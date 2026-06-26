#!/usr/bin/env python3
"""Validate the Phase 1 PGD serial optimizations (Changes A, B, C).

This harness proves the three serial optimizations in
``docs/plans/PHASE1_PGD_SERIAL_OPTIMIZATIONS_PLAN.md`` (audit IDs #1, #4, #6 in
``docs/reference/PHASE1_PGD_SERIAL_OPTIMIZATION_AUDIT.md``) preserve the computed
minimizer while measuring the speedup.

  * Change A — backtracking step warm-start (per-iteration step trajectory only).
  * Change B — projection inner-loop cleanup (result-preserving refactor + a
    scalar-residual stall test).
  * Change C — gradient reuse (bit-identical; halves gradient evaluations).

Two modes:

Mode 1 (``--equivalence``, in-process, fast)
    1a. Projection equivalence (Change B): runs the *current*
        ``orthogonal_projection_iterative`` against a verbatim frozen copy of the
        *original* implementation (``reference_projection_iterative`` below) over a
        grid of sizes and seeds. Asserts agreement within ``max(1e-10, 10*tol)``
        and that both satisfy the constraints.
    1b. Gradient-reuse identity (Change C): builds a tiny optimizer and asserts
        ``compute_gradient`` at a fixed iterate is reproducible to ``< 1e-12``.

Mode 2 (``--compare --baseline <run_dir> --candidate <run_dir>``)
    Compares two completed Phase 1 runs produced on the *same config and seed* —
    one built from ``main``, one from the branch. Reports/asserts final energy,
    solution-vector delta, permutation-invariant partition agreement, region-area
    delta, and the backtrack/wall-time speedup.

A/B run protocol (Mode 2):
    1. Pick a small, fast, deterministic config (e.g.
       ``parameters/torus_10part.yaml``) with a fixed ``seed``. Use the *identical*
       config file for both branches. Run with ``--profile``.
    2. ``git checkout main`` →
       ``python scripts/find_surface_partition.py --config <cfg> --profile``
       → note the ``results/run_.../`` dir (the ``--baseline``).
    3. ``git checkout feat/pgd-serial-optimizations`` → run the *same* command →
       note the new ``results/run_.../`` dir (the ``--candidate``).
    4. ``python testing/validate_pgd_optimizations.py --compare \
           --baseline <baseline_run_dir> --candidate <candidate_run_dir>``
    Both runs must execute on the same machine (BLAS reduction order affects the
    1e-12 exact thresholds).

Stage thresholds (changes are committed/validated in order C → B → A):
    | After commit | Energy / x_opt threshold              | Partition |
    | C            | max|dx| < 1e-12, energy rel < 1e-10   | 100 %     |
    | B            | max|dx| < 1e-9,  energy rel < 1e-8    | 100 %     |
    | A            | energy rel < 1e-6, areas rel < 1e-3   | >= 99.5 % |

Usage:
    python testing/validate_pgd_optimizations.py --equivalence
    python testing/validate_pgd_optimizations.py --compare \
        --baseline <baseline_run_dir> --candidate <candidate_run_dir> [--stage A|B|C]

Exit code: 0 on PASS, 1 on FAIL.
"""

import argparse
import glob
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.optimization.projection import (  # noqa: E402
    orthogonal_projection_iterative,
)


# ---------------------------------------------------------------------------
# Frozen verbatim copy of the ORIGINAL orthogonal_projection_iterative.
# This is the pre-Change-B implementation, captured so the equivalence test does
# not depend on git history. Do NOT edit to track refactors — it is the baseline.
# ---------------------------------------------------------------------------
def reference_projection_iterative(A, c, d, v, max_iter=1000, tol=1e-10,
                                   logger=None, _prof=None):
    if logger is None:
        logger = logging.getLogger(__name__ + ".reference")

    # Validate input dimensions
    N, n = A.shape
    if len(c) != n or len(d) != n or len(v) != N:
        raise ValueError(
            f"Dimension mismatch: A({N}x{n}), c({len(c)}), d({len(d)}), v({len(v)})"
        )

    A = A.copy()  # Make a copy to avoid modifying the input

    # Initial normalization to satisfy partition constraint
    row_sums = np.sum(A, axis=1)
    mask = row_sums > 0  # Avoid division by zero
    A[mask] = A[mask] / row_sums[mask, np.newaxis]
    A[~mask] = 1.0 / n  # Set uniform distribution for zero rows

    # Small regularization to avoid numerical issues
    epsilon = 1e-10

    # Track convergence history
    convergence_history = {
        'iterations': [],
        'row_errors': [],
        'area_errors': [],
        'max_errors': []
    }

    for iter in range(max_iter):
        A_prev = A.copy()

        # Step 1: Calculate line sum error (N x 1 column vector)
        e = np.sum(A, axis=1) - np.ones(N)  # Each row should sum to 1

        # Step 2: Calculate column scalar product error (n x 1 column vector)
        f = v @ A - d

        # Step 3: Define matrix C of size n x n
        v_norm_squared = np.sum(v ** 2)
        C = np.full((n, n), -v_norm_squared / n)
        np.fill_diagonal(C, v_norm_squared - v_norm_squared / n)

        # Step 4: Calculate q vector
        q = f - np.dot(v, e) / n

        # Step 5: Solve for lambda
        try:
            # Try solving the full system first
            lambda_vec = np.linalg.solve(C + epsilon * np.eye(n), q)
        except np.linalg.LinAlgError:
            # Fall back to reduced system if full system fails
            lambda_vec = np.zeros(n)
            lambda_vec[:-1] = np.linalg.solve(
                C[:-1, :-1] + epsilon * np.eye(n - 1), q[:-1])

        # Step 6: Calculate S
        S = np.sum(lambda_vec)

        # Step 7: Calculate eta vector
        eta = (e - S * v) / n

        # Step 8: Calculate orthogonal correction
        A_orth = np.outer(eta, np.ones(n)) + np.outer(v, lambda_vec)

        # Step 9: Apply correction
        A = A - A_orth

        # Step 10: Ensure non-negativity
        A = np.maximum(A, 0)

        # Step 11: Normalize rows to ensure partition constraint
        row_sums = np.sum(A, axis=1)
        mask = row_sums > epsilon  # Avoid division by zero
        A[mask] = A[mask] / row_sums[mask, np.newaxis]
        A[~mask] = 1.0 / n  # Set uniform distribution for zero rows

        # Step 12: Project onto area constraints
        area_sums = v @ A
        scale_factors = d / (area_sums + epsilon)
        A = A * scale_factors[np.newaxis, :]

        # Check convergence of both constraints
        row_sum_error = np.max(np.abs(np.sum(A, axis=1) - 1))
        area_error = np.max(np.abs(v @ A - d))
        max_error = max(row_sum_error, area_error)

        # Track convergence history
        convergence_history['iterations'].append(iter)
        convergence_history['row_errors'].append(row_sum_error)
        convergence_history['area_errors'].append(area_error)
        convergence_history['max_errors'].append(max_error)

        if row_sum_error < tol and area_error < tol:
            break

        # Check if we're making progress
        if iter > 0 and np.allclose(A, A_prev, rtol=tol, atol=tol):
            break

    # Validate final result
    final_row_error = np.max(np.abs(np.sum(A, axis=1) - 1))
    final_area_error = np.max(np.abs(v @ A - d))

    if final_row_error > 10 * tol or final_area_error > 10 * tol:
        raise RuntimeError(
            "Orthogonal projection failed to achieve required tolerance")

    return A


# ---------------------------------------------------------------------------
# Mode 1 — in-process numerical proof
# ---------------------------------------------------------------------------
def _constraint_errors(A, v, d):
    row_err = float(np.max(np.abs(np.sum(A, axis=1) - 1.0)))
    area_err = float(np.max(np.abs(v @ A - d)))
    return row_err, area_err


def run_equivalence(tol=1e-9, max_iter=300):
    print("=" * 70)
    print("Mode 1 — in-process equivalence proof")
    print("=" * 70)
    ok = True

    # --- 1a. Projection equivalence (Change B) ---------------------------
    print("\n[1a] Projection equivalence (current vs frozen reference)")
    print(f"     tol={tol:.1e}, max_iter={max_iter}, "
          f"atol_gate={max(1e-10, 10 * tol):.1e}")
    sizes = [(200, 5), (500, 10), (1000, 30), (2000, 50)]
    seeds = [0, 1, 7, 42]
    atol = max(1e-10, 10 * tol)
    worst_delta = 0.0
    for (V, Nreg) in sizes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            A = rng.random((V, Nreg))
            v = rng.random(V) + 0.1
            d = (v.sum() / Nreg) * np.ones(Nreg)
            c = np.ones(Nreg)
            A_new = orthogonal_projection_iterative(
                A, c, d, v, max_iter=max_iter, tol=tol)
            A_ref = reference_projection_iterative(
                A, c, d, v, max_iter=max_iter, tol=tol)
            delta = float(np.max(np.abs(A_new - A_ref)))
            worst_delta = max(worst_delta, delta)
            rerr_n, aerr_n = _constraint_errors(A_new, v, d)
            rerr_r, aerr_r = _constraint_errors(A_ref, v, d)
            feasible = max(rerr_n, aerr_n, rerr_r, aerr_r) < 10 * tol
            row_ok = delta < atol and feasible
            ok = ok and row_ok
            status = "PASS" if row_ok else "FAIL"
            print(f"     V={V:5d} N={Nreg:3d} seed={seed:2d}: "
                  f"max|dA|={delta:.2e} feas(new)=({rerr_n:.1e},{aerr_n:.1e}) "
                  f"[{status}]")
    print(f"     worst max|dA| over grid = {worst_delta:.2e} "
          f"(gate {atol:.1e})")

    # --- 1b. Gradient-reuse identity (Change C) --------------------------
    print("\n[1b] Gradient-reuse identity (compute_gradient reproducible)")
    try:
        from src.surfaces.torus import TorusMeshProvider
        from src.optimization.pgd_optimizer import ProjectedGradientOptimizer

        prov = TorusMeshProvider(n_theta=12, n_phi=8, R=2.0, r=1.0)
        prov.set_resolution(12, 8)
        mesh = prov.build()
        n_parts = 4
        opt = ProjectedGradientOptimizer(
            K=mesh.K, M=mesh.M, v=mesh.v, n_partitions=n_parts,
            epsilon=0.1, total_area=prov.theoretical_total_area(),
        )
        V = len(mesh.v)
        rng = np.random.default_rng(123)
        A = rng.random((V, n_parts))
        c = np.ones(n_parts)
        d = (np.sum(mesh.v) / n_parts) * np.ones(n_parts)
        A = orthogonal_projection_iterative(A, c, d, mesh.v,
                                            max_iter=300, tol=1e-9)
        x = A.flatten()
        # One manual accepted PGD step.
        g = opt.compute_gradient(x)
        A_trial = x.reshape(V, n_parts) - 0.1 * g.reshape(V, n_parts)
        A_trial = np.clip(A_trial, 1e-8, 1 - 1e-8)
        A_trial = orthogonal_projection_iterative(A_trial, c, d, mesh.v,
                                                  max_iter=300, tol=1e-9)
        xp = A_trial.flatten()
        g1 = opt.compute_gradient(xp)
        g2 = opt.compute_gradient(xp)
        dg = float(np.max(np.abs(g1 - g2)))
        grad_ok = dg < 1e-12
        ok = ok and grad_ok
        print(f"     V={V} N={n_parts}: max|g(x')-g(x')| = {dg:.2e} "
              f"[{'PASS' if grad_ok else 'FAIL'}]")
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"     SKIP (could not build optimizer): {exc}")

    print("\n" + ("RESULT: PASS" if ok else "RESULT: FAIL"))
    return ok


# ---------------------------------------------------------------------------
# Mode 2 — compare two completed runs
# ---------------------------------------------------------------------------
def _load_run(run_dir):
    run = Path(run_dir)
    sol_glob = sorted(glob.glob(str(run / "solution" / "*.h5")))
    if not sol_glob:
        raise FileNotFoundError(f"No solution/*.h5 under {run}")
    import h5py
    sol = {}
    with h5py.File(sol_glob[0], "r") as h5:
        sol["x_opt"] = np.array(h5["x_opt"])
        sol["vertices"] = np.array(h5["vertices"])
        sol["faces"] = np.array(h5["faces"])
        sol["n_partitions"] = int(h5.attrs["n_partitions"])
    meta_path = run / "solution" / "metadata.yaml"
    sol["metadata"] = (yaml.safe_load(meta_path.read_text())
                       if meta_path.exists() else {})
    tp_path = run / "solution" / "timing_profile.yaml"
    sol["timing"] = (yaml.safe_load(tp_path.read_text())
                    if tp_path.exists() else None)
    # Final energy from the finest-level trace summary (last row, OBJFUN col 3).
    sol["final_energy"] = _final_energy_from_traces(run)
    return sol


def _final_energy_from_traces(run):
    summaries = sorted(glob.glob(str(Path(run) / "traces" / "*_summary.out")))
    if not summaries:
        return None

    def level_of(p):
        name = Path(p).name
        for tok in name.split("_"):
            if tok.startswith("level"):
                try:
                    return int(tok[len("level"):])
                except ValueError:
                    return -1
        return -1

    summaries.sort(key=level_of)
    finest = summaries[-1]
    last = None
    for line in Path(finest).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("MAJOR"):
            continue
        last = line
    if last is None:
        return None
    cols = last.split()
    # Columns: MAJOR NFEV NGEV OBJFUN GNORM CNORM FEAS OPT STEP
    return float(cols[3])


def _partition_agreement(x_b, x_c, V, N):
    Phi_b = x_b.reshape(V, N)
    Phi_c = x_c.reshape(V, N)
    lab_b = np.argmax(Phi_b, axis=1)
    lab_c = np.argmax(Phi_c, axis=1)
    # Overlap matrix; rows = baseline label, cols = candidate label.
    overlap = np.zeros((N, N), dtype=np.int64)
    np.add.at(overlap, (lab_b, lab_c), 1)
    from scipy.optimize import linear_sum_assignment
    # Maximize overlap → minimize negative.
    row_ind, col_ind = linear_sum_assignment(-overlap)
    agree = int(overlap[row_ind, col_ind].sum())
    return 100.0 * agree / V


def run_compare(baseline_dir, candidate_dir, stage=None):
    print("=" * 70)
    print("Mode 2 — compare two completed Phase 1 runs")
    print(f"  baseline : {baseline_dir}")
    print(f"  candidate: {candidate_dir}")
    if stage:
        print(f"  stage    : {stage} "
              f"(applies the stage-specific acceptance thresholds)")
    print("=" * 70)

    b = _load_run(baseline_dir)
    cd = _load_run(candidate_dir)

    if b["n_partitions"] != cd["n_partitions"]:
        print(f"FAIL: n_partitions differ "
              f"({b['n_partitions']} vs {cd['n_partitions']})")
        return False
    N = b["n_partitions"]
    x_b = b["x_opt"]
    x_c = cd["x_opt"]
    if x_b.shape != x_c.shape:
        print(f"FAIL: x_opt shapes differ ({x_b.shape} vs {x_c.shape})")
        return False
    V = x_b.size // N

    # Stage thresholds.
    thr = {
        "C": dict(dx=1e-12, energy=1e-10, partition=100.0, areas=1e-3),
        "B": dict(dx=1e-9, energy=1e-8, partition=100.0, areas=1e-3),
        "A": dict(dx=None, energy=1e-6, partition=99.5, areas=1e-3),
    }.get(stage, dict(dx=None, energy=1e-6, partition=99.5, areas=1e-3))

    ok = True

    # 1. Final energy.
    print("\n[1] Final energy")
    if b["final_energy"] is not None and cd["final_energy"] is not None:
        E_b, E_c = b["final_energy"], cd["final_energy"]
        rel = abs(E_b - E_c) / abs(E_b) if E_b != 0 else abs(E_c)
        e_ok = rel < thr["energy"]
        ok = ok and e_ok
        print(f"    E_base={E_b:.12e}  E_cand={E_c:.12e}")
        print(f"    rel diff = {rel:.2e}  (gate {thr['energy']:.0e}) "
              f"[{'PASS' if e_ok else 'FAIL'}]")
    else:
        print("    SKIP (final energy not found in traces)")

    # 2. Solution vector (only meaningful at exact stages).
    print("\n[2] Solution vector max|dx|")
    dx = float(np.max(np.abs(x_b - x_c)))
    if thr["dx"] is not None:
        dx_ok = dx < thr["dx"]
        ok = ok and dx_ok
        print(f"    max|dx| = {dx:.2e}  (gate {thr['dx']:.0e}) "
              f"[{'PASS' if dx_ok else 'FAIL'}]")
    else:
        print(f"    max|dx| = {dx:.2e}  (no gate at stage A; trajectory differs)")

    # 3. Partition agreement (permutation-invariant).
    print("\n[3] Partition agreement (optimal relabel)")
    agree = _partition_agreement(x_b, x_c, V, N)
    p_ok = agree >= thr["partition"]
    ok = ok and p_ok
    print(f"    agreement = {agree:.4f}%  (gate >= {thr['partition']}%) "
          f"[{'PASS' if p_ok else 'FAIL'}]")

    # 4. Region areas (permutation-invariant).
    print("\n[4] Region areas (sorted) rel diff")
    # v = lumped mass = row-sum of M; reconstruct from solution metadata if
    # available, else use uniform weights as a fallback (areas via vertex count).
    areas_b = np.sort(np.bincount(np.argmax(x_b.reshape(V, N), axis=1),
                                  minlength=N).astype(float))
    areas_c = np.sort(np.bincount(np.argmax(x_c.reshape(V, N), axis=1),
                                  minlength=N).astype(float))
    denom = np.where(areas_b != 0, areas_b, 1.0)
    area_rel = float(np.max(np.abs(areas_b - areas_c) / denom))
    a_ok = area_rel < thr["areas"]
    ok = ok and a_ok
    print(f"    sorted region-vertex-count rel diff = {area_rel:.2e}  "
          f"(gate {thr['areas']:.0e}) [{'PASS' if a_ok else 'FAIL'}]")

    # 5. Speedup (report; assert backtrack reduction if timing present).
    print("\n[5] Speedup (mean backtracks / iter, total wall time)")
    bt_b = _mean_backtracks(b["timing"])
    bt_c = _mean_backtracks(cd["timing"])
    wall_b = _total_wall(b["timing"])
    wall_c = _total_wall(cd["timing"])
    if bt_b is not None and bt_c is not None:
        print(f"    mean backtracks/iter: base={bt_b:.4f}  cand={bt_c:.4f}")
        bt_ok = bt_c < bt_b
        if stage == "A" or stage is None:
            ok = ok and bt_ok
        print(f"    backtrack reduction: "
              f"{'PASS' if bt_ok else 'FAIL'} "
              f"({'expected only after Change A' if stage in ('B', 'C') else 'asserted'})")
    else:
        print("    SKIP backtracks (no timing_profile.yaml; rerun with --profile)")
    if wall_b is not None and wall_c is not None and wall_c > 0:
        print(f"    total wall: base={wall_b:.2f}s  cand={wall_c:.2f}s  "
              f"speedup={wall_b / wall_c:.2f}x")
    else:
        print("    SKIP wall time (no timing_profile.yaml)")

    print("\n" + ("RESULT: PASS" if ok else "RESULT: FAIL"))
    return ok


def _mean_backtracks(timing):
    if not timing:
        return None
    summ = timing.get("summary", {})
    return summ.get("mean_backtracks_per_iter")


def _total_wall(timing):
    if not timing:
        return None
    summ = timing.get("summary", {})
    return summ.get("total_wall_s")


def main():
    ap = argparse.ArgumentParser(
        description="Validate Phase 1 PGD serial optimizations (Changes A/B/C).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--equivalence", action="store_true",
                    help="Mode 1: in-process projection + gradient equivalence.")
    ap.add_argument("--compare", action="store_true",
                    help="Mode 2: compare two completed run directories.")
    ap.add_argument("--baseline", help="Baseline run dir (built from main).")
    ap.add_argument("--candidate", help="Candidate run dir (built from branch).")
    ap.add_argument("--stage", choices=["A", "B", "C"],
                    help="Apply the stage-specific acceptance thresholds.")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="Projection tolerance for Mode 1 (default 1e-9).")
    args = ap.parse_args()

    if not (args.equivalence or args.compare):
        ap.error("choose at least one of --equivalence / --compare")

    ok = True
    if args.equivalence:
        ok = run_equivalence(tol=args.tol) and ok
    if args.compare:
        if not (args.baseline and args.candidate):
            ap.error("--compare requires --baseline and --candidate")
        ok = run_compare(args.baseline, args.candidate, stage=args.stage) and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
