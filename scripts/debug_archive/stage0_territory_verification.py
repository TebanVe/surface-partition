#!/usr/bin/env python3
"""Stage 0 pre-implementation verification for the territory-aware plan.

Re-verifies, directly on an existing Phase 1 solution (no new run), the two
measured findings of docs/plans/PHASE1_N1000_VALIDITY_PLAN.md that gate the
territory-aware implementation (docs/plans/PHASE1_TERRITORY_AWARE_IMPLEMENTATION_PLAN.md
Stage 0):

  [A] The accounting identity  T_k - Abar = gain_k - lost_k  (validity plan
      section 2.1): with continuous mass pinned at Abar, the winner-take-all
      territory deviation of every cell equals its interface-band mass
      exchange. Reported: max identity error, corr(rel area deviation,
      lost-paint fraction), mean/std lost fraction.

  [B] The frozen-optimizer finding (validity plan section 2.3): the final
      iterate is NOT constrained-stationary. Reported: ||g||, the
      reduced-gradient KKT residual ||g_t|| (gradient projected onto the
      tangent of the active constraints restricted to the free set, duals via
      Gauss-Seidel sweeps), the fraction of bound-pinned entries, and the
      projection non-idempotency (movement + energy cost of projecting the
      already-feasible iterate).

  [C] (optional, --descent-steps K) A K-step tangential-descent probe from the
      "converged" iterate: energy recovered and whether the winner map moves
      (the "P2 alone does not fix validity" check, validity plan section 2.4).

Usage:
    python scripts/debug_archive/stage0_territory_verification.py \
        --run results/run_20260714_224821_surftorus_npart300_..._seed61803399 \
        [--descent-steps 40] [--gs-sweeps 10]

Exit code 0 always (diagnostic; numbers are printed for the report).
"""

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.mesh.tri_mesh import TriMesh  # noqa: E402
from src.optimization.pgd_optimizer import ProjectedGradientOptimizer  # noqa: E402
from src.optimization.projection import orthogonal_projection_iterative  # noqa: E402
from src.partition.find_contours import detect_area_imbalance  # noqa: E402


def load_run(run_dir):
    run = Path(run_dir)
    candidates = sorted(
        p for p in glob.glob(str(run / "solution" / "*.h5"))
        if not p.endswith("_refined_contours.h5")
    )
    if not candidates:
        raise FileNotFoundError(f"No base solution under {run / 'solution'}")
    sol_path = candidates[-1]
    with h5py.File(sol_path, "r") as f:
        x_opt = np.array(f["x_opt"])
        vertices = np.array(f["vertices"])
        faces = np.array(f["faces"])
        n_partitions = int(f.attrs["n_partitions"])
        lam = float(f.attrs["lambda_penalty"])
    meta_path = run / "solution" / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "x_opt": x_opt, "vertices": vertices, "faces": faces,
        "n_partitions": n_partitions, "lambda_penalty": lam,
        "meta": meta or {}, "solution_path": sol_path,
    }


def accounting_identity(U, v, N):
    """[A] T_k - Abar = gain_k - lost_k, and the lost-fraction statistics."""
    Vn = U.shape[0]
    winners = np.argmax(U, axis=1)
    A = float(v.sum())
    Abar = A / N
    T = np.bincount(winners, weights=v, minlength=N)      # hard WTA territory
    mass = v @ U                                          # continuous mass per cell
    u_win = U[np.arange(Vn), winners]                     # winner's own density
    kept = np.bincount(winners, weights=v * u_win, minlength=N)
    lost = mass - kept                                    # own mass on lost vertices
    gain_k = np.bincount(                                 # foreign mass on won verts
        winners, weights=v * (1.0 - u_win), minlength=N)
    identity_err = np.max(np.abs((T - Abar) - (gain_k - lost)))
    rel_dev = (T - Abar) / Abar
    lost_frac = lost / mass
    corr = float(np.corrcoef(rel_dev, lost_frac)[0, 1])
    return {
        "Abar": Abar,
        "mass_residual": float(np.max(np.abs(mass - Abar))),
        "identity_max_err": float(identity_err),
        "corr_reldev_lostfrac": corr,
        "lost_frac_mean": float(lost_frac.mean()),
        "lost_frac_std": float(lost_frac.std()),
        "lost_frac_min": float(lost_frac.min()),
        "lost_frac_max": float(lost_frac.max()),
        "worst_rel_dev": float(np.abs(rel_dev).max()),
        "T": T, "rel_dev": rel_dev,
    }


def free_set(U, G, lb=1e-6, ub_margin=1e-6):
    """Entries NOT pinned at a box bound with outward (bound-infeasible) gradient."""
    at_lo = U <= lb
    at_hi = U >= 1.0 - ub_margin
    pinned = (at_lo & (G > 0)) | (at_hi & (G < 0))
    return ~pinned


def reduced_gradient(G, v, free, sweeps=10, alpha0=None, beta0=None):
    """Project G onto the tangent of {row-sum, weighted-column-sum} constraints
    restricted to the free set: g_t = (G - alpha 1^T - v beta^T) on free entries.

    alpha (V,) and beta (Ncells,) solve the coupled normal equations
        sum_{k in F_i} (G_ik - alpha_i - v_i beta_k) = 0
        sum_{i in F_k} v_i (G_ik - alpha_i - v_i beta_k) = 0
    by Gauss-Seidel block sweeps (each sweep = two O(V*N) reductions).
    """
    V, N = G.shape
    Fm = free.astype(np.float64)
    alpha = np.zeros(V) if alpha0 is None else alpha0.copy()
    beta = np.zeros(N) if beta0 is None else beta0.copy()
    row_cnt = Fm.sum(axis=1)                      # |F_i|
    col_vv = (v[:, None] ** 2 * Fm).sum(axis=0)   # sum_i v_i^2 over free
    row_safe = np.maximum(row_cnt, 1.0)
    col_safe = np.maximum(col_vv, 1e-300)
    for _ in range(sweeps):
        # alpha block: given beta
        num_a = (Fm * (G - v[:, None] * beta[None, :])).sum(axis=1)
        alpha = np.where(row_cnt > 0, num_a / row_safe, 0.0)
        # beta block: given alpha
        num_b = (Fm * v[:, None] * (G - alpha[:, None])).sum(axis=0)
        beta = np.where(col_vv > 0, num_b / col_safe, 0.0)
    Gt = (G - alpha[:, None] - v[:, None] * beta[None, :]) * Fm
    return Gt, alpha, beta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="Phase 1 run directory")
    ap.add_argument("--gs-sweeps", type=int, default=10)
    ap.add_argument("--descent-steps", type=int, default=0,
                    help="Optional tangential-descent probe length (0 = skip)")
    ap.add_argument("--descent-step-size", type=float, default=None,
                    help="Step size for the probe (default: 1/||g_t||_inf-scaled)")
    args = ap.parse_args()

    sol = load_run(args.run)
    N = sol["n_partitions"]
    x = sol["x_opt"]
    V = x.size // N
    U = x.reshape(V, N)
    print("=" * 72)
    print("Stage 0 verification — territory-aware plan")
    print(f"  run          : {args.run}")
    print(f"  solution     : {sol['solution_path']}")
    print(f"  V x N        : {V} x {N}")
    print(f"  lambda       : {sol['lambda_penalty']}")

    print("\nBuilding mesh matrices (P1 assembly)...")
    t0 = time.time()
    mesh = TriMesh(sol["vertices"], sol["faces"])
    mesh.compute_matrices()
    eps_meta = sol["meta"].get("final_epsilon")
    stats = mesh.get_mesh_statistics()
    epsilon = float(eps_meta) if eps_meta else float(
        np.sqrt(stats["mean_triangle_area"]))
    print(f"  done in {time.time() - t0:.1f}s; epsilon = {epsilon:.6e}")

    # ------------------------------------------------------------------ [A]
    print("\n[A] Accounting identity  T_k - Abar = gain_k - lost_k")
    acc = accounting_identity(U, mesh.v, N)
    print(f"  continuous-mass residual max|m_k - Abar| : {acc['mass_residual']:.3e}")
    print(f"  identity max error                       : {acc['identity_max_err']:.3e}")
    print(f"  corr(rel area dev, lost fraction)        : {acc['corr_reldev_lostfrac']:.4f}")
    print(f"  lost-paint fraction mean / std           : "
          f"{acc['lost_frac_mean'] * 100:.1f}% / {acc['lost_frac_std'] * 100:.1f}%")
    print(f"  lost-paint fraction range                : "
          f"[{acc['lost_frac_min'] * 100:.1f}%, {acc['lost_frac_max'] * 100:.1f}%]")
    print(f"  worst |T_k - Abar|/Abar                  : {acc['worst_rel_dev'] * 100:.2f}%")
    gate = detect_area_imbalance(U, mesh.v, N)
    print(f"  detect_area_imbalance: n_imbalanced = {gate['n_imbalanced']}, "
          f"worst cell {gate['worst_cell']} at {gate['worst_rel_dev'] * 100:.2f}%")

    # ------------------------------------------------------------------ [B]
    print("\n[B] Frozen-optimizer finding (KKT residual + projection noise)")
    opt = ProjectedGradientOptimizer(
        K=mesh.K, M=mesh.M, v=mesh.v, n_partitions=N,
        epsilon=epsilon, lambda_penalty=sol["lambda_penalty"],
    )
    E0 = opt.compute_energy(x)
    g = opt.compute_gradient(x)
    G = g.reshape(V, N)
    gnorm = float(np.linalg.norm(g))
    free = free_set(U, G)
    pinned_frac = 1.0 - free.mean()
    Gt, alpha, beta = reduced_gradient(G, mesh.v, free, sweeps=args.gs_sweeps)
    gtnorm = float(np.linalg.norm(Gt))
    print(f"  E(x)                        : {E0:.10e}")
    print(f"  ||g||                       : {gnorm:.4f}")
    print(f"  bound-pinned entry fraction : {pinned_frac * 100:.2f}%")
    print(f"  ||g_t|| (KKT residual)      : {gtnorm:.4f}   "
          f"({gtnorm / gnorm * 100:.1f}% of ||g|| is feasible descent)")

    # Projection non-idempotency at the already-feasible iterate
    c = np.ones(N)
    d = (np.sum(mesh.v) / N) * np.ones(N)
    Ap = orthogonal_projection_iterative(
        U.copy(), c, d, mesh.v, max_iter=300, tol=1e-8)
    move_inf = float(np.max(np.abs(Ap - U)))
    move_rms = float(np.sqrt(np.mean((Ap - U) ** 2)))
    Ep = opt.compute_energy(Ap.flatten())
    print(f"  projection non-idempotency  : max|P(x)-x| = {move_inf:.3e}, "
          f"rms = {move_rms:.3e}")
    print(f"  energy cost of projecting   : E(P(x)) - E(x) = {Ep - E0:+.3e}")

    # ------------------------------------------------------------------ [C]
    if args.descent_steps > 0:
        print(f"\n[C] Tangential-descent probe ({args.descent_steps} steps)")
        winners0 = np.argmax(U, axis=1)
        Ucur = U.copy()
        Ecur = E0
        s = args.descent_step_size
        if s is None:
            # Conservative: the frozen run's last accepted scale was ~1e-12,
            # but the tangential direction is feasible; use the Armijo-scale
            # step the validity plan probe used (s such that s*||g_t||_inf
            # stays well inside the box).
            s = 0.1 / max(float(np.max(np.abs(Gt))), 1e-30)
        alpha_ws, beta_ws = alpha, beta
        accepted = 0
        for it in range(args.descent_steps):
            trial = np.clip(Ucur - s * Gt, 1e-8, 1 - 1e-8)
            trial = orthogonal_projection_iterative(
                trial, c, d, mesh.v, max_iter=300, tol=1e-8)
            Et = opt.compute_energy(trial.flatten())
            if Et < Ecur:
                Ucur, Ecur = trial, Et
                accepted += 1
                Gc = opt.compute_gradient(Ucur.flatten()).reshape(V, N)
                fr = free_set(Ucur, Gc)
                Gt, alpha_ws, beta_ws = reduced_gradient(
                    Gc, mesh.v, fr, sweeps=args.gs_sweeps,
                    alpha0=alpha_ws, beta0=beta_ws)
            else:
                s *= 0.5
        winners1 = np.argmax(Ucur, axis=1)
        moved = int(np.sum(winners0 != winners1))
        gate1 = detect_area_imbalance(Ucur, mesh.v, N)
        print(f"  accepted steps              : {accepted}/{args.descent_steps}")
        print(f"  energy recovered            : {E0 - Ecur:+.4f} "
              f"({E0:.4f} -> {Ecur:.4f})")
        print(f"  winner-map vertices moved   : {moved}/{V}")
        print(f"  n_imbalanced after probe    : {gate1['n_imbalanced']} "
              f"(worst {gate1['worst_rel_dev'] * 100:.2f}%)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
