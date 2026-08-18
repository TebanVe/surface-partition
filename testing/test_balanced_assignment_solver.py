#!/usr/bin/env python3
"""Phase 0a acceptance gates for the shared balanced-assignment solver.

Approaches A, B and C all need the same primitive -- balanced assignment given
arbitrary per-vertex scores -- which ships inside the balanced readout as
`solve_dual_offsets`. Generalizing it must not disturb A, and must converge on
the score scales B and C actually produce.

GATE 1  approach A is undisturbed: the shipped readout reproduces a recorded
        reference exactly, with normalization off.

GATE 2  foreign score scales converge, measured on the state the arms are
        actually in.

Gate 2 was rewritten on 2026-08-17 after adversarial review refuted its first
version. Three things were wrong with it, and the fixes define what it now does:

* **It posed a cold start.** B and C are iterative; from outer iteration 1 they
  assign from an already-balanced partition, not from a raw geometric guess.
  Scoring the one-shot problem measured a state the arms occupy for a single
  iteration and recover from unaided. Gate 2 now scores *settled* outer
  iterations and reports the cold start as informational only.
* **Its bar was absolute and indefensible** -- 2 x vertex granularity, which
  approach A itself fails on its home scores (the shipped dual stalls at 2.12%
  against a 2.02% bar at V=47,488/N=300). A gate that the shipped solver fails
  cannot be evidence about anything else. The bar is now calibrated per fixture
  against a strong-reference run: the default config must land within 25% of
  what a long, tuned run achieves on that same score matrix, or inside the
  granularity floor, whichever is looser.
* **It never tested the production configuration** V=114,144/N=300, and used a
  single seed. Both fixed.

Usage:
    python testing/test_balanced_assignment_solver.py           # both gates
    python testing/test_balanced_assignment_solver.py --gate2   # foreign only
    python testing/test_balanced_assignment_solver.py --quick   # skip the big pin
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import factorized

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.partition.balanced_readout import (  # noqa: E402
    BalancedReadoutConfig,
    assignment_margin_scale,
    solve_dual_offsets,
)
from src.surfaces.torus import TorusMeshProvider  # noqa: E402

# (n_theta, n_phi, N, outer_iterations, seeds). The third entry is the
# production configuration the first version of this gate never tested.
# Three seeds, not two. Reviews 3 and 4 both flagged n=2 as thin for a marginal
# verdict, and the production pin's own seed spread (0.888 vs 0.702 under the old
# protocol) was several times the margin it was being judged by.
FIXTURES = [
    (224, 212, 300, 4, (0, 1, 2)),
    (348, 328, 100, 4, (0, 1, 2)),
    (348, 328, 300, 4, (0, 1, 2)),
]
# Genuine long reference. MUST exceed the shipped budget and MUST be set through
# normalized_dual_iters: setting dual_iters here was DEAD from 6bb78cb onward,
# because the normalized path reads normalized_dual_iters, so the "strong
# reference" silently became the identical config under test -- which is why
# settled equalled strong-ref to three decimals and made the plan's
# "settled ~= strong-ref => converging" claim circular.
STRONG_ITERS = 4000

# The fixtures' own internal assignments use a PINNED reference config, never
# the solver under test. Without this the fixture trajectory degenerates along
# with a broken solver (cells die), so a broken solver is graded on the easy
# problem it created for itself -- measured: a psi=0 solver reached 80.15% on B
# and still passed, because the fixture had collapsed around it.
# ALL five knobs the normalized path reads are pinned. Pinning only two left the
# fixtures free to change silently if a default moved (normalized_eta0,
# dual_decay, score_margin_percentile), which would regenerate different fixtures
# for the next comparison without anyone noticing.
REFERENCE_CFG = dict(
    normalize_scores=True,
    normalized_dual_iters=400,
    normalized_eta0=2.0,
    dual_decay=0.02,
    score_margin_percentile=50.0,
)

# Incumbent-parity floor, imported from the harness so ONE number governs both
# this gate and the harness's own area lane (they disagreed before: 2 x
# granularity there vs A's stall here, which at the production pin meant the
# fault flag fired at 0.841% while this gate declared 0a done at 2.022%).
#
# Superseded values, kept as a warning: the first floor used A's stall as
# RECORDED in the shipped runs -- 2.1215% / 2.0218%. Those are A running
# UN-normalized. The same solver on the same log densities with
# normalize_scores ON and the same 400-iteration budget reaches 0.9072% and
# 1.8384%, so the recorded figures are a configuration artifact, and importing
# them let an arm clear a bar the incumbent only fails because of a setting.
from src.partition.arm_harness import assignment_quality_bar  # noqa: E402


def build(n_theta, n_phi):
    return TorusMeshProvider(n_theta=n_theta, n_phi=n_phi, R=1.0, r=0.6).build()


def farthest_point_seeds(vertices, n_seeds, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(vertices)))]
    d = np.linalg.norm(vertices - vertices[idx[0]], axis=1)
    for _ in range(n_seeds - 1):
        nxt = int(np.argmax(d))
        idx.append(nxt)
        d = np.minimum(d, np.linalg.norm(vertices - vertices[nxt], axis=1))
    return np.array(idx)


def edge_graph(mesh):
    f = mesh.faces
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    w = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    n = len(mesh.vertices)
    g = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return g.maximum(g.T)


def mean_edge_length(mesh):
    f = mesh.faces
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return float(np.mean(np.linalg.norm(
        mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)))


def _assign(scores, v, target):
    """Fixture-internal assignment -- ALWAYS the pinned reference config.

    Deliberately not the solver under test: see REFERENCE_CFG.
    """
    cfg = BalancedReadoutConfig(**REFERENCE_CFG)
    _, labels, worst = solve_dual_offsets(scores, v, target, cfg)
    return labels, worst


def fixtures_C(mesh, n_cells, n_outer, seed):
    """Iterated C: Lloyd on a capacity-constrained geodesic power diagram.

    Yields (outer_iteration, score_matrix). Iteration 0 is the cold start from
    raw farthest-point seeds; later iterations start from the previous balanced
    partition, which is the state C is actually in.
    """
    v, target = mesh.v, float(mesh.v.sum()) / n_cells
    graph = edge_graph(mesh)
    sites = farthest_point_seeds(mesh.vertices, n_cells, seed)
    for it in range(n_outer):
        scores = -(dijkstra(graph, indices=sites, directed=False).T ** 2)
        yield it, scores
        labels, _ = _assign(scores, v, target)
        # Lloyd step: v-weighted centroid, snapped back to the nearest vertex.
        new_sites = []
        for k in range(n_cells):
            m = labels == k
            if not m.any():
                new_sites.append(sites[k])
                continue
            c = (mesh.vertices[m] * v[m, None]).sum(0) / v[m].sum()
            new_sites.append(int(np.argmin(
                np.linalg.norm(mesh.vertices - c, axis=1))))
        sites = np.array(new_sites)


def fixtures_B(mesh, n_cells, n_outer, seed, c=4.0):
    """Iterated B: MBO -- diffuse the indicators, then balanced-threshold.

    Started from a BALANCED assignment (one C-scores assignment), which is what
    B does in practice and what removes the one workload the incumbent solver
    cannot handle. c=4 rather than 2: the plan's own tau criterion puts c=2 in
    the freeze regime, where the field is near-binary and balance is not
    representable as argmax(y + psi) by ANY method.
    """
    v, target = mesh.v, float(mesh.v.sum()) / n_cells
    graph = edge_graph(mesh)
    sites = farthest_point_seeds(mesh.vertices, n_cells, seed)
    d2 = -(dijkstra(graph, indices=sites, directed=False).T ** 2)
    labels, _ = _assign(d2, v, target)

    tau = (c * mean_edge_length(mesh)) ** 2
    solve = factorized((mesh.M + tau * mesh.K).tocsc())
    for it in range(n_outer):
        chi = np.zeros((len(mesh.vertices), n_cells))
        chi[np.arange(len(mesh.vertices)), labels] = 1.0
        scores = np.column_stack(
            [solve(mesh.M @ chi[:, k]) for k in range(n_cells)])
        yield it, scores
        labels, _ = _assign(scores, v, target)


def run_gate2(quick=False):
    print("=" * 76)
    print("GATE 2 -- foreign score scales converge on the arms' ACTUAL state")
    print("=" * 76)
    ok = True
    fixtures = FIXTURES[:2] if quick else FIXTURES
    for n_theta, n_phi, n_cells, n_outer, seeds in fixtures:
        mesh = build(n_theta, n_phi)
        v = mesh.v
        target = float(v.sum()) / n_cells
        gran = float(v.max()) / target
        print(f"\n--- V={len(v)}  N={n_cells}  granularity {gran*100:.3f}%")

        for arm, gen in (("C", fixtures_C), ("B", fixtures_B)):
            for seed in seeds:
                cold = None
                settled = []
                last_scores = None
                bar = assignment_quality_bar(v, n_cells)
                for it, scores in gen(mesh, n_cells, n_outer, seed):
                    # NO early stop here, deliberately. Early stopping is a
                    # PRODUCTION feature (B calls this solver every MBO step and
                    # should not burn budget once converged), but inside the gate
                    # it destroys the measurement: the solver halts the instant
                    # it crosses the bar, so every reported number hugs the bar
                    # from below and says nothing about achieved quality.
                    # Measured 2026-08-18 at the production pin: with early stop
                    # 0.885%, without it 0.620% on the identical scores. The
                    # verdict was unaffected -- both pass -- but the first run of
                    # this gate with early stop on reported numbers that were
                    # artefacts of the stopping rule, not of the solver.
                    cfg = BalancedReadoutConfig(normalize_scores=True)
                    # Grade from the RETURNED LABELS, not the solver's own
                    # best_worst. The measurement must not run through the code
                    # under test -- that is the pattern three reviews kept
                    # finding. (Gate 3 already did this.)
                    _, labels, _ = solve_dual_offsets(scores, v, target, cfg)
                    areas = np.zeros(n_cells)
                    np.add.at(areas, labels, v)
                    worst = float(np.abs((areas - target) / target).max())
                    if it == 0:
                        cold = worst
                    else:
                        settled.append(worst)
                    last_scores = scores

                # Context only. NEVER a bar term: it is computed with the solver
                # under test, so a broken solver would inflate its own bar in
                # proportion to how broken it is. Measured: six deliberately
                # crippled solvers, including one returning psi=0 with no
                # iterations at all, passed the previous self-referential bar.
                strong_cfg = BalancedReadoutConfig(
                    normalize_scores=True, normalized_dual_iters=STRONG_ITERS)
                _, _, strong = solve_dual_offsets(
                    last_scores, v, target, strong_cfg)

                # MEDIAN, not min. min ratchets monotonically with sampling:
                # measured on the production pin, C seed 0 over 8 outer
                # iterations gives min 0.799% but median 0.919% with no
                # convergence trend, and the min alone moved 0.919 -> 0.888 ->
                # 0.799 purely by sampling more.
                got = float(np.median(settled)) if settled else float("inf")
                passed = got <= bar
                ok &= passed
                sig = assignment_margin_scale(last_scores)
                print(f"  {arm} seed{seed}: cold {cold*100:7.3f}%  "
                      f"settled {got*100:6.3f}%  | strong-ref {strong*100:6.3f}%"
                      f"  bar {bar*100:6.3f}%  sigma {sig:.1e}  "
                      f"[{'PASS' if passed else 'FAIL'}]")
    print("\n  cold-start figures are INFORMATIONAL: the arms occupy that state "
          "for one\n  outer iteration and recover from it unaided.")
    return ok


def run_gate_selftest():
    """Gate 3 -- does gate 2 actually CATCH a broken solver?

    A gate nobody can fail is not evidence. The first version of gate 2 passed
    six deliberately crippled solvers, the weakest of which returned psi = 0 and
    did no iterations at all (45.4% on C, 80.2% on B, against self-scaled bars of
    57.2% and 124.3%). Two defects caused it: the bar included
    1.25 x strong-reference computed WITH THE SOLVER UNDER TEST, and the fixture
    trajectory degenerated along with the broken solver.

    This keeps the gate honest by asserting the null solver fails it. If this
    ever passes, gate 2's verdicts are worthless regardless of what they say.
    """
    print("=" * 76)
    print("GATE 3 -- gate 2 rejects a do-nothing solver (anti-theatre check)")
    print("=" * 76)
    n_theta, n_phi, n_cells = FIXTURES[0][:3]
    mesh = build(n_theta, n_phi)
    v = mesh.v
    target = float(v.sum()) / n_cells
    bar = assignment_quality_bar(v, n_cells)
    print(f"  V={len(v)} N={n_cells}  bar {bar*100:.3f}%")

    ok = True
    for arm, gen in (("C", fixtures_C), ("B", fixtures_B)):
        devs = []
        for it, scores in gen(mesh, n_cells, 4, 0):
            if it == 0:
                continue
            labels = np.argmax(scores, axis=1)          # psi = 0, no iterations
            areas = np.zeros(n_cells)
            np.add.at(areas, labels, v)
            devs.append(float(np.abs((areas - target) / target).max()))
        med = float(np.median(devs))
        caught = med > bar
        ok &= caught
        print(f"  {arm}: psi_zero settled(median) {med*100:7.3f}%  "
              f"[{'PASS -- correctly rejected' if caught else 'FAIL -- GATE IS THEATRE'}]")
    return ok


def run_gate_production_earlystop():
    """Gate 4 -- early stopping is safe INSIDE an iterated loop (B's real use).

    `dual_early_stop_rel` exists for production: B calls the assignment solver on
    every MBO step and should not burn its whole budget once converged. But that
    path had NO test. The failure it could hide is quiet and compounding -- if
    early stopping halts on a lucky iterate, every outer step starts from a
    slightly worse partition and the damage accumulates over the trajectory,
    while each individual assignment still looks "good enough".

    Method: run the same iterated C loop twice, with early stopping on and off
    INSIDE the loop, then grade BOTH final states with the same full-budget,
    no-early-stop solve. That isolates "did the trajectory degrade" from "did the
    last measurement stop early", which is precisely the confusion that made the
    first v4 gate run report numbers hugging the bar.
    """
    print("=" * 76)
    print("GATE 4 -- early stopping does not degrade an iterated trajectory")
    print("=" * 76)
    import time
    n_theta, n_phi, n_cells = FIXTURES[0][:3]
    mesh = build(n_theta, n_phi)
    v = mesh.v
    target = float(v.sum()) / n_cells
    bar = assignment_quality_bar(v, n_cells)
    graph = edge_graph(mesh)
    n_outer = 5
    print(f"  V={len(v)} N={n_cells}  bar {bar*100:.3f}%  {n_outer} outer iterations")

    def lloyd(cfg_kwargs):
        sites = farthest_point_seeds(mesh.vertices, n_cells, 0)
        t0 = time.perf_counter()
        for _ in range(n_outer):
            scores = -(dijkstra(graph, indices=sites, directed=False).T ** 2)
            cfg = BalancedReadoutConfig(normalize_scores=True, **cfg_kwargs)
            _, labels, _ = solve_dual_offsets(scores, v, target, cfg)
            new_sites = []
            for k in range(n_cells):
                m = labels == k
                if not m.any():
                    new_sites.append(sites[k])
                    continue
                c = (mesh.vertices[m] * v[m, None]).sum(0) / v[m].sum()
                new_sites.append(int(np.argmin(
                    np.linalg.norm(mesh.vertices - c, axis=1))))
            sites = np.array(new_sites)
        wall = time.perf_counter() - t0
        # Grade the FINAL state identically for both arms: full budget, no stop.
        final = -(dijkstra(graph, indices=sites, directed=False).T ** 2)
        _, _, worst = solve_dual_offsets(
            final, v, target, BalancedReadoutConfig(normalize_scores=True))
        return worst, wall

    off, off_t = lloyd({})
    on, on_t = lloyd({"dual_early_stop_rel": bar})
    print(f"  early stop OFF: final quality {off*100:6.3f}%   loop {off_t:6.1f}s")
    print(f"  early stop ON : final quality {on*100:6.3f}%   loop {on_t:6.1f}s"
          f"   ({off_t/max(on_t,1e-9):.1f}x faster)")

    # Degradation must be small relative to the bar, and it must be faster --
    # if it is not faster there is no reason to accept ANY degradation.
    degraded = on - off
    ok_quality = on <= bar and degraded <= 0.25 * bar
    ok_speed = on_t < off_t
    print(f"  degradation {degraded*100:+.3f} pts (allowed +{0.25*bar*100:.3f}) "
          f"[{'ok' if ok_quality else 'FAIL'}]   "
          f"speedup [{'ok' if ok_speed else 'FAIL'}]")
    return bool(ok_quality and ok_speed)


def run_gate1():
    """A is undisturbed: reproduce a recorded readout reference exactly."""
    print("=" * 76)
    print("GATE 1 -- approach A (log densities) is byte-for-byte undisturbed")
    print("=" * 76)
    root = Path(__file__).resolve().parents[1]
    hits = sorted(root.glob(
        "results/run_20260716_152451*/solution/surface_*.h5"))
    if not hits:
        print("  SKIP: reference solution not on disk")
        return None

    import h5py
    from src.mesh.tri_mesh import TriMesh
    from src.partition.balanced_readout import apply_balanced_readout

    with h5py.File(hits[0], "r") as f:
        x = f["x_opt"][:]
        verts, faces = f["vertices"][:], f["faces"][:]
        n_cells = int(f.attrs["n_partitions"])
    mesh = TriMesh(verts, faces)
    dens = x.reshape(len(verts), n_cells)

    cfg = BalancedReadoutConfig()
    assert cfg.normalize_scores is False, "default must keep A unchanged"
    res = apply_balanced_readout(dens, faces, verts, mesh.v, n_cells, cfg)
    st = res["stages"]["repaired"]
    reb = res.get("rebalance_report", {})
    got = (st["area_imbalance"]["n_imbalanced"],
           round(st["area_imbalance"]["worst_rel_dev"] * 100, 2),
           st["disconnected_cells"]["n_fragmented"],
           reb.get("n_moves"), reb.get("sweeps_used"))
    exp = (0, 1.80, 0, 2469, 54)
    print("  reference : 0 imbalanced, worst 1.80%, 0 fragmented, "
          "2469 moves, 54 sweeps")
    print(f"  recomputed: {got[0]} imbalanced, worst {got[1]}%, {got[2]} "
          f"fragmented, {got[3]} moves, {got[4]} sweeps")
    passed = got == exp
    print(f"  [{'PASS' if passed else 'FAIL'}]")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1", action="store_true")
    ap.add_argument("--gate2", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="gate 3 only: check gate 2 rejects a null solver")
    ap.add_argument("--production", action="store_true",
                    help="gate 4 only: early stopping is safe in an iterated loop")
    ap.add_argument("--quick", action="store_true",
                    help="skip the V=114,144/N=300 production pin")
    a = ap.parse_args()
    both = not (a.gate1 or a.gate2 or a.selftest or a.production)
    results = []
    if both or a.gate1:
        results.append(run_gate1())
    if both or a.selftest:
        results.append(run_gate_selftest())
    if both or a.production:
        results.append(run_gate_production_earlystop())
    if both or a.gate2:
        results.append(run_gate2(quick=a.quick))
    hard = [r for r in results if r is not None]
    print("\n" + "=" * 76)
    print("RESULT:", "PASS" if all(hard) else "FAIL")
    return 0 if all(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
