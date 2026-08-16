#!/usr/bin/env python3
"""Phase 0a acceptance gates for the shared balanced-assignment solver.

Approaches A, B and C all need the same primitive -- balanced assignment given
arbitrary per-vertex scores -- which ships inside the balanced readout as
`solve_dual_offsets`. Generalizing it must not disturb A, and must actually
converge on the score scales B and C produce.

Gate 1 (A is undisturbed): the shipped readout path reproduces a recorded
reference exactly, with normalization off.

Gate 2 (foreign scales converge): synthetic `-d^2` (C) and diffused-indicator
(B) score matrices reach worst |area dev| <= max(1%, 2 x vertex granularity) at
the two meshes the plan pins. A flat 1% bar is INFEASIBLE at V=47,488/N=300,
where the one-vertex granularity floor alone is ~1.01%.

Usage:
    python testing/test_balanced_assignment_solver.py            # both gates
    python testing/test_balanced_assignment_solver.py --gate2    # foreign only
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

# (n_theta, n_phi, N) -- the meshes the plan pins for arm comparison.
PINNED = [(224, 212, 300), (348, 328, 100)]


def build(n_theta, n_phi):
    prov = TorusMeshProvider(n_theta=n_theta, n_phi=n_phi, R=1.0, r=0.6)
    return prov.build()


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
    """Sparse graph of mesh edges weighted by Euclidean edge length."""
    f = mesh.faces
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    w = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    n = len(mesh.vertices)
    g = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return g.maximum(g.T)


def scores_C(mesh, n_cells):
    """Approach C: -d^2, geodesic distance approximated on the edge graph."""
    seeds = farthest_point_seeds(mesh.vertices, n_cells)
    d = dijkstra(edge_graph(mesh), indices=seeds, directed=False)
    return -(d.T ** 2)


def scores_B(mesh, n_cells, c=2.0):
    """Approach B: one diffusion step of one-hot indicators, tau = (c*h)^2."""
    seeds = farthest_point_seeds(mesh.vertices, n_cells)
    lab = np.argmin(
        np.stack([np.linalg.norm(mesh.vertices - mesh.vertices[s], axis=1)
                  for s in seeds]), axis=0)
    chi = np.zeros((len(mesh.vertices), n_cells))
    chi[np.arange(len(mesh.vertices)), lab] = 1.0
    f = mesh.faces
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    h = float(np.mean(np.linalg.norm(
        mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)))
    tau = (c * h) ** 2
    solve = factorized((mesh.M + tau * mesh.K).tocsc())
    return np.column_stack([solve(mesh.M @ chi[:, k]) for k in range(n_cells)])


def run_gate2():
    print("=" * 74)
    print("GATE 2 -- foreign score scales converge")
    print("=" * 74)
    ok = True
    for n_theta, n_phi, n_cells in PINNED:
        mesh = build(n_theta, n_phi)
        v = mesh.v
        V = len(v)
        target = float(v.sum()) / n_cells
        granularity = float(v.max()) / target
        bar = max(0.01, 2.0 * granularity)
        print(f"\n--- V={V}, N={n_cells} | granularity {granularity*100:.3f}% "
              f"| bar max(1%, 2x) = {bar*100:.3f}%")

        for name, scores in (("C: -d^2", scores_C(mesh, n_cells)),
                             ("B: diffused", scores_B(mesh, n_cells))):
            sigma = assignment_margin_scale(scores)
            row = []
            for norm in (False, True):
                cfg = BalancedReadoutConfig(normalize_scores=norm)
                _, _, worst = solve_dual_offsets(scores, v, target, cfg)
                row.append(worst)
            off, on = row
            passed = on <= bar
            ok &= passed
            print(f"  {name:<14} margin sigma={sigma:.3e} | "
                  f"norm OFF {off*100:8.3f}%  ->  norm ON {on*100:7.3f}%  "
                  f"[{'PASS' if passed else 'FAIL'}]")
    return ok


def run_gate1():
    """A is undisturbed: reproduce a recorded readout reference exactly."""
    print("=" * 74)
    print("GATE 1 -- approach A (log densities) is byte-for-byte undisturbed")
    print("=" * 74)
    root = Path(__file__).resolve().parents[1]
    hits = sorted(root.glob(
        "results/run_20260716_152451*/solution/surface_*.h5"))
    if not hits:
        print("  SKIP: reference solution not on disk")
        return None

    import h5py
    from src.partition.balanced_readout import apply_balanced_readout

    with h5py.File(hits[0], "r") as f:
        x = f["x_opt"][:]
        verts, faces = f["vertices"][:], f["faces"][:]
        n_cells = int(f.attrs["n_partitions"])
    from src.mesh.tri_mesh import TriMesh
    mesh = TriMesh(verts, faces)
    dens = x.reshape(len(verts), n_cells)

    cfg = BalancedReadoutConfig()
    assert cfg.normalize_scores is False, "default must keep A unchanged"
    res = apply_balanced_readout(
        dens, faces, verts, mesh.v, n_cells, cfg
    )
    st = res["stages"]["repaired"]
    reb = res.get("rebalance_report", {})
    got = (st["area_imbalance"]["n_imbalanced"],
           round(st["area_imbalance"]["worst_rel_dev"] * 100, 2),
           st["disconnected_cells"]["n_fragmented"],
           reb.get("n_moves"),
           reb.get("sweeps_used"))
    # Recorded 2026-08-15 from scripts/balanced_readout.py on this solution.
    exp = (0, 1.80, 0, 2469, 54)
    print(f"  reference : 0 imbalanced, worst 1.80%, 0 fragmented, "
          f"2469 moves, 54 sweeps")
    print(f"  recomputed: {got[0]} imbalanced, worst {got[1]}%, {got[2]} "
          f"fragmented, {got[3]} moves, {got[4]} sweeps")
    passed = got == exp
    print(f"  [{'PASS' if passed else 'FAIL'}]")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1", action="store_true")
    ap.add_argument("--gate2", action="store_true")
    a = ap.parse_args()
    both = not (a.gate1 or a.gate2)
    results = []
    if both or a.gate1:
        results.append(run_gate1())
    if both or a.gate2:
        results.append(run_gate2())
    hard = [r for r in results if r is not None]
    print("\n" + "=" * 74)
    print("RESULT:", "PASS" if all(hard) else "FAIL")
    return 0 if all(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
