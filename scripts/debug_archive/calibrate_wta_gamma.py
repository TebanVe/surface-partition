#!/usr/bin/env python3
"""Calibrate the WTA balance strength gamma (territory-aware plan, section 2.5).

Procedure (recorded in docs/math/07-phase1-wta-balance/):
  1. Load a converged Phase 1 solution and rebuild its P1 matrices.
  2. Identify the interface-band vertices: those where the winning density is
     below a crispness cutoff (default u_win < 0.9), i.e. where the soft
     assignment w is spread over >1 cell. Only there can the balance force act
     (interface-band localization property).
  3. At each band vertex compute the local double-well gradient magnitude
     |g_well| = |(2/eps) (M q) (1 - 2u)| for the winning cell.
  4. Compute the balance-force magnitude a synthetic 5% territory deficit of
     the winning cell would produce there (all other cells balanced):
         |g_bal| = (gamma * p / Abar) * v_i * (u^(p-1)/S_i) * 0.05*(1 - w_i,win)
  5. gamma is chosen so that median_band(|g_bal| / |g_well|) = target ratio
     (default 0.10, the middle of the plan's 5-15% window).

Usage:
    python scripts/debug_archive/calibrate_wta_gamma.py \
        --run results/run_20260714_224821_... [--p 2] [--target-ratio 0.10]
"""

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from src.mesh.tri_mesh import TriMesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--target-ratio", type=float, default=0.10)
    ap.add_argument("--deficit", type=float, default=0.05)
    ap.add_argument("--band-cutoff", type=float, default=0.9,
                    help="Band vertex: winning density below this (default 0.9)")
    args = ap.parse_args()

    run = Path(args.run)
    sol_path = sorted(
        p for p in glob.glob(str(run / "solution" / "*.h5"))
        if not p.endswith("_refined_contours.h5")
    )[-1]
    with h5py.File(sol_path, "r") as f:
        x = np.array(f["x_opt"])
        vertices = np.array(f["vertices"])
        faces = np.array(f["faces"])
        N = int(f.attrs["n_partitions"])
    V = x.size // N
    U = x.reshape(V, N)

    print(f"solution: {sol_path}")
    print(f"V x N = {V} x {N}, p = {args.p}, deficit = {args.deficit}, "
          f"target ratio = {args.target_ratio}")
    t0 = time.time()
    mesh = TriMesh(vertices, faces)
    mesh.compute_matrices()
    stats = mesh.get_mesh_statistics()
    eps = float(np.sqrt(stats["mean_triangle_area"]))
    v = mesh.v
    A = float(v.sum())
    Abar = A / N
    print(f"matrices in {time.time() - t0:.1f}s; eps = {eps:.6e}, "
          f"Abar = {Abar:.6e}")

    # Double-well gradient field, per cell, restricted to the winner's column.
    winners = np.argmax(U, axis=1)
    u_win = U[np.arange(V), winners]
    Mq = np.empty((V,), dtype=np.float64)
    # (M q_k)_i for k = winner(i): loop over cells, scatter the winning rows.
    for k in range(N):
        rows = np.where(winners == k)[0]
        if rows.size == 0:
            continue
        u = U[:, k]
        q = u * (1 - u)
        Mq[rows] = (mesh.M @ q)[rows]
    g_well = np.abs((2.0 / eps) * Mq * (1.0 - 2.0 * u_win))

    # Soft weights at each vertex (power p).
    Up = U ** args.p
    S = Up.sum(axis=1)
    w_win = Up[np.arange(V), winners] / S

    band = u_win < args.band_cutoff
    n_band = int(band.sum())
    print(f"band vertices (u_win < {args.band_cutoff}): {n_band} "
          f"({n_band / V * 100:.1f}% of vertices)")

    # Balance force magnitude per unit gamma for a `deficit` of the winner:
    # r_win = -deficit, others 0 -> |r_win - m_i| = deficit * (1 - w_i,win).
    pref = (args.p / Abar) * v * (u_win ** (args.p - 1) / S)
    g_bal_unit = pref * args.deficit * (1.0 - w_win)

    ratio_unit = g_bal_unit[band] / np.maximum(g_well[band], 1e-300)
    med = float(np.median(ratio_unit))
    q25, q75 = np.percentile(ratio_unit, [25, 75])
    gamma = args.target_ratio / med
    print(f"per-unit-gamma force ratio on the band: median = {med:.4e}, "
          f"IQR = [{q25:.4e}, {q75:.4e}]")
    print(f"==> gamma for median ratio {args.target_ratio}: "
          f"gamma = {gamma:.3f}")
    for g in (round(gamma, 1), round(gamma), 2 * round(gamma)):
        r = g * med
        print(f"    gamma = {g:>6}: median band ratio = {r * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
