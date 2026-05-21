#!/usr/bin/env python3
"""Verify the degenerate-triangle branch of the analytical Steiner derivatives.

When a triple point's triangle has an angle >= 120 degrees the Fermat point
collapses onto the obtuse vertex.  The derivatives there are non-smooth and the
analytical code takes a dedicated branch:
    S          = obtuse vertex
    dS/dp_k    = I  if k == obtuse  else 0
    d2S/dp_k.. = 0

This case almost never fires on a real mesh, so it is exercised here on a
hand-built synthetic PartitionArrays whose single triple point is a 125-degree
isoceles triangle.

Checks:
  1. _compute_tp_steiner detects the degeneracy and reports the obtuse slot.
  2. _compute_dS_dp returns I on the obtuse slot, 0 on the others.
  3. _compute_d2S_dp2 is identically zero.
  4. The analytical gradient and Hessians stay finite (no 0/0).
  5. For each cell whose VP pair contains the obtuse vertex, the Steiner
     perimeter contribution ||p_a-S|| + ||p_b-S|| - ||p_a-p_b|| is zero.

Usage:
    python testing/test_steiner_degenerate_case.py

Exit code: 0 on PASS, 1 on FAIL.  (No --solution: the geometry is synthetic.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.partition.partition_arrays import PartitionArrays  # noqa: E402
from src.partition import vectorized_steiner as vs  # noqa: E402


def _build_degenerate_pa():
    """Synthetic PartitionArrays: one triple point, a 125-degree triangle.

    VP k sits exactly on triangle corner k (edge_v1 = corner, lambda = 1).
    The apex angle at corner 0 is 125 degrees, so corner 0 is the obtuse
    vertex and slot 0 is degenerate.
    """
    ang = np.deg2rad(125.0)
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([np.cos(ang), np.sin(ang), 0.0])     # 125 deg from +x at p0
    # 6 mesh vertices: 3 triangle corners + 3 distinct dummies (edge_v2 ends).
    vertices = np.array([
        p0, p1, p2,
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], p2 + np.array([0.0, 0.0, 1.0]),
    ], dtype=np.float64)

    n_vp, n_cells = 3, 3
    i32 = lambda a: np.asarray(a, dtype=np.int32)        # noqa: E731

    # 6 lower-triangle Hessian entries for the 3 VPs {0,1,2}.
    hess_row = i32([0, 1, 1, 2, 2, 2])
    hess_col = i32([0, 0, 1, 0, 1, 2])
    hess_offset_map = {(int(r), int(c)): k
                       for k, (r, c) in enumerate(zip(hess_row, hess_col))}
    tp_hess_off = np.empty((1, 3, 3), dtype=np.int32)
    for i in range(3):
        for j in range(3):
            tp_hess_off[0, i, j] = hess_offset_map[(max(i, j), min(i, j))]

    empty_i = i32([])
    return PartitionArrays(
        vp_edge_v1=i32([0, 1, 2]),
        vp_edge_v2=i32([3, 4, 5]),
        vp_lambda=np.ones(n_vp, dtype=np.float64),       # VP k -> corner k
        seg_vp1=empty_i, seg_vp2=empty_i,
        seg_cell_a=empty_i, seg_cell_b=empty_i,
        btri_idx=empty_i, btri_cell=empty_i, btri_n_inside=empty_i,
        btri_v_in=np.empty((0, 2), dtype=np.int32),
        btri_v_out=np.empty((0, 2), dtype=np.int32),
        btri_vp1=empty_i, btri_vp2=empty_i,
        cell_interior_area=np.zeros(n_cells),
        n_cells=n_cells, n_active_vp=n_vp,
        active_to_absolute=i32([0, 1, 2]),
        area_affected_vps=empty_i,
        vertices=vertices,
        tp_vp_indices=i32([[0, 1, 2]]),
        n_triple_points=1,
        tp_contrib_tp_idx=i32([0, 0, 0]),
        tp_contrib_cell=i32([0, 1, 2]),
        tp_contrib_vp1=i32([0, 1, 2]),
        tp_contrib_vp2=i32([1, 2, 0]),
        tp_contrib_mesh_vertex=i32([3, 4, 5]),
        tp_affected_vps=i32([0, 1, 2]),
        tp_contrib_slot1=i32([0, 1, 2]),
        tp_contrib_slot2=i32([1, 2, 0]),
        jac_row=empty_i, jac_col=empty_i,
        nnz_lookup=-np.ones((n_cells - 1, n_vp), dtype=np.int32),
        hess_row=hess_row, hess_col=hess_col,
        hess_offset_map=hess_offset_map,
        tp_hess_off=tp_hess_off,
    )


def main() -> int:
    vs._STEINER_SELF_CHECK = True
    pa = _build_degenerate_pa()
    p0 = pa.vertices[0]
    ok = True

    print("Degenerate Steiner case — synthetic 125-degree triple point")

    # 1. Degeneracy detection.
    p, S, degen_mask, obtuse_idx = vs._compute_tp_steiner(pa)
    c1 = bool(degen_mask[0]) and int(obtuse_idx[0]) == 0 \
        and np.allclose(S[0], p0)
    print(f"  [1] detection: degen={bool(degen_mask[0])}, "
          f"obtuse_slot={int(obtuse_idx[0])}, S==p0={np.allclose(S[0], p0)}"
          f"   [{'ok' if c1 else 'FAIL'}]")
    ok &= c1

    # 2. dS/dp_k: identity on the obtuse slot, zero elsewhere.
    geom = vs._compute_tp_geometry(pa)
    dS_dp = vs._compute_dS_dp(geom)
    c2 = (np.allclose(dS_dp[0, 0], np.eye(3))
          and np.allclose(dS_dp[0, 1], 0.0)
          and np.allclose(dS_dp[0, 2], 0.0))
    print(f"  [2] dS/dp_k = delta_{{k,obtuse}} I   [{'ok' if c2 else 'FAIL'}]")
    ok &= c2

    # 3. d2S/dp2 identically zero.
    d2S = vs._compute_d2S_dp2(geom, dS_dp)
    c3 = np.allclose(d2S, 0.0)
    print(f"  [3] d2S/dp2 == 0   (max|.|={np.max(np.abs(d2S)):.2e})   "
          f"[{'ok' if c3 else 'FAIL'}]")
    ok &= c3

    # 4. Analytical gradient / Hessians stay finite.
    grad = vs.compute_steiner_perimeter_gradient_analytical(pa)
    hp = vs.compute_steiner_perimeter_hessian_analytical(pa)
    ha = vs.compute_steiner_area_hessian_analytical(pa, np.ones(pa.n_cells - 1))
    jac = vs.compute_steiner_area_jacobian_analytical(pa)
    c4 = all(np.all(np.isfinite(x)) for x in (grad, hp, ha, jac))
    print(f"  [4] analytical gradient/Jacobian/Hessians finite "
          f"[{'ok' if c4 else 'FAIL'}]")
    ok &= c4

    # 5. Steiner perimeter contribution is zero for cells touching the obtuse VP.
    pos = vs._compute_vp_positions(pa)
    worst = 0.0
    for row in range(len(pa.tp_contrib_tp_idx)):
        a, b = pa.tp_contrib_vp1[row], pa.tp_contrib_vp2[row]
        if 0 not in (a, b):                  # obtuse VP is index 0
            continue
        contrib = (np.linalg.norm(pos[a] - S[0]) + np.linalg.norm(pos[b] - S[0])
                   - np.linalg.norm(pos[a] - pos[b]))
        worst = max(worst, abs(contrib))
    c5 = worst < 1e-12
    print(f"  [5] degenerate Steiner perimeter contribution == 0 "
          f"(max|.|={worst:.2e})   [{'ok' if c5 else 'FAIL'}]")
    ok &= c5

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
