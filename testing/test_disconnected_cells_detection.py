"""Smoke test for detect_disconnected_cells (Phase 1 connectivity gate).

A cell whose winner-take-all territory splits into disconnected islands passes
both detect_dormant_cells (each piece is crisp) and detect_area_imbalance (the
pieces sum to the target area), so connectivity is a third, independent validity
check. This test builds a tiny hand-made mesh with a deliberately split cell and
verifies (1) a genuine split is flagged and (2) a below-threshold stray piece is
treated as speckle and NOT flagged.

Run:
    python testing/test_disconnected_cells_detection.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.partition.find_contours import detect_disconnected_cells


def _strip_mesh(n_cols=6, n_rows=2):
    """A flat triangulated strip: n_rows x n_cols grid, no periodic wrap.

    Returns (vertices unused here, faces). Vertex index = row * n_cols + col.
    """
    faces = []
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            a = r * n_cols + c
            b = r * n_cols + (c + 1)
            d = (r + 1) * n_cols + c
            e = (r + 1) * n_cols + (c + 1)
            faces.append([a, b, d])
            faces.append([b, e, d])
    return np.asarray(faces, dtype=int)


def main() -> int:
    n_cols, n_rows = 6, 2
    n_vertices = n_cols * n_rows
    faces = _strip_mesh(n_cols, n_rows)

    # Cell 0 owns columns {0,1} and {4,5} (two blocks separated by cell 1's
    # columns {2,3}) -> cell 0's territory is disconnected into 2 pieces.
    densities = np.zeros((n_vertices, 2))
    for r in range(n_rows):
        for c in range(n_cols):
            k = 0 if c in (0, 1, 4, 5) else 1
            densities[r * n_cols + c, k] = 1.0

    failures = []

    # --- Case 1: uniform mass -> both blocks are significant, cell 0 flagged ---
    mass = np.ones(n_vertices)
    res = detect_disconnected_cells(densities, faces, mass)
    if res['n_components_per_cell'][0] != 2:
        failures.append(
            f"cell 0 raw component count = {res['n_components_per_cell'][0]}, "
            f"expected 2"
        )
    if res['n_components_per_cell'][1] != 1:
        failures.append(
            f"cell 1 raw component count = {res['n_components_per_cell'][1]}, "
            f"expected 1"
        )
    if res['fragmented'] != [0]:
        failures.append(f"fragmented = {res['fragmented']}, expected [0]")
    if res['n_fragmented'] != 1:
        failures.append(f"n_fragmented = {res['n_fragmented']}, expected 1")
    if res['fragmented'] == [0]:
        det = res['details'][0]
        if det['n_components'] != 2 or len(det['component_areas']) != 2:
            failures.append(f"details wrong: {det}")
        # two equal 4-vertex blocks, target = 12/2 = 6 -> stray_rel = 4/6
        if abs(det['stray_rel'] - (4.0 / 6.0)) > 1e-9:
            failures.append(
                f"stray_rel = {det['stray_rel']}, expected {4.0/6.0}"
            )
    print(f"[case 1: uniform mass] fragmented={res['fragmented']} "
          f"worst_stray_rel={res['worst_stray_rel']:.4f}  "
          f"-> {'OK' if not failures else 'FAIL'}")

    # --- Case 2: right block is tiny -> below threshold -> speckle, NOT flagged ---
    mass2 = np.ones(n_vertices)
    for r in range(n_rows):
        for c in (4, 5):
            mass2[r * n_cols + c] = 1e-3
    res2 = detect_disconnected_cells(densities, faces, mass2)
    case2_fail = []
    if res2['n_components_per_cell'][0] != 2:
        case2_fail.append(
            f"raw component count still expected 2, got "
            f"{res2['n_components_per_cell'][0]}"
        )
    if res2['fragmented'] != []:
        case2_fail.append(
            f"tiny stray should be speckle (not flagged); got "
            f"fragmented={res2['fragmented']}"
        )
    failures.extend(case2_fail)
    print(f"[case 2: tiny stray]   fragmented={res2['fragmented']} "
          f"(raw components for cell 0 = {res2['n_components_per_cell'][0]})  "
          f"-> {'OK' if not case2_fail else 'FAIL'}")

    # --- Case 3: fully connected cells -> nothing flagged ---
    dens3 = np.zeros((n_vertices, 2))
    for r in range(n_rows):
        for c in range(n_cols):
            k = 0 if c < 3 else 1
            dens3[r * n_cols + c, k] = 1.0
    res3 = detect_disconnected_cells(dens3, faces, np.ones(n_vertices))
    case3_fail = []
    if res3['n_fragmented'] != 0 or res3['fragmented']:
        case3_fail.append(f"expected no fragments; got {res3['fragmented']}")
    if res3['n_components_per_cell'] != [1, 1]:
        case3_fail.append(
            f"expected [1, 1] components; got {res3['n_components_per_cell']}"
        )
    failures.extend(case3_fail)
    print(f"[case 3: connected]    fragmented={res3['fragmented']}  "
          f"-> {'OK' if not case3_fail else 'FAIL'}")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All connectivity-detector checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
