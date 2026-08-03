#!/usr/bin/env python
"""Run the three Phase 1 validity gates on any solution or per-level checkpoint.

``run_relaxation`` only evaluates ``detect_dormant_cells`` /
``detect_area_imbalance`` / ``detect_disconnected_cells`` on the FINAL solution
(``src/pipeline/relaxation.py`` lines 443-447), so a run that is still on the
mesh ladder -- or one that died before the last level -- never reports them.
The connectivity gate in particular is the only test for a cell whose
winner-take-all territory has split into disconnected islands: such a cell
passes the dormant and area-imbalance checks (its pieces sum to the target
area and each is crisp), so neither of the other two numbers says anything
about it.

This runs all three against a ``solution/*.h5`` or ``solution/checkpoint_level*.h5``
written by Phase 1, so mid-ladder validity can be inspected without waiting for
(or restarting) the run.

Usage:
    python testing/check_fragmentation.py <path_to_solution_or_checkpoint.h5>
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mesh.tri_mesh import TriMesh  # noqa: E402
from src.partition.find_contours import (  # noqa: E402
    detect_area_imbalance,
    detect_disconnected_cells,
    detect_dormant_cells,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Phase 1 solution or checkpoint HDF5')
    args = parser.parse_args()

    with h5py.File(args.checkpoint, 'r') as f:
        x_opt = np.array(f['x_opt'])
        vertices = np.array(f['vertices'])
        faces = np.array(f['faces'])
        n_partitions = int(f.attrs['n_partitions'])
        completed_levels = int(f.attrs.get('completed_levels', -1))
        wta_active = f.attrs.get('wta_active', None)

    n_vertices = vertices.shape[0]
    densities = x_opt.reshape(n_vertices, n_partitions)

    print(f"file              : {args.checkpoint}")
    print(f"levels completed  : {completed_levels}")
    print(f"mesh              : V={n_vertices}, N={n_partitions}")
    if wta_active is not None:
        print(f"wta_active        : {bool(wta_active)}")
    print("assembling FEM matrices (this takes a few seconds) ...")

    mesh = TriMesh(vertices, faces)

    dormant = detect_dormant_cells(densities)
    imbalance = detect_area_imbalance(densities, mesh.v, n_partitions)
    disconnected = detect_disconnected_cells(densities, faces, mesh.v)

    print()
    print("--- dormant gate ---")
    print(f"  dead cells      : {len(dormant['dead'])} {dormant['dead'][:10]}")
    print(f"  weak cells      : {len(dormant['weak'])} {dormant['weak'][:10]}")
    print(f"  min peak density: {min(dormant['max_density_per_cell']):.4f}")

    print()
    print("--- area-imbalance gate (threshold "
          f"{imbalance['rel_threshold']:.0%}) ---")
    print(f"  n_imbalanced    : {imbalance['n_imbalanced']}")
    print(f"  worst cell      : {imbalance['worst_cell']} "
          f"at {imbalance['worst_rel_dev']:.2%}")
    print(f"  imbalanced      : {imbalance['imbalanced'][:10]}")

    print()
    print("--- connectivity gate (fragment threshold "
          f"{disconnected['fragment_rel_threshold']:.0%} of target) ---")
    print(f"  n_fragmented    : {disconnected['n_fragmented']}")
    print(f"  fragmented      : {disconnected['fragmented'][:10]}")
    if disconnected['n_fragmented']:
        print(f"  worst cell      : {disconnected['worst_cell']} "
              f"stray {disconnected['worst_stray_rel']:.2%} of target")
        for detail in disconnected['details'][:5]:
            print(f"    {detail}")

    n_multi = sum(1 for c in disconnected['n_components_per_cell'] if c > 1)
    print(f"  cells with >1 raw component (incl. speckle): {n_multi}")

    print()
    ok = (not dormant['dead']
          and imbalance['n_imbalanced'] == 0
          and disconnected['n_fragmented'] == 0)
    print("VERDICT: " + ("all three gates PASS" if ok else "GATE FAILURE — "
          f"dead={len(dormant['dead'])}, "
          f"imbalanced={imbalance['n_imbalanced']}, "
          f"fragmented={disconnected['n_fragmented']}"))


if __name__ == '__main__':
    main()
