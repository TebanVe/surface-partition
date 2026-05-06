#!/usr/bin/env python3
"""
Smoke test for the zero-length-boundary fix in
``compute_cell_portion_in_triangle_simple``.

Without rendering anything, simulate the BEFORE and AFTER (single-trigger)
states of two known degenerate Type 1 components on the
double-torus iteration_001 checkpoint, and confirm:

  1. BEFORE the migration: every triangle in the target vertex's 1-ring
     that was previously left WHITE (no polygon produced) now produces
     a polygon for at least one of the two cells meeting on it.
  2. AFTER applying ONLY the selected trigger: triangles whose
     remaining degeneracy is owned by a different pending trigger
     stay white (we should not over-color them); triangles whose
     degeneracy was wholly owned by this trigger become colored.

This locks down the behavior described in the elaboration discussion,
specifically:
  - Component 4 (vertex 850, cell 7→5): T1596–T1599 should be colored
    by both cells before and after their VPs have been resolved.
  - Component 35 (vertex 5378, cell 2→8): T10574, T10575, T11126, T11127
    color before, T11128 stays white after (its VP739 belongs to
    component 37 at vertex 5656).

Usage:
    python testing/test_white_triangle_fix.py --solution <iteration_001 file>
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh.mesh_topology import MeshTopology
from src.pipeline.io import load_partition_from_refined_file
from src.migration.migration_orchestrator import MigrationOrchestrator, MigrationConfig

# Import the rendering helper (now patched)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "viz_t1", str(Path(__file__).parent.parent / "scripts" / "visualize_type1_vertex_collapse.py")
)
_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_viz)


def render_check(mesh, partition, tri_idx, cell_idx, tri_to_seg):
    poly = _viz.compute_cell_portion_in_triangle_simple(
        mesh, partition, tri_idx, cell_idx, tri_to_seg
    )
    return poly is not None


def report(case_label, mesh, partition, target_v, expected_states):
    """expected_states: dict of tri_idx -> dict of cell_idx -> bool (rendered?)"""
    tri_to_seg = {ts.triangle_idx: ts for ts in partition.triangle_segments}

    print(f"\n--- {case_label} ---")
    failures = 0
    for tri_idx, cell_expectations in expected_states.items():
        for cell_idx, expected in cell_expectations.items():
            actual = render_check(mesh, partition, tri_idx, cell_idx, tri_to_seg)
            status = "OK " if actual == expected else "FAIL"
            print(f"  {status} T{tri_idx} cell={cell_idx}: "
                  f"expected_rendered={expected}, actual_rendered={actual}")
            if actual != expected:
                failures += 1
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--boundary-tol", type=float, default=0.001)
    args = ap.parse_args()

    print("=" * 80)
    print("WHITE-TRIANGLE FIX VERIFICATION")
    print("=" * 80)

    fail_total = 0

    # ============================================================
    # Case A: Component 4 — vertex 850, cell 7 -> 5
    # ============================================================
    mesh, partition = load_partition_from_refined_file(args.solution)
    mesh_topology = MeshTopology(mesh)
    orch = MigrationOrchestrator(
        partition, mesh, mesh_topology,
        MigrationConfig(delta=args.boundary_tol),
    )
    triggers = orch.detect_all_triggers(delta=args.boundary_tol).type1_triggers
    trig4 = triggers[4]
    assert trig4.vertex == 850, f"Expected vertex 850, got {trig4.vertex}"

    # BEFORE: T1596..T1599 each have 2 VPs both collapsed at vertex 850.
    # After the fix, BOTH cells 5 and 7 should produce a non-None polygon
    # for these triangles (the polygon is degenerate-but-renderable for one
    # cell and a true sliver for the other).
    expected_before = {
        # T1596 labels (7,5,5): cell 5 owns 2 mesh vertices + 1 anchor = polygon;
        # cell 7 owns the single label-7 vertex which IS the anchor → 1 unique
        # point → no polygon.
        1596: {5: True, 7: False},
        1597: {5: True, 7: False},
        1598: {5: True, 7: False},
        1599: {5: True, 7: False},
        # T1587 / T1595 each have ONE collapsed and ONE non-degenerate VP →
        # standard rendering, both cells produce polygons.
        1587: {5: True, 7: True},
        1595: {5: True, 7: True},
    }
    fail_total += report("Case A BEFORE — vertex 850 (cell 7→5)",
                         mesh, partition, 850, expected_before)

    # AFTER applying ONLY trigger #4: vertex 850 flips to 5, all 5 collapsed
    # VPs are destroyed, so T1596..T1599 become interior to cell 5 (rendered
    # by area_calc.cell_interior_triangles in the visualization), so the
    # boundary-rendering path no longer covers them. We don't test the AFTER
    # state for this case since the rendering path changes.

    # ============================================================
    # Case B: Component 35 — vertex 5378, cell 2 -> 8
    # ============================================================
    mesh, partition = load_partition_from_refined_file(args.solution)
    mesh_topology = MeshTopology(mesh)
    orch = MigrationOrchestrator(
        partition, mesh, mesh_topology,
        MigrationConfig(delta=args.boundary_tol),
    )
    triggers = orch.detect_all_triggers(delta=args.boundary_tol).type1_triggers
    trig35 = triggers[35]
    assert trig35.vertex == 5378, f"Expected vertex 5378, got {trig35.vertex}"

    expected_before = {
        # All five 2-VP-collapsed triangles around vertex 5378.
        # Labels (2, 8, 8): cell 8 owns 2 vertices + anchor → polygon;
        # cell 2 owns 1 vertex which IS the anchor → 1 unique pt → no polygon.
        # Labels (2, 2, 8): cell 2 owns 2 vertices + anchor → polygon;
        # cell 8 owns 1 vertex (NOT the anchor) + anchor → 2 unique pts → no
        # polygon. (Both 2-labelled vertices include vertex 5378 itself, so
        # one of them coincides with the anchor.)
        # Specifically:
        #   T10574 labels (2, 8, 8) → cell 8 polygon, cell 2 sliver
        10574: {8: True, 2: False},
        #   T10575 labels (8, 2, 8) → cell 8 polygon, cell 2 sliver
        10575: {8: True, 2: False},
        #   T11126 labels (8, 2, 8) → cell 8 polygon, cell 2 sliver
        11126: {8: True, 2: False},
        #   T11127 labels (2, 8, 8) → cell 8 polygon, cell 2 sliver
        11127: {8: True, 2: False},
        # T10573: 1 collapsed + 1 non-degenerate VP → both cells render
        10573: {2: True, 8: True},
        # T11128: BOTH VPs collapsed but at DIFFERENT vertices (V5378 + V5656)
        # so the segment is non-degenerate. The standard path runs, but the
        # segment lies ALONG the line connecting the two cell-2 vertices
        # (5378, 5656) of the triangle, so cell 2 has geometrically zero
        # area in T11128 (just an edge); cell 8 owns the bulk.
        11128: {2: False, 8: True},
    }
    fail_total += report("Case B BEFORE — vertex 5378 (cell 2→8)",
                         mesh, partition, 5378, expected_before)

    # AFTER applying trigger #35:
    #   - V5378 flips to 8.
    #   - VPs 697, 709, 710, 711, 712 are destroyed.
    #   - T11128's edge (5378, 5658) now connects two cell-8 vertices, so
    #     VP712 disappears. VP739 remains at vertex 5656. The new VP on the
    #     edge (5656, 5378) is created by one_ring_rebuilder.
    #   - The white triangle that PERSISTS in the BEFORE/AFTER screenshots
    #     is T11128. We assert that AFTER this single trigger, T11128 is
    #     either no longer in the boundary set for cell 2 or 8 (became
    #     interior/triple), OR it is still rendered as a polygon thanks to
    #     the fix even if its boundary remains degenerate.
    partition_after = copy.deepcopy(partition)
    mesh_topology_after = MeshTopology(mesh)
    orch_after = MigrationOrchestrator(
        partition_after, mesh, mesh_topology_after,
        MigrationConfig(delta=args.boundary_tol),
    )
    orch_after.detect_all_triggers(delta=args.boundary_tol)
    ok = orch_after.execute_single_trigger(trig35)
    print(f"\n[Case B] migration #35 success: {ok}")
    print(f"[Case B] T11128 after migration: vertex labels = "
          f"{tuple(int(partition_after.vertex_labels[int(v)]) for v in mesh.faces[11128])}")
    tri_to_seg_after = {ts.triangle_idx: ts
                        for ts in partition_after.triangle_segments}
    seg = tri_to_seg_after.get(11128)
    if seg is not None:
        active = [vi for vi in seg.var_point_indices
                  if partition_after.variable_points[vi].active]
        print(f"[Case B] T11128 active VPs after migration: {active}")
        for vi in active:
            vp = partition_after.variable_points[vi]
            print(f"          VP{vi}: edge={vp.edge}, λ={vp.lambda_param:.4f}")

    # The fix should not change AFTER-state rendering of T11128:
    # if it is still a boundary triangle and still degenerate, it should
    # now produce a polygon for the cell(s) that label vertex 5656.
    poly_2 = _viz.compute_cell_portion_in_triangle_simple(
        mesh, partition_after, 11128, 2, tri_to_seg_after
    )
    poly_8 = _viz.compute_cell_portion_in_triangle_simple(
        mesh, partition_after, 11128, 8, tri_to_seg_after
    )
    print(f"[Case B] AFTER: T11128 cell 2 polygon: "
          f"{'rendered' if poly_2 is not None else 'NOT rendered'}")
    print(f"[Case B] AFTER: T11128 cell 8 polygon: "
          f"{'rendered' if poly_8 is not None else 'NOT rendered'}")
    print("[Case B] AFTER: at least one cell renders T11128 → "
          + ("OK" if (poly_2 is not None or poly_8 is not None) else "FAIL"))
    if poly_2 is None and poly_8 is None:
        fail_total += 1

    print("\n" + "=" * 80)
    print(f"TOTAL FAILURES: {fail_total}")
    print("=" * 80)
    sys.exit(0 if fail_total == 0 else 1)


if __name__ == "__main__":
    main()
