#!/usr/bin/env python3
"""
Diagnostic: Type 1 triggers at triple-point vertices.

Loads a refined partition, detects Type 1 triggers, and cross-references them
with triple-point triangle vertices to confirm whether triggers with >3
approaching VPs correspond to vertices in (or adjacent to) triple-point
triangles — which should NOT be handled as Type 1 migrations.

Usage:
    python testing/test_type1_triple_point_overlap.py \
        --solution results/.../iteration_001_....h5 \
        --boundary-tol 0.001
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.io import load_partition_from_refined_file
from src.mesh.mesh_topology import MeshTopology
from src.partition.steiner_handler import SteinerHandler
from src.migration.migration_detector import detect_type1_triggers
from src.migration import migration_utils
from src.logging_config import get_logger, setup_logging


def get_one_ring_cells(vertex, mesh_topology, vertex_labels):
    """Return the set of distinct cell labels in the one-ring of a vertex."""
    cells = set()
    for edge in mesh_topology.get_edges_at_vertex(vertex):
        other = edge[1] if edge[0] == vertex else edge[0]
        cells.add(int(vertex_labels[other]))
    return cells


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Type 1 triggers at triple-point vertices"
    )
    parser.add_argument(
        "--solution", required=True,
        help="Path to refined contours HDF5 file"
    )
    parser.add_argument(
        "--boundary-tol", type=float, default=1e-3,
        help="Detection delta for Type 1 triggers (default: 1e-3)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)

    # -- Load partition --
    print(f"Loading: {args.solution}")
    mesh, partition = load_partition_from_refined_file(args.solution, verbose=True)
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Partition: {len(partition.variable_points)} VPs, {partition.n_cells} cells")

    # -- Build topology --
    mesh_topology = MeshTopology(mesh)

    # -- Detect triple points --
    steiner_handler = SteinerHandler(mesh, partition)
    triple_points = steiner_handler.triple_points
    print(f"Triple points detected: {len(triple_points)}")

    # Build set of vertices belonging to triple-point triangles
    tp_vertex_set = set()
    # Map: vertex -> list of (triangle_idx, cell_indices) for reporting
    vertex_to_tp_info = defaultdict(list)
    for tp in triple_points:
        for v in tp.vertex_indices:
            tp_vertex_set.add(v)
            vertex_to_tp_info[v].append((tp.triangle_idx, tp.cell_indices))

    print(f"Vertices in triple-point triangles: {len(tp_vertex_set)}")

    # -- Run Type 1 detection --
    triggers = detect_type1_triggers(partition, mesh_topology, delta=args.boundary_tol)
    print(f"\nType 1 triggers detected: {len(triggers)}")

    # -- Classify triggers --
    triggers_3vp = []
    triggers_gt3vp = []
    for t in triggers:
        if len(t.approaching_vps) > 3:
            triggers_gt3vp.append(t)
        else:
            triggers_3vp.append(t)

    # -- Report triggers with >3 VPs --
    print("\n" + "=" * 80)
    print(f"TYPE 1 TRIGGERS WITH >3 APPROACHING VPs ({len(triggers_gt3vp)} found)")
    print("=" * 80)

    vertex_labels = partition.vertex_labels
    in_tp_triangle_count = 0

    for t in sorted(triggers_gt3vp, key=lambda x: -len(x.approaching_vps)):
        n_vps = len(t.approaching_vps)
        in_tp = t.vertex in tp_vertex_set

        if in_tp:
            in_tp_triangle_count += 1

        print(f"\n  Vertex {t.vertex} (cell {t.current_cell} -> {t.target_cell}): "
              f"{n_vps} approaching VPs")

        # Triple-point membership
        if in_tp:
            for tri_idx, cells in vertex_to_tp_info[t.vertex]:
                print(f"    In triple-point triangle? YES "
                      f"(triangle {tri_idx}, cells {sorted(cells)})")
        else:
            print(f"    In triple-point triangle? NO")

        # One-ring cell analysis
        one_ring_cells = get_one_ring_cells(t.vertex, mesh_topology, vertex_labels)
        non_current = one_ring_cells - {t.current_cell}
        print(f"    One-ring cells: {sorted(one_ring_cells)} "
              f"({len(non_current)} distinct non-current cell(s))")

        # VP details
        print(f"    VP details:")
        for vp_idx in t.approaching_vps:
            vp = partition.variable_points[vp_idx]
            other_v = vp.edge[1] if vp.edge[0] == t.vertex else vp.edge[0]
            other_cell = int(vertex_labels[other_v])
            dist = migration_utils.compute_boundary_distance(vp)
            print(f"      VP {vp_idx}: edge={vp.edge}, other_vertex={other_v}, "
                  f"other_cell={other_cell}, dist={dist:.6f}")

    # -- Also check if any 3-VP triggers are in triple-point triangles --
    triggers_3vp_in_tp = [t for t in triggers_3vp if t.vertex in tp_vertex_set]

    # -- Summary --
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total Type 1 triggers: {len(triggers)}")
    print(f"    With exactly 3 VPs: {len(triggers_3vp)}")
    print(f"    With >3 VPs: {len(triggers_gt3vp)}")
    print(f"  Of >3 VP triggers in triple-point triangles: "
          f"{in_tp_triangle_count} / {len(triggers_gt3vp)}")
    print(f"  Of 3 VP triggers in triple-point triangles: "
          f"{len(triggers_3vp_in_tp)} / {len(triggers_3vp)}")
    print()

    if in_tp_triangle_count > 0:
        print("  CONCLUSION: Type 1 detection is firing on vertices that belong")
        print("  to triple-point triangles. These should be excluded from Type 1")
        print("  analysis — they represent 3-cell junctions, not 2-cell boundary flips.")
    elif len(triggers_gt3vp) > 0:
        print("  NOTE: >3 VP triggers exist but are NOT in triple-point triangles.")
        print("  These may be high-valence vertices deeply enclosed by the target cell,")
        print("  or vertices adjacent to (but not part of) triple-point triangles.")
    else:
        print("  All Type 1 triggers have exactly 3 approaching VPs — no anomalies.")


if __name__ == "__main__":
    main()
