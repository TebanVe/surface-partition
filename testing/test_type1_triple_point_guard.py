#!/usr/bin/env python3
"""
Smoke test for the triple-point safety guard in detect_type1_triggers.

Compares Type 1 trigger counts with and without the SteinerHandler argument
on a real refined-checkpoint file. Reports:

  - Total Type 1 triggers without the guard
  - Total Type 1 triggers with the guard
  - Triggers rejected by the guard, grouped by reason
    (vertex on triple-point triangle, or approaching VP on a triple-point VP)

Usage:
    python testing/test_type1_triple_point_guard.py \
        --solution <path>/iteration_*.h5 \
        --boundary-tol 0.001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh.mesh_topology import MeshTopology
from src.partition.steiner_handler import SteinerHandler
from src.pipeline.io import load_partition_from_refined_file
from src.migration.migration_detector import detect_type1_triggers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--boundary-tol", type=float, default=0.001)
    args = parser.parse_args()

    print("=" * 80)
    print("TRIPLE-POINT GUARD SMOKE TEST")
    print("=" * 80)

    mesh, partition = load_partition_from_refined_file(args.solution)
    mesh_topology = MeshTopology(mesh)
    steiner_handler = SteinerHandler(mesh, partition)

    print(f"\nMesh:          {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    print(f"Partition:     {partition.n_cells} cells, "
          f"{sum(1 for vp in partition.variable_points if vp.active)} active VPs")
    print(f"Triple points: {len(steiner_handler.triple_points)}")
    print(f"Boundary tol:  {args.boundary_tol}")

    print("\n--- Detection WITHOUT guard ---")
    triggers_no_guard = detect_type1_triggers(
        partition, mesh_topology, delta=args.boundary_tol, steiner_handler=None,
    )
    print(f"Type 1 triggers (no guard): {len(triggers_no_guard)}")

    print("\n--- Detection WITH guard ---")
    triggers_with_guard = detect_type1_triggers(
        partition, mesh_topology, delta=args.boundary_tol,
        steiner_handler=steiner_handler,
    )
    print(f"Type 1 triggers (with guard): {len(triggers_with_guard)}")

    rejected_vertices = (
        {t.vertex for t in triggers_no_guard}
        - {t.vertex for t in triggers_with_guard}
    )
    print(f"Triggers rejected by guard: {len(rejected_vertices)}")

    if rejected_vertices:
        triple_vp_set = set()
        triple_vertex_set = set()
        for tp in steiner_handler.triple_points:
            triple_vp_set.update(int(vi) for vi in tp.var_point_indices)
            triple_vertex_set.update(int(v) for v in tp.vertex_indices)

        print()
        for v in sorted(rejected_vertices):
            trig = next(t for t in triggers_no_guard if t.vertex == v)
            reasons = []
            if v in triple_vertex_set:
                reasons.append("vertex on triple-point triangle")
            shared_vps = [vp for vp in trig.approaching_vps if vp in triple_vp_set]
            if shared_vps:
                reasons.append(f"approaching VPs on triple-point: {shared_vps}")
            print(f"  Vertex {v} ({trig.current_cell} -> {trig.target_cell}, "
                  f"#VPs={trig.n_boundary_vps}): {' AND '.join(reasons)}")

    if 850 in {t.vertex for t in triggers_with_guard}:
        print("\n[OK] Vertex 850 (the >3-VP example) still passes the guard "
              "(no triple point in its 1-ring).")
    elif 850 in {t.vertex for t in triggers_no_guard}:
        print("\n[WARN] Vertex 850 was rejected by the guard — would be a "
              "regression for the documented case. Investigate.")

    print("\nDone.")


if __name__ == "__main__":
    main()
