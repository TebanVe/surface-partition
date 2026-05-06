#!/usr/bin/env python3
"""
Diagnostic script to analyze the white-triangle / dot-in-middle issue
reported in the Type 1 visualization for double torus 10-partition.

For the specific iteration file:
  iteration_001_20260506_093157.h5

Region 7, Component 4 (>3 approaching VPs).

Usage:
    python testing/diagnose_white_triangles.py --solution <path>
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh.mesh_topology import MeshTopology
from src.partition.steiner_handler import SteinerHandler
from src.partition.area_calculator import AreaCalculator
from src.pipeline.io import load_partition_from_refined_file
from src.migration.migration_orchestrator import MigrationOrchestrator, MigrationConfig
from src.migration import migration_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--component-index", type=int, default=4)
    parser.add_argument("--region", type=int, default=7)
    parser.add_argument("--boundary-tol", type=float, default=0.001)
    args = parser.parse_args()

    print("=" * 80)
    print("WHITE-TRIANGLE DIAGNOSTIC")
    print("=" * 80)

    mesh, partition = load_partition_from_refined_file(args.solution)
    mesh_topology = MeshTopology(mesh)

    print(f"\nMesh:      {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    print(f"Partition: {partition.n_cells} cells, "
          f"{sum(1 for vp in partition.variable_points if vp.active)} active VPs "
          f"({len(partition.variable_points)} total)")

    # Get the same trigger that visualization picks
    orchestrator = MigrationOrchestrator(
        partition, mesh, mesh_topology,
        MigrationConfig(delta=args.boundary_tol),
    )
    detection = orchestrator.detect_all_triggers(delta=args.boundary_tol)
    type1_triggers = detection.type1_triggers

    print(f"\nType 1 triggers detected: {len(type1_triggers)}")
    if args.component_index >= len(type1_triggers):
        print(f"ERROR: Component {args.component_index} out of range")
        return

    trig = type1_triggers[args.component_index]
    target_vertex = trig.vertex
    print()
    print(f"Selected trigger #{args.component_index}:")
    print(f"  vertex          = {target_vertex}")
    print(f"  current cell    = {trig.current_cell}")
    print(f"  target cell     = {trig.target_cell}")
    print(f"  approaching VPs = {trig.approaching_vps}")
    print(f"  n_boundary_vps  = {trig.n_boundary_vps}")
    print(f"  min lambda dist = {trig.min_lambda_distance:.6e}")

    # ------------------------------------------------------------------
    # 1) Triangles around the target vertex
    # ------------------------------------------------------------------
    target_pos = mesh.vertices[target_vertex]
    triangles_at_vertex = []
    for tri_idx, face in enumerate(mesh.faces):
        if target_vertex in (int(face[0]), int(face[1]), int(face[2])):
            triangles_at_vertex.append(tri_idx)
    print(f"\nVertex {target_vertex}: 3D pos = {target_pos}")
    print(f"  valence (triangles in 1-ring): {len(triangles_at_vertex)}")
    print(f"  triangle indices: {triangles_at_vertex}")

    # ------------------------------------------------------------------
    # 2) For every triangle in the 1-ring, classify it
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRIANGLE-BY-TRIANGLE STATE IN THE 1-RING")
    print("=" * 80)

    vertex_labels = partition.vertex_labels

    # build tri_idx -> TriangleSegment lookup
    tri_idx_to_seg = {ts.triangle_idx: ts for ts in partition.triangle_segments}

    # build SteinerHandler for triple-point lookup
    steiner_handler = SteinerHandler(mesh, partition)
    triple_point_tri_set = {tp.triangle_idx for tp in steiner_handler.triple_points}

    # build AreaCalculator to know how it categorized triangles
    area_calc = AreaCalculator(mesh, partition)

    interior_for = defaultdict(list)
    boundary_for = defaultdict(list)
    for cidx, tlist in area_calc.cell_interior_triangles.items():
        for ti in tlist:
            interior_for[ti].append(cidx)
    for cidx, tlist in area_calc.cell_boundary_triangles.items():
        for ti in tlist:
            boundary_for[ti].append(cidx)

    print()
    header = (f"{'tri':>6} {'v_labels':>14} {'#labels':>7}  "
              f"{'#VPs':>4}  {'VPs (idx,edge,λ,cells)':<60}  "
              f"{'TP?':<4} {'Interior(cells)':<18} {'Boundary(cells)':<18}  "
              f"{'rendered?':<10}")
    print(header)
    print("-" * len(header))

    for tri_idx in triangles_at_vertex:
        face = mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        labs = (int(vertex_labels[v1]), int(vertex_labels[v2]), int(vertex_labels[v3]))
        n_distinct = len(set(labs))

        ts = tri_idx_to_seg.get(tri_idx)
        if ts is None:
            vp_count = 0
            vp_brief = "-"
        else:
            vp_descs = []
            active_vps = [vi for vi in ts.var_point_indices
                          if partition.variable_points[vi].active]
            for vi in active_vps:
                vp = partition.variable_points[vi]
                cells = sorted(list(vp.belongs_to_cells))
                vp_descs.append(f"({vi},{vp.edge},λ={vp.lambda_param:.3f},{cells})")
            vp_count = len(active_vps)
            vp_brief = "; ".join(vp_descs) if vp_descs else "-"

        is_tp = "Yes" if tri_idx in triple_point_tri_set else "No"
        int_cells = interior_for.get(tri_idx, [])
        bnd_cells = boundary_for.get(tri_idx, [])

        will_render = (
            len(int_cells) > 0
            or len(bnd_cells) > 0
            or tri_idx in triple_point_tri_set
        )

        # truncate vp_brief for display
        vp_disp = vp_brief if len(vp_brief) <= 60 else vp_brief[:57] + "..."

        rendered_label = "YES" if will_render else "*** NO ***"

        print(f"T{tri_idx:>5} {str(labs):>14} {n_distinct:>7}  "
              f"{vp_count:>4}  {vp_disp:<60}  "
              f"{is_tp:<4} {str(int_cells):<18} {str(bnd_cells):<18}  "
              f"{rendered_label:<10}")

    # ------------------------------------------------------------------
    # 3) Summarize VPs around the target vertex
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VPs ON EDGES INCIDENT TO TARGET VERTEX")
    print("=" * 80)

    print(f"\n{'edge':<14} {'other_label':<11} {'vp_idx':<7} "
          f"{'λ':<10} {'active':<6} {'belongs_to_cells'}")
    print("-" * 78)
    for edge in mesh_topology.get_edges_at_vertex(target_vertex):
        other = edge[1] if edge[0] == target_vertex else edge[0]
        other_label = int(vertex_labels[other])
        normalized = tuple(sorted(edge))
        vp_idx = partition.edge_to_varpoint.get(normalized)
        if vp_idx is None:
            vp_str = "-"
            lam_str = "-"
            act_str = "-"
            cells_str = "-"
        else:
            vp = partition.variable_points[vp_idx]
            vp_str = str(vp_idx)
            lam_str = f"{vp.lambda_param:.4f}"
            act_str = str(vp.active)
            cells_str = str(sorted(list(vp.belongs_to_cells)))
        print(f"{str(normalized):<14} {other_label:<11} {vp_str:<7} "
              f"{lam_str:<10} {act_str:<6} {cells_str}")

    # ------------------------------------------------------------------
    # 4) Specifically inspect the triple-point-by-vertex-label triangles
    #    and check why they are not registered in steiner_handler.triple_points
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRIANGLES WITH 3 DIFFERENT VERTEX LABELS (TRIPLE-POINT BY LABELS)")
    print("=" * 80)

    bad_triple_label_tris = []
    for tri_idx in triangles_at_vertex:
        face = mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        labs = (int(vertex_labels[v1]), int(vertex_labels[v2]), int(vertex_labels[v3]))
        if len(set(labs)) == 3:
            bad_triple_label_tris.append((tri_idx, labs))

    if not bad_triple_label_tris:
        print("(none)")
    else:
        for tri_idx, labs in bad_triple_label_tris:
            print(f"\nT{tri_idx}: vertex labels {labs}")
            print(f"  Registered as triple point in SteinerHandler? "
                  f"{'YES' if tri_idx in triple_point_tri_set else 'NO  *** MISSING ***'}")
            ts = tri_idx_to_seg.get(tri_idx)
            if ts is None:
                print("  No TriangleSegment exists.")
                continue
            face = mesh.faces[tri_idx]
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            edges_of_tri = [
                tuple(sorted([v1, v2])),
                tuple(sorted([v2, v3])),
                tuple(sorted([v1, v3])),
            ]
            for e in edges_of_tri:
                vp_idx = partition.edge_to_varpoint.get(e)
                end_labs = (int(vertex_labels[e[0]]), int(vertex_labels[e[1]]))
                expected = end_labs[0] != end_labs[1]
                if vp_idx is None:
                    have = "no VP"
                else:
                    vp = partition.variable_points[vp_idx]
                    have = (f"VP#{vp_idx} active={vp.active} λ={vp.lambda_param:.3f} "
                            f"belongs={sorted(vp.belongs_to_cells)}")
                print(f"  edge {e} (labels {end_labs}): expected_VP={expected}, {have}")

            print(f"  TriangleSegment.var_point_indices = {ts.var_point_indices} "
                  f"(active among them: "
                  f"{[vi for vi in ts.var_point_indices if partition.variable_points[vi].active]})")

    # ------------------------------------------------------------------
    # 5) Verify what `compute_cell_portion_in_triangle_simple` would do
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RENDER PATH SIMULATION FOR FOCUS REGION")
    print("=" * 80)
    print(f"Region (cell_idx) = {args.region}\n")

    cell = args.region
    interior = set(area_calc.cell_interior_triangles.get(cell, []))
    boundary = set(area_calc.cell_boundary_triangles.get(cell, []))
    print(f"Cell {cell}: {len(interior)} interior tris, {len(boundary)} boundary tris")
    print(f"Cell {cell} triangles in 1-ring of target vertex {target_vertex}:")
    for tri_idx in triangles_at_vertex:
        if tri_idx in interior:
            cls = "interior"
        elif tri_idx in boundary:
            cls = "boundary"
        elif tri_idx in triple_point_tri_set:
            tp = next(tp for tp in steiner_handler.triple_points
                      if tp.triangle_idx == tri_idx)
            cls = (f"triple-point (cells={sorted(tp.cell_indices)}, "
                   f"region in cells: {cell in tp.cell_indices})")
        else:
            cls = "*** NOT IN ANY CATEGORY → WHITE ***"
        print(f"  T{tri_idx}: {cls}")

    print("\nDone.")


if __name__ == "__main__":
    main()
