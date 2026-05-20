#!/usr/bin/env python3
"""
Cross-reference white triangles in a Type 1 component's 1-ring with OTHER
pending Type 1 triggers.

For the selected component:
  1. List the full VP state of every triangle in the target vertex's 1-ring.
  2. For each VP that is collapsed (λ=0 or λ=1), identify which mesh vertex
     it sits at.
  3. Cross-reference those collapsed-target vertices with the full Type 1
     trigger list to identify which OTHER pending migration would resolve
     each white triangle.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh.mesh_topology import MeshTopology
from src.partition.steiner_handler import SteinerHandler
from src.partition.area_calculator import AreaCalculator
from src.pipeline.io import load_partition_from_refined_file
from src.migration.migration_orchestrator import MigrationOrchestrator, MigrationConfig
from src.migration import migration_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--component-index", type=int, default=35)
    ap.add_argument("--boundary-tol", type=float, default=0.001)
    args = ap.parse_args()

    mesh, partition = load_partition_from_refined_file(args.solution)
    mesh_topology = MeshTopology(mesh)

    orch = MigrationOrchestrator(
        partition, mesh, mesh_topology,
        MigrationConfig(delta=args.boundary_tol),
    )
    detection = orch.detect_all_triggers(delta=args.boundary_tol)
    triggers = detection.type1_triggers
    print(f"Total Type 1 triggers: {len(triggers)}")

    trig = triggers[args.component_index]
    target_v = trig.vertex
    print(f"\nThis component (#{args.component_index}):")
    print(f"  vertex          = {target_v}")
    print(f"  cell flip       = {trig.current_cell} -> {trig.target_cell}")
    print(f"  approaching VPs = {trig.approaching_vps}")

    vertex_to_trigger = {t.vertex: (i, t) for i, t in enumerate(triggers)}

    triangles_at_v = [
        ti for ti, face in enumerate(mesh.faces)
        if target_v in (int(face[0]), int(face[1]), int(face[2]))
    ]

    print(f"\n1-ring of vertex {target_v}: {len(triangles_at_v)} triangles")
    print("=" * 100)
    print(f"{'tri':>6}  {'v_labels':>10}  {'VPs (full)':<70}")
    print("-" * 100)

    tri_to_seg = {ts.triangle_idx: ts for ts in partition.triangle_segments}

    for ti in triangles_at_v:
        face = mesh.faces[ti]
        vlabs = tuple(int(partition.vertex_labels[int(face[k])]) for k in range(3))
        ts = tri_to_seg.get(ti)
        if ts is None:
            print(f"T{ti:<5}  {str(vlabs):<10}  -")
            continue

        vp_descs = []
        for vi in ts.var_point_indices:
            vp = partition.variable_points[vi]
            if not vp.active:
                continue
            tgt = migration_utils.identify_target_vertex(vp)
            collapsed = vp.lambda_param < 1e-6 or vp.lambda_param > 1 - 1e-6
            collapsed_tag = f"→V{tgt}" if collapsed else "(non-degenerate)"
            vp_descs.append(
                f"VP{vi}({vp.edge}, λ={vp.lambda_param:.4f}) {collapsed_tag}"
            )
        print(f"T{ti:<5}  {str(vlabs):<10}  {' | '.join(vp_descs)}")

    # Cross-reference collapsed target vertices in this 1-ring with other triggers
    print("\n" + "=" * 100)
    print("WHICH OTHER PENDING TRIGGERS RESOLVE THE COLLAPSED VPs IN THIS 1-RING?")
    print("=" * 100)

    seen = set()
    for ti in triangles_at_v:
        ts = tri_to_seg.get(ti)
        if ts is None:
            continue
        for vi in ts.var_point_indices:
            vp = partition.variable_points[vi]
            if not vp.active:
                continue
            tgt = migration_utils.identify_target_vertex(vp)
            collapsed = vp.lambda_param < 1e-6 or vp.lambda_param > 1 - 1e-6
            if not collapsed or tgt is None:
                continue
            if (vi, tgt) in seen:
                continue
            seen.add((vi, tgt))

            if tgt == target_v:
                tag = f"THIS migration (component #{args.component_index})"
            elif tgt in vertex_to_trigger:
                idx, t2 = vertex_to_trigger[tgt]
                tag = (f"OTHER migration → component #{idx}: vertex {tgt}, "
                       f"flip {t2.current_cell}→{t2.target_cell}")
            else:
                tag = "NO matching trigger (collapsed but not in trigger list)"
            print(f"  T{ti}: VP{vi} → vertex {tgt}  ::  {tag}")

    # For each triangle, check whether THIS migration alone would resolve it
    print("\n" + "=" * 100)
    print("PER-TRIANGLE: WILL IT BE COLORED AFTER THIS MIGRATION?")
    print("=" * 100)

    own_vp_set = set(trig.approaching_vps)

    for ti in triangles_at_v:
        ts = tri_to_seg.get(ti)
        if ts is None:
            print(f"  T{ti}: no segment (probably interior)")
            continue
        active_vps = [vi for vi in ts.var_point_indices
                      if partition.variable_points[vi].active]
        if not active_vps:
            print(f"  T{ti}: no active VPs (interior to one cell)")
            continue

        collapsed_vps = []
        non_degen_vps = []
        for vi in active_vps:
            vp = partition.variable_points[vi]
            if vp.lambda_param < 1e-6 or vp.lambda_param > 1 - 1e-6:
                collapsed_vps.append(vi)
            else:
                non_degen_vps.append(vi)

        # If at least one VP is non-degenerate, the triangle was already colored
        # before AND after.
        if non_degen_vps:
            verdict = "ALREADY COLORED (≥1 non-degenerate VP)"
        else:
            collapsed_in_this = [vi for vi in collapsed_vps if vi in own_vp_set]
            collapsed_outside = [vi for vi in collapsed_vps if vi not in own_vp_set]
            if collapsed_outside and not collapsed_in_this:
                verdict = (f"WHITE BEFORE & AFTER — collapsed VPs {collapsed_outside} "
                           f"belong to OTHER migrations")
            elif collapsed_in_this and not collapsed_outside:
                verdict = (f"WHITE BEFORE → COLORED AFTER (all collapsed VPs "
                           f"{collapsed_in_this} are part of this migration)")
            elif collapsed_in_this and collapsed_outside:
                verdict = (f"WHITE BEFORE → STILL WHITE AFTER (mix: "
                           f"this={collapsed_in_this}, other={collapsed_outside})")
            else:
                verdict = "??? unclear"
        print(f"  T{ti}: {verdict}")


if __name__ == "__main__":
    main()
