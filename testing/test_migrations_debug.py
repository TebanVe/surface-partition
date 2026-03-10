#!/usr/bin/env python3
"""
Test script: Migration Debug (Type 2 then Type 1, no optimization, no export)

Purpose: Investigate VP lambda values before and after Type 2 migrations to
diagnose why VPs that are correctly separated into different components before
Type 2 migrations end up merged into a spurious component afterward.

For each Type 2 migration, the script logs the full edge/lambda/target state of
all watched VPs and reports any changes. After Type 2 migrations it runs the
full Type 1 component analysis and shows which components contain the watched VPs.

Usage:
    # Watch default debug VPs (1621, 1624, 1625)
    python testing/test_migrations_debug.py \\
        --solution results/run_xyz/*_iterationN_refined_contours.h5 \\
        --boundary-tol 0.001

    # Specify VPs to watch
    python testing/test_migrations_debug.py \\
        --solution results/run_xyz/*_iterationN_refined_contours.h5 \\
        --watch-vps 1621 1624 1625

Author: Perimeter Refinement Testing
Date: March 2026
"""

import os
import sys
import re
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

from examples.data_loader import load_partition_from_refined_file
from src.core.tri_mesh import TriMesh
from src.core.mesh_topology import MeshTopology
from src.core.topology_switcher import TopologySwitcher
from src.core.steiner_handler import SteinerHandler
from src.core.area_calculator import AreaCalculator
from src.core.perimeter_calculator import PerimeterCalculator
from src.core.perimeter_optimizer import PerimeterOptimizer
from src.core.type1_component_analyzer import Type1ComponentAnalyzer
from src.core.type2_migration_history import Type2MigrationHistory
from src.core.type2_migration_io import load_type2_migration_history
from src.core import migration_utils
from src.logging_config import get_logger, setup_logging


# ============================================================================
# VP STATE HELPERS
# ============================================================================

def vp_one_liner(partition, vp_idx):
    """Return a one-line state summary for a single VP."""
    if vp_idx >= len(partition.variable_points):
        return f"VP {vp_idx}: OUT OF RANGE (total={len(partition.variable_points)})"
    vp = partition.variable_points[vp_idx]
    target = migration_utils.identify_target_vertex(vp)
    dist = migration_utils.compute_boundary_distance(vp)
    return (f"VP {vp_idx}: edge={vp.edge}, λ={vp.lambda_param:.6f}, "
            f"approaches={target}, dist={dist:.6f}")


def snapshot_vps(partition, vp_indices):
    """Capture (edge, lambda, target_vertex) for each watched VP index."""
    snap = {}
    for idx in vp_indices:
        if idx < len(partition.variable_points):
            vp = partition.variable_points[idx]
            target = migration_utils.identify_target_vertex(vp)
            snap[idx] = (tuple(vp.edge), vp.lambda_param, target)
        else:
            snap[idx] = None
    return snap


def log_snapshot(label, snap, logger):
    """Log the full state of all VPs in a snapshot."""
    logger.info(f"  [{label}]")
    for idx in sorted(snap.keys()):
        s = snap[idx]
        if s is None:
            logger.info(f"    VP {idx}: does not exist")
        else:
            edge, lam, tgt = s
            dist = lam if lam < 0.5 else (1.0 - lam)
            logger.info(f"    VP {idx}: edge={edge}, λ={lam:.6f}, "
                        f"approaches={tgt}, dist_to_tgt={dist:.6f}")


def log_diff(label, before, after, logger):
    """Log only VPs that changed between two snapshots."""
    all_keys = sorted(set(before.keys()) | set(after.keys()))
    changes = []
    for idx in all_keys:
        b, a = before.get(idx), after.get(idx)
        if b is None and a is not None:
            edge, lam, tgt = a
            changes.append(f"    VP {idx}: CREATED  edge={edge}, λ={lam:.6f}, approaches={tgt}")
        elif b is not None and a is None:
            changes.append(f"    VP {idx}: DELETED")
        elif b != a:
            be, bl, bt = b
            ae, al, at = a
            parts = []
            if be != ae:
                parts.append(f"edge {be}→{ae}")
            if bl != al:
                parts.append(f"λ {bl:.6f}→{al:.6f}")
            if bt != at:
                parts.append(f"approaches {bt}→{at}  *** TARGET FLIP ***")
            changes.append(f"    VP {idx}: CHANGED  " + " | ".join(parts))

    if changes:
        logger.info(f"  [{label}] VP changes:")
        for c in changes:
            logger.info(c)
    else:
        logger.info(f"  [{label}] No changes to watched VPs")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Debug Type 2 + Type 1 migrations without optimization or export'
    )
    parser.add_argument('--solution', required=True,
                        help='Path to refined contours HDF5 file (iterationN file)')
    parser.add_argument('--migration-type', choices=['type1', 'type2', 'both'],
                        default='both',
                        help='Which migrations to apply (default: both)')
    parser.add_argument('--distance-preservation', default='preserve',
                        help='VP placement strategy (default: preserve)')
    parser.add_argument('--boundary-tol', type=float, default=1e-3,
                        help='Boundary tolerance for detection (default: 1e-3)')
    parser.add_argument('--watch-vps', type=int, nargs='+', default=[],
                        help='VP indices to track across all migrations '
                             '(default: none; use e.g. --watch-vps 1621 1624 1625)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging — passes DEBUG to all core modules '
                             '(topology_switcher, type1_component_analyzer, etc.), '
                             'producing highly verbose output. Overrides --log-level.')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO; --debug overrides this)')
    args = parser.parse_args()

    log_level = 'DEBUG' if args.debug else args.log_level
    log_file = setup_logging(log_level=log_level)
    logger = get_logger(__name__)

    if args.debug:
        logger.info("Debug mode enabled — all core module DEBUG messages will be shown")

    logger.info("="*80)
    logger.info("MIGRATION DEBUG — no optimization, no export")
    logger.info("="*80)
    logger.info(f"Input file   : {args.solution}")
    logger.info(f"Migrations   : {args.migration_type}")
    logger.info(f"Dist. pres.  : {args.distance_preservation}")
    logger.info(f"Boundary tol : {args.boundary_tol}")
    if args.watch_vps:
        logger.info(f"Watching VPs : {args.watch_vps}")
    logger.info("="*80)

    # ------------------------------------------------------------------ load
    logger.info("")
    logger.info("LOADING DATA")
    logger.info("-"*80)
    mesh, partition = load_partition_from_refined_file(args.solution, verbose=True)
    logger.info(f"Mesh       : {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    logger.info(f"Partition  : {len(partition.variable_points)} VPs, {partition.n_cells} cells")

    # Load migration history
    migration_history = Type2MigrationHistory()
    try:
        import h5py
        with h5py.File(args.solution, 'r') as f:
            loaded_history = load_type2_migration_history(f)
        if loaded_history is not None:
            migration_history = loaded_history
            logger.info(f"Migration history loaded: {len(migration_history.records)} records")
        else:
            logger.info("No migration history in file — starting fresh")
    except Exception as e:
        logger.warning(f"Could not load migration history: {e}")

    # Detect current iteration from filename for history bookkeeping
    iter_match = re.search(r'_iteration(\d+)_refined_contours\.h5$', args.solution)
    starting_iteration = int(iter_match.group(1)) if iter_match else 1
    current_iteration = starting_iteration + 1
    logger.info(f"Detected start iteration: {starting_iteration}, "
                f"treating as iteration {current_iteration}")

    total_area = float(mesh.M.sum())
    target_area = total_area / partition.n_cells
    logger.info(f"Target area per cell: {target_area:.10f}")

    watch_set = list(args.watch_vps)

    # ------------------------------------------------------------------ initial snapshot
    if watch_set:
        logger.info("")
        logger.info("="*80)
        logger.info("INITIAL STATE OF WATCHED VPs")
        logger.info("="*80)
        log_snapshot("initial", snapshot_vps(partition, watch_set), logger)

    # ------------------------------------------------------------------ init optimizer
    # We need the optimizer only to call check_topology_switches_needed.
    # All heavy allocations are deferred.
    area_calc  = AreaCalculator(mesh, partition)
    perim_calc = PerimeterCalculator(mesh, partition)
    steiner_handler = SteinerHandler(mesh, partition)
    mesh_topology = MeshTopology(mesh)
    optimizer = PerimeterOptimizer(
        partition, mesh, target_area,
        area_calc=area_calc,
        perim_calc=perim_calc,
        steiner_handler=steiner_handler
    )

    # ------------------------------------------------------------------ detect switches
    logger.info("")
    logger.info("="*80)
    logger.info("DETECTING TOPOLOGY SWITCHES")
    logger.info("="*80)

    switches_needed, switch_info = optimizer.check_topology_switches_needed(
        tol=args.boundary_tol
    )

    if not switches_needed:
        logger.info("No topology switches detected — partition appears converged")
        return 0

    # Collect Type 1 VP candidates
    type1_candidates = [
        i for i, vp in enumerate(partition.variable_points)
        if vp.on_boundary(tol=args.boundary_tol)
    ]
    # Collect Type 2 (triple point) candidates
    type2_candidates = optimizer.steiner_handler.get_boundary_triple_points(
        tol=args.boundary_tol
    )

    logger.info(f"Type 1 VP candidates  : {len(type1_candidates)}")
    logger.info(f"Type 2 triple points  : {len(type2_candidates)}")

    # ================================================================== TYPE 2
    if args.migration_type in ['type2', 'both'] and type2_candidates:
        logger.info("")
        logger.info("="*80)
        logger.info(f"TYPE 2 MIGRATIONS ({len(type2_candidates)} triple points)")
        logger.info("="*80)

        switcher = TopologySwitcher(mesh, partition, mesh_topology)
        switcher.type2_migration_history = migration_history
        migration_history.current_iteration = current_iteration

        n_vps_before = len(partition.variable_points)
        migrations_ok = 0
        migrations_fail = 0

        for tp in type2_candidates:
            # Resolve index in steiner_handler.triple_points
            tp_idx = None
            for idx, handler_tp in enumerate(optimizer.steiner_handler.triple_points):
                if handler_tp.triangle_idx == tp.triangle_idx:
                    tp_idx = idx
                    break

            if tp_idx is None:
                logger.error(f"  Could not find triple point (triangle {tp.triangle_idx}) "
                             "in steiner_handler")
                continue

            logger.info("")
            logger.info(f"  --- Triple point {tp_idx} (triangle {tp.triangle_idx}) ---")
            logger.info(f"  Triple-point VPs: {list(tp.var_point_indices)}")
            for vp_idx in tp.var_point_indices:
                logger.info(f"    {vp_one_liner(partition, vp_idx)}")

            snap_before = snapshot_vps(partition, watch_set) if watch_set else {}

            try:
                result = switcher.apply_type2_switch_v4(
                    steiner_handler=optimizer.steiner_handler,
                    triple_point_idx=tp_idx,
                    distance_preservation=args.distance_preservation
                )
                ok = result.get('success', False)
            except Exception as exc:
                logger.error(f"  Exception: {exc}")
                ok = False
                result = {'success': False, 'error': str(exc)}

            if ok:
                n_vps_after = len(partition.variable_points)
                delta = n_vps_after - n_vps_before
                logger.info(f"  ✓ success — VP count: {n_vps_before} → {n_vps_after} ({delta:+d})")
                n_vps_before = n_vps_after
                migrations_ok += 1
            else:
                logger.error(f"  ✗ FAILED — {result.get('error', 'unknown error')}")
                migrations_fail += 1

            if watch_set:
                snap_after = snapshot_vps(partition, watch_set)
                log_diff(f"after Type2 tp={tp_idx} tri={tp.triangle_idx}",
                         snap_before, snap_after, logger)

        logger.info("")
        logger.info(f"Type 2 complete: {migrations_ok} succeeded, {migrations_fail} failed")
        logger.info(f"Total VPs now: {len(partition.variable_points)}")

    # ------------------------------------------------------------------ post-Type2 snapshot
    if watch_set:
        logger.info("")
        logger.info("="*80)
        logger.info("WATCHED VP STATE AFTER ALL TYPE 2 MIGRATIONS")
        logger.info("="*80)
        log_snapshot("post-Type2", snapshot_vps(partition, watch_set), logger)

    # ================================================================== TYPE 1
    if args.migration_type in ['type1', 'both']:
        logger.info("")
        logger.info("="*80)
        logger.info("TYPE 1 COMPONENT ANALYSIS (post-Type2 state)")
        logger.info("="*80)

        # Reinitialize SteinerHandler and MeshTopology for the updated partition
        steiner_handler2 = SteinerHandler(mesh, partition)
        mesh_topology2  = MeshTopology(mesh)

        import logging as _logging
        # Suppress verbose sub-logger output during analysis (unless debug mode)
        _sw_log = _logging.getLogger('src.core.topology_switcher')
        _orig_sw = _sw_log.level
        if not args.debug:
            _sw_log.setLevel(_logging.WARNING)

        analyzer = Type1ComponentAnalyzer(
            mesh, partition, mesh_topology2,
            steiner_handler=steiner_handler2
        )

        analysis_result = analyzer.run_full_analysis(
            boundary_tol=args.boundary_tol,
            conflict_strategy='exclude_one',
            build_migration_plan=True,
            protect_type2=True
        )

        _sw_log.setLevel(_orig_sw)

        to_migrate    = analysis_result.get('to_migrate', [])
        excluded      = analysis_result.get('excluded', [])
        type2_excluded = analysis_result.get('type2_excluded', [])
        migration_plan = analysis_result.get('migration_plan', [])

        logger.info(f"Components to migrate         : {len(to_migrate)}")
        logger.info(f"Components excluded (pre-filter): {len(excluded)}")
        logger.info(f"Components excluded (Type2 prot): {len(type2_excluded)}")
        logger.info(f"Migration plan entries         : {len(migration_plan)}")

        # Show every component that contains a watched VP
        if watch_set:
            logger.info("")
            logger.info("  Components/plan entries containing watched VPs:")
            found = False
            all_comps = to_migrate + excluded + type2_excluded
            for comp in all_comps:
                comp_vps = set(comp.get('vp_indices', []))
                if comp_vps & set(watch_set):
                    found = True
                    status = ("TO MIGRATE" if comp in to_migrate
                              else "TYPE2-EXCL" if comp in type2_excluded
                              else "EXCLUDED")
                    logger.info(
                        f"    Component {comp.get('index')} [{status}]: "
                        f"VPs={comp.get('vp_indices')}, "
                        f"target={comp.get('target_vertex')}, "
                        f"size={comp.get('size')}, "
                        f"dist={comp.get('min_distance', float('nan')):.6f}"
                    )
                    for vp_idx in sorted(comp_vps):
                        logger.info(f"      {vp_one_liner(partition, vp_idx)}")

            for entry in migration_plan:
                entry_vps = {entry['migrating_vp'],
                             entry['left_neighbor'],
                             entry['right_neighbor']}
                entry_vps.update(entry.get('auxiliary_component', []))
                if entry_vps & set(watch_set):
                    found = True
                    logger.info(
                        f"    Plan entry comp={entry['component_idx']}: "
                        f"migrating={entry['migrating_vp']}, "
                        f"aux={entry['auxiliary_component']}, "
                        f"L={entry['left_neighbor']}, R={entry['right_neighbor']}"
                    )

            if not found:
                logger.info("    (none of the watched VPs appear in any component or plan entry)")

        # ---------------------------------------------------------------- apply Type 1
        if migration_plan:
            logger.info("")
            logger.info("="*80)
            logger.info(f"APPLYING TYPE 1 MIGRATIONS ({len(migration_plan)} entries)")
            logger.info("="*80)

            switcher2 = TopologySwitcher(mesh, partition, mesh_topology2)
            migrations_ok2  = 0
            migrations_fail2 = 0

            for plan_entry in migration_plan:
                comp     = plan_entry['component']
                comp_idx = comp['index']

                # Pre-execution revalidation: verify auxiliary VPs still on
                # edges incident to the target vertex
                target_vertex = comp['target_vertex']
                aux_vps = plan_entry['auxiliary_component']
                skip = False
                for vp_idx in aux_vps:
                    vp = partition.variable_points[vp_idx]
                    if target_vertex not in vp.edge:
                        logger.warning(f"  ⚠ Component {comp_idx}: VP {vp_idx} on edge {vp.edge} "
                                     f"no longer incident to target vertex {target_vertex} — skipping")
                        skip = True
                        break
                if skip:
                    migrations_fail2 += 1
                    continue

                snap_before = snapshot_vps(partition, watch_set) if watch_set else {}

                try:
                    result = switcher2.apply_type1_switch_v2(
                        component=comp,
                        distance_preservation=args.distance_preservation,
                        migrating_vp=plan_entry['migrating_vp'],
                        auxiliary_component=plan_entry['auxiliary_component'],
                        left_neighbor=plan_entry['left_neighbor'],
                        right_neighbor=plan_entry['right_neighbor']
                    )
                    ok = result.get('success', False)
                except Exception as exc:
                    logger.error(f"  ✗ Component {comp_idx}: Exception — {exc}")
                    ok = False
                    result = {'success': False, 'error': str(exc)}

                if ok:
                    logger.info(f"  ✓ Component {comp_idx}: Migration successful")
                    migrations_ok2 += 1
                    partition.rebuild_triangle_segments_from_current_vps()
                else:
                    logger.error(f"  ✗ Component {comp_idx}: FAILED — "
                                 f"{result.get('error', 'unknown')}")
                    migrations_fail2 += 1

                if watch_set:
                    snap_after = snapshot_vps(partition, watch_set)
                    log_diff(f"after Type1 comp={comp_idx}",
                             snap_before, snap_after, logger)

            logger.info("")
            logger.info(f"Type 1 complete: {migrations_ok2} succeeded, "
                        f"{migrations_fail2} failed")

    # ------------------------------------------------------------------ final snapshot
    if watch_set:
        logger.info("")
        logger.info("="*80)
        logger.info("FINAL STATE OF WATCHED VPs")
        logger.info("="*80)
        log_snapshot("final", snapshot_vps(partition, watch_set), logger)

    logger.info("")
    logger.info("="*80)
    logger.info("DEBUG COMPLETE — no optimization, no export")
    logger.info(f"Total VPs: {len(partition.variable_points)}")
    logger.info("="*80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
