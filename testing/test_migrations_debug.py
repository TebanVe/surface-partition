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
import h5py

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.io import load_partition_from_refined_file
from src.mesh.mesh_topology import MeshTopology
from src.migration import migration_utils
from src.logging_config import get_logger, setup_logging
from src.migration.migration_orchestrator import MigrationOrchestrator, MigrationConfig


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
                             '(migration_orchestrator, migration_detector, etc.), '
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

    # Detect current iteration from HDF5 attrs, then filename fallback
    starting_iteration = None
    try:
        with h5py.File(args.solution, 'r') as f:
            val = f.attrs.get('iteration_number')
            if val is not None:
                starting_iteration = int(val)
    except Exception:
        pass
    if starting_iteration is None:
        iter_match = re.search(r'iteration[_]?(\d+)', os.path.basename(args.solution))
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

    # MigrationOrchestrator path
    mesh_topology = MeshTopology(mesh)
    orchestrator = MigrationOrchestrator(
        partition, mesh, mesh_topology,
        MigrationConfig(delta=args.boundary_tol)
    )
    detection = orchestrator.detect_all_triggers(delta=args.boundary_tol)

    if not detection.type1_triggers and not detection.type2_triggers:
        logger.info("No topology switches detected — partition appears converged")
        return 0

    if args.migration_type == 'type1':
        triggers = detection.type1_triggers
    elif args.migration_type == 'type2':
        triggers = detection.type2_triggers
    else:
        triggers = detection.type2_triggers + detection.type1_triggers

    logger.info(f"Executing {len(triggers)} triggers (Type2: {len(detection.type2_triggers)}, "
                f"Type1: {len(detection.type1_triggers)})")

    for i, trigger in enumerate(triggers):
        snap_before = snapshot_vps(partition, watch_set) if watch_set else {}
        try:
            ok = orchestrator.execute_single_trigger(trigger)
        except Exception as exc:
            logger.error(f"  Exception on trigger {i}: {exc}")
            ok = False

        if ok:
            logger.info(f"  ✓ Trigger {i}: success")
        else:
            logger.error(f"  ✗ Trigger {i}: FAILED")

        if watch_set:
            snap_after = snapshot_vps(partition, watch_set)
            log_diff(f"after trigger {i}", snap_before, snap_after, logger)

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
