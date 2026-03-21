#!/usr/bin/env python3
"""
Test script: Migration and Continue Optimization (with Iterative Support)

This script implements the migrate→optimize workflow starting from a refined partition:
1. Load refined contours from HDF5 file (iteration file or first refined file)
2. ITERATIVE LOOP (until convergence or max_iterations):
   a. Detect topology switches needed
   b. Apply selected migrations (Type 1, Type 2, or both)
   c. Rebuild calculators with updated partition
   d. Run optimization iteration
   e. Export results with correct iteration numbering
3. Converge when no more switches are detected

Key features:
- Can start from any iteration file (_iterationN_refined_contours.h5)
- Automatically detects starting iteration number from filename
- Maintains sequential iteration numbering in output files
- Default: single cycle (max-iterations=1) for backward compatibility
- Set --max-iterations higher for full iterative refinement
- Type 2 migrations applied before Type 1 (proven order)

This differs from refine_perimeter_iterative.py in workflow order:
- refine_perimeter_iterative.py: OPTIMIZE → MIGRATE (starts from base solution)
- test_migration_and_continue.py: MIGRATE → OPTIMIZE (starts from refined file)

Usage:
    # Single cycle (backward compatible)
    python testing/test_migration_and_continue.py \\
        --solution results/run_xyz/*_refined_contours.h5 \\
        --migration-type both

    # Iterative refinement from iteration 1
    python testing/test_migration_and_continue.py \\
        --solution results/run_xyz/*_iteration1_refined_contours.h5 \\
        --migration-type both \\
        --max-iterations 10

    # Resume from iteration 5
    python testing/test_migration_and_continue.py \\
        --solution results/run_xyz/*_iteration5_refined_contours.h5 \\
        --migration-type both \\
        --max-iterations 5

Author: Perimeter Refinement Testing
Date: February 2026
"""

import os
import sys
import argparse
import time
import numpy as np
import h5py

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

# Pre-parse --use-legacy before conditional imports
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--use-legacy', action='store_true',
                         help='Use legacy migration path (TopologySwitcher, Type1ComponentAnalyzer, etc.)')
_pre_args, _ = _pre_parser.parse_known_args()
_use_legacy = _pre_args.use_legacy

if _use_legacy:
    from src.core.topology_switcher_legacy import TopologySwitcher
    from src.core.type1_component_analyzer import Type1ComponentAnalyzer
    from src.core.type2_migration_history import Type2MigrationHistory
    from src.core.type2_migration_io import load_type2_migration_history, save_type2_migration_history
else:
    from src.core.migration_orchestrator import MigrationOrchestrator, MigrationConfig
    from src.core.migration_types import TriplePointHistory

from examples.data_loader import load_partition_from_refined_file
from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.core.area_calculator import AreaCalculator
from src.core.perimeter_calculator import PerimeterCalculator
from src.core.steiner_handler import SteinerHandler
from src.core.perimeter_optimizer import PerimeterOptimizer
from src.core.mesh_topology import MeshTopology
from src.logging_config import get_logger, setup_logging


def apply_type1_migrations(partition, mesh, vp_candidates, logger, boundary_tol=0.01, 
                          distance_preservation='preserve', steiner_handler=None):
    """Apply Type 1 migrations for boundary VPs.
    
    Args:
        partition: PartitionContour object
        mesh: TriMesh object
        vp_candidates: List of candidate VP indices
        logger: Logger instance
        boundary_tol: Boundary tolerance (default: 0.01)
        distance_preservation: VP placement strategy after migration (default: 'preserve')
        steiner_handler: Optional pre-created SteinerHandler (for efficiency)
    """
    logger.info("")
    logger.info("="*80)
    logger.info(f"APPLYING TYPE 1 MIGRATIONS ({len(vp_candidates)} candidates)")
    logger.info(f"  Boundary tolerance: {boundary_tol}")
    logger.info(f"  Distance preservation: {distance_preservation}")
    logger.info("="*80)
    
    if len(vp_candidates) == 0:
        logger.info("No Type 1 migrations to apply")
        logger.info("="*80)
        return 0
    
    # Create topology objects
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    analyzer = Type1ComponentAnalyzer(mesh, partition, mesh_topology, 
                                     steiner_handler=steiner_handler)  # Reuse if provided
    
    # Run full component analysis
    logger.info("\nStep 1: Running full component analysis...")
    logger.info("="*80)
    
    # Temporarily suppress verbose switcher logging during analysis
    import logging
    switcher_logger = logging.getLogger('src.core.topology_switcher')
    original_level = switcher_logger.level
    switcher_logger.setLevel(logging.WARNING)
    
    analysis_result = analyzer.run_full_analysis(
        boundary_tol=boundary_tol, 
        conflict_strategy='exclude_one',
        build_migration_plan=True,  # Request pre-computed migration plan
        protect_type2=True  # Enable Type 2 protection (topology-based)
    )
    
    # Restore logging level
    switcher_logger.setLevel(original_level)
    
    to_migrate = analysis_result['to_migrate']
    excluded = analysis_result['excluded']
    type2_excluded = analysis_result.get('type2_excluded', [])
    migration_plan = analysis_result.get('migration_plan', [])
    
    logger.info(f"\nComponents selected for migration: {len(to_migrate)}")
    logger.info(f"Components excluded (conflicts): {len(excluded)}")
    logger.info(f"Components excluded (Type 2 protection): {len(type2_excluded)}")
    if migration_plan:
        logger.info(f"Migration plan entries: {len(migration_plan)}")
    logger.info("="*80)
    
    # Log boundary_segment consistency check before Type 1 migrations
    # This helps diagnose whether Type 2 direct manipulation left stale segments
    canonical_pairs = set()
    for ts in partition.triangle_segments:
        vi = ts.var_point_indices
        if len(vi) == 2:
            canonical_pairs.add((min(vi[0], vi[1]), max(vi[0], vi[1])))
        elif len(vi) == 3:
            for a in range(3):
                for b in range(a + 1, 3):
                    canonical_pairs.add((min(vi[a], vi[b]), max(vi[a], vi[b])))
    
    live_pairs = set()
    for seg in partition.boundary_segments:
        live_pairs.add((min(seg.vp_idx_1, seg.vp_idx_2), max(seg.vp_idx_1, seg.vp_idx_2)))
    
    extra_in_live = live_pairs - canonical_pairs
    missing_in_live = canonical_pairs - live_pairs
    
    if extra_in_live or missing_in_live:
        logger.warning(f"⚠ PRE-TYPE1 boundary_segments DRIFT:")
        logger.warning(f"  Canonical (from triangle_segments): {len(canonical_pairs)} pairs")
        logger.warning(f"  Live (boundary_segments): {len(live_pairs)} pairs")
        if extra_in_live:
            logger.warning(f"  Phantom segments ({len(extra_in_live)}):")
            for pair in sorted(extra_in_live):
                logger.warning(f"    VP {pair[0]} ↔ VP {pair[1]}")
        if missing_in_live:
            logger.warning(f"  Missing segments ({len(missing_in_live)}):")
            for pair in sorted(missing_in_live):
                logger.warning(f"    VP {pair[0]} ↔ VP {pair[1]}")
    else:
        logger.info(f"✓ PRE-TYPE1 boundary_segments consistent: {len(live_pairs)} pairs match")
    
    # Log boundary_segments for each migration plan component's VPs
    for plan_entry in migration_plan:
        aux = plan_entry['auxiliary_component']
        comp_idx = plan_entry['component']['index']
        aux_set = set(aux)
        relevant_segs = []
        for seg in partition.boundary_segments:
            if seg.vp_idx_1 in aux_set or seg.vp_idx_2 in aux_set:
                relevant_segs.append((seg.vp_idx_1, seg.vp_idx_2))
        logger.debug(f"  Component {comp_idx} aux={aux}: "
                     f"{len(relevant_segs)} boundary_segments touching these VPs: {relevant_segs}")
    
    if not migration_plan:
        logger.info("\n⚠ No components selected for migration after analysis")
        return 0
    
    migrations_applied = 0
    failures = []
    
    # Temporarily suppress verbose switcher logging during migration
    import logging
    switcher_logger = logging.getLogger('src.core.topology_switcher')
    contour_logger = logging.getLogger('src.core.contour_partition')
    original_switcher_level = switcher_logger.level
    original_contour_level = contour_logger.level
    switcher_logger.setLevel(logging.WARNING)
    contour_logger.setLevel(logging.WARNING)
    
    logger.info("\nApplying migrations using pre-computed plan...")
    
    # Apply migrations using pre-computed plan
    for i, plan_entry in enumerate(migration_plan):
        comp = plan_entry['component']
        comp_idx = comp['index']
        
        # Pre-execution revalidation: verify auxiliary VPs are still on edges
        # incident to the target vertex (earlier migrations may have moved neighbors)
        target_vertex = comp['target_vertex']
        aux_vps = plan_entry['auxiliary_component']
        migrating_vp = plan_entry['migrating_vp']
        left_nb = plan_entry['left_neighbor']
        right_nb = plan_entry['right_neighbor']
        
        skip_migration = False
        for vp_idx in aux_vps:
            vp = partition.variable_points[vp_idx]
            if target_vertex not in vp.edge:
                logger.warning(f"  ⚠ Component {comp_idx}: VP {vp_idx} is on edge {vp.edge} "
                             f"which does NOT contain target vertex {target_vertex} — "
                             f"skipping (invalidated by earlier migration)")
                skip_migration = True
                break
        
        if skip_migration:
            failures.append((comp_idx, 'Pre-execution revalidation failed (VP edge no longer incident to target vertex)'))
            continue
        
        try:
            # Use pre-computed migration details (efficient path)
            result = switcher.apply_type1_switch_v2(
                component=comp,
                distance_preservation=distance_preservation,
                migrating_vp=migrating_vp,
                auxiliary_component=aux_vps,
                left_neighbor=left_nb,
                right_neighbor=right_nb
            )
            
            if result['success']:
                logger.info(f"  ✓ Component {comp_idx}: Migration successful")
                migrations_applied += 1
                
                # Rebuild boundary_segments after each migration so subsequent
                # migrations see accurate segment connectivity
                partition.rebuild_triangle_segments_from_current_vps()
            else:
                logger.error(f"\n  ✗ Component {comp_idx}: Migration FAILED")
                logger.error(f"    VPs: {comp['vp_indices']}")
                logger.error(f"    Target vertex: {comp['target_vertex']}")
                logger.error(f"    Error: {result.get('error', 'Unknown error')}")
                failures.append((comp_idx, result.get('error', 'Unknown')))
        
        except Exception as e:
            # Detailed exception message
            logger.error(f"\n  ✗ Component {comp_idx}: Exception during migration")
            logger.error(f"    VPs: {comp['vp_indices']}")
            logger.error(f"    Target vertex: {comp['target_vertex']}")
            logger.error(f"    Exception: {str(e)}")
            failures.append((comp_idx, str(e)))
    
    # Restore logging levels
    switcher_logger.setLevel(original_switcher_level)
    contour_logger.setLevel(original_contour_level)
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info(f"Type 1 migrations completed: {migrations_applied}/{len(migration_plan)}")
    if failures:
        logger.warning(f"  Failed components: {len(failures)}")
        for comp_idx, error in failures:
            logger.warning(f"    Component {comp_idx}: {error}")
    logger.info("="*80)
    
    return migrations_applied


def apply_type2_migrations(partition, mesh, tp_candidates, logger, steiner_handler=None, 
                          distance_preservation='preserve', migration_history=None, iteration_number=None):
    """Apply Type 2 migrations for boundary triple points.
    
    Args:
        partition: PartitionContour object
        mesh: TriMesh object
        tp_candidates: List of candidate triple point indices
        logger: Logger instance
        steiner_handler: Optional pre-created SteinerHandler (for efficiency)
        distance_preservation: VP placement strategy after migration
        migration_history: Optional Type2MigrationHistory object (for reverse migrations)
        iteration_number: Current iteration number (for history recording)
    """
    logger.info("")
    logger.info("="*80)
    logger.info(f"APPLYING TYPE 2 MIGRATIONS ({len(tp_candidates)} candidates)")
    logger.info(f"  Distance preservation: {distance_preservation}")
    if migration_history is not None:
        logger.info(f"  Migration history: {len(migration_history.records)} triple points tracked")
    logger.info("="*80)
    
    if len(tp_candidates) == 0:
        logger.info("No Type 2 migrations to apply")
        logger.info("="*80)
        return 0
    
    # Create topology objects
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    
    # Attach migration history if provided
    if migration_history is not None:
        switcher.type2_migration_history = migration_history
        migration_history.current_iteration = iteration_number
        logger.info(f"Attached migration history (iteration {iteration_number})")
    
    # Reuse steiner_handler if provided, otherwise create new
    if steiner_handler is None:
        steiner_handler = SteinerHandler(mesh, partition)
    
    migrations_applied = 0
    failures = []
    
    # Temporarily suppress verbose switcher logging during migration
    import logging
    switcher_logger = logging.getLogger('src.core.topology_switcher')
    contour_logger = logging.getLogger('src.core.contour_partition')
    original_switcher_level = switcher_logger.level
    original_contour_level = contour_logger.level
    switcher_logger.setLevel(logging.WARNING)
    contour_logger.setLevel(logging.WARNING)
    
    logger.info("\nApplying migrations...")
    
    # Apply migrations for each boundary triple point
    # tp_candidates contains TriplePoint objects, not indices
    for i, tp in enumerate(tp_candidates):
        # Find the index of this triple point in steiner_handler.triple_points
        tp_idx = None
        for idx, handler_tp in enumerate(steiner_handler.triple_points):
            if handler_tp.triangle_idx == tp.triangle_idx:
                tp_idx = idx
                break
        
        if tp_idx is None:
            logger.error(f"  ✗ Triple point {i} (triangle {tp.triangle_idx}): Could not find in steiner_handler")
            continue
        
        try:
            seg_count_before = len(partition.boundary_segments)
            
            result = switcher.apply_type2_switch_v4(
                steiner_handler=steiner_handler,
                triple_point_idx=tp_idx,
                distance_preservation=distance_preservation
            )
            
            if result['success']:
                seg_count_after = len(partition.boundary_segments)
                logger.info(f"  ✓ Triple point {tp_idx} (triangle {tp.triangle_idx}): Migration successful")
                logger.info(f"    boundary_segments: {seg_count_before} → {seg_count_after} "
                           f"(net {seg_count_after - seg_count_before:+d})")
                
                # Consistency check: compare boundary_segments VP pairs
                # against what triangle_segments would produce from scratch
                canonical_pairs = set()
                for ts in partition.triangle_segments:
                    vi = ts.var_point_indices
                    if len(vi) == 2:
                        canonical_pairs.add((min(vi[0], vi[1]), max(vi[0], vi[1])))
                    elif len(vi) == 3:
                        for a in range(3):
                            for b in range(a + 1, 3):
                                canonical_pairs.add((min(vi[a], vi[b]), max(vi[a], vi[b])))
                
                live_pairs = set()
                for seg in partition.boundary_segments:
                    live_pairs.add((min(seg.vp_idx_1, seg.vp_idx_2), max(seg.vp_idx_1, seg.vp_idx_2)))
                
                extra_in_live = live_pairs - canonical_pairs
                missing_in_live = canonical_pairs - live_pairs
                
                if extra_in_live or missing_in_live:
                    logger.warning(f"    ⚠ boundary_segments DRIFT detected after Type 2 tp={tp_idx}:")
                    if extra_in_live:
                        logger.warning(f"      Phantom segments (in boundary_segments but not triangle_segments): "
                                      f"{len(extra_in_live)}")
                        for pair in sorted(extra_in_live)[:5]:
                            logger.warning(f"        VP {pair[0]} ↔ VP {pair[1]}")
                    if missing_in_live:
                        logger.warning(f"      Missing segments (in triangle_segments but not boundary_segments): "
                                      f"{len(missing_in_live)}")
                        for pair in sorted(missing_in_live)[:5]:
                            logger.warning(f"        VP {pair[0]} ↔ VP {pair[1]}")
                else:
                    logger.info(f"    ✓ boundary_segments consistent with triangle_segments "
                               f"({len(live_pairs)} pairs match)")
                
                migrations_applied += 1
            else:
                # Detailed failure message (ERROR - operation failed)
                logger.error(f"\n  ✗ Triple point {tp_idx}: Migration FAILED")
                logger.error(f"    Triangle: {tp.triangle_idx}")
                logger.error(f"    VPs: {tp.var_point_indices}")
                logger.error(f"    Cells: {tp.cell_indices}")
                logger.error(f"    Error: {result.get('error', 'Unknown error')}")
                failures.append((tp_idx, result.get('error', 'Unknown')))
        
        except Exception as e:
            # Detailed exception message
            logger.error(f"\n  ✗ Triple point {tp_idx}: Exception during migration")
            logger.error(f"    Triangle: {tp.triangle_idx}")
            logger.error(f"    VPs: {tp.var_point_indices}")
            logger.error(f"    Cells: {tp.cell_indices}")
            logger.error(f"    Exception: {str(e)}")
            failures.append((tp_idx, str(e)))
    
    # Restore logging levels
    switcher_logger.setLevel(original_switcher_level)
    contour_logger.setLevel(original_contour_level)
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info(f"Type 2 migrations completed: {migrations_applied}/{len(tp_candidates)}")
    if failures:
        logger.warning(f"  Failed triple points: {len(failures)}")
        for tp_idx, error in failures:
            logger.warning(f"    Triple point {tp_idx}: {error}")
    logger.info("="*80)
    
    return migrations_applied


def run_optimization_with_optimizer(optimizer, target_area, max_iter, tolerance, method, logger):
    """Run one optimization iteration using existing optimizer.
    
    Args:
        optimizer: Pre-created PerimeterOptimizer instance
        target_area: Target area per cell
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        method: Optimization method
        logger: Logger instance
    """
    logger.info("")
    logger.info("="*80)
    logger.info("RUNNING OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Target area: {target_area:.10f}")
    logger.info(f"Max iterations: {max_iter}")
    logger.info(f"Tolerance: {tolerance}")
    logger.info(f"Method: {method}")
    
    # Calculate initial perimeter before optimization
    x0 = optimizer.partition.get_variable_vector()
    initial_perimeter = optimizer.objective(x0)
    logger.info(f"\nInitial perimeter: {initial_perimeter:.10f}")
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(
        max_iter=max_iter,
        tol=tolerance,
        method=method
    )
    opt_time = time.time() - start_time
    
    # Get optimization info
    opt_info = optimizer.get_optimization_info(result)
    
    # Calculate perimeter reduction metrics
    final_perimeter = opt_info['final_perimeter']
    perimeter_reduction = initial_perimeter - final_perimeter
    percent_reduction = (perimeter_reduction / initial_perimeter * 100) if initial_perimeter > 0 else 0.0
    
    # Calculate constraint violations (area violations)
    final_areas = np.array(opt_info['final_areas'])
    constraint_violations = final_areas - target_area
    
    # Add calculated values to opt_info for export
    opt_info['initial_perimeter'] = float(initial_perimeter)
    opt_info['perimeter_reduction'] = float(perimeter_reduction)
    opt_info['percent_reduction'] = float(percent_reduction)
    opt_info['final_constraint_violations'] = constraint_violations.tolist()
    
    logger.info(f"\nOptimization completed in {opt_time:.2f}s")
    logger.info(f"  Success: {opt_info['success']}")
    logger.info(f"  Message: {opt_info['message']}")
    logger.info(f"  Iterations: {opt_info['n_iterations']}")
    logger.info(f"  Function evaluations: {opt_info['n_function_evals']}")
    logger.info(f"  Final perimeter: {final_perimeter:.10f}")
    logger.info(f"  Improvement: {perimeter_reduction:.10f} ({percent_reduction:.4f}%)")
    
    # Check constraint violations
    logger.info(f"\nConstraint violations:")
    for i, viol in enumerate(constraint_violations):
        logger.info(f"  Cell {i}: {viol:.2e}")
    
    max_violation = np.max(np.abs(constraint_violations))
    logger.info(f"  Max violation: {max_violation:.2e}")
    
    logger.info("="*80)
    
    return result, opt_info


def _check_indicator_vp_consistency(partition, mesh, logger):
    """
    Roundtrip consistency check: verify that partition.indicator_functions
    produces exactly the same set of boundary edges (and thus VP count) as
    the live partition.variable_points.

    This catches the case where topology migrations updated VP edges in memory
    but forgot to flip the corresponding mesh-vertex labels in indicator_functions.

    Returns True if consistent, False if a mismatch is detected.
    """
    import numpy as np
    from src.find_contours import ContourAnalyzer

    logger.info("  Running indicator_functions ↔ VP roundtrip consistency check...")

    # Reconstruct boundary edges purely from indicator_functions
    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
    reconstructed_edges = set()
    for face in mesh.faces:
        v0, v1, v2 = face
        l0, l1, l2 = vertex_labels[v0], vertex_labels[v1], vertex_labels[v2]
        if l0 != l1:
            reconstructed_edges.add(tuple(sorted((v0, v1))))
        if l1 != l2:
            reconstructed_edges.add(tuple(sorted((v1, v2))))
        if l0 != l2:
            reconstructed_edges.add(tuple(sorted((v0, v2))))

    # Collect edges from live (active) VPs only — inactive VPs are not
    # represented in indicator_functions after migrations.
    live_edges = {vp.edge for vp in partition.variable_points if getattr(vp, 'active', True)}

    only_in_reconstructed = reconstructed_edges - live_edges
    only_in_live = live_edges - reconstructed_edges

    if only_in_reconstructed or only_in_live:
        logger.error("  ✗ CONSISTENCY CHECK FAILED: indicator_functions do not match live VPs")
        logger.error(f"    Live VPs      : {len(live_edges)} unique edges")
        logger.error(f"    Reconstructed : {len(reconstructed_edges)} unique edges from indicator_functions")
        logger.error(f"    Edges in indicator_functions but NOT in live VPs ({len(only_in_reconstructed)}): "
                     f"{sorted(only_in_reconstructed)[:10]}{'...' if len(only_in_reconstructed) > 10 else ''}")
        logger.error(f"    Edges in live VPs but NOT in indicator_functions ({len(only_in_live)}): "
                     f"{sorted(only_in_live)[:10]}{'...' if len(only_in_live) > 10 else ''}")
        logger.error("  *** This will cause a VP count mismatch when the file is reloaded! ***")
        return False
    else:
        logger.info(f"  ✓ Consistent: {len(live_edges)} boundary edges match in both live VPs and indicator_functions")
        return True


def export_results(partition, opt_info, output_path, logger, migration_history=None, mesh=None):
    """Export results to HDF5 file in same format as input, including updated indicator functions."""
    logger.info("")
    logger.info("="*80)
    logger.info("EXPORTING RESULTS")
    logger.info("="*80)
    logger.info(f"Output file: {output_path}")

    # Run consistency check before writing (requires mesh for face iteration)
    if mesh is not None:
        _check_indicator_vp_consistency(partition, mesh, logger)
    else:
        logger.warning("  Skipping consistency check: mesh not provided to export_results")

    # Get optimized lambda parameters and edges for ACTIVE VPs only.
    # indicator_functions reflects active topology, so lambda/edge arrays must match.
    active_lambdas = []
    active_edges = []
    for vp in partition.variable_points:
        if getattr(vp, 'active', True):
            active_lambdas.append(vp.lambda_param)
            active_edges.append(vp.edge)

    lambda_opt = np.array(active_lambdas)
    vp_edges = np.array(active_edges, dtype=np.int64)

    # Get indicator functions (updated after migrations)
    indicator_functions = partition.indicator_functions

    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Save lambda parameters (active VPs only)
        f.create_dataset('lambda_parameters', data=lambda_opt)

        # Save VP edge associations (active VPs only)
        f.create_dataset('vp_edges', data=vp_edges)
        
        # Save indicator functions (critical for visualization after migrations)
        f.create_dataset('indicator_functions', data=indicator_functions)
        
        # Save metadata
        f.attrs['n_variable_points'] = len(lambda_opt)
        f.attrs['n_cells'] = partition.n_cells
        f.attrs['final_perimeter'] = opt_info['final_perimeter']
        f.attrs['optimization_success'] = opt_info['success']
        f.attrs['optimization_iterations'] = opt_info['n_iterations']
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save migration history (legacy path only)
        if _use_legacy and migration_history is not None and len(migration_history.records) > 0:
            save_type2_migration_history(f, migration_history)
            logger.info(f"✓ Saved migration history: {len(migration_history.records)} triple points tracked")
        
        # Save optimization info
        opt_grp = f.create_group('optimization_info')
        opt_grp.attrs['initial_perimeter'] = opt_info['initial_perimeter']
        opt_grp.attrs['final_perimeter'] = opt_info['final_perimeter']
        opt_grp.attrs['perimeter_reduction'] = opt_info['perimeter_reduction']
        opt_grp.attrs['percent_reduction'] = opt_info['percent_reduction']
        opt_grp.create_dataset('constraint_violations', data=opt_info['final_constraint_violations'])
    
    logger.info("✓ Results exported successfully")
    logger.info(f"  {len(lambda_opt)} lambda parameters saved")
    logger.info(f"  Indicator functions saved: {indicator_functions.shape}")
    logger.info(f"  Metadata and optimization info included")
    
    # Perimeter roundtrip check: reload the file and compare perimeter
    if mesh is not None:
        try:
            from examples.data_loader import load_partition_from_refined_file
            from src.core.perimeter_calculator import PerimeterCalculator
            
            mesh_reloaded, partition_reloaded = load_partition_from_refined_file(output_path, verbose=False)
            
            lambda_reloaded = partition_reloaded.get_variable_vector()
            partition_reloaded.set_variable_vector(lambda_reloaded)
            
            perim_calc_reloaded = PerimeterCalculator(mesh_reloaded, partition_reloaded)
            regular_perimeter_reloaded = perim_calc_reloaded.compute_total_perimeter(lambda_reloaded)
            
            steiner_handler_reloaded = SteinerHandler(mesh_reloaded, partition_reloaded)
            steiner_perimeter_reloaded = steiner_handler_reloaded.get_total_perimeter_contribution()
            
            perimeter_reloaded = regular_perimeter_reloaded + steiner_perimeter_reloaded
            
            in_memory_perimeter = opt_info['final_perimeter']
            rel_diff = abs(perimeter_reloaded - in_memory_perimeter) / max(in_memory_perimeter, 1e-12)
            
            if rel_diff < 1e-4:
                logger.info(f"  ✓ Roundtrip perimeter check PASSED: "
                          f"in-memory={in_memory_perimeter:.6f}, reloaded={perimeter_reloaded:.6f} "
                          f"(rel_diff={rel_diff:.2e})")
                logger.info(f"    Regular={regular_perimeter_reloaded:.6f}, "
                          f"Steiner={steiner_perimeter_reloaded:.6f}")
            else:
                logger.warning(f"  ⚠ Roundtrip perimeter check FAILED: "
                             f"in-memory={in_memory_perimeter:.6f}, reloaded={perimeter_reloaded:.6f} "
                             f"(rel_diff={rel_diff:.2e})")
                logger.warning(f"    Regular={regular_perimeter_reloaded:.6f}, "
                             f"Steiner={steiner_perimeter_reloaded:.6f}")
                logger.warning(f"    VP count in-memory: {len(lambda_opt)}, "
                             f"reloaded: {len(lambda_reloaded)}")
        except Exception as e:
            logger.warning(f"  ⚠ Roundtrip perimeter check skipped due to error: {e}")
    
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Test migration and continue optimization workflow'
    )
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to refined contours .h5 file')
    parser.add_argument('--migration-type', type=str, default='both',
                       choices=['type1', 'type2', 'both'],
                       help='Which migrations to apply (default: both)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output file (default: auto-generated in same directory as input)')
    parser.add_argument('--max-opt-iter', type=int, default=1000,
                       help='Maximum optimization iterations (default: 1000)')
    parser.add_argument('--tolerance', type=float, default=1e-7,
                       help='Optimization convergence tolerance (default: 1e-7)')
    parser.add_argument('--boundary-tol', type=float, default=0.01,
                       help='Threshold for boundary point detection (default: 0.01)')
    parser.add_argument('--method', type=str, default='SLSQP',
                       choices=['SLSQP', 'trust-constr'],
                       help='Optimization method (default: SLSQP)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging (verbose output with detailed diagnostics)')
    parser.add_argument('--log-level', type=str, default=None,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO, or DEBUG if --debug flag is used). '
                            'Note: --debug flag overrides this setting.')
    parser.add_argument('--distance-preservation', type=str, default='preserve',
                       help='VP placement strategy after Type 1 migration: '
                            '"preserve" (default, maintains original distance to target vertex), '
                            '"midpoint" (places at edge midpoint, λ=0.5), '
                            'or a number between 0.0 and 1.0 (custom distance)')
    parser.add_argument('--max-iterations', type=int, default=1,
                       help='Maximum refinement iterations (default: 1 for single cycle, '
                            'set higher for iterative refinement until convergence)')
    parser.add_argument('--use-legacy', action='store_true',
                       help='Use legacy migration path (TopologySwitcher, Type1ComponentAnalyzer, etc.)')
    
    args = parser.parse_args()
    
    # Validate distance_preservation argument
    distance_preservation = args.distance_preservation
    if distance_preservation not in ['preserve', 'midpoint']:
        # Try to parse as float
        try:
            dist_value = float(distance_preservation)
            if dist_value < 0.0 or dist_value > 1.0:
                print(f"ERROR: Custom distance must be between 0.0 and 1.0, got {dist_value}")
                return 1
            # Valid custom distance
        except ValueError:
            print(f"ERROR: Invalid distance_preservation value: '{distance_preservation}'")
            print("Must be 'preserve', 'midpoint', or a number between 0.0 and 1.0")
            return 1
    
    # Setup logging: --debug flag takes precedence over --log-level
    if args.debug:
        log_level = 'DEBUG'
    elif args.log_level:
        log_level = args.log_level
    else:
        log_level = 'INFO'
    
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    if args.debug:
        logger.info("Debug mode enabled - verbose logging active")
    
    logger.info("="*80)
    logger.info("TEST: Migration and Continue Optimization")
    logger.info("="*80)
    logger.info(f"Input file: {args.solution}")
    logger.info(f"Migration type: {args.migration_type}")
    logger.info(f"Distance preservation: {args.distance_preservation}")
    logger.info(f"Max iterations: {args.max_iterations}")
    
    # Check input file exists
    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1
    
    # Detect starting iteration number from filename
    import re
    iteration_match = re.search(r'_iteration(\d+)_refined_contours\.h5$', args.solution)
    
    # Also detect if boundary_tol is in the filename
    btol_match = re.search(r'_btol([\d.]+(?:e-?\d+)?)_', args.solution)
    has_btol_in_name = btol_match is not None
    
    if iteration_match:
        starting_iteration = int(iteration_match.group(1))
        logger.info(f"Detected input as iteration {starting_iteration} file")
    else:
        starting_iteration = 1
        logger.info("Input appears to be first refined file (iteration 1)")
    
    # Determine base output path for iteration files
    if args.output is None:
        # Remove iteration number and _refined_contours suffix to get base path
        if iteration_match:
            # Remove existing iteration suffix
            base_path = args.solution.replace(f'_iteration{starting_iteration}_refined_contours.h5', '')
        else:
            base_path = args.solution.replace('_refined_contours.h5', '')
        
        # Also remove btol from base path if present (we'll add it back consistently)
        if btol_match:
            base_path = base_path.replace(f'_btol{btol_match.group(1)}', '')
    else:
        # User provided output path - derive base from it
        base_path = args.output.replace(f'_iteration{starting_iteration + 1}_refined_contours.h5', '')
        base_path = base_path.replace('_refined_contours.h5', '')
        # Remove btol if present
        if btol_match:
            base_path = base_path.replace(f'_btol{btol_match.group(1)}', '')
    
    logger.info(f"Base output path: {base_path}")
    logger.info(f"Iteration files will be: {base_path}_btol{args.boundary_tol}_iteration{{N}}_refined_contours.h5")
    
    # Create output directory if needed
    output_dir = os.path.dirname(base_path) if os.path.dirname(base_path) else '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Stage 1: Load refined data
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("STAGE 1: Loading refined data...")
    
    try:
        mesh, partition = load_partition_from_refined_file(args.solution, verbose=True)
    except Exception as e:
        logger.error(f"Failed to load refined data: {str(e)}")
        return 1
    
    # -------------------------------------------------------------------------
    # Stage 2: Initialize optimizer and target area
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("INITIAL STATE DIAGNOSTICS")
    logger.info("="*80)
    
    # Mesh info
    logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
    logger.info(f"Partition: {partition.n_cells} cells, {len(partition.variable_points)} VPs")
    
    # Calculate target area (equal division)
    total_area = float(mesh.M.sum())
    target_area = total_area / partition.n_cells
    logger.info(f"Target area per cell: {target_area:.10f}")
    logger.info("="*80)
    
    # -------------------------------------------------------------------------
    # Initialize Migration History (BEFORE loop)
    # -------------------------------------------------------------------------
    
    if _use_legacy:
        migration_history = Type2MigrationHistory()
        if starting_iteration > 1:
            try:
                with h5py.File(args.solution, 'r') as f:
                    migration_history = load_type2_migration_history(f)
                logger.info(f"Loaded migration history from {args.solution}")
                logger.info(f"  Tracked triple points: {len(migration_history.records)}")
                for orig_tri, record in migration_history.records.items():
                    logger.info(f"    Triangle {orig_tri}: {record.triangle_sequence}")
            except Exception as e:
                logger.warning(f"Could not load migration history: {e}")
                logger.warning("Starting with empty history")
                migration_history = Type2MigrationHistory()
        else:
            logger.info("Starting with empty migration history (iteration 1)")
    else:
        migration_history = None
        logger.info("Migration history managed by MigrationOrchestrator")
    
    # -------------------------------------------------------------------------
    # Stage 3: ITERATIVE REFINEMENT LOOP
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING ITERATIVE REFINEMENT LOOP")
    logger.info("="*80)
    logger.info(f"Starting from iteration {starting_iteration}")
    logger.info(f"Maximum iterations: {args.max_iterations}")
    logger.info("="*80)
    
    global_start_time = time.time()
    converged = False
    total_type1_migrations = 0
    total_type2_migrations = 0
    
    for iteration_idx in range(args.max_iterations):
        current_iteration = starting_iteration + iteration_idx + 1
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"ITERATION {current_iteration}")
        logger.info("="*80)
        iteration_start_time = time.time()
        
        # ---------------------------------------------------------------------
        # Phase 1: Create fresh optimizer
        # ---------------------------------------------------------------------
        logger.info("Creating optimizer...")
        optimizer = PerimeterOptimizer(partition, mesh, target_area)
        
        lambda_vec = partition.get_variable_vector()
        total_perimeter = optimizer.perim_calc.compute_total_perimeter(lambda_vec)
        steiner_perimeter = optimizer.steiner_handler.get_total_perimeter_contribution()
        current_perimeter = total_perimeter + steiner_perimeter
        
        active_vps = sum(1 for vp in partition.variable_points if vp.active)
        logger.info(f"Current state:")
        logger.info(f"  VPs: {len(partition.variable_points)} ({active_vps} active)")
        logger.info(f"  Triple points: {len(optimizer.steiner_handler.triple_points)}")
        logger.info(f"  Perimeter: {current_perimeter:.10f}")
        
        # ---------------------------------------------------------------------
        # Phase 2: Detect topology switches
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("Detecting topology switches...")
        
        if _use_legacy:
            switches_needed, switch_info = optimizer.check_topology_switches_needed(
                tol=args.boundary_tol
            )
            no_switches = not switches_needed
        else:
            mesh_topology = MeshTopology(mesh)
            orchestrator = MigrationOrchestrator(
                partition, mesh, mesh_topology,
                MigrationConfig(delta=args.boundary_tol)
            )
            detection = orchestrator.detect_all_triggers(delta=args.boundary_tol)
            no_switches = not detection.type1_triggers and not detection.type2_triggers
        
        if no_switches:
            logger.info("✓ No topology switches detected - CONVERGED")
            converged = True
            output_path = f"{base_path}_btol{args.boundary_tol}_iteration{current_iteration}_refined_contours.h5"
            opt_info = {
                'success': True,
                'n_iterations': 0,
                'n_function_evals': 0,
                'initial_perimeter': current_perimeter,
                'final_perimeter': current_perimeter,
                'perimeter_reduction': 0.0,
                'percent_reduction': 0.0,
                'final_constraint_violations': (optimizer.area_calc.compute_all_cell_areas(lambda_vec) - target_area).tolist(),
                'message': 'Converged - no switches needed'
            }
            export_results(partition, opt_info, output_path, logger, migration_history, mesh=mesh)
            break
        
        if _use_legacy:
            type1_candidates = []
            for vp_idx, vp in enumerate(partition.variable_points):
                if vp.on_boundary(tol=args.boundary_tol):
                    type1_candidates.append(vp_idx)
            type2_candidates = optimizer.steiner_handler.get_boundary_triple_points(tol=args.boundary_tol)
            logger.info(f"✓ Switches detected:")
            logger.info(f"  Type 1 candidates: {len(type1_candidates)}")
            logger.info(f"  Type 2 candidates: {len(type2_candidates)}")
        else:
            logger.info(f"✓ Switches detected: {len(detection.type1_triggers)} Type 1, "
                       f"{len(detection.type2_triggers)} Type 2")
        
        # ---------------------------------------------------------------------
        # Phase 3: Apply migrations (Type 2 first, then Type 1)
        # ---------------------------------------------------------------------
        iteration_migrations = 0
        
        if _use_legacy:
            if args.migration_type in ['type2', 'both'] and len(type2_candidates) > 0:
                migrations = apply_type2_migrations(
                    partition, mesh, type2_candidates, logger,
                    steiner_handler=optimizer.steiner_handler,
                    distance_preservation=args.distance_preservation,
                    migration_history=migration_history,
                    iteration_number=current_iteration
                )
                iteration_migrations += migrations
                total_type2_migrations += migrations
            
            if args.migration_type in ['type2', 'both'] and len(type2_candidates) > 0 and iteration_migrations > 0:
                logger.info("")
                logger.info("Rebuilding boundary_segments after Type 2 migrations...")
                partition.rebuild_triangle_segments_from_current_vps()
                logger.info(f"  boundary_segments rebuilt: {len(partition.boundary_segments)} segments")
            
            if args.migration_type in ['type1', 'both'] and len(type1_candidates) > 0:
                migrations = apply_type1_migrations(
                    partition, mesh, type1_candidates, logger,
                    boundary_tol=args.boundary_tol,
                    distance_preservation=args.distance_preservation,
                    steiner_handler=optimizer.steiner_handler
                )
                iteration_migrations += migrations
                total_type1_migrations += migrations
        else:
            mig_result = orchestrator.execute_migrations(mode='batch')
            
            if mig_result.failed:
                logger.error("")
                logger.error("="*80)
                logger.error("MIGRATION FAILURE")
                logger.error("="*80)
                logger.error(f"Error: {mig_result.error_message}")
                logger.error("="*80)
                return 1
            
            iteration_migrations = (mig_result.type1_applied +
                                    mig_result.type2_forward_applied +
                                    mig_result.type2_rollbacks_applied)
            total_type1_migrations += mig_result.type1_applied
            total_type2_migrations += mig_result.type2_forward_applied + mig_result.type2_rollbacks_applied
            
            logger.info(f"  Type 1 applied: {mig_result.type1_applied}")
            logger.info(f"  Type 2 forward: {mig_result.type2_forward_applied}")
            logger.info(f"  Type 2 rollbacks: {mig_result.type2_rollbacks_applied}")
        
        if iteration_migrations == 0:
            logger.warning("No migrations were applied despite switches detected")
            logger.warning("This may indicate tolerance issues or all migrations failed")
            converged = True
            break
        
        logger.info(f"✓ Iteration {current_iteration} migrations: {iteration_migrations}")
        
        # ---------------------------------------------------------------------
        # Phase 4: Optimize after migrations
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info("Starting optimization after migrations...")
        
        optimizer = PerimeterOptimizer(partition, mesh, target_area)
        
        try:
            result, opt_info = run_optimization_with_optimizer(
                optimizer, target_area,
                args.max_opt_iter, args.tolerance, args.method, logger
            )
        except Exception as e:
            logger.error(f"Optimization failed with exception: {str(e)}")
            logger.error("ABORTING - Please debug the optimization failure")
            return 1
        
        if not opt_info['success']:
            logger.warning("="*80)
            logger.warning("OPTIMIZATION DID NOT FULLY CONVERGE")
            logger.warning("="*80)
            logger.warning(f"Reason: {opt_info['message']}")
            logger.warning(f"Iterations: {opt_info['n_iterations']}")
            logger.warning(f"Perimeter reduction achieved: {opt_info['perimeter_reduction']:.6f} ({opt_info['percent_reduction']:.4f}%)")
            logger.warning(f"Max constraint violation: {np.max(np.abs(opt_info['final_constraint_violations'])):.2e}")
            logger.warning("")
            logger.warning("Continuing to save results (partial convergence may still be useful)")
            logger.warning("Consider:")
            logger.warning("  - Increasing --max-opt-iter for more iterations")
            logger.warning("  - Relaxing --tolerance if constraint violations are acceptable")
            logger.warning("  - Running another iteration starting from the saved file")
            logger.warning("="*80)
        
        # ---------------------------------------------------------------------
        # Phase 5: Export iteration results
        # ---------------------------------------------------------------------
        output_path = f"{base_path}_btol{args.boundary_tol}_iteration{current_iteration}_refined_contours.h5"
        export_results(partition, opt_info, output_path, logger, migration_history, mesh=mesh)
        
        iteration_time = time.time() - iteration_start_time
        logger.info(f"✓ Iteration {current_iteration} completed in {iteration_time:.2f}s")
        logger.info(f"  Final perimeter: {opt_info['final_perimeter']:.10f}")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - global_start_time
    
    logger.info("")
    logger.info("="*80)
    logger.info("ITERATIVE REFINEMENT COMPLETED")
    logger.info("="*80)
    logger.info(f"Status: {'CONVERGED' if converged else 'MAX ITERATIONS REACHED'}")
    logger.info(f"Total iterations: {iteration_idx + 1}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Total Type 1 migrations: {total_type1_migrations}")
    logger.info(f"Total Type 2 migrations: {total_type2_migrations}")
    logger.info(f"Final output: {output_path}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
