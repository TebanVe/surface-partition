#!/usr/bin/env python3
"""
Iterative Perimeter Refinement with Automatic Topology Migrations

This script implements the complete iterative refinement workflow:
1. Load initial solution from PGD/SLSQP relaxation optimization
2. Extract contours via indicator functions
3. Build PartitionContour data structure
4. ITERATIVE LOOP (until convergence or max_iterations):
   a. Run constrained perimeter optimization
   b. Export pre-switch state for visualization
   c. Detect topology switches needed
   d. Apply Type 2 migrations (triple points)
   e. Apply Type 1 migrations (boundary VPs)
   f. Reinitialize for next iteration
5. Export final converged state

Key features:
- Uses latest v2/v4 migration methods with full component analysis
- Exports state at each iteration before switches (visualizable)
- Exits immediately on first migration failure with detailed diagnostics
- Automatic per-iteration optimization progress logging
- Type 2 migrations applied before Type 1 (proven order)

Usage:
    python testing/refine_perimeter_iterative.py \\
        --solution results/run_xyz/solution_level0.h5 \\
        --max-iterations 10 \\
        --tolerance 1e-7 \\
        --boundary-tol 0.01

Author: Perimeter Refinement Team
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

from src.find_contours import ContourAnalyzer
from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.core.perimeter_optimizer import PerimeterOptimizer
from src.core.mesh_topology import MeshTopology
from src.logging_config import get_logger, setup_logging

# Early parse for --use-legacy (before migration-specific imports)
_preparser = argparse.ArgumentParser(add_help=False)
_preparser.add_argument('--use-legacy', action='store_true', help='Use legacy TopologySwitcher')
_preargs, _ = _preparser.parse_known_args()

if _preargs.use_legacy:
    from src.core.topology_switcher_legacy import TopologySwitcher
    from src.core.type1_component_analyzer import Type1ComponentAnalyzer
    from src.core.type2_migration_io import load_type2_migration_history, save_type2_migration_history
else:
    from src.core.migration_orchestrator import MigrationOrchestrator, MigrationConfig
    from src.core.migration_types import TriplePointHistory


def apply_type1_migrations(partition, mesh, vp_candidates, logger, boundary_tol=0.01, 
                          distance_preservation='preserve', steiner_handler=None):
    """
    Apply Type 1 migrations for boundary VPs.
    
    Returns immediately on first failure with detailed failure info.
    
    Args:
        partition: PartitionContour object
        mesh: TriMesh object
        vp_candidates: List of candidate VP indices
        logger: Logger instance
        boundary_tol: Boundary tolerance (default: 0.01)
        distance_preservation: VP placement strategy after migration
        steiner_handler: Optional pre-created SteinerHandler (for efficiency)
    
    Returns:
        dict with keys:
            - migrations_applied: int
            - failed: bool
            - failure_type: str (if failed=True)
            - component: dict (if failed=True)
            - component_idx: int (if failed=True)
            - error: str (if failed=True)
            - result: dict (if failed=True)
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
        return {'migrations_applied': 0, 'failed': False}
    
    # Create topology objects
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    analyzer = Type1ComponentAnalyzer(mesh, partition, mesh_topology, 
                                     steiner_handler=steiner_handler)
    
    # Run full component analysis
    logger.info("\nStep 1: Running full component analysis...")
    logger.info("="*80)
    
    # Temporarily suppress verbose switcher logging
    import logging
    switcher_logger = logging.getLogger('src.core.topology_switcher')
    original_level = switcher_logger.level
    switcher_logger.setLevel(logging.WARNING)
    
    analysis_result = analyzer.run_full_analysis(
        boundary_tol=boundary_tol, 
        conflict_strategy='exclude_one',
        build_migration_plan=True,
        protect_type2=True  # Enable Type 2 protection (topology-based)
    )
    
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
    
    if not migration_plan:
        logger.info("\n⚠ No components selected for migration after analysis")
        return {'migrations_applied': 0, 'failed': False}
    
    migrations_applied = 0
    
    # Temporarily suppress verbose logging during migration
    contour_logger = logging.getLogger('src.core.contour_partition')
    original_contour_level = contour_logger.level
    switcher_logger.setLevel(logging.WARNING)
    contour_logger.setLevel(logging.WARNING)
    
    logger.info("\nApplying migrations using pre-computed plan...")
    
    # Apply migrations using pre-computed plan
    for i, plan_entry in enumerate(migration_plan):
        comp = plan_entry['component']
        comp_idx = comp['index']
        
        try:
            # Use pre-computed migration details
            result = switcher.apply_type1_switch_v2(
                component=comp,
                distance_preservation=distance_preservation,
                migrating_vp=plan_entry['migrating_vp'],
                auxiliary_component=plan_entry['auxiliary_component'],
                left_neighbor=plan_entry['left_neighbor'],
                right_neighbor=plan_entry['right_neighbor']
            )
            
            if result['success']:
                logger.info(f"  ✓ Component {comp_idx}: Migration successful")
                migrations_applied += 1
            else:
                # RETURN IMMEDIATELY ON FIRST FAILURE
                switcher_logger.setLevel(original_level)
                contour_logger.setLevel(original_contour_level)
                return {
                    'migrations_applied': migrations_applied,
                    'failed': True,
                    'failure_type': 'type1',
                    'component': comp,
                    'component_idx': comp_idx,
                    'error': result.get('error', 'Unknown error'),
                    'result': result
                }
        
        except Exception as e:
            # RETURN IMMEDIATELY ON EXCEPTION
            switcher_logger.setLevel(original_level)
            contour_logger.setLevel(original_contour_level)
            return {
                'migrations_applied': migrations_applied,
                'failed': True,
                'failure_type': 'type1',
                'component': comp,
                'component_idx': comp_idx,
                'error': str(e),
                'result': {'exception': True}
            }
    
    # Restore logging levels
    switcher_logger.setLevel(original_level)
    contour_logger.setLevel(original_contour_level)
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info(f"Type 1 migrations completed: {migrations_applied}/{len(migration_plan)}")
    logger.info("="*80)
    
    return {'migrations_applied': migrations_applied, 'failed': False}


def apply_type2_migrations(partition, mesh, tp_candidates, logger, steiner_handler=None, 
                          distance_preservation='preserve', migration_history=None, iteration_number=None):
    """
    Apply Type 2 migrations for boundary triple points.
    
    Returns immediately on first failure with detailed failure info.
    
    Args:
        partition: PartitionContour object
        mesh: TriMesh object
        tp_candidates: List of TriplePoint objects
        logger: Logger instance
        steiner_handler: Optional pre-created SteinerHandler (for efficiency)
        distance_preservation: VP placement strategy after migration
        migration_history: Optional Type2MigrationHistory object (for reverse migrations)
        iteration_number: Current iteration number (for history recording)
    
    Returns:
        dict with keys:
            - migrations_applied: int
            - reversed_migrations: int
            - failed: bool
            - failure_type: str (if failed=True)
            - triple_point: TriplePoint (if failed=True)
            - tp_idx: int (if failed=True)
            - error: str (if failed=True)
            - result: dict (if failed=True)
    """
    logger.info("")
    logger.info("="*80)
    logger.info(f"APPLYING TYPE 2 MIGRATIONS ({len(tp_candidates)} candidates)")
    logger.info(f"  Distance preservation: {distance_preservation}")
    if migration_history is not None:
        logger.info(f"  Migration history: {len(migration_history.records)} triple points tracked")
        logger.info(f"  Iteration: {iteration_number}")
    logger.info("="*80)
    
    if len(tp_candidates) == 0:
        logger.info("No Type 2 migrations to apply")
        logger.info("="*80)
        return {'migrations_applied': 0, 'reversed_migrations': 0, 'failed': False}
    
    # Create topology objects
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    
    # Attach migration history and set current iteration
    if migration_history is not None:
        switcher.type2_migration_history = migration_history
        migration_history.current_iteration = iteration_number
    
    # Reuse steiner_handler if provided
    if steiner_handler is None:
        from src.core.steiner_handler import SteinerHandler
        steiner_handler = SteinerHandler(mesh, partition)
    
    migrations_applied = 0
    reversed_migrations = 0
    
    # Temporarily suppress verbose logging
    import logging
    switcher_logger = logging.getLogger('src.core.topology_switcher')
    contour_logger = logging.getLogger('src.core.contour_partition')
    original_switcher_level = switcher_logger.level
    original_contour_level = contour_logger.level
    switcher_logger.setLevel(logging.WARNING)
    contour_logger.setLevel(logging.WARNING)
    
    logger.info("\nApplying migrations...")
    
    # Apply migrations for each boundary triple point
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
            result = switcher.apply_type2_switch_v4(
                steiner_handler=steiner_handler,
                triple_point_idx=tp_idx,
                distance_preservation=distance_preservation
            )
            
            if result['success']:
                if result.get('reversed', False):
                    logger.info(f"  ✓ Triple point {tp_idx} (triangle {tp.triangle_idx}): REVERSE migration successful (reversed {result.get('num_reversed', 1)} migration(s))")
                    reversed_migrations += result.get('num_reversed', 1)
                else:
                    logger.info(f"  ✓ Triple point {tp_idx} (triangle {tp.triangle_idx}): Forward migration successful")
                migrations_applied += 1
            else:
                # RETURN IMMEDIATELY ON FIRST FAILURE
                switcher_logger.setLevel(original_switcher_level)
                contour_logger.setLevel(original_contour_level)
                return {
                    'migrations_applied': migrations_applied,
                    'reversed_migrations': reversed_migrations,
                    'failed': True,
                    'failure_type': 'type2',
                    'triple_point': tp,
                    'tp_idx': tp_idx,
                    'error': result.get('error', 'Unknown error'),
                    'result': result
                }
        
        except Exception as e:
            # RETURN IMMEDIATELY ON EXCEPTION
            switcher_logger.setLevel(original_switcher_level)
            contour_logger.setLevel(original_contour_level)
            return {
                'migrations_applied': migrations_applied,
                'reversed_migrations': reversed_migrations,
                'failed': True,
                'failure_type': 'type2',
                'triple_point': tp,
                'tp_idx': tp_idx,
                'error': str(e),
                'result': {'exception': True}
            }
    
    # Restore logging levels
    switcher_logger.setLevel(original_switcher_level)
    contour_logger.setLevel(original_contour_level)
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info(f"Type 2 migrations completed: {migrations_applied}/{len(tp_candidates)}")
    if reversed_migrations > 0:
        logger.info(f"  Forward migrations: {migrations_applied - reversed_migrations}")
        logger.info(f"  Reverse migrations: {reversed_migrations}")
    logger.info("="*80)
    
    return {'migrations_applied': migrations_applied, 'reversed_migrations': reversed_migrations, 'failed': False}


def export_intermediate_state(partition, iteration_number, base_output_path, opt_info, logger, migration_history=None, boundary_tol=None):
    """
    Export intermediate partition state in exact format as test_migration_and_continue.py.
    
    This ensures visualization scripts can read iteration files identically to test outputs.
    
    Args:
        partition: PartitionContour object
        iteration_number: Current iteration number (1, 2, 3, ...)
        base_output_path: Base path without extension (e.g., "results/run_xyz/solution_level0")
        opt_info: Dictionary with optimization results
        logger: Logger instance
        migration_history: Optional Type2MigrationHistory object to save
        boundary_tol: Optional boundary tolerance value to include in filename
    
    Returns:
        str: Path to exported file
    """
    # Construct filename with boundary_tol if provided
    if boundary_tol is not None:
        output_path = f"{base_output_path}_btol{boundary_tol}_iteration{iteration_number}_refined_contours.h5"
    else:
        output_path = f"{base_output_path}_iteration{iteration_number}_refined_contours.h5"
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"EXPORTING ITERATION {iteration_number} STATE")
    logger.info("="*80)
    logger.info(f"Output file: {output_path}")
    
    # Get optimized lambda parameters and edges for ACTIVE VPs only.
    # indicator_functions reflects active topology, so lambda/edge arrays must match.
    active_lambdas = []
    active_edges = []
    for vp in partition.variable_points:
        if getattr(vp, 'active', True):
            active_lambdas.append(vp.lambda_param)
            active_edges.append(vp.edge)

    lambda_opt = np.array(active_lambdas)
    vp_edges_arr = np.array(active_edges, dtype=np.int64)
    
    # Get indicator functions (updated after migrations)
    indicator_functions = partition.indicator_functions
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Save lambda parameters (active VPs only)
        f.create_dataset('lambda_parameters', data=lambda_opt)
        
        # Save VP edge associations (active VPs only)
        f.create_dataset('vp_edges', data=vp_edges_arr)
        
        # Save indicator functions (critical for visualization after migrations)
        f.create_dataset('indicator_functions', data=indicator_functions)
        
        # Save metadata
        f.attrs['n_variable_points'] = len(lambda_opt)
        f.attrs['n_cells'] = partition.n_cells
        f.attrs['final_perimeter'] = opt_info['final_perimeter']
        f.attrs['optimization_success'] = opt_info['success']
        f.attrs['optimization_iterations'] = opt_info['n_iterations']
        f.attrs['iteration_number'] = iteration_number
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save optimization info
        opt_grp = f.create_group('optimization_info')
        opt_grp.attrs['initial_perimeter'] = opt_info['initial_perimeter']
        opt_grp.attrs['final_perimeter'] = opt_info['final_perimeter']
        opt_grp.attrs['perimeter_reduction'] = opt_info['perimeter_reduction']
        opt_grp.attrs['percent_reduction'] = opt_info['percent_reduction']
        opt_grp.create_dataset('constraint_violations', data=opt_info['final_constraint_violations'])
        
        # Save Type 2 migration history (NEW!)
        # NOTE: Currently saves EVERY iteration for testing/debugging
        #       See "Future Optimization" section in TYPE2_REVERSE_MIGRATION_IMPLEMENTATION_PLAN.md
        if migration_history is not None and len(migration_history.records) > 0:
            save_type2_migration_history(f, migration_history)
            logger.info(f"✓ Saved migration history: {len(migration_history.records)} triple points tracked")
            for orig_tri, record in migration_history.records.items():
                logger.debug(f"  Triangle {orig_tri}: {record.triangle_sequence}")
    
    logger.info("✓ State exported successfully")
    logger.info(f"  {len(lambda_opt)} lambda parameters saved")
    logger.info(f"  Indicator functions saved: {indicator_functions.shape}")
    logger.info(f"  Perimeter: {opt_info['final_perimeter']:.10f}")
    logger.info(f"  Can be visualized with existing visualization scripts")
    logger.info("="*80)
    
    return output_path


def handle_type1_migration_failure(failure_info, iteration_file, boundary_tol, logger):
    """Handle Type 1 migration failure with detailed logging."""
    comp = failure_info['component']
    comp_idx = failure_info['component_idx']
    error = failure_info['error']
    
    logger.error("")
    logger.error("="*80)
    logger.error("MIGRATION FAILURE - TYPE 1")
    logger.error("="*80)
    logger.error(f"Component {comp_idx} migration FAILED")
    logger.error("")
    logger.error("Component details:")
    logger.error(f"  VPs involved: {comp['vp_indices']}")
    logger.error(f"  Target vertex: {comp['target_vertex']}")
    logger.error(f"  Cells involved: {comp['cell_indices']}")
    logger.error(f"  Distance to target: {comp['distance']:.6f}")
    logger.error(f"  Component size: {len(comp['vp_indices'])} VPs")
    logger.error("")
    logger.error(f"Error: {error}")
    logger.error("")
    logger.error("="*80)
    logger.error("DIAGNOSTIC RECOMMENDATION")
    logger.error("="*80)
    logger.error("Visualize this component to investigate the failure:")
    logger.error("")
    logger.error(f"  python examples/visualize_type1_vertex_collapse.py \\")
    logger.error(f"    --solution {iteration_file} \\")
    logger.error(f"    --component-index {comp_idx} \\")
    logger.error(f"    --boundary-tol {boundary_tol} \\")
    logger.error(f"    --state before \\")
    logger.error(f"    --show-vps \\")
    logger.error(f"    --show-steiner")
    logger.error("")
    logger.error(f"State before failure saved to: {iteration_file}")
    logger.error("="*80)


def handle_type2_migration_failure(failure_info, iteration_file, boundary_tol, logger):
    """Handle Type 2 migration failure with detailed logging."""
    tp = failure_info['triple_point']
    tp_idx = failure_info['tp_idx']
    error = failure_info['error']
    
    logger.error("")
    logger.error("="*80)
    logger.error("MIGRATION FAILURE - TYPE 2")
    logger.error("="*80)
    logger.error(f"Triple point {tp_idx} migration FAILED")
    logger.error("")
    logger.error("Triple point details:")
    logger.error(f"  Triangle: {tp.triangle_idx}")
    logger.error(f"  VPs involved: {tp.var_point_indices}")
    logger.error(f"  Cells involved: {tp.cell_indices}")
    logger.error("")
    logger.error(f"Error: {error}")
    logger.error("")
    logger.error("="*80)
    logger.error("DIAGNOSTIC RECOMMENDATION")
    logger.error("="*80)
    logger.error("Visualize this triple point to investigate the failure:")
    logger.error("")
    logger.error(f"  python examples/visualize_type2_triple_point.py \\")
    logger.error(f"    --solution {iteration_file} \\")
    logger.error(f"    --triple-point-index {tp_idx} \\")
    logger.error(f"    --state before \\")
    logger.error(f"    --show-vps \\")
    logger.error(f"    --show-steiner")
    logger.error("")
    logger.error(f"State before failure saved to: {iteration_file}")
    logger.error("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Iterative perimeter refinement with automatic topology migrations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic refinement with default settings
  python testing/refine_perimeter_iterative.py \\
      --solution results/run_xyz/solution_level0.h5 \\
      --max-iterations 10
  
  # Custom settings
  python testing/refine_perimeter_iterative.py \\
      --solution results/run_xyz/solution_level0.h5 \\
      --max-iterations 20 \\
      --tolerance 1e-8 \\
      --boundary-tol 0.005 \\
      --distance-preservation midpoint \\
      --method trust-constr
        """
    )
    
    # Required arguments
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input solution .h5 file from PGD/SLSQP optimization')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output refined contours .h5 file (default: auto-generated)')
    
    parser.add_argument('--max-iterations', type=int, required=True,
                       help='Maximum number of topology iteration cycles (REQUIRED)')
    
    parser.add_argument('--max-opt-iter', type=int, default=1000,
                       help='Maximum optimization iterations per topology (default: 1000)')
    
    parser.add_argument('--tolerance', type=float, default=1e-7,
                       help='Optimization convergence tolerance (default: 1e-7)')
    
    parser.add_argument('--boundary-tol', type=float, default=1e-3,
                       help='Threshold for boundary point detection (default: 1e-3)')
    
    parser.add_argument('--distance-preservation', type=str, default='preserve',
                       help='VP placement strategy after Type 1 migration: '
                            '"preserve" (default, maintains distance to target), '
                            '"midpoint" (places at edge midpoint), '
                            'or a float between 0.0-1.0 (custom distance)')
    
    parser.add_argument('--method', type=str, default='SLSQP',
                       choices=['SLSQP', 'trust-constr'],
                       help='Optimization method (default: SLSQP)')
    
    parser.add_argument('--use-legacy', action='store_true',
                       help='Use legacy TopologySwitcher')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging (verbose output with detailed diagnostics)')
    
    parser.add_argument('--log-level', type=str, default=None,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO, or DEBUG if --debug flag is used). '
                            'Note: --debug flag overrides this setting.')
    
    args = parser.parse_args()
    
    # Validate distance_preservation
    distance_preservation = args.distance_preservation
    if distance_preservation not in ['preserve', 'midpoint']:
        try:
            dist_value = float(distance_preservation)
            if dist_value < 0.0 or dist_value > 1.0:
                print(f"ERROR: Custom distance must be between 0.0 and 1.0, got {dist_value}")
                return 1
        except ValueError:
            print(f"ERROR: Invalid distance_preservation: '{distance_preservation}'")
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
    logger.info("Iterative Perimeter Refinement with Automatic Migrations")
    logger.info("="*80)
    logger.info(f"Migration path: {'legacy (TopologySwitcher)' if args.use_legacy else 'MigrationOrchestrator'}")
    logger.info(f"Input solution: {args.solution}")
    logger.info(f"Maximum iterations: {args.max_iterations}")
    logger.info(f"Distance preservation: {args.distance_preservation}")
    
    # Check input file exists
    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1
    
    # Determine output paths
    if args.output is None:
        base_path = args.solution.replace('.h5', '')
        args.output = f"{base_path}_refined_contours.h5"
    
    # Base output path for iteration files (without extension)
    base_output = args.output.replace('_refined_contours.h5', '').replace('.h5', '')
    
    logger.info(f"Output file: {args.output}")
    logger.info(f"Iteration files: {base_output}_btol{args.boundary_tol}_iteration{{N}}_refined_contours.h5")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Load solution and extract initial contours
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("STEP 1: Loading solution and extracting contours")
    logger.info("="*80)
    
    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    
    n_vertices = analyzer.vertices.shape[0]
    n_partitions = analyzer.densities.shape[1]
    mesh_dim = analyzer.vertices.shape[1]
    
    logger.info(f"Mesh: {n_vertices} vertices, {n_partitions} partitions, dimension={mesh_dim}")
    
    # Compute indicator functions
    indicators = analyzer.compute_indicator_functions()
    
    # Extract contours with topology information
    logger.info("Extracting contours with boundary topology information...")
    raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
    logger.info(f"Extracted topology from {sum(len(v) for v in boundary_topology.values())} boundary triangles")
    
    # -------------------------------------------------------------------------
    # Step 2: Build mesh and partition contour data structures
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("STEP 2: Building data structures")
    logger.info("="*80)
    
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    total_area = float(mesh.M.sum())
    target_area = total_area / n_partitions
    
    logger.info(f"Total mesh area: {total_area:.6f}")
    logger.info(f"Target area per cell: {target_area:.6f}")
    
    partition = PartitionContour(mesh, indicators, boundary_topology=boundary_topology)
    
    # -------------------------------------------------------------------------
    # Step 3: Iterative refinement loop
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING ITERATIVE REFINEMENT")
    logger.info("="*80)
    logger.info(f"Maximum iterations: {args.max_iterations}")
    logger.info(f"Convergence tolerance: {args.tolerance}")
    logger.info(f"Boundary detection tolerance: {args.boundary_tol}")
    logger.info("="*80)
    
    # -------------------------------------------------------------------------
    # Initialize Type 2 Migration History (legacy path only)
    # -------------------------------------------------------------------------
    if args.use_legacy:
        from src.core.type2_migration_history import Type2MigrationHistory
        migration_history = Type2MigrationHistory()
        logger.info("Initialized empty Type 2 migration history")
    else:
        migration_history = None
    
    # -------------------------------------------------------------------------
    # Main iteration loop
    # -------------------------------------------------------------------------
    
    global_start_time = time.time()
    converged = False
    topology_iteration = 0
    total_type1_migrations = 0
    total_type2_migrations = 0
    iteration_files = []
    
    while not converged and topology_iteration < args.max_iterations:
        iteration_number = topology_iteration + 1
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"ITERATION {iteration_number}/{args.max_iterations}")
        logger.info("="*80)
        
        # ---------------------------------------------------------------------
        # PHASE 1: Optimize Current Topology
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info(f"Phase 1: Optimizing current topology...")
        logger.info("-"*80)
        
        # Create fresh optimizer each iteration (VP count may change after migrations)
        optimizer = PerimeterOptimizer(partition, mesh, target_area)
        
        # Calculate initial perimeter
        x0 = partition.get_variable_vector()
        iter_initial_perimeter = optimizer.objective(x0)
        logger.info(f"Perimeter at start of iteration: {iter_initial_perimeter:.10f}")
        
        # Run optimization (automatically logs every 10 iterations)
        opt_start_time = time.time()
        result = optimizer.optimize(
            max_iter=args.max_opt_iter,
            tol=args.tolerance,
            method=args.method
        )
        opt_elapsed = time.time() - opt_start_time
        
        # Get optimization info
        opt_info = optimizer.get_optimization_info(result)
        
        # Calculate metrics
        final_perimeter = opt_info['final_perimeter']
        perimeter_reduction = iter_initial_perimeter - final_perimeter
        percent_reduction = (perimeter_reduction / iter_initial_perimeter * 100) if iter_initial_perimeter > 0 else 0.0
        final_areas = np.array(opt_info['final_areas'])
        constraint_violations = final_areas - target_area
        
        # Add to opt_info for export
        opt_info['initial_perimeter'] = float(iter_initial_perimeter)
        opt_info['perimeter_reduction'] = float(perimeter_reduction)
        opt_info['percent_reduction'] = float(percent_reduction)
        opt_info['final_constraint_violations'] = constraint_violations.tolist()
        
        logger.info(f"\nOptimization completed in {opt_elapsed:.2f}s")
        logger.info(f"  Success: {opt_info['success']}")
        logger.info(f"  Iterations: {opt_info['n_iterations']}")
        logger.info(f"  Final perimeter: {final_perimeter:.10f}")
        logger.info(f"  Improvement: {perimeter_reduction:.10f} ({percent_reduction:.4f}%)")
        logger.info(f"  Max area violation: {np.max(np.abs(constraint_violations)):.2e}")
        
        if not opt_info['success']:
            logger.error("\nOptimization failed to converge!")
            logger.error(f"Message: {opt_info['message']}")
            logger.error("Exporting current state and stopping")
            
            # Export current state
            failure_file = export_intermediate_state(
                partition, iteration_number, base_output, opt_info, logger, migration_history, args.boundary_tol
            )
            logger.error(f"State saved to: {failure_file}")
            return 1
        
        # ---------------------------------------------------------------------
        # PHASE 2: Export Pre-Switch State
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info(f"Phase 2: Exporting pre-switch state...")
        logger.info("-"*80)
        
        iteration_file = export_intermediate_state(
            partition, iteration_number, base_output, opt_info, logger, migration_history, args.boundary_tol
        )
        iteration_files.append(iteration_file)
        
        # ---------------------------------------------------------------------
        # PHASE 3: Detect Topology Switches
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info(f"Phase 3: Detecting topology switches...")
        logger.info("-"*80)
        
        if args.use_legacy:
            switches_needed, switch_info = optimizer.check_topology_switches_needed(
                tol=args.boundary_tol
            )
            
            if not switches_needed:
                logger.info("")
                logger.info("="*80)
                logger.info("CONVERGENCE ACHIEVED")
                logger.info("="*80)
                logger.info("No topology switches needed")
                logger.info(f"Final perimeter: {final_perimeter:.10f}")
                converged = True
                break
            
            # Extract candidates manually
            type1_candidates = []
            for vp_idx, vp in enumerate(partition.variable_points):
                if vp.on_boundary(tol=args.boundary_tol):
                    type1_candidates.append(vp_idx)
            
            type2_candidates = optimizer.steiner_handler.get_boundary_triple_points(
                tol=args.boundary_tol
            )
            
            logger.info(f"Switches detected:")
            logger.info(f"  Type 1 candidates (boundary VPs): {len(type1_candidates)}")
            logger.info(f"  Type 2 candidates (boundary triple points): {len(type2_candidates)}")
        else:
            # New MigrationOrchestrator API
            mesh_topology = MeshTopology(mesh)
            orchestrator = MigrationOrchestrator(
                partition, mesh, mesh_topology,
                MigrationConfig(delta=args.boundary_tol)
            )
            detection = orchestrator.detect_all_triggers(delta=args.boundary_tol)
            
            if not detection.type1_triggers and not detection.type2_triggers:
                logger.info("")
                logger.info("="*80)
                logger.info("CONVERGENCE ACHIEVED")
                logger.info("="*80)
                logger.info("No topology switches needed")
                logger.info(f"Final perimeter: {final_perimeter:.10f}")
                converged = True
                break
            
            logger.info(f"Switches detected: {len(detection.type1_triggers)} Type 1, {len(detection.type2_triggers)} Type 2")
        
        # ---------------------------------------------------------------------
        # PHASE 4: Apply Migrations (Type 2 first, then Type 1)
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info(f"Phase 4: Applying migrations...")
        logger.info("-"*80)
        
        iteration_migrations = 0
        
        if args.use_legacy:
            type1_result = {'migrations_applied': 0}
            type2_result = {'migrations_applied': 0}
            # Type 2 migrations first
            if len(type2_candidates) > 0:
                type2_result = apply_type2_migrations(
                    partition, mesh, type2_candidates, logger,
                    steiner_handler=optimizer.steiner_handler,
                    distance_preservation=args.distance_preservation,
                    migration_history=migration_history,
                    iteration_number=iteration_number
                )
                
                # Check for failure
                if type2_result.get('failed'):
                    handle_type2_migration_failure(
                        type2_result, iteration_file, args.boundary_tol, logger
                    )
                    return 1
                
                iteration_migrations += type2_result['migrations_applied']
                total_type2_migrations += type2_result['migrations_applied']
            
            # Type 1 migrations second
            if len(type1_candidates) > 0:
                type1_result = apply_type1_migrations(
                    partition, mesh, type1_candidates, logger,
                    boundary_tol=args.boundary_tol,
                    distance_preservation=args.distance_preservation,
                    steiner_handler=optimizer.steiner_handler
                )
                
                # Check for failure
                if type1_result.get('failed'):
                    handle_type1_migration_failure(
                        type1_result, iteration_file, args.boundary_tol, logger
                    )
                    return 1
                
                iteration_migrations += type1_result['migrations_applied']
                total_type1_migrations += type1_result['migrations_applied']
            
            _t1_count = type1_result.get('migrations_applied', 0)
            _t2_count = type2_result.get('migrations_applied', 0)
        else:
            # New MigrationOrchestrator API
            mig_result = orchestrator.execute_migrations(mode='batch')
            
            if mig_result.failed:
                logger.error("")
                logger.error("="*80)
                logger.error("MIGRATION FAILURE")
                logger.error("="*80)
                logger.error(f"Error: {mig_result.error_message}")
                logger.error(f"State saved to: {iteration_file}")
                logger.error("="*80)
                return 1
            
            iteration_migrations = mig_result.type1_applied + mig_result.type2_forward_applied + mig_result.type2_rollbacks_applied
            total_type1_migrations += mig_result.type1_applied
            total_type2_migrations += mig_result.type2_forward_applied + mig_result.type2_rollbacks_applied
            
            _t1_count = mig_result.type1_applied
            _t2_count = mig_result.type2_forward_applied + mig_result.type2_rollbacks_applied
        
        if iteration_migrations == 0:
            logger.warning("")
            logger.warning("="*80)
            logger.warning("NO MIGRATIONS APPLIED")
            logger.warning("="*80)
            logger.warning("Switches were detected but no migrations succeeded")
            logger.warning("Cannot make progress - stopping")
            logger.warning(f"State saved to: {iteration_file}")
            logger.warning("="*80)
            return 1
        
        logger.info(f"\nMigrations applied in iteration {iteration_number}: {iteration_migrations}")
        logger.info(f"  Type 1: {_t1_count}")
        logger.info(f"  Type 2: {_t2_count}")
        
        # ---------------------------------------------------------------------
        # PHASE 5: Prepare for Next Iteration
        # ---------------------------------------------------------------------
        logger.info("")
        logger.info(f"Phase 5: Preparing for next iteration...")
        logger.info("-"*80)
        
        active_vps = sum(1 for vp in partition.variable_points if vp.active)
        logger.info(f"  Total VPs: {len(partition.variable_points)} ({active_vps} active)")
        logger.info(f"  Optimizer will be rebuilt at start of next iteration")
        
        logger.info("")
        logger.info("="*80)
        logger.info(f"ITERATION {iteration_number} COMPLETE")
        logger.info("="*80)
        
        topology_iteration += 1
    
    # Check if max iterations reached
    if not converged:
        logger.warning("")
        logger.warning("="*80)
        logger.warning("MAX ITERATIONS REACHED")
        logger.warning("="*80)
        logger.warning(f"Stopped after {args.max_iterations} iterations")
        logger.warning("Partition has not fully converged")
        logger.warning("Consider increasing --max-iterations if further refinement needed")
        logger.warning("="*80)
    
    global_elapsed = time.time() - global_start_time
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    final_file = iteration_files[-1] if iteration_files else None
    
    logger.info("")
    logger.info("="*80)
    logger.info("REFINEMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {global_elapsed:.2f}s")
    logger.info(f"Total iterations: {topology_iteration}")
    logger.info(f"Total migrations applied: {total_type1_migrations + total_type2_migrations}")
    logger.info(f"  Type 1: {total_type1_migrations}")
    logger.info(f"  Type 2: {total_type2_migrations}")
    logger.info(f"Final perimeter: {final_perimeter:.10f}")
    logger.info(f"Max area violation: {np.max(np.abs(constraint_violations)):.2e}")
    logger.info(f"Convergence: {'Yes' if converged else 'No (max iterations reached)'}")
    logger.info("")
    logger.info("Output files:")
    if iteration_files:
        for f in iteration_files:
            logger.info(f"  - {f}")
        logger.info(f"Final pre-migration state: {final_file}")
    logger.info("")
    if final_file:
        logger.info("To continue with migration + optimization:")
        logger.info(f"  python testing/test_migration_and_continue.py --solution {final_file} --migration-type both")
        logger.info("")
        logger.info("To visualize:")
        logger.info(f"  python examples/visualize_partition.py --solution {final_file}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
