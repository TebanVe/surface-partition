#!/usr/bin/env python3
"""
Test script: Migration and Continue Optimization

This script tests the full workflow cycle starting from a refined partition:
1. Load refined contours from HDF5 file
2. Detect topology switches needed
3. Apply selected migrations (Type 1, Type 2, or both)
4. Rebuild calculators with updated partition
5. Run one optimization iteration
6. Export results in same format as input

This differs from refine_perimeter.py in that migrations happen FIRST,
then optimization continues. The refine_perimeter.py optimizes first
until switches are detected.

Usage:
    python testing/test_migration_and_continue.py \\
        --solution results/run_xyz/*_refined_contours.h5 \\
        --migration-type both \\
        --max-opt-iter 1000 \\
        --tolerance 1e-7

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

from examples.data_loader import load_partition_from_refined_file
from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.core.area_calculator import AreaCalculator
from src.core.perimeter_calculator import PerimeterCalculator
from src.core.steiner_handler import SteinerHandler
from src.core.perimeter_optimizer import PerimeterOptimizer
from src.core.mesh_topology import MeshTopology
from src.core.topology_switcher import TopologySwitcher
from src.core.type1_component_analyzer import Type1ComponentAnalyzer
from src.logging_config import get_logger, setup_logging


def compute_initial_diagnostics(mesh, partition, logger):
    """Compute and log initial state diagnostics."""
    logger.info("="*80)
    logger.info("INITIAL STATE DIAGNOSTICS")
    logger.info("="*80)
    
    # Mesh info
    logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
    
    # Partition info
    logger.info(f"Partition: {partition.n_cells} cells, {len(partition.variable_points)} VPs")
    
    # Triple points
    steiner_handler = SteinerHandler(mesh, partition)
    logger.info(f"Triple points: {len(steiner_handler.triple_points)}")
    
    # Compute current perimeter
    perim_calc = PerimeterCalculator(mesh, partition)
    lambda_vec = partition.get_variable_vector()
    partition.set_variable_vector(lambda_vec)
    
    total_perimeter = perim_calc.compute_total_perimeter(lambda_vec)
    steiner_perimeter = steiner_handler.get_total_perimeter_contribution()
    combined_perimeter = total_perimeter + steiner_perimeter
    
    logger.info(f"Current perimeter:")
    logger.info(f"  Regular: {total_perimeter:.10f}")
    logger.info(f"  Steiner: {steiner_perimeter:.10f}")
    logger.info(f"  Total:   {combined_perimeter:.10f}")
    
    # Compute current areas
    area_calc = AreaCalculator(mesh, partition)
    areas = area_calc.compute_all_cell_areas(lambda_vec)
    
    # Add Steiner contributions
    steiner_areas = steiner_handler.get_total_area_contribution()
    for cell_idx, area_contrib in steiner_areas.items():
        if cell_idx < len(areas):
            areas[cell_idx] += area_contrib
    
    total_area = np.sum(areas)
    logger.info(f"Current areas:")
    logger.info(f"  Total area: {total_area:.10f}")
    for i, area in enumerate(areas):
        logger.info(f"  Cell {i}: {area:.10f}")
    
    logger.info("="*80)
    
    return combined_perimeter, areas, steiner_handler


def detect_topology_switches(partition, mesh, steiner_handler, boundary_tol, logger):
    """Detect which topology switches are needed."""
    logger.info("")
    logger.info("="*80)
    logger.info("DETECTING TOPOLOGY SWITCHES")
    logger.info("="*80)
    
    # Create temporary optimizer just for switch detection
    target_area = 1.0  # Dummy value, not used for detection
    temp_optimizer = PerimeterOptimizer(partition, mesh, target_area)
    temp_optimizer.steiner_handler = steiner_handler  # Use existing steiner_handler
    
    switches_needed, switch_info = temp_optimizer.check_topology_switches_needed(tol=boundary_tol)
    
    logger.info(f"Switches detected: {switches_needed}")
    logger.info(f"  Type 1 candidates (boundary VPs): {switch_info['n_boundary_points']}")
    logger.info(f"  Type 2 candidates (boundary triple points): {switch_info['n_boundary_triple_points']}")
    
    # Get detailed lists
    type1_candidates = []
    type2_candidates = []
    
    if switches_needed:
        # Type 1: Scan VPs for boundary conditions
        for vp_idx, vp in enumerate(partition.variable_points):
            if vp.on_boundary(tol=boundary_tol):
                type1_candidates.append(vp_idx)
        
        # Type 2: Get boundary triple points
        boundary_tps = steiner_handler.get_boundary_triple_points(tol=boundary_tol)
        type2_candidates = boundary_tps
        
        logger.info(f"\nType 1 VP indices: {type1_candidates}")
        logger.info(f"Type 2 triple point count: {len(type2_candidates)}")
    
    logger.info("="*80)
    
    return switches_needed, type1_candidates, type2_candidates


def apply_type1_migrations(partition, mesh, vp_candidates, logger, boundary_tol=0.01, distance_preservation='preserve'):
    """Apply Type 1 migrations for boundary VPs.
    
    Args:
        partition: PartitionContour object
        mesh: TriMesh object
        vp_candidates: List of candidate VP indices
        logger: Logger instance
        boundary_tol: Boundary tolerance (default: 0.01)
        distance_preservation: VP placement strategy after migration (default: 'preserve')
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
    analyzer = Type1ComponentAnalyzer(mesh, partition, mesh_topology)
    
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
        build_migration_plan=True  # Request pre-computed migration plan
    )
    
    # Restore logging level
    switcher_logger.setLevel(original_level)
    
    to_migrate = analysis_result['to_migrate']
    excluded = analysis_result['excluded']
    migration_plan = analysis_result.get('migration_plan', [])
    
    logger.info(f"\nComponents selected for migration: {len(to_migrate)}")
    logger.info(f"Components excluded: {len(excluded)}")
    if migration_plan:
        logger.info(f"Migration plan entries: {len(migration_plan)}")
    logger.info("="*80)
    
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
        
        try:
            # Use pre-computed migration details (efficient path)
            result = switcher.apply_type1_switch_v2(
                component=comp,
                distance_preservation=distance_preservation,
                migrating_vp=plan_entry['migrating_vp'],
                auxiliary_component=plan_entry['auxiliary_component'],
                left_neighbor=plan_entry['left_neighbor'],
                right_neighbor=plan_entry['right_neighbor']
            )
            
            if result['success']:
                # Brief success message
                logger.info(f"  ✓ Component {comp_idx}: Migration successful")
                migrations_applied += 1
            else:
                # Detailed failure message
                logger.warning(f"\n  ✗ Component {comp_idx}: Migration FAILED")
                logger.warning(f"    VPs: {comp['vp_indices']}")
                logger.warning(f"    Target vertex: {comp['target_vertex']}")
                logger.warning(f"    Error: {result.get('error', 'Unknown error')}")
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


def apply_type2_migrations(partition, mesh, tp_candidates, logger):
    """Apply Type 2 migrations for boundary triple points."""
    logger.info("")
    logger.info("="*80)
    logger.info(f"APPLYING TYPE 2 MIGRATIONS ({len(tp_candidates)} candidates)")
    logger.info("="*80)
    
    if len(tp_candidates) == 0:
        logger.info("No Type 2 migrations to apply")
        logger.info("="*80)
        return 0
    
    # Create topology objects
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
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
    for i, tp_idx in enumerate(tp_candidates):
        tp = steiner_handler.triple_points[tp_idx]
        
        try:
            result = switcher.apply_type2_switch_v4(
                steiner_handler=steiner_handler,
                triple_point_idx=tp_idx,
                distance_preservation='preserve'
            )
            
            if result['success']:
                # Brief success message
                logger.info(f"  ✓ Triple point {tp_idx} (triangle {tp.triangle_idx}): Migration successful")
                migrations_applied += 1
            else:
                # Detailed failure message
                logger.warning(f"\n  ✗ Triple point {tp_idx}: Migration FAILED")
                logger.warning(f"    Triangle: {tp.triangle_idx}")
                logger.warning(f"    VPs: {tp.var_point_indices}")
                logger.warning(f"    Cells: {tp.cell_indices}")
                logger.warning(f"    Error: {result.get('error', 'Unknown error')}")
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


def rebuild_calculators(mesh, partition, logger):
    """Rebuild calculators after migrations."""
    logger.info("")
    logger.info("="*80)
    logger.info("REBUILDING CALCULATORS")
    logger.info("="*80)
    
    area_calc = AreaCalculator(mesh, partition)
    perim_calc = PerimeterCalculator(mesh, partition)
    steiner_handler = SteinerHandler(mesh, partition)
    
    logger.info(f"✓ AreaCalculator rebuilt")
    logger.info(f"✓ PerimeterCalculator rebuilt")
    logger.info(f"✓ SteinerHandler rebuilt ({len(steiner_handler.triple_points)} triple points)")
    logger.info("="*80)
    
    return area_calc, perim_calc, steiner_handler


def run_optimization(partition, mesh, target_area, max_iter, tolerance, method, logger):
    """Run one optimization iteration."""
    logger.info("")
    logger.info("="*80)
    logger.info("RUNNING OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Target area: {target_area:.10f}")
    logger.info(f"Max iterations: {max_iter}")
    logger.info(f"Tolerance: {tolerance}")
    logger.info(f"Method: {method}")
    
    # Create optimizer
    optimizer = PerimeterOptimizer(partition, mesh, target_area)
    
    # Calculate initial perimeter before optimization
    x0 = partition.get_variable_vector()
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


def export_results(partition, opt_info, output_path, logger):
    """Export results to HDF5 file in same format as input."""
    logger.info("")
    logger.info("="*80)
    logger.info("EXPORTING RESULTS")
    logger.info("="*80)
    logger.info(f"Output file: {output_path}")
    
    # Get optimized lambda parameters
    lambda_opt = partition.get_variable_vector()
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Save lambda parameters (same format as input)
        f.create_dataset('lambda_parameters', data=lambda_opt)
        
        # Save metadata
        f.attrs['n_variable_points'] = len(lambda_opt)
        f.attrs['n_cells'] = partition.n_cells
        f.attrs['final_perimeter'] = opt_info['final_perimeter']
        f.attrs['optimization_success'] = opt_info['success']
        f.attrs['optimization_iterations'] = opt_info['n_iterations']
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save optimization info
        opt_grp = f.create_group('optimization_info')
        opt_grp.attrs['initial_perimeter'] = opt_info['initial_perimeter']
        opt_grp.attrs['final_perimeter'] = opt_info['final_perimeter']
        opt_grp.attrs['perimeter_reduction'] = opt_info['perimeter_reduction']
        opt_grp.attrs['percent_reduction'] = opt_info['percent_reduction']
        opt_grp.create_dataset('constraint_violations', data=opt_info['final_constraint_violations'])
    
    logger.info("✓ Results exported successfully")
    logger.info(f"  {len(lambda_opt)} lambda parameters saved")
    logger.info(f"  Metadata and optimization info included")
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
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--distance-preservation', type=str, default='preserve',
                       help='VP placement strategy after Type 1 migration: '
                            '"preserve" (default, maintains original distance to target vertex), '
                            '"midpoint" (places at edge midpoint, λ=0.5), '
                            'or a number between 0.0 and 1.0 (custom distance)')
    
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
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("TEST: Migration and Continue Optimization")
    logger.info("="*80)
    logger.info(f"Input file: {args.solution}")
    logger.info(f"Migration type: {args.migration_type}")
    logger.info(f"Distance preservation: {args.distance_preservation}")
    
    # Check input file exists
    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1
    
    # Determine output path (same directory as input if not specified)
    if args.output is None:
        base_path = args.solution.replace('_refined_contours.h5', '')
        args.output = f"{base_path}_iteration2_refined_contours.h5"
    
    logger.info(f"Output file: {args.output}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
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
    # Stage 2: Initial diagnostics
    # -------------------------------------------------------------------------
    initial_perimeter, initial_areas, steiner_handler = compute_initial_diagnostics(
        mesh, partition, logger
    )
    
    # Calculate target area (equal division)
    total_area = np.sum(initial_areas)
    target_area = total_area / partition.n_cells
    logger.info(f"Target area per cell: {target_area:.10f}")
    
    # -------------------------------------------------------------------------
    # Stage 3: Detect topology switches
    # -------------------------------------------------------------------------
    switches_needed, type1_candidates, type2_candidates = detect_topology_switches(
        partition, mesh, steiner_handler, args.boundary_tol, logger
    )
    
    if not switches_needed:
        logger.info("\nNo topology switches detected - aborting test")
        logger.info("This partition is already converged for perimeter refinement")
        return 0
    
    # -------------------------------------------------------------------------
    # Stage 4: Apply migrations based on type
    # -------------------------------------------------------------------------
    total_migrations = 0
    
    if args.migration_type in ['type1', 'both']:
        migrations = apply_type1_migrations(
            partition, mesh, type1_candidates, logger, 
            boundary_tol=args.boundary_tol,
            distance_preservation=args.distance_preservation
        )
        total_migrations += migrations
    
    if args.migration_type in ['type2', 'both']:
        migrations = apply_type2_migrations(partition, mesh, type2_candidates, logger)
        total_migrations += migrations
    
    if total_migrations == 0:
        logger.warning("\nNo migrations were applied - aborting test")
        logger.warning("Check boundary tolerance or migration candidates")
        return 1
    
    logger.info(f"\nTotal migrations applied: {total_migrations}")
    
    # -------------------------------------------------------------------------
    # Stage 5: Rebuild calculators
    # -------------------------------------------------------------------------
    area_calc, perim_calc, steiner_handler = rebuild_calculators(mesh, partition, logger)
    
    # -------------------------------------------------------------------------
    # Stage 6: Run optimization
    # -------------------------------------------------------------------------
    try:
        result, opt_info = run_optimization(
            partition, mesh, target_area,
            args.max_opt_iter, args.tolerance, args.method, logger
        )
    except Exception as e:
        logger.error(f"Optimization failed with exception: {str(e)}")
        logger.error("ABORTING TEST - Please debug the optimization failure")
        return 1
    
    if not opt_info['success']:
        logger.error("Optimization did not converge successfully")
        logger.error(f"Reason: {opt_info['message']}")
        logger.error("ABORTING TEST - Please review optimization parameters")
        return 1
    
    # -------------------------------------------------------------------------
    # Stage 7: Export results
    # -------------------------------------------------------------------------
    export_results(partition, opt_info, args.output, logger)
    
    # -------------------------------------------------------------------------
    # Stage 8: Post-optimization analysis
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("POST-OPTIMIZATION ANALYSIS")
    logger.info("="*80)
    
    # Check for NEW switches needed
    new_switches_needed, new_switch_info = PerimeterOptimizer(
        partition, mesh, target_area
    ).check_topology_switches_needed(tol=args.boundary_tol)
    
    logger.info(f"New switches needed: {new_switches_needed}")
    if new_switches_needed:
        logger.info(f"  New Type 1 candidates: {new_switch_info['n_boundary_points']}")
        logger.info(f"  New Type 2 candidates: {new_switch_info['n_boundary_triple_points']}")
    else:
        logger.info("  Partition has converged (no more switches needed)")
    
    logger.info("="*80)
    
    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("TEST COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Migrations applied: {total_migrations}")
    logger.info(f"Initial perimeter: {initial_perimeter:.10f}")
    logger.info(f"Final perimeter: {opt_info['final_perimeter']:.10f}")
    logger.info(f"Perimeter reduction: {initial_perimeter - opt_info['final_perimeter']:.10f}")
    logger.info(f"Output saved to: {args.output}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
