#!/usr/bin/env python3
"""
Perimeter refinement script for partition optimization.

This script implements the complete perimeter refinement workflow from Section 5
of "Partitions of Minimal Length on Manifolds" by Bogosel and Oudet.

Workflow:
1. Load relaxed solution from PGD/SLSQP optimization
2. Extract contours via indicator functions
3. Build PartitionContour data structure
4. Run constrained perimeter optimization
5. Check for topology switches
6. Iterate if needed (topology changes)
7. Save refined contours and results

Usage:
    python examples/refine_perimeter.py --solution results/run_xyz/solution_level0.h5 \\
        --output results/run_xyz/solution_level0_refined_contours.h5 \\
        --max-iterations 10 --tolerance 1e-7
"""

import os
import sys
import argparse
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from find_contours import ContourAnalyzer
from core.tri_mesh import TriMesh
from core.contour_partition import PartitionContour
from core.perimeter_optimizer import PerimeterOptimizer
from logging_config import get_logger, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description='Refine partition contours by minimizing perimeter with area constraints'
    )
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input solution .h5 file from PGD/SLSQP optimization')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output refined contours .h5 file (default: auto-generated)')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum number of topology switch iterations (default: 10)')
    parser.add_argument('--max-opt-iter', type=int, default=1000,
                       help='Maximum optimization iterations per topology (default: 1000)')
    parser.add_argument('--tolerance', type=float, default=1e-7,
                       help='Optimization convergence tolerance (default: 1e-7)')
    parser.add_argument('--boundary-tol', type=float, default=1e-3,
                       help='Threshold for boundary point detection (default: 1e-3)')
    parser.add_argument('--method', type=str, default='SLSQP', choices=['SLSQP', 'trust-constr'],
                       help='Optimization method (default: SLSQP)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--no-switch', action='store_true',
                       help='Disable topology switching (run optimization only, save pre-switch state)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("Perimeter Refinement for Partition Optimization")
    logger.info("="*80)
    logger.info(f"Input solution: {args.solution}")
    
    # Check input file exists
    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1
    
    # Determine output path
    if args.output is None:
        base_path = args.solution.replace('.h5', '')
        args.output = f"{base_path}_refined_contours.h5"
    
    logger.info(f"Output file: {args.output}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Load solution and extract initial contours
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Loading solution and extracting contours...")
    
    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    
    n_vertices = analyzer.vertices.shape[0]
    n_partitions = analyzer.densities.shape[1]
    mesh_dim = analyzer.vertices.shape[1]
    
    logger.info(f"Mesh: {n_vertices} vertices, {n_partitions} partitions, dimension={mesh_dim}")
    
    # Compute indicator functions
    indicators = analyzer.compute_indicator_functions()
    
    # NEW: Extract contours with topology information
    # This avoids redundant triangle scanning in PartitionContour initialization
    logger.info("Extracting contours with boundary topology information...")
    raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
    logger.info(f"Extracted topology from {sum(len(v) for v in boundary_topology.values())} boundary triangles")
    
    # -------------------------------------------------------------------------
    # Step 2: Build mesh and partition contour data structures
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Building data structures...")
    
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    total_area = float(mesh.M.sum())
    target_area = total_area / n_partitions

    # In case that the theoretical area of the manifold will be considered, it needs to be introduced here
    # total_area = provider.theoretical_total_area()

    logger.info(f"Total mesh area: {total_area:.6f}")
    logger.info(f"Target area per cell: {target_area:.6f}")
    
    # NEW: Pass boundary_topology to avoid redundant triangle scanning
    partition = PartitionContour(mesh, indicators, boundary_topology=boundary_topology)
    
    # -------------------------------------------------------------------------
    # Step 3: Iterative optimization with topology switches
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Running perimeter optimization...")
    
    global_start_time = time.time()
    converged = False
    topology_iteration = 0
    
    # Track best result across topology iterations
    best_result = None
    best_perimeter = float('inf')
    
    while not converged and topology_iteration < args.max_iterations:
        logger.info("")
        logger.info(f"--- Topology Iteration {topology_iteration + 1}/{args.max_iterations} ---")
        
        # Create optimizer for current topology
        optimizer = PerimeterOptimizer(partition, mesh, target_area)
        
        # Run optimization
        result = optimizer.optimize(
            max_iter=args.max_opt_iter,
            tol=args.tolerance,
            method=args.method
        )
        
        # Get optimization info
        opt_info = optimizer.get_optimization_info(result)
        final_perimeter = opt_info['final_perimeter']
        
        # Track best result
        if final_perimeter < best_perimeter:
            best_perimeter = final_perimeter
            best_result = result
            best_opt_info = opt_info
        
        # Check if topology switches are needed
        switches_needed, switch_info = optimizer.check_topology_switches_needed(
            tol=args.boundary_tol
        )
        
        if not switches_needed:
            logger.info("")
            logger.info("Convergence achieved: No topology switches needed")
            converged = True
        else:
            logger.info("")
            
            # Print detailed diagnostics for boundary triple points
            if switch_info['n_boundary_triple_points'] > 0:
                optimizer.diagnose_boundary_triple_points(tol=args.boundary_tol)
            
            # Check if switching is disabled
            if args.no_switch:
                logger.info("")
                logger.info("="*60)
                logger.info("TOPOLOGY SWITCHES DETECTED BUT --no-switch FLAG IS SET")
                logger.info(f"  Pure boundary VPs (Type 1 candidates): {switch_info['n_boundary_points']}")
                n_tp_boundary = switch_info.get('n_triple_point_boundary_vps', 0)
                if n_tp_boundary > 0:
                    logger.info(f"  Triple point boundary VPs (handle via Type 2): {n_tp_boundary}")
                logger.info(f"  Boundary triple points (Type 2 candidates): {switch_info['n_boundary_triple_points']}")
                logger.info(f"  Total boundary VPs: {switch_info.get('n_total_boundary_vps', switch_info['n_boundary_points'])}")
                logger.info("Saving pre-switch state and stopping.")
                logger.info("="*60)
                converged = True  # Stop iteration, save current state
            else:
                logger.info("Topology switches detected - applying switches")
                logger.info("")
                
                # Apply topology switches
                n_moves = optimizer.apply_topology_switches(switch_info, switch_tol=args.boundary_tol)
                
                if n_moves > 0:
                    # Re-initialize calculators after switches
                    optimizer.reinitialize_after_switches()
                    logger.info(f"Continuing to next topology iteration...")
                else:
                    # Couldn't apply switches - stop here
                    logger.warning("Could not apply topology switches - stopping")
                    converged = True
        
        topology_iteration += 1
    
    global_elapsed = time.time() - global_start_time
    
    # -------------------------------------------------------------------------
    # Step 4: Save refined contours
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 4: Saving refined contours...")
    
    # Ensure partition has best result
    partition.set_variable_vector(best_result.x)
    
    # Save to HDF5
    partition.save_refined_contours(
        output_path=args.output,
        perimeter=best_perimeter,
        areas=best_opt_info['final_areas'],
        optimization_info=best_opt_info
    )
    
    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("REFINEMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {global_elapsed:.2f}s")
    logger.info(f"Topology iterations: {topology_iteration}")
    logger.info(f"Final perimeter: {best_perimeter:.6f}")
    logger.info(f"Max area violation: {best_opt_info['max_area_violation']:.2e}")
    logger.info(f"Output saved to: {args.output}")
    logger.info("")
    logger.info("To visualize refined contours, run:")
    logger.info(f"  python examples/surface_visualization.py --solution {args.solution} --refined")
    logger.info("")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

