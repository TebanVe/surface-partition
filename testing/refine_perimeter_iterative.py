#!/usr/bin/env python3
"""
Unified Iterative Perimeter Refinement with Automatic Topology Migrations

This script implements the complete iterative refinement workflow:
1. Auto-detect input file type (base solution or iteration checkpoint)
2. Three-way entry point:
   - Base solution (x_opt present): start fresh -> Optimize first
   - Checkpoint with pending_migration=True: resume -> Migrate first
   - Checkpoint with pending_migration=False: already converged -> exit
3. ITERATIVE LOOP (Optimize -> Detect -> Export -> Migrate):
   a. Run constrained perimeter optimization
   b. Detect topology switches needed
   c. Export state with pending_migration flag
   d. If converged: exit
   e. Apply Type 2 migrations (triple points)
   f. Apply Type 1 migrations (boundary VPs)
   g. Loop

Key features:
- Dual file loading: base solution (x_opt) or iteration checkpoint (lambda_parameters)
- pending_migration boolean stored in every exported HDF5 for unambiguous resume
- --save-iterations flag (default off; only final converged file saved)
- Auto-detect starting iteration number from filename via regex
- Full diagnostic suite: VP consistency checks, perimeter roundtrip, drift detection
- MigrationOrchestrator handles Type 1/2 detection, execution, and rollback
- Exits immediately on first migration failure with detailed diagnostics

Usage:
    # Fresh run from base solution
    python testing/refine_perimeter_iterative.py \\
        --solution results/run_xyz/solution_level0.h5 \\
        --max-iterations 10

    # Resume from iteration checkpoint (auto-detected)
    python testing/refine_perimeter_iterative.py \\
        --solution results/run_xyz/solution_level0_btol0.01_iteration3_refined_contours.h5 \\
        --max-iterations 10

    # Save all intermediate checkpoints (enables mid-run resume)
    python testing/refine_perimeter_iterative.py \\
        --solution results/run_xyz/solution_level0.h5 \\
        --max-iterations 10 --save-iterations

Author: Perimeter Refinement Team
Date: February 2026
"""

import os
import sys
import argparse
import time
import re
import numpy as np
import h5py

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.data_loader import load_partition_from_refined_file
from src.partition.find_contours import ContourAnalyzer
from src.mesh.tri_mesh import TriMesh
from src.partition.contour_partition import PartitionContour
from src.optimization.perimeter_optimizer import PerimeterOptimizer
from src.mesh.mesh_topology import MeshTopology
from src.logging_config import get_logger, setup_logging

from src.migration.migration_orchestrator import MigrationOrchestrator, MigrationConfig


def detect_file_type(path):
    """Auto-detect whether an HDF5 file is a base solution or iteration checkpoint.

    Returns:
        'base' -- file contains x_opt, enter at Optimize
        'checkpoint_pending' -- iteration file with pending_migration=True, enter at Migrate
        'checkpoint_converged' -- iteration file with pending_migration=False, done
    """
    with h5py.File(path, 'r') as f:
        if 'x_opt' in f:
            return 'base'
        pending = bool(f.attrs.get('pending_migration', True))
        return 'checkpoint_pending' if pending else 'checkpoint_converged'


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
    from src.partition.find_contours import ContourAnalyzer

    logger.info("  Running indicator_functions <-> VP roundtrip consistency check...")

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

    live_edges = {vp.edge for vp in partition.variable_points if getattr(vp, 'active', True)}

    only_in_reconstructed = reconstructed_edges - live_edges
    only_in_live = live_edges - reconstructed_edges

    if only_in_reconstructed or only_in_live:
        logger.error("  CONSISTENCY CHECK FAILED: indicator_functions do not match live VPs")
        logger.error(f"    Live VPs      : {len(live_edges)} unique edges")
        logger.error(f"    Reconstructed : {len(reconstructed_edges)} unique edges from indicator_functions")
        logger.error(f"    Edges in indicator_functions but NOT in live VPs ({len(only_in_reconstructed)}): "
                     f"{sorted(only_in_reconstructed)[:10]}{'...' if len(only_in_reconstructed) > 10 else ''}")
        logger.error(f"    Edges in live VPs but NOT in indicator_functions ({len(only_in_live)}): "
                     f"{sorted(only_in_live)[:10]}{'...' if len(only_in_live) > 10 else ''}")
        logger.error("  *** This will cause a VP count mismatch when the file is reloaded! ***")
        return False
    else:
        logger.info(f"  Consistent: {len(live_edges)} boundary edges match in both live VPs and indicator_functions")
        return True


def export_intermediate_state(partition, iteration_number, base_output_path, opt_info, logger,
                              boundary_tol=None, pending_migration=True, mesh=None):
    """
    Export intermediate partition state to HDF5.

    Every exported file includes a pending_migration boolean attribute that
    makes the checkpoint self-describing for unambiguous resume.

    Args:
        partition: PartitionContour object
        iteration_number: Current iteration number (1, 2, 3, ...)
        base_output_path: Base path without extension
        opt_info: Dictionary with optimization results
        logger: Logger instance
        boundary_tol: Optional boundary tolerance value to include in filename
        pending_migration: Whether topology migration is pending (True) or converged (False)
        mesh: TriMesh object (enables consistency check and roundtrip verification)

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
    logger.info(f"pending_migration: {pending_migration}")

    # Pre-write consistency check
    if mesh is not None:
        _check_indicator_vp_consistency(partition, mesh, logger)

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
        f.attrs['pending_migration'] = bool(pending_migration)

        # Save optimization info
        opt_grp = f.create_group('optimization_info')
        opt_grp.attrs['initial_perimeter'] = opt_info['initial_perimeter']
        opt_grp.attrs['final_perimeter'] = opt_info['final_perimeter']
        opt_grp.attrs['perimeter_reduction'] = opt_info['perimeter_reduction']
        opt_grp.attrs['percent_reduction'] = opt_info['percent_reduction']
        opt_grp.create_dataset('constraint_violations', data=opt_info['final_constraint_violations'])

    logger.info("State exported successfully")
    logger.info(f"  {len(lambda_opt)} lambda parameters saved")
    logger.info(f"  Indicator functions saved: {indicator_functions.shape}")
    logger.info(f"  Perimeter: {opt_info['final_perimeter']:.10f}")
    logger.info(f"  pending_migration: {pending_migration}")

    # Post-write roundtrip perimeter check
    if mesh is not None:
        try:
            from src.partition.perimeter_calculator import PerimeterCalculator
            from src.partition.steiner_handler import SteinerHandler

            mesh_r, partition_r = load_partition_from_refined_file(output_path, verbose=False)

            lambda_r = partition_r.get_variable_vector()
            partition_r.set_variable_vector(lambda_r)

            perim_calc_r = PerimeterCalculator(mesh_r, partition_r)
            regular_perimeter_r = perim_calc_r.compute_total_perimeter(lambda_r)

            steiner_handler_r = SteinerHandler(mesh_r, partition_r)
            steiner_perimeter_r = steiner_handler_r.get_total_perimeter_contribution()

            perimeter_r = regular_perimeter_r + steiner_perimeter_r

            in_memory_perimeter = opt_info['final_perimeter']
            rel_diff = abs(perimeter_r - in_memory_perimeter) / max(in_memory_perimeter, 1e-12)

            if rel_diff < 1e-4:
                logger.info(f"  Roundtrip perimeter check PASSED: "
                          f"in-memory={in_memory_perimeter:.6f}, reloaded={perimeter_r:.6f} "
                          f"(rel_diff={rel_diff:.2e})")
                logger.info(f"    Regular={regular_perimeter_r:.6f}, Steiner={steiner_perimeter_r:.6f}")
            else:
                logger.warning(f"  Roundtrip perimeter check FAILED: "
                             f"in-memory={in_memory_perimeter:.6f}, reloaded={perimeter_r:.6f} "
                             f"(rel_diff={rel_diff:.2e})")
                logger.warning(f"    Regular={regular_perimeter_r:.6f}, Steiner={steiner_perimeter_r:.6f}")
                logger.warning(f"    VP count in-memory: {len(lambda_opt)}, reloaded: {len(lambda_r)}")
        except Exception as e:
            logger.warning(f"  Roundtrip perimeter check skipped: {e}")

    logger.info("="*80)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Unified iterative perimeter refinement with automatic topology migrations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Fresh run from base solution
  python testing/refine_perimeter_iterative.py \\
      --solution results/run_xyz/solution_level0.h5 \\
      --max-iterations 10

  # Resume from iteration checkpoint (auto-detected)
  python testing/refine_perimeter_iterative.py \\
      --solution results/run_xyz/solution_level0_btol0.01_iteration3_refined_contours.h5 \\
      --max-iterations 10

  # Save all intermediate checkpoints (enables mid-run resume)
  python testing/refine_perimeter_iterative.py \\
      --solution results/run_xyz/solution_level0.h5 \\
      --max-iterations 20 --save-iterations
        """
    )

    # Required arguments
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input HDF5 file (base solution or iteration checkpoint)')

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
                       choices=['SLSQP', 'trust-constr', 'ipopt'],
                       help='Optimization method (default: SLSQP)')

    parser.add_argument('--lbfgs-memory', type=int, default=6,
                       help='L-BFGS history size for IPOPT hessian approximation. '
                            'Higher values capture more curvature but use more memory. '
                            'IPOPT default is 6; try 20-50 for better refinement quality. '
                            'Ignored for SLSQP / trust-constr.')

    parser.add_argument('--allow-partial-convergence', action='store_true',
                       help='When the optimizer hits max_iter (or another non-fatal stopping '
                            'condition) without fully converging, treat the result as '
                            'acceptable and continue to the migration phase rather than '
                            'stopping. Fatal IPOPT statuses (infeasible, diverged, NaN, etc.) '
                            'still halt the run. Ignored for SLSQP / trust-constr.')

    parser.add_argument('--best-iterate', action='store_true',
                       help='Track the best feasible iterate during IPOPT optimization and '
                            'return it instead of the last iterate when the final point is '
                            'worse (e.g. due to restoration phase). Prevents perimeter '
                            'regressions between outer iterations. Ignored for SLSQP / '
                            'trust-constr.')

    parser.add_argument('--exact-hessian', action='store_true',
                       help='Provide IPOPT with an analytical Hessian of the '
                            'Lagrangian instead of the L-BFGS approximation. '
                            'Gives exact curvature for smoother boundaries. '
                            'Requires more computation per iteration. '
                            'Ignored for SLSQP / trust-constr.')

    parser.add_argument('--save-iterations', action='store_true',
                       help='Save an HDF5 checkpoint after each optimization step. '
                            'Enables mid-run resume. Default: off (only final converged result saved).')

    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging (verbose output with detailed diagnostics)')

    parser.add_argument('--log-level', type=str, default=None,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO, or DEBUG if --debug flag is used). '
                            'Note: --debug flag overrides this setting.')

    vectorized_group = parser.add_mutually_exclusive_group()
    vectorized_group.add_argument(
        '--use-vectorized', action='store_true', default=True, dest='use_vectorized',
        help='Use vectorized evaluation path (default: enabled)')
    vectorized_group.add_argument(
        '--no-vectorized', action='store_false', dest='use_vectorized',
        help='Disable vectorized evaluation; use original per-element calculators')

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
    logger.info("Unified Iterative Perimeter Refinement with Automatic Migrations")
    logger.info("="*80)
    logger.info(f"Input: {args.solution}")
    logger.info(f"Maximum iterations: {args.max_iterations}")
    logger.info(f"Distance preservation: {args.distance_preservation}")
    logger.info(f"Save iterations: {args.save_iterations}")

    # Check input file exists
    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1

    # -------------------------------------------------------------------------
    # Parse filename metadata
    # -------------------------------------------------------------------------
    iteration_match = re.search(r'_iteration(\d+)_refined_contours\.h5$', args.solution)
    btol_match = re.search(r'_btol([\d.]+(?:e-?\d+)?)_', args.solution)

    # -------------------------------------------------------------------------
    # Auto-detect file type and load
    # -------------------------------------------------------------------------
    file_type = detect_file_type(args.solution)
    logger.info(f"File type detected: {file_type}")

    if file_type == 'base':
        # Fresh run -- load via ContourAnalyzer
        logger.info("")
        logger.info("="*80)
        logger.info("Loading base solution via ContourAnalyzer")
        logger.info("="*80)

        analyzer = ContourAnalyzer(args.solution)
        analyzer.load_results(use_initial_condition=False)

        n_vertices = analyzer.vertices.shape[0]
        n_partitions = analyzer.densities.shape[1]
        mesh_dim = analyzer.vertices.shape[1]

        logger.info(f"Mesh: {n_vertices} vertices, {n_partitions} partitions, dimension={mesh_dim}")

        indicators = analyzer.compute_indicator_functions()

        logger.info("Extracting contours with boundary topology information...")
        raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
        logger.info(f"Extracted topology from {sum(len(v) for v in boundary_topology.values())} boundary triangles")

        mesh = TriMesh(analyzer.vertices, analyzer.faces)
        partition = PartitionContour(mesh, indicators, boundary_topology=boundary_topology)

        starting_iteration = 0
        enter_at = 'optimize'

    elif file_type == 'checkpoint_pending':
        # Resume -- migration was pending when this file was saved
        logger.info("")
        logger.info("="*80)
        logger.info("Loading iteration checkpoint (pending migration)")
        logger.info("="*80)

        try:
            mesh, partition = load_partition_from_refined_file(args.solution, verbose=True)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 1

        starting_iteration = int(iteration_match.group(1)) if iteration_match else 1
        enter_at = 'migrate'
        logger.info(f"Will resume from iteration {starting_iteration} (migrate first, then optimize)")

    elif file_type == 'checkpoint_converged':
        # Already converged -- nothing to do
        logger.info("Input file is a converged result (pending_migration=False). Nothing to do.")
        return 0

    # -------------------------------------------------------------------------
    # Determine output paths
    # -------------------------------------------------------------------------
    if args.output is None:
        if file_type == 'base':
            base_path = args.solution.replace('.h5', '')
            args.output = f"{base_path}_refined_contours.h5"
        else:
            # Checkpoint: strip iteration and btol to get base path
            base_path = args.solution
            if iteration_match:
                base_path = base_path.replace(
                    f'_iteration{starting_iteration}_refined_contours.h5', '')
            else:
                base_path = base_path.replace('_refined_contours.h5', '').replace('.h5', '')
            if btol_match:
                base_path = base_path.replace(f'_btol{btol_match.group(1)}', '')
            args.output = f"{base_path}_refined_contours.h5"

    # Base output path for iteration files (without extension)
    base_output = args.output.replace('_refined_contours.h5', '').replace('.h5', '')

    logger.info(f"Output base: {base_output}")
    logger.info(f"Iteration files: {base_output}_btol{args.boundary_tol}_iteration{{N}}_refined_contours.h5")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Compute target area
    # -------------------------------------------------------------------------
    total_area = float(mesh.M.sum())
    n_cells = partition.n_cells
    target_area = total_area / n_cells

    logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
    logger.info(f"Partition: {n_cells} cells, {len(partition.variable_points)} VPs")
    logger.info(f"Total mesh area: {total_area:.6f}")
    logger.info(f"Target area per cell: {target_area:.6f}")

    # -------------------------------------------------------------------------
    # Main iteration loop
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING ITERATIVE REFINEMENT")
    logger.info("="*80)
    logger.info(f"Entry point: {enter_at}")
    logger.info(f"Starting iteration: {starting_iteration}")
    logger.info(f"Maximum topology iterations: {args.max_iterations}")
    logger.info(f"Save iterations: {args.save_iterations}")
    logger.info("="*80)

    global_start_time = time.time()
    converged = False
    topology_iteration = 0
    total_type1_migrations = 0
    total_type2_migrations = 0
    iteration_files = []
    first_pass = True
    final_perimeter = None
    constraint_violations = None

    while not converged and topology_iteration < args.max_iterations:
        iteration_number = starting_iteration + topology_iteration + 1

        # -----------------------------------------------------------------
        # ENTRY POINT: MIGRATE (resume from checkpoint with pending migration)
        # The loaded state was optimized+detected but not yet migrated.
        # Re-detect switches and apply migration, then loop to optimize.
        # -----------------------------------------------------------------
        if first_pass and enter_at == 'migrate':
            first_pass = False
            logger.info("")
            logger.info("="*80)
            logger.info(f"RESUMING: Applying pending migration from iteration {starting_iteration}")
            logger.info("="*80)

            mesh_topology = MeshTopology(mesh)
            orchestrator = MigrationOrchestrator(
                partition, mesh, mesh_topology,
                MigrationConfig(delta=args.boundary_tol))
            detection = orchestrator.detect_all_triggers(delta=args.boundary_tol)

            if not detection.type1_triggers and not detection.type2_triggers:
                logger.warning("File had pending_migration=True but no switches found on resume")
                converged = True
                break

            logger.info(f"Re-detected: {len(detection.type1_triggers)} Type 1, "
                      f"{len(detection.type2_triggers)} Type 2")

            mig_result = orchestrator.execute_migrations(mode='batch')
            if mig_result.failed:
                logger.error(f"Migration failure on resume: {mig_result.error_message}")
                return 1
            total_type1_migrations += mig_result.type1_applied
            total_type2_migrations += (mig_result.type2_forward_applied +
                                      mig_result.type2_rollbacks_applied)

            _check_indicator_vp_consistency(partition, mesh, logger)
            logger.info("Pending migration applied. Continuing to optimization.")
            continue  # Don't increment topology_iteration; resume-migrate is not a new cycle

        first_pass = False

        logger.info("")
        logger.info("="*80)
        logger.info(f"ITERATION {iteration_number}/{starting_iteration + args.max_iterations}")
        logger.info("="*80)

        # -----------------------------------------------------------------
        # PHASE 1: Optimize Current Topology
        # -----------------------------------------------------------------
        logger.info("")
        logger.info(f"Phase 1: Optimizing current topology...")
        logger.info("-"*80)

        # Create fresh optimizer each iteration (VP count may change after migrations)
        optimizer = PerimeterOptimizer(partition, mesh, target_area,
                                       use_vectorized=args.use_vectorized)

        # Calculate initial perimeter
        x0 = partition.get_variable_vector()
        iter_initial_perimeter = optimizer.objective(x0)
        logger.info(f"Perimeter at start of iteration: {iter_initial_perimeter:.10f}")

        # Run optimization (automatically logs every 10 iterations)
        opt_start_time = time.time()
        result = optimizer.optimize(
            max_iter=args.max_opt_iter,
            tol=args.tolerance,
            method=args.method,
            lbfgs_memory=args.lbfgs_memory,
            best_iterate=args.best_iterate,
            exact_hessian=args.exact_hessian,
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
            # IPOPT status codes that are non-fatal stopping conditions:
            #   -1  Maximum_Iterations_Exceeded
            #   -4  Maximum_CpuTime_Exceeded
            #    5  User_Requested_Stop
            # All other non-zero statuses are genuine failures (infeasible,
            # diverged, NaN in callbacks, linear algebra breakdown, etc.).
            IPOPT_NONFATAL = {-1, -4, 5}
            ipopt_status = opt_info.get('status', None)
            is_nonfatal = (ipopt_status in IPOPT_NONFATAL)

            if args.allow_partial_convergence and is_nonfatal:
                logger.warning(f"\nOptimizer did not fully converge (status={ipopt_status}: "
                               f"{opt_info['message']})")
                logger.warning("--allow-partial-convergence is set: continuing to migration phase.")
            else:
                logger.error("\nOptimization failed to converge!")
                logger.error(f"Message: {opt_info['message']}")
                logger.error("Exporting current state and stopping")

                failure_file = export_intermediate_state(
                    partition, iteration_number, base_output, opt_info, logger,
                    boundary_tol=args.boundary_tol,
                    pending_migration=True, mesh=mesh
                )
                logger.info(f"State saved to: {failure_file}")
                return 1

        # -----------------------------------------------------------------
        # PHASE 2: Detect Topology Switches (before export so file knows its state)
        # -----------------------------------------------------------------
        logger.info("")
        logger.info(f"Phase 2: Detecting topology switches...")
        logger.info("-"*80)

        pending_migration = False

        mesh_topology = MeshTopology(mesh)
        orchestrator = MigrationOrchestrator(
            partition, mesh, mesh_topology,
            MigrationConfig(delta=args.boundary_tol)
        )
        detection = orchestrator.detect_all_triggers(delta=args.boundary_tol)

        if detection.type1_triggers or detection.type2_triggers:
            pending_migration = True
            logger.info(f"Switches detected: {len(detection.type1_triggers)} Type 1, "
                      f"{len(detection.type2_triggers)} Type 2")
        else:
            logger.info("No topology switches needed")

        # -----------------------------------------------------------------
        # PHASE 3: Export (with pending_migration flag)
        # -----------------------------------------------------------------
        is_last_iteration = (topology_iteration + 1 == args.max_iterations)
        should_export = args.save_iterations or (not pending_migration) or is_last_iteration

        iteration_file = None
        if should_export:
            logger.info("")
            logger.info(f"Phase 3: Exporting state (pending_migration={pending_migration})...")
            logger.info("-"*80)

            iteration_file = export_intermediate_state(
                partition, iteration_number, base_output, opt_info, logger,
                boundary_tol=args.boundary_tol,
                pending_migration=pending_migration, mesh=mesh
            )
            iteration_files.append(iteration_file)

        # -----------------------------------------------------------------
        # Check convergence
        # -----------------------------------------------------------------
        if not pending_migration:
            logger.info("")
            logger.info("="*80)
            logger.info("CONVERGENCE ACHIEVED")
            logger.info("="*80)
            logger.info("No topology switches needed")
            logger.info(f"Final perimeter: {final_perimeter:.10f}")
            converged = True
            break

        # -----------------------------------------------------------------
        # PHASE 4: Apply Migrations (Type 2 first, then Type 1)
        # -----------------------------------------------------------------
        logger.info("")
        logger.info(f"Phase 4: Applying migrations...")
        logger.info("-"*80)

        iteration_migrations = 0

        mig_result = orchestrator.execute_migrations(mode='batch')

        if mig_result.failed:
            logger.error("")
            logger.error("="*80)
            logger.error("MIGRATION FAILURE")
            logger.error("="*80)
            logger.error(f"Error: {mig_result.error_message}")
            diag_file = iteration_file if iteration_file else args.solution
            logger.error(f"State saved to: {diag_file}")
            logger.error("="*80)
            return 1

        iteration_migrations = (mig_result.type1_applied +
                               mig_result.type2_forward_applied +
                               mig_result.type2_rollbacks_applied)
        total_type1_migrations += mig_result.type1_applied
        total_type2_migrations += (mig_result.type2_forward_applied +
                                  mig_result.type2_rollbacks_applied)

        _t1_count = mig_result.type1_applied
        _t2_count = mig_result.type2_forward_applied + mig_result.type2_rollbacks_applied

        if iteration_migrations == 0:
            logger.warning("")
            logger.warning("="*80)
            logger.warning("NO MIGRATIONS APPLIED")
            logger.warning("="*80)
            logger.warning("Switches were detected but no migrations succeeded")
            logger.warning("Cannot make progress - stopping")
            diag_file = iteration_file if iteration_file else args.solution
            logger.warning(f"State saved to: {diag_file}")
            logger.warning("="*80)
            return 1

        logger.info(f"\nMigrations applied in iteration {iteration_number}: {iteration_migrations}")
        logger.info(f"  Type 1: {_t1_count}")
        logger.info(f"  Type 2: {_t2_count}")

        # Post-migration consistency check
        _check_indicator_vp_consistency(partition, mesh, logger)

        # -----------------------------------------------------------------
        # Prepare for next iteration
        # -----------------------------------------------------------------
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
        logger.warning(f"Stopped after {args.max_iterations} topology iterations")
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
    logger.info(f"Total topology iterations: {topology_iteration}")
    logger.info(f"Total migrations applied: {total_type1_migrations + total_type2_migrations}")
    logger.info(f"  Type 1: {total_type1_migrations}")
    logger.info(f"  Type 2: {total_type2_migrations}")
    if final_perimeter is not None:
        logger.info(f"Final perimeter: {final_perimeter:.10f}")
    if constraint_violations is not None:
        logger.info(f"Max area violation: {np.max(np.abs(constraint_violations)):.2e}")
    logger.info(f"Convergence: {'Yes' if converged else 'No (max iterations reached)'}")
    logger.info("")
    logger.info("Output files:")
    if iteration_files:
        for f in iteration_files:
            logger.info(f"  - {f}")
        logger.info(f"Final state: {final_file}")
    logger.info("")
    if final_file:
        logger.info("To resume (if not converged):")
        logger.info(f"  python testing/refine_perimeter_iterative.py --solution {final_file} "
                   f"--max-iterations {args.max_iterations}")
        logger.info("")
        logger.info("To visualize:")
        logger.info(f"  python examples/visualize_partition.py --solution {final_file}")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
