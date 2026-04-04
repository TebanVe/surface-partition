#!/usr/bin/env python3
"""
Unified Iterative Perimeter Refinement with Automatic Topology Migrations

Thin CLI wrapper that delegates to PipelineOrchestrator for all refinement logic.

Usage:
    # Fresh run from base solution
    python testing/refine_perimeter_iterative.py \
        --solution results/run_xyz/solution_level0.h5 \
        --max-iterations 10

    # Resume from iteration checkpoint (auto-detected)
    python testing/refine_perimeter_iterative.py \
        --solution results/run_xyz/solution_level0_btol0.01_iteration3_refined_contours.h5 \
        --max-iterations 10

    # Save all intermediate checkpoints (enables mid-run resume)
    python testing/refine_perimeter_iterative.py \
        --solution results/run_xyz/solution_level0.h5 \
        --max-iterations 20 --save-iterations
"""

import os
import sys
import argparse

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import get_logger, setup_logging
from src.pipeline.pipeline_orchestrator import (
    PipelineOrchestrator, RefinementConfig, detect_file_type, derive_output_paths,
)
from src.pipeline.io import load_partition_from_base_file, load_partition_from_refined_file


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

    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input HDF5 file (base solution or iteration checkpoint)')

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

    logger.info("=" * 80)
    logger.info("Unified Iterative Perimeter Refinement with Automatic Migrations")
    logger.info("=" * 80)
    logger.info(f"Input: {args.solution}")
    logger.info(f"Maximum iterations: {args.max_iterations}")
    logger.info(f"Distance preservation: {args.distance_preservation}")
    logger.info(f"Save iterations: {args.save_iterations}")

    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1

    config = RefinementConfig(
        max_iterations=args.max_iterations,
        max_opt_iter=args.max_opt_iter,
        tolerance=args.tolerance,
        boundary_tol=args.boundary_tol,
        method=args.method,
        lbfgs_memory=args.lbfgs_memory,
        best_iterate=args.best_iterate,
        exact_hessian=args.exact_hessian,
        allow_partial_convergence=args.allow_partial_convergence,
        use_vectorized=args.use_vectorized,
        save_iterations=args.save_iterations,
        distance_preservation=distance_preservation,
    )

    file_type = detect_file_type(args.solution)

    if file_type == 'checkpoint_converged':
        logger.info("Input file is a converged result (pending_migration=False). Nothing to do.")
        return 0

    _final_output, base_output = derive_output_paths(
        args.solution, file_type, output_override=args.output)

    if file_type == 'base':
        mesh, partition = load_partition_from_base_file(args.solution, verbose=True)
    else:
        mesh, partition = load_partition_from_refined_file(args.solution, verbose=True)

    orch = PipelineOrchestrator(mesh, partition, config, logger=logger)
    result = orch.run_refinement_loop(args.solution, base_output_path=base_output)

    if result.get('error'):
        logger.error(f"Refinement ended with error: {result['error']}")
        return 1

    final_file = result.get('iteration_files', [None])[-1] if result.get('iteration_files') else None
    if final_file:
        logger.info("")
        logger.info("To resume (if not converged):")
        logger.info(f"  python testing/refine_perimeter_iterative.py --solution {final_file} "
                     f"--max-iterations {args.max_iterations}")
        logger.info("")
        logger.info("To visualize:")
        logger.info(f"  python examples/visualize_partition.py --solution {final_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
