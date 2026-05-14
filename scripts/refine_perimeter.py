#!/usr/bin/env python3
"""
Iterative Perimeter Refinement with Automatic Topology Migrations

CLI wrapper that delegates to PipelineOrchestrator for all refinement logic.

Usage:
    # Fresh run from experiment config
    python scripts/refine_perimeter.py \
        --solution results/run_xyz/solution/surface_....h5 \
        --config parameters/torus_10part.yaml

    # CLI overrides (--max-iterations is required if not in --config)
    python scripts/refine_perimeter.py \
        --solution results/run_xyz/solution/surface_....h5 \
        --max-iterations 10 --method ipopt --exact-hessian

    # Resume from iteration checkpoint (auto-detected)
    python scripts/refine_perimeter.py \
        --solution results/run_xyz/refinement/ipopt_btol0.001/iteration_003_20260410_120523.h5 \
        --config parameters/torus_10part.yaml

    # Save all intermediate checkpoints (enables mid-run resume)
    python scripts/refine_perimeter.py \
        --solution results/run_xyz/solution/surface_....h5 \
        --config parameters/torus_10part.yaml --save-iterations
"""

import os
import sys
import argparse
import yaml

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import get_logger, setup_logging
from src.pipeline.pipeline_orchestrator import (
    PipelineOrchestrator, RefinementConfig, detect_file_type, derive_output_paths,
    build_campaign_name,
)
from src.pipeline.io import load_partition_from_base_file, load_partition_from_refined_file


def _build_config(args):
    """Build RefinementConfig from YAML + CLI overrides.

    YAML values are loaded first; any explicitly provided CLI flag
    takes precedence.
    """
    yaml_params = {}
    if args.config:
        with open(args.config, 'r') as f:
            yaml_params = yaml.safe_load(f) or {}

    if yaml_params:
        config = RefinementConfig.from_yaml_dict(yaml_params)
    else:
        config = RefinementConfig()

    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.max_opt_iter is not None:
        config.max_opt_iter = args.max_opt_iter
    if args.tolerance is not None:
        config.tolerance = args.tolerance
    if args.boundary_tol is not None:
        config.boundary_tol = args.boundary_tol
    if args.method is not None:
        config.method = args.method
    if args.lbfgs_memory is not None:
        config.lbfgs_memory = args.lbfgs_memory
    if args.distance_preservation is not None:
        config.distance_preservation = args.distance_preservation
    if args.use_vectorized is not None:
        config.use_vectorized = args.use_vectorized

    # Boolean flags: only override if explicitly passed on CLI
    if args.allow_partial_convergence:
        config.allow_partial_convergence = True
    if args.best_iterate:
        config.best_iterate = True
    if args.exact_hessian:
        config.exact_hessian = True
    if args.save_iterations:
        config.save_iterations = True
    if args.profile:
        config.profile = True

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Iterative perimeter refinement with automatic topology migrations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # From experiment config
  python scripts/refine_perimeter.py \\
      --solution results/run_xyz/solution/surface_....h5 \\
      --config parameters/torus_10part.yaml

  # CLI only (--max-iterations required)
  python scripts/refine_perimeter.py \\
      --solution results/run_xyz/solution/surface_....h5 \\
      --max-iterations 10 --method ipopt --exact-hessian

  # Save all intermediate checkpoints
  python scripts/refine_perimeter.py \\
      --solution results/run_xyz/solution/surface_....h5 \\
      --config parameters/torus_10part.yaml --save-iterations
        """
    )

    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input HDF5 file (base solution or iteration checkpoint)')

    parser.add_argument('--config', type=str, default=None,
                       help='Path to experiment YAML with a refinement: section. '
                            'CLI flags override YAML values.')

    parser.add_argument('--output', type=str, default=None,
                       help='Output directory override (default: auto-generated campaign dir)')

    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Maximum number of topology iteration cycles '
                            '(required if not set in --config)')

    parser.add_argument('--max-opt-iter', type=int, default=None,
                       help='Maximum optimization iterations per topology (default: 1000)')

    parser.add_argument('--tolerance', type=float, default=None,
                       help='Optimization convergence tolerance (default: 1e-7)')

    parser.add_argument('--boundary-tol', type=float, default=None,
                       help='Threshold for boundary point detection (default: 1e-3)')

    parser.add_argument('--distance-preservation', type=str, default=None,
                       help='VP placement strategy after Type 1 migration: '
                            '"preserve" (default), "midpoint", or a float 0.0-1.0')

    parser.add_argument('--method', type=str, default=None,
                       choices=['SLSQP', 'trust-constr', 'ipopt'],
                       help='Optimization method (default: SLSQP)')

    parser.add_argument('--lbfgs-memory', type=int, default=None,
                       help='L-BFGS history size for IPOPT (default: 6; try 20-50). '
                            'Ignored for SLSQP / trust-constr.')

    parser.add_argument('--allow-partial-convergence', action='store_true', default=False,
                       help='Continue to migration phase even if optimizer hits max_iter. '
                            'Ignored for SLSQP / trust-constr.')

    parser.add_argument('--best-iterate', action='store_true', default=False,
                       help='Track best feasible iterate during IPOPT. '
                            'Ignored for SLSQP / trust-constr.')

    parser.add_argument('--exact-hessian', action='store_true', default=False,
                       help='Use analytical Hessian instead of L-BFGS approximation. '
                            'Ignored for SLSQP / trust-constr.')

    parser.add_argument('--save-iterations', action='store_true', default=False,
                       help='Save checkpoint after each optimization step.')

    parser.add_argument('--profile', action='store_true', default=False,
                       help='Enable timing profiling. Writes timing_profile.yaml per campaign '
                            'with per-callback wall-clock times and Steiner recomputation counts. '
                            'Zero overhead when omitted.')

    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    parser.add_argument('--log-level', type=str, default=None,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO, or DEBUG if --debug)')

    vectorized_group = parser.add_mutually_exclusive_group()
    vectorized_group.add_argument(
        '--use-vectorized', action='store_true', default=None, dest='use_vectorized',
        help='Use vectorized evaluation path (default: enabled)')
    vectorized_group.add_argument(
        '--no-vectorized', action='store_false', dest='use_vectorized',
        help='Disable vectorized evaluation')

    args = parser.parse_args()

    # Validate distance_preservation if provided
    if args.distance_preservation is not None:
        dp = args.distance_preservation
        if dp not in ['preserve', 'midpoint']:
            try:
                dist_value = float(dp)
                if dist_value < 0.0 or dist_value > 1.0:
                    print(f"ERROR: Custom distance must be between 0.0 and 1.0, got {dist_value}")
                    return 1
            except ValueError:
                print(f"ERROR: Invalid distance_preservation: '{dp}'")
                print("Must be 'preserve', 'midpoint', or a number between 0.0 and 1.0")
                return 1

    config = _build_config(args)

    if config.max_iterations <= 0:
        print("ERROR: --max-iterations is required (via --config or CLI)")
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
    logger.info("Iterative Perimeter Refinement with Automatic Migrations")
    logger.info("=" * 80)
    logger.info(f"Input: {args.solution}")
    if args.config:
        logger.info(f"Config: {args.config}")
    logger.info(f"Method: {config.method}")
    logger.info(f"Maximum iterations: {config.max_iterations}")
    logger.info(f"Boundary tolerance: {config.boundary_tol}")
    logger.info(f"Distance preservation: {config.distance_preservation}")
    logger.info(f"Save iterations: {config.save_iterations}")

    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1

    file_type = detect_file_type(args.solution)

    if file_type == 'checkpoint_converged':
        logger.info("Input file is a converged result (pending_migration=False). Nothing to do.")
        return 0

    campaign_dir = derive_output_paths(
        args.solution, file_type, config=config, output_override=args.output)

    if campaign_dir:
        os.makedirs(campaign_dir, exist_ok=True)

    campaign_log_path = os.path.join(campaign_dir, 'refinement.log')
    import logging as _logging
    _root = _logging.getLogger()
    _fmt = _logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    _fh = _logging.FileHandler(campaign_log_path)
    _fh.setLevel(_logging.DEBUG)
    _fh.setFormatter(_fmt)
    _root.addHandler(_fh)
    logger.info(f"Logging to: {campaign_log_path}")

    if file_type == 'base':
        mesh, partition = load_partition_from_base_file(args.solution, verbose=True)
    else:
        mesh, partition = load_partition_from_refined_file(args.solution, verbose=True)

    orch = PipelineOrchestrator(mesh, partition, config, logger=logger)
    result = orch.run_refinement_loop(args.solution, output_dir=campaign_dir)

    if result.get('error'):
        logger.error(f"Refinement ended with error: {result['error']}")
        return 1

    final_file = result.get('iteration_files', [None])[-1] if result.get('iteration_files') else None
    if final_file:
        logger.info("")
        logger.info("To resume (if not converged):")
        resume_cmd = f"  python scripts/refine_perimeter.py --solution {final_file}"
        if args.config:
            resume_cmd += f" --config {args.config}"
        else:
            resume_cmd += f" --max-iterations {config.max_iterations}"
        logger.info(resume_cmd)
        logger.info("")
        logger.info("To visualize:")
        logger.info(f"  python scripts/visualize_partition.py --solution {final_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
