#!/usr/bin/env python3
"""Γ-convergence relaxation — CLI wrapper.

Delegates all computation to src.pipeline.relaxation.run_relaxation().

Usage:
    python scripts/find_surface_partition.py --input parameters/input.yaml --surface torus
    python scripts/find_surface_partition.py --input parameters/input.yaml --solution-dir /tmp/out
"""

import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import setup_logging, get_logger
from src.pipeline.relaxation import run_relaxation, RelaxationConfig


def main():
    parser = argparse.ArgumentParser(
        description='Generic surface partition optimization'
    )
    parser.add_argument('--input', type=str, help='Path to input YAML')
    parser.add_argument('--solution-dir', type=str,
                        help='Directory to save solutions')
    parser.add_argument('--surface', type=str, default='torus',
                        help='Surface type (torus)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to a prior solution HDF5 to warm-start from. '
                             'config.refinement_levels must be greater than the '
                             'completed_levels stored in that file.')
    args = parser.parse_args()

    setup_logging(log_level='INFO', log_to_console=True, log_to_file=False)
    logger = get_logger(__name__)

    params = {}
    if args.input:
        with open(args.input, 'r') as f:
            params = yaml.safe_load(f) or {}

    config = RelaxationConfig.from_yaml_dict(params) if params else RelaxationConfig()

    if args.surface == 'torus':
        from src.surfaces.torus import TorusMeshProvider
        n_theta = int(params.get('n_theta', 32))
        n_phi = int(params.get('n_phi', 24))
        R = float(params.get('R', 1.0))
        r = float(params.get('r', 0.3))
        n_theta_increment = int(params.get('n_theta_increment', 0))
        n_phi_increment = int(params.get('n_phi_increment', 0))
        provider = TorusMeshProvider(
            n_theta, n_phi, R, r,
            n_theta_increment=n_theta_increment,
            n_phi_increment=n_phi_increment,
        )
    else:
        raise ValueError(f"Unsupported surface type: {args.surface}")

    result = run_relaxation(
        provider, config,
        output_dir=args.solution_dir,
        logger=logger,
        warm_start_path=args.resume_from,
    )

    logger.info(f"Solution file saved: {result.solution_path}")
    print(f"\nSurface partition optimization complete.")
    print(f"Results saved in: {result.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
