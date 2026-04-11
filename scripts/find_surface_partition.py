#!/usr/bin/env python3
"""Γ-convergence relaxation — CLI wrapper.

Delegates all computation to src.pipeline.relaxation.run_relaxation().

Usage:
    python scripts/find_surface_partition.py --config parameters/torus_10part.yaml
    python scripts/find_surface_partition.py --config parameters/torus_10part.yaml --surface torus
    python scripts/find_surface_partition.py --config parameters/torus_10part.yaml --solution-dir /tmp/out
"""

import os
import sys
import shutil
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import setup_logging, get_logger
from src.pipeline.relaxation import run_relaxation, RelaxationConfig


def _resolve_surface_params(params, surface_name):
    """Extract surface parameters from a sectioned or flat YAML dict."""
    surface_section = params.get('surface', {})
    if isinstance(surface_section, dict) and surface_name in surface_section:
        return surface_section[surface_name]
    # Legacy flat format: surface params live at the top level
    return params


def main():
    parser = argparse.ArgumentParser(
        description='Generic surface partition optimization'
    )
    parser.add_argument('--config', type=str, dest='config',
                        help='Path to experiment YAML (sectioned format)')
    parser.add_argument('--input', type=str, dest='config',
                        help='(deprecated alias for --config)')
    parser.add_argument('--solution-dir', type=str,
                        help='Directory to save solutions')
    parser.add_argument('--surface', type=str, default=None,
                        help='Surface type (default: from YAML or "torus")')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to a prior solution HDF5 to warm-start from. '
                             'config.refinement_levels must be greater than the '
                             'completed_levels stored in that file.')
    args = parser.parse_args()

    setup_logging(log_level='INFO', log_to_console=True, log_to_file=False)
    logger = get_logger(__name__)

    params = {}
    config_path = None
    if args.config:
        config_path = os.path.abspath(args.config)
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f) or {}

    config = RelaxationConfig.from_yaml_dict(params) if params else RelaxationConfig()

    surface_name = args.surface
    if surface_name is None:
        surface_name = params.get('experiment', {}).get('surface', 'torus')

    if surface_name == 'torus':
        from src.surfaces.torus import TorusMeshProvider
        sp = _resolve_surface_params(params, 'torus')
        provider = TorusMeshProvider(
            int(sp.get('n_theta', 32)),
            int(sp.get('n_phi', 24)),
            float(sp.get('R', 1.0)),
            float(sp.get('r', 0.3)),
            n_theta_increment=int(sp.get('n_theta_increment', 0)),
            n_phi_increment=int(sp.get('n_phi_increment', 0)),
        )
    else:
        raise ValueError(f"Unsupported surface type: {surface_name}")

    result = run_relaxation(
        provider, config,
        output_dir=args.solution_dir,
        logger=logger,
        warm_start_path=args.resume_from,
    )

    if config_path is not None:
        dest = os.path.join(result.output_dir, 'experiment.yaml')
        shutil.copy2(config_path, dest)
        logger.info(f"Experiment config copied to: {dest}")

    logger.info(f"Solution file saved: {result.solution_path}")
    print(f"\nSurface partition optimization complete.")
    print(f"Results saved in: {result.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
