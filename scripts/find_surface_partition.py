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
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable Phase 1 timing profiling. Writes '
                             'solution/timing_profile.yaml with per-level '
                             'wall-clock breakdown by callback. '
                             'Zero overhead when omitted.')
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

    if args.profile:
        config.profile = True

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

    elif surface_name == 'ellipsoid':
        from src.surfaces.ellipsoid import EllipsoidMeshProvider
        sp = _resolve_surface_params(params, 'ellipsoid')
        provider = EllipsoidMeshProvider(
            int(sp.get('n_theta', 32)),
            int(sp.get('n_phi', 64)),
            float(sp.get('a', 1.0)),
            float(sp.get('b', 1.0)),
            float(sp.get('c', 0.7)),
            n_theta_increment=int(sp.get('n_theta_increment', 0)),
            n_phi_increment=int(sp.get('n_phi_increment', 0)),
        )

    elif surface_name == 'double_torus':
        from src.surfaces.double_torus import DoubleTorusMeshProvider
        sp = _resolve_surface_params(params, 'double_torus')
        provider = DoubleTorusMeshProvider(
            int(sp.get('n_grid_x', 100)),
            int(sp.get('n_grid_y', 100)),
            int(sp.get('n_grid_z', 100)),
            c=float(sp.get('c', 0.03)),
            n_grid_x_increment=int(sp.get('n_grid_x_increment', 0)),
            n_grid_y_increment=int(sp.get('n_grid_y_increment', 0)),
        )

    elif surface_name == 'banchoff_chmutov':
        from src.surfaces.banchoff_chmutov import BanchoffChmutovMeshProvider
        sp = _resolve_surface_params(params, 'banchoff_chmutov')
        provider = BanchoffChmutovMeshProvider(
            int(sp.get('n_grid_x', 100)),
            int(sp.get('n_grid_y', 100)),
            int(sp.get('n_grid_z', 100)),
            n_grid_x_increment=int(sp.get('n_grid_x_increment', 0)),
            n_grid_y_increment=int(sp.get('n_grid_y_increment', 0)),
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

    dc = result.dormant_cells or {}
    if dc.get('dead') or dc.get('weak'):
        print("\n" + "=" * 70)
        print("⚠️  WARNING: DORMANT CELLS DETECTED")
        print(f"   This is NOT a valid {result.n_partitions}-region partition, even though")
        print("   it is a consistent continuous solution (equal areas satisfied).")
        if dc.get('dead'):
            print(f"   Dead cells (vanished): {dc['dead']}  ->  "
                  f"{dc['n_effective']}/{dc['n_cells']} effective regions")
        if dc.get('weak'):
            print(f"   Weak cells (peak density < {dc['weak_threshold']}): {dc['weak']}")
        print("   Increase the initial mesh resolution (n_theta/n_phi) and re-run.")
        print("=" * 70)

    ai = result.area_imbalance or {}
    if ai.get('imbalanced'):
        print("\n" + "=" * 70)
        print("⚠️  WARNING: DISCRETE AREA IMBALANCE")
        print("   Phase 1's continuous equal areas are satisfied, but the discrete")
        print("   (winner-take-all) cell areas are not — so Phase 2 refinement will")
        print("   likely RAISE the perimeter and stall at local infeasibility.")
        print(f"   Worst cell {ai['worst_cell']}: {ai['worst_rel_dev'] * 100:.1f}% off "
              f"target (= Phase 2 iter-0 constraint violation {ai['worst_abs_dev']:.4g}).")
        print(f"   {ai['n_imbalanced']} cell(s) over {ai['rel_threshold'] * 100:.0f}%: "
              f"{ai['imbalanced']}")
        print("   A finer mesh does NOT reliably help; try other seeds and/or tune")
        print("   lambda_penalty. See docs/reference/winner_take_all_partition_gap.md")
        print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
