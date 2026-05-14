#!/usr/bin/env python3
"""
Timing analyzer for Phase 2 perimeter refinement profiling data.

Reads experiment_index.yaml and produces scaling figures from
timing_profile.yaml data collected by --profile runs.

Usage:
    python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/
    python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/ \
        --campaign ipopt_btol0.001_lbfgs30_hess
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

CB_COLORS = {
    'hessian':     '#d62728',
    'jacobian':    '#ff7f0e',
    'gradient':    '#2ca02c',
    'objective':   '#1f77b4',
    'constraints': '#9467bd',
    'overhead':    '#8c564b',
}


def _load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def _load_runs_with_timing(experiment_dir: str, campaign: str = None) -> list:
    """Return list of run-entry dicts that have timing data."""
    index_path = os.path.join(experiment_dir, "experiment_index.yaml")
    if not os.path.isfile(index_path):
        logger.error(f"No experiment_index.yaml found in {experiment_dir}")
        return []

    index = _load_yaml(index_path)
    runs = index.get("runs", [])

    timed = []
    for run in runs:
        if "timing_campaigns" not in run:
            continue
        tc = run["timing_campaigns"]
        if campaign:
            if campaign not in tc:
                continue
            run = dict(run)
            run["_selected_campaign"] = campaign
            run["_tdata"] = tc[campaign]
        else:
            # Use first available campaign
            first = next(iter(tc))
            run = dict(run)
            run["_selected_campaign"] = first
            run["_tdata"] = tc[first]
        timed.append(run)

    if not timed:
        logger.warning("No runs with timing data found. "
                       "Re-run refinement with --profile and then --mode collect.")
    return timed


def _power_law_fit(x, y):
    """Fit log(y) = a*log(x) + b via least squares. Returns (exponent, label_str)."""
    mask = (np.array(x) > 0) & (np.array(y) > 0)
    xm, ym = np.array(x)[mask], np.array(y)[mask]
    if len(xm) < 2:
        return None, None
    coeffs = np.polyfit(np.log(xm), np.log(ym), 1)
    alpha = coeffs[0]
    return alpha, f"O(N^{alpha:.2f})"


def plot_callback_time_vs_ncells(runs: list, out_dir: str):
    """4-subplot figure: total wall time per callback type vs n_cells."""
    callbacks = ['hessian', 'jacobian', 'gradient', 'objective']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Callback Wall Time vs Number of Cells', fontsize=14)

    for ax, cb in zip(axes.flat, callbacks):
        xs, ys, labels = [], [], []
        for run in runs:
            td = run["_tdata"]
            x = run.get("n_cells")
            y = td.get("callbacks", {}).get(cb, {}).get("total_wall_s")
            if x is not None and y is not None and y > 0:
                xs.append(x)
                ys.append(y)
                labels.append(str(x))

        if xs:
            ax.scatter(xs, ys, color=CB_COLORS[cb], zorder=3, s=60)
            for xi, yi, lab in zip(xs, ys, labels):
                ax.annotate(lab, (xi, yi), textcoords='offset points',
                            xytext=(4, 4), fontsize=8)
            alpha, label = _power_law_fit(xs, ys)
            if alpha is not None:
                x_fit = np.linspace(min(xs), max(xs), 100)
                coeffs = np.polyfit(np.log(xs), np.log(ys), 1)
                y_fit = np.exp(coeffs[1]) * x_fit ** coeffs[0]
                ax.plot(x_fit, y_fit, '--', color=CB_COLORS[cb],
                        alpha=0.7, label=label)
                ax.legend(fontsize=9)
            if len(set(xs)) > 1:
                ax.set_xscale('log')
            if len(set(ys)) > 1 and max(ys) / (min(ys) + 1e-15) > 10:
                ax.set_yscale('log')

        ax.set_xlabel('n_cells')
        ax.set_ylabel('Total wall time (s)')
        ax.set_title(cb.capitalize())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'callback_time_vs_ncells.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_steiner_recomps_vs_ntp(runs: list, out_dir: str):
    """Scatter: Steiner recomputations per Hessian call vs n_triple_points."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Steiner Recomputations per Hessian Call vs Triple Points', fontsize=12)

    xs, ys = [], []
    for run in runs:
        td = run["_tdata"]
        x = run.get("n_triple_points")
        y = (td.get("callbacks", {})
               .get("hessian", {})
               .get("steiner_recomps_per_call_mean"))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
            ax.annotate(str(run.get("n_cells", "")), (x, y),
                        textcoords='offset points', xytext=(4, 4), fontsize=8)

    if xs:
        ax.scatter(xs, ys, color=CB_COLORS['hessian'], zorder=3, s=70,
                   label='Observed')
        # O(n_tp²) reference line
        x_ref = np.linspace(max(1, min(xs) * 0.8), max(xs) * 1.2, 100)
        # Scale to pass through the first data point
        scale = ys[0] / (xs[0] ** 2 + 1e-15) if xs else 1.0
        ax.plot(x_ref, scale * x_ref ** 2, 'k--', alpha=0.5, label='O(n_tp²) ref')
        ax.legend(fontsize=9)
        if len(set(xs)) > 1:
            ax.set_xscale('log')
        if len(set(ys)) > 1 and max(ys) / (min(ys) + 1e-15) > 10:
            ax.set_yscale('log')

    ax.set_xlabel('n_triple_points')
    ax.set_ylabel('Steiner recomps per Hessian call')
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'steiner_recomps_vs_ntp.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_cost_breakdown(runs: list, out_dir: str):
    """Stacked horizontal bar: % wall time per callback type, one bar per run."""
    if not runs:
        return

    labels = []
    data = {cb: [] for cb in ('hessian', 'jacobian', 'gradient',
                               'objective', 'constraints', 'overhead')}

    for run in runs:
        td = run["_tdata"]
        n_cells = run.get("n_cells", "?")
        labels.append(f"n={n_cells}")
        summary = td.get("summary", {})
        total = summary.get("total_wall_s", 0) or 1.0
        for cb in ('hessian', 'jacobian', 'gradient', 'objective', 'constraints'):
            pct = summary.get(f"{cb}_pct_wall", 0.0) or 0.0
            data[cb].append(pct)
        data['overhead'].append(summary.get("overhead_pct_wall", 0.0) or 0.0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(runs) * 0.7 + 2)))
    ax.set_title('Cost Breakdown by Callback Type', fontsize=12)

    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for cb in ('hessian', 'jacobian', 'gradient', 'objective', 'constraints', 'overhead'):
        vals = np.array(data[cb])
        ax.barh(y, vals, left=left, color=CB_COLORS[cb],
                label=cb.capitalize(), height=0.6)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('% of total wall time')
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right', fontsize=8, ncol=3)
    ax.grid(True, axis='x', alpha=0.3)

    path = os.path.join(out_dir, 'cost_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_hessian_time_vs_nvps(runs: list, out_dir: str):
    """Scatter: mean Hessian callback time vs n_active_vps."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Mean Hessian Callback Time vs Active Variable Points', fontsize=12)

    xs, ys = [], []
    for run in runs:
        td = run["_tdata"]
        x = run.get("n_active_vps")
        y = td.get("callbacks", {}).get("hessian", {}).get("mean_wall_s")
        if x is not None and y is not None and y > 0:
            xs.append(x)
            ys.append(y)
            ax.annotate(str(run.get("n_cells", "")), (x, y),
                        textcoords='offset points', xytext=(4, 4), fontsize=8)

    if xs:
        ax.scatter(xs, ys, color=CB_COLORS['hessian'], zorder=3, s=70)
        alpha, label = _power_law_fit(xs, ys)
        if alpha is not None:
            x_fit = np.linspace(min(xs), max(xs), 100)
            coeffs = np.polyfit(np.log(xs), np.log(ys), 1)
            y_fit = np.exp(coeffs[1]) * x_fit ** coeffs[0]
            ax.plot(x_fit, y_fit, '--', color=CB_COLORS['hessian'],
                    alpha=0.7, label=label)
            ax.legend(fontsize=9)
        if len(set(xs)) > 1:
            ax.set_xscale('log')
        if len(set(ys)) > 1 and max(ys) / (min(ys) + 1e-15) > 10:
            ax.set_yscale('log')

    ax.set_xlabel('n_active_vps')
    ax.set_ylabel('Mean Hessian call time (s)')
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'hessian_time_vs_nvps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_ipopt_iters_vs_ncells(runs: list, out_dir: str):
    """Scatter: total IPOPT iterations vs n_cells."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('Total IPOPT Iterations vs Number of Cells', fontsize=12)

    for run in runs:
        td = run["_tdata"]
        x = run.get("n_cells")
        y = td.get("ipopt_summary", {}).get("ipopt_iter_count")
        if x is not None and y is not None:
            ax.scatter(x, y, color='steelblue', s=70, zorder=3)
            ax.annotate(str(x), (x, y), textcoords='offset points',
                        xytext=(4, 4), fontsize=8)

    ax.set_xlabel('n_cells')
    ax.set_ylabel('Total IPOPT iterations')
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'ipopt_iters_vs_ncells.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Timing analysis for Phase 2 profiling data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/
  python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/ \\
      --campaign ipopt_btol0.001_lbfgs30_hess
        """
    )
    parser.add_argument('--experiment-dir', required=True,
                        help='Experiment directory containing experiment_index.yaml')
    parser.add_argument('--campaign', default=None,
                        help='Campaign name to select (default: first with timing data)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for figures '
                             '(default: {experiment_dir}/analysis/timing/)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    if not HAS_MPL:
        logger.error("matplotlib is required. Install with: pip install matplotlib")
        return 1

    experiment_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(experiment_dir):
        logger.error(f"Directory not found: {experiment_dir}")
        return 1

    out_dir = args.output_dir or os.path.join(experiment_dir, "analysis", "timing")
    os.makedirs(out_dir, exist_ok=True)

    runs = _load_runs_with_timing(experiment_dir, campaign=args.campaign)
    if not runs:
        logger.warning("No runs with timing data. Exiting.")
        return 0

    logger.info(f"Loaded {len(runs)} run(s) with timing data")
    if args.campaign:
        logger.info(f"Campaign filter: {args.campaign}")
    else:
        campaigns = {r['_selected_campaign'] for r in runs}
        logger.info(f"Campaigns present: {sorted(campaigns)}")

    plot_callback_time_vs_ncells(runs, out_dir)
    plot_steiner_recomps_vs_ntp(runs, out_dir)
    plot_cost_breakdown(runs, out_dir)
    plot_hessian_time_vs_nvps(runs, out_dir)
    plot_ipopt_iters_vs_ncells(runs, out_dir)

    logger.info(f"All figures written to: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
