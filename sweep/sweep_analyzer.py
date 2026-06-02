#!/usr/bin/env python3
"""Experiment-wide analysis and visualization for parameter sweeps.

Reads experiment_index.yaml and generates comparison plots across all runs
in an experiment directory: heatmaps, line plots with error bars, convergence
overlays, and parameter sensitivity charts.

Usage:
    python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/
    python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/ --x-param lambda_penalty --y-param initial_N
    python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/ --metric final_energy
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import yaml

# Project root is one level above sweep/
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from src.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment_index(experiment_dir: str) -> dict:
    index_path = os.path.join(experiment_dir, "experiment_index.yaml")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"experiment_index.yaml not found in {experiment_dir}. "
            f"Run 'parameter_sweep.py --mode collect' first."
        )
    with open(index_path, "r") as f:
        return yaml.safe_load(f) or {}


def _get_completed_runs(index: dict) -> List[dict]:
    return [
        r for r in index.get("runs", [])
        if r.get("status") != "failed" or r.get("perimeter") is not None
    ]


def _get_param(run: dict, param: str) -> Any:
    """Retrieve a parameter value from a run entry.

    Checks run['params'][param] first, then top-level run[param].
    """
    val = run.get("params", {}).get(param)
    if val is not None:
        return val
    return run.get(param)


def _detect_varying_params(runs: List[dict]) -> List[str]:
    """Identify parameters that vary across runs."""
    all_params: Dict[str, set] = defaultdict(set)
    for r in runs:
        for k, v in r.get("params", {}).items():
            if v is not None:
                all_params[k].add(str(v))
    return [k for k, vals in all_params.items() if len(vals) > 1]


# ---------------------------------------------------------------------------
# Load trace data for convergence overlay
# ---------------------------------------------------------------------------

def _load_trace_data(
    experiment_dir: str, run_entry: dict
) -> Optional[Tuple[List[float], List[int]]]:
    """Load energy-vs-iteration data from a run's trace summary files."""
    run_dir = os.path.join(experiment_dir, run_entry["directory"])
    meta_path = os.path.join(run_dir, "solution", "metadata.yaml")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f) or {}
    except Exception:
        return None

    levels = meta.get("levels", [])
    energies: List[float] = []
    cum_iter = 0
    iterations: List[int] = []

    for lm in sorted(levels, key=lambda x: x.get("level", 0)):
        summary_file = lm.get("files", {}).get("summary")
        if summary_file and not os.path.exists(summary_file):
            basename = os.path.basename(summary_file)
            summary_file = os.path.join(run_dir, "traces", basename)
        if not summary_file or not os.path.exists(summary_file):
            continue
        try:
            with open(summary_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("MAJOR"):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        energies.append(float(parts[3]))
                        iterations.append(cum_iter + int(parts[0]))
            n_iter = lm.get("iters", {}).get("num_iterations", 0)
            cum_iter += int(n_iter)
        except Exception:
            continue

    if not energies:
        return None
    return energies, iterations


# ---------------------------------------------------------------------------
# Plot: Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    runs: List[dict],
    x_param: str,
    y_param: str,
    metric: str,
    output_path: str,
    surface: Optional[str] = None,
    n_partitions: Optional[int] = None,
) -> None:
    """Heatmap of *metric* vs (*x_param*, *y_param*).

    Runs that did not converge are marked with hatching.
    When multiple runs share the same (x, y) combination (e.g., different
    seeds), the metric values are averaged.
    """
    data: Dict[Tuple, List[float]] = defaultdict(list)
    converged_map: Dict[Tuple, List[bool]] = defaultdict(list)

    for r in runs:
        xv = _get_param(r, x_param)
        yv = _get_param(r, y_param)
        mv = r.get(metric)
        if xv is None or yv is None or mv is None:
            continue
        key = (float(xv), float(yv))
        data[key].append(float(mv))
        converged_map[key].append(bool(r.get("converged", False)))

    if not data:
        logger.warning("No data for heatmap; skipping")
        return

    x_vals = sorted(set(k[0] for k in data))
    y_vals = sorted(set(k[1] for k in data))

    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    conv_grid = np.ones((len(y_vals), len(x_vals)), dtype=bool)

    for (xv, yv), vals in data.items():
        xi = x_vals.index(xv)
        yi = y_vals.index(yv)
        grid[yi, xi] = np.mean(vals)
        conv_grid[yi, xi] = all(converged_map[(xv, yv)])

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:g}" for v in y_vals])
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)

    for yi_idx in range(len(y_vals)):
        for xi_idx in range(len(x_vals)):
            val = grid[yi_idx, xi_idx]
            if np.isnan(val):
                ax.text(xi_idx, yi_idx, "N/A", ha="center", va="center", fontsize=8)
                continue
            txt = f"{val:.2f}"
            if not conv_grid[yi_idx, xi_idx]:
                txt += "\n(NC)"
            ax.text(xi_idx, yi_idx, txt, ha="center", va="center", fontsize=8)

    title_parts = [f"{metric} vs ({x_param}, {y_param})"]
    if surface:
        title_parts.append(f"surface={surface}")
    if n_partitions:
        title_parts.append(f"npart={n_partitions}")
    ax.set_title(" | ".join(title_parts), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: Line plots
# ---------------------------------------------------------------------------

def plot_line_charts(
    runs: List[dict],
    varying_params: List[str],
    metric: str,
    output_path: str,
    surface: Optional[str] = None,
    n_partitions: Optional[int] = None,
) -> None:
    """Line plot of *metric* vs each varying parameter.

    Includes error bars when multiple seeds exist for the same parameter
    combination.
    """
    n_plots = len(varying_params)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    axes = axes[0]

    for ax, param in zip(axes, varying_params):
        groups: Dict[float, List[float]] = defaultdict(list)
        for r in runs:
            pv = _get_param(r, param)
            mv = r.get(metric)
            if pv is None or mv is None:
                continue
            groups[float(pv)].append(float(mv))

        if not groups:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{metric} vs {param}")
            continue

        xs = sorted(groups.keys())
        means = [np.mean(groups[x]) for x in xs]
        stds = [np.std(groups[x]) if len(groups[x]) > 1 else 0.0 for x in xs]

        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, linewidth=2)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {param}")
        ax.grid(True, alpha=0.3)

    title_parts = [f"Parameter Effects on {metric}"]
    if surface:
        title_parts.append(f"surface={surface}")
    if n_partitions:
        title_parts.append(f"npart={n_partitions}")
    fig.suptitle(" | ".join(title_parts), fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Line chart saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: Convergence comparison overlay
# ---------------------------------------------------------------------------

def plot_convergence_overlay(
    experiment_dir: str,
    runs: List[dict],
    output_path: str,
    max_runs: int = 20,
    surface: Optional[str] = None,
    n_partitions: Optional[int] = None,
) -> None:
    """Overlay energy-vs-iteration curves for selected runs."""
    fig, ax = plt.subplots(figsize=(12, 7))

    plotted = 0
    for r in runs:
        if plotted >= max_runs:
            break
        trace = _load_trace_data(experiment_dir, r)
        if trace is None:
            continue
        energies, iters = trace
        params = r.get("params", {})
        label_parts = [f"{k}={v}" for k, v in params.items() if v is not None]
        label = ", ".join(label_parts[:3]) or r.get("directory", "")[:30]
        ax.plot(iters, energies, linewidth=1.2, alpha=0.8, label=label)
        plotted += 1

    if plotted == 0:
        logger.warning("No trace data found for convergence overlay; skipping")
        plt.close()
        return

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=max(1, plotted // 10))

    title_parts = ["Convergence Comparison"]
    if surface:
        title_parts.append(f"surface={surface}")
    if n_partitions:
        title_parts.append(f"npart={n_partitions}")
    ax.set_title(" | ".join(title_parts), fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Convergence overlay saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: Parameter sensitivity (variance decomposition)
# ---------------------------------------------------------------------------

def plot_parameter_sensitivity(
    runs: List[dict],
    varying_params: List[str],
    metric: str,
    output_path: str,
    surface: Optional[str] = None,
    n_partitions: Optional[int] = None,
) -> None:
    """Bar chart showing how much each parameter affects the metric.

    Uses the ratio of inter-group variance to total variance as a rough
    measure of importance.
    """
    all_vals = [r.get(metric) for r in runs if r.get(metric) is not None]
    if len(all_vals) < 2:
        logger.warning("Not enough data for sensitivity analysis; skipping")
        return

    total_var = np.var(all_vals)
    if total_var < 1e-15:
        logger.warning("Zero variance in metric; skipping sensitivity plot")
        return

    importances: Dict[str, float] = {}
    for param in varying_params:
        groups: Dict[Any, List[float]] = defaultdict(list)
        for r in runs:
            pv = _get_param(r, param)
            mv = r.get(metric)
            if pv is not None and mv is not None:
                groups[pv].append(float(mv))

        if len(groups) < 2:
            importances[param] = 0.0
            continue

        group_means = [np.mean(g) for g in groups.values()]
        between_var = np.var(group_means)
        importances[param] = between_var / total_var

    if not importances:
        return

    params_sorted = sorted(importances, key=importances.get, reverse=True)
    vals_sorted = [importances[p] for p in params_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(params_sorted)), vals_sorted, color="#4C72B0", alpha=0.85)
    ax.set_yticks(range(len(params_sorted)))
    ax.set_yticklabels(params_sorted)
    ax.set_xlabel("Relative Importance (between-group var / total var)")
    ax.set_xlim(0, max(vals_sorted) * 1.15 if vals_sorted else 1)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, v in zip(bars, vals_sorted):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=9,
        )

    title_parts = [f"Parameter Sensitivity ({metric})"]
    if surface:
        title_parts.append(f"surface={surface}")
    if n_partitions:
        title_parts.append(f"npart={n_partitions}")
    ax.set_title(" | ".join(title_parts), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Sensitivity chart saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment-wide sweep analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Path to experiment directory (contains experiment_index.yaml)",
    )
    parser.add_argument(
        "--x-param", default="lambda_penalty",
        help="Parameter for x-axis (default: lambda_penalty)",
    )
    parser.add_argument(
        "--y-param", default="initial_N",
        help="Parameter for y-axis in heatmaps (default: initial_N)",
    )
    parser.add_argument(
        "--metric", default="perimeter",
        help="Metric to plot (default: perimeter; alternatives: final_energy, "
             "total_iterations, relax_timing_total_wall_s, "
             "relax_timing_projection_pct_wall, relax_timing_mean_projection_inner_iters, "
             "and other relax_timing_* scalars written by --profile Phase 1 runs)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for plots (default: {experiment-dir}/analysis/)",
    )
    args = parser.parse_args()

    setup_logging(log_level="INFO", log_to_console=True, log_to_file=False)

    index = load_experiment_index(args.experiment_dir)
    runs = _get_completed_runs(index)
    surface = index.get("surface")
    n_partitions = index.get("n_partitions")

    if not runs:
        print("No completed runs found in experiment index.")
        return 1

    print(f"Loaded {len(runs)} completed runs for {surface} npart={n_partitions}")

    output_dir = args.output_dir or os.path.join(args.experiment_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    varying = _detect_varying_params(runs)
    print(f"Varying parameters: {varying}")

    if len(varying) >= 2 or (args.x_param in varying and args.y_param != args.x_param):
        plot_heatmap(
            runs, args.x_param, args.y_param, args.metric,
            os.path.join(output_dir, f"heatmap_{args.metric}.png"),
            surface=surface, n_partitions=n_partitions,
        )

    if varying:
        plot_line_charts(
            runs, varying, args.metric,
            os.path.join(output_dir, f"line_{args.metric}.png"),
            surface=surface, n_partitions=n_partitions,
        )

    plot_convergence_overlay(
        args.experiment_dir, runs,
        os.path.join(output_dir, "convergence_overlay.png"),
        surface=surface, n_partitions=n_partitions,
    )

    if len(varying) >= 1:
        plot_parameter_sensitivity(
            runs, varying, args.metric,
            os.path.join(output_dir, f"sensitivity_{args.metric}.png"),
            surface=surface, n_partitions=n_partitions,
        )

    print(f"\nAnalysis plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
