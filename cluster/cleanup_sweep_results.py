#!/usr/bin/env python3
"""
Prune run directories from a sweep experiment, keeping only the N best by perimeter.

Historical records are preserved:
  - sweeps/  directory (CSV summaries + per-run configs) is never touched
  - A cleanup_log.yaml is written/appended to record every deletion with full provenance

experiment_index.yaml is NOT modified. It will be rebuilt from surviving directories
automatically the next time `--mode collect` is run.

Usage:
    # Preview what would be deleted (safe, no changes)
    python cluster/cleanup_sweep_results.py --experiment-dir results/torus_npart10/

    # Delete, keeping the 10 best by perimeter (default)
    python cluster/cleanup_sweep_results.py --experiment-dir results/torus_npart10/ --execute

    # Keep only the 5 best
    python cluster/cleanup_sweep_results.py --experiment-dir results/torus_npart10/ --keep 5 --execute

    # Rank by a different metric (lower = better)
    python cluster/cleanup_sweep_results.py --experiment-dir results/torus_npart10/ --metric final_energy --execute
"""

import argparse
import os
import shutil
import sys
import yaml
from datetime import datetime
from typing import Optional


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def du_mb(path: str) -> float:
    """Return approximate directory size in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            fp = os.path.join(dirpath, fname)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 ** 2)


def load_runs(experiment_dir: str) -> list:
    """Load run entries from experiment_index.yaml."""
    index_path = os.path.join(experiment_dir, "experiment_index.yaml")
    if not os.path.isfile(index_path):
        print(f"ERROR: No experiment_index.yaml found in {experiment_dir}", file=sys.stderr)
        sys.exit(1)
    index = load_yaml(index_path)
    runs = index.get("runs", [])
    if not runs:
        print("No runs found in experiment_index.yaml.", file=sys.stderr)
        sys.exit(1)
    return runs


def rank_runs(runs: list, metric: str, experiment_dir: str):
    """
    Split runs into (rankable, unrankable) and sort rankable ascending by metric.
    Runs whose directory no longer exists on disk are treated as unrankable.
    """
    rankable = []
    unrankable = []

    for run in runs:
        directory = run.get("directory", "")
        run_path = os.path.join(experiment_dir, directory)
        if not os.path.isdir(run_path):
            unrankable.append((run, "directory missing on disk"))
            continue
        value = run.get(metric)
        if value is None:
            unrankable.append((run, f"metric '{metric}' not available"))
        else:
            rankable.append((run, float(value)))

    rankable.sort(key=lambda x: x[1])
    return rankable, unrankable


def print_table(title: str, rows: list, metric: str):
    if not rows:
        return
    print(f"\n{title}")
    print(f"  {'Rank':<6} {metric:<12} {'sweep_origin':<30} directory")
    print(f"  {'-'*6} {'-'*12} {'-'*30} {'-'*60}")
    for i, (run, value) in enumerate(rows, 1):
        origin = run.get("sweep_origin", "—")
        val_str = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"  {i:<6} {val_str:<12} {origin:<30} {run['directory']}")


def append_cleanup_log(experiment_dir: str, kept: list, deleted: list,
                       metric: str, keep_n: int):
    log_path = os.path.join(experiment_dir, "cleanup_log.yaml")

    existing = []
    if os.path.isfile(log_path):
        with open(log_path) as f:
            existing = yaml.safe_load(f) or []

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metric": metric,
        "keep_n": keep_n,
        "kept": [
            {
                "directory": run["directory"],
                "perimeter": run.get("perimeter"),
                "params": run.get("params"),
                "sweep_origin": run.get("sweep_origin"),
            }
            for run, _ in kept
        ],
        "deleted": [
            {
                "directory": run["directory"],
                "perimeter": run.get("perimeter"),
                "params": run.get("params"),
                "sweep_origin": run.get("sweep_origin"),
                "reason": reason,
            }
            for run, reason in deleted
        ],
    }
    existing.append(entry)

    with open(log_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nCleanup log written: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prune sweep run directories, keeping the N best by perimeter."
    )
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Path to the experiment directory (contains experiment_index.yaml)"
    )
    parser.add_argument(
        "--keep", type=int, default=10,
        help="Number of best runs to keep (default: 10)"
    )
    parser.add_argument(
        "--metric", default="perimeter",
        help="Metric to rank by, lower = better (default: perimeter)"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually delete directories. Without this flag, only a preview is shown."
    )
    args = parser.parse_args()

    experiment_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(experiment_dir):
        print(f"ERROR: Directory not found: {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(experiment_dir)
    rankable, unrankable = rank_runs(runs, args.metric, experiment_dir)

    total_on_disk = len(rankable)
    keep_n = min(args.keep, total_on_disk)
    to_keep = rankable[:keep_n]
    to_delete_ranked = list(rankable[keep_n:])  # list of (run, float_value)

    # Unrankable runs present on disk are also candidates for deletion
    unrankable_on_disk = [
        (run, reason) for run, reason in unrankable
        if os.path.isdir(os.path.join(experiment_dir, run.get("directory", "")))
    ]
    # Combine: ranked deletions store (run, float); unrankable store (run, str reason)
    to_delete = [(run, f"ranked {keep_n + i + 1}/{total_on_disk}") for i, (run, _) in enumerate(to_delete_ranked)]
    to_delete += unrankable_on_disk

    # --- Summary ---
    print(f"\nExperiment: {experiment_dir}")
    print(f"Metric:     {args.metric} (lower = better)")
    print(f"Total runs on disk: {total_on_disk}")
    print(f"Keeping:    {keep_n} best")
    print(f"Deleting:   {len(to_delete)} ({len(to_delete_ranked)} ranked + "
          f"{len(unrankable_on_disk)} unrankable)")

    print_table("KEEP", to_keep, args.metric)

    if to_delete_ranked:
        # Rebuild with float values for the table display
        print_table("DELETE (ranked, below cutoff)",
                    [(run, value) for (run, value) in rankable[keep_n:]], args.metric)

    if unrankable_on_disk:
        print(f"\nDELETE (unrankable — missing '{args.metric}' or other issue):")
        for run, reason in unrankable_on_disk:
            print(f"  {run['directory']}  [{reason}]")

    if not to_delete:
        print("\nNothing to delete.")
        return

    # --- Estimate disk savings ---
    print("\nEstimating disk usage of directories to delete...")
    total_mb = 0.0
    for run, _ in to_delete:
        run_path = os.path.join(experiment_dir, run["directory"])
        mb = du_mb(run_path)
        total_mb += mb
        print(f"  {mb:8.1f} MB  {run['directory']}")
    print(f"\n  Total to free: {total_mb:.1f} MB  ({total_mb/1024:.2f} GB)")

    if not args.execute:
        print("\n[DRY RUN] No files deleted. Re-run with --execute to delete.")
        return

    # --- Confirmation ---
    print(f"\nAbout to permanently delete {len(to_delete)} run director"
          f"{'y' if len(to_delete) == 1 else 'ies'}.")
    print("The sweeps/ directory and experiment_index.yaml will NOT be modified.")
    answer = input("Confirm deletion? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    # --- Execute ---
    deleted_log = []
    freed_mb = 0.0
    for run, reason in to_delete:
        run_path = os.path.join(experiment_dir, run["directory"])
        mb = du_mb(run_path)
        shutil.rmtree(run_path)
        freed_mb += mb
        print(f"  Deleted ({mb:.1f} MB): {run['directory']}")
        deleted_log.append((run, reason))

    print(f"\nDone. Freed {freed_mb:.1f} MB ({freed_mb/1024:.2f} GB).")
    print(f"Kept {keep_n} runs. Run `--mode collect` to rebuild experiment_index.yaml.")

    append_cleanup_log(experiment_dir, to_keep, deleted_log, args.metric, args.keep)


if __name__ == "__main__":
    main()
