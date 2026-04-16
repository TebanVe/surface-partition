#!/usr/bin/env python3
"""Parameter space exploration tool for Phase 1 relaxation.

Reads a sweep YAML specification, generates parameter combinations (grid or
paired strategy, with grouped pairing support), creates per-run YAML configs,
and executes them via find_surface_partition.py.

Usage:
    python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml
    python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode local-parallel --workers 4
    python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode generate-only
    python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode collect
"""

import os
import sys
import csv
import copy
import re
import shutil
import argparse
import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Project root is one level above sweep/
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from src.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(data: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Nested dict helpers
# ---------------------------------------------------------------------------

def set_nested(d: dict, dotpath: str, value: Any) -> None:
    """Set a value in a nested dict using dot-notation (e.g. 'relaxation.seed')."""
    keys = dotpath.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested(d: dict, dotpath: str, default: Any = None) -> Any:
    """Get a value from a nested dict using dot-notation."""
    keys = dotpath.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def generate_combinations(
    sweep_dict: dict, strategy: str
) -> List[Dict[str, Any]]:
    """Generate parameter combinations from a sweep specification.

    Separates grouped entries (sub-dicts whose values are themselves dicts of
    equal-length lists) from ungrouped entries (plain lists).  Parameters
    inside a group are zipped together.  Groups and ungrouped parameters are
    then combined according to *strategy* (``"grid"`` or ``"paired"``).
    """
    axes: List[List[Dict[str, Any]]] = []

    for key, value in sweep_dict.items():
        if isinstance(value, dict):
            inner_keys = list(value.keys())
            inner_lists = list(value.values())
            lengths = [len(lst) for lst in inner_lists]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"Group '{key}' has unequal list lengths: "
                    f"{dict(zip(inner_keys, lengths))}"
                )
            group_combos = [
                {k: v for k, v in zip(inner_keys, vals)}
                for vals in zip(*inner_lists)
            ]
            axes.append(group_combos)
        elif isinstance(value, list):
            axes.append([{key: v} for v in value])
        else:
            raise ValueError(
                f"Sweep entry '{key}' must be a list or a group dict, "
                f"got {type(value).__name__}"
            )

    if not axes:
        return [{}]

    if strategy == "grid":
        return [
            _merge_dicts(combo) for combo in itertools.product(*axes)
        ]
    elif strategy == "paired":
        lengths = [len(ax) for ax in axes]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Paired strategy requires all axes to have the same length, "
                f"got lengths {lengths}"
            )
        return [_merge_dicts(combo) for combo in zip(*axes)]
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def _merge_dicts(dicts) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for d in dicts:
        merged.update(d)
    return merged


# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------

def derive_experiment_identity(
    base_config: dict,
) -> Tuple[str, int]:
    """Return (surface, n_partitions) from a base config dict."""
    surface = base_config.get("experiment", {}).get("surface", "torus")
    n_partitions = base_config.get("relaxation", {}).get("n_partitions", 3)
    return surface, int(n_partitions)


# ---------------------------------------------------------------------------
# Post-run analysis
# ---------------------------------------------------------------------------

def find_solution_h5(run_dir: str) -> Optional[str]:
    """Find the .h5 solution file inside a run directory."""
    solution_dir = os.path.join(run_dir, "solution")
    if os.path.isdir(solution_dir):
        for fname in os.listdir(solution_dir):
            if fname.endswith(".h5"):
                return os.path.join(solution_dir, fname)
    return None


def run_post_analysis(run_dir: str) -> None:
    """Generate analysis plots and partition screenshots for a completed run."""
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. Optimization metrics (matplotlib — always available)
    try:
        import matplotlib
        matplotlib.use("agg")
        from scripts.optimization_analyzer import analyze_optimization_run

        analyze_optimization_run(run_dir)
        logger.info(f"Optimization analysis complete for {run_dir}")
    except Exception as exc:
        logger.warning(f"Optimization analysis failed for {run_dir}: {exc}")

    # 2. Partition screenshots (PyVista — run in subprocess to isolate segfaults)
    solution_h5 = find_solution_h5(run_dir)
    if solution_h5:
        try:
            _render_screenshots_subprocess(solution_h5, analysis_dir)
        except Exception as exc:
            logger.warning(f"Partition screenshots failed for {run_dir}: {exc}")


def _render_screenshots_subprocess(solution_h5: str, output_dir: str) -> None:
    """Run screenshot rendering in a subprocess to isolate VTK segfaults."""
    script = (
        "import sys; sys.path.insert(0, '.');\n"
        "from src.visualization.partition_screenshots import "
        "render_partition_screenshots;\n"
        f"r = render_partition_screenshots({solution_h5!r}, {output_dir!r});\n"
        "print(f'Screenshots: {{len(r)}}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_PROJECT_ROOT,
    )
    if result.returncode == 0 and "Screenshots:" in result.stdout:
        logger.info(f"Screenshot subprocess: {result.stdout.strip()}")
    elif result.returncode != 0:
        logger.debug(
            f"Screenshot subprocess exited {result.returncode} "
            f"(PyVista may not be available)"
        )


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def _find_existing_run(experiment_dir: str, combo: Dict[str, Any]) -> Optional[str]:
    """Find an existing run directory whose config matches *combo*.

    Checks each ``run_*`` subdirectory for ``solution/metadata.yaml`` and
    compares the relevant parameters.  Parameters that are not found in
    the metadata are skipped (they may be internal PGD settings not recorded
    in the output).
    """
    for entry in sorted(os.listdir(experiment_dir)):
        if not entry.startswith("run_"):
            continue
        meta_path = os.path.join(experiment_dir, entry, "solution", "metadata.yaml")
        if not os.path.isfile(meta_path):
            continue
        try:
            meta = _load_yaml(meta_path)
        except Exception:
            continue
        inp = meta.get("input_parameters", {})
        match = True
        matched_any = False
        for dotpath, value in combo.items():
            parts = dotpath.split(".")
            key = parts[-1]
            meta_val = inp.get(key)
            if key.startswith("n_grid_") or key.startswith("n_theta") or key.startswith("n_phi"):
                levels = meta.get("levels", [])
                if levels:
                    meta_val = levels[0].get(key, meta_val)
            if meta_val is None:
                sp = inp.get("surface_params", {})
                meta_val = sp.get(key, meta_val)
            if meta_val is None:
                continue
            matched_any = True
            try:
                if abs(float(meta_val) - float(value)) > 1e-9:
                    match = False
                    break
            except (TypeError, ValueError):
                if str(meta_val) != str(value):
                    match = False
                    break
        if match and matched_any:
            return os.path.join(experiment_dir, entry)
    return None


def run_single(
    config_path: str,
    experiment_dir: str,
    combo: Dict[str, Any],
    resume: bool = False,
    dry_run: bool = False,
) -> Optional[str]:
    """Execute a single find_surface_partition.py run.

    The subprocess creates a ``run_{timestamp}_...`` directory under the
    default ``results/`` location.  After completion we move it into the
    experiment directory so all runs for a (surface, npart) pair are grouped.

    Returns the run directory on success, or None.
    """
    if resume:
        existing = _find_existing_run(experiment_dir, combo)
        if existing:
            logger.info(f"Skipping (resume): {existing}")
            return existing

    script = os.path.join(_PROJECT_ROOT, "scripts", "find_surface_partition.py")
    cmd = [
        sys.executable,
        script,
        "--config",
        str(config_path),
    ]

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return None

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT
    )

    if result.returncode != 0:
        logger.error(
            f"Run failed (exit {result.returncode}):\n"
            f"  config: {config_path}\n"
            f"  stderr: {result.stderr[-500:] if result.stderr else '(empty)'}"
        )
        return None

    run_dir = _parse_run_dir_from_output(result.stdout)
    if run_dir is None:
        logger.warning("Could not determine run directory from subprocess output")
        return None

    # Move into experiment directory if it's not already there
    run_dir = os.path.abspath(run_dir)
    experiment_dir_abs = os.path.abspath(experiment_dir)
    if not run_dir.startswith(experiment_dir_abs + os.sep):
        dest = os.path.join(experiment_dir_abs, os.path.basename(run_dir))
        if os.path.exists(dest):
            logger.warning(f"Destination already exists, skipping move: {dest}")
        else:
            shutil.move(run_dir, dest)
            run_dir = dest

    run_post_analysis(run_dir)
    return run_dir


def _parse_run_dir_from_output(stdout: str) -> Optional[str]:
    """Extract 'Results saved in: ...' path from find_surface_partition.py output."""
    for line in stdout.splitlines():
        m = re.search(r"Results saved in:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return None


def run_parallel(
    configs: List[Tuple[str, Dict[str, Any]]],
    experiment_dir: str,
    n_workers: int,
    resume: bool = False,
    dry_run: bool = False,
) -> List[Optional[str]]:
    """Execute runs in parallel via ThreadPoolExecutor."""
    results: List[Optional[str]] = [None] * len(configs)

    try:
        from tqdm import tqdm

        pbar = tqdm(total=len(configs), desc="Sweep runs")
    except ImportError:
        pbar = None

    def _run(idx_cfg_combo):
        idx, (cfg, combo) = idx_cfg_combo
        return idx, run_single(cfg, experiment_dir, combo, resume=resume, dry_run=dry_run)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run, (i, cc)): i for i, cc in enumerate(configs)
        }
        for future in as_completed(futures):
            idx, run_dir = future.result()
            results[idx] = run_dir
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()
    return results


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def _extract_run_metrics(run_dir: str) -> Optional[dict]:
    """Extract metrics from a single run directory's metadata.yaml."""
    meta_path = os.path.join(run_dir, "solution", "metadata.yaml")
    if not os.path.isfile(meta_path):
        return None
    try:
        meta = _load_yaml(meta_path)
    except Exception:
        return None

    inp = meta.get("input_parameters", {})
    levels = meta.get("levels", [])

    total_iterations = 0
    initial_N = None
    final_N = None
    for lm in levels:
        n_iter = lm.get("iters", {}).get("num_iterations", 0)
        total_iterations += int(n_iter)
        N = lm.get("N")
        if N is not None:
            if initial_N is None:
                initial_N = int(N)
            final_N = int(N)

    params: Dict[str, Any] = {
        "lambda_penalty": inp.get("lambda_penalty"),
        "seed": inp.get("seed"),
    }
    res_labels = inp.get("resolution_labels", [])
    if levels:
        for lab in res_labels:
            val = levels[0].get(lab)
            if val is not None:
                params[lab] = int(val)

    return {
        "directory": os.path.basename(run_dir),
        "params": params,
        "status": "completed",
        "perimeter": meta.get("initial_perimeter"),
        "final_energy": meta.get("final_energy"),
        "initial_N": initial_N,
        "final_N": final_N,
        "converged": meta.get("success", False),
        "total_iterations": total_iterations,
    }


def collect_results(
    experiment_dir: str,
    sweep_spec: Optional[dict] = None,
    sweep_id: Optional[str] = None,
) -> dict:
    """Scan experiment directory, update experiment_index.yaml, and optionally
    write a sweep-specific summary CSV.

    Returns the updated index dict.
    """
    surface = None
    n_partitions = None

    existing_index_path = os.path.join(experiment_dir, "experiment_index.yaml")
    existing_index: dict = {}
    if os.path.isfile(existing_index_path):
        existing_index = _load_yaml(existing_index_path) or {}
        surface = existing_index.get("surface")
        n_partitions = existing_index.get("n_partitions")

    existing_runs_by_dir = {}
    for run_entry in existing_index.get("runs", []):
        existing_runs_by_dir[run_entry["directory"]] = run_entry

    run_entries = []
    for entry in sorted(os.listdir(experiment_dir)):
        if not entry.startswith("run_"):
            continue
        run_path = os.path.join(experiment_dir, entry)
        if not os.path.isdir(run_path):
            continue

        metrics = _extract_run_metrics(run_path)
        if metrics is None:
            if entry in existing_runs_by_dir:
                run_entries.append(existing_runs_by_dir[entry])
            continue

        if surface is None:
            meta = _load_yaml(
                os.path.join(run_path, "solution", "metadata.yaml")
            )
            inp = meta.get("input_parameters", {})
            surface = inp.get("surface")
            n_partitions = inp.get("n_partitions")

        if entry in existing_runs_by_dir:
            old = existing_runs_by_dir[entry]
            if "sweep_origin" in old and "sweep_origin" not in metrics:
                metrics["sweep_origin"] = old["sweep_origin"]

        if sweep_id and "sweep_origin" not in metrics:
            metrics["sweep_origin"] = sweep_id

        analysis_dir = os.path.join(run_path, "analysis")
        expected = ["refinement_optimization_metrics.png"]
        if not all(
            os.path.isfile(os.path.join(analysis_dir, f)) for f in expected
        ):
            logger.info(f"Generating missing analysis for {entry}")
            run_post_analysis(run_path)

        run_entries.append(metrics)

    index = {
        "surface": surface,
        "n_partitions": n_partitions,
        "runs": run_entries,
    }
    _save_yaml(index, existing_index_path)
    logger.info(
        f"Updated {existing_index_path} with {len(run_entries)} runs"
    )

    if sweep_id and sweep_spec:
        sweeps_dir = os.path.join(experiment_dir, "sweeps")
        os.makedirs(sweeps_dir, exist_ok=True)
        csv_path = os.path.join(sweeps_dir, f"{sweep_id}_summary.csv")
        _write_summary_csv(csv_path, run_entries)
        logger.info(f"Sweep summary CSV: {csv_path}")

    return index


def _write_summary_csv(csv_path: str, runs: List[dict]) -> None:
    if not runs:
        return
    all_param_keys: List[str] = []
    for r in runs:
        for k in r.get("params", {}):
            if k not in all_param_keys:
                all_param_keys.append(k)

    fieldnames = (
        ["directory", "status", "perimeter", "final_energy"]
        + all_param_keys
        + ["initial_N", "final_N", "converged", "total_iterations"]
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            row = {
                "directory": run.get("directory", ""),
                "status": run.get("status", ""),
                "perimeter": run.get("perimeter", ""),
                "final_energy": run.get("final_energy", ""),
                "initial_N": run.get("initial_N", ""),
                "final_N": run.get("final_N", ""),
                "converged": run.get("converged", ""),
                "total_iterations": run.get("total_iterations", ""),
            }
            for k in all_param_keys:
                row[k] = run.get("params", {}).get(k, "")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for Phase 1 relaxation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep", required=True, help="Path to sweep specification YAML"
    )
    parser.add_argument(
        "--mode",
        default="local-sequential",
        choices=["local-sequential", "local-parallel", "generate-only", "collect"],
        help="Execution mode (default: local-sequential)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers (local-parallel only)"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Base directory for results"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose metadata.yaml already exists",
    )
    args = parser.parse_args()

    setup_logging(log_level="INFO", log_to_console=True, log_to_file=False)

    sweep_spec = _load_yaml(args.sweep)
    sweep_name = sweep_spec.get("name", "sweep")
    base_config_path = sweep_spec.get("base_config")
    if not base_config_path:
        print("ERROR: sweep spec missing 'base_config'")
        return 1

    base_config = _load_yaml(base_config_path)
    surface, n_partitions = derive_experiment_identity(base_config)

    experiment_dir = os.path.join(args.output_dir, f"{surface}_npart{n_partitions}")
    os.makedirs(experiment_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_id = f"{timestamp}_{sweep_name}"

    if args.mode == "collect":
        collect_results(experiment_dir, sweep_spec, sweep_id)
        print(f"Results collected in {experiment_dir}/experiment_index.yaml")
        return 0

    sweep_params = sweep_spec.get("sweep", {})
    strategy = sweep_spec.get("strategy", "grid")
    combos = generate_combinations(sweep_params, strategy)

    print(f"Sweep: {sweep_name}")
    print(f"Surface: {surface}, N partitions: {n_partitions}")
    print(f"Strategy: {strategy} -> {len(combos)} parameter combinations")
    print(f"Experiment directory: {experiment_dir}")

    sweeps_dir = os.path.join(experiment_dir, "sweeps")
    os.makedirs(sweeps_dir, exist_ok=True)
    shutil.copy2(args.sweep, os.path.join(sweeps_dir, f"{sweep_id}.yaml"))

    configs: List[Tuple[str, Dict[str, Any]]] = []
    for i, combo in enumerate(combos, 1):
        run_config = copy.deepcopy(base_config)
        for dotpath, value in combo.items():
            set_nested(run_config, dotpath, value)

        config_path = os.path.join(
            sweeps_dir, f"{sweep_id}_run{i:03d}.yaml"
        )
        _save_yaml(run_config, config_path)
        configs.append((config_path, combo))

        if args.dry_run or args.mode == "generate-only":
            override_strs = [f"{k}={v}" for k, v in combo.items()]
            print(f"  Run {i:3d}: {', '.join(override_strs)}")
        print(config_path)

    if args.mode == "generate-only":
        print(f"\nGenerated {len(configs)} configs in {sweeps_dir}")
        return 0

    if args.mode == "local-sequential":
        for config_path, combo in configs:
            run_single(
                config_path,
                experiment_dir,
                combo,
                resume=args.resume,
                dry_run=args.dry_run,
            )
    elif args.mode == "local-parallel":
        run_parallel(
            configs,
            experiment_dir,
            args.workers,
            resume=args.resume,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        collect_results(experiment_dir, sweep_spec, sweep_id)
        print(f"\nSweep complete. Results in {experiment_dir}")
        print(f"Index: {experiment_dir}/experiment_index.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
