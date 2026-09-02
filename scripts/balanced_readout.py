#!/usr/bin/env python3
"""
Balanced Readout: equal-area, connected extraction from a Phase 1 solution.

Post-processing stage between Phase 1 and Phase 2. Reads a relaxation solution,
replaces the plain winner-take-all argmax with a dual-shifted argmax that makes
the discrete cell areas equal, repairs disconnected cells, and writes a new
solution file in the standard Phase 1 schema so every downstream tool
(visualize_partition_fast, refine_perimeter, export_partition,
testing/check_fragmentation) consumes it unchanged.

The source solution is never modified. Output lands in a campaign directory under
<run_root>/readout/, mirroring how Phase 2 campaigns live under <run_root>/refinement/.

Usage:
    python scripts/balanced_readout.py \
        --solution results/run_xyz/solution/surface_....h5

    # Explicit campaign name (default: dualshift_gate{threshold}_repair)
    python scripts/balanced_readout.py \
        --solution results/run_xyz/solution/surface_....h5 \
        --campaign dualshift_gate0.05_repair

    # Dual shifts only, no connectivity repair
    python scripts/balanced_readout.py \
        --solution results/run_xyz/solution/surface_....h5 --no-repair
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging_config import get_logger, setup_logging
from src.mesh.tri_mesh import TriMesh
from src.partition.balanced_readout import (
    BalancedReadoutConfig,
    apply_balanced_readout,
)

logger = get_logger(__name__)

# Attributes copied verbatim from the source solution so the derived file is a
# drop-in for every reader of the Phase 1 schema.
INHERITED_ATTRS = (
    "n_partitions",
    "surface",
    "completed_levels",
    "lambda_penalty",
    "seed",
    "optimizer",
    "use_analytic",
    "var1",
    "var2",
    "resolution_labels",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Balanced readout of a Phase 1 relaxation solution"
    )
    p.add_argument("--solution", required=True, help="Path to the Phase 1 solution .h5")
    p.add_argument("--campaign", default=None, help="Campaign directory name")
    p.add_argument(
        "--gate-threshold",
        type=float,
        default=0.05,
        help="Relative area deviation counted as imbalanced (default: 0.05)",
    )
    p.add_argument("--dual-iters", type=int, default=400)
    p.add_argument("--max-repair-sweeps", type=int, default=200)
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="Apply the dual shifts only; skip connectivity repair",
    )
    return p.parse_args()


def _to_builtin(obj):
    """Recursively convert numpy scalars/arrays so yaml.safe_dump accepts them."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    args = parse_args()
    solution_path = Path(args.solution).resolve()
    if not solution_path.is_file():
        print(f"ERROR: solution not found: {solution_path}")
        return 1

    # <run_root>/solution/<file>.h5  ->  <run_root>/readout/<campaign>/
    run_root = solution_path.parent.parent
    campaign = args.campaign or f"dualshift_gate{args.gate_threshold}_repair"
    out_dir = run_root / "readout" / campaign
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_level="INFO", log_to_file=False, log_to_console=True)
    fh = logging.FileHandler(out_dir / "readout.log", mode="w")
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(fh)

    logger.info("Balanced readout")
    logger.info("source solution: %s", solution_path)
    logger.info("output campaign: %s", out_dir)

    with h5py.File(solution_path, "r") as f:
        vertices = f["vertices"][:]
        faces = f["faces"][:]
        densities = f["x_opt"][:]
        x0 = f["x0"][:] if "x0" in f else None
        n_partitions = int(f.attrs["n_partitions"])
        src_attrs = {k: f.attrs[k] for k in INHERITED_ATTRS if k in f.attrs}

    densities = densities.reshape(vertices.shape[0], n_partitions)
    mesh = TriMesh(vertices, faces)
    logger.info(
        "mesh: %d vertices, %d faces, %d partitions",
        vertices.shape[0],
        faces.shape[0],
        n_partitions,
    )

    config = BalancedReadoutConfig(
        gate_threshold=args.gate_threshold,
        dual_iters=args.dual_iters,
        max_repair_sweeps=args.max_repair_sweeps,
        repair_enabled=not args.no_repair,
    )
    result = apply_balanced_readout(
        densities, faces, vertices, mesh.v, n_partitions, config
    )

    out_h5 = out_dir / "solution_balanced.h5"
    tmp_h5 = out_h5.with_suffix(".h5.tmp")
    with h5py.File(tmp_h5, "w") as f:
        f.create_dataset("x_opt", data=result["densities"].ravel())
        f.create_dataset("x0", data=x0 if x0 is not None else densities.ravel())
        f.create_dataset("vertices", data=vertices)
        f.create_dataset("faces", data=faces)
        # Provenance and the full readout state: psi plus both labelings make the
        # source densities exactly recoverable and every relabeling auditable.
        f.create_dataset("psi", data=result["psi"])
        f.create_dataset("labels_final", data=result["labels"].astype(np.int32))
        f.create_dataset("labels_source", data=result["source_labels"].astype(np.int32))
        for k, v in src_attrs.items():
            f.attrs[k] = v
        f.attrs["derived_from"] = os.path.relpath(solution_path, out_dir)
        f.attrs["readout_method"] = "dual_shift_argmax+connectivity_repair"
        f.attrs["readout_timestamp"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["n_relabeled"] = int(result["n_relabeled_vs_source"])
        f.attrs["gate_threshold"] = float(args.gate_threshold)
    tmp_h5.replace(out_h5)
    logger.info("wrote %s", out_h5)

    with open(out_dir / "readout.yaml", "w") as f:
        yaml.safe_dump(
            _to_builtin(
                {
                    "stage": "balanced_readout",
                    "script": "scripts/balanced_readout.py",
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "source_solution": os.path.relpath(solution_path, run_root),
                    "campaign": campaign,
                    "config": result["config"],
                }
            ),
            f,
            sort_keys=False,
        )

    stages = result["stages"]
    metadata = {
        "balanced_readout": {
            "source_solution": os.path.relpath(solution_path, run_root),
            "target_area": result["target_area"],
            "vertex_granularity_rel": result["vertex_granularity_rel"],
            "dual_worst_rel_dev": result["dual_worst_rel_dev"],
            "n_relabeled_vs_source": result["n_relabeled_vs_source"],
            "psi_min": float(np.min(result["psi"])),
            "psi_max": float(np.max(result["psi"])),
            "island_report": result["island_report"],
            "rebalance_report": result["rebalance_report"],
            "config": result["config"],
        },
        "stages": {
            name: {
                "area_imbalance": s["area_imbalance"],
                "disconnected_cells": s["disconnected_cells"],
                "label_boundary_length": s["label_boundary_length"],
            }
            for name, s in stages.items()
        },
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(_to_builtin(metadata), f, sort_keys=False)

    src, rep = stages["source"], stages["repaired"]
    print("\n" + "=" * 68)
    print("BALANCED READOUT")
    print("=" * 68)
    for name, s in (("source  ", src), ("repaired", rep)):
        print(
            f"  {name}: {s['area_imbalance']['n_imbalanced']:3d} imbalanced "
            f"(worst {s['area_imbalance']['worst_rel_dev']*100:6.2f}%), "
            f"{s['disconnected_cells']['n_fragmented']:2d} fragmented, "
            f"boundary length {s['label_boundary_length']:.4f}"
        )
    dl = (rep["label_boundary_length"] - src["label_boundary_length"]) / src[
        "label_boundary_length"
    ]
    print(f"\n  boundary length change: {dl*100:+.2f}%")
    print(f"  vertices relabeled:     {result['n_relabeled_vs_source']}")
    if result["rebalance_report"]:
        rr = result["rebalance_report"]
        print(
            f"  repair strain:          {rr['n_moves']} moves, "
            f"{rr['n_blocked_by_connectivity']} blocked by connectivity, "
            f"sweeps {rr['sweeps_used']}"
            + ("  [HIT SWEEP CAP]" if rr["hit_sweep_cap"] else "")
        )
    ok = (
        rep["area_imbalance"]["n_imbalanced"] == 0
        and rep["disconnected_cells"]["n_fragmented"] == 0
    )
    print(f"\n  VALID PARTITION: {'YES' if ok else 'NO'}")
    print(f"\n  output: {out_h5}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
