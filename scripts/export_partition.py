#!/usr/bin/env python3
"""
Export a Phase-2 checkpoint to the link-list-torus partition HDF5 schema.

This is a thin CLI wrapper around `src.export.export_partition`. It loads the
checkpoint and base solution via the existing pipeline I/O, materialises the
Steiner geometry, and writes one self-contained file in the schema documented
in `../link-list-torus/docs/design/PARTITION_INPUT_ASSESSMENT.md`.

Usage:
    python scripts/export_partition.py \
        --solution results/<run>/refinement/<campaign>/iteration_NNN_*.h5 \
        --config parameters/torus_10part.yaml \
        --output results/<run>/partition/torus_partition_<run-id>.h5
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export import export_partition
from src.partition.steiner_handler import SteinerHandler
from src.pipeline.io import find_base_solution_path, load_partition_from_refined_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a Phase-2 checkpoint to the link-list-torus schema"
    )
    parser.add_argument(
        "--solution", type=str, required=True,
        help="Path to a Phase-2 iteration checkpoint HDF5 file."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the experiment YAML containing surface.torus radii."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HDF5 path. Default: <run_dir>/partition/torus_partition_<run-id>.h5"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Raise an error if the checkpoint has pending_migration=True. "
             "Default is permissive (warn and continue, writing finalised=False)."
    )
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.solution)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return 1

    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    if "surface" not in config or "torus" not in config["surface"]:
        print("ERROR: config has no surface.torus section (R, r, n_theta, n_phi)")
        return 1

    run_dir = Path(checkpoint_path).parent.parent.parent
    source_run_id = run_dir.name

    with h5py.File(checkpoint_path, "r") as f:
        source_iteration = int(f.attrs["iteration_number"])
        final_perimeter = float(f.attrs["final_perimeter"])
        pending_migration = bool(f.attrs.get("pending_migration", False))

    base_solution_path = find_base_solution_path(checkpoint_path, verbose=True)
    with h5py.File(base_solution_path, "r") as f:
        seed = int(f.attrs["seed"])

    mesh, partition = load_partition_from_refined_file(checkpoint_path, verbose=True)
    steiner_handler = SteinerHandler(mesh, partition)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        partition_dir = run_dir / "partition"
        partition_dir.mkdir(exist_ok=True)
        output_path = str(partition_dir / f"torus_partition_{source_run_id}.h5")

    export_partition(
        partition=partition,
        mesh=mesh,
        steiner_handler=steiner_handler,
        config=config,
        output_path=output_path,
        source_run_id=source_run_id,
        source_iteration=source_iteration,
        seed=seed,
        final_perimeter=final_perimeter,
        pending_migration=pending_migration,
        strict=args.strict,
    )

    print(f"Exported partition to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
