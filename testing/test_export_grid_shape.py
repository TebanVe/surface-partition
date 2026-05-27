#!/usr/bin/env python3
"""
Smoke test for the grid_shape invariant on the consolidated partition export.

Exports a Phase-2 checkpoint via ``src.export.export_partition`` to a temporary
HDF5 file and asserts:

    /mesh.attrs["grid_shape"][0] * /mesh.attrs["grid_shape"][1]
        == /mesh/vertices.shape[0]

This is the external contract documented in the downstream link-list-torus
reader (Verification Check #4 of its PARTITION_FILE_FORMAT.md §8). A failure
indicates the exporter is writing pre-refinement initial dimensions instead of
the final post-refinement grid that matches the mesh actually stored in
``/mesh/vertices``.

Usage:
    python testing/test_export_grid_shape.py \\
        --solution <path>/iteration_*.h5 \\
        --config parameters/torus_10part.yaml
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import h5py
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export import export_partition
from src.partition.steiner_handler import SteinerHandler
from src.pipeline.io import find_base_solution_path, load_partition_from_refined_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the grid_shape invariant on a fresh partition export"
    )
    parser.add_argument("--solution", required=True, help="Phase-2 checkpoint HDF5")
    parser.add_argument("--config", required=True, help="Experiment YAML")
    args = parser.parse_args()

    print("=" * 80)
    print("EXPORT grid_shape INVARIANT SMOKE TEST")
    print("=" * 80)

    checkpoint_path = os.path.abspath(args.solution)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    with h5py.File(checkpoint_path, "r") as f:
        source_iteration = int(f.attrs["iteration_number"])
        final_perimeter = float(f.attrs["final_perimeter"])
        pending_migration = bool(f.attrs.get("pending_migration", False))

    base_solution_path = find_base_solution_path(checkpoint_path, verbose=False)
    with h5py.File(base_solution_path, "r") as f:
        seed = int(f.attrs["seed"])
        n_theta_final = int(f.attrs["var1"])
        n_phi_final = int(f.attrs["var2"])

    mesh, partition = load_partition_from_refined_file(checkpoint_path, verbose=False)
    steiner_handler = SteinerHandler(mesh, partition)

    source_run_id = Path(checkpoint_path).parent.parent.parent.name
    V_input = mesh.vertices.shape[0]

    print(f"\nCheckpoint:       {checkpoint_path}")
    print(f"Base solution:    {base_solution_path}")
    print(f"n_theta_final:    {n_theta_final}")
    print(f"n_phi_final:      {n_phi_final}")
    print(f"mesh.vertices:    {V_input}")
    print(f"Expected product: {n_theta_final * n_phi_final}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_partition.h5")
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
            n_theta_final=n_theta_final,
            n_phi_final=n_phi_final,
            strict=False,
        )

        with h5py.File(output_path, "r") as f:
            gs = f["mesh"].attrs["grid_shape"]
            V_out = f["mesh/vertices"].shape[0]
            gs0, gs1 = int(gs[0]), int(gs[1])

    print(f"\nExported grid_shape: [{gs0}, {gs1}]")
    print(f"Exported V:          {V_out}")
    print(f"Product:             {gs0 * gs1}")

    ok = (gs0 * gs1 == V_out) and (V_out == V_input)
    if ok:
        print("\nRESULT: PASS")
        return 0
    print(
        f"\nRESULT: FAIL  (gs0*gs1={gs0 * gs1}, V_out={V_out}, V_input={V_input})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
