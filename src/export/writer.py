"""
Assemble and write the link-list-torus partition export HDF5.

The schema is documented in `../link-list-torus/docs/design/PARTITION_INPUT_ASSESSMENT.md`
and reproduced in §5 of `prompts/prompt.md`.
"""

from datetime import datetime, timezone
import logging
import os
import subprocess
from typing import Optional

import h5py
import numpy as np

from .rep3_builder import build_representation_3


logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def export_partition(
    partition,
    mesh,
    steiner_handler,
    config: dict,
    output_path: str,
    source_run_id: str,
    source_iteration: int,
    seed: int,
    final_perimeter: float,
    pending_migration: bool,
    n_theta_final: int,
    n_phi_final: int,
    strict: bool = False,
) -> None:
    """Export a finalised torus partition to the link-list-torus HDF5 schema.

    All inputs are already-loaded Python objects; this function does not open
    any input files. It builds Representation 3, packages the canonical
    snapshot, and writes one self-contained HDF5 file at ``output_path``.

    ``n_theta_final`` and ``n_phi_final`` are the structured-grid dimensions
    of the actual mesh in ``mesh.vertices`` after Phase-1 multi-level
    refinement. They are sourced from the base-solution HDF5 (attrs ``var1``
    and ``var2``) by the caller, not from the experiment config (which holds
    only the pre-refinement initial values).

    The ``finalised`` flag is set to ``not pending_migration``. When
    ``pending_migration`` is True, a warning is emitted (or, with
    ``strict=True``, an error is raised) but the export is not blocked unless
    ``strict`` is set.
    """
    if pending_migration:
        msg = (
            "This checkpoint has a pending topology migration; the partition "
            "state may shift in the next iteration. Exporting with "
            "finalised=False."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
        print(f"WARNING: {msg}")

    torus_cfg = config["surface"]["torus"]
    R = float(torus_cfg["R"])
    r = float(torus_cfg["r"])
    n_theta = int(n_theta_final)
    n_phi = int(n_phi_final)

    active_vps = [vp for vp in partition.variable_points if vp.active]
    n_vp = len(active_vps)
    n_tp = len(steiner_handler.triple_points)
    n_cells = int(partition.n_cells)

    vertex_labels = partition.indicator_functions.argmax(axis=1).astype(np.int32)
    V = mesh.vertices.shape[0]

    if n_theta * n_phi != V:
        raise ValueError(
            f"grid_shape inconsistency: n_theta_final * n_phi_final = "
            f"{n_theta} * {n_phi} = {n_theta * n_phi}, but mesh.vertices has "
            f"{V} rows. The structured grid dimensions must match the actual "
            f"mesh stored in /mesh/vertices."
        )

    sub_vertices, sub_faces, face_labels = build_representation_3(
        partition, mesh, active_vps, steiner_handler
    )

    vp_edges = np.empty((n_vp, 2), dtype=np.int64)
    vp_lambda = np.empty(n_vp, dtype=np.float64)
    for k, vp in enumerate(active_vps):
        vp_edges[k, 0] = vp.edge[0]
        vp_edges[k, 1] = vp.edge[1]
        vp_lambda[k] = vp.lambda_param

    triangle_index = np.empty(n_tp, dtype=np.int32)
    cell_triple = np.empty((n_tp, 3), dtype=np.int32)
    steiner_xyz = np.empty((n_tp, 3), dtype=np.float64)
    for k, tp in enumerate(steiner_handler.triple_points):
        triangle_index[k] = tp.triangle_idx
        if len(tp.cell_indices) != 3:
            raise ValueError(
                f"Triple point at triangle {tp.triangle_idx} has "
                f"{len(tp.cell_indices)} cells, expected 3"
            )
        cell_triple[k] = np.asarray(tp.cell_indices, dtype=np.int32)
        steiner_xyz[k] = tp.steiner_point

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    finalised = not pending_migration
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with h5py.File(output_path, "w") as f:
        f.attrs["schema_version"] = "1.1"
        f.attrs["surface"] = "torus"
        f.attrs["n_cells"] = n_cells
        f.attrs["R"] = R
        f.attrs["r"] = r
        f.attrs["finalised"] = bool(finalised)
        f.attrs["source_run_id"] = source_run_id
        f.attrs["source_iteration"] = int(source_iteration)
        f.attrs["seed"] = int(seed)
        f.attrs["final_perimeter"] = float(final_perimeter)
        f.attrs["surface_partition_git_sha"] = _git_sha()
        f.attrs["created"] = created

        mesh_grp = f.create_group("mesh")
        mesh_grp.create_dataset("vertices", data=mesh.vertices.astype(np.float64))
        mesh_grp.create_dataset("faces", data=mesh.faces.astype(np.int32))
        mesh_grp.attrs["grid_shape"] = np.array([n_theta, n_phi], dtype=np.int32)
        mesh_grp.attrs["vertex_order"] = (
            "theta-major row-major: vertex[i*n_phi + j] is at (theta_i, phi_j)"
        )

        part_grp = f.create_group("partition")
        part_grp.create_dataset("sub_vertices", data=sub_vertices)
        part_grp.create_dataset("sub_faces", data=sub_faces)
        part_grp.create_dataset("face_labels", data=face_labels)
        part_grp.attrs["n_original_vertices"] = V

        snap_grp = f.create_group("snapshot")
        snap_grp.create_dataset("vertex_labels", data=vertex_labels)
        snap_grp.create_dataset("vp_edges", data=vp_edges)
        snap_grp.create_dataset("vp_lambda", data=vp_lambda)
        snap_grp.attrs["n_variable_points"] = n_vp

        tp_grp = snap_grp.create_group("triple_points")
        tp_grp.create_dataset("triangle_index", data=triangle_index)
        tp_grp.create_dataset("cell_triple", data=cell_triple)
        tp_grp.create_dataset("steiner_xyz", data=steiner_xyz)

    logger.info(
        f"Exported partition to {output_path}: V={V}, n_vp={n_vp}, n_tp={n_tp}, "
        f"n_cells={n_cells}, finalised={finalised}"
    )
