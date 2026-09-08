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
from ..surfaces.factory import (
    is_structured,
    resolve_surface_params,
    surface_name_from_config,
)


logger = logging.getLogger(__name__)

#: Zero level set of each implicit surface, as a human-readable expression. The
#: generalisation of the torus's ``R``/``r``: ``f`` and ``grad f`` give the
#: on-surface residual check, exact normals, and projection onto the surface
#: alike. See docs/reference/PARTITION_EXPORT_SCHEMA_GENERAL.md §5.
_IMPLICIT_EXPR = {
    "double_torus": "(x*(x-1)**2*(x-2) + y**2)**2 + z**2 - c",
    "banchoff_chmutov": "T4(x) + T4(y) + T4(z),  T4(t) = 8*t**4 - 8*t**2 + 1",
}

#: Which config keys are the surface's own shape parameters, per surface.
_SURFACE_PARAM_KEYS = {
    "torus": ("R", "r"),
    "ellipsoid": ("a", "b", "c"),
    "double_torus": ("c",),
    "banchoff_chmutov": (),
}

#: Marching-cubes surfaces; ``voxel_size`` is meaningful only for these, and the
#: residual tolerance scales with it as ``K * h**2`` (spec §5.1).
_IMPLICIT_SURFACES = frozenset(_IMPLICIT_EXPR)


def _euler_characteristic(faces: np.ndarray) -> int:
    """``chi = V - E + F`` from the face list alone."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return int(faces.max() + 1) - int(e.shape[0]) + int(faces.shape[0])


def _provider_bbox(surface_name: str, surface_params: dict):
    """The provider's marching-cubes sampling box, ``((xmin,xmax), ...)``.

    The sampling box, not the mesh extent: the level set generally does not reach
    the box faces, so vertex extents would understate the voxel size.
    """
    from ..surfaces.factory import build_provider

    provider = build_provider({"surface": {surface_name: dict(surface_params)}})
    return provider.bounding_box()


def _write_surface_group(
    f, surface_name, surface_params, mesh, resolution, structured
) -> None:
    """Write the self-describing ``/surface`` group of schema 2.0.

    Genus is *computed* from the exported mesh rather than tabulated per surface:
    it doubles as a check that what we are exporting is a closed surface.
    """
    grp = f.create_group("surface")
    grp.attrs["name"] = surface_name
    is_implicit = surface_name in _IMPLICIT_SURFACES
    grp.attrs["kind"] = "implicit" if is_implicit else "parametric"
    grp.attrs["structured"] = bool(structured)

    chi = _euler_characteristic(np.asarray(mesh.faces))
    grp.attrs["euler_characteristic"] = int(chi)
    grp.attrs["genus"] = int((2 - chi) // 2)

    v = np.asarray(mesh.vertices)
    grp.attrs["bbox"] = np.stack([v.min(axis=0), v.max(axis=0)], axis=1).astype(
        np.float64
    )

    params = grp.create_group("params")
    for key in _SURFACE_PARAM_KEYS.get(surface_name, ()):
        if key in surface_params:
            params.attrs[key] = float(surface_params[key])

    if is_implicit:
        grp.attrs["implicit_expr"] = _IMPLICIT_EXPR[surface_name]
        # Voxel spacing along x. Two things this must NOT use: the config's
        # n_grid_x (that is the BASE level, not the refined one this mesh came
        # from), and the vertex extent (the level set does not reach the sampling
        # box). Take the FINAL resolution and the provider's own bounding box.
        n_gx = int(resolution[0])
        (xmin, xmax), _, _ = _provider_bbox(surface_name, surface_params)
        if n_gx > 1:
            grp.attrs["voxel_size"] = (xmax - xmin) / (n_gx - 1)

    if structured:
        grp.attrs["resolution"] = np.asarray(resolution, dtype=np.int32)


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
    force_finalised: bool = False,
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
    ``strict`` is set. Passing ``force_finalised=True`` overrides this and
    writes ``finalised=True`` regardless of ``pending_migration``, together
    with an explanatory ``finalised_note`` — the reproducible equivalent of
    hand-patching the attr when the Phase-2 refinement is stuck in the
    migration-cycling plateau and the best iterate is the accepted final
    result (see CLAUDE.md 'Phase 2 migration-cycling plateau'). ``strict`` and
    ``force_finalised`` are mutually exclusive.
    """
    if strict and force_finalised:
        raise ValueError("strict and force_finalised are mutually exclusive")

    finalised_note = None
    if pending_migration:
        if force_finalised:
            finalised_note = (
                f"Best iterate (iteration {int(source_iteration)}, perimeter "
                f"{float(final_perimeter):.3f}) exported with finalised=True via "
                f"force_finalised despite pending_migration=True. Typical cause: "
                f"the Phase-2 refinement reached the migration-cycling plateau "
                f"(perimeter oscillates on migrations with no further reduction, "
                f"so pending_migration never clears). The minimum-perimeter iterate "
                f"is the accepted final deliverable. See CLAUDE.md 'Phase 2 "
                f"migration-cycling plateau'."
            )
            msg = (
                "Checkpoint has a pending topology migration, but force_finalised "
                "is set: writing finalised=True (accepted migration-cycling "
                "plateau best iterate)."
            )
            logger.warning(msg)
            print(f"NOTE: {msg}")
        else:
            msg = (
                "This checkpoint has a pending topology migration; the partition "
                "state may shift in the next iteration. Exporting with "
                "finalised=False."
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            print(f"WARNING: {msg}")

    surface_name = surface_name_from_config(config)
    surface_params = resolve_surface_params(config, surface_name)
    structured = is_structured(surface_name)
    n_theta = int(n_theta_final)
    n_phi = int(n_phi_final)
    if surface_name == "torus":
        R = float(surface_params["R"])
        r = float(surface_params["r"])

    active_vps = [vp for vp in partition.variable_points if vp.active]
    n_vp = len(active_vps)
    n_tp = len(steiner_handler.triple_points)
    n_cells = int(partition.n_cells)

    vertex_labels = partition.indicator_functions.argmax(axis=1).astype(np.int32)
    V = mesh.vertices.shape[0]

    # The product identity is a property of a STRUCTURED mesh, not of the schema.
    # Enforced for the torus -- the surface the 1.1 contract is written for -- and
    # skipped where it cannot hold: on a marching-cubes mesh the resolution pair is
    # a sampling grid and V is whatever the level set intersects.
    if structured and n_theta * n_phi != V:
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

    finalised = force_finalised or (not pending_migration)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with h5py.File(output_path, "w") as f:
        # Version namespace is FORKED, not bumped: the torus keeps writing 1.1
        # byte-identically so the downstream reader -- which validates
        # schema_version == "1.1" AND surface == "torus" -- cannot be affected.
        # It refuses a 2.0 file at its first check, which is correct behaviour.
        # See docs/reference/PARTITION_EXPORT_SCHEMA_GENERAL.md.
        f.attrs["schema_version"] = "1.1" if surface_name == "torus" else "2.0"
        f.attrs["surface"] = surface_name
        f.attrs["n_cells"] = n_cells
        if surface_name == "torus":
            f.attrs["R"] = R
            f.attrs["r"] = r
        f.attrs["finalised"] = bool(finalised)
        if finalised_note is not None:
            f.attrs["finalised_note"] = finalised_note
        f.attrs["source_run_id"] = source_run_id
        f.attrs["source_iteration"] = int(source_iteration)
        f.attrs["seed"] = int(seed)
        f.attrs["final_perimeter"] = float(final_perimeter)
        f.attrs["surface_partition_git_sha"] = _git_sha()
        f.attrs["created"] = created

        mesh_grp = f.create_group("mesh")
        mesh_grp.create_dataset("vertices", data=mesh.vertices.astype(np.float64))
        mesh_grp.create_dataset("faces", data=mesh.faces.astype(np.int32))
        if surface_name == "torus":
            mesh_grp.attrs["grid_shape"] = np.array([n_theta, n_phi], dtype=np.int32)
            mesh_grp.attrs["vertex_order"] = (
                "theta-major row-major: vertex[i*n_phi + j] is at (theta_i, phi_j)"
            )
        else:
            _write_surface_group(
                f, surface_name, surface_params, mesh, (n_theta, n_phi), structured
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
