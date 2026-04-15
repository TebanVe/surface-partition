"""Offscreen partition rendering for automated analysis.

Renders partition solutions from multiple camera angles and saves as PNG
screenshots.  Requires PyVista; functions return empty lists if it is not
installed so callers can skip silently.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

CAMERA_ANGLES: Dict[str, dict] = {
    "front": {"position": (0, -3, 0), "viewup": (0, 0, 1)},
    "side": {"position": (3, 0, 0), "viewup": (0, 0, 1)},
    "top": {"position": (0, 0, 3), "viewup": (0, 1, 0)},
    "oblique": {"position": (2, -2, 1.5), "viewup": (0, 0, 1)},
}

_PALETTE = [
    "#FFE5B4", "#E0BBE4", "#FFDAC1", "#B5EAD7", "#C7CEEA",
    "#FFB7B2", "#FFDFD3", "#E2F0CB", "#B4F8C8", "#A0C4FF",
    "#FFC6FF", "#FFCFD2", "#FDE2E4", "#FAD2E1", "#BEE1E6",
    "#D4E6F1", "#D5F5E3", "#FDEBD0", "#F9EBEA", "#EAF2F8",
]


def render_partition_screenshots(
    solution_path: str,
    output_dir: str,
    angles: Optional[Dict[str, dict]] = None,
    window_size: tuple = (1200, 900),
) -> List[str]:
    """Render partition from multiple camera angles and save as PNGs.

    Args:
        solution_path: Path to .h5 solution file (base or refined).
        output_dir: Directory to save screenshot PNGs.
        angles: Camera angle dict (defaults to CAMERA_ANGLES).
        window_size: Render resolution (width, height).

    Returns:
        List of saved file paths, empty if PyVista is not available or
        rendering fails.
    """
    try:
        import pyvista as pv
    except ImportError:
        logger.debug("PyVista not installed — skipping partition screenshots")
        return []

    # Skip rendering in environments where OpenGL is unavailable (headless
    # servers, sandboxes) to avoid a potential segfault from VTK.
    import os as _os

    if _os.environ.get("DISPLAY") == "" or (
        sys.platform == "linux" and "DISPLAY" not in _os.environ
    ):
        logger.debug(
            "No DISPLAY set — skipping offscreen screenshots to avoid segfault"
        )
        return []

    # Quick sanity check: verify VTK can create an offscreen context.
    try:
        _test = pv.Plotter(off_screen=True, window_size=(64, 64))
        _test.close()
        del _test
    except Exception:
        logger.debug("PyVista offscreen probe failed — skipping screenshots")
        return []

    angles = angles or CAMERA_ANGLES
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        mesh, partition, area_calc, steiner_handler = _load_partition_data(
            solution_path
        )
    except Exception as exc:
        logger.warning(f"Could not load partition for screenshots: {exc}")
        return []

    # Compute bounding box for camera distance scaling
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_extent = np.linalg.norm(bbox_max - bbox_min)
    cam_distance = bbox_extent * 1.8

    n_cells = partition.n_cells
    tri_idx_to_segment = {
        ts.triangle_idx: ts for ts in partition.triangle_segments
    }

    saved: List[str] = []
    for name, cam in angles.items():
        try:
            plotter = pv.Plotter(off_screen=True, window_size=window_size)

            _add_partition_mesh(
                plotter, mesh, partition, area_calc, steiner_handler,
                n_cells, tri_idx_to_segment,
            )

            # Scale camera position relative to bounding box
            direction = np.array(cam["position"], dtype=float)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            cam_pos = bbox_center + direction * cam_distance
            viewup = cam["viewup"]
            plotter.camera_position = [
                cam_pos.tolist(),
                bbox_center.tolist(),
                list(viewup),
            ]

            out_path = str(Path(output_dir) / f"partition_{name}.png")
            plotter.screenshot(out_path)
            plotter.close()
            saved.append(out_path)
        except Exception as exc:
            logger.warning(f"Screenshot '{name}' failed: {exc}")

    return saved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_partition_data(solution_path: str):
    """Load mesh, partition, area calculator, and Steiner handler."""
    import os
    import h5py

    from ..mesh.tri_mesh import TriMesh
    from ..partition.find_contours import ContourAnalyzer
    from ..partition.contour_partition import PartitionContour
    from ..partition.area_calculator import AreaCalculator
    from ..partition.steiner_handler import SteinerHandler
    from ..pipeline.io import load_partition_from_refined_file, load_partition_from_base_file

    # Detect base vs refined
    is_refined = False
    try:
        with h5py.File(solution_path, "r") as f:
            is_refined = "lambda_parameters" in f
    except Exception:
        pass

    if is_refined:
        mesh, partition = load_partition_from_refined_file(solution_path)
    else:
        mesh, partition = load_partition_from_base_file(solution_path)

    area_calc = AreaCalculator(mesh, partition)
    steiner_handler = SteinerHandler(mesh, partition)

    return mesh, partition, area_calc, steiner_handler


def _add_partition_mesh(
    plotter,
    mesh,
    partition,
    area_calc,
    steiner_handler,
    n_cells: int,
    tri_idx_to_segment: dict,
) -> None:
    """Build and add colored partition cells to a PyVista plotter."""
    import pyvista as pv

    from .partition_helpers import (
        compute_cell_portion_in_triangle_simple,
        compute_triple_point_cell_portion,
    )

    for cell_idx in range(n_cells):
        all_verts_parts = []
        all_faces_parts = []
        vertex_offset = 0

        # Interior triangles (vectorized)
        interior_tris = area_calc.cell_interior_triangles.get(cell_idx, [])
        if interior_tris:
            int_arr = np.asarray(interior_tris, dtype=int)
            faces_3 = mesh.faces[int_arr]
            verts_int = mesh.vertices[faces_3].reshape(-1, 3)
            n = len(int_arr)
            base = np.arange(n, dtype=np.int64) * 3
            int_faces = np.empty(4 * n, dtype=np.int64)
            int_faces[0::4] = 3
            int_faces[1::4] = base
            int_faces[2::4] = base + 1
            int_faces[3::4] = base + 2
            all_verts_parts.append(verts_int)
            all_faces_parts.append(int_faces)
            vertex_offset = 3 * n

        # Boundary triangles
        boundary_tris = area_calc.cell_boundary_triangles.get(cell_idx, [])
        for tri_idx in boundary_tris:
            poly_verts = compute_cell_portion_in_triangle_simple(
                mesh, partition, tri_idx, cell_idx, tri_idx_to_segment
            )
            if poly_verts is None:
                continue
            nv = len(poly_verts)
            all_verts_parts.append(poly_verts)
            face_entry = np.empty(1 + nv, dtype=np.int64)
            face_entry[0] = nv
            face_entry[1:] = np.arange(vertex_offset, vertex_offset + nv)
            all_faces_parts.append(face_entry)
            vertex_offset += nv

        # Triple-point (Steiner) triangles
        for tp in steiner_handler.triple_points:
            if cell_idx not in tp.cell_indices:
                continue
            polygons = compute_triple_point_cell_portion(
                mesh, partition, steiner_handler, tp.triangle_idx, cell_idx
            )
            if polygons is None:
                continue
            for poly_verts in polygons:
                nv = len(poly_verts)
                all_verts_parts.append(poly_verts)
                face_entry = np.empty(1 + nv, dtype=np.int64)
                face_entry[0] = nv
                face_entry[1:] = np.arange(vertex_offset, vertex_offset + nv)
                all_faces_parts.append(face_entry)
                vertex_offset += nv

        if not all_verts_parts:
            continue

        all_vertices = np.vstack(all_verts_parts)
        all_faces = np.concatenate(all_faces_parts)
        color = _PALETTE[cell_idx % len(_PALETTE)]
        region_mesh = pv.PolyData(all_vertices, faces=all_faces)
        plotter.add_mesh(region_mesh, color=color, opacity=1.0, show_edges=False)
