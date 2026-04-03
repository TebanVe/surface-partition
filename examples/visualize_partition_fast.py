#!/usr/bin/env python3
"""
Visualize Partition Results — Fast Renderer

Drop-in replacement for visualize_partition.py with three key optimizations:

  1. Interior triangles assembled via NumPy fancy indexing (vectorized, not a
     Python loop), eliminating ~490 k Python iterations for a 506 k-face mesh.
  2. The tri_idx_to_segment lookup dict is built ONCE before the per-cell loop
     instead of being rebuilt for every cell.
  3. show_edges defaults to False (toggle with --show-edges).

Boundary clipping (compute_cell_portion_in_triangle_simple) and Steiner-point
handling are intentionally unchanged — cell boundaries remain geometrically
precise.

The script scales to any number of cells; a progress line is printed per cell
so long runs stay observable.

Usage (base solution):
    python examples/visualize_partition_fast.py \\
        --solution path/to/solution.h5

Usage (refined contours, highlight region 2):
    python examples/visualize_partition_fast.py \\
        --solution path/to/*_refined_contours.h5 \\
        --region 2 \\
        --show-steiner \\
        --vp-size 0.0004 \\
        --steiner-size 0.0008 \\
        --opacity 1

Author: Partition Visualization — fast renderer
Date: March 2026
"""

import os
import sys
import re
import time
import argparse
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista is required.  Install with: pip install pyvista")
    sys.exit(1)

from src.mesh.tri_mesh import TriMesh
from src.partition.contour_partition import PartitionContour
from src.partition.steiner_handler import SteinerHandler
from src.partition.area_calculator import AreaCalculator
from src.partition.perimeter_calculator import PerimeterCalculator

from src.visualization.partition_helpers import (
    compute_cell_portion_in_triangle_simple,
    compute_triple_point_cell_portion,
    add_steiner_visualization,
    add_vp_visualization,
)
from src.pipeline.io import (
    load_partition_from_refined_file,
    load_partition_from_base_file,
)

# ---------------------------------------------------------------------------
# Colour palette — cycles automatically for any number of cells
# ---------------------------------------------------------------------------
_PALETTE = [
    '#FFE5B4', '#E0BBE4', '#FFDAC1', '#B5EAD7', '#C7CEEA',
    '#FFB7B2', '#FFDFD3', '#E2F0CB', '#B4F8C8', '#A0C4FF',
    '#FFC6FF', '#FFCFD2', '#FDE2E4', '#FAD2E1', '#BEE1E6',
    '#D4E6F1', '#D5F5E3', '#FDEBD0', '#F9EBEA', '#EAF2F8',
]


# ---------------------------------------------------------------------------
# File detection helpers (shared with visualize_partition.py)
# ---------------------------------------------------------------------------

def _is_refined_file(path: str) -> bool:
    if 'refined_contours' in os.path.basename(path):
        return True
    try:
        with h5py.File(path, 'r') as f:
            return 'lambda_parameters' in f
    except Exception:
        return False


def _get_iteration_number(path: str) -> Optional[int]:
    try:
        with h5py.File(path, 'r') as f:
            val = f.attrs.get('iteration_number')
            if val is not None:
                return int(val)
    except Exception:
        pass
    m = re.search(r'_iteration(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def _build_source_label(is_refined: bool, solution_path: str) -> str:
    if not is_refined:
        return "Base"
    iter_num = _get_iteration_number(solution_path)
    return f"Refined - iter {iter_num}" if iter_num is not None else "Refined"


# ---------------------------------------------------------------------------
# Smart loader (identical logic to visualize_partition.py)
# ---------------------------------------------------------------------------

def load_partition_smart(solution_path: str, use_initial: bool = False,
                         verbose: bool = False):
    refined = _is_refined_file(solution_path)
    if refined:
        if verbose:
            print("Detected refined contours file")
        mesh, partition = load_partition_from_refined_file(solution_path, verbose=verbose)
    else:
        if verbose:
            print("Detected base solution file")
        mesh, partition = load_partition_from_base_file(
            solution_path, use_initial=use_initial, verbose=verbose
        )
    if verbose:
        print(f"  ✓ Successfully loaded partition with {len(partition.variable_points)} VPs")
    return mesh, partition, refined


# ---------------------------------------------------------------------------
# Fast per-cell renderer
# ---------------------------------------------------------------------------

def _render_region_fast(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    cell_idx: int,
    color: str,
    tri_idx_to_segment: Dict,
    opacity: float = 1.0,
    show_edges: bool = False,
    backface_culling: bool = False,
) -> None:
    """
    Render one cell with precise boundaries.

    Interior triangles are assembled with a single NumPy fancy-index operation.
    Boundary clipping and Steiner handling are kept exactly as in the original
    render_single_region_simple.
    """
    all_verts_parts = []   # list of (K, 3) float arrays
    all_faces_parts = []   # list of flat int arrays (PyVista face format)
    vertex_offset = 0

    # ------------------------------------------------------------------
    # 1. Interior triangles — vectorized (no Python loop over each face)
    # ------------------------------------------------------------------
    interior_tris = area_calc.cell_interior_triangles.get(cell_idx, [])
    if interior_tris:
        int_arr = np.asarray(interior_tris, dtype=int)   # (N,)
        faces_3 = mesh.faces[int_arr]                    # (N, 3)
        verts_int = mesh.vertices[faces_3].reshape(-1, 3)  # (3N, 3)
        n = len(int_arr)

        # PyVista face format: [3, v0, v1, v2,  3, v3, v4, v5, ...]
        base = np.arange(n, dtype=np.int64) * 3
        int_faces = np.empty(4 * n, dtype=np.int64)
        int_faces[0::4] = 3
        int_faces[1::4] = base
        int_faces[2::4] = base + 1
        int_faces[3::4] = base + 2

        all_verts_parts.append(verts_int)
        all_faces_parts.append(int_faces)
        vertex_offset = 3 * n

    # ------------------------------------------------------------------
    # 2. Boundary triangles — exact clipping (Python loop, ~1-2 k/cell)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3. Triple-point (Steiner) triangles — exact, small count
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Assemble single PolyData and add to plotter
    # ------------------------------------------------------------------
    if not all_verts_parts:
        return

    all_vertices = np.vstack(all_verts_parts)
    all_faces = np.concatenate(all_faces_parts)

    region_mesh = pv.PolyData(all_vertices, faces=all_faces)
    plotter.add_mesh(
        region_mesh,
        color=color,
        opacity=opacity,
        show_edges=show_edges,
        edge_color='lightgray',
        line_width=0.5,
        backface_culling=backface_culling,
    )


# ---------------------------------------------------------------------------
# Multi-cell renderer
# ---------------------------------------------------------------------------

def render_all_regions_fast(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    target_region: Optional[int] = None,
    target_color: str = '#FF8C42',
    opacity: float = 1.0,
    show_edges: bool = False,
) -> None:
    """
    Render all cells using the fast renderer.

    The tri_idx_to_segment lookup is built here ONCE and reused across all
    cells instead of being rebuilt inside each per-cell call.
    """
    n_cells = partition.n_cells
    print(f"  Pre-building segment lookup for {len(partition.triangle_segments)} segments...")
    t0 = time.perf_counter()

    # Build once — shared across all cells
    tri_idx_to_segment: Dict = {
        ts.triangle_idx: ts for ts in partition.triangle_segments
    }

    t1 = time.perf_counter()
    print(f"  ✓ Segment lookup ready ({t1 - t0:.2f}s)")
    print(f"  Rendering {n_cells} regions...")

    for cell_idx in range(n_cells):
        t_cell = time.perf_counter()

        if target_region is not None and cell_idx == target_region:
            color = target_color
            cell_opacity = 1.0
        else:
            color = _PALETTE[cell_idx % len(_PALETTE)]
            cell_opacity = opacity

        _render_region_fast(
            plotter, mesh, partition, area_calc, steiner_handler,
            cell_idx, color, tri_idx_to_segment,
            opacity=cell_opacity,
            show_edges=show_edges,
        )

        elapsed = time.perf_counter() - t_cell
        int_count = len(area_calc.cell_interior_triangles.get(cell_idx, []))
        bnd_count = len(area_calc.cell_boundary_triangles.get(cell_idx, []))
        print(f"    Cell {cell_idx:>3d}: {int_count:>7,} interior + {bnd_count:>5,} boundary  "
              f"({elapsed:.2f}s)")

    print(f"  ✓ Region rendering complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualize partition results — fast renderer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--solution', required=True,
                        help='Path to .h5 file (base solution or refined_contours)')
    parser.add_argument('--use-initial', action='store_true',
                        help='Use initial condition (x0) instead of optimized result')

    parser.add_argument('--region', type=int, default=None,
                        help='Region/cell index to highlight (optional)')

    parser.add_argument('--show-vps', action='store_true',
                        help='Show all variable points')
    parser.add_argument('--show-steiner', action='store_true',
                        help='Show Steiner points and void triangles')
    parser.add_argument('--show-edges', action='store_true',
                        help='Render mesh edges (slow on large meshes; off by default)')
    parser.add_argument('--intense-color', default='#FF8C42',
                        help='Highlight colour for --region (default: #FF8C42)')

    parser.add_argument('--vp-size', type=float, default=0.0005,
                        help='VP marker radius (default: 0.0005)')
    parser.add_argument('--steiner-size', type=float, default=0.000005,
                        help='Steiner point marker radius (default: 0.000005)')
    parser.add_argument('--opacity', type=float, default=1.0,
                        help='Cell opacity, 0–1 (default: 1.0)')

    parser.add_argument('--apply-zoom', action='store_true',
                        help='Zoom camera onto the highlighted region')
    parser.add_argument('--zoom-factor', type=float, default=0.1,
                        help='Zoom factor — smaller means closer (default: 0.1)')

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    t_start = time.perf_counter()

    print("=" * 80)
    print("PARTITION VISUALIZATION  (fast renderer)")
    print("=" * 80)
    print(f"Loading: {args.solution}")

    if not os.path.exists(args.solution):
        print(f"ERROR: File not found: {args.solution}")
        return 1

    try:
        mesh, partition, is_refined = load_partition_smart(
            args.solution, use_initial=args.use_initial, verbose=True
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    source_label = _build_source_label(is_refined, args.solution)
    active_vps = sum(1 for vp in partition.variable_points if vp.active)
    print(f"✓ Loaded partition ({source_label}): {partition.n_cells} cells, "
          f"{active_vps} active VPs ({len(partition.variable_points)} total)")

    # -----------------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------------
    print("\nInitializing handlers...")
    area_calc = AreaCalculator(mesh, partition)
    steiner_handler = SteinerHandler(mesh, partition)
    print(f"✓ Found {len(steiner_handler.triple_points)} triple points")

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    lambda_vec = partition.get_variable_vector()
    perim_calc = PerimeterCalculator(mesh, partition)
    regular_perimeter = perim_calc.compute_total_perimeter(lambda_vec)
    steiner_perimeter = steiner_handler.get_total_perimeter_contribution()
    total_perimeter = regular_perimeter + steiner_perimeter
    total_area = float(mesh.M.sum())
    target_area = total_area / partition.n_cells

    print(f"\nMetrics:")
    print(f"  Total perimeter : {total_perimeter:.6f}")
    print(f"    Regular: {regular_perimeter:.6f}  |  Steiner: {steiner_perimeter:.6f}")
    print(f"  Total surface area : {total_area:.6f}")
    print(f"  Target area/cell   : {target_area:.6f}")

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------
    print("\nCreating visualization...")
    plotter = pv.Plotter()

    t_render = time.perf_counter()
    render_all_regions_fast(
        plotter, mesh, partition, area_calc, steiner_handler,
        target_region=args.region,
        target_color=args.intense_color,
        opacity=args.opacity,
        show_edges=args.show_edges,
    )
    print(f"  Render build time: {time.perf_counter() - t_render:.2f}s")

    if args.show_steiner:
        add_steiner_visualization(plotter, steiner_handler, partition, mesh,
                                  args.steiner_size)
        print(f"  Adding {len(steiner_handler.triple_points)} Steiner points and void triangles...")

    if args.show_vps:
        all_vp_indices = [i for i, vp in enumerate(partition.variable_points) if vp.active]
        vp_colors = ['red'] * len(all_vp_indices)
        vp_labels = [f'VP{i}' for i in all_vp_indices]
        add_vp_visualization(plotter, partition, mesh, all_vp_indices,
                             vp_colors, vp_labels, args.vp_size)

    if args.apply_zoom and args.region is not None:
        interior_tris = area_calc.cell_interior_triangles.get(args.region, [])
        boundary_tris = area_calc.cell_boundary_triangles.get(args.region, [])
        all_tris = interior_tris + boundary_tris
        if all_tris:
            tris_arr = np.asarray(all_tris, dtype=int)
            faces_3 = mesh.faces[tris_arr]
            centroids = mesh.vertices[faces_3].mean(axis=1)
            focus_point = centroids.mean(axis=0)
            camera_offset = np.array([args.zoom_factor,
                                      args.zoom_factor * 0.5,
                                      args.zoom_factor * 0.5])
            plotter.camera_position = [
                focus_point + camera_offset,
                focus_point,
                (0, 0, 1),
            ]
            plotter.camera.clipping_range = (
                args.zoom_factor * 0.1,
                args.zoom_factor * 10,
            )

    title_parts = [f"Partition ({source_label}): {partition.n_cells} cells"]
    if args.region is not None:
        title_parts.append(f"Region {args.region}")
    title_parts.append(f"Perimeter: {total_perimeter:.4f}")
    title_parts.append(f"Target area/cell: {target_area:.4f}")
    plotter.add_title(" | ".join(title_parts), font_size=12)

    total_elapsed = time.perf_counter() - t_start
    print(f"\n✓ Visualization ready  (total setup: {total_elapsed:.2f}s)")
    print("=" * 80)

    plotter.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())
