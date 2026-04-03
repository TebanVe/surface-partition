#!/usr/bin/env python3
"""
Visualize Partition Results (Simple Cell Viewer)

Simple visualization for viewing partition states without migration logic.
Accepts both base solution .h5 files and refined_contours .h5 files.

Usage (refined contours):
    python examples/visualize_partition.py \
        --solution path/to/*_refined_contours.h5 \
        --region 2 \
        --show-steiner \
        --vp-size 0.0004 \
        --steiner-size 0.0008

Usage (base solution):
    python examples/visualize_partition.py \
        --solution path/to/solution.h5

Author: Partition Visualization
Date: February 2026
"""

import os
import sys
import argparse
import numpy as np
import re
import h5py
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista is required. Install with: pip install pyvista")
    sys.exit(1)

from src.mesh.tri_mesh import TriMesh
from src.partition.contour_partition import PartitionContour
from src.partition.steiner_handler import SteinerHandler
from src.partition.area_calculator import AreaCalculator
from src.partition.perimeter_calculator import PerimeterCalculator

from src.visualization.partition_helpers import (
    render_single_region_simple,
    add_steiner_visualization,
    add_vp_visualization,
)
from src.pipeline.io import load_partition_from_refined_file, load_partition_from_base_file


def _is_refined_file(path):
    """Detect whether an .h5 file is a refined contours file or a base solution."""
    if 'refined_contours' in os.path.basename(path):
        return True
    try:
        with h5py.File(path, 'r') as f:
            return 'lambda_parameters' in f
    except Exception:
        return False


def _get_iteration_number(path):
    """Extract iteration number from a refined contours file.

    Checks HDF5 attrs first, then falls back to parsing the filename.
    Returns None for base solution files or when no iteration is identified.
    """
    try:
        with h5py.File(path, 'r') as f:
            iter_num = f.attrs.get('iteration_number')
            if iter_num is not None:
                return int(iter_num)
    except Exception:
        pass

    match = re.search(r'_iteration(\d+)', os.path.basename(path))
    if match:
        return int(match.group(1))

    return None


def _build_source_label(is_refined, solution_path):
    """Build a human-readable source label for the plot title."""
    if not is_refined:
        return "Base"
    iter_num = _get_iteration_number(solution_path)
    if iter_num is not None:
        return f"Refined - iter {iter_num}"
    return "Refined"


def load_partition_smart(solution_path, use_initial=False, verbose=False):
    """
    Load partition from either a base solution or refined contours file.

    Detection logic:
    - If the filename contains 'refined_contours' or the file has
      'lambda_parameters', delegate to load_partition_from_refined_file.
    - Otherwise treat it as a base solution file.

    Returns:
        tuple: (mesh, partition, is_refined) -- the boolean indicates which
               loader was used so callers can adjust titles / messages.
    """
    refined = _is_refined_file(solution_path)

    try:
        if refined:
            if verbose:
                print(f"Detected refined contours file")
            mesh, partition = load_partition_from_refined_file(solution_path, verbose=verbose)
        else:
            if verbose:
                print(f"Detected base solution file")
            mesh, partition = load_partition_from_base_file(
                solution_path, use_initial=use_initial, verbose=verbose
            )

        if verbose:
            print(f"  ✓ Successfully loaded partition with {len(partition.variable_points)} VPs")

        return mesh, partition, refined

    except Exception as e:
        raise RuntimeError(f"Failed to load partition: {str(e)}")


def render_regions_opaque(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    target_region: Optional[int] = None,
    target_color: str = '#FF8C42',
    opacity: float = 1.0
):
    """Render all regions with higher opacity for better depth perception."""
    # Pastel color palette (matches visualize_type1_vertex_collapse.py)
    pale_palette = [
        '#FFE5B4', '#E0BBE4', '#FFDAC1', '#B5EAD7', '#C7CEEA',
        '#FFB7B2', '#FFDFD3', '#E2F0CB', '#B4F8C8', '#A0C4FF',
        '#FFC6FF', '#FFCFD2', '#FDE2E4', '#FAD2E1', '#BEE1E6'
    ]
    
    print(f"  Rendering all {partition.n_cells} regions (opacity: {opacity})...")
    
    for cell_idx in range(partition.n_cells):
        if target_region is not None and cell_idx == target_region:
            color = target_color
            cell_opacity = 1.0
        else:
            color = pale_palette[cell_idx % len(pale_palette)]
            cell_opacity = opacity
        
        render_single_region_simple(
            plotter, mesh, partition, area_calc, steiner_handler,
            cell_idx, color, opacity=cell_opacity, backface_culling=False
        )
    
    print(f"  ✓ Region rendering complete")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize partition results (simple cell viewer)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--solution', required=True,
                       help='Path to .h5 file (base solution or refined_contours)')
    parser.add_argument('--use-initial', action='store_true',
                       help='Use initial condition (x0) instead of optimized result (base solution only)')
    
    # Optional highlighting
    parser.add_argument('--region', type=int, default=None,
                       help='Region/cell index to highlight (optional)')
    
    # Visualization options
    parser.add_argument('--show-vps', action='store_true',
                       help='Show all variable points')
    parser.add_argument('--show-steiner', action='store_true',
                       help='Show Steiner points and void triangles')
    parser.add_argument('--intense-color', default='#FF8C42',
                       help='Color for highlighted region (default: #FF8C42)')
    
    # Size parameters
    parser.add_argument('--vp-size', type=float, default=0.0005,
                       help='Size of VP markers (default: 0.0005)')
    parser.add_argument('--steiner-size', type=float, default=0.000005,
                       help='Size of Steiner point markers (default: 0.000005)')
    parser.add_argument('--opacity', type=float, default=1.0,
                       help='Opacity of cell colors (0.0-1.0, default: 1.0 for stronger colors)')
    
    # Camera control
    parser.add_argument('--apply-zoom', action='store_true',
                       help='Apply camera zoom to highlighted region')
    parser.add_argument('--zoom-factor', type=float, default=0.1,
                       help='Zoom factor (smaller = closer, default: 0.1)')
    
    args = parser.parse_args()
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("="*80)
    print("PARTITION VISUALIZATION")
    print("="*80)
    print(f"Loading: {args.solution}")
    
    if not os.path.exists(args.solution):
        print(f"ERROR: Solution file not found: {args.solution}")
        return 1
    
    try:
        mesh, partition, is_refined = load_partition_smart(
            args.solution, use_initial=args.use_initial, verbose=True
        )
    except Exception as e:
        print(f"ERROR: Failed to load partition: {e}")
        return 1
    
    source_label = _build_source_label(is_refined, args.solution)
    active_vps = sum(1 for vp in partition.variable_points if vp.active)
    print(f"✓ Loaded partition ({source_label}): {partition.n_cells} cells, "
          f"{active_vps} VPs ({len(partition.variable_points)} total)")
    
    # ========================================================================
    # Initialize Handlers
    # ========================================================================
    
    print("\nInitializing handlers...")
    area_calc = AreaCalculator(mesh, partition)
    steiner_handler = SteinerHandler(mesh, partition)
    print(f"✓ Found {len(steiner_handler.triple_points)} triple points")

    # ========================================================================
    # Compute Metrics
    # ========================================================================

    lambda_vec = partition.get_variable_vector()
    perim_calc = PerimeterCalculator(mesh, partition)
    regular_perimeter = perim_calc.compute_total_perimeter(lambda_vec)
    steiner_perimeter = steiner_handler.get_total_perimeter_contribution()
    total_perimeter = regular_perimeter + steiner_perimeter

    total_area = float(mesh.M.sum())
    target_area = total_area / partition.n_cells

    print(f"\nMetrics:")
    print(f"  Total perimeter: {total_perimeter:.6f}")
    print(f"    Regular: {regular_perimeter:.6f}  |  Steiner: {steiner_perimeter:.6f}")
    print(f"  Total surface area: {total_area:.6f}")
    print(f"  Target area/cell:   {target_area:.6f}")
    
    # ========================================================================
    # Create Visualization
    # ========================================================================
    
    print("\nCreating visualization...")
    plotter = pv.Plotter()
    
    # Render all cells (uses working function from visualize_type2_triple_point.py)
    render_regions_opaque(
        plotter, mesh, partition, area_calc, steiner_handler,
        target_region=args.region,
        target_color=args.intense_color,
        opacity=args.opacity
    )
    
    # Add Steiner visualization if requested
    if args.show_steiner:
        add_steiner_visualization(plotter, steiner_handler, partition, mesh, args.steiner_size)
    
    # Add VP visualization if requested
    if args.show_vps:
        all_vp_indices = [i for i, vp in enumerate(partition.variable_points) if vp.active]
        vp_colors = ['red'] * len(all_vp_indices)
        vp_labels = [f'VP{i}' for i in all_vp_indices]
        add_vp_visualization(plotter, partition, mesh, all_vp_indices, vp_colors, vp_labels, args.vp_size)
    
    # Apply camera zoom if requested
    if args.apply_zoom and args.region is not None:
        # Find centroid of highlighted region
        interior_tris = area_calc.cell_interior_triangles.get(args.region, [])
        boundary_tris = area_calc.cell_boundary_triangles.get(args.region, [])
        all_tris = interior_tris + boundary_tris
        
        if all_tris:
            centroids = []
            for tri_idx in all_tris:
                face = mesh.faces[tri_idx]
                tri_vertices = mesh.vertices[face]
                centroids.append(np.mean(tri_vertices, axis=0))
            
            focus_point = np.mean(centroids, axis=0)
            camera_offset = np.array([args.zoom_factor, args.zoom_factor * 0.5, args.zoom_factor * 0.5])
            camera_position = focus_point + camera_offset
            
            plotter.camera_position = [camera_position, focus_point, (0, 0, 1)]
            plotter.camera.clipping_range = (args.zoom_factor * 0.1, args.zoom_factor * 10)
    
    # Set title
    title_parts = [f"Partition ({source_label}): {partition.n_cells} cells"]
    if args.region is not None:
        title_parts.append(f"Region {args.region}")
    title_parts.append(f"Perimeter: {total_perimeter:.4f}")
    title_parts.append(f"Target area/cell: {target_area:.4f}")

    plotter.add_title(" | ".join(title_parts), font_size=12)
    
    print("✓ Visualization ready")
    print("="*80)
    
    # Show visualization
    plotter.show()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
