#!/usr/bin/env python3
"""
Visualize Partition Results (Simple Cell Viewer)

Simple visualization for viewing partition states without migration logic.

Usage:
    python examples/visualize_partition.py \
        --solution path/to/*_refined_contours.h5 \
        --region 2 \
        --show-steiner \
        --vp-size 0.0004 \
        --steiner-size 0.0008

Author: Partition Visualization
Date: February 2026
"""

import os
import sys
import argparse
import numpy as np
import re
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista is required. Install with: pip install pyvista")
    sys.exit(1)

from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.core.steiner_handler import SteinerHandler
from src.core.area_calculator import AreaCalculator

# Import working functions from existing script
from examples.visualize_type2_triple_point import (
    render_single_region_simple,
    add_steiner_visualization,
    add_vp_visualization
)
from examples.data_loader import load_partition_from_refined_file


def load_partition_smart(refined_path, verbose=False):
    """
    Load partition from refined contours file.
    
    This function properly delegates to load_partition_from_refined_file which:
    - For original refined files: Reconstructs from base solution + lambda params
    - For iteration files: Uses stored partition state (handles changed VP count after migrations)
    
    The key insight: Iteration files store FULL partition state, not just lambda parameters.
    """
    if verbose:
        print(f"Loading from refined contours file...")
        print(f"  Refined: {refined_path}")
    
    try:
        # Use the proper loader that handles all cases correctly
        # This loader knows how to handle iteration files with changed VP counts
        mesh, partition = load_partition_from_refined_file(refined_path, verbose=verbose)
        
        if verbose:
            print(f"  ✓ Successfully loaded partition with {len(partition.variable_points)} VPs")
        
        return mesh, partition
    
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
                       help='Path to refined_contours.h5 file')
    
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
        mesh, partition = load_partition_smart(args.solution, verbose=True)
    except Exception as e:
        print(f"ERROR: Failed to load partition: {e}")
        return 1
    
    print(f"✓ Loaded partition: {partition.n_cells} cells, {len(partition.variable_points)} VPs")
    
    # ========================================================================
    # Initialize Handlers
    # ========================================================================
    
    print("\nInitializing handlers...")
    area_calc = AreaCalculator(mesh, partition)
    steiner_handler = SteinerHandler(mesh, partition)
    print(f"✓ Found {len(steiner_handler.triple_points)} triple points")
    
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
        all_vp_indices = list(range(len(partition.variable_points)))
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
    title_parts = [f"Partition: {partition.n_cells} cells"]
    if args.region is not None:
        title_parts.append(f"Region {args.region}")
    
    plotter.add_title(" | ".join(title_parts), font_size=14)
    
    print("✓ Visualization ready")
    print("="*80)
    
    # Show visualization
    plotter.show()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
