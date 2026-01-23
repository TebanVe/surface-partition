#!/usr/bin/env python3
"""
Visualize Type 2 migration (Triple Point Migration).

Clean, focused implementation for testing and debugging Type 2 strategy.
Only includes necessary code - shows BEFORE state for now.

Usage:
    python examples/visualize_type2_triple_point.py \
        --solution <path>/*_refined_contours.h5 \
        --region 2 \
        --triple-point-index 0 \
        --state before \
        --show-vps \
        --show-steiner \
        --apply-zoom \
        --zoom-factor 0.05 --vp-size 0.0004 --boundary-tol 0.1

Author: Type 2 triple point migration visualization
Date: January 2026
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista is required. Install with: pip install pyvista")
    sys.exit(1)

from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.core.mesh_topology import MeshTopology
from src.core.topology_switcher import TopologySwitcher
from src.core.steiner_handler import SteinerHandler, TriplePoint
from src.core.area_calculator import AreaCalculator

# Import data loading from the reference script
from examples.visualize_precise_region import load_partition_from_refined_file


# ============================================================================
# Visualization Functions (Copied from visualize_type1_vertex_collapse.py)
# ============================================================================

def _order_polygon_vertices(points: np.ndarray, mesh: TriMesh, tri_idx: int) -> np.ndarray:
    """
    Order polygon vertices counter-clockwise around triangle centroid.
    
    Args:
        points: (N, 3) array of polygon vertices
        mesh: TriMesh (for normal computation)
        tri_idx: Triangle index (for normal computation)
    
    Returns:
        Ordered vertices (N, 3)
    """
    if len(points) < 3:
        return points
    
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Get triangle normal
    face = mesh.faces[tri_idx]
    v1, v2, v3 = [mesh.vertices[int(i)] for i in face]
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    
    # Compute angles from centroid
    v0 = points[0] - centroid
    v0 = v0 / (np.linalg.norm(v0) + 1e-12)
    
    angles = []
    for p in points:
        v = p - centroid
        v = v / (np.linalg.norm(v) + 1e-12)
        
        # Angle using atan2 in plane perpendicular to normal
        cos_angle = np.dot(v0, v)
        sin_angle = np.dot(np.cross(v0, v), normal)
        angle = np.arctan2(sin_angle, cos_angle)
        angles.append(angle)
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def _vertex_on_cell_side_for_viz(
    vertex_pos: np.ndarray,
    boundary_p1: np.ndarray,
    boundary_p2: np.ndarray,
    cell_idx: int,
    mesh: TriMesh,
    tri_idx: int,
    partition: PartitionContour
) -> bool:
    """
    Determine if a mesh vertex belongs to the given cell side of a boundary.
    
    Uses indicator_functions to check vertex cell assignment.
    
    Args:
        vertex_pos: 3D position of mesh vertex
        boundary_p1, boundary_p2: Boundary segment endpoints (VPs)
        cell_idx: Cell index to check
        mesh: TriMesh
        tri_idx: Triangle index (not used but kept for compatibility)
        partition: PartitionContour with indicator_functions
    
    Returns:
        True if vertex is on cell_idx side
    """
    # Find which mesh vertex this is
    distances = np.linalg.norm(mesh.vertices - vertex_pos, axis=1)
    vertex_idx = np.argmin(distances)
    
    # Check if vertex belongs to cell_idx using indicator_functions
    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
    vertex_cell = int(vertex_labels[vertex_idx])
    
    return vertex_cell == cell_idx


def compute_cell_portion_in_triangle_simple(
    mesh: TriMesh,
    partition: PartitionContour,
    tri_idx: int,
    cell_idx: int,
    tri_idx_to_segment: Optional[Dict] = None
) -> Optional[np.ndarray]:
    """
    Compute polygon for cell portion of boundary triangle (simplified - no crossing cache).
    
    Returns:
        (N, 3) array of polygon vertices, or None if triangle doesn't contribute
    """
    # Get mesh vertices and labels
    face = mesh.faces[tri_idx]
    v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
    labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
    
    # Check if triangle is boundary via VPs (only path - no crossing cache)
    is_boundary = False
    vp_positions = []
    
    if tri_idx_to_segment is not None:
        tri_seg = tri_idx_to_segment.get(tri_idx)
        if tri_seg:
            for vp_idx in tri_seg.var_point_indices:
                vp = partition.variable_points[vp_idx]
                if cell_idx in vp.belongs_to_cells:
                    is_boundary = True
                    vp_positions.append(vp.evaluate(mesh.vertices))
    else:
        for tri_seg in partition.triangle_segments:
            if tri_seg.triangle_idx == tri_idx:
                for vp_idx in tri_seg.var_point_indices:
                    vp = partition.variable_points[vp_idx]
                    if cell_idx in vp.belongs_to_cells:
                        is_boundary = True
                        vp_positions.append(vp.evaluate(mesh.vertices))
                break
    
    if not is_boundary:
        return None
    
    # Standard case: Use VP positions and indicator_functions
    vertices_inside = []
    
    if len(vp_positions) >= 2:
        # 2+ VPs define the boundary line
        vp_pos1 = vp_positions[0]
        vp_pos2 = vp_positions[1]
        
        for v in [v1, v2, v3]:
            v_pos = mesh.vertices[v]
            if _vertex_on_cell_side_for_viz(v_pos, vp_pos1, vp_pos2, cell_idx, 
                                           mesh, tri_idx, partition):
                vertices_inside.append(v_pos)
    else:
        # Fallback to indicator_functions
        for v, lab in zip([v1, v2, v3], labels):
            if lab == cell_idx:
                vertices_inside.append(mesh.vertices[v])
    
    # Construct polygon
    all_points = vertices_inside + vp_positions
    
    if len(all_points) < 3:
        return None
    
    return _order_polygon_vertices(np.array(all_points), mesh, tri_idx)


def compute_triple_point_cell_portion(
    mesh: TriMesh,
    partition: PartitionContour,
    steiner_handler: SteinerHandler,
    tri_idx: int,
    cell_idx: int
) -> Optional[List[np.ndarray]]:
    """
    Compute the portion of a triple point triangle belonging to cell_idx.
    
    Returns BOTH void interior AND corner region.
    
    Returns:
        List of polygons (numpy arrays) or None if not found:
        - [0]: Void interior wedge (VP1, VP2, steiner)
        - [1]: Corner triangle (mesh_vertex, VP1, VP2) [if exists]
    """
    # Find the triple point for this triangle
    triple_point = None
    for tp in steiner_handler.triple_points:
        if tp.triangle_idx == tri_idx:
            triple_point = tp
            break
    
    if not triple_point or cell_idx not in triple_point.cell_indices:
        return None
    
    steiner_pos = triple_point.compute_steiner_point()
    
    # Get the two VPs that bound this cell (from triple point mapping)
    if cell_idx not in triple_point.cell_to_varpoint_pair:
        return None
    
    vp_idx1, vp_idx2 = triple_point.cell_to_varpoint_pair[cell_idx]
    vp1 = partition.variable_points[vp_idx1]
    vp2 = partition.variable_points[vp_idx2]
    vp1_pos = vp1.evaluate(mesh.vertices)
    vp2_pos = vp2.evaluate(mesh.vertices)
    
    polygons = []
    
    # Polygon 1: Void interior wedge (Steiner + two VPs)
    # REVERSED vertex order to make normal point outward
    void_wedge = np.array([steiner_pos, vp2_pos, vp1_pos])
    polygons.append(void_wedge)
    
    # Polygon 2: Corner triangle (if cell has a mesh vertex)
    if cell_idx in triple_point.cell_to_mesh_vertex:
        mesh_vertex_idx = triple_point.cell_to_mesh_vertex[cell_idx]
        mesh_vertex_pos = mesh.vertices[mesh_vertex_idx]
        # REVERSED vertex order to make normal point outward
        corner_triangle = np.array([mesh_vertex_pos, vp2_pos, vp1_pos])
        polygons.append(corner_triangle)
    
    return polygons


def render_single_region_simple(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    cell_idx: int,
    color: str,
    opacity: float = 1.0,
    backface_culling: bool = False,
):
    """
    Render ONE region with precise boundaries (simplified for vertex-collapse).
    
    SIMPLIFIED VERSION:
    - Uses compute_cell_portion_in_triangle_simple (no crossing cache)
    - Includes triple point rendering (Steiner subdivisions)
    - Faster, cleaner code for vertex-collapse strategy
    
    Args:
        backface_culling: If True, hide back-facing triangles (default: False to match Type 1)
    """
    # Pre-index triangle_segments for O(1) lookup
    tri_idx_to_segment = {}
    for tri_seg in partition.triangle_segments:
        tri_idx_to_segment[tri_seg.triangle_idx] = tri_seg
    
    # Collect all polygon vertices
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Interior triangles (full)
    interior_tris = area_calc.cell_interior_triangles.get(cell_idx, [])
    for tri_idx in interior_tris:
        face = mesh.faces[tri_idx]
        vertices = mesh.vertices[[int(face[0]), int(face[1]), int(face[2])]]
        all_vertices.append(vertices)
        all_faces.extend([3, vertex_offset, vertex_offset+1, vertex_offset+2])
        vertex_offset += 3
    
    # Boundary triangles (partial)
    boundary_tris = area_calc.cell_boundary_triangles.get(cell_idx, [])
    for tri_idx in boundary_tris:
        poly_vertices = compute_cell_portion_in_triangle_simple(
            mesh, partition, tri_idx, cell_idx, tri_idx_to_segment
        )
        if poly_vertices is not None:
            n_verts = len(poly_vertices)
            all_vertices.append(poly_vertices)
            face_indices = [n_verts] + list(range(vertex_offset, vertex_offset + n_verts))
            all_faces.extend(face_indices)
            vertex_offset += n_verts
    
    # Triple point triangles (Steiner subdivisions)
    # Handles void interior wedge + corner triangle for each cell
    for tp in steiner_handler.triple_points:
        if cell_idx in tp.cell_indices:
            polygons = compute_triple_point_cell_portion(
                mesh, partition, steiner_handler, tp.triangle_idx, cell_idx
            )
            if polygons is not None:
                # Add all polygons (void interior + corner)
                for poly_vertices in polygons:
                    n_verts = len(poly_vertices)
                    all_vertices.append(poly_vertices)
                    face_indices = [n_verts] + list(range(vertex_offset, vertex_offset + n_verts))
                    all_faces.extend(face_indices)
                    vertex_offset += n_verts
    
    # Create single mesh from all polygons
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        region_mesh = pv.PolyData(all_vertices, faces=all_faces)
        plotter.add_mesh(region_mesh, color=color, opacity=opacity, 
                        show_edges=True, edge_color='lightgray', line_width=0.5,
                        backface_culling=backface_culling)


def render_regions(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    target_region: Optional[int] = None,
    target_color: str = '#FF8C42',  # Bright warm orange
    backface_culling: bool = False,
):
    """Render all regions with precise boundaries (simplified for vertex-collapse).
    
    Args:
        backface_culling: If True, hide back-facing triangles (default: False to match Type 1)
    """
    # Warm, vibrant color palette
    warm_palette = [
        '#FFB366',  # Warm orange
        '#FF8C66',  # Coral
        '#FFD966',  # Golden yellow
        '#FF9999',  # Salmon pink
        '#FFCC99',  # Peach
        '#FFB3BA',  # Light pink
        '#FFDFBA',  # Light peach
        '#FFFFBA',  # Light yellow
        '#BAFFC9',  # Light mint
        '#BAE1FF',  # Light blue
        '#E0BBE4',  # Light lavender
        '#FFDFD3',  # Light coral
    ]
    
    print(f"  Rendering all {partition.n_cells} regions (precise boundaries)...")
    
    for cell_idx in range(partition.n_cells):
        if target_region is not None and cell_idx == target_region:
            color = target_color
            opacity = 1.0
        else:
            color = warm_palette[cell_idx % len(warm_palette)]
            opacity = 0.85
        
        # Use simplified rendering function (no crossing cache)
        render_single_region_simple(
            plotter, mesh, partition, area_calc, steiner_handler,
            cell_idx, color, opacity=opacity, backface_culling=backface_culling
        )
    
    print(f"  ✓ Region rendering complete")


def add_steiner_visualization(
    plotter: pv.Plotter,
    steiner_handler: SteinerHandler,
    partition: PartitionContour,
    mesh: TriMesh,
    steiner_size: float = 0.000005
):
    """Add Steiner points and void triangles to visualization."""
    triple_points = steiner_handler.triple_points
    
    if not triple_points:
        return
    
    print(f"  Adding {len(triple_points)} Steiner points and void triangles...")
    
    for tp in triple_points:
        # Compute Steiner point
        steiner_pt = tp.compute_steiner_point()
        
        # Add Steiner point as red sphere
        sphere = pv.Sphere(radius=steiner_size, center=steiner_pt)
        plotter.add_mesh(sphere, color='red', opacity=1.0)
        
        # Get VP positions for void triangle
        vp_positions = []
        for vp_idx in tp.var_point_indices:
            vp = partition.variable_points[vp_idx]
            vp_pos = vp.evaluate(mesh.vertices)
            vp_positions.append(vp_pos)
        
        if len(vp_positions) == 3:
            # Draw void triangle edges (cyan)
            for i in range(3):
                p1 = vp_positions[i]
                p2 = vp_positions[(i + 1) % 3]
                line = pv.Line(p1, p2)
                plotter.add_mesh(line, color='cyan', line_width=2, opacity=0.7)


def add_vp_visualization(
    plotter: pv.Plotter,
    partition: PartitionContour,
    mesh: TriMesh,
    vp_indices: List[int],
    vp_colors: List[str],
    vp_labels: List[str],
    vp_size: float = 0.0005
):
    """Add VP spheres with labels to visualization."""
    for vp_idx, color, label in zip(vp_indices, vp_colors, vp_labels):
        vp = partition.variable_points[vp_idx]
        vp_pos = vp.evaluate(mesh.vertices)
        
        # Add sphere
        sphere = pv.Sphere(radius=vp_size, center=vp_pos)
        plotter.add_mesh(sphere, color=color, opacity=1.0)
        
        # Add label
        if label:
            label_pos = vp_pos + np.array([0, 0, vp_size * 3])
            plotter.add_point_labels(
                [label_pos], [label],
                font_size=10, text_color='black',
                shape_color='white', shape_opacity=0.7,
                always_visible=True, point_size=8
            )


def add_triple_point_triangle_labels(
    plotter: pv.Plotter,
    mesh: TriMesh,
    mesh_topology: MeshTopology,
    triple_point: TriplePoint,
    partition: PartitionContour
):
    """
    Add triangle labels for all triangles involved with the triple point.
    
    Labels:
    - The void triangle itself
    - All triangles sharing edges with the 3 VPs
    """
    triangles_to_label = set()
    
    # Add the void triangle
    triangles_to_label.add(triple_point.triangle_idx)
    
    # Add triangles for each VP in the triple point
    for vp_idx in triple_point.var_point_indices:
        vp = partition.variable_points[vp_idx]
        vp_edge = tuple(sorted(vp.edge))
        triangles_at_edge = mesh_topology.get_triangles_sharing_edge(vp_edge)
        triangles_to_label.update(triangles_at_edge)
    
    print(f"  Labeling {len(triangles_to_label)} triangles: {sorted(triangles_to_label)}")
    
    # Add labels
    for tri_idx in triangles_to_label:
        face = mesh.faces[tri_idx]
        tri_vertices = mesh.vertices[face]
        centroid = np.mean(tri_vertices, axis=0)
        
        # Offset slightly along normal for visibility
        v1, v2, v3 = tri_vertices
        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        label_pos = centroid + normal * 0.00005
        
        plotter.add_point_labels(
            points=[label_pos],
            labels=[f"T{tri_idx}"],
            font_size=14,
            text_color='black',
            render_points_as_spheres=True,
            point_size=12,
            point_color='yellow',
            always_visible=True
        )


def compute_steiner_distance_to_boundary(triple_point: TriplePoint, mesh: TriMesh) -> float:
    """
    Compute distance from Steiner point to closest edge of the triangle.
    
    Args:
        triple_point: TriplePoint object
        mesh: TriMesh
    
    Returns:
        Distance to closest edge
    """
    steiner_pos = triple_point.compute_steiner_point()
    tri_idx = triple_point.triangle_idx
    face = mesh.faces[tri_idx]
    
    # Get triangle vertices
    v0, v1, v2 = [mesh.vertices[int(i)] for i in face]
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    
    min_dist = float('inf')
    for edge_v1, edge_v2 in edges:
        # Distance from point to line segment
        dist = triple_point._point_to_segment_distance(steiner_pos, edge_v1, edge_v2)
        min_dist = min(min_dist, dist)
    
    return min_dist


def run_visualization(args):
    """Main visualization routine."""
    print("="*80)
    print("TYPE 2 TRIPLE POINT VISUALIZATION")
    print("="*80)
    print(f"Refined contours file: {args.solution}")
    print(f"Target region: {args.region}")
    print(f"Switch type: type2")
    print()
    
    # Load partition
    print("Loading partition data...")
    mesh, partition = load_partition_from_refined_file(args.solution)
    print("\n✓ Loaded partition state from refined file\n")
    
    # Initialize topology components
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    steiner_handler = SteinerHandler(mesh, partition)
    
    # Triple point analysis
    print("Analyzing Triple Points for Type 2 Migration...")
    
    all_triple_points = steiner_handler.triple_points
    print(f"✓ Found {len(all_triple_points)} total triple points")
    
    # Filter boundary triple points
    boundary_triple_points = steiner_handler.get_boundary_triple_points(tol=args.boundary_tol)
    print(f"✓ Found {len(boundary_triple_points)} boundary triple points (within boundary_tol={args.boundary_tol})")
    print()
    
    if not boundary_triple_points:
        print("ERROR: No boundary triple points found for Type 2 migration")
        return
    
    # Display triple point table
    print("="*80)
    print("BOUNDARY TRIPLE POINTS (Type 2 Migration Candidates)")
    print("="*80)
    print(f"{'Idx':<5} {'Triangle':<10} {'VPs':<30} {'Cells':<15} {'Dist to Boundary':<20}")
    print("-" * 80)
    
    for i, tp in enumerate(boundary_triple_points):
        vp_str = str(tp.var_point_indices)
        cells_str = str(sorted(tp.cell_indices))
        dist = compute_steiner_distance_to_boundary(tp, mesh)
        print(f"{i:<5} {tp.triangle_idx:<10} {vp_str:<30} {cells_str:<15} {dist:<20.6f}")
    
    print()
    
    # Select triple point
    if args.triple_point_index >= len(boundary_triple_points):
        print(f"ERROR: Triple point index {args.triple_point_index} out of range (max: {len(boundary_triple_points)-1})")
        return
    
    selected_tp = boundary_triple_points[args.triple_point_index]
    
    print(f"Selected Triple Point: {args.triple_point_index}")
    print(f"  Triangle: {selected_tp.triangle_idx}")
    print(f"  VPs: {selected_tp.var_point_indices}")
    
    # Get VP edges
    vp_edges = []
    for vp_idx in selected_tp.var_point_indices:
        vp = partition.variable_points[vp_idx]
        vp_edges.append(vp.edge)
        print(f"    VP {vp_idx}: edge {vp.edge}, λ = {vp.lambda_param:.6f}")
    
    print(f"  Cells: {sorted(selected_tp.cell_indices)}")
    
    dist = compute_steiner_distance_to_boundary(selected_tp, mesh)
    print(f"  Steiner distance to boundary: {dist:.6f}")
    print()
    
    # ========================================================================
    # BEFORE STATE
    # ========================================================================
    
    print("="*60)
    print("BEFORE Type 2 Migration")
    print("="*60)
    print()
    
    # Initialize AreaCalculator (for rendering only)
    print("  Initializing AreaCalculator...")
    area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
    print()
    
    # Get Steiner position for camera focus
    steiner_pos = selected_tp.compute_steiner_point()
    
    print("Rendering BEFORE state...")
    print()
    
    # Create plotter
    plotter_before = pv.Plotter()
    plotter_before.add_title(
        f"Type 2 BEFORE: Triple Point at Triangle {selected_tp.triangle_idx} (triple-point-index={args.triple_point_index})",
        font_size=12
    )
    
    # Render all regions with precise boundaries (same as Type 1)
    render_regions(
        plotter_before, mesh, partition, area_calc, steiner_handler,
        target_region=None,  # No special highlighting for Type 2
        backface_culling=args.enable_backface_culling
    )
    
    # Add Steiner visualization if requested
    if args.show_steiner:
        add_steiner_visualization(plotter_before, steiner_handler, partition, mesh, 
                                args.steiner_size)
    
    # Add VP visualization if requested
    if args.show_vps:
        vp_colors = ['red', 'blue', 'green']
        vp_labels = [
            f"VP{i+1}\nidx={vp_idx}\nedge={vp_edges[i]}"
            for i, vp_idx in enumerate(selected_tp.var_point_indices)
        ]
        
        print("Triple Point VPs:")
        for i, vp_idx in enumerate(selected_tp.var_point_indices):
            color_name = vp_colors[i].capitalize()
            print(f"  {color_name} VP: {vp_idx}")
            print(f"    Edge: {vp_edges[i]}")
            vp = partition.variable_points[vp_idx]
            print(f"    Lambda: {vp.lambda_param:.6f}")
        print()
        
        add_vp_visualization(
            plotter_before, partition, mesh,
            selected_tp.var_point_indices,
            vp_colors,
            vp_labels,
            args.vp_size
        )
    
    # Add triangle labels
    add_triple_point_triangle_labels(plotter_before, mesh, mesh_topology, 
                                     selected_tp, partition)
    
    # Apply zoom if requested
    if args.apply_zoom:
        print(f"  Applying zoom to Steiner point at {steiner_pos}")
        camera_offset = np.array([args.zoom_factor, args.zoom_factor * 0.5, args.zoom_factor * 0.5])
        camera_position = steiner_pos + camera_offset
        
        plotter_before.camera_position = [
            camera_position,
            steiner_pos,
            (0, 0, 1)
        ]
        plotter_before.camera.clipping_range = (args.zoom_factor * 0.1, args.zoom_factor * 10)
    
    print()
    print("Opening PyVista window (BEFORE state)...")
    plotter_before.show()
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Type 2 migration (Triple Point Migration)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--solution', required=True,
                       help='Path to refined_contours.h5 file')
    parser.add_argument('--region', type=int, required=True,
                       help='Region/cell index to visualize')
    
    # Migration parameters
    parser.add_argument('--triple-point-index', type=int, default=0,
                       help='Triple point index to visualize (default: 0)')
    parser.add_argument('--boundary-tol', type=float, default=0.1,
                       help='Boundary tolerance for triple point detection (default: 0.1)')
    
    # Visualization options
    parser.add_argument('--state', choices=['before', 'after', 'both'], default='before',
                       help='Which state to show (default: before)')
    parser.add_argument('--intense-color', default='#FF8C42',
                       help='Color for target region (default: bright warm orange #FF8C42)')
    parser.add_argument('--show-vps', action='store_true',
                       help='Show variable points as spheres')
    parser.add_argument('--show-steiner', action='store_true',
                       help='Show Steiner points and void triangles')
    parser.add_argument('--enable-backface-culling', action='store_true',
                       help='Hide back-facing triangles (default: show all faces like Type 1)')
    
    # Camera/zoom options
    parser.add_argument('--apply-zoom', action='store_true',
                       help='Apply zoom to focus on triple point')
    parser.add_argument('--zoom-factor', type=float, default=0.1,
                       help='Zoom factor (smaller = more zoomed) (default: 0.1)')
    parser.add_argument('--vp-size', type=float, default=0.0005,
                       help='Size of VP spheres (default: 0.0005)')
    parser.add_argument('--steiner-size', type=float, default=0.000005,
                       help='Size of Steiner point spheres (default: 0.000005)')
    
    args = parser.parse_args()
    
    run_visualization(args)


if __name__ == '__main__':
    main()
