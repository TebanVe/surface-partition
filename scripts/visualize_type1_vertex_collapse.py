#!/usr/bin/env python3
"""
Visualize Type 1 migration using vertex-collapse strategy.

Clean, focused implementation for testing and debugging the new approach.
Only includes necessary code - no legacy methods.

Usage:
    python scripts/visualize_type1_vertex_collapse.py \\
        --solution <path>/*_refined_contours.h5 \\
        --region 2 \\
        --component-index 0 \\
        --state before \\
        --show-vps \\
        --show-steiner \\
        --apply-zoom \\
        --zoom-factor 0.05 --vp-size 0.0004 --boundary-tol 0.01

Author: Clean implementation for vertex-collapse strategy
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

from src.mesh.tri_mesh import TriMesh
from src.partition.contour_partition import PartitionContour
from src.mesh.mesh_topology import MeshTopology
from src.partition.steiner_handler import SteinerHandler
from src.partition.area_calculator import AreaCalculator
from src.migration import migration_utils
from src.migration.migration_orchestrator import MigrationOrchestrator, MigrationConfig
from src.migration.migration_detector import detect_type1_triggers

from src.pipeline.io import load_partition_from_refined_file


# ============================================================================
# Simplified Visualization Functions (Vertex-Collapse Only - No Crossing Cache)
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
    Determine if a vertex is on the cell_idx side of the boundary segment.
    
    SIMPLIFIED for vertex-collapse: Uses only VP edge endpoints and vertex_labels.
    Does NOT use crossing cache.
    
    Args:
        vertex_pos: 3D position of mesh vertex
        boundary_p1: First boundary point (VP position)
        boundary_p2: Second boundary point (VP position)
        cell_idx: Cell index we're checking
        mesh: TriMesh
        tri_idx: Triangle index
        partition: PartitionContour
        
    Returns:
        True if vertex is on cell_idx side of boundary
    """
    # Vector from boundary_p1 to boundary_p2
    segment_vec = boundary_p2 - boundary_p1
    seg_len = np.linalg.norm(segment_vec)
    if seg_len < 1e-12:
        return False
    
    # Vector from boundary_p1 to vertex
    to_vertex = vertex_pos - boundary_p1
    
    # Get triangle normal
    face = mesh.faces[tri_idx]
    v1, v2, v3 = [mesh.vertices[int(i)] for i in face]
    normal = np.cross(v2 - v1, v3 - v1)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-12:
        return False
    normal = normal / normal_len
    
    # Cross product gives signed area (determines which side of line)
    cross = np.cross(segment_vec, to_vertex)
    signed_area = np.dot(cross, normal)
    
    # STRATEGY 1: Use VPs on this triangle's edges to find reference
    ref_vertex = None
    ref_is_cell_idx_side = False
    
    for tri_seg in partition.triangle_segments:
        if tri_seg.triangle_idx == tri_idx:
            for vp_idx in tri_seg.var_point_indices:
                vp = partition.variable_points[vp_idx]
                if not vp.active:
                    continue
                edge = vp.edge
                v_a, v_b = edge
                vertex_labels = partition.vertex_labels
                
                # One endpoint should be on each side
                label_a = vertex_labels[v_a]
                label_b = vertex_labels[v_b]
                
                if label_a == cell_idx and label_b != cell_idx:
                    ref_vertex = mesh.vertices[v_a]
                    ref_is_cell_idx_side = True
                    break
                elif label_b == cell_idx and label_a != cell_idx:
                    ref_vertex = mesh.vertices[v_b]
                    ref_is_cell_idx_side = True
                    break
                elif label_a != cell_idx:
                    ref_vertex = mesh.vertices[v_a]
                    ref_is_cell_idx_side = False
                    break
                elif label_b != cell_idx:
                    ref_vertex = mesh.vertices[v_b]
                    ref_is_cell_idx_side = False
                    break
            break
    
    # STRATEGY 2: Fallback to vertex_labels on triangle vertices
    if ref_vertex is None:
        vertex_labels = partition.vertex_labels
        
        # Find any vertex labeled as cell_idx
        for v_idx in [int(face[0]), int(face[1]), int(face[2])]:
            if vertex_labels[v_idx] == cell_idx:
                ref_vertex = mesh.vertices[v_idx]
                ref_is_cell_idx_side = True
                break
        
        # If none found, find vertex NOT labeled as cell_idx
        if ref_vertex is None:
            for v_idx in [int(face[0]), int(face[1]), int(face[2])]:
                if vertex_labels[v_idx] != cell_idx:
                    ref_vertex = mesh.vertices[v_idx]
                    ref_is_cell_idx_side = False
                    break
    
    if ref_vertex is None:
        return False
    
    # Compute which side the reference vertex is on
    ref_to_vertex = ref_vertex - boundary_p1
    ref_cross = np.cross(segment_vec, ref_to_vertex)
    ref_signed = np.dot(ref_cross, normal)
    
    # Determine if vertex is on cell_idx side
    same_side_as_ref = (signed_area > 0) == (ref_signed > 0)
    
    if ref_is_cell_idx_side:
        return same_side_as_ref
    else:
        return not same_side_as_ref


def compute_cell_portion_in_triangle_simple(
    mesh: TriMesh,
    partition: PartitionContour,
    tri_idx: int,
    cell_idx: int,
    tri_idx_to_segment: Optional[Dict[int, any]] = None
) -> Optional[np.ndarray]:
    """
    Compute vertices of the polygon representing cell_idx's portion in tri_idx.
    
    SIMPLIFIED VERSION for vertex-collapse:
    - Only uses VP-based logic (no crossing cache)
    - Assumes segments stay within single triangles
    
    Args:
        mesh: TriMesh
        partition: PartitionContour
        tri_idx: Triangle index
        cell_idx: Cell index
        tri_idx_to_segment: Pre-indexed dict mapping triangle_idx to TriangleSegment
        
    Returns:
        (N, 3) array of polygon vertices, or None if triangle doesn't contribute
    """
    # Get mesh vertices and labels
    face = mesh.faces[tri_idx]
    v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
    vertex_labels = partition.vertex_labels
    labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
    
    # Check if triangle is boundary via VPs (only path - no crossing cache)
    is_boundary = False
    vp_positions = []
    
    if tri_idx_to_segment is not None:
        tri_seg = tri_idx_to_segment.get(tri_idx)
        if tri_seg:
            for vp_idx in tri_seg.var_point_indices:
                vp = partition.variable_points[vp_idx]
                if not vp.active:
                    continue
                if cell_idx in vp.belongs_to_cells:
                    is_boundary = True
                    vp_positions.append(vp.evaluate(mesh.vertices))
    else:
        for tri_seg in partition.triangle_segments:
            if tri_seg.triangle_idx == tri_idx:
                for vp_idx in tri_seg.var_point_indices:
                    vp = partition.variable_points[vp_idx]
                    if not vp.active:
                        continue
                    if cell_idx in vp.belongs_to_cells:
                        is_boundary = True
                        vp_positions.append(vp.evaluate(mesh.vertices))
                break
    
    if not is_boundary:
        return None
    
    # Standard case: Use VP positions and vertex_labels
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
        # Fallback to vertex_labels
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
    
    # Compute Steiner point
    vp_positions_for_steiner = [partition.evaluate_variable_point(vi) for vi in triple_point.var_point_indices]
    steiner_pos = triple_point.compute_steiner_point(vp_positions=vp_positions_for_steiner)
    
    # Get the two VPs that bound this cell (from triple point mapping)
    if cell_idx not in triple_point.cell_to_varpoint_pair:
        return None
    
    vp_idx1, vp_idx2 = triple_point.cell_to_varpoint_pair[cell_idx]
    vp_pos1 = partition.evaluate_variable_point(vp_idx1)
    vp_pos2 = partition.evaluate_variable_point(vp_idx2)
    
    # POLYGON 1: Void interior wedge
    # The cell's portion of the void is bounded by the two VPs and Steiner point
    void_wedge = np.array([
        vp_pos1,
        vp_pos2,
        steiner_pos
    ])
    
    # POLYGON 2: Corner triangle (if this cell has a mesh vertex)
    polygons = [void_wedge]
    
    if cell_idx in triple_point.cell_to_mesh_vertex:
        mesh_vertex_idx = triple_point.cell_to_mesh_vertex[cell_idx]
        vertex_pos = mesh.vertices[mesh_vertex_idx]
        
        # Corner triangle: mesh vertex to the two adjacent VPs
        corner_triangle = np.array([
            vertex_pos,
            vp_pos1,
            vp_pos2
        ])
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
):
    """
    Render ONE region with precise boundaries (simplified for vertex-collapse).
    
    SIMPLIFIED VERSION:
    - Uses compute_cell_portion_in_triangle_simple (no crossing cache)
    - Includes triple point rendering (Steiner subdivisions)
    - Faster, cleaner code for vertex-collapse strategy
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
                        show_edges=True, edge_color='lightgray', line_width=0.5)


def compute_region_area(area_calc: AreaCalculator, cell_idx: int, lambda_vec: np.ndarray) -> Dict:
    """Compute and report area for a region."""
    area = area_calc.compute_cell_area(cell_idx, lambda_vec)
    
    interior_tris = area_calc.cell_interior_triangles.get(cell_idx, [])
    boundary_tris = area_calc.cell_boundary_triangles.get(cell_idx, [])
    
    interior_area = area_calc.cell_interior_area.get(cell_idx, 0.0)
    boundary_area = area - interior_area
    
    return {
        'total_area': area,
        'interior_area': interior_area,
        'boundary_area': boundary_area,
        'n_interior_triangles': len(interior_tris),
        'n_boundary_triangles': len(boundary_tris)
    }


def render_regions(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    target_region: int,
    target_color: str = '#FF8C42',  # Bright warm orange for better VP visibility
):
    """Render all regions with precise boundaries (simplified for vertex-collapse)."""
    # Pastel colors for all regions
    pale_palette = [
        '#FFE5B4', '#E0BBE4', '#FFDAC1', '#B5EAD7', '#C7CEEA',
        '#FFB7B2', '#FFDFD3', '#E2F0CB', '#B4F8C8', '#A0C4FF',
        '#FFC6FF', '#FFCFD2', '#FDE2E4', '#FAD2E1', '#BEE1E6'
    ]
    
    print(f"  Rendering all {partition.n_cells} regions (precise boundaries)...")
    
    for cell_idx in range(partition.n_cells):
        if cell_idx == target_region:
            color = target_color
            opacity = 1.0
        else:
            color = pale_palette[cell_idx % len(pale_palette)]
            opacity = 0.8
        
        # Use simplified rendering function (no crossing cache)
        render_single_region_simple(
            plotter, mesh, partition, area_calc, steiner_handler,
            cell_idx, color, opacity=opacity
        )
    
    print(f"  ✓ Region rendering complete")


def add_vp_visualization(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    vp_indices: List[int],
    vp_labels: List[str],
    vp_size: float = 0.0005,
):
    """Add variable points as colored spheres with labels."""
    if not vp_indices:
        return
    
    # Color scheme for 5 VPs: migrating, 2 neighbors, 2 secondary neighbors
    # Using distinct, easily distinguishable colors
    colors = [
        'yellow',      # Migrating VP
        'blue',        # Neighbor 1
        'magenta',     # Neighbor 2
        'cyan',        # Secondary neighbor 1
        'orange'       # Secondary neighbor 2
    ]
    color_names = ['YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'ORANGE']
    
    print(f"  Adding {len(vp_indices)} highlighted variable points (migration-related)...")
    print(f"    VP color scheme:")
    
    for i, (vp_idx, label) in enumerate(zip(vp_indices, vp_labels)):
        if vp_idx >= len(partition.variable_points):
            print(f"      WARNING: VP {vp_idx} out of range")
            continue
        
        vp = partition.variable_points[vp_idx]
        if not vp.active:
            print(f"      WARNING: VP {vp_idx} is inactive, skipping")
            continue
        pos = vp.evaluate(mesh.vertices)
        sphere = pv.Sphere(radius=vp_size, center=pos)
        
        color = colors[i % len(colors)]
        color_name = color_names[i % len(color_names)]
        
        print(f"      VP {vp_idx} ({label}): {color_name}")
        plotter.add_mesh(sphere, color=color, opacity=0.9)


def add_steiner_visualization(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    steiner_handler: SteinerHandler,
    steiner_size: float = 0.000005,
):
    """Add Steiner points and void triangles."""
    triple_points = steiner_handler.triple_points
    
    if not triple_points:
        return
    
    print(f"  Adding {len(triple_points)} Steiner points and void triangles...")
    
    for tp in triple_points:
        # Compute Steiner point
        vp_positions_for_steiner = [partition.evaluate_variable_point(vi) for vi in tp.var_point_indices]
        steiner_pos = tp.compute_steiner_point(vp_positions=vp_positions_for_steiner)
        
        # Add Steiner point as red sphere
        steiner_sphere = pv.Sphere(radius=steiner_size, center=steiner_pos)
        plotter.add_mesh(steiner_sphere, color='red', opacity=1.0)
        
        # Add void triangle edges (cyan lines connecting VPs)
        vp_positions = []
        for vp_idx in tp.var_point_indices:
            vp = partition.variable_points[vp_idx]
            vp_positions.append(vp.evaluate(mesh.vertices))
        
        # Draw edges of void triangle
        for i in range(3):
            j = (i + 1) % 3
            line = pv.Line(vp_positions[i], vp_positions[j])
            plotter.add_mesh(line, color='cyan', line_width=2, opacity=0.7)


def add_edge_visualization(
    plotter: pv.Plotter,
    mesh: TriMesh,
    current_edge: Optional[Tuple[int, int]] = None,
    target_edge: Optional[Tuple[int, int]] = None,
):
    """Add current and target edge visualization."""
    if current_edge is None and target_edge is None:
        return
    
    print(f"  Adding edge visualization for migration...")
    
    # Current edge (before migration) - RED
    if current_edge is not None:
        v1_idx, v2_idx = current_edge
        v1_pos = mesh.vertices[v1_idx]
        v2_pos = mesh.vertices[v2_idx]
        current_line = pv.Line(v1_pos, v2_pos)
        plotter.add_mesh(current_line, color='red', line_width=5, opacity=0.9)
        print(f"    Current edge: {current_edge} (RED)")
    
    # Target edge (after migration) - GREEN
    if target_edge is not None:
        v1_idx, v2_idx = target_edge
        v1_pos = mesh.vertices[v1_idx]
        v2_pos = mesh.vertices[v2_idx]
        target_line = pv.Line(v1_pos, v2_pos)
        plotter.add_mesh(target_line, color='lime', line_width=5, opacity=0.9)
        print(f"    Target edge: {target_edge} (GREEN)")


def add_target_vertex_triangle_labels(
    plotter, 
    mesh: TriMesh, 
    target_vertex: int,
    label_all: bool = False,
    apply_zoom: bool = False,
    zoom_factor: float = 0.1,
    label_font_size: int = 14
):
    """
    Add labels to triangles.
    
    If label_all=False (default): Only labels triangles sharing target_vertex (methodology triangles)
    If label_all=True: Labels all triangles in mesh, with methodology triangles using different styling
    
    Args:
        plotter: PyVista plotter
        mesh: TriMesh object
        target_vertex: Index of the target vertex
        label_all: If True, label all triangles
        apply_zoom: If True and label_all=True, use spatial filtering
        zoom_factor: Distance threshold for spatial filtering
    """
    if target_vertex is None:
        return
    
    # Find all triangles that contain the target vertex (methodology triangles)
    triangles_with_target = []
    for tri_idx, face in enumerate(mesh.faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        if target_vertex in [v1, v2, v3]:
            triangles_with_target.append(tri_idx)
    
    methodology_triangles = set(triangles_with_target)
    print(f"  Triangles sharing target vertex {target_vertex}: {len(triangles_with_target)} triangles")
    print(f"  Triangle indices: {triangles_with_target}")
    print()
    
    # Add labels to triangles
    if label_all:
        target_pos = mesh.vertices[target_vertex]
        
        if apply_zoom:
            # Use spatial filtering for performance
            max_distance = zoom_factor * 50  # Increased from 10 to capture more triangles
            print(f"  Using spatial filtering: labeling triangles within {max_distance:.4f} units of target vertex...")
            
            # Batch collect
            methodology_centroids = []
            methodology_labels = []
            other_centroids = []
            other_labels = []
            
            labeled_count = 0
            for tri_idx in range(len(mesh.faces)):
                face = mesh.faces[tri_idx]
                v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
                centroid = (mesh.vertices[v1] + mesh.vertices[v2] + mesh.vertices[v3]) / 3.0
                
                # Offset for visibility
                edge1 = mesh.vertices[v2] - mesh.vertices[v1]
                edge2 = mesh.vertices[v3] - mesh.vertices[v1]
                normal = np.cross(edge1, edge2)
                normal_len = np.linalg.norm(normal)
                if normal_len > 1e-12:
                    normal = normal / normal_len
                    centroid = centroid + normal * 0.0001
                
                # Check distance
                dist = np.linalg.norm(centroid - target_pos)
                if dist < max_distance or tri_idx in methodology_triangles:
                    if tri_idx in methodology_triangles:
                        methodology_centroids.append(centroid)
                        methodology_labels.append(f"T{tri_idx}")
                    else:
                        other_centroids.append(centroid)
                        other_labels.append(f"T{tri_idx}")
                    labeled_count += 1
            
            print(f"  Labeling {labeled_count} triangles (filtered from {len(mesh.faces)})...")
            
            # Batch add
            if methodology_centroids:
                plotter.add_point_labels(
                    np.array(methodology_centroids),
                    methodology_labels,
                    font_size=label_font_size,
                    text_color='black',
                    point_color='yellow',
                    point_size=12,
                    render_points_as_spheres=True,
                    shape_color='gray',
                    shape_opacity=0.9,
                    always_visible=True
                )
            
            if other_centroids:
                plotter.add_point_labels(
                    np.array(other_centroids),
                    other_labels,
                    font_size=label_font_size,
                    text_color='black',
                    shape_color='white',
                    shape_opacity=0.9,
                    always_visible=True
                )
        else:
            # No zoom - use batching only
            print(f"  Labeling all {len(mesh.faces)} triangles using batched approach...")
            
            # Batch collect
            methodology_centroids = []
            methodology_labels = []
            other_centroids = []
            other_labels = []
            
            for tri_idx in range(len(mesh.faces)):
                face = mesh.faces[tri_idx]
                v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
                centroid = (mesh.vertices[v1] + mesh.vertices[v2] + mesh.vertices[v3]) / 3.0
                
                # Offset for visibility
                edge1 = mesh.vertices[v2] - mesh.vertices[v1]
                edge2 = mesh.vertices[v3] - mesh.vertices[v1]
                normal = np.cross(edge1, edge2)
                normal_len = np.linalg.norm(normal)
                if normal_len > 1e-12:
                    normal = normal / normal_len
                    centroid = centroid + normal * 0.0001
                
                if tri_idx in methodology_triangles:
                    methodology_centroids.append(centroid)
                    methodology_labels.append(f"T{tri_idx}")
                else:
                    other_centroids.append(centroid)
                    other_labels.append(f"T{tri_idx}")
            
            # Batch add
            if methodology_centroids:
                plotter.add_point_labels(
                    np.array(methodology_centroids),
                    methodology_labels,
                    font_size=label_font_size,
                    text_color='black',
                    point_color='yellow',
                    point_size=12,
                    render_points_as_spheres=True,
                    shape_color='gray',
                    shape_opacity=0.9,
                    always_visible=True
                )
            
            if other_centroids:
                plotter.add_point_labels(
                    np.array(other_centroids),
                    other_labels,
                    font_size=label_font_size,
                    text_color='black',
                    shape_color='white',
                    shape_opacity=0.9,
                    always_visible=True
                )
    else:
        # Original behavior: only methodology triangles
        for tri_idx in triangles_with_target:
            face = mesh.faces[tri_idx]
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            
            # Compute triangle centroid
            centroid = (mesh.vertices[v1] + mesh.vertices[v2] + mesh.vertices[v3]) / 3.0
            
            # Offset centroid slightly away from surface along triangle normal
            edge1 = mesh.vertices[v2] - mesh.vertices[v1]
            edge2 = mesh.vertices[v3] - mesh.vertices[v1]
            normal = np.cross(edge1, edge2)
            normal_len = np.linalg.norm(normal)
            if normal_len > 1e-12:
                normal = normal / normal_len
                centroid = centroid + normal * 0.0001
            
            # Add label with better visibility (methodology triangles)
            plotter.add_point_labels(
                centroid,
                [f"T{tri_idx}"],
                font_size=label_font_size,
                text_color='black',
                point_color='yellow',
                point_size=12,
                render_points_as_spheres=True,
                shape_color='gray',
                shape_opacity=0.9,
                always_visible=True
            )


def apply_camera_zoom(
    plotter: pv.Plotter,
    camera_focus: np.ndarray,
    zoom_factor: float,
):
    """Apply camera zoom to focus point."""
    # Set camera position relative to focus point
    offset = np.array([zoom_factor, zoom_factor * 0.5, zoom_factor * 0.5])
    camera_position = camera_focus + offset
    
    plotter.camera_position = [
        camera_position,  # Camera location
        camera_focus,     # Focal point
        (0, 0, 1)        # View up direction
    ]
    
    # Adjust clipping range for close-up views
    plotter.camera.clipping_range = (zoom_factor * 0.1, zoom_factor * 10)


def run_visualization(args):
    """Main visualization routine."""
    print("="*80)
    print("PRECISE REGION VISUALIZATION")
    print("="*80)
    print(f"Refined contours file: {args.solution}")
    print(f"Target region: {args.region}")
    print(f"Switch type: type1")
    print()
    
    # Load partition
    print("Loading partition data...")
    mesh, partition = load_partition_from_refined_file(args.solution)
    print("\n✓ Loaded partition state from refined file\n")
    
    # Initialize topology components
    mesh_topology = MeshTopology(mesh)
    steiner_handler = SteinerHandler(mesh, partition)

    orchestrator = MigrationOrchestrator(
        partition, mesh, mesh_topology,
        MigrationConfig(delta=args.boundary_tol)
    )

    print("Analyzing Type 1 migration via MigrationOrchestrator (trigger-based detection)...")
    print()

    detection = orchestrator.detect_all_triggers(delta=args.boundary_tol)
    type1_triggers = detection.type1_triggers

    if not type1_triggers:
        print("No Type 1 triggers detected.")
        return

    print("="*80)
    print("DETECTED TYPE 1 TRIGGERS")
    print("="*80)
    print(f"{'Idx':<5} {'Vertex':<8} {'Dist':<10} {'Cells':<15} {'#VPs':<6} {'VPs':<30}")
    print("-" * 80)

    for i, trig in enumerate(type1_triggers):
        vp_list = str(trig.approaching_vps[:5]) + ("..." if len(trig.approaching_vps) > 5 else "")
        cells_str = f"{trig.current_cell} -> {trig.target_cell}"
        print(f"{i:<5} {trig.vertex:<8} {trig.min_lambda_distance:<10.6f} {cells_str:<15} {trig.n_boundary_vps:<6} {vp_list:<30}")

    print()

    if args.component_index >= len(type1_triggers):
        print(f"ERROR: Trigger index {args.component_index} out of range (max: {len(type1_triggers)-1})")
        return

    selected_trigger = type1_triggers[args.component_index]
    component_vps = selected_trigger.approaching_vps
    target_vertex = selected_trigger.vertex

    print(f"✓ Selected Trigger {args.component_index} for migration:")
    print(f"  Vertex: {selected_trigger.vertex}")
    print(f"  Cell transition: {selected_trigger.current_cell} -> {selected_trigger.target_cell}")
    print(f"  Approaching VPs: {component_vps}")
    print(f"  Min distance: {selected_trigger.min_lambda_distance:.6f}")
    print()
    # ========================================================================
    # PRE-COMPUTE MIGRATION VPs FOR VISUALIZATION
    # ========================================================================
    
    print("Pre-computing migration VPs for visualization...")

    print(f"  Trigger approaching VPs: {selected_trigger.approaching_vps}")
    for vp_idx in selected_trigger.approaching_vps:
        vp_obj = partition.variable_points[vp_idx]
        vp_target = migration_utils.identify_target_vertex(vp_obj)
        print(f"    VP {vp_idx}: edge {vp_obj.edge}, "
              f"λ={vp_obj.lambda_param:.6f}, "
              f"approaches vertex {vp_target}")

    preview_migrating_vp = min(
        selected_trigger.approaching_vps,
        key=lambda vi: migration_utils.compute_boundary_distance(partition.variable_points[vi])
    ) if selected_trigger.approaching_vps else None
    preview_left = None
    preview_right = None

    if preview_migrating_vp is not None:
        preview_migrating_vp_obj = partition.variable_points[preview_migrating_vp]
        preview_old_edge = preview_migrating_vp_obj.edge
    else:
        preview_old_edge = None
    preview_target_edge = None

    print(f"✓ Closest approaching VP: {preview_migrating_vp}")
    print()
    
    # ========================================================================
    # BEFORE STATE
    # ========================================================================
    
    # Initialize AreaCalculator
    print("  Initializing AreaCalculator BEFORE migration...")
    area_calc_before = AreaCalculator(mesh, partition)
    print()
    
    # Compute area
    lambda_vec = partition.get_variable_vector()
    area_info_before = compute_region_area(area_calc_before, args.region, lambda_vec)
    
    print("="*60)
    print("BEFORE Type 1 Migration")
    print("="*60)
    print()
    print(f"Region {args.region} Geometry (BEFORE):")
    print(f"  Interior triangles: {area_info_before['n_interior_triangles']:,} (area: {area_info_before['interior_area']:.4f})")
    print(f"  Boundary triangles: {area_info_before['n_boundary_triangles']:,} (area: {area_info_before['boundary_area']:.4f})")
    print(f"  TOTAL AREA: {area_info_before['total_area']:.6f}")
    
    # Create plotter for BEFORE
    plotter_before = pv.Plotter()
    
    # Render regions
    render_regions(
        plotter_before, mesh, partition, area_calc_before, steiner_handler,
        args.region, args.intense_color
    )
    
    if args.show_vps and preview_migrating_vp is not None:
        highlight_vps = list(selected_trigger.approaching_vps)
        vp_labels = [f'VP{vi}' for vi in highlight_vps]
        add_vp_visualization(plotter_before, mesh, partition, highlight_vps, vp_labels, args.vp_size)
    
    # Add Steiner points if requested
    if args.show_steiner:
        add_steiner_visualization(plotter_before, mesh, partition, steiner_handler, args.steiner_size)
    
    # Add edge visualization if we have the preview info
    if preview_old_edge is not None and preview_target_edge is not None:
        add_edge_visualization(plotter_before, mesh, preview_old_edge, preview_target_edge)
    
    # Add triangle labels for triangles sharing target vertex
    add_target_vertex_triangle_labels(plotter_before, mesh, target_vertex, 
                                     args.label_all, args.apply_zoom, args.zoom_factor,
                                     args.label_font_size)
    
    # Apply camera zoom if requested - focus on target vertex
    if args.apply_zoom:
        target_pos = mesh.vertices[target_vertex]
        apply_camera_zoom(plotter_before, target_pos, args.zoom_factor)
    
    # Set title and show BEFORE (non-blocking, keeps window open while script continues)
    title = f"Region {args.region} - BEFORE Type 1 (Component {args.component_index}"
    if preview_migrating_vp is not None:
        title += f", VP {preview_migrating_vp}"
    title += ")"
    plotter_before.add_title(title, font_size=14)
    plotter_before.show(interactive=False, auto_close=False)
    
    # ========================================================================
    # PERFORM MIGRATION
    # ========================================================================
    
    print("\n" + "="*60)
    print("PERFORMING TYPE 1 MIGRATION")
    print("="*60)
    print(f"Migrating trigger {args.component_index}...")
    print()

    success = orchestrator.execute_single_trigger(selected_trigger)
    if not success:
        print("❌ ERROR: Migration failed!")
        print("⚠ BEFORE figure is displayed. Close it to exit.")
        input("\nPress Enter to close and exit...")
        return

    print("\n" + "="*60)
    print("✓ MIGRATION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"  Trigger vertex: {selected_trigger.vertex}")
    print(f"  Cell transition: {selected_trigger.current_cell} -> {selected_trigger.target_cell}")
    print("="*60)
    print()

    migrating_vp_idx = min(
        selected_trigger.approaching_vps,
        key=lambda vi: migration_utils.compute_boundary_distance(partition.variable_points[vi])
    )
    left_neighbor = None
    right_neighbor = None
    old_edge = None
    target_edge = None
    steiner_handler = orchestrator.steiner_handler
    
    # ========================================================================
    # AFTER STATE
    # ========================================================================
    
    # Re-initialize AreaCalculator with updated data structures
    print("  Initializing AreaCalculator AFTER migration...")
    area_calc_after = AreaCalculator(mesh, partition)
    print()
    
    # Compute area after migration
    lambda_vec_after = partition.get_variable_vector()
    area_info_after = compute_region_area(area_calc_after, args.region, lambda_vec_after)
    
    print("="*60)
    print("AFTER Type 1 Migration")
    print("="*60)
    print()
    print(f"Region {args.region} Geometry (AFTER):")
    print(f"  Interior triangles: {area_info_after['n_interior_triangles']:,} (area: {area_info_after['interior_area']:.4f})")
    print(f"  Boundary triangles: {area_info_after['n_boundary_triangles']:,} (area: {area_info_after['boundary_area']:.4f})")
    print(f"  TOTAL AREA: {area_info_after['total_area']:.6f}")
    print()
    print(f"Area change: {area_info_after['total_area'] - area_info_before['total_area']:.9f}")
    print()
    
    # Get new VP positions after migration
    migrating_vp_after = partition.variable_points[migrating_vp_idx]
    new_edge_migrating = migrating_vp_after.edge
    
    print(f"VP positions after migration:")
    print(f"  Migrating VP {migrating_vp_idx}: {old_edge} → {new_edge_migrating}")
    print(f"    Lambda: {migrating_vp_after.lambda_param:.6f}")
    
    if left_neighbor is not None:
        left_neighbor_vp_after = partition.variable_points[left_neighbor]
        new_edge_left = left_neighbor_vp_after.edge
        print(f"  Left neighbor {left_neighbor}: edge → {new_edge_left}")
        print(f"    Lambda: {left_neighbor_vp_after.lambda_param:.6f}")
    if right_neighbor is not None:
        right_neighbor_vp_after = partition.variable_points[right_neighbor]
        new_edge_right = right_neighbor_vp_after.edge
        print(f"  Right neighbor {right_neighbor}: edge → {new_edge_right}")
        print(f"    Lambda: {right_neighbor_vp_after.lambda_param:.6f}")
    print()
    
    # Create plotter for AFTER
    plotter_after = pv.Plotter()
    
    # Render regions with updated partition
    render_regions(
        plotter_after, mesh, partition, area_calc_after, steiner_handler,
        args.region, args.intense_color
    )
    
    if args.show_vps:
        highlight_vps = [vi for vi in selected_trigger.approaching_vps
                         if partition.variable_points[vi].active]
        vp_labels = [f'VP{vi}' for vi in highlight_vps]
        add_vp_visualization(plotter_after, mesh, partition, highlight_vps, vp_labels, args.vp_size)
    
    # Add Steiner points if requested
    if args.show_steiner:
        add_steiner_visualization(plotter_after, mesh, partition, steiner_handler, args.steiner_size)
    
    # Add edge visualization (show new edges)
    add_edge_visualization(plotter_after, mesh, new_edge_migrating, None)
    
    # Add triangle labels for triangles sharing target vertex
    add_target_vertex_triangle_labels(plotter_after, mesh, target_vertex,
                                     args.label_all, args.apply_zoom, args.zoom_factor,
                                     args.label_font_size)
    
    # Apply camera zoom if requested (focus on target vertex)
    if args.apply_zoom:
        target_pos = mesh.vertices[target_vertex]
        apply_camera_zoom(plotter_after, target_pos, args.zoom_factor)
    
    # Set title and show AFTER
    plotter_after.add_title(
        f"Region {args.region} - AFTER Type 1 (Component {args.component_index}, VP {migrating_vp_idx})", 
        font_size=14
    )
    plotter_after.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Type 1 migration using vertex-collapse strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--solution', required=True,
                       help='Path to refined_contours.h5 file')
    parser.add_argument('--region', type=int, required=True,
                       help='Region/cell index to visualize')
    
    # Migration parameters
    parser.add_argument('--component-index', type=int, default=0,
                       help='Trigger index (within MigrationOrchestrator detection list) to migrate '
                            '(default: 0)')
    parser.add_argument('--boundary-tol', type=float, default=0.1,
                       help='Boundary tolerance for VP detection (default: 0.1)')

    # Visualization options
    parser.add_argument('--state', choices=['before', 'after', 'both'], default='before',
                       help='Which state to show (default: before)')
    parser.add_argument('--intense-color', default='#FF8C42',
                       help='Color for target region (default: bright warm orange #FF8C42)')
    parser.add_argument('--show-vps', action='store_true',
                       help='Show variable points as spheres')
    parser.add_argument('--show-steiner', action='store_true',
                       help='Show Steiner points and void triangles')
    parser.add_argument('--label-all', action='store_true',
                       help='Label all triangles in mesh (methodology triangles: gray box, others: white box; '
                            'recommended with --apply-zoom to reduce clutter)')
    parser.add_argument('--label-font-size', type=int, default=14,
                       help='Font size for triangle labels (default: 14)')
    
    # Camera/zoom options
    parser.add_argument('--apply-zoom', action='store_true',
                       help='Apply zoom to focus on migration area')
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
