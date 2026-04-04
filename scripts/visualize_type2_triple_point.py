#!/usr/bin/env python3
"""
Visualize Type 2 migration (Triple Point Migration).

Clean, focused implementation for testing and debugging Type 2 strategy.
Only includes necessary code - shows BEFORE state for now.

Usage:
    python scripts/visualize_type2_triple_point.py \
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

from src.mesh.tri_mesh import TriMesh
from src.partition.contour_partition import PartitionContour
from src.mesh.mesh_topology import MeshTopology
from src.partition.steiner_handler import SteinerHandler, TriplePoint
from src.partition.area_calculator import AreaCalculator

# Pre-parse --use-legacy before conditional imports
import argparse as _argparse
from src.migration.migration_orchestrator import MigrationOrchestrator, MigrationConfig

# Import data loading from the reference script
from src.pipeline.io import load_partition_from_refined_file


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
    
    vertex_labels = partition.vertex_labels
    vertex_cell = int(vertex_labels[vertex_idx])
    
    return vertex_cell == cell_idx


def compute_cell_portion_in_triangle_simple(
    mesh: TriMesh,
    partition: PartitionContour,
    tri_idx: int,
    cell_idx: int,
    tri_idx_to_segment: Optional[Dict] = None,
    debug: bool = False
) -> Optional[np.ndarray]:
    """
    Compute polygon for cell portion of boundary triangle (simplified - no crossing cache).
    
    Returns:
        (N, 3) array of polygon vertices, or None if triangle doesn't contribute
    """
    face = mesh.faces[tri_idx]
    v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
    vertex_labels = partition.vertex_labels
    labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
    
    if debug and tri_idx in [18150, 18152]:
        print(f"\n{'='*60}")
        print(f"DEBUG: compute_cell_portion for Triangle {tri_idx}, Cell {cell_idx}")
        print(f"{'='*60}")
        print(f"Triangle vertices: {v1}, {v2}, {v3}")
        print(f"Vertex labels: {labels}")
    
    # Check if triangle is boundary via VPs (only path - no crossing cache)
    is_boundary = False
    vp_positions = []
    vp_indices_in_tri = []
    
    if tri_idx_to_segment is not None:
        tri_seg = tri_idx_to_segment.get(tri_idx)
        if tri_seg:
            if debug and tri_idx in [18150, 18152]:
                print(f"Triangle segment found with VPs: {tri_seg.var_point_indices}")
            
            for vp_idx in tri_seg.var_point_indices:
                vp = partition.variable_points[vp_idx]
                if not getattr(vp, 'active', True):
                    continue
                
                if debug and tri_idx in [18150, 18152]:
                    print(f"  VP {vp_idx}:")
                    print(f"    edge: {vp.edge}")
                    print(f"    belongs_to_cells: {vp.belongs_to_cells}")
                    print(f"    cell {cell_idx} in belongs_to_cells? {cell_idx in vp.belongs_to_cells}")
                
                if cell_idx in vp.belongs_to_cells:
                    is_boundary = True
                    vp_pos = vp.evaluate(mesh.vertices)
                    vp_positions.append(vp_pos)
                    vp_indices_in_tri.append(vp_idx)
                    
                    if debug and tri_idx in [18150, 18152]:
                        print(f"    → INCLUDED (position: {vp_pos})")
    else:
        for tri_seg in partition.triangle_segments:
            if tri_seg.triangle_idx == tri_idx:
                for vp_idx in tri_seg.var_point_indices:
                    vp = partition.variable_points[vp_idx]
                    if not getattr(vp, 'active', True):
                        continue
                    if cell_idx in vp.belongs_to_cells:
                        is_boundary = True
                        vp_positions.append(vp.evaluate(mesh.vertices))
                        vp_indices_in_tri.append(vp_idx)
                break
    
    if debug and tri_idx in [18150, 18152]:
        print(f"Is boundary for cell {cell_idx}? {is_boundary}")
        print(f"VP positions collected: {len(vp_positions)} (VPs: {vp_indices_in_tri})")
    
    if not is_boundary:
        return None
    
    # Standard case: Use VP positions and indicator_functions
    vertices_inside = []
    
    if len(vp_positions) >= 2:
        # 2+ VPs define the boundary line
        vp_pos1 = vp_positions[0]
        vp_pos2 = vp_positions[1]
        
        if debug and tri_idx in [18150, 18152]:
            print(f"Using VP-based vertex selection (2+ VPs)")
            print(f"Checking which triangle vertices are on cell {cell_idx} side:")
        
        for v in [v1, v2, v3]:
            v_pos = mesh.vertices[v]
            is_inside = _vertex_on_cell_side_for_viz(v_pos, vp_pos1, vp_pos2, cell_idx, 
                                           mesh, tri_idx, partition)
            
            if debug and tri_idx in [18150, 18152]:
                print(f"  Vertex {v} (cell {vertex_labels[v]}): {'INSIDE' if is_inside else 'OUTSIDE'}")
            
            if is_inside:
                vertices_inside.append(v_pos)
    else:
        # Fallback to indicator_functions
        if debug and tri_idx in [18150, 18152]:
            print(f"Using indicator_functions fallback (< 2 VPs)")
        
        for v, lab in zip([v1, v2, v3], labels):
            if lab == cell_idx:
                vertices_inside.append(mesh.vertices[v])
                
                if debug and tri_idx in [18150, 18152]:
                    print(f"  Vertex {v} included (label matches cell {cell_idx})")
    
    # Construct polygon
    all_points = vertices_inside + vp_positions
    
    if debug and tri_idx in [18150, 18152]:
        print(f"Final polygon construction:")
        print(f"  Vertices inside: {len(vertices_inside)}")
        print(f"  VP positions: {len(vp_positions)}")
        print(f"  Total points: {len(all_points)}")
        
        if len(all_points) < 3:
            print(f"  ❌ RETURNING None - not enough points for polygon!")
        else:
            print(f"  ✓ Returning polygon with {len(all_points)} vertices")
    
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
    
    vp_positions_for_steiner = [partition.evaluate_variable_point(vi) for vi in triple_point.var_point_indices]
    steiner_pos = triple_point.compute_steiner_point(vp_positions=vp_positions_for_steiner)
    
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
        # Enable debug for problematic triangles
        debug_enabled = tri_idx in [18150, 18152]
        
        poly_vertices = compute_cell_portion_in_triangle_simple(
            mesh, partition, tri_idx, cell_idx, tri_idx_to_segment, debug=debug_enabled
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
    opacity: float = 1.0,
):
    """Render all regions with precise boundaries (simplified for vertex-collapse).
    
    Args:
        backface_culling: If True, hide back-facing triangles (default: False to match Type 1)
        opacity: Opacity for non-highlighted regions (default: 1.0)
    """
    # Pastel color palette (matches visualize_partition.py)
    pale_palette = [
        '#FFE5B4', '#E0BBE4', '#FFDAC1', '#B5EAD7', '#C7CEEA',
        '#FFB7B2', '#FFDFD3', '#E2F0CB', '#B4F8C8', '#A0C4FF',
        '#FFC6FF', '#FFCFD2', '#FDE2E4', '#FAD2E1', '#BEE1E6'
    ]
    
    print(f"  Rendering all {partition.n_cells} regions (precise boundaries, opacity: {opacity})...")
    
    for cell_idx in range(partition.n_cells):
        if target_region is not None and cell_idx == target_region:
            color = target_color
            cell_opacity = 1.0
        else:
            color = pale_palette[cell_idx % len(pale_palette)]
            cell_opacity = opacity
        
        # Use simplified rendering function (no crossing cache)
        render_single_region_simple(
            plotter, mesh, partition, area_calc, steiner_handler,
            cell_idx, color, opacity=cell_opacity, backface_culling=backface_culling
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
        vp_positions_for_steiner = [partition.evaluate_variable_point(vi) for vi in tp.var_point_indices]
        steiner_pt = tp.compute_steiner_point(vp_positions=vp_positions_for_steiner)
        
        # Add Steiner point as orange sphere
        sphere = pv.Sphere(radius=steiner_size, center=steiner_pt)
        plotter.add_mesh(sphere, color='orange', opacity=1.0)
        
        vp_positions = []
        for vp_idx in tp.var_point_indices:
            vp = partition.variable_points[vp_idx]
            if not getattr(vp, 'active', True):
                continue
            vp_pos = vp.evaluate(mesh.vertices)
            vp_positions.append(vp_pos)
        
        if len(vp_positions) == 3:
            # Draw void triangle edges (cyan)
            for i in range(3):
                p1 = vp_positions[i]
                p2 = vp_positions[(i + 1) % 3]
                line = pv.Line(p1, p2)
                plotter.add_mesh(line, color='cyan', line_width=2, opacity=0.7)


def add_triangle_label(
    plotter: pv.Plotter,
    mesh: TriMesh,
    tri_idx: int,
    label_text: str,
    color: str = 'black',
    font_size: int = 12,
    shape_color: str = 'gray',
    shape_opacity: float = 0.8
):
    """Add a text label at the centroid of a triangle with customizable styling."""
    # Get triangle vertices
    face = mesh.faces[tri_idx]
    v0, v1, v2 = [mesh.vertices[int(i)] for i in face]
    
    # Compute centroid
    centroid = (v0 + v1 + v2) / 3.0
    
    # Add text at centroid with customizable box background
    plotter.add_point_labels(
        [centroid],
        [label_text],
        font_size=font_size,
        text_color=color,
        font_family='arial',
        shape_color=shape_color,
        shape_opacity=shape_opacity,
        always_visible=True
    )


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
                font_size=9, text_color='black',
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
    
    # Add labels using the standard add_triangle_label function
    for tri_idx in triangles_to_label:
        add_triangle_label(plotter, mesh, tri_idx, f"T{tri_idx}", color='black', font_size=12)


def compute_steiner_distance_to_boundary(triple_point: TriplePoint, mesh: TriMesh,
                                        partition: PartitionContour) -> float:
    """
    Compute distance from Steiner point to closest edge of the triangle.
    
    Args:
        triple_point: TriplePoint object
        mesh: TriMesh
        partition: PartitionContour (for Steiner point computation)
    
    Returns:
        Distance to closest edge
    """
    vp_positions_for_steiner = [partition.evaluate_variable_point(vi) for vi in triple_point.var_point_indices]
    steiner_pos = triple_point.compute_steiner_point(vp_positions=vp_positions_for_steiner)
    tri_idx = triple_point.triangle_idx
    face = mesh.faces[tri_idx]
    
    v0, v1, v2 = [mesh.vertices[int(i)] for i in face]
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    
    min_dist = float('inf')
    for edge_v1, edge_v2 in edges:
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
    
    # Load migration history if present
    print("Checking for Type 2 migration history...")
    migration_history = None
    try:
        import h5py
        from src.migration.type2_migration_io import load_type2_migration_history
        with h5py.File(args.solution, 'r') as f:
            migration_history = load_type2_migration_history(f)
        if len(migration_history.records) > 0:
            print(f"✓ Found migration history: {len(migration_history.records)} triple points tracked")
            for orig_tri, record in migration_history.records.items():
                print(f"  Triangle {orig_tri}: Path = {record.triangle_sequence}")
                print(f"                    Iterations = {record.iteration_sequence}")
        else:
            print("  No migration history in file")
    except Exception as e:
        print(f"  No migration history found ({e})")
    print()
    
    # Initialize topology components
    mesh_topology = MeshTopology(mesh)
    if _preargs.use_legacy:
        switcher = TopologySwitcher(mesh, partition, mesh_topology)
        if migration_history is not None:
            switcher.type2_migration_history = migration_history
            print("✓ Migration history attached to topology switcher")
            print()
    else:
        orchestrator = MigrationOrchestrator(
            partition, mesh, mesh_topology,
            MigrationConfig(delta=args.boundary_tol)
        )
    
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
        dist = compute_steiner_distance_to_boundary(tp, mesh, partition)
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
    
    dist = compute_steiner_distance_to_boundary(selected_tp, mesh, partition)
    print(f"  Steiner distance to boundary: {dist:.6f}")
    print()
    
    # ========================================================================
    # TYPE 2 MIGRATION ANALYSIS (New Strategy)
    # ========================================================================
    
    print("="*80)
    print("TYPE 2 MIGRATION ANALYSIS (Topological VP Selection)")
    print("="*80)
    print()
    
    if _preargs.use_legacy:
        # ------- Legacy path: TopologySwitcher analysis -------
        migration_result = switcher.apply_type2_switch_v3(steiner_handler, args.triple_point_index)

        if not migration_result['success']:
            print(f"❌ ERROR: Migration analysis failed: {migration_result.get('error', 'Unknown error')}")
            return

        if 'warning' in migration_result:
            print(f"⚠️  WARNING: {migration_result['warning']}")
            print()

        print("Migration Analysis Results:")
        print(f"  Triple triangle: {migration_result['triple_triangle_idx']}")
        print(f"  Target triangle: {migration_result['target_triangle_idx']}")
        print(f"  Shared edge (Steiner approaching): {migration_result['shared_edge']}")
        print(f"  Target edge (free edge in target triangle): {migration_result['target_edge']}")
        print()

        print("VP Classification:")
        print(f"  Anchor VP: {migration_result['anchor_vp_idx']} (on shared edge)")
        print(f"  Migrating VP: {migration_result['migrating_vp_idx']} (will move to target edge)")
        print(f"    Edge: {migration_result['migrating_vp_edge']}")
        if migration_result['shared_vertex'] is not None:
            print(f"    Shared vertex with target edge: {migration_result['shared_vertex']}")
        print(f"  Non-migrating VP: {migration_result['non_migrating_vp_idx']} (stays in place)")
        print(f"    Edge: {migration_result['non_migrating_vp_edge']}")
        print()

        print("Connectivity Analysis:")
        for vp_idx, info in migration_result['connectivity_analysis'].items():
            print(f"  VP {vp_idx}:")
            print(f"    Edge: {info['edge']}")
            print(f"    Shared vertices with target edge: {info['shared_vertices']} (count={info['num_shared']})")
        print()

        migrating_vp_idx = migration_result['migrating_vp_idx']
        triple_vp_set = set(selected_tp.var_point_indices)

        def get_neighbors(vp_idx):
            """Get all neighbors of a VP via boundary_segments."""
            neighbors = []
            for seg in partition.boundary_segments:
                if vp_idx == seg.vp_idx_1:
                    neighbors.append(seg.vp_idx_2)
                elif vp_idx == seg.vp_idx_2:
                    neighbors.append(seg.vp_idx_1)
            return neighbors

        all_neighbors = get_neighbors(migrating_vp_idx)
        direct_outer_neighbors = [vp for vp in all_neighbors if vp not in triple_vp_set]

        second_level_neighbors = []
        for direct_neighbor in direct_outer_neighbors:
            neighbor_neighbors = get_neighbors(direct_neighbor)
            second_level = [vp for vp in neighbor_neighbors if vp != migrating_vp_idx]
            second_level_neighbors.extend(second_level)

        outer_neighbors = direct_outer_neighbors + second_level_neighbors

        print(f"Migrating VP {migrating_vp_idx} Neighbor Analysis:")
        print(f"  All immediate neighbors: {all_neighbors}")
        print(f"  Triple triangle VPs (excluded): {list(triple_vp_set)}")
        print(f"  Direct outer neighbor (Level 1): {direct_outer_neighbors}")
        print(f"  Second-level neighbors (Level 2): {second_level_neighbors}")
        print(f"  Total outer neighbors to display: {outer_neighbors}")
        print()

        if outer_neighbors:
            print("Outer Neighbor Details:")
            for i, vp_idx in enumerate(outer_neighbors):
                vp = partition.variable_points[vp_idx]
                level = "Level 1 (direct)" if vp_idx in direct_outer_neighbors else "Level 2 (indirect)"
                print(f"  Outer Neighbor {i+1} [{level}]: VP {vp_idx}")
                print(f"    Edge: {vp.edge}")
                print(f"    Lambda: {vp.lambda_param:.6f}")
            print()

        print("="*80)
        print("KEY TRIANGLES FOR TYPE 2 MIGRATION")
        print("="*80)
        print()

        triangle_result = switcher._identify_type2_migration_triangles(migration_result)

        if not triangle_result['success']:
            print(f"❌ ERROR: Triangle identification failed: {triangle_result.get('error', 'Unknown error')}")
            return

        print("Triangle Identification Results:")
        print(f"  T_second_VP: {triangle_result['T_second_VP']}")
        print(f"    Contains segment: VP{triangle_result['vp_context']['direct_outer_neighbor']} -- VP{triangle_result['vp_context']['second_level_neighbor']}")
        print(f"    Free edge: {triangle_result['T_second_VP_free_edge']}")
        print()
        print(f"  T_adjacent_to_T_second: {triangle_result['T_adjacent_to_T_second']}")
        print(f"    Shares free edge {triangle_result['T_second_VP_free_edge']} with T_second_VP")
        print()
        print(f"  T_shared_edge_with_target: {triangle_result['T_shared_edge_with_target']}")
        print(f"    Shares target edge {migration_result['target_edge']} with target triangle")
        print()

    else:
        # ------- New path: MigrationOrchestrator -------
        print("Using MigrationOrchestrator trigger-based detection")
        print()

        detection = orchestrator.detect_all_triggers(delta=args.boundary_tol)
        print(f"  Detected {len(detection.type2_triggers)} Type 2 triggers")
        for i, trig in enumerate(detection.type2_triggers):
            print(f"    [{i}] {trig}")
        print()

        migration_result = None
        triangle_result = {'success': False}
        outer_neighbors = []
        migrating_vp_idx = None
        direct_outer_neighbors = []
    
    # ========================================================================
    # BEFORE STATE
    # ========================================================================
    
    print("="*60)
    print("BEFORE Type 2 Migration")
    print("="*60)
    print()
    
    # Initialize AreaCalculator (for rendering only)
    print("  Initializing AreaCalculator...")
    area_calc = AreaCalculator(mesh, partition)
    print()
    
    # Get Steiner position for camera focus
    vp_positions_for_steiner = [partition.evaluate_variable_point(vi) for vi in selected_tp.var_point_indices]
    steiner_pos = selected_tp.compute_steiner_point(vp_positions=vp_positions_for_steiner)
    
    # ========================================================================
    # DETERMINE WHICH STATES TO SHOW
    # ========================================================================
    
    show_before = args.state in ['before', 'both']
    show_after = args.state in ['after', 'both']
    
    # ========================================================================
    # BEFORE STATE
    # ========================================================================
    
    if show_before:
        print("="*80)
        print("RENDERING BEFORE STATE")
        print("="*80)
        
        # Create separate plotter for BEFORE
        plotter_before = pv.Plotter()
        plotter_before.add_title(
            f"Type 2 BEFORE: Triple Point at Triangle {selected_tp.triangle_idx} (triple-point-index={args.triple_point_index})",
            font_size=12
        )
        
        # Render all regions with precise boundaries
        render_regions(
            plotter_before, mesh, partition, area_calc, steiner_handler,
            target_region=None,
            backface_culling=args.enable_backface_culling,
            opacity=args.opacity
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
            
            # Add outer neighbor VPs
            if outer_neighbors:
                migrating_color_idx = None
                for i, vp_idx in enumerate(selected_tp.var_point_indices):
                    if vp_idx == migrating_vp_idx:
                        migrating_color_idx = i
                        break
                
                if migrating_color_idx == 0:
                    outer_colors = ['darkred', 'indianred']
                elif migrating_color_idx == 1:
                    outer_colors = ['darkblue', 'lightblue']
                elif migrating_color_idx == 2:
                    outer_colors = ['darkgreen', 'lightgreen']
                else:
                    outer_colors = ['purple', 'violet']
                
                while len(outer_colors) < len(outer_neighbors):
                    outer_colors.append('gray')
                
                outer_labels = []
                for i, vp_idx in enumerate(outer_neighbors):
                    vp = partition.variable_points[vp_idx]
                    outer_labels.append(f"Outer{i+1}\nidx={vp_idx}\nedge={vp.edge}")
                
                print("Outer Neighbor VPs:")
                for i, vp_idx in enumerate(outer_neighbors):
                    print(f"  {outer_colors[i].capitalize()} VP: {vp_idx}")
                print()
                
                add_vp_visualization(
                    plotter_before, partition, mesh,
                    outer_neighbors,
                    outer_colors[:len(outer_neighbors)],
                    outer_labels,
                    args.vp_size
                )
        
        # Add target edge highlighting
        target_edge = migration_result['target_edge']
        v1_pos = mesh.vertices[target_edge[0]]
        v2_pos = mesh.vertices[target_edge[1]]
        target_edge_line = pv.Line(v1_pos, v2_pos)
        plotter_before.add_mesh(target_edge_line, color='limegreen', line_width=5, opacity=1.0)
        print(f"Target edge {target_edge} highlighted in green")
        print()
        
        # Add T_second_VP free edge highlighting
        if triangle_result['success']:
            free_edge = triangle_result['T_second_VP_free_edge']
            v1_pos_free = mesh.vertices[free_edge[0]]
            v2_pos_free = mesh.vertices[free_edge[1]]
            free_edge_line = pv.Line(v1_pos_free, v2_pos_free)
            plotter_before.add_mesh(free_edge_line, color='gold', line_width=4, opacity=1.0)
            print(f"T_second_VP free edge {free_edge} highlighted in yellow/gold")
            print()
        
        # Add triangle labels
        print("Adding triangle labels...")
        
        # Label the 6 methodology triangles explicitly (not using add_triple_point_triangle_labels)
        if triangle_result['success']:
            # Find T_first_VP using segment_to_triangle map
            migrating_vp_idx = migration_result['migrating_vp_idx']
            # Get first_level neighbor from outer_neighbors list
            first_level_vp_idx = outer_neighbors[0] if outer_neighbors else None
            
            T_first_VP = None
            if first_level_vp_idx is not None:
                seg_key = tuple(sorted([migrating_vp_idx, first_level_vp_idx]))
                T_first_VP = partition.segment_to_triangle.get(seg_key)
            
            methodology_triangles = {
                migration_result['triple_triangle_idx']: f"T{migration_result['triple_triangle_idx']}",
                migration_result['target_triangle_idx']: f"T{migration_result['target_triangle_idx']}",
                triangle_result['T_second_VP']: f"T{triangle_result['T_second_VP']}",
                triangle_result['T_adjacent_to_T_second']: f"T{triangle_result['T_adjacent_to_T_second']}",
                triangle_result['T_shared_edge_with_target']: f"T{triangle_result['T_shared_edge_with_target']}"
            }
            
            if T_first_VP is not None:
                methodology_triangles[T_first_VP] = f"T{T_first_VP}"
            
            # Label triangles based on --label-all flag
            if args.label_all:
                # Use spatial filtering when zoomed in for performance
                if args.apply_zoom:
                    # Only label triangles within distance of zoom focus
                    max_distance = args.zoom_factor * 50  # Increased from 10 to capture more triangles
                    print(f"  Labeling triangles within {max_distance:.4f} units of focus point...")
                    
                    # Batch collect centroids and labels
                    methodology_centroids = []
                    methodology_labels = []
                    other_centroids = []
                    other_labels = []
                    
                    labeled_count = 0
                    for tri_idx in range(len(mesh.faces)):
                        face = mesh.faces[tri_idx]
                        v0, v1, v2 = [mesh.vertices[int(i)] for i in face]
                        centroid = (v0 + v1 + v2) / 3.0
                        
                        # Check if within distance OR is methodology triangle
                        dist = np.linalg.norm(centroid - steiner_pos)
                        if dist < max_distance or tri_idx in methodology_triangles:
                            if tri_idx in methodology_triangles:
                                methodology_centroids.append(centroid)
                                methodology_labels.append(methodology_triangles[tri_idx])
                            else:
                                other_centroids.append(centroid)
                                other_labels.append(f"T{tri_idx}")
                            labeled_count += 1
                    
                    print(f"  Labeling {labeled_count} triangles (filtered from {len(mesh.faces)})...")
                    print(f"    Methodology triangles: {len(methodology_centroids)}")
                    print(f"    Other triangles: {len(other_centroids)}")
                    
                    # Batch add methodology triangles (gray background)
                    if methodology_centroids:
                        plotter_before.add_point_labels(
                            np.array(methodology_centroids),
                            methodology_labels,
                            font_size=args.label_font_size,
                            text_color='black',
                            shape_color='gray',
                            shape_opacity=0.8,
                            always_visible=True
                        )
                    
                    # Batch add other triangles (white background)
                    if other_centroids:
                        plotter_before.add_point_labels(
                            np.array(other_centroids),
                            other_labels,
                            font_size=args.label_font_size,
                            text_color='black',
                            shape_color='white',
                            shape_opacity=0.9,
                            always_visible=True
                        )
                else:
                    # No zoom - use batching only (no spatial filter)
                    print(f"  Labeling all {len(mesh.faces)} triangles using batched approach...")
                    
                    # Batch collect centroids and labels
                    methodology_centroids = []
                    methodology_labels = []
                    other_centroids = []
                    other_labels = []
                    
                    for tri_idx in range(len(mesh.faces)):
                        face = mesh.faces[tri_idx]
                        v0, v1, v2 = [mesh.vertices[int(i)] for i in face]
                        centroid = (v0 + v1 + v2) / 3.0
                        
                        if tri_idx in methodology_triangles:
                            methodology_centroids.append(centroid)
                            methodology_labels.append(methodology_triangles[tri_idx])
                        else:
                            other_centroids.append(centroid)
                            other_labels.append(f"T{tri_idx}")
                    
                    # Batch add methodology triangles
                    if methodology_centroids:
                        plotter_before.add_point_labels(
                            np.array(methodology_centroids),
                            methodology_labels,
                            font_size=args.label_font_size,
                            text_color='black',
                            shape_color='gray',
                            shape_opacity=0.8,
                            always_visible=True
                        )
                    
                    # Batch add other triangles
                    if other_centroids:
                        plotter_before.add_point_labels(
                            np.array(other_centroids),
                            other_labels,
                            font_size=args.label_font_size,
                            text_color='black',
                            shape_color='white',
                            shape_opacity=0.6,
                            always_visible=True
                        )
            else:
                # Original behavior: only methodology triangles
                for tri_idx, label in methodology_triangles.items():
                    add_triangle_label(plotter_before, mesh, tri_idx, label, 
                                     color='black', font_size=args.label_font_size,
                                     shape_color='gray', shape_opacity=0.8)
        
        print("Triangle labels added")
        print()
        
        # Apply zoom
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
        
        print("✓ BEFORE state rendered")
        print()
        
        # Show BEFORE window
        if args.state == 'both':
            # Non-blocking: keep window open while showing AFTER
            print("Opening PyVista window (BEFORE state)...")
            plotter_before.show(interactive=False, auto_close=False)
        else:
            # Blocking: only showing BEFORE
            print("Opening PyVista window (BEFORE state)...")
            plotter_before.show()

    
    # ========================================================================
    # AFTER STATE
    # ========================================================================
    
    if show_after:
        print("="*80)
        print("RENDERING AFTER STATE (Applying Type 2 Migration)")
        print("="*80)
        
        import copy
        partition_after = copy.deepcopy(partition)
        mesh_topology_after = MeshTopology(mesh)
        steiner_handler_after = SteinerHandler(mesh, partition_after)
        
        if _preargs.use_legacy:
            switcher_after = TopologySwitcher(mesh, partition_after, mesh_topology_after)
            print("Applying apply_type2_switch_v4()...")
            migration_v4_result = switcher_after.apply_type2_switch_v4(
                steiner_handler_after,
                args.triple_point_index,
                distance_preservation='preserve'
            )
        else:
            orchestrator_after = MigrationOrchestrator(
                partition_after, mesh, mesh_topology_after,
                MigrationConfig(delta=args.boundary_tol)
            )
            orchestrator_after.detect_all_triggers(delta=args.boundary_tol)
            print("Applying MigrationOrchestrator.execute_migrations()...")
            migration_v4_result = orchestrator_after.execute_migrations(mode='batch')
        
        if not migration_v4_result['success']:
            print()
            print("="*80)
            print("⚠️  ERROR: Type 2 migration failed!")
            print("="*80)
            print(f"Reason: {migration_v4_result.get('error', 'Unknown error')}")
            print()
            print("The BEFORE state has been rendered successfully.")
            print("Since AFTER state cannot be generated, only BEFORE will be displayed.")
            print("="*80)
            print()
            
            # If we already showed BEFORE in non-blocking mode, show it again in blocking mode
            if args.state == 'both' and show_before:
                print("Showing BEFORE state window (migration failed, AFTER unavailable)...")
                plotter_before.show()
            
            # Exit gracefully
            print("\n" + "="*80)
            print("Visualization complete (BEFORE state only)")
            print("="*80)
            return
        
        print()
        print("✓ Type 2 migration applied successfully")
        print(f"  VP count change: +{migration_v4_result['vp_count_change']}")
        print(f"  Segment count change: +{migration_v4_result['segment_count_change']}")
        print(f"  New triple point in triangle: {migration_v4_result['target_triangle_idx']}")
        print()
        
        # CRITICAL: Reinitialize SteinerHandler to detect NEW triple points
        print("Reinitializing SteinerHandler to detect new triple points...")
        steiner_handler_after = SteinerHandler(mesh, partition_after)
        print(f"✓ Detected {len(steiner_handler_after.triple_points)} triple points after migration")
        print()
        
        # Reinitialize area calculator for updated partition
        area_calc_after = AreaCalculator(mesh, partition_after)
        
        # Create separate plotter for AFTER
        plotter_after = pv.Plotter()
        plotter_after.add_title(
            f"Type 2 AFTER: New Triple Point at Triangle {migration_v4_result['target_triangle_idx']}",
            font_size=12
        )
        
        # Render all regions with updated partition
        render_regions(
            plotter_after, mesh, partition_after, area_calc_after, steiner_handler_after,
            target_region=None,
            backface_culling=args.enable_backface_culling,
            opacity=args.opacity
        )
        
        # Add Steiner visualization (new triple point)
        if args.show_steiner:
            add_steiner_visualization(plotter_after, steiner_handler_after, partition_after, mesh, 
                                    args.steiner_size)
        
        # Add VP visualization for AFTER state
        if args.show_vps:
            # Get VP indices from migration result
            vp_close_idx = migration_v4_result['vp_close_to_steiner_idx']
            steiner_vp_idx = migration_v4_result['steiner_vp_idx']
            outer_neighbor_idx = migration_v4_result['outer_neighbor_vp_close_idx']
            stationary_idx = migration_v4_result['stationary_vp_idx']
            migrated_idx = migration_v4_result['migrating_vp_idx']
            
            # Map VP indices to their BEFORE colors (preserve identity)
            # BEFORE: triple_vp_indices = [vp1, vp2, vp3] with colors ['red', 'blue', 'green']
            # Need to find which VP is which
            original_triple_vps = selected_tp.var_point_indices
            vp_color_map = {}
            original_colors = ['red', 'blue', 'green']
            for i, vp_idx in enumerate(original_triple_vps):
                vp_color_map[vp_idx] = original_colors[i]
            
            # Get colors for each VP based on original identity
            vp_close_color = vp_color_map.get(vp_close_idx, 'yellow')
            migrated_color = vp_color_map.get(migrated_idx, 'blue')
            stationary_color = vp_color_map.get(stationary_idx, 'red')
            
            # Determine color for outer_neighbor based on vp_close_to_steiner color
            if vp_close_color == 'red':
                outer_neighbor_color = 'darkred'
            elif vp_close_color == 'blue':
                outer_neighbor_color = 'darkblue'
            elif vp_close_color == 'green':
                outer_neighbor_color = 'darkgreen'
            else:
                outer_neighbor_color = 'purple'
            
            # Show new triple point VPs in new triple triangle
            new_triple_vps = [vp_close_idx, steiner_vp_idx, outer_neighbor_idx]
            new_triple_colors = [vp_close_color, 'orange', outer_neighbor_color]
            new_triple_labels = [
                f"VP_close\nidx={vp_close_idx}\n(moved)",
                f"Steiner_VP\nidx={steiner_vp_idx}\n(NEW!)",
                f"Outer\nidx={outer_neighbor_idx}"
            ]
            
            print("New Triple Point VPs:")
            for i, vp_idx in enumerate(new_triple_vps):
                print(f"  {new_triple_colors[i].capitalize()} VP: {vp_idx}")
                vp = partition_after.variable_points[vp_idx]
                print(f"    Edge: {vp.edge}")
                print(f"    Lambda: {vp.lambda_param:.6f}")
            print()
            
            add_vp_visualization(
                plotter_after, partition_after, mesh,
                new_triple_vps,
                new_triple_colors,
                new_triple_labels,
                args.vp_size
            )
            
            # Show other key VPs (stationary and migrated)
            other_vps = [stationary_idx, migrated_idx]
            other_colors = [stationary_color, migrated_color]
            other_labels = [
                f"Stationary\nidx={stationary_idx}",
                f"Migrated\nidx={migrated_idx}\n(moved)"
            ]
            
            # Also show the outer neighbors of migrated VP if available
            if outer_neighbors:
                # Color variations based on migrated VP color
                if migrated_color == 'red':
                    outer_migrated_colors = ['darkred', 'indianred']
                elif migrated_color == 'blue':
                    outer_migrated_colors = ['darkblue', 'lightblue']
                elif migrated_color == 'green':
                    outer_migrated_colors = ['darkgreen', 'lightgreen']
                else:
                    outer_migrated_colors = ['purple', 'violet']
                
                # Add outer neighbors to visualization
                for i, vp_idx in enumerate(outer_neighbors[:2]):  # Show up to 2
                    other_vps.append(vp_idx)
                    other_colors.append(outer_migrated_colors[i] if i < len(outer_migrated_colors) else 'gray')
                    other_labels.append(f"Outer{i+1}\nidx={vp_idx}")
            
            print("Other Key VPs:")
            for i, vp_idx in enumerate(other_vps):
                print(f"  {other_colors[i].capitalize()} VP: {vp_idx}")
                vp = partition_after.variable_points[vp_idx]
                print(f"    Edge: {vp.edge}")
            print()
            
            add_vp_visualization(
                plotter_after, partition_after, mesh,
                other_vps,
                other_colors,
                other_labels,
                args.vp_size
            )
        
        # ========================================================================
        # DEBUG: Check for white regions (triangles without proper cell assignment)
        # ========================================================================
        
        print("="*80)
        print("DEBUG: Checking for rendering issues (white polygons)")
        print("="*80)
        
        # Check triangle_segments consistency
        triangles_with_vps = set()
        for tri_seg in partition_after.triangle_segments:
            if len(tri_seg.var_point_indices) > 0:
                triangles_with_vps.add(tri_seg.triangle_idx)
        
        print(f"Triangles with VPs: {len(triangles_with_vps)}")
        
        vertex_labels = partition_after.vertex_labels
        indicator_max_values = np.max(partition_after.indicator_functions, axis=1)
        ambiguous_vertices = np.where(indicator_max_values < 0.9)[0]  # Vertices with weak cell assignment
        
        if len(ambiguous_vertices) > 0:
            print(f"⚠️  Found {len(ambiguous_vertices)} vertices with weak cell assignment:")
            for v_idx in ambiguous_vertices[:10]:  # Show first 10
                print(f"  Vertex {v_idx}: max value = {indicator_max_values[v_idx]:.3f}")
        else:
            print("✓ All vertices have strong cell assignments")
        
        # Check for triangles near the migration area
        affected_tri_set = set(migration_v4_result.get('affected_triangles', []))
        print(f"Affected triangles (rebuilt): {len(affected_tri_set)}")
        
        # Check for boundary triangles that might not be fully categorized
        boundary_tri_count = 0
        for tri_seg in partition_after.triangle_segments:
            if 0 < len(tri_seg.var_point_indices) < 3:
                boundary_tri_count += 1
        
        print(f"Boundary triangles (with 1-2 VPs): {boundary_tri_count}")
        print("="*80)
        print()
        
        # ========================================================================
        # DEBUG: Check which cells include problematic triangles as boundaries
        # ========================================================================
        print("="*80)
        print("DEBUG: Which cells claim triangles 18150 and 18152 as boundaries?")
        print("="*80)
        
        for tri_idx in [18150, 18152]:
            cells_with_this_tri = []
            for cell_idx in range(partition_after.n_cells):
                boundary_tris = area_calc_after.cell_boundary_triangles.get(cell_idx, [])
                if tri_idx in boundary_tris:
                    cells_with_this_tri.append(cell_idx)
            print(f"Triangle {tri_idx}: claimed by cells {cells_with_this_tri}")
        
        print("="*80)
        print()
        
        # ========================================================================
        # DEBUG: Inspect the 6 methodology triangles in detail
        # ========================================================================
        print("="*80)
        print("DEBUG: Inspecting 6 methodology triangles in AFTER state")
        print("="*80)
        
        # Build methodology triangles list
        methodology_tris = [
            migration_result['triple_triangle_idx'],
            migration_result['target_triangle_idx'],
            triangle_result['T_second_VP'],
            triangle_result['T_adjacent_to_T_second'],
            triangle_result['T_shared_edge_with_target']
        ]
        
        # Add T_first_VP if we found it
        if T_first_VP is not None:
            methodology_tris.append(T_first_VP)
        
        for tri_idx in sorted(methodology_tris):
            tri_segs = [ts for ts in partition_after.triangle_segments if ts.triangle_idx == tri_idx]
            if tri_segs:
                ts = tri_segs[0]
                print(f"Triangle {tri_idx}:")
                print(f"  VP indices: {ts.var_point_indices}")
                print(f"  Boundary edges: {ts.boundary_edges}")
                print(f"  Vertex labels: {[partition_after.vertex_labels[vi] for vi in ts.vertex_indices]}")
                
                # Check each VP
                for vp_idx in ts.var_point_indices:
                    vp = partition_after.variable_points[vp_idx]
                    print(f"    VP {vp_idx}: edge={vp.edge}, λ={vp.lambda_param:.6f}")
                    print(f"      belongs_to_cells: {vp.belongs_to_cells}")
                    
                    # Check if edge is in triangle
                    if vp.edge not in ts.boundary_edges:
                        print(f"      ⚠️  WARNING: VP edge {vp.edge} NOT in triangle's boundary_edges!")
            else:
                print(f"Triangle {tri_idx}: NO triangle_segment entry!")
        
        print("="*80)
        print()
        
        # Add triangle labels (same as BEFORE state)
        if triangle_result['success']:
            print("Adding triangle labels...")
            
            # Find T_first_VP using segment_to_triangle map from BEFORE partition
            # (not AFTER, because the segment moved during migration!)
            migrating_vp_idx = migration_result['migrating_vp_idx']
            first_level_vp_idx = outer_neighbors[0] if outer_neighbors else None
            
            T_first_VP = None
            if first_level_vp_idx is not None:
                seg_key = tuple(sorted([migrating_vp_idx, first_level_vp_idx]))
                # Use BEFORE partition's segment_to_triangle map
                T_first_VP = partition.segment_to_triangle.get(seg_key)
                if T_first_VP is not None:
                    print(f"  Found T_first_VP: {T_first_VP} (from BEFORE partition)")
            
            # Label the 6 methodology triangles explicitly
            methodology_triangles = {}
            
            # Add all triangles
            methodology_triangles[migration_result['triple_triangle_idx']] = f"T{migration_result['triple_triangle_idx']}"
            methodology_triangles[migration_result['target_triangle_idx']] = f"T{migration_result['target_triangle_idx']}"
            methodology_triangles[triangle_result['T_second_VP']] = f"T{triangle_result['T_second_VP']}"
            methodology_triangles[triangle_result['T_adjacent_to_T_second']] = f"T{triangle_result['T_adjacent_to_T_second']}"
            methodology_triangles[triangle_result['T_shared_edge_with_target']] = f"T{triangle_result['T_shared_edge_with_target']}"
            
            if T_first_VP is not None:
                methodology_triangles[T_first_VP] = f"T{T_first_VP}"
            
            print(f"  Labeling {len(methodology_triangles)} methodology triangles:")
            for tri_idx in sorted(methodology_triangles.keys()):
                print(f"    {methodology_triangles[tri_idx]}")
            
            # Label triangles based on --label-all flag
            if args.label_all:
                # Use spatial filtering when zoomed in for performance
                if args.apply_zoom:
                    # Only label triangles within distance of zoom focus
                    max_distance = args.zoom_factor * 50  # Increased from 10 to capture more triangles
                    print(f"  Labeling triangles within {max_distance:.4f} units of focus point...")
                    
                    # Batch collect centroids and labels
                    methodology_centroids = []
                    methodology_labels = []
                    other_centroids = []
                    other_labels = []
                    
                    labeled_count = 0
                    for tri_idx in range(len(mesh.faces)):
                        face = mesh.faces[tri_idx]
                        v0, v1, v2 = [mesh.vertices[int(i)] for i in face]
                        centroid = (v0 + v1 + v2) / 3.0
                        
                        # Check if within distance OR is methodology triangle
                        dist = np.linalg.norm(centroid - steiner_pos)
                        if dist < max_distance or tri_idx in methodology_triangles:
                            if tri_idx in methodology_triangles:
                                methodology_centroids.append(centroid)
                                methodology_labels.append(methodology_triangles[tri_idx])
                            else:
                                other_centroids.append(centroid)
                                other_labels.append(f"T{tri_idx}")
                            labeled_count += 1
                    
                    print(f"  Labeling {labeled_count} triangles (filtered from {len(mesh.faces)})...")
                    print(f"    Methodology triangles: {len(methodology_centroids)}")
                    print(f"    Other triangles: {len(other_centroids)}")
                    
                    # Batch add methodology triangles (gray background)
                    if methodology_centroids:
                        plotter_after.add_point_labels(
                            np.array(methodology_centroids),
                            methodology_labels,
                            font_size=args.label_font_size,
                            text_color='black',
                            shape_color='gray',
                            shape_opacity=0.8,
                            always_visible=True
                        )
                    
                    # Batch add other triangles (white background)
                    if other_centroids:
                        plotter_after.add_point_labels(
                            np.array(other_centroids),
                            other_labels,
                            font_size=args.label_font_size,
                            text_color='black',
                            shape_color='white',
                            shape_opacity=0.6,
                            always_visible=True
                        )
                else:
                    # No zoom - use batching only (no spatial filter)
                    print(f"  Labeling all {len(mesh.faces)} triangles using batched approach...")
                    
                    # Batch collect centroids and labels
                    methodology_centroids = []
                    methodology_labels = []
                    other_centroids = []
                    other_labels = []
                    
                    for tri_idx in range(len(mesh.faces)):
                        face = mesh.faces[tri_idx]
                        v0, v1, v2 = [mesh.vertices[int(i)] for i in face]
                        centroid = (v0 + v1 + v2) / 3.0
                        
                        if tri_idx in methodology_triangles:
                            methodology_centroids.append(centroid)
                            methodology_labels.append(methodology_triangles[tri_idx])
                        else:
                            other_centroids.append(centroid)
                            other_labels.append(f"T{tri_idx}")
                    
                    # Batch add methodology triangles
                    if methodology_centroids:
                        plotter_after.add_point_labels(
                            np.array(methodology_centroids),
                            methodology_labels,
                            font_size=12,
                            text_color='black',
                            shape_color='gray',
                            shape_opacity=0.8,
                            always_visible=True
                        )
                    
                    # Batch add other triangles
                    if other_centroids:
                        plotter_after.add_point_labels(
                            np.array(other_centroids),
                            other_labels,
                            font_size=12,
                            text_color='black',
                            shape_color='white',
                            shape_opacity=0.6,
                            always_visible=True
                        )
            else:
                # Original behavior: only methodology triangles
                for tri_idx, label in methodology_triangles.items():
                    add_triangle_label(plotter_after, mesh, tri_idx, label,
                                     color='black', font_size=args.label_font_size,
                                     shape_color='gray', shape_opacity=0.8)
            
            print("Triangle labels added")
            print()

        
        # Apply zoom (same as BEFORE)
        if args.apply_zoom:
            camera_offset = np.array([args.zoom_factor, args.zoom_factor * 0.5, args.zoom_factor * 0.5])
            camera_position = steiner_pos + camera_offset
            
            plotter_after.camera_position = [
                camera_position,
                steiner_pos,
                (0, 0, 1)
            ]
            plotter_after.camera.clipping_range = (args.zoom_factor * 0.1, args.zoom_factor * 10)
        
        print("✓ AFTER state rendered")
        print()
        
        # Show AFTER window (always blocking - last window)
        print("Opening PyVista window (AFTER state)...")
        plotter_after.show()
    
    # ========================================================================
    # COMPLETE
    # ========================================================================
    
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
    parser.add_argument('--label-all', action='store_true',
                       help='Label all triangles in mesh (methodology triangles: gray box, others: white box; '
                            'recommended with --apply-zoom to reduce clutter)')
    parser.add_argument('--label-font-size', type=int, default=12,
                       help='Font size for triangle labels (default: 12)')
    
    # Camera/zoom options
    parser.add_argument('--apply-zoom', action='store_true',
                       help='Apply zoom to focus on triple point')
    parser.add_argument('--zoom-factor', type=float, default=0.1,
                       help='Zoom factor (smaller = more zoomed) (default: 0.1)')
    parser.add_argument('--vp-size', type=float, default=0.0005,
                       help='Size of VP spheres (default: 0.0005)')
    parser.add_argument('--steiner-size', type=float, default=0.000005,
                       help='Size of Steiner point spheres (default: 0.000005)')
    parser.add_argument('--opacity', type=float, default=1.0,
                       help='Opacity of non-highlighted regions (0.0-1.0, default: 1.0)')
    parser.add_argument('--use-legacy', action='store_true',
                       help='Use legacy TopologySwitcher instead of MigrationOrchestrator')
    
    args = parser.parse_args()
    
    run_visualization(args)


if __name__ == '__main__':
    main()
