#!/usr/bin/env python3
"""
Visualize Type 1 migration using vertex-collapse strategy.

Clean, focused implementation for testing and debugging the new approach.
Only includes necessary code - no legacy methods.

Usage:
    python examples/visualize_type1_vertex_collapse.py \\
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

from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.core.mesh_topology import MeshTopology
from src.core.topology_switcher import TopologySwitcher
from src.core.steiner_handler import SteinerHandler
from src.core.area_calculator import AreaCalculator

# Import data loading from the reference script
from examples.visualize_precise_region import load_partition_from_refined_file


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
    
    SIMPLIFIED for vertex-collapse: Uses only VP edge endpoints and indicator_functions.
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
                # Use VP's edge endpoints as potential references
                edge = vp.edge
                v_a, v_b = edge
                vertex_labels = np.argmax(partition.indicator_functions, axis=1)
                
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
    
    # STRATEGY 2: Fallback to indicator_functions on triangle vertices
    if ref_vertex is None:
        vertex_labels = np.argmax(partition.indicator_functions, axis=1)
        
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
    - Skips triple point handling for now (can add later if needed)
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
    vp_size: float = 0.0005,
):
    """Add variable points as colored spheres."""
    if not vp_indices:
        return
    
    # Color scheme: migrating VP (yellow), neighbor 1 (blue), neighbor 2 (magenta)
    colors = ['yellow', 'blue', 'magenta']
    color_names = ['YELLOW', 'BLUE', 'MAGENTA']
    
    print(f"  Adding {len(vp_indices)} highlighted variable points (migration-related)...")
    print(f"    VP color scheme:")
    
    for i, vp_idx in enumerate(vp_indices):
        if vp_idx >= len(partition.variable_points):
            print(f"      WARNING: VP {vp_idx} out of range")
            continue
        
        vp = partition.variable_points[vp_idx]
        pos = vp.evaluate(mesh.vertices)
        sphere = pv.Sphere(radius=vp_size, center=pos)
        
        color = colors[i % len(colors)]
        color_name = color_names[i % len(color_names)]
        
        print(f"      VP {vp_idx}: {color_name}")
        plotter.add_mesh(sphere, color=color, opacity=0.9)
        print(f"      Added VP {vp_idx} at position {pos} with color {color}")


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
        steiner_pos = tp.compute_steiner_point()
        
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
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    steiner_handler = SteinerHandler(mesh, partition)
    
    # Component analysis
    print("Analyzing Type 1 migration...")
    print("Using vertex-collapse strategy (component-based migration)")
    print()
    
    # Step 1: Get boundary VPs (excluding triple point VPs)
    boundary_vps = switcher.get_non_triple_point_boundary_vps(boundary_tol=args.boundary_tol)
    boundary_vps_set = set(boundary_vps)
    
    if not boundary_vps:
        print("ERROR: No boundary VPs found for Type 1 migration")
        return
    
    # Step 2: Find connected components
    components = switcher.find_connected_components(boundary_vps_set)
    print(f"✓ Found {len(components)} connected component(s)")
    print()
    
    # Step 3: Analyze each component
    component_info = []
    for i, comp_vps in enumerate(components):
        info = switcher.analyze_component(comp_vps)
        info['index'] = i
        component_info.append(info)
    
    # Step 4: Detect conflicts
    conflicts, chain_warnings = switcher.detect_proximity_conflicts(component_info)
    
    # Step 5: Select components for migration
    to_migrate, deferred = switcher.select_components_for_migration(component_info, conflicts)
    
    # Display component table
    print("="*80)
    print("AVAILABLE COMPONENTS FOR MIGRATION")
    print("="*80)
    print(f"{'Idx':<5} {'Size':<6} {'Dist':<10} {'Cells':<10} {'Status':<15} {'VPs':<30}")
    print("-" * 80)
    
    for comp in component_info:
        status = "TO MIGRATE" if comp in to_migrate else "DEFERRED"
        vp_list = str(comp['vp_indices'][:5]) + ("..." if len(comp['vp_indices']) > 5 else "")
        
        # Get cells that this component separates
        all_cells = set()
        for vp_idx in comp['vp_indices']:
            vp = partition.variable_points[vp_idx]
            all_cells.update(vp.belongs_to_cells)
        cells_str = str(sorted(list(all_cells)))
        
        print(f"{comp['index']:<5} {comp['size']:<6} {comp['min_distance']:<10.6f} {cells_str:<10} {status:<15} {vp_list:<30}")
    
    print()
    
    # Select component
    if args.component_index >= len(component_info):
        print(f"ERROR: Component index {args.component_index} out of range (max: {len(component_info)-1})")
        return
    
    selected_component = component_info[args.component_index]
    
    # Identify migrating VP and neighbors
    component_vps = selected_component['vp_indices']
    migrating_vp_idx = min(component_vps, 
                           key=lambda vp: switcher.compute_boundary_distance(vp))
    
    left_neighbor, right_neighbor = switcher._get_two_neighbors(migrating_vp_idx)
    
    migrating_vp = partition.variable_points[migrating_vp_idx]
    current_edge = migrating_vp.edge
    target_vertex = switcher._identify_target_vertex(migrating_vp)
    target_edge = switcher._find_opposite_edge(current_edge, target_vertex) if target_vertex else None
    
    # Display selection
    print(f"✓ Selected Component {args.component_index} for migration:")
    print(f"  Size: {selected_component['size']} VPs")
    print(f"  VPs: {component_vps}")
    print(f"  Target vertex: {target_vertex}")
    print(f"  Min distance: {selected_component['min_distance']:.6f}")
    print(f"  Status: {'TO MIGRATE' if selected_component in to_migrate else 'DEFERRED'}")
    print()
    print(f"  Migrating VP: {migrating_vp_idx}")
    print(f"  Neighbors: {left_neighbor}, {right_neighbor}")
    print(f"  Current edge: {current_edge}")
    print(f"  Lambda: {migrating_vp.lambda_param:.6f}")
    print()
    
    # Initialize AreaCalculator
    print("  Initializing AreaCalculator BEFORE migration...")
    area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
    print()
    
    # Compute area
    lambda_vec = partition.get_variable_vector()
    area_info = compute_region_area(area_calc, args.region, lambda_vec)
    
    print("="*60)
    print("BEFORE Type 1 Migration")
    print("="*60)
    print()
    print(f"Region {args.region} Geometry (BEFORE):")
    print(f"  Interior triangles: {area_info['n_interior_triangles']:,} (area: {area_info['interior_area']:.4f})")
    print(f"  Boundary triangles: {area_info['n_boundary_triangles']:,} (area: {area_info['boundary_area']:.4f})")
    print(f"  TOTAL AREA: {area_info['total_area']:.6f}")
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Render regions
    render_regions(
        plotter, mesh, partition, area_calc, steiner_handler,
        args.region, args.intense_color
    )
    
    # Add VPs if requested
    if args.show_vps:
        highlight_vps = [migrating_vp_idx, left_neighbor, right_neighbor]
        add_vp_visualization(plotter, mesh, partition, highlight_vps, args.vp_size)
    
    # Add Steiner points if requested
    if args.show_steiner:
        add_steiner_visualization(plotter, mesh, partition, steiner_handler, args.steiner_size)
    
    # Add edge visualization
    add_edge_visualization(plotter, mesh, current_edge, target_edge)
    
    # Apply camera zoom if requested
    if args.apply_zoom:
        vp_pos = partition.evaluate_variable_point(migrating_vp_idx)
        apply_camera_zoom(plotter, vp_pos, args.zoom_factor)
    
    # Set title and show
    plotter.add_title(f"Region {args.region} - BEFORE Type 1 (Component {args.component_index})", font_size=14)
    plotter.show()


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
                       help='Component index to migrate (default: 0)')
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
