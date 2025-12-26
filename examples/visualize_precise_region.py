#!/usr/bin/env python3
"""
Visualize precise region boundaries using exact triangle portions.

This script renders partition regions with EXACT geometric boundaries, not
approximate majority-vote coloring. It's designed to validate:
- Triangle categorization (interior vs boundary)
- Segment crossing cache accuracy
- Area calculation correctness
- Boundary tracking after topology switches

IMPORTANT: This script requires BOTH files in the same directory:
1. Base solution file: <name>.h5 (contains mesh geometry + densities)
2. Refined contours file: <name>_refined_contours.h5 (contains optimized λ values)

You only need to provide the refined_contours.h5 file path; the script will
automatically find the base solution file.

Usage:
    # Current state, no migration
    python visualize_precise_region.py \\
        --solution <path>/*_refined_contours.h5 --region 2
    
    # Before/after Type 1 migration
    python visualize_precise_region.py \\
        --solution <path>/*_refined_contours.h5 --region 2 \\
        --switch-type type1 --state both
    
    # Before/after Type 2 migration  
    python visualize_precise_region.py \\
        --solution <path>/*_refined_contours.h5 --region 2 \\
        --switch-type type2 --state both

Author: Esteban Velez
Date: December 2025
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

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: simple progress indicator
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total or (len(iterable) if iterable else 0)
            self.n = 0
            if desc:
                print(f"{desc}...")
        
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.desc:
                print(f"{self.desc}... Done!")
        
        def update(self, n=1):
            self.n += n

from src.find_contours import ContourAnalyzer
from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.core.mesh_topology import MeshTopology
from src.core.topology_switcher import TopologySwitcher
from src.core.steiner_handler import SteinerHandler
from src.core.area_calculator import AreaCalculator


def load_partition_from_refined_file(refined_path):
    """
    Load partition from refined_contours.h5 file and base solution file.
    
    The refined file contains:
    - lambda_parameters: optimized VP positions
    - Metadata about optimization
    
    The base solution file contains:
    - vertices, faces: mesh geometry
    - x_opt (densities): for indicator functions
    
    Returns:
        tuple: (mesh, partition) ready for visualization
    """
    import h5py
    
    print(f"Loading from refined contours file...")
    print(f"  Refined: {refined_path}")
    
    # Derive base solution path
    base_solution_path = refined_path.replace('_refined_contours.h5', '.h5')
    print(f"  Base solution: {base_solution_path}")
    
    if not os.path.exists(base_solution_path):
        raise FileNotFoundError(
            f"Base solution file not found: {base_solution_path}\n"
            f"The refined_contours.h5 file needs the corresponding base solution file "
            f"(without _refined_contours) in the same directory."
        )
    
    # Load base solution (mesh + densities)
    analyzer = ContourAnalyzer(base_solution_path)
    analyzer.load_results(use_initial_condition=False)
    
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    print(f"  ✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Compute indicator functions
    indicator_functions = analyzer.compute_indicator_functions()
    n_cells = indicator_functions.shape[1]
    print(f"  ✓ Computed indicator functions: {n_cells} cells")
    
    # Extract boundary topology (efficient initialization)
    raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
    n_boundary_triangles = sum(len(v) for v in boundary_topology.values())
    print(f"  ✓ Extracted boundary topology: {n_boundary_triangles} boundary triangles")
    
    # Create partition (this initializes VPs with default λ=0.5)
    partition = PartitionContour(mesh, indicator_functions, boundary_topology=boundary_topology)
    print(f"  ✓ Created partition: {len(partition.variable_points)} VPs")
    
    # Load optimized λ parameters from refined file
    with h5py.File(refined_path, 'r') as f:
        if 'lambda_parameters' in f:
            lambda_opt = f['lambda_parameters'][:]
            partition.set_variable_vector(lambda_opt)
            print(f"  ✓ Applied optimized λ values: {len(lambda_opt)} parameters")
            
            # Verify match
            if len(lambda_opt) != len(partition.variable_points):
                raise ValueError(
                    f"Mismatch: refined file has {len(lambda_opt)} λ parameters, "
                    f"but partition has {len(partition.variable_points)} VPs"
                )
        else:
            raise ValueError("No lambda_parameters found in refined file")
    
    return mesh, partition


def compute_boundary_distance(partition, vp_idx: int) -> float:
    """
    Compute how far a boundary VP is from its target vertex.
    
    For λ < 0.5: distance = λ (approaching edge[1])
    For λ > 0.5: distance = (1 - λ) (approaching edge[0])
    """
    vp = partition.variable_points[vp_idx]
    return min(vp.lambda_param, 1.0 - vp.lambda_param)


def find_connected_components(boundary_vps_set, partition):
    """Find connected components of boundary VPs."""
    from collections import defaultdict
    
    adjacency = defaultdict(set)
    for segment in partition.boundary_segments:
        vp1, vp2 = segment.vp_idx_1, segment.vp_idx_2
        if vp1 in boundary_vps_set and vp2 in boundary_vps_set:
            adjacency[vp1].add(vp2)
            adjacency[vp2].add(vp1)
    
    visited = set()
    components = []
    
    for vp_idx in boundary_vps_set:
        if vp_idx in visited:
            continue
        
        component = set()
        stack = [vp_idx]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            component.add(current)
            
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        components.append(component)
    
    return components


def filter_connected_boundary_vps(boundary_vps, partition):
    """
    Filter boundary VPs to keep only one per connected component.
    Returns filtered list with one VP per connected component (closest to vertex).
    """
    if not boundary_vps:
        return []
    
    boundary_set = set(boundary_vps)
    components = find_connected_components(boundary_set, partition)
    
    vps_to_keep = []
    
    print(f"  Found {len(components)} connected component(s) among {len(boundary_vps)} boundary VPs")
    
    for i, component in enumerate(components):
        if len(component) == 1:
            vp_idx = list(component)[0]
            vps_to_keep.append(vp_idx)
        else:
            vps_with_dist = [
                (compute_boundary_distance(partition, vp), vp)
                for vp in component
            ]
            vps_with_dist.sort()
            
            closest_dist, closest_vp = vps_with_dist[0]
            vps_to_keep.append(closest_vp)
            
            print(f"    Component {i+1}: {len(component)} connected VPs, "
                  f"keeping VP {closest_vp} (distance={closest_dist:.6f})")
    
    return vps_to_keep


def compute_cell_portion_in_triangle(
    mesh: TriMesh,
    partition: PartitionContour,
    tri_idx: int,
    cell_idx: int,
    tri_idx_to_segment: Optional[Dict[int, any]] = None
) -> Optional[np.ndarray]:
    """
    Compute vertices of the polygon representing cell_idx's portion in tri_idx.
    
    Args:
        tri_idx_to_segment: Pre-indexed dict mapping triangle_idx to TriangleSegment (for speed)
    
    Returns:
        vertices: (N, 3) array where N is number of polygon vertices, or None if
                  triangle doesn't contribute to this cell's boundary
    """
    # Check if this triangle is in the cell's boundary (O(1) lookup if pre-indexed)
    is_boundary = False
    
    if tri_idx_to_segment is not None:
        # Fast path: O(1) lookup
        tri_seg = tri_idx_to_segment.get(tri_idx)
        if tri_seg:
            for vp_idx in tri_seg.var_point_indices:
                vp = partition.variable_points[vp_idx]
                if cell_idx in vp.belongs_to_cells:
                    is_boundary = True
                    break
    else:
        # Slow path: O(N) scan (fallback)
        for tri_seg in partition.triangle_segments:
            if tri_seg.triangle_idx == tri_idx:
                for vp_idx in tri_seg.var_point_indices:
                    vp = partition.variable_points[vp_idx]
                    if cell_idx in vp.belongs_to_cells:
                        is_boundary = True
                        break
                if is_boundary:
                    break
    
    # Also check segment_crossing_cache
    if not is_boundary and tri_idx in partition.segment_crossing_cache:
        for crossing in partition.segment_crossing_cache[tri_idx]:
            if crossing.cell_idx == cell_idx:
                is_boundary = True
                break
    
    if not is_boundary:
        return None
    
    # Get mesh vertices and labels
    face = mesh.faces[tri_idx]
    v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
    labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
    n_inside = sum(1 for lab in labels if lab == cell_idx)
    
    # Collect boundary points
    vp_positions = []
    crossing_positions = []
    
    # From triangle_segments (VPs in this triangle) - use pre-index if available
    if tri_idx_to_segment is not None:
        tri_seg = tri_idx_to_segment.get(tri_idx)
        if tri_seg:
            for vp_idx in tri_seg.var_point_indices:
                vp = partition.variable_points[vp_idx]
                if cell_idx in vp.belongs_to_cells:
                    pos = vp.evaluate(mesh.vertices)
                    vp_positions.append(pos)
    else:
        # Fallback: scan all
        for tri_seg in partition.triangle_segments:
            if tri_seg.triangle_idx == tri_idx:
                for vp_idx in tri_seg.var_point_indices:
                    vp = partition.variable_points[vp_idx]
                    if cell_idx in vp.belongs_to_cells:
                        pos = vp.evaluate(mesh.vertices)
                        vp_positions.append(pos)
    
    # From segment_crossing_cache (crossings in this triangle)
    if tri_idx in partition.segment_crossing_cache:
        for crossing in partition.segment_crossing_cache[tri_idx]:
            if crossing.cell_idx == cell_idx:
                # Add crossing points that are NOT VP positions
                for pos in [crossing.entry_point, crossing.exit_point]:
                    is_vp_pos = False
                    for vp_pos in vp_positions:
                        if np.linalg.norm(pos - vp_pos) < 1e-8:
                            is_vp_pos = True
                            break
                    if not is_vp_pos:
                        # Check if already in list
                        is_duplicate = False
                        for existing_pos in crossing_positions:
                            if np.linalg.norm(pos - existing_pos) < 1e-8:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            crossing_positions.append(pos)
    
    boundary_points = vp_positions + crossing_positions
    
    # Get mesh vertices inside cell
    vertices_inside = []
    for v, lab in zip([v1, v2, v3], labels):
        if lab == cell_idx:
            vertices_inside.append(mesh.vertices[v])
    
    # Construct polygon
    all_points = vertices_inside + boundary_points
    
    if len(all_points) < 3:
        return None
    
    # Order points to form valid polygon (counter-clockwise)
    return _order_polygon_vertices(np.array(all_points), mesh, tri_idx)


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


def compute_triple_point_cell_portion(
    mesh: TriMesh,
    partition: PartitionContour,
    steiner_handler: SteinerHandler,
    tri_idx: int,
    cell_idx: int
) -> Optional[np.ndarray]:
    """
    Compute the portion of a triple point triangle belonging to cell_idx.
    Uses Steiner tree subdivision.
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
    steiner_pos = triple_point.compute_steiner_point()
    
    # Find the VP belonging to this cell
    cell_vp_idx = None
    for vp_idx in triple_point.var_point_indices:
        vp = partition.variable_points[vp_idx]
        if cell_idx in vp.belongs_to_cells:
            cell_vp_idx = vp_idx
            break
    
    if cell_vp_idx is None:
        return None
    
    # Get the two other VPs (neighbors in void triangle)
    other_vps = [vp for vp in triple_point.var_point_indices if vp != cell_vp_idx]
    
    # The cell portion is bounded by:
    # - Two Steiner branches to neighboring VPs
    # - Two void triangle edges from cell_vp to neighbors
    cell_vp_pos = partition.evaluate_variable_point(cell_vp_idx)
    neighbor1_pos = partition.evaluate_variable_point(other_vps[0])
    neighbor2_pos = partition.evaluate_variable_point(other_vps[1])
    
    # Return as quadrilateral (cell_vp, neighbor1, steiner, neighbor2)
    polygon = np.array([
        cell_vp_pos,
        neighbor1_pos,
        steiner_pos,
        neighbor2_pos
    ])
    
    return polygon


def render_single_region_precise(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    cell_idx: int,
    color: str,
    opacity: float = 1.0,
    show_progress: bool = True,
    use_exact_boundaries: bool = True
):
    """
    Render ONE region with exact geometric boundaries.
    
    Args:
        plotter: PyVista plotter
        mesh: TriMesh
        partition: PartitionContour
        area_calc: AreaCalculator (for categorized triangles)
        steiner_handler: SteinerHandler (for triple points)
        cell_idx: Cell index to render
        color: Color for this region
        opacity: Opacity (1.0 = fully opaque)
        show_progress: Show progress bar
        use_exact_boundaries: If False, render simple triangles (faster for non-target regions)
    """
    # Pre-index triangle_segments for O(1) lookup (HUGE speedup!)
    tri_idx_to_segment = None
    if use_exact_boundaries:
        tri_idx_to_segment = {}
        for tri_seg in partition.triangle_segments:
            tri_idx_to_segment[tri_seg.triangle_idx] = tri_seg
    
    # Collect ALL vertices and faces in numpy arrays (FAST approach like other scripts!)
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Interior triangles (full) - always fast
    interior_tris = area_calc.cell_interior_triangles.get(cell_idx, [])
    if show_progress and len(interior_tris) > 1000:
        desc = f"  Processing interior triangles (cell {cell_idx})"
        iterator = tqdm(interior_tris, desc=desc, leave=False)
    else:
        iterator = interior_tris
    
    for tri_idx in iterator:
        face = mesh.faces[tri_idx]
        vertices = mesh.vertices[[int(face[0]), int(face[1]), int(face[2])]]
        all_vertices.append(vertices)
        # Face: [3, v0, v1, v2] with local indices
        all_faces.extend([3, vertex_offset, vertex_offset+1, vertex_offset+2])
        vertex_offset += 3
    
    # Boundary triangles (partial or full, depending on mode)
    boundary_tris = area_calc.cell_boundary_triangles.get(cell_idx, [])
    
    if use_exact_boundaries:
        # Exact polygon construction (for target region)
        if show_progress and len(boundary_tris) > 100:
            desc = f"  Processing exact boundaries (cell {cell_idx})"
            iterator = tqdm(boundary_tris, desc=desc, leave=False)
        else:
            iterator = boundary_tris
        
        for tri_idx in iterator:
            poly_vertices = compute_cell_portion_in_triangle(
                mesh, partition, tri_idx, cell_idx, tri_idx_to_segment
            )
            if poly_vertices is not None:
                n_verts = len(poly_vertices)
                all_vertices.append(poly_vertices)
                # Face: [n, v0, v1, ..., vn-1] with local indices
                face_indices = [n_verts] + list(range(vertex_offset, vertex_offset + n_verts))
                all_faces.extend(face_indices)
                vertex_offset += n_verts
    else:
        # Fast mode: just render full triangles (approximate, for non-target regions)
        for tri_idx in boundary_tris:
            face = mesh.faces[tri_idx]
            vertices = mesh.vertices[[int(face[0]), int(face[1]), int(face[2])]]
            all_vertices.append(vertices)
            all_faces.extend([3, vertex_offset, vertex_offset+1, vertex_offset+2])
            vertex_offset += 3
    
    # Triple point triangles (Steiner subdivisions)
    if use_exact_boundaries:
        for tp in steiner_handler.triple_points:
            if cell_idx in tp.cell_indices:
                poly_vertices = compute_triple_point_cell_portion(
                    mesh, partition, steiner_handler, tp.triangle_idx, cell_idx
                )
                if poly_vertices is not None:
                    n_verts = len(poly_vertices)
                    all_vertices.append(poly_vertices)
                    face_indices = [n_verts] + list(range(vertex_offset, vertex_offset + n_verts))
                    all_faces.extend(face_indices)
                    vertex_offset += n_verts
    
    # Create single mesh from all collected vertices and faces (FAST - O(n)!)
    if all_vertices:
        if show_progress:
            print(f"  Creating mesh from {len(all_vertices)} polygons ({vertex_offset} vertices)...")
        
        # Concatenate all vertices into single array
        vertices_combined = np.vstack(all_vertices)
        
        # Convert face list to numpy array
        faces_combined = np.array(all_faces, dtype=np.int64)
        
        # Create PyVista mesh (single operation - FAST!)
        region_mesh = pv.PolyData(vertices_combined, faces_combined)
        
        # Add to plotter with edges (like reference scripts)
        plotter.add_mesh(
            region_mesh, 
            color=color, 
            opacity=opacity, 
            show_edges=True,           # Show mesh edges on region
            edge_color='gray',         # Match reference scripts
            line_width=0.5,            # Match reference scripts
            backface_culling=True      # Hide back faces (no confusing rear view)
        )
        
        if show_progress:
            print(f"  ✓ Rendered {len(all_vertices)} polygons in single mesh")


def render_region_precise(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    target_region: int,
    target_color: str = 'orangered',
    show_vps: bool = False,
    show_steiner: bool = False,
    show_mesh_triangles: bool = False,
    vp_size: float = 0.0005,
    steiner_size: float = 0.000005,
    title: str = "Precise Region Visualization",
    target_only: bool = False,
    highlight_vp_indices: list = None  # NEW: Only highlight specific VPs
):
    """
    Render ALL regions with precise boundaries.
    
    Args:
        target_only: If True, only render target region (much faster, ~1 minute)
        highlight_vp_indices: List of VP indices to highlight (for migrations)
    """
    # Light pastel palette for all regions (like other scripts)
    pale_palette = [
        '#c6dbef', '#c7e9c0', '#fdd0a2', '#e5d8bd', '#d9d9d9', '#f2f0f7',
        '#e7e1ef', '#fee0d2', '#ffffcc', '#d0e1f9', '#fde0ef', '#e0ecf4'
    ]
    
    # NOTE: Mesh edges will be displayed on region meshes directly (like reference scripts)
    # No need for separate base mesh
    
    # Render all regions with precise boundaries (using pastel colors)
    if target_only:
        print(f"  Rendering target region {target_region} (precise boundaries)...")
        regions_to_render = [target_region]
    else:
        print(f"  Rendering all {partition.n_cells} regions (precise boundaries)...")
        regions_to_render = range(partition.n_cells)
    
    for cell_idx in regions_to_render:
        color = pale_palette[cell_idx % len(pale_palette)]
        render_single_region_precise(
            plotter, mesh, partition, area_calc, steiner_handler,
            cell_idx, color, opacity=0.8,
            show_progress=(cell_idx == target_region),  # Progress only for first
            use_exact_boundaries=True  # Always use exact boundaries
        )
    
    print(f"  ✓ Region rendering complete")
    
    # NOTE: Boundary contours are already part of the precise region rendering above
    # No need to draw them separately - it would be redundant and slow
    
    # Add VPs if requested (only highlight specific VPs for migrations)
    if show_vps and highlight_vp_indices is not None:
        n_vps = len(highlight_vp_indices)
        print(f"  Adding {n_vps} highlighted variable points (migration-related)...")
        
        if n_vps > 100:
            iterator = tqdm(highlight_vp_indices, desc="  Drawing VPs", leave=False)
        else:
            iterator = highlight_vp_indices
        
        for vp_idx in iterator:
            vp = partition.variable_points[vp_idx]
            pos = vp.evaluate(mesh.vertices)
            sphere = pv.Sphere(radius=vp_size, center=pos)
            # Color by cell membership
            if target_region in vp.belongs_to_cells:
                vp_color = 'yellow'
            else:
                vp_color = 'cyan'
            plotter.add_mesh(sphere, color=vp_color, opacity=0.8)
    
    # Add Steiner points if requested
    if show_steiner:
        n_tps = len(steiner_handler.triple_points)
        print(f"  Adding {n_tps} Steiner points and void triangles...")
        
        if n_tps > 10:
            iterator = tqdm(steiner_handler.triple_points, 
                          desc="  Drawing Steiner points", leave=False)
        else:
            iterator = steiner_handler.triple_points
        
        for tp in iterator:
            steiner_pt = tp.compute_steiner_point()
            sphere = pv.Sphere(radius=steiner_size, center=steiner_pt)
            plotter.add_mesh(sphere, color='red', opacity=0.9)
            
            # Draw void triangle
            void_verts = []
            for vp_idx in tp.var_point_indices:
                vp = partition.variable_points[vp_idx]
                pos = vp.evaluate(mesh.vertices)
                void_verts.append(pos)
            
            for i in range(3):
                v1 = void_verts[i]
                v2 = void_verts[(i + 1) % 3]
                line = pv.Line(v1, v2)
                plotter.add_mesh(line, color='cyan', line_width=2, opacity=0.7)
    
    # NOTE: Mesh edges are already added at the beginning (line 617-632) efficiently
    # Don't duplicate here with individual line segments - it's VERY slow!
    
    plotter.add_title(title, font_size=14)



def get_neighbors_from_triangle_segments(partition, vp_idx):
    """Get neighbor VPs from triangle_segments."""
    neighbors = set()
    for tri_seg in partition.triangle_segments:
        if vp_idx in tri_seg.var_point_indices:
            for other_vp in tri_seg.var_point_indices:
                if other_vp != vp_idx:
                    neighbors.add(other_vp)
    return list(neighbors)


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


def run_visualization(args):
    """Main visualization routine."""
    print("="*80)
    print("PRECISE REGION VISUALIZATION")
    print("="*80)
    print(f"Refined contours file: {args.solution}")
    print(f"Target region: {args.region}")
    print(f"Switch type: {args.switch_type}")
    print()
    
    # Check file exists
    if not os.path.exists(args.solution):
        print(f"ERROR: File not found: {args.solution}")
        return
    
    # Verify it's a refined_contours file
    if '_refined_contours.h5' not in args.solution:
        print("WARNING: This script is designed for refined_contours.h5 files")
        print(f"         Your file: {args.solution}")
        print(f"         Expected: *_refined_contours.h5")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Load partition from refined file
    print("\nLoading partition data...")
    try:
        mesh, partition = load_partition_from_refined_file(args.solution)
    except Exception as e:
        print(f"ERROR: Failed to load refined contours file")
        print(f"       {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✓ Loaded partition state from refined file")
    
    # Initialize components
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    
    # No switch - just show current state
    if args.switch_type == 'none':
        print("\nRendering current state (no migration)...")
        
        print("  Initializing AreaCalculator (VP-based, optimized)...")
        area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
        
        print("  Initializing SteinerHandler...")
        steiner_handler = SteinerHandler(mesh, partition)
        
        # Compute area for target region
        lambda_vec = partition.get_variable_vector()
        area_info = compute_region_area(area_calc, args.region, lambda_vec)
        
        print(f"\nRegion {args.region} Geometry:")
        print(f"  Interior triangles: {area_info['n_interior_triangles']:,} "
              f"(area: {area_info['interior_area']:.4f})")
        print(f"  Boundary triangles: {area_info['n_boundary_triangles']:,} "
              f"(area: {area_info['boundary_area']:.4f})")
        print(f"  TOTAL AREA: {area_info['total_area']:.6f}")
        
        # Render
        plotter = pv.Plotter()
        render_region_precise(
            plotter, mesh, partition, area_calc, steiner_handler,
            args.region, args.intense_color,
            show_vps=args.show_vps,
            show_steiner=args.show_steiner,
            show_mesh_triangles=args.show_mesh_triangles,
            vp_size=args.vp_size,
            steiner_size=args.steiner_size,
            title=f"Precise Region {args.region} - Current State",
            target_only=args.target_only,
            highlight_vp_indices=None  # No migration, no VPs to highlight
        )
        plotter.show()
        
        return
    
    # Type 1 or Type 2 switch
    if args.switch_type == 'type1':
        print("\nAnalyzing Type 1 migration...")
        
        # Find boundary VPs
        boundary_vps = partition.get_boundary_variable_points(tol=args.boundary_tol)
        
        # Filter out triple point VPs
        steiner_handler = SteinerHandler(mesh, partition)
        triple_point_vp_indices = set()
        for tp in steiner_handler.triple_points:
            triple_point_vp_indices.update(tp.var_point_indices)
        
        non_triple_boundary_vps = [vp for vp in boundary_vps 
                                   if vp not in triple_point_vp_indices]
        
        if not non_triple_boundary_vps:
            print("ERROR: No boundary VPs found for Type 1 migration")
            return
        
        # Filter connected components
        filtered_vps = filter_connected_boundary_vps(non_triple_boundary_vps, partition)
        
        # Sort by distance
        filtered_vps_sorted = sorted(
            filtered_vps,
            key=lambda vp_idx: compute_boundary_distance(partition, vp_idx)
        )
        
        # Select closest VP
        vp_idx = filtered_vps_sorted[0]
        vp = partition.variable_points[vp_idx]
        
        print(f"\nSelected VP {vp_idx}:")
        print(f"  Edge: {vp.edge}")
        print(f"  Lambda: {vp.lambda_param:.6f}")
        print(f"  Distance to vertex: {compute_boundary_distance(partition, vp_idx):.6f}")
        
        # BEFORE state
        if args.state in ['before', 'both']:
            print("\n" + "="*60)
            print("BEFORE Type 1 Migration")
            print("="*60)
            
            print("  Initializing AreaCalculator (VP-based, optimized)...")
            area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
            lambda_vec = partition.get_variable_vector()
            area_info = compute_region_area(area_calc, args.region, lambda_vec)
            
            print(f"\nRegion {args.region} Geometry (BEFORE):")
            print(f"  Interior triangles: {area_info['n_interior_triangles']:,} "
                  f"(area: {area_info['interior_area']:.4f})")
            print(f"  Boundary triangles: {area_info['n_boundary_triangles']:,} "
                  f"(area: {area_info['boundary_area']:.4f})")
            print(f"  TOTAL AREA: {area_info['total_area']:.6f}")
            
            plotter = pv.Plotter()
            
            # Get neighbors to highlight
            neighbors_before = get_neighbors_from_triangle_segments(partition, vp_idx)
            highlight_vps_before = [vp_idx] + neighbors_before
            
            render_region_precise(
                plotter, mesh, partition, area_calc, steiner_handler,
                args.region, args.intense_color,
                show_vps=args.show_vps,
                show_steiner=args.show_steiner,
                show_mesh_triangles=args.show_mesh_triangles,
                vp_size=args.vp_size,
                steiner_size=args.steiner_size,
                title=f"Region {args.region} - BEFORE Type 1 (VP {vp_idx})",
                target_only=args.target_only,
                highlight_vp_indices=highlight_vps_before if args.show_vps else None
            )
            
            if args.save_before:
                plotter.screenshot(args.save_before)
                print(f"✓ Saved BEFORE state to {args.save_before}")
            
            if args.state == 'before':
                plotter.show()
                return
            else:
                plotter.show(interactive_update=True)
        
        # Apply Type 1 switch
        if args.state in ['after', 'both']:
            print("\n" + "="*60)
            print("Applying Type 1 Migration...")
            print("="*60)
            
            old_edge = vp.edge
            success = switcher.apply_type1_switch(vp_idx, tol=args.boundary_tol)
            
            if not success:
                print("ERROR: Type 1 switch failed!")
                return
            
            new_edge = vp.edge
            print(f"✓ VP {vp_idx} moved: {old_edge} → {new_edge}")
            
            # Rebuild
            print("  Rebuilding triangle segments...")
            partition.rebuild_triangle_segments_from_current_vps()
            print("  Classifying segments...")
            switcher.classify_all_segments()
            
            # Re-initialize with VP-based categorization
            print("  Re-initializing AreaCalculator (VP-based)...")
            area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
            print("  Re-initializing SteinerHandler...")
            steiner_handler = SteinerHandler(mesh, partition)
            
            # AFTER state
            print("\n" + "="*60)
            print("AFTER Type 1 Migration")
            print("="*60)
            
            lambda_vec = partition.get_variable_vector()
            area_info_after = compute_region_area(area_calc, args.region, lambda_vec)
            
            print(f"\nRegion {args.region} Geometry (AFTER):")
            print(f"  Interior triangles: {area_info_after['n_interior_triangles']:,} "
                  f"(area: {area_info_after['interior_area']:.4f})")
            print(f"  Boundary triangles: {area_info_after['n_boundary_triangles']:,} "
                  f"(area: {area_info_after['boundary_area']:.4f})")
            print(f"  TOTAL AREA: {area_info_after['total_area']:.6f}")
            
            if args.state == 'both':
                area_diff = area_info_after['total_area'] - area_info['total_area']
                print(f"\n  Area difference: {area_diff:.2e}")
                if abs(area_diff) < 1e-6:
                    print(f"  ✓ Area conserved!")
                else:
                    print(f"  ⚠ Area changed!")
            
            plotter = pv.Plotter()
            
            # Get neighbors after migration to highlight
            neighbors_after = get_neighbors_from_triangle_segments(partition, vp_idx)
            highlight_vps_after = [vp_idx] + neighbors_after
            
            render_region_precise(
                plotter, mesh, partition, area_calc, steiner_handler,
                args.region, args.intense_color,
                show_vps=args.show_vps,
                show_steiner=args.show_steiner,
                show_mesh_triangles=args.show_mesh_triangles,
                vp_size=args.vp_size,
                steiner_size=args.steiner_size,
                title=f"Region {args.region} - AFTER Type 1 (VP {vp_idx} moved)",
                target_only=args.target_only,
                highlight_vp_indices=highlight_vps_after if args.show_vps else None
            )
            
            if args.save_after:
                plotter.screenshot(args.save_after)
                print(f"✓ Saved AFTER state to {args.save_after}")
            
            plotter.show()
    
    elif args.switch_type == 'type2':
        print("\nAnalyzing Type 2 migration...")
        
        steiner_handler = SteinerHandler(mesh, partition)
        boundary_tps = steiner_handler.get_boundary_triple_points(tol=args.boundary_tol)
        
        if not boundary_tps:
            print("ERROR: No boundary triple points found for Type 2 migration")
            return
        
        tp = boundary_tps[0]
        print(f"\nSelected triple point at triangle {tp.triangle_idx}:")
        print(f"  VPs: {tp.var_point_indices}")
        print(f"  Cells: {tp.cell_indices}")
        
        # BEFORE state
        if args.state in ['before', 'both']:
            print("\n" + "="*60)
            print("BEFORE Type 2 Migration")
            print("="*60)
            
            print("  Initializing AreaCalculator (VP-based, optimized)...")
            area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
            lambda_vec = partition.get_variable_vector()
            area_info = compute_region_area(area_calc, args.region, lambda_vec)
            
            print(f"\nRegion {args.region} Geometry (BEFORE):")
            print(f"  Interior triangles: {area_info['n_interior_triangles']:,} "
                  f"(area: {area_info['interior_area']:.4f})")
            print(f"  Boundary triangles: {area_info['n_boundary_triangles']:,} "
                  f"(area: {area_info['boundary_area']:.4f})")
            print(f"  TOTAL AREA: {area_info['total_area']:.6f}")
            
            plotter = pv.Plotter()
            
            # Highlight VPs involved in triple point
            highlight_vps_before = list(tp.var_point_indices)
            
            render_region_precise(
                plotter, mesh, partition, area_calc, steiner_handler,
                args.region, args.intense_color,
                show_vps=args.show_vps,
                show_steiner=args.show_steiner,
                show_mesh_triangles=args.show_mesh_triangles,
                vp_size=args.vp_size,
                steiner_size=args.steiner_size,
                title=f"Region {args.region} - BEFORE Type 2 (TP at T{tp.triangle_idx})",
                target_only=args.target_only,
                highlight_vp_indices=highlight_vps_before if args.show_vps else None
            )
            
            if args.save_before:
                plotter.screenshot(args.save_before)
                print(f"✓ Saved BEFORE state to {args.save_before}")
            
            if args.state == 'before':
                plotter.show()
                return
            else:
                plotter.show(interactive_update=True)
        
        # Apply Type 2 switch
        if args.state in ['after', 'both']:
            print("\n" + "="*60)
            print("Applying Type 2 Migration...")
            print("="*60)
            
            old_triangle = tp.triangle_idx
            success = switcher.apply_type2_switch(tp, tol=args.boundary_tol)
            
            if not success:
                print("ERROR: Type 2 switch failed!")
                return
            
            print(f"✓ Triple point migrated from triangle {old_triangle}")
            
            # Rebuild
            print("  Rebuilding triangle segments...")
            partition.rebuild_triangle_segments_from_current_vps()
            print("  Classifying segments...")
            switcher.classify_all_segments()
            
            # Re-initialize with VP-based categorization
            print("  Re-initializing AreaCalculator (VP-based)...")
            area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
            print("  Re-initializing SteinerHandler...")
            steiner_handler = SteinerHandler(mesh, partition)
            
            # Find new triple point location
            for tp_new in steiner_handler.triple_points:
                if any(vp_idx in tp.var_point_indices for vp_idx in tp_new.var_point_indices):
                    print(f"  New location: triangle {tp_new.triangle_idx}")
                    break
            
            # AFTER state
            print("\n" + "="*60)
            print("AFTER Type 2 Migration")
            print("="*60)
            
            lambda_vec = partition.get_variable_vector()
            area_info_after = compute_region_area(area_calc, args.region, lambda_vec)
            
            print(f"\nRegion {args.region} Geometry (AFTER):")
            print(f"  Interior triangles: {area_info_after['n_interior_triangles']:,} "
                  f"(area: {area_info_after['interior_area']:.4f})")
            print(f"  Boundary triangles: {area_info_after['n_boundary_triangles']:,} "
                  f"(area: {area_info_after['boundary_area']:.4f})")
            print(f"  TOTAL AREA: {area_info_after['total_area']:.6f}")
            
            if args.state == 'both':
                area_diff = area_info_after['total_area'] - area_info['total_area']
                print(f"\n  Area difference: {area_diff:.2e}")
                if abs(area_diff) < 1e-6:
                    print(f"  ✓ Area conserved!")
                else:
                    print(f"  ⚠ Area changed!")
            
            plotter = pv.Plotter()
            
            # Find new triple point and highlight its VPs
            new_tp_vps = []
            for tp_new in steiner_after.triple_points:
                if any(vp_idx in tp.var_point_indices for vp_idx in tp_new.var_point_indices):
                    new_tp_vps = list(tp_new.var_point_indices)
                    break
            
            render_region_precise(
                plotter, mesh, partition, area_calc, steiner_handler,
                args.region, args.intense_color,
                show_vps=args.show_vps,
                show_steiner=args.show_steiner,
                show_mesh_triangles=args.show_mesh_triangles,
                vp_size=args.vp_size,
                steiner_size=args.steiner_size,
                title=f"Region {args.region} - AFTER Type 2 (TP migrated)",
                target_only=args.target_only,
                highlight_vp_indices=new_tp_vps if args.show_vps else None
            )
            
            if args.save_after:
                plotter.screenshot(args.save_after)
                print(f"✓ Saved AFTER state to {args.save_after}")
            
            plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize precise region boundaries with exact triangle portions",
        epilog="NOTE: Requires both <name>.h5 and <name>_refined_contours.h5 in same directory. "
               "Provide the _refined_contours.h5 path; base file will be auto-detected."
    )
    parser.add_argument('--solution', required=True,
                       help='Path to *_refined_contours.h5 file (output of refine_perimeter.py). '
                            'Base solution file must be in same directory.')
    parser.add_argument('--region', type=int, required=True,
                       help='Region (cell) index to highlight with precise boundaries (0-indexed)')
    parser.add_argument('--switch-type', choices=['none', 'type1', 'type2'], default='none',
                       help='Type of topology switch to analyze (default: none)')
    parser.add_argument('--state', choices=['before', 'after', 'both'], default='both',
                       help='Which state to show (default: both)')
    parser.add_argument('--intense-color', default='orangered',
                       help='Color for highlighted region (default: orangered)')
    parser.add_argument('--show-vps', action='store_true',
                       help='Show variable points as spheres')
    parser.add_argument('--show-steiner', action='store_true',
                       help='Show Steiner points and void triangles')
    parser.add_argument('--show-mesh-triangles', action='store_true',
                       help='Show mesh triangle edges (for debugging)')
    parser.add_argument('--vp-size', type=float, default=0.0005,
                       help='Size of VP spheres (default: 0.0005)')
    parser.add_argument('--steiner-size', type=float, default=0.000005,
                       help='Size of Steiner point spheres (default: 0.000005)')
    parser.add_argument('--boundary-tol', type=float, default=0.1,
                       help='Threshold for boundary detection (default: 0.1)')
    parser.add_argument('--target-only', action='store_true',
                       help='Render ONLY target region (skip other regions for speed, 100x faster)')
    parser.add_argument('--save-before', type=str,
                       help='Path to save BEFORE state image')
    parser.add_argument('--save-after', type=str,
                       help='Path to save AFTER state image')
    
    args = parser.parse_args()
    
    # Validate region index
    if args.region < 0:
        print(f"ERROR: Region index must be >= 0")
        return 1
    
    run_visualization(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

