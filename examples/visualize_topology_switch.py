#!/usr/bin/env python3
"""
Visualize topology switches (Type 1 and Type 2) using PyVista.

Shows the partition with colored cells, boundary contours, triple points,
and highlights the VP and triangles affected by the switch.

Uses the same structure as surface_visualization.py.

Usage:
    python visualize_topology_switch.py \
        --solution <path_to_h5> \
        --switch-type type1|type2 \
        --state before|after

Author: Esteban Velez
Date: December 2025
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista is required. Install with: pip install pyvista")
    sys.exit(1)

from src.find_contours import ContourAnalyzer
from src.core.contour_partition import PartitionContour
from src.core.tri_mesh import TriMesh
from src.core.mesh_topology import MeshTopology
from src.core.topology_switcher import TopologySwitcher
from src.core.steiner_handler import SteinerHandler


def load_partition_contour_from_analyzer(analyzer):
    """Create PartitionContour with Phase 1 structures from ContourAnalyzer data.
    
    CRITICAL: Must use boundary_topology to match refine_perimeter.py initialization.
    The refined_contours.h5 lambda values were saved with VP indices from this path.
    """
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    indicator_functions = analyzer.compute_indicator_functions()
    # Use boundary_topology - must match the path used in refine_perimeter.py
    raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
    partition = PartitionContour(mesh, indicator_functions, boundary_topology=boundary_topology)
    return mesh, partition


def load_optimized_lambda_from_file(refined_path, partition):
    """Load optimized λ values from refined HDF5 file and apply to partition."""
    import h5py
    try:
        with h5py.File(refined_path, 'r') as f:
            if 'lambda_parameters' in f:
                lambda_opt = f['lambda_parameters'][:]
                partition.set_variable_vector(lambda_opt)
                return lambda_opt
    except Exception as e:
        print(f"Warning: Could not load lambda parameters: {e}")
    return None


def get_steiner_info(mesh, partition, steiner_handler):
    """Extract Steiner point information for visualization."""
    steiner_info = {
        'steiner_points': [],
        'void_triangles': [],
        'triple_point_data': []
    }
    for tp in steiner_handler.triple_points:
        steiner_pt = tp.compute_steiner_point()
        steiner_info['steiner_points'].append(steiner_pt)
        
        void_vertices = []
        for vp_idx in tp.var_point_indices:
            vp = partition.variable_points[vp_idx]
            pos = vp.evaluate(mesh.vertices)
            void_vertices.append(pos)
        steiner_info['void_triangles'].append(np.array(void_vertices))
        
        steiner_info['triple_point_data'].append({
            'triangle_idx': tp.triangle_idx,
            'cell_indices': tp.cell_indices,
            'var_point_indices': tp.var_point_indices
        })
    return steiner_info


def plot_partition_with_switch_highlight(
    vertices: np.ndarray,
    faces: np.ndarray,
    contours: dict,
    triangle_labels: np.ndarray,
    steiner_info: dict = None,
    highlight_vp_positions: list = None,
    highlight_vp_colors: list = None,
    highlight_vp_labels: list = None,  # NEW: Labels for VPs (e.g., "VP1: (9075, 9076)")
    highlight_triangles: list = None,
    highlight_segments: list = None,
    highlight_vertices: list = None,
    highlight_vertex_colors: list = None,
    highlight_edges: list = None,  # NEW: List of (v1_idx, v2_idx, color) to highlight mesh edges
    triangle_indices_to_label: list = None,  # NEW: List of triangle indices to label
    title: str = "Topology Switch Visualization",
    steiner_size: float = 0.008,
    vp_size: float = 0.01,
    camera_focus: np.ndarray = None,  # NEW: Point to focus camera on
    camera_zoom: float = None,  # NEW: Zoom level (distance from focus point)
):
    """
    Plot partition with contours, Steiner points, and switch highlights.
    
    Based on plot_mesh_with_contours_pyvista from plot_utils_3d.py
    """
    # Create PyVista mesh
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(vertices, faces_pv)
    
    # Add cell labels for coloring
    mesh.cell_data['region'] = triangle_labels
    
    # Light pastel palette for partitions
    light_palette = [
        '#c6dbef', '#c7e9c0', '#fdd0a2', '#e5d8bd', '#d9d9d9', '#f2f0f7',
        '#e7e1ef', '#fee0d2', '#ffffcc', '#d0e1f9', '#fde0ef', '#e0ecf4'
    ]
    
    plotter = pv.Plotter()
    plotter.add_title(title, font_size=12)
    
    # Add mesh with partition coloring
    plotter.add_mesh(
        mesh,
        scalars='region',
        categories=True,
        cmap=light_palette,
        show_scalar_bar=False,
        opacity=0.9,
        show_edges=True,
        edge_color='gray',
        line_width=0.5,
    )
    
    # Add contours (boundary segments)
    color_list = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf', '#e377c2', '#7f7f7f']
    if contours is not None:
        for region_idx, segments in contours.items():
            color = color_list[region_idx % len(color_list)]
            for seg in segments:
                if seg.shape[1] == 3:  # 3D
                    line = pv.Line(seg[0], seg[1])
                    plotter.add_mesh(line, color=color, line_width=4)
    
    # Add Steiner points and void triangles
    if steiner_info is not None:
        steiner_points = steiner_info.get('steiner_points', [])
        void_triangles = steiner_info.get('void_triangles', [])
        
        # Draw Steiner points as red spheres
        for sp in steiner_points:
            if sp.shape[0] == 3:
                sphere = pv.Sphere(radius=steiner_size, center=sp)
                plotter.add_mesh(sphere, color='red', opacity=0.9)
        
        # Draw void triangles and Steiner tree branches
        for idx, void_tri in enumerate(void_triangles):
            if void_tri.shape == (3, 3):
                # Draw the 3 edges of the void triangle (cyan wireframe)
                for i in range(3):
                    v1 = void_tri[i]
                    v2 = void_tri[(i + 1) % 3]
                    line = pv.Line(v1, v2)
                    plotter.add_mesh(line, color='cyan', line_width=3, opacity=0.8)
                
                # Draw Steiner tree branches
                if idx < len(steiner_points):
                    sp = steiner_points[idx]
                    for i in range(3):
                        v = void_tri[i]
                        line = pv.Line(sp, v)
                        plotter.add_mesh(line, color='magenta', line_width=2, opacity=0.7)
    
    # Highlight specific VP positions (the VP being moved)
    if highlight_vp_positions is not None:
        vp_colors = highlight_vp_colors or ['yellow'] * len(highlight_vp_positions)
        vp_labels = highlight_vp_labels or [None] * len(highlight_vp_positions)
        for i, (pos, color) in enumerate(zip(highlight_vp_positions, vp_colors)):
            sphere = pv.Sphere(radius=vp_size, center=pos)
            plotter.add_mesh(sphere, color=color, opacity=1.0)
            
            # Add label if provided
            if i < len(vp_labels) and vp_labels[i] is not None:
                # Offset label slightly above the VP
                label_pos = pos + np.array([0, 0, vp_size * 3])
                plotter.add_point_labels(
                    [label_pos], [vp_labels[i]], 
                    font_size=10, text_color='black',
                    shape_color='white', shape_opacity=0.7,
                    always_visible=True
                )
    
    # Highlight specific triangles (affected by switch)
    if highlight_triangles is not None:
        for tri_idx in highlight_triangles:
            face = faces[tri_idx]
            tri_verts = vertices[face]
            # Draw triangle edges in yellow
            for i in range(3):
                v1 = tri_verts[i]
                v2 = tri_verts[(i + 1) % 3]
                line = pv.Line(v1, v2)
                plotter.add_mesh(line, color='yellow', line_width=5)
    
    # Highlight specific segments (cross-triangle segments)
    if highlight_segments is not None:
        for seg_info in highlight_segments:
            pos1, pos2, color = seg_info
            line = pv.Line(pos1, pos2)
            plotter.add_mesh(line, color=color, line_width=6, opacity=0.9)
    
    # Highlight specific mesh vertices (e.g., target vertex)
    if highlight_vertices is not None:
        vertex_colors = highlight_vertex_colors or ['purple'] * len(highlight_vertices)
        for pos, color in zip(highlight_vertices, vertex_colors):
            # Use a cube to distinguish from VP spheres
            cube = pv.Cube(center=pos, x_length=vp_size*1.5, y_length=vp_size*1.5, z_length=vp_size*1.5)
            plotter.add_mesh(cube, color=color, opacity=1.0)
    
    # Highlight specific mesh edges (e.g., current edge, target edge)
    if highlight_edges is not None:
        for edge_info in highlight_edges:
            v1_idx, v2_idx, color = edge_info[:3]
            line_width = edge_info[3] if len(edge_info) > 3 else 8
            v1_pos = vertices[v1_idx]
            v2_pos = vertices[v2_idx]
            line = pv.Line(v1_pos, v2_pos)
            plotter.add_mesh(line, color=color, line_width=line_width, opacity=1.0)
    
    # Add triangle labels if requested
    if triangle_indices_to_label is not None and len(triangle_indices_to_label) > 0:
        for tri_idx in triangle_indices_to_label:
            if tri_idx < len(faces):
                # Compute centroid of triangle
                face = faces[tri_idx]
                tri_vertices = vertices[face]
                centroid = np.mean(tri_vertices, axis=0)
                
                # Add label at centroid
                plotter.add_point_labels(
                    points=[centroid],
                    labels=[f"T{tri_idx}"],
                    font_size=12,
                    text_color='black',
                    render_points_as_spheres=False,
                    point_size=0.001,
                    always_visible=True
                )
    
    # Apply camera zoom if requested
    if camera_focus is not None and camera_zoom is not None:
        # Set camera position relative to focus point
        # Calculate a good viewing angle (slightly above and to the side)
        offset = np.array([camera_zoom, camera_zoom * 0.5, camera_zoom * 0.5])
        camera_position = camera_focus + offset
        
        plotter.camera_position = [
            camera_position,  # Camera location
            camera_focus,     # Focal point
            (0, 0, 1)        # View up direction
        ]
        
        # Adjust clipping range for close-up views
        plotter.camera.clipping_range = (camera_zoom * 0.1, camera_zoom * 10)
    
    return plotter


def get_neighbors_from_triangle_segments(partition, vp_idx):
    """Get neighbor VPs from triangle_segments."""
    neighbors = set()
    for tri_seg in partition.triangle_segments:
        if vp_idx in tri_seg.var_point_indices:
            for other_vp in tri_seg.var_point_indices:
                if other_vp != vp_idx:
                    neighbors.add(other_vp)
    return list(neighbors)


def compute_boundary_distance(partition, vp_idx: int) -> float:
    """
    Compute how far a boundary VP is from its target vertex.
    
    For λ < 0.5: VP approaching edge[1], distance = λ
    For λ > 0.5: VP approaching edge[0], distance = (1 - λ)
    
    Returns:
        Distance in [0, 0.5], where smaller = closer to target vertex
    """
    vp = partition.variable_points[vp_idx]
    if vp.lambda_param < 0.5:
        return vp.lambda_param
    else:
        return 1.0 - vp.lambda_param


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
    
    Returns:
        Filtered list with one VP per connected component
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
            
            deferred = [vp for _, vp in vps_with_dist[1:]]
            print(f"    Component {i+1}: {len(component)} connected VPs")
            print(f"      Keeping VP {closest_vp} (distance={closest_dist:.6f})")
            print(f"      Deferring: {deferred}")
    
    return vps_to_keep


def get_triangles_to_label_type1(partition, mesh_topology, vp_idx, neighbor_vps):
    """
    Get all triangles containing the moved VP or its neighbors for Type 1 switch.
    
    Args:
        partition: PartitionContour
        mesh_topology: MeshTopology
        vp_idx: Index of VP being moved
        neighbor_vps: List of neighbor VP indices
        
    Returns:
        List of triangle indices to label
    """
    triangles_to_label = set()
    
    # Add triangles for moved VP
    moved_vp_edge = partition.variable_points[vp_idx].edge
    triangles_to_label.update(
        mesh_topology.get_triangles_sharing_edge(tuple(sorted(moved_vp_edge)))
    )
    
    # Add triangles for each neighbor VP
    for n_idx in neighbor_vps:
        n_edge = partition.variable_points[n_idx].edge
        triangles_to_label.update(
            mesh_topology.get_triangles_sharing_edge(tuple(sorted(n_edge)))
        )
    
    return list(triangles_to_label)


def get_triangles_to_label_type2(partition, mesh_topology, triple_point):
    """
    Get all triangles containing the triple point VPs AND their neighbors for Type 2 switch.
    
    Args:
        partition: PartitionContour
        mesh_topology: MeshTopology
        triple_point: TriplePoint object
        
    Returns:
        List of triangle indices to label
    """
    triangles_to_label = set()
    
    # Add the void triangle itself
    triangles_to_label.add(triple_point.triangle_idx)
    
    # Add triangles for each VP in the triple point
    for vp_idx in triple_point.var_point_indices:
        vp_edge = partition.variable_points[vp_idx].edge
        triangles_to_label.update(
            mesh_topology.get_triangles_sharing_edge(tuple(sorted(vp_edge)))
        )
    
    # NEW: Add triangles for neighbors of each triple point VP
    for vp_idx in triple_point.var_point_indices:
        # Find neighbors using boundary_segments
        for seg in partition.boundary_segments:
            if seg.vp_idx_1 == vp_idx or seg.vp_idx_2 == vp_idx:
                # Get the other VP in this segment (the neighbor)
                neighbor_idx = seg.vp_idx_2 if seg.vp_idx_1 == vp_idx else seg.vp_idx_1
                
                # Add triangles containing this neighbor's edge
                neighbor_edge = partition.variable_points[neighbor_idx].edge
                triangles_to_label.update(
                    mesh_topology.get_triangles_sharing_edge(tuple(sorted(neighbor_edge)))
                )
    
    return list(triangles_to_label)


def run_visualization(args):
    """Main visualization routine."""
    # Load data using ContourAnalyzer (same as surface_visualization.py)
    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    
    # Create mesh and partition
    mesh, partition = load_partition_contour_from_analyzer(analyzer)
    
    # Load refined lambda values if available
    refined_path = args.solution.replace('.h5', '_refined_contours.h5')
    if os.path.exists(refined_path):
        lambda_opt = load_optimized_lambda_from_file(refined_path, partition)
        if lambda_opt is not None:
            print(f"✅ Loaded refined λ values from: {refined_path}")
    
    # Initialize topology components
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    steiner_handler = SteinerHandler(mesh, partition)
    
    print(f"Loaded: {len(partition.variable_points)} VPs, "
          f"{len(partition.triangle_segments)} triangle segments, "
          f"{len(steiner_handler.triple_points)} triple points")
    
    # Get contours and labels for visualization
    contours = partition.to_visualization_format()
    triangle_labels = analyzer.label_triangles_from_indicator()
    steiner_info = get_steiner_info(mesh, partition, steiner_handler)
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # =========================================================================
    # TYPE 1 SWITCH
    # =========================================================================
    if args.switch_type in ['type1', 'both']:
        print("\n" + "=" * 60)
        print("TYPE 1 SWITCH")
        print("=" * 60)
        
        # Find boundary VP not part of triple point
        boundary_vps = partition.get_boundary_variable_points(tol=0.1)
        triple_point_vp_indices = set()
        for tp in steiner_handler.triple_points:
            triple_point_vp_indices.update(tp.var_point_indices)
        
        non_triple_boundary_vps = [vp for vp in boundary_vps if vp not in triple_point_vp_indices]
        
        if non_triple_boundary_vps:
            # Filter connected VPs (keep closest in each component)
            print("\nFiltering connected boundary VPs...")
            filtered_vps = filter_connected_boundary_vps(non_triple_boundary_vps, partition)
            
            # Sort by distance to target vertex
            filtered_vps_sorted = sorted(
                filtered_vps,
                key=lambda vp_idx: compute_boundary_distance(partition, vp_idx)
            )
            
            # Select closest VP
            vp_idx = filtered_vps_sorted[0]
            print(f"Selected VP {vp_idx} (closest to target vertex)")
            
            vp = partition.variable_points[vp_idx]
            old_edge = vp.edge
            neighbors_before = get_neighbors_from_triangle_segments(partition, vp_idx)
            
            print(f"\nVP {vp_idx}: edge {old_edge}, λ = {vp.lambda_param:.4f}")
            print(f"Distance to target: {compute_boundary_distance(partition, vp_idx):.6f}")
            print(f"Neighbors: {neighbors_before}")
            
            # Calculate target vertex using the ACTUAL method from TopologySwitcher
            # CRITICAL λ CONVENTION: position = λ * edge[0] + (1-λ) * edge[1]
            # So: λ > 0.5 → approaching edge[0], λ < 0.5 → approaching edge[1]
            target_vertex_idx = switcher._identify_target_vertex(vp)
            target_vertex_pos = vertices[target_vertex_idx]
            other_vertex_idx = old_edge[1] if target_vertex_idx == old_edge[0] else old_edge[0]
            other_vertex_pos = vertices[other_vertex_idx]
            
            print(f"Target vertex: {target_vertex_idx} (λ = {vp.lambda_param:.4f}, {'>' if vp.lambda_param > 0.5 else '<='} 0.5)")
            print(f"Other vertex: {other_vertex_idx}")
            
            # Get affected triangles
            old_edge_norm = tuple(sorted(old_edge))
            affected_tris = mesh_topology.get_triangles_sharing_edge(old_edge_norm)
            
            # VP position
            vp_pos = partition.evaluate_variable_point(vp_idx)
            
            # Show BEFORE state
            if args.state in ['before', 'both']:
                print("\nShowing BEFORE state...")
                
                # Collect all VPs involved (main VP + neighbors)
                all_involved_vps = [vp_idx] + neighbors_before
                vp_positions_before = []
                vp_colors_before = []
                vp_labels_before = []
                
                # Main VP (being moved) - RED
                vp_positions_before.append(vp_pos)
                vp_colors_before.append('red')
                label = f"VP_moving (idx={vp_idx})\nedge: ({old_edge[0]}, {old_edge[1]})\nλ = {vp.lambda_param:.3f}"
                vp_labels_before.append(label)
                print(f"  {label.replace(chr(10), ' ')}")
                
                # Neighbor VPs - CYAN
                # Colors for neighbor VP edges
                neighbor_edge_colors = ['#00BFFF', '#8B4513', '#FF6347', '#9370DB']  # Sky Blue, Saddle Brown, Coral, Violet
                neighbor_edge_color_names = ['Sky Blue', 'Saddle Brown', 'Coral', 'Violet']
                
                for i, n_idx in enumerate(neighbors_before):
                    n_vp = partition.variable_points[n_idx]
                    n_pos = partition.evaluate_variable_point(n_idx)
                    n_edge = n_vp.edge
                    
                    vp_positions_before.append(n_pos)
                    vp_colors_before.append('cyan')
                    n_label = f"VP_neighbor{i+1} (idx={n_idx})\nedge: ({n_edge[0]}, {n_edge[1]})"
                    vp_labels_before.append(n_label)
                    print(f"  {n_label.replace(chr(10), ' ')}")
                
                # Highlight segments from VP to neighbors
                highlight_segs = []
                for n_idx in neighbors_before:
                    n_pos = partition.evaluate_variable_point(n_idx)
                    highlight_segs.append((vp_pos, n_pos, 'blue'))
                
                # Prepare target vertex highlight if requested
                highlight_verts = None
                highlight_vert_colors = None
                if args.show_target_vertex:
                    # Purple cube = target vertex (where algorithm will move VP)
                    # White cube = other vertex (alternative)
                    highlight_verts = [target_vertex_pos, other_vertex_pos]
                    highlight_vert_colors = ['purple', 'white']
                
                # Highlight edges: current edge (Hot Pink) and target edge (Lime Green)
                highlight_edge_list = []
                
                # Current edge in Hot Pink - "VP is here"
                highlight_edge_list.append((old_edge[0], old_edge[1], '#FF1493', 10))  # Hot Pink
                print(f"  Current edge (Hot Pink): {old_edge}")
                
                # Neighbor VP edges in distinct colors
                for i, n_idx in enumerate(neighbors_before):
                    n_vp = partition.variable_points[n_idx]
                    n_edge = n_vp.edge
                    color = neighbor_edge_colors[i % len(neighbor_edge_colors)]
                    color_name = neighbor_edge_color_names[i % len(neighbor_edge_color_names)]
                    highlight_edge_list.append((n_edge[0], n_edge[1], color, 8))
                    print(f"  Neighbor {i+1} edge ({color_name}): {n_edge}")
                
                # Get the ACTUAL target edge that will be selected (not all candidates)
                best_target_edge = switcher.get_best_target_edge_for_type1(vp_idx, tol=0.1)
                if best_target_edge:
                    highlight_edge_list.append((best_target_edge[0], best_target_edge[1], '#00FF00', 10))  # Lime Green
                    print(f"  Target edge (Lime Green): {best_target_edge}")
                else:
                    print(f"  WARNING: No valid target edge found!")
                
                # Collect triangles to label (all triangles containing VP or neighbors)
                triangles_to_label_before = get_triangles_to_label_type1(
                    partition, mesh_topology, vp_idx, neighbors_before
                )
                print(f"  Labeling {len(triangles_to_label_before)} triangles: {sorted(triangles_to_label_before)}")
                
                plotter = plot_partition_with_switch_highlight(
                    vertices, faces, contours, triangle_labels,
                    steiner_info=steiner_info,
                    highlight_vp_positions=vp_positions_before,
                    highlight_vp_colors=vp_colors_before,
                    highlight_vp_labels=vp_labels_before,
                    highlight_triangles=affected_tris,
                    triangle_indices_to_label=triangles_to_label_before,
                    highlight_segments=highlight_segs,
                    highlight_vertices=highlight_verts,
                    highlight_vertex_colors=highlight_vert_colors,
                    highlight_edges=highlight_edge_list,
                    title=f"Type 1 BEFORE: VP {vp_idx} on edge {old_edge} (target vertex: {target_vertex_idx})",
                    steiner_size=args.steiner_size,
                    vp_size=args.vp_size,
                    camera_focus=vp_pos if args.apply_zoom else None,
                    camera_zoom=args.zoom_factor if args.apply_zoom else None,
                )
                plotter.show(interactive_update=True)
            
            # Apply switch
            if args.state in ['after', 'both']:
                print("\nApplying Type 1 switch...")
                success = switcher.apply_type1_switch(vp_idx, tol=0.1)
                
                if success:
                    partition.rebuild_triangle_segments_from_current_vps()
                    
                    # Classify segments and populate crossing cache
                    switcher.classify_all_segments()
                    
                    new_edge = vp.edge
                    new_vp_pos = partition.evaluate_variable_point(vp_idx)
                    
                    # Get new affected triangles
                    new_edge_norm = tuple(sorted(new_edge))
                    new_affected_tris = mesh_topology.get_triangles_sharing_edge(new_edge_norm)
                    
                    # Update contours and steiner info
                    contours_after = partition.to_visualization_format()
                    steiner_after = SteinerHandler(mesh, partition)
                    steiner_info_after = get_steiner_info(mesh, partition, steiner_after)
                    
                    print(f"VP {vp_idx} moved: {old_edge} → {new_edge}")
                    
                    # Collect all VPs involved (main VP + neighbors) - AFTER
                    neighbors_after = get_neighbors_from_triangle_segments(partition, vp_idx)
                    vp_positions_after = []
                    vp_colors_after = []
                    vp_labels_after = []
                    
                    # Main VP (was moved) - GREEN
                    vp_positions_after.append(new_vp_pos)
                    vp_colors_after.append('green')
                    label = f"VP_moved (idx={vp_idx})\nedge: ({new_edge[0]}, {new_edge[1]})\nλ = {vp.lambda_param:.3f}"
                    vp_labels_after.append(label)
                    print(f"  {label.replace(chr(10), ' ')}")
                    
                    # Neighbor VPs (original neighbors) - YELLOW
                    # Colors for neighbor VP edges (same as BEFORE for consistency)
                    neighbor_edge_colors = ['#00BFFF', '#8B4513', '#FF6347', '#9370DB']  # Sky Blue, Saddle Brown, Coral, Violet
                    neighbor_edge_color_names = ['Sky Blue', 'Saddle Brown', 'Coral', 'Violet']
                    
                    for i, n_idx in enumerate(neighbors_before):
                        n_vp = partition.variable_points[n_idx]
                        n_pos = partition.evaluate_variable_point(n_idx)
                        n_edge = n_vp.edge
                        
                        vp_positions_after.append(n_pos)
                        vp_colors_after.append('yellow')
                        n_label = f"VP_neighbor{i+1} (idx={n_idx})\nedge: ({n_edge[0]}, {n_edge[1]})"
                        vp_labels_after.append(n_label)
                        print(f"  {n_label.replace(chr(10), ' ')}")
                    
                    # Highlight segments from VP to neighbors (original neighbors)
                    highlight_segs = []
                    for n_idx in neighbors_before:
                        n_pos = partition.evaluate_variable_point(n_idx)
                        highlight_segs.append((new_vp_pos, n_pos, 'white'))
                    
                    # Highlight BOTH edges: old edge (Hot Pink) and new edge (Lime Green)
                    highlight_edge_list = []
                    
                    # Old edge in Hot Pink - "VP was here"
                    highlight_edge_list.append((old_edge[0], old_edge[1], '#FF1493', 10))  # Hot Pink
                    print(f"  Old edge (Hot Pink): {old_edge}")
                    
                    # New edge in Lime Green - "VP moved here"
                    highlight_edge_list.append((new_edge[0], new_edge[1], '#00FF00', 10))  # Lime Green
                    print(f"  New edge (Lime Green): {new_edge}")
                    
                    # Neighbor VP edges in distinct colors (same as BEFORE)
                    for i, n_idx in enumerate(neighbors_before):
                        n_vp = partition.variable_points[n_idx]
                        n_edge = n_vp.edge
                        color = neighbor_edge_colors[i % len(neighbor_edge_colors)]
                        color_name = neighbor_edge_color_names[i % len(neighbor_edge_color_names)]
                        highlight_edge_list.append((n_edge[0], n_edge[1], color, 8))
                        print(f"  Neighbor {i+1} edge ({color_name}): {n_edge}")
                    
                    # Extract crossing point from cache (if available)
                    # The crossing happens on the shared edge between old and new triangles
                    crossing_positions = []
                    crossing_colors = []
                    crossing_labels = []
                    crossing_edges_set = set()  # Track unique crossing edges
                    
                    if partition.segment_crossing_cache:
                        print("\nSegment crossing information:")
                        
                        # Get VP positions for comparison
                        vp_positions_dict = {}
                        for n_idx in neighbors_before + [vp_idx]:
                            vp_pos = partition.evaluate_variable_point(n_idx)
                            vp_positions_dict[n_idx] = vp_pos
                        
                        for tri_idx, crossings in partition.segment_crossing_cache.items():
                            for crossing in crossings:
                                seg_key = crossing.segment
                                # Check if this crossing involves the moved VP
                                if vp_idx in seg_key:
                                    # Get VP positions for this segment
                                    vp1_pos = vp_positions_dict.get(seg_key[0])
                                    vp2_pos = vp_positions_dict.get(seg_key[1])
                                    
                                    # Check if entry/exit points are at VP positions (within tolerance)
                                    tol = 1e-8
                                    entry_is_vp1 = vp1_pos is not None and np.linalg.norm(crossing.entry_point - vp1_pos) < tol
                                    entry_is_vp2 = vp2_pos is not None and np.linalg.norm(crossing.entry_point - vp2_pos) < tol
                                    exit_is_vp1 = vp1_pos is not None and np.linalg.norm(crossing.exit_point - vp1_pos) < tol
                                    exit_is_vp2 = vp2_pos is not None and np.linalg.norm(crossing.exit_point - vp2_pos) < tol
                                    
                                    entry_is_vp = entry_is_vp1 or entry_is_vp2
                                    exit_is_vp = exit_is_vp1 or exit_is_vp2
                                    
                                    # Determine what to display and plot
                                    if entry_is_vp and exit_is_vp:
                                        # Both endpoints are VPs (normal segment in single triangle)
                                        print(f"  Triangle {tri_idx}: Normal segment (both VPs in triangle)")
                                    elif entry_is_vp:
                                        # Entry is VP, exit is computed crossing
                                        vp_id = seg_key[0] if entry_is_vp1 else seg_key[1]
                                        print(f"  Triangle {tri_idx}: VP {vp_id} at entry, computed crossing at exit edge {crossing.exit_edge}")
                                        crossing_positions.append(crossing.exit_point)
                                        crossing_colors.append('magenta')
                                        crossing_labels.append(f"Crossing\nedge={crossing.exit_edge}\ntri={tri_idx}")
                                        # Track crossing edge
                                        crossing_edges_set.add(tuple(sorted(crossing.exit_edge)))
                                    elif exit_is_vp:
                                        # Exit is VP, entry is computed crossing
                                        vp_id = seg_key[0] if exit_is_vp1 else seg_key[1]
                                        print(f"  Triangle {tri_idx}: Computed crossing at entry edge {crossing.entry_edge}, VP {vp_id} at exit")
                                        crossing_positions.append(crossing.entry_point)
                                        crossing_colors.append('magenta')
                                        crossing_labels.append(f"Crossing\nedge={crossing.entry_edge}\ntri={tri_idx}")
                                        # Track crossing edge
                                        crossing_edges_set.add(tuple(sorted(crossing.entry_edge)))
                                    elif crossing.is_vertex_crossing:
                                        # Vertex crossing (neither VP is at entry/exit)
                                        print(f"  Triangle {tri_idx}: Vertex crossing at vertex {crossing.entry_vertex}")
                                        crossing_positions.append(crossing.entry_point)
                                        crossing_colors.append('magenta')
                                        crossing_labels.append(f"Vertex\nv={crossing.entry_vertex}\ntri={tri_idx}")
                                    else:
                                        # True edge crossing (intermediate triangle)
                                        print(f"  Triangle {tri_idx}: Computed edge crossing")
                                        print(f"    Entry edge {crossing.entry_edge}: {crossing.entry_point}")
                                        print(f"    Exit edge {crossing.exit_edge}: {crossing.exit_point}")
                                        crossing_positions.append(crossing.entry_point)
                                        crossing_colors.append('magenta')
                                        crossing_labels.append(f"Crossing\nedge={crossing.entry_edge}\ntri={tri_idx}")
                                        # Track both edges for intermediate triangle crossings
                                        crossing_edges_set.add(tuple(sorted(crossing.entry_edge)))
                                        crossing_edges_set.add(tuple(sorted(crossing.exit_edge)))
                        
                        # Add crossing edges to highlight list if requested
                        if args.show_crossing_edges and crossing_edges_set:
                            print("\nCrossing edges (highlighted in Cyan):")
                            for edge in sorted(crossing_edges_set):
                                print(f"  Edge {edge}")
                                highlight_edge_list.append((edge[0], edge[1], '#00FFFF', 8))  # Cyan
                    
                    # Combine VP positions with crossing positions
                    all_positions = vp_positions_after + crossing_positions
                    all_colors = vp_colors_after + crossing_colors
                    all_labels = vp_labels_after + crossing_labels
                    
                    # Collect triangles to label (all triangles containing VP or neighbors AFTER migration)
                    triangles_to_label_after = get_triangles_to_label_type1(
                        partition, mesh_topology, vp_idx, neighbors_before
                    )
                    print(f"  Labeling {len(triangles_to_label_after)} triangles: {sorted(triangles_to_label_after)}")
                    
                    plotter = plot_partition_with_switch_highlight(
                        vertices, faces, contours_after, triangle_labels,
                        steiner_info=steiner_info_after,
                        highlight_vp_positions=all_positions,
                        highlight_vp_colors=all_colors,
                        highlight_vp_labels=all_labels,
                        highlight_triangles=new_affected_tris,
                        triangle_indices_to_label=triangles_to_label_after,
                        highlight_segments=highlight_segs,
                        highlight_edges=highlight_edge_list,
                        title=f"Type 1 AFTER: VP {vp_idx} moved to edge {new_edge}",
                        steiner_size=args.steiner_size,
                        vp_size=args.vp_size,
                        camera_focus=new_vp_pos if args.apply_zoom else None,
                        camera_zoom=args.zoom_factor if args.apply_zoom else None,
                    )
                    plotter.show()
                else:
                    print("Type 1 switch failed!")
        else:
            print("No non-triple boundary VPs found for Type 1 switch")
    
    # =========================================================================
    # TYPE 2 SWITCH - Reload data if we did Type 1
    # =========================================================================
    if args.switch_type in ['type2', 'both']:
        if args.switch_type == 'both':
            print("\n" + "=" * 60)
            print("Reloading data for Type 2...")
            analyzer = ContourAnalyzer(args.solution)
            analyzer.load_results(use_initial_condition=False)
            mesh, partition = load_partition_contour_from_analyzer(analyzer)
            if os.path.exists(refined_path):
                load_optimized_lambda_from_file(refined_path, partition)
            mesh_topology = MeshTopology(mesh)
            switcher = TopologySwitcher(mesh, partition, mesh_topology)
            steiner_handler = SteinerHandler(mesh, partition)
            contours = partition.to_visualization_format()
            triangle_labels = analyzer.label_triangles_from_indicator()
            steiner_info = get_steiner_info(mesh, partition, steiner_handler)
            vertices = mesh.vertices
            faces = mesh.faces
        
        print("\n" + "=" * 60)
        print("TYPE 2 SWITCH (Triple Point Migration)")
        print("=" * 60)
        
        boundary_tps = steiner_handler.get_boundary_triple_points(tol=0.1)
        
        if boundary_tps:
            tp = boundary_tps[0]
            print(f"Triple point at triangle {tp.triangle_idx}")
            print(f"VPs: {tp.var_point_indices}")
            
            # Capture VP edges before
            vp_edges_before = {vp_idx: partition.variable_points[vp_idx].edge 
                              for vp_idx in tp.var_point_indices}
            
            # VP positions
            vp_positions = [partition.evaluate_variable_point(vp_idx) 
                          for vp_idx in tp.var_point_indices]
            
            # Show BEFORE state
            if args.state in ['before', 'both']:
                print("\nShowing BEFORE state...")
                
                # Create labels with VP index and edge vertices
                vp_labels_before = []
                for i, vp_idx in enumerate(tp.var_point_indices):
                    edge = vp_edges_before[vp_idx]
                    label = f"VP{i+1} (idx={vp_idx})\nedge: ({edge[0]}, {edge[1]})"
                    vp_labels_before.append(label)
                    print(f"  {label.replace(chr(10), ' ')}")
                
                # Calculate centroid of triple point for camera focus
                steiner_pos = tp.compute_steiner_point()
                
                # Collect triangles to label (void triangle + all triangles containing triple point VPs)
                triangles_to_label_before_type2 = get_triangles_to_label_type2(
                    partition, mesh_topology, tp
                )
                print(f"  Labeling {len(triangles_to_label_before_type2)} triangles: {sorted(triangles_to_label_before_type2)}")
                
                plotter = plot_partition_with_switch_highlight(
                    vertices, faces, contours, triangle_labels,
                    steiner_info=steiner_info,
                    highlight_vp_positions=vp_positions,
                    highlight_vp_colors=['red', 'blue', 'green'],
                    highlight_vp_labels=vp_labels_before,
                    highlight_triangles=[tp.triangle_idx],
                    triangle_indices_to_label=triangles_to_label_before_type2,
                    title=f"Type 2 BEFORE: Triple point at triangle {tp.triangle_idx}",
                    steiner_size=args.steiner_size,
                    vp_size=args.vp_size,
                    camera_focus=steiner_pos if args.apply_zoom else None,
                    camera_zoom=args.zoom_factor if args.apply_zoom else None,
                )
                plotter.show(interactive_update=True)
            
            # Apply switch
            if args.state in ['after', 'both']:
                print("\nApplying Type 2 switch...")
                success = switcher.apply_type2_switch(tp, tol=0.1)
                
                if success:
                    partition.rebuild_triangle_segments_from_current_vps()
                    
                    # Detect which VP moved
                    moved_vp_idx = None
                    for vp_idx, old_edge in vp_edges_before.items():
                        current_edge = partition.variable_points[vp_idx].edge
                        if current_edge != old_edge:
                            moved_vp_idx = vp_idx
                            print(f"Moved VP: {vp_idx}, edge {old_edge} → {current_edge}")
                            break
                    
                    # Re-detect triple points
                    steiner_after = SteinerHandler(mesh, partition)
                    steiner_info_after = get_steiner_info(mesh, partition, steiner_after)
                    contours_after = partition.to_visualization_format()
                    
                    # Find new triple point containing the moved VP
                    new_tp = None
                    for tp_candidate in steiner_after.triple_points:
                        if moved_vp_idx in tp_candidate.var_point_indices:
                            new_tp = tp_candidate
                            break
                    
                    if new_tp:
                        print(f"New triple point at triangle {new_tp.triangle_idx}")
                        
                        # VP positions after
                        new_vp_positions = [partition.evaluate_variable_point(vp_idx) 
                                           for vp_idx in new_tp.var_point_indices]
                        
                        # Colors and labels: moved VP in orange, others in cyan
                        new_vp_colors = []
                        new_vp_labels = []
                        for vp_idx in new_tp.var_point_indices:
                            edge = partition.variable_points[vp_idx].edge
                            if vp_idx == moved_vp_idx:
                                new_vp_colors.append('orange')
                                label = f"VP1 MOVED (idx={vp_idx})\nedge: ({edge[0]}, {edge[1]})"
                            else:
                                new_vp_colors.append('cyan')
                                label = f"VP (idx={vp_idx})\nedge: ({edge[0]}, {edge[1]})"
                            new_vp_labels.append(label)
                            print(f"  {label.replace(chr(10), ' ')}")
                        
                        # Analyze crossing points (if any) - diagnostic only for Type 2
                        # NOTE: Crossing edge highlighting (cyan) is only for Type 1 migration, not Type 2
                        switcher.classify_all_segments()
                        
                        if partition.segment_crossing_cache:
                            print("\nSegment crossing information (diagnostic only):")
                            
                            # Get VP positions for comparison
                            vp_positions_dict = {}
                            for vp_idx in new_tp.var_point_indices:
                                vp_pos = partition.evaluate_variable_point(vp_idx)
                                vp_positions_dict[vp_idx] = vp_pos
                            
                            for tri_idx, crossings in partition.segment_crossing_cache.items():
                                for crossing in crossings:
                                    seg_key = crossing.segment
                                    # Check if this crossing involves any VP from the triple point
                                    if seg_key[0] in new_tp.var_point_indices or seg_key[1] in new_tp.var_point_indices:
                                        # Get VP positions for this segment
                                        vp1_pos = vp_positions_dict.get(seg_key[0])
                                        vp2_pos = vp_positions_dict.get(seg_key[1])
                                        
                                        # Check if entry/exit points are at VP positions (within tolerance)
                                        tol = 1e-8
                                        entry_is_vp1 = vp1_pos is not None and np.linalg.norm(crossing.entry_point - vp1_pos) < tol
                                        entry_is_vp2 = vp2_pos is not None and np.linalg.norm(crossing.entry_point - vp2_pos) < tol
                                        exit_is_vp1 = vp1_pos is not None and np.linalg.norm(crossing.exit_point - vp1_pos) < tol
                                        exit_is_vp2 = vp2_pos is not None and np.linalg.norm(crossing.exit_point - vp2_pos) < tol
                                        
                                        entry_is_vp = entry_is_vp1 or entry_is_vp2
                                        exit_is_vp = exit_is_vp1 or exit_is_vp2
                                        
                                        # Determine what to display (diagnostic only, no edge highlighting)
                                        if entry_is_vp and exit_is_vp:
                                            # Both endpoints are VPs (normal segment in single triangle)
                                            print(f"  Triangle {tri_idx}: Normal segment (both VPs in triangle)")
                                        elif entry_is_vp:
                                            # Entry is VP, exit is computed crossing
                                            vp_id = seg_key[0] if entry_is_vp1 else seg_key[1]
                                            print(f"  Triangle {tri_idx}: VP {vp_id} at entry, computed crossing at exit edge {crossing.exit_edge}")
                                        elif exit_is_vp:
                                            # Exit is VP, entry is computed crossing
                                            vp_id = seg_key[0] if exit_is_vp1 else seg_key[1]
                                            print(f"  Triangle {tri_idx}: Computed crossing at entry edge {crossing.entry_edge}, VP {vp_id} at exit")
                                        elif crossing.is_vertex_crossing:
                                            # Vertex crossing (neither VP is at entry/exit)
                                            print(f"  Triangle {tri_idx}: Vertex crossing at vertex {crossing.entry_vertex}")
                                        else:
                                            # True edge crossing (intermediate triangle)
                                            print(f"  Triangle {tri_idx}: Computed edge crossing")
                                            print(f"    Entry edge {crossing.entry_edge}: {crossing.entry_point}")
                                            print(f"    Exit edge {crossing.exit_edge}: {crossing.exit_point}")
                        
                        # Calculate new Steiner position for camera focus
                        new_steiner_pos = new_tp.compute_steiner_point()
                        
                        # No crossing edge highlighting for Type 2 (only for Type 1)
                        highlight_edge_list = None
                        
                        # Collect triangles to label (void triangle + all triangles containing triple point VPs AFTER migration)
                        triangles_to_label_after_type2 = get_triangles_to_label_type2(
                            partition, mesh_topology, new_tp
                        )
                        print(f"  Labeling {len(triangles_to_label_after_type2)} triangles: {sorted(triangles_to_label_after_type2)}")
                        
                        plotter = plot_partition_with_switch_highlight(
                            vertices, faces, contours_after, triangle_labels,
                            steiner_info=steiner_info_after,
                            highlight_vp_positions=new_vp_positions,
                            highlight_vp_colors=new_vp_colors,
                            highlight_vp_labels=new_vp_labels,
                            highlight_triangles=[new_tp.triangle_idx, tp.triangle_idx],
                            triangle_indices_to_label=triangles_to_label_after_type2,
                            highlight_edges=highlight_edge_list if highlight_edge_list else None,
                            title=f"Type 2 AFTER: Triple point migrated to triangle {new_tp.triangle_idx}",
                            steiner_size=args.steiner_size,
                            vp_size=args.vp_size,
                            camera_focus=new_steiner_pos if args.apply_zoom else None,
                            camera_zoom=args.zoom_factor if args.apply_zoom else None,
                        )
                        plotter.show()
                else:
                    print("Type 2 switch failed!")
        else:
            print("No boundary triple points found for Type 2 switch")
    
    print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize topology switches with partition coloring, contours, and Steiner points"
    )
    parser.add_argument(
        '--solution', type=str, required=True,
        help='Path to solution .h5 file (same as surface_visualization.py)'
    )
    parser.add_argument(
        '--switch-type', type=str, choices=['type1', 'type2', 'both'], default='type1',
        help='Which switch type to visualize (default: type1)'
    )
    parser.add_argument(
        '--state', type=str, choices=['before', 'after', 'both'], default='both',
        help='Which state to show (default: both)'
    )
    parser.add_argument(
        '--vp-size', type=float, default=0.01,
        help='Size of VP spheres (default: 0.01)'
    )
    parser.add_argument(
        '--steiner-size', type=float, default=0.008,
        help='Size of Steiner point spheres (default: 0.008)'
    )
    parser.add_argument(
        '--show-target-vertex', action='store_true',
        help='Show target vertex for Type 1 switch (purple cube)'
    )
    parser.add_argument(
        '--apply-zoom', action='store_true',
        help='Automatically zoom and focus camera on the VP being migrated'
    )
    parser.add_argument(
        '--zoom-factor', type=float, default=0.05,
        help='Zoom level (default: 0.05, smaller = more zoomed in)'
    )
    parser.add_argument(
        '--show-crossing-edges', action='store_true',
        help='Highlight mesh edges where segment crossings occur (displayed in Cyan)'
    )
    
    args = parser.parse_args()
    run_visualization(args)


if __name__ == '__main__':
    main()
