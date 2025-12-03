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
    """Create PartitionContour with Phase 1 structures from ContourAnalyzer data."""
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    indicator_functions = analyzer.compute_indicator_functions()
    partition = PartitionContour(mesh, indicator_functions)
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
    highlight_triangles: list = None,
    highlight_segments: list = None,
    highlight_vertices: list = None,
    highlight_vertex_colors: list = None,
    title: str = "Topology Switch Visualization",
    steiner_size: float = 0.008,
    vp_size: float = 0.01,
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
        for pos, color in zip(highlight_vp_positions, vp_colors):
            sphere = pv.Sphere(radius=vp_size, center=pos)
            plotter.add_mesh(sphere, color=color, opacity=1.0)
    
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
            vp_idx = non_triple_boundary_vps[0]
            vp = partition.variable_points[vp_idx]
            old_edge = vp.edge
            neighbors_before = get_neighbors_from_triangle_segments(partition, vp_idx)
            
            print(f"VP {vp_idx}: edge {old_edge}, λ = {vp.lambda_param:.4f}")
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
                
                plotter = plot_partition_with_switch_highlight(
                    vertices, faces, contours, triangle_labels,
                    steiner_info=steiner_info,
                    highlight_vp_positions=[vp_pos],
                    highlight_vp_colors=['red'],
                    highlight_triangles=affected_tris,
                    highlight_segments=highlight_segs,
                    highlight_vertices=highlight_verts,
                    highlight_vertex_colors=highlight_vert_colors,
                    title=f"Type 1 BEFORE: VP {vp_idx} on edge {old_edge} (target vertex: {target_vertex_idx})",
                    steiner_size=args.steiner_size,
                    vp_size=args.vp_size,
                )
                plotter.show(interactive_update=True)
            
            # Apply switch
            if args.state in ['after', 'both']:
                print("\nApplying Type 1 switch...")
                success = switcher.apply_type1_switch(vp_idx, tol=0.1)
                
                if success:
                    partition.rebuild_triangle_segments_from_current_vps()
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
                    
                    # Highlight segments from VP to neighbors (original neighbors)
                    highlight_segs = []
                    for n_idx in neighbors_before:
                        n_pos = partition.evaluate_variable_point(n_idx)
                        highlight_segs.append((new_vp_pos, n_pos, 'orange'))
                    
                    plotter = plot_partition_with_switch_highlight(
                        vertices, faces, contours_after, triangle_labels,
                        steiner_info=steiner_info_after,
                        highlight_vp_positions=[new_vp_pos],
                        highlight_vp_colors=['green'],
                        highlight_triangles=new_affected_tris,
                        highlight_segments=highlight_segs,
                        title=f"Type 1 AFTER: VP {vp_idx} moved to edge {new_edge}",
                        steiner_size=args.steiner_size,
                        vp_size=args.vp_size,
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
                
                plotter = plot_partition_with_switch_highlight(
                    vertices, faces, contours, triangle_labels,
                    steiner_info=steiner_info,
                    highlight_vp_positions=vp_positions,
                    highlight_vp_colors=['red', 'blue', 'green'],
                    highlight_triangles=[tp.triangle_idx],
                    title=f"Type 2 BEFORE: Triple point at triangle {tp.triangle_idx}",
                    steiner_size=args.steiner_size,
                    vp_size=args.vp_size,
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
                        
                        # Colors: moved VP in orange, others in original colors
                        new_vp_colors = []
                        for vp_idx in new_tp.var_point_indices:
                            if vp_idx == moved_vp_idx:
                                new_vp_colors.append('orange')
                            else:
                                new_vp_colors.append('cyan')
                        
                        plotter = plot_partition_with_switch_highlight(
                            vertices, faces, contours_after, triangle_labels,
                            steiner_info=steiner_info_after,
                            highlight_vp_positions=new_vp_positions,
                            highlight_vp_colors=new_vp_colors,
                            highlight_triangles=[new_tp.triangle_idx, tp.triangle_idx],
                            title=f"Type 2 AFTER: Triple point migrated to triangle {new_tp.triangle_idx}",
                            steiner_size=args.steiner_size,
                            vp_size=args.vp_size,
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
    
    args = parser.parse_args()
    run_visualization(args)


if __name__ == '__main__':
    main()
