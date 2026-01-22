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


def load_partition_from_refined_file(refined_path, verbose=False):
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
    
    if verbose:
        print(f"Loading from refined contours file...")
        print(f"  Refined: {refined_path}")
    
    # Derive base solution path
    base_solution_path = refined_path.replace('_refined_contours.h5', '.h5')
    if verbose:
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
    if verbose:
        print(f"  ✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Compute indicator functions
    indicator_functions = analyzer.compute_indicator_functions()
    n_cells = indicator_functions.shape[1]
    if verbose:
        print(f"  ✓ Computed indicator functions: {n_cells} cells")
    
    # Extract boundary topology (efficient initialization)
    raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
    n_boundary_triangles = sum(len(v) for v in boundary_topology.values())
    if verbose:
        print(f"  ✓ Extracted boundary topology: {n_boundary_triangles} boundary triangles")
    
    # Create partition (this initializes VPs with default λ=0.5)
    partition = PartitionContour(mesh, indicator_functions, boundary_topology=boundary_topology)
    if verbose:
        print(f"  ✓ Created partition: {len(partition.variable_points)} VPs")
    
    # Load optimized λ parameters from refined file
    with h5py.File(refined_path, 'r') as f:
        if 'lambda_parameters' in f:
            lambda_opt = f['lambda_parameters'][:]
            partition.set_variable_vector(lambda_opt)
            if verbose:
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


def debug_type1_convergence_analysis(mesh, partition, mesh_topology, vp_idx, boundary_tol=0.1, verbose=False):
    """
    DEBUG SECTION 7: Analyze VP convergence pattern BEFORE Type 1 migration.
    
    Tests the hypothesis that the migrating VP and its neighbors are all
    converging toward the same mesh vertex.
    """
    if not verbose:
        return {'target_vertex': None, 'target_pos': None, 'all_share_target': False, 'all_close': False}
    
    print("\n" + "="*80)
    print("DEBUG SECTION 7: TYPE 1 VP CONVERGENCE ANALYSIS (BEFORE MIGRATION)")
    print("="*80)
    
    vp = partition.variable_points[vp_idx]
    vp_pos = vp.evaluate(mesh.vertices)
    vp_edge = vp.edge
    v_a, v_b = vp_edge
    
    print(f"\n━━━ MIGRATING VP {vp_idx} ━━━")
    print(f"  Edge: ({v_a}, {v_b})")
    print(f"  Lambda: {vp.lambda_param:.6f}")
    print(f"  Position: {vp_pos}")
    print(f"  Belongs to cells: {vp.belongs_to_cells}")
    
    # Determine target vertex based on lambda
    # VP position = (1-λ) * v_a + λ * v_b
    # If λ → 0: VP approaches v_a
    # If λ → 1: VP approaches v_b
    if vp.lambda_param < 0.5:
        target_vertex_idx = v_a
        distance_to_target = vp.lambda_param  # Distance parameter (0 = at v_a)
    else:
        target_vertex_idx = v_b
        distance_to_target = 1.0 - vp.lambda_param  # Distance parameter (0 = at v_b)
    
    target_vertex_pos = mesh.vertices[target_vertex_idx]
    actual_distance = np.linalg.norm(vp_pos - target_vertex_pos)
    
    print(f"\n  TARGET VERTEX ANALYSIS:")
    print(f"    λ = {vp.lambda_param:.6f} {'< 0.5 → approaching v_a' if vp.lambda_param < 0.5 else '>= 0.5 → approaching v_b'}")
    print(f"    Target vertex index: {target_vertex_idx}")
    print(f"    Target vertex position: {target_vertex_pos}")
    print(f"    Distance to target (λ-based): {distance_to_target:.6f}")
    print(f"    Distance to target (actual): {actual_distance:.6f}")
    print(f"    Is close to vertex? {distance_to_target < boundary_tol} (tol={boundary_tol})")
    
    # Find neighbor VPs from boundary_segments
    print(f"\n━━━ NEIGHBOR VP ANALYSIS ━━━")
    neighbor_vps = []
    for seg in partition.boundary_segments:
        if seg.vp_idx_1 == vp_idx:
            neighbor_vps.append(seg.vp_idx_2)
        elif seg.vp_idx_2 == vp_idx:
            neighbor_vps.append(seg.vp_idx_1)
    
    print(f"  Found {len(neighbor_vps)} neighbor VPs: {neighbor_vps}")
    
    neighbor_convergence_info = []
    for i, neighbor_idx in enumerate(neighbor_vps):
        neighbor_vp = partition.variable_points[neighbor_idx]
        neighbor_pos = neighbor_vp.evaluate(mesh.vertices)
        neighbor_edge = neighbor_vp.edge
        n_v_a, n_v_b = neighbor_edge
        
        print(f"\n  NEIGHBOR VP {neighbor_idx}:")
        print(f"    Edge: ({n_v_a}, {n_v_b})")
        print(f"    Lambda: {neighbor_vp.lambda_param:.6f}")
        print(f"    Position: {neighbor_pos}")
        
        # Check if target vertex is on neighbor's edge
        target_on_neighbor_edge = target_vertex_idx in [n_v_a, n_v_b]
        print(f"    Target vertex {target_vertex_idx} on this edge? {target_on_neighbor_edge}")
        
        # Distance from neighbor VP to target vertex
        dist_to_target = np.linalg.norm(neighbor_pos - target_vertex_pos)
        print(f"    Distance to target vertex: {dist_to_target:.6f}")
        print(f"    Is close to target? {dist_to_target < boundary_tol * 2} (tol={boundary_tol * 2})")
        
        # If target is on neighbor's edge, compute lambda distance
        if target_on_neighbor_edge:
            if target_vertex_idx == n_v_a:
                neighbor_lambda_dist = neighbor_vp.lambda_param
            else:
                neighbor_lambda_dist = 1.0 - neighbor_vp.lambda_param
            print(f"    Lambda distance to target: {neighbor_lambda_dist:.6f}")
        else:
            neighbor_lambda_dist = None
        
        neighbor_convergence_info.append({
            'idx': neighbor_idx,
            'edge': neighbor_edge,
            'lambda': neighbor_vp.lambda_param,
            'pos': neighbor_pos,
            'target_on_edge': target_on_neighbor_edge,
            'dist_to_target': dist_to_target,
            'lambda_dist': neighbor_lambda_dist
        })
    
    # Summary
    print(f"\n━━━ CONVERGENCE SUMMARY ━━━")
    print(f"  Target vertex: {target_vertex_idx}")
    print(f"  Target position: {target_vertex_pos}")
    print(f"\n  VP distances to target vertex {target_vertex_idx}:")
    print(f"    VP {vp_idx} (migrating): {actual_distance:.6f} {'✓ CLOSE' if distance_to_target < boundary_tol else ''}")
    
    all_share_target = True
    all_close = distance_to_target < boundary_tol
    
    for info in neighbor_convergence_info:
        close_marker = '✓ CLOSE' if info['dist_to_target'] < boundary_tol * 2 else ''
        shares_marker = '(shares edge)' if info['target_on_edge'] else '(different edge)'
        print(f"    VP {info['idx']} (neighbor): {info['dist_to_target']:.6f} {close_marker} {shares_marker}")
        
        if not info['target_on_edge']:
            all_share_target = False
        if info['dist_to_target'] >= boundary_tol * 2:
            all_close = False
    
    print(f"\n  ★ ALL NEIGHBORS SHARE TARGET VERTEX? {all_share_target}")
    print(f"  ★ ALL VPs CLOSE TO TARGET? {all_close}")
    
    if all_share_target and all_close:
        print(f"\n  ★★★ VERTEX COLLAPSE PATTERN DETECTED! ★★★")
        print(f"  All {len(neighbor_vps) + 1} VPs are converging to vertex {target_vertex_idx}")
        print(f"  This suggests a simpler migration approach may be possible.")
    
    # List all edges meeting at target vertex
    print(f"\n━━━ VERTEX STAR TOPOLOGY ━━━")
    print(f"  Edges emanating from vertex {target_vertex_idx}:")
    
    # Get all triangles containing the target vertex
    target_triangles = []
    for tri_idx, face in enumerate(mesh.faces):
        if target_vertex_idx in [int(face[0]), int(face[1]), int(face[2])]:
            target_triangles.append(tri_idx)
    
    # Extract edges from these triangles
    edges_at_vertex = set()
    for tri_idx in target_triangles:
        face = mesh.faces[tri_idx]
        verts = [int(face[0]), int(face[1]), int(face[2])]
        for i in range(3):
            edge = tuple(sorted([verts[i], verts[(i+1) % 3]]))
            if target_vertex_idx in edge:
                edges_at_vertex.add(edge)
    
    print(f"  Total edges at vertex: {len(edges_at_vertex)}")
    
    # Check which edges have VPs
    for edge in sorted(edges_at_vertex):
        vp_on_edge = None
        for other_vp_idx, other_vp in enumerate(partition.variable_points):
            if tuple(sorted(other_vp.edge)) == edge:
                vp_on_edge = other_vp_idx
                break
        
        if vp_on_edge is not None:
            other_vp = partition.variable_points[vp_on_edge]
            dist = np.linalg.norm(other_vp.evaluate(mesh.vertices) - target_vertex_pos)
            marker = ""
            if vp_on_edge == vp_idx:
                marker = "← MIGRATING VP"
            elif vp_on_edge in neighbor_vps:
                marker = "← NEIGHBOR VP"
            print(f"    Edge {edge}: VP {vp_on_edge} (dist={dist:.6f}) {marker}")
        else:
            print(f"    Edge {edge}: no VP")
    
    print("\n" + "="*80)
    
    return {
        'target_vertex': target_vertex_idx,
        'target_pos': target_vertex_pos,
        'all_share_target': all_share_target,
        'all_close': all_close,
        'neighbor_info': neighbor_convergence_info
    }


def debug_all_type1_candidates_convergence(mesh, partition, boundary_tol=0.1, verbose=False):
    """
    DEBUG SECTION 8: Test ALL Type 1 migration candidates for vertex convergence.
    
    Analyzes CONNECTED COMPONENTS of boundary VPs (not individual VPs).
    Each component represents a group of VPs that are converging together.
    Tests if all VPs in each component share a common mesh vertex.
    """
    if not verbose:
        return {}
    
    print("\n" + "="*80)
    print("DEBUG SECTION 8: CONNECTED COMPONENT VERTEX CONVERGENCE ANALYSIS")
    print("="*80)
    
    # Get all boundary VPs (Type 1 candidates)
    boundary_vps = partition.get_boundary_variable_points(tol=boundary_tol)
    boundary_vps_set = set(boundary_vps)
    
    print(f"\n  Boundary tolerance: {boundary_tol}")
    print(f"  Total boundary VPs: {len(boundary_vps)}")
    
    # Find connected components (same logic as filter_connected_boundary_vps)
    from collections import defaultdict
    
    adjacency = defaultdict(set)
    for seg in partition.boundary_segments:
        vp1, vp2 = seg.vp_idx_1, seg.vp_idx_2
        if vp1 in boundary_vps_set and vp2 in boundary_vps_set:
            adjacency[vp1].add(vp2)
            adjacency[vp2].add(vp1)
    
    visited = set()
    components = []
    
    for vp_idx in boundary_vps:
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
    
    print(f"  Connected components found: {len(components)}")
    
    # Analyze each component
    print(f"\n━━━ ANALYZING EACH CONNECTED COMPONENT ━━━\n")
    
    # Statistics
    total_components = len(components)
    all_share_vertex = 0
    partially_share = 0
    no_common_vertex = 0
    
    component_details = []
    
    for comp_idx, component in enumerate(components):
        comp_vps = list(component)
        n_vps = len(comp_vps)
        
        # Get all edges in this component
        all_edges = []
        all_vertices = set()
        for vp_idx in comp_vps:
            vp = partition.variable_points[vp_idx]
            edge = tuple(sorted(vp.edge))
            all_edges.append(edge)
            all_vertices.update(vp.edge)
        
        # Find common vertex shared by ALL VPs in component
        common_vertex = None
        for v in all_vertices:
            if all(v in edge for edge in all_edges):
                common_vertex = v
                break
        
        # Calculate positions and distances
        positions = []
        for vp_idx in comp_vps:
            vp = partition.variable_points[vp_idx]
            positions.append(vp.evaluate(mesh.vertices))
        
        # Max pairwise distance
        max_dist = 0.0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                max_dist = max(max_dist, d)
        
        # Centroid of component
        centroid = np.mean(positions, axis=0) if positions else None
        
        # Distance to common vertex (if found)
        if common_vertex is not None:
            vertex_pos = mesh.vertices[common_vertex]
            dist_to_vertex = np.linalg.norm(centroid - vertex_pos) if centroid is not None else None
        else:
            dist_to_vertex = None
        
        # Categorize
        if common_vertex is not None:
            all_share_vertex += 1
            pattern = "ALL_SHARE_VERTEX"
        else:
            # Check how many vertices are shared by multiple edges
            vertex_counts = defaultdict(int)
            for edge in all_edges:
                for v in edge:
                    vertex_counts[v] += 1
            
            max_shared = max(vertex_counts.values()) if vertex_counts else 0
            if max_shared >= n_vps - 1:
                partially_share += 1
                pattern = "PARTIALLY_SHARE"
            else:
                no_common_vertex += 1
                pattern = "NO_COMMON"
        
        component_details.append({
            'component_idx': comp_idx + 1,
            'n_vps': n_vps,
            'vp_indices': comp_vps,
            'edges': all_edges,
            'common_vertex': common_vertex,
            'pattern': pattern,
            'max_pairwise_dist': max_dist,
            'dist_to_common_vertex': dist_to_vertex
        })
    
    # Print statistics
    print(f"━━━ CONVERGENCE STATISTICS ━━━\n")
    print(f"  Total connected components: {total_components}")
    print(f"  ")
    print(f"  VERTEX SHARING PATTERNS:")
    if total_components > 0:
        print(f"    All VPs share common vertex:  {all_share_vertex:3d} ({100*all_share_vertex/total_components:5.1f}%)")
        print(f"    Partially share vertex:       {partially_share:3d} ({100*partially_share/total_components:5.1f}%)")
        print(f"    No common vertex:             {no_common_vertex:3d} ({100*no_common_vertex/total_components:5.1f}%)")
    
    # Spatial clustering analysis
    spatially_close = sum(1 for d in component_details if d['max_pairwise_dist'] < 0.001)
    print(f"\n  SPATIAL CLUSTERING (max pairwise distance < 0.001):")
    if total_components > 0:
        print(f"    Spatially clustered:          {spatially_close:3d} ({100*spatially_close/total_components:5.1f}%)")
    
    # Component size distribution
    size_1 = sum(1 for d in component_details if d['n_vps'] == 1)
    size_2 = sum(1 for d in component_details if d['n_vps'] == 2)
    size_3 = sum(1 for d in component_details if d['n_vps'] == 3)
    size_4_plus = [d for d in component_details if d['n_vps'] >= 4]
    print(f"\n  COMPONENT SIZE DISTRIBUTION:")
    print(f"    1 VP:  {size_1:3d}")
    print(f"    2 VPs: {size_2:3d}")
    print(f"    3 VPs: {size_3:3d}")
    print(f"    4+ VPs: {len(size_4_plus):3d}")
    
    # Show details of unusual sizes (1 VP or 4+ VPs)
    if size_1 > 0:
        single_vp_components = [d for d in component_details if d['n_vps'] == 1]
        print(f"\n  SINGLE VP COMPONENTS (isolated boundary VPs):")
        for d in single_vp_components:
            print(f"    Component {d['component_idx']}: VP {d['vp_indices']}, edge {d['edges']}")
    
    if size_4_plus:
        print(f"\n  LARGE COMPONENTS (4+ VPs):")
        for d in size_4_plus:
            print(f"    Component {d['component_idx']}: {d['n_vps']} VPs {d['vp_indices']}")
            print(f"      Edges: {d['edges']}")
            print(f"      Common vertex: {d['common_vertex']}")
            print(f"      Max pairwise distance: {d['max_pairwise_dist']:.6f}")
    
    # Show examples
    print(f"\n━━━ EXAMPLE COMPONENTS ━━━")
    
    # Examples where all share vertex
    share_examples = [d for d in component_details if d['pattern'] == 'ALL_SHARE_VERTEX'][:5]
    if share_examples:
        print(f"\n  ALL VPs SHARE COMMON VERTEX (first {len(share_examples)}):")
        for ex in share_examples:
            print(f"    Component {ex['component_idx']}: {ex['n_vps']} VPs {ex['vp_indices']}")
            print(f"      Edges: {ex['edges']}")
            print(f"      Common vertex: {ex['common_vertex']}")
            print(f"      Max pairwise distance: {ex['max_pairwise_dist']:.6f}")
            print(f"      Distance to common vertex: {ex['dist_to_common_vertex']:.6f}")
    
    # Examples where they don't all share
    non_share_examples = [d for d in component_details if d['pattern'] != 'ALL_SHARE_VERTEX'][:5]
    if non_share_examples:
        print(f"\n  DO NOT ALL SHARE COMMON VERTEX (first {len(non_share_examples)}):")
        for ex in non_share_examples:
            print(f"    Component {ex['component_idx']}: {ex['n_vps']} VPs {ex['vp_indices']}")
            print(f"      Edges: {ex['edges']}")
            print(f"      Pattern: {ex['pattern']}")
            print(f"      Max pairwise distance: {ex['max_pairwise_dist']:.6f}")
    
    # Final conclusion
    print(f"\n━━━ CONCLUSION ━━━")
    if total_components > 0:
        share_pct = 100 * all_share_vertex / total_components
        cluster_pct = 100 * spatially_close / total_components
        
        if share_pct > 95:
            print(f"\n  ★★★ HYPOTHESIS STRONGLY CONFIRMED ★★★")
            print(f"  {share_pct:.1f}% of connected components have ALL VPs sharing a common vertex.")
            print(f"  {cluster_pct:.1f}% are spatially clustered (within 0.001).")
            print(f"  ")
            print(f"  IMPLICATION: Type 1 migration candidates are groups of 2-3 VPs")
            print(f"  all converging to the SAME mesh vertex. A simpler 'vertex collapse'")
            print(f"  operation could replace the complex edge-hopping migration.")
        elif share_pct > 80:
            print(f"\n  ★★ HYPOTHESIS MOSTLY CONFIRMED ★★")
            print(f"  {share_pct:.1f}% of components share a common vertex.")
        elif share_pct > 50:
            print(f"\n  ★ HYPOTHESIS PARTIALLY CONFIRMED ★")
            print(f"  {share_pct:.1f}% of components share a common vertex.")
        else:
            print(f"\n  ✗ HYPOTHESIS NOT CONFIRMED")
            print(f"  Only {share_pct:.1f}% of components share a common vertex.")
    
    print("\n" + "="*80)
    
    return {
        'total_components': total_components,
        'all_share_vertex': all_share_vertex,
        'partially_share': partially_share,
        'no_common_vertex': no_common_vertex,
        'details': component_details
    }


def debug_orphaned_triangles(mesh, partition, area_calc, target_cell_idx, verbose=False):
    """
    DEBUG SECTION 2: Identify triangles that are not claimed by ANY cell.
    
    These "orphaned" triangles would appear as white gaps in visualization.
    """
    if not verbose:
        # Still check for orphaned triangles but don't print details
        n_triangles = len(mesh.faces)
        n_cells = partition.n_cells
        
        # Track which triangles are claimed by each cell
        all_claimed = set()
        for cell_idx in range(n_cells):
            interior = set(area_calc.cell_interior_triangles.get(cell_idx, []))
            boundary = set(area_calc.cell_boundary_triangles.get(cell_idx, []))
            all_claimed |= (interior | boundary)
        
        # Find orphaned triangles
        all_triangles = set(range(n_triangles))
        orphaned = list(all_triangles - all_claimed)
        
        return {'orphaned': len(orphaned) > 0, 'count': len(orphaned), 'orphaned_centroid': None}
    
    print("\n" + "="*80)
    print("DEBUG SECTION 2: ORPHANED TRIANGLE ANALYSIS (AFTER MIGRATION)")
    print("="*80)
    
    n_triangles = len(mesh.faces)
    n_cells = partition.n_cells
    
    # Track which triangles are claimed by each cell
    claimed_by_cell = {cell_idx: set() for cell_idx in range(n_cells)}
    
    for cell_idx in range(n_cells):
        interior = set(area_calc.cell_interior_triangles.get(cell_idx, []))
        boundary = set(area_calc.cell_boundary_triangles.get(cell_idx, []))
        claimed_by_cell[cell_idx] = interior | boundary
    
    # Find triangles claimed by at least one cell
    all_claimed = set()
    for cell_idx in range(n_cells):
        all_claimed |= claimed_by_cell[cell_idx]
    
    # Find orphaned triangles
    all_triangles = set(range(n_triangles))
    orphaned = all_triangles - all_claimed
    
    print(f"\n━━━ TRIANGLE COVERAGE SUMMARY ━━━")
    print(f"  Total mesh triangles: {n_triangles:,}")
    print(f"  Triangles claimed by at least one cell: {len(all_claimed):,}")
    print(f"  ORPHANED triangles (not claimed by any cell): {len(orphaned):,}")
    
    if len(orphaned) == 0:
        print(f"\n  ✓ No orphaned triangles! All triangles are categorized.")
        print("="*80)
        return {'orphaned': [], 'orphaned_details': []}
    
    print(f"\n  ⚠ WARNING: {len(orphaned)} orphaned triangles detected!")
    print(f"  These will appear as WHITE GAPS in visualization.")
    
    # Analyze orphaned triangles
    print(f"\n━━━ ORPHANED TRIANGLE DETAILS ━━━")
    
    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
    orphaned_details = []
    
    # Build quick lookup for triangle_segments
    tri_to_vps = {}
    for ts in partition.triangle_segments:
        tri_to_vps[ts.triangle_idx] = ts.var_point_indices
    
    # Show details for first 20 orphaned triangles
    show_count = min(20, len(orphaned))
    print(f"\n  Showing details for first {show_count} orphaned triangles:\n")
    
    for i, tri_idx in enumerate(sorted(orphaned)[:show_count]):
        face = mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
        
        # Check if in crossing cache
        in_cache = tri_idx in partition.segment_crossing_cache
        
        # Check VPs on edges
        vps_on_tri = tri_to_vps.get(tri_idx, [])
        
        # Determine reason for orphaned status
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            reason = "Single label but not in any cell's list (BUG?)"
        elif len(vps_on_tri) == 0 and not in_cache:
            reason = "Mixed labels + NO VPs + NOT in crossing cache → ORPHANED"
        else:
            reason = "Unknown reason (needs investigation)"
        
        print(f"  Triangle {tri_idx}:")
        print(f"    Vertices: {v1}, {v2}, {v3}")
        print(f"    Vertex labels: {labels} ({'mixed' if len(unique_labels) > 1 else 'uniform'})")
        print(f"    VPs on edges: {vps_on_tri if vps_on_tri else 'None'}")
        print(f"    In crossing cache: {in_cache}")
        print(f"    → REASON: {reason}")
        print()
        
        orphaned_details.append({
            'tri_idx': tri_idx,
            'vertices': (v1, v2, v3),
            'labels': labels,
            'vps': vps_on_tri,
            'in_cache': in_cache,
            'reason': reason
        })
    
    if len(orphaned) > show_count:
        print(f"  ... and {len(orphaned) - show_count} more orphaned triangles")
    
    # Compute centroid of orphaned triangles for spatial analysis
    print(f"\n━━━ SPATIAL ANALYSIS OF ORPHANED TRIANGLES ━━━")
    
    orphaned_centroids_list = []
    for tri_idx in orphaned:
        face = mesh.faces[tri_idx]
        centroid = np.mean(mesh.vertices[[int(face[0]), int(face[1]), int(face[2])]], axis=0)
        orphaned_centroids_list.append(centroid)
    
    overall_centroid = None
    if len(orphaned_centroids_list) > 0:
        orphaned_centroids = np.array(orphaned_centroids_list)
        overall_centroid = np.mean(orphaned_centroids, axis=0)
        print(f"  Centroid of orphaned region: {overall_centroid}")
        
        # Check distance spread
        distances = np.linalg.norm(orphaned_centroids - overall_centroid, axis=1)
        print(f"  Distance spread from centroid: min={distances.min():.6f}, max={distances.max():.6f}, mean={distances.mean():.6f}")
    else:
        print(f"  No orphaned triangles to analyze spatially.")
    
    print("\n" + "="*80)
    
    return {
        'orphaned': list(orphaned),
        'orphaned_details': orphaned_details,
        'orphaned_centroid': overall_centroid
    }


def debug_categorization_comparison(mesh, partition, area_calc_before, area_calc_after, target_cell_idx, verbose=False):
    """
    DEBUG SECTION 1: Compare triangle categorization BEFORE vs AFTER migration.
    
    Identifies triangles that changed status (lost/gained boundary status).
    """
    if not verbose:
        return
    
    print("\n" + "="*80)
    print("DEBUG SECTION 1: TRIANGLE CATEGORIZATION COMPARISON")
    print("="*80)
    
    n_cells = partition.n_cells
    
    # Get before/after sets for each cell
    print(f"\n━━━ CATEGORIZATION COUNTS ━━━\n")
    print(f"  {'Cell':<6} {'Interior BEFORE':>16} {'Interior AFTER':>15} {'Boundary BEFORE':>16} {'Boundary AFTER':>15}")
    print(f"  {'-'*6} {'-'*16} {'-'*15} {'-'*16} {'-'*15}")
    
    changes = {}
    
    for cell_idx in range(n_cells):
        int_before = set(area_calc_before.cell_interior_triangles.get(cell_idx, []))
        int_after = set(area_calc_after.cell_interior_triangles.get(cell_idx, []))
        bnd_before = set(area_calc_before.cell_boundary_triangles.get(cell_idx, []))
        bnd_after = set(area_calc_after.cell_boundary_triangles.get(cell_idx, []))
        
        print(f"  {cell_idx:<6} {len(int_before):>16,} {len(int_after):>15,} {len(bnd_before):>16,} {len(bnd_after):>15,}")
        
        # Track changes
        all_before = int_before | bnd_before
        all_after = int_after | bnd_after
        
        lost = all_before - all_after
        gained = all_after - all_before
        
        changes[cell_idx] = {
            'lost': lost,
            'gained': gained,
            'int_before': int_before,
            'int_after': int_after,
            'bnd_before': bnd_before,
            'bnd_after': bnd_after
        }
    
    # Focus on target cell
    print(f"\n━━━ CHANGES FOR TARGET CELL {target_cell_idx} ━━━")
    
    target_changes = changes[target_cell_idx]
    lost = target_changes['lost']
    gained = target_changes['gained']
    
    print(f"\n  Triangles LOST (were in cell {target_cell_idx}, now not): {len(lost)}")
    print(f"  Triangles GAINED (were not in cell {target_cell_idx}, now are): {len(gained)}")
    
    # Get vertex labels for triangle info
    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
    
    if lost:
        print(f"\n  LOST triangles (first 10):")
        
        for tri_idx in sorted(lost)[:10]:
            face = mesh.faces[tri_idx]
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
            
            was_interior = tri_idx in target_changes['int_before']
            was_boundary = tri_idx in target_changes['bnd_before']
            
            print(f"    Triangle {tri_idx}: labels={labels}, was_interior={was_interior}, was_boundary={was_boundary}")
    
    if gained:
        print(f"\n  GAINED triangles (first 10):")
        for tri_idx in sorted(gained)[:10]:
            face = mesh.faces[tri_idx]
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
            
            is_interior = tri_idx in target_changes['int_after']
            is_boundary = tri_idx in target_changes['bnd_after']
            
            print(f"    Triangle {tri_idx}: labels={labels}, is_interior={is_interior}, is_boundary={is_boundary}")
    
    # Overall mesh coverage
    print(f"\n━━━ OVERALL MESH COVERAGE ━━━")
    
    all_before_total = set()
    all_after_total = set()
    for cell_idx in range(n_cells):
        all_before_total |= changes[cell_idx]['int_before'] | changes[cell_idx]['bnd_before']
        all_after_total |= changes[cell_idx]['int_after'] | changes[cell_idx]['bnd_after']
    
    n_triangles = len(mesh.faces)
    print(f"  Total mesh triangles: {n_triangles:,}")
    print(f"  Covered BEFORE migration: {len(all_before_total):,}")
    print(f"  Covered AFTER migration: {len(all_after_total):,}")
    print(f"  LOST coverage: {len(all_before_total - all_after_total):,}")
    print(f"  GAINED coverage: {len(all_after_total - all_before_total):,}")
    
    print("\n" + "="*80)
    
    return changes


def debug_migrated_vp(mesh, partition, mesh_topology, vp_idx, target_cell_idx, verbose=False):
    """
    Debug function to analyze what's happening around a migrated VP.
    
    Prints detailed information about crossing cache, triangles, and polygon computation.
    """
    if not verbose:
        return
    print("\n" + "="*80)
    print("DEBUG: ANALYZING MIGRATED VP AND SURROUNDING TRIANGULAR")
    print("="*80)
    
    vp = partition.variable_points[vp_idx]
    vp_pos = vp.evaluate(mesh.vertices)
    new_edge = tuple(sorted(vp.edge))
    
    print(f"\n1. MIGRATED VP INFO:")
    print(f"   VP index: {vp_idx}")
    print(f"   New edge: {new_edge}")
    print(f"   Position: {vp_pos}")
    print(f"   Belongs to cells: {vp.belongs_to_cells}")
    
    # Find triangles containing the VP
    triangles_with_vp = mesh_topology.get_triangles_sharing_edge(new_edge)
    print(f"\n2. TRIANGLES CONTAINING MIGRATED VP:")
    print(f"   Triangles: {triangles_with_vp}")
    
    for tri_idx in triangles_with_vp:
        face = mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        vertex_labels = np.argmax(partition.indicator_functions, axis=1)
        labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
        print(f"\n   Triangle {tri_idx}:")
        print(f"     Vertices: {v1}, {v2}, {v3}")
        print(f"     Vertex labels (indicator_functions): {labels}")
        
        # Check if in crossing cache
        if tri_idx in partition.segment_crossing_cache:
            print(f"     ✓ IN crossing cache:")
            for crossing in partition.segment_crossing_cache[tri_idx]:
                print(f"       Segment: {crossing.segment}")
                print(f"       Cell pair: {crossing.cell_pair}")
                print(f"       Entry: {crossing.entry_point} on edge {crossing.entry_edge}")
                print(f"       Exit: {crossing.exit_point} on edge {crossing.exit_edge}")
        else:
            print(f"     ✗ NOT in crossing cache")
        
        # Check triangle_segments
        tri_seg = None
        for ts in partition.triangle_segments:
            if ts.triangle_idx == tri_idx:
                tri_seg = ts
                break
        if tri_seg:
            print(f"     Triangle_segments VP indices: {tri_seg.var_point_indices}")
        else:
            print(f"     NOT in triangle_segments")
    
    # Find segments connected to this VP
    print(f"\n3. BOUNDARY SEGMENTS CONNECTED TO VP {vp_idx}:")
    connected_segments = []
    neighbor_vp_indices = []
    for seg in partition.boundary_segments:
        if seg.vp_idx_1 == vp_idx or seg.vp_idx_2 == vp_idx:
            connected_segments.append(seg)
            other_idx = seg.vp_idx_2 if seg.vp_idx_1 == vp_idx else seg.vp_idx_1
            neighbor_vp_indices.append(other_idx)
            other_vp = partition.variable_points[other_idx]
            other_pos = other_vp.evaluate(mesh.vertices)
            print(f"   Segment ({seg.vp_idx_1}, {seg.vp_idx_2}):")
            print(f"     Type: {seg.segment_type}")
            print(f"     Cell pair: {seg.cell_pair}")
            print(f"     Crossing triangles: {seg.crossing_triangles}")
            print(f"     Neighbor VP {other_idx} position: {other_pos}")
            print(f"     Neighbor VP {other_idx} edge: {other_vp.edge}")
    
    # Compute segment directions and verify if they pass through vertices
    print(f"\n3b. SEGMENT GEOMETRY ANALYSIS:")
    for neighbor_idx in neighbor_vp_indices:
        neighbor_vp = partition.variable_points[neighbor_idx]
        neighbor_pos = neighbor_vp.evaluate(mesh.vertices)
        
        # Direction from VP 1688 to neighbor
        direction = neighbor_pos - vp_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-12:
            direction_unit = direction / direction_norm
        else:
            direction_unit = np.zeros(3)
        
        print(f"\n   Segment VP {vp_idx} → VP {neighbor_idx}:")
        print(f"     VP {vp_idx} pos: {vp_pos}")
        print(f"     VP {neighbor_idx} pos: {neighbor_pos}")
        print(f"     Direction (unit): {direction_unit}")
        print(f"     Segment length: {direction_norm:.6f}")
        
        # Check if segment passes near any triangle vertices
        for tri_idx in triangles_with_vp:
            face = mesh.faces[tri_idx]
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            
            print(f"\n     Triangle {tri_idx} vertex distances from segment line:")
            for v, v_name in [(v1, "v1"), (v2, "v2"), (v3, "v3")]:
                v_pos = mesh.vertices[v]
                
                # Distance from vertex to line (VP_pos, VP_pos + direction)
                # Using cross product formula: |AP x direction| / |direction|
                ap = v_pos - vp_pos
                cross = np.cross(ap, direction_unit)
                dist_to_line = np.linalg.norm(cross)
                
                # Also compute parameter t where closest point is on segment
                # t = dot(ap, direction_unit)
                t = np.dot(ap, direction_unit)
                
                print(f"       {v_name} (idx={v}): dist_to_line = {dist_to_line:.8f}, t = {t:.6f}")
                if dist_to_line < 1e-5:
                    print(f"         ⚠ VERY CLOSE TO SEGMENT LINE!")
                    if 0 <= t <= direction_norm:
                        print(f"         ⚠ AND WITHIN SEGMENT BOUNDS! Segment passes through/near this vertex!")
    
    # Print full crossing cache
    print(f"\n4. FULL CROSSING CACHE ({len(partition.segment_crossing_cache)} triangles):")
    for tri_idx, crossings in partition.segment_crossing_cache.items():
        for crossing in crossings:
            print(f"   Triangle {tri_idx}: segment {crossing.segment}, cells {crossing.cell_pair}")
    
    # Test compute_cell_portion_in_triangle for triangles around VP
    print(f"\n5. TESTING compute_cell_portion_in_triangle FOR CELL {target_cell_idx}:")
    
    # Build tri_idx_to_segment for fast lookup
    tri_idx_to_segment = {}
    for ts in partition.triangle_segments:
        tri_idx_to_segment[ts.triangle_idx] = ts
    
    for tri_idx in triangles_with_vp:
        print(f"\n   Triangle {tri_idx}:")
        
        # Check crossing first
        crossing = _find_crossing_for_cell_viz(partition, tri_idx, target_cell_idx)
        if crossing:
            print(f"     _find_crossing_for_cell_viz: FOUND")
            print(f"       Entry: {crossing.entry_point}")
            print(f"       Exit: {crossing.exit_point}")
            
            # Test side determination
            face = mesh.faces[tri_idx]
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            print(f"     Side determination for each vertex:")
            for v, v_name in [(v1, "v1"), (v2, "v2"), (v3, "v3")]:
                v_pos = mesh.vertices[v]
                is_inside = _vertex_on_cell_side_for_viz(
                    v_pos, crossing.entry_point, crossing.exit_point, 
                    target_cell_idx, mesh, tri_idx, partition
                )
                vertex_labels = np.argmax(partition.indicator_functions, axis=1)
                print(f"       {v_name} (idx={v}, label={vertex_labels[v]}): on_cell_side = {is_inside}")
        else:
            print(f"     _find_crossing_for_cell_viz: NOT FOUND")
        
        # Call compute_cell_portion_in_triangle
        result = compute_cell_portion_in_triangle(mesh, partition, tri_idx, target_cell_idx, tri_idx_to_segment)
        if result is None:
            print(f"     compute_cell_portion_in_triangle: RETURNS None!")
        else:
            print(f"     compute_cell_portion_in_triangle: {len(result)} points")
    
    # Also check intermediate triangles from crossing cache
    print(f"\n6. TESTING INTERMEDIATE TRIANGLES FROM CROSSING CACHE:")
    tested = set(triangles_with_vp)
    for tri_idx in partition.segment_crossing_cache.keys():
        if tri_idx not in tested:
            # Check if this crossing involves the target cell
            for crossing in partition.segment_crossing_cache[tri_idx]:
                if crossing.involves_cell(target_cell_idx):
                    tested.add(tri_idx)
                    print(f"\n   Triangle {tri_idx} (intermediate):")
                    print(f"     Crossing: segment {crossing.segment}, cells {crossing.cell_pair}")
                    
                    result = compute_cell_portion_in_triangle(mesh, partition, tri_idx, target_cell_idx, tri_idx_to_segment)
                    if result is None:
                        print(f"     compute_cell_portion_in_triangle: RETURNS None!")
                    else:
                        print(f"     compute_cell_portion_in_triangle: {len(result)} points")
                    break
    
    # CRITICAL DEBUG: Check if crossing cache triangles are in boundary_triangles
    from src.core.area_calculator import AreaCalculator
    area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
    
    print(f"\n7. CHECKING IF CROSSING CACHE TRIANGLES ARE IN BOUNDARY LIST:")
    boundary_tris_for_cell = area_calc.cell_boundary_triangles.get(target_cell_idx, [])
    interior_tris_for_cell = area_calc.cell_interior_triangles.get(target_cell_idx, [])
    print(f"   Cell {target_cell_idx} has {len(boundary_tris_for_cell)} boundary triangles")
    print(f"   Cell {target_cell_idx} has {len(interior_tris_for_cell)} interior triangles")
    
    all_crossing_tris = list(partition.segment_crossing_cache.keys())
    for tri_idx in all_crossing_tris:
        in_boundary = tri_idx in boundary_tris_for_cell
        in_interior = tri_idx in interior_tris_for_cell
        print(f"   Triangle {tri_idx}: in_boundary={in_boundary}, in_interior={in_interior}")
        
        if not in_boundary and not in_interior:
            print(f"     ⚠ NOT IN ANY LIST FOR CELL {target_cell_idx}!")
            # Check indicator_functions
            face = mesh.faces[tri_idx]
            vertex_labels = np.argmax(partition.indicator_functions, axis=1)
            labels = [vertex_labels[int(face[0])], vertex_labels[int(face[1])], vertex_labels[int(face[2])]]
            print(f"     Vertex labels: {labels}")
    
    # Also check for BOTH cells in cell_pair
    print(f"\n8. CHECKING CATEGORIZATION FOR BOTH CELLS IN CROSSING:")
    for tri_idx, crossings in partition.segment_crossing_cache.items():
        for crossing in crossings:
            cell_a, cell_b = crossing.cell_pair
            boundary_a = area_calc.cell_boundary_triangles.get(cell_a, [])
            boundary_b = area_calc.cell_boundary_triangles.get(cell_b, [])
            in_a = tri_idx in boundary_a
            in_b = tri_idx in boundary_b
            print(f"   Triangle {tri_idx} (cells {cell_a}, {cell_b}): in_cell_{cell_a}_boundary={in_a}, in_cell_{cell_b}_boundary={in_b}")
    
    # NEW DEBUG: Print actual polygon coordinates for BOTH cells
    print(f"\n9. ACTUAL POLYGON COORDINATES FOR BOTH CELLS:")
    for tri_idx in triangles_with_vp:
        print(f"\n   Triangle {tri_idx}:")
        
        # Get triangle vertices for reference
        face = mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        print(f"     Mesh vertices:")
        print(f"       v1 ({v1}): {mesh.vertices[v1]}")
        print(f"       v2 ({v2}): {mesh.vertices[v2]}")
        print(f"       v3 ({v3}): {mesh.vertices[v3]}")
        
        # Get crossing info
        crossing = _find_crossing_for_cell_viz(partition, tri_idx, target_cell_idx)
        if crossing:
            print(f"     Crossing entry: {crossing.entry_point}")
            print(f"     Crossing exit: {crossing.exit_point}")
        
        # Compute polygon for BOTH cells in the cell_pair
        for cell_idx in crossing.cell_pair if crossing else [target_cell_idx]:
            result = compute_cell_portion_in_triangle(mesh, partition, tri_idx, cell_idx, tri_idx_to_segment)
            print(f"\n     Cell {cell_idx} polygon:")
            if result is None:
                print(f"       RETURNS None!")
            else:
                print(f"       {len(result)} points:")
                for i, pt in enumerate(result):
                    print(f"         [{i}]: {pt}")
                    
                # Check if polygon covers full triangle or partial
                tri_area = 0.5 * np.linalg.norm(np.cross(
                    mesh.vertices[v2] - mesh.vertices[v1],
                    mesh.vertices[v3] - mesh.vertices[v1]
                ))
                # Compute polygon area (approximate for non-planar)
                if len(result) >= 3:
                    poly_area = 0.0
                    centroid = np.mean(result, axis=0)
                    for i in range(len(result)):
                        p1 = result[i]
                        p2 = result[(i+1) % len(result)]
                        poly_area += 0.5 * np.linalg.norm(np.cross(p1 - centroid, p2 - centroid))
                    print(f"       Triangle area: {tri_area:.6f}")
                    print(f"       Polygon area: {poly_area:.6f}")
                    print(f"       Ratio: {poly_area/tri_area:.2%}")
    
    # Check what the OTHER cell (cell 1) gets for these triangles
    other_cell = 1 if target_cell_idx == 2 else 2
    print(f"\n10. SANITY CHECK - CELL {other_cell} POLYGONS:")
    for tri_idx in triangles_with_vp:
        result = compute_cell_portion_in_triangle(mesh, partition, tri_idx, other_cell, tri_idx_to_segment)
        if result is None:
            print(f"   Triangle {tri_idx}: Cell {other_cell} gets None")
        else:
            print(f"   Triangle {tri_idx}: Cell {other_cell} gets {len(result)} points")
            # Compute area ratio
            face = mesh.faces[tri_idx]
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            tri_area = 0.5 * np.linalg.norm(np.cross(
                mesh.vertices[v2] - mesh.vertices[v1],
                mesh.vertices[v3] - mesh.vertices[v1]
            ))
            if len(result) >= 3:
                poly_area = 0.0
                centroid = np.mean(result, axis=0)
                for i in range(len(result)):
                    p1 = result[i]
                    p2 = result[(i+1) % len(result)]
                    poly_area += 0.5 * np.linalg.norm(np.cross(p1 - centroid, p2 - centroid))
                print(f"     Area ratio: {poly_area/tri_area:.2%} of triangle")
                
                # The TWO polygons should sum to 100%
                result_target = compute_cell_portion_in_triangle(mesh, partition, tri_idx, target_cell_idx, tri_idx_to_segment)
                if result_target is not None and len(result_target) >= 3:
                    poly_area_target = 0.0
                    centroid_t = np.mean(result_target, axis=0)
                    for i in range(len(result_target)):
                        p1 = result_target[i]
                        p2 = result_target[(i+1) % len(result_target)]
                        poly_area_target += 0.5 * np.linalg.norm(np.cross(p1 - centroid_t, p2 - centroid_t))
                    combined = (poly_area + poly_area_target) / tri_area
                    print(f"     Combined ratio (cell {other_cell} + cell {target_cell_idx}): {combined:.2%}")
                    if abs(combined - 1.0) > 0.01:
                        print(f"     ⚠ WARNING: Combined != 100%! Gap or overlap detected!")
    
    print("\n" + "="*80)
    print("END DEBUG")
    print("="*80 + "\n")


def compute_cell_portion_in_triangle(
    mesh: TriMesh,
    partition: PartitionContour,
    tri_idx: int,
    cell_idx: int,
    tri_idx_to_segment: Optional[Dict[int, any]] = None
) -> Optional[np.ndarray]:
    """
    Compute vertices of the polygon representing cell_idx's portion in tri_idx.
    
    PRIORITY ORDER (same as area_calculator):
    1. Check segment_crossing_cache FIRST - use entry_point/exit_point directly
    2. Fall back to standard VP/indicator_functions logic
    
    Args:
        tri_idx_to_segment: Pre-indexed dict mapping triangle_idx to TriangleSegment (for speed)
    
    Returns:
        vertices: (N, 3) array where N is number of polygon vertices, or None if
                  triangle doesn't contribute to this cell's boundary
    """
    # Get mesh vertices and labels
    face = mesh.faces[tri_idx]
    v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
    labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
    
    # PRIORITY 1: Check segment_crossing_cache FIRST
    # This handles ALL cross-triangle segments (intermediate + endpoint triangles)
    # The entry_point and exit_point are already computed by _compute_and_cache_crossings()
    crossing = _find_crossing_for_cell_viz(partition, tri_idx, cell_idx)
    
    if crossing is not None:
        # Use the pre-computed entry/exit points directly
        entry_point = crossing.entry_point
        exit_point = crossing.exit_point
        
        # Determine which mesh vertices are on cell_idx side
        vertices_inside = []
        for v in [v1, v2, v3]:
            v_pos = mesh.vertices[v]
            if _vertex_on_cell_side_for_viz(v_pos, entry_point, exit_point, cell_idx, 
                                            mesh, tri_idx, partition):
                vertices_inside.append(v_pos)
        
        # Boundary points are entry and exit (deduplicated)
        boundary_points = [entry_point]
        if np.linalg.norm(exit_point - entry_point) > 1e-8:
            boundary_points.append(exit_point)
        
        # Construct polygon
        all_points = vertices_inside + boundary_points
        if len(all_points) < 3:
            return None
        return _order_polygon_vertices(np.array(all_points), mesh, tri_idx)
    
    # PRIORITY 2: Standard logic - check if triangle is boundary via VPs
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


def _find_crossing_for_cell_viz(partition: PartitionContour, tri_idx: int, cell_idx: int):
    """
    Find crossing info for a specific cell in this triangle.
    
    Returns:
        SegmentCrossingInfo if found, None otherwise
    """
    if tri_idx not in partition.segment_crossing_cache:
        return None
    
    for crossing in partition.segment_crossing_cache[tri_idx]:
        if hasattr(crossing, 'involves_cell'):
            if crossing.involves_cell(cell_idx):
                return crossing
        elif crossing.cell_idx == cell_idx:
            return crossing
    
    return None


def _vertex_on_cell_side_for_viz(vertex_pos: np.ndarray, entry_point: np.ndarray,
                                  exit_point: np.ndarray, cell_idx: int,
                                  mesh: TriMesh, tri_idx: int, 
                                  partition: PartitionContour) -> bool:
    """
    Determine if a vertex is on the cell_idx side of the crossing segment.
    
    Uses cross product with triangle normal to determine signed side.
    IMPROVED: Tries multiple strategies to find a reference point, not just
    indicator_functions (which may be stale after topology switches).
    """
    # Vector from entry to exit
    segment_vec = exit_point - entry_point
    seg_len = np.linalg.norm(segment_vec)
    if seg_len < 1e-12:
        return False
    
    # Vector from entry to vertex
    to_vertex = vertex_pos - entry_point
    
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
    
    # STRATEGY 1: Use crossing cache to find cell_pair and determine orientation
    ref_vertex = None
    ref_is_cell_idx_side = True
    
    if tri_idx in partition.segment_crossing_cache:
        for crossing in partition.segment_crossing_cache[tri_idx]:
            if hasattr(crossing, 'cell_pair'):
                cell_pair = crossing.cell_pair
                if cell_idx in cell_pair:
                    # Find the other cell in pair
                    other_cell = cell_pair[0] if cell_pair[1] == cell_idx else cell_pair[1]
                    
                    # Look for a vertex labeled as other_cell (opposite side reference)
                    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
                    for v_idx in [int(face[0]), int(face[1]), int(face[2])]:
                        if vertex_labels[v_idx] == other_cell:
                            ref_vertex = mesh.vertices[v_idx]
                            ref_is_cell_idx_side = False  # Reference is on OPPOSITE side
                            break
                    
                    if ref_vertex is not None:
                        break
    
    # STRATEGY 2: Use VPs on this triangle's edges to find reference
    if ref_vertex is None:
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
    
    # STRATEGY 3: Fallback to indicator_functions on triangle vertices
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
    ref_to_vertex = ref_vertex - entry_point
    ref_cross = np.cross(segment_vec, ref_to_vertex)
    ref_signed = np.dot(ref_cross, normal)
    
    # Determine if vertex is on cell_idx side
    same_side_as_ref = (signed_area > 0) == (ref_signed > 0)
    
    if ref_is_cell_idx_side:
        return same_side_as_ref
    else:
        return not same_side_as_ref


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
) -> Optional[List[np.ndarray]]:
    """
    Compute the portion of a triple point triangle belonging to cell_idx.
    
    CRITICAL FIX: Now returns BOTH void interior AND corner region.
    
    Returns:
        List of polygons (numpy arrays) or None if not found:
        - [0]: Void interior quadrilateral (VP, neighbor1, steiner, neighbor2)
        - [1]: Corner triangle (mesh_vertex, VP positions)
        
    Previously only returned void interior, causing white gaps in visualization.
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
    
    # Get the two VPs that bound this cell (from triple point mapping)
    if cell_idx not in triple_point.cell_to_varpoint_pair:
        return None
    
    vp_idx1, vp_idx2 = triple_point.cell_to_varpoint_pair[cell_idx]
    vp_pos1 = partition.evaluate_variable_point(vp_idx1)
    vp_pos2 = partition.evaluate_variable_point(vp_idx2)
    
    # POLYGON 1: Void interior quadrilateral
    # The cell's portion of the void is bounded by the two VPs and Steiner point
    # Note: Ordering matters for correct winding
    void_quad = np.array([
        vp_pos1,
        vp_pos2,
        steiner_pos
    ])  # Triangle, not quad - let PyVista triangulate if needed
    
    # POLYGON 2: Corner triangle (CRITICAL FIX - was missing!)
    # Get the mesh vertex that belongs to this cell
    polygons = [void_quad]
    
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
    # CRITICAL FIX: Now handles multiple polygons per triple point (void + corner)
    if use_exact_boundaries:
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
        # NOTE: backface_culling disabled because it causes triple point polygons
        # to disappear (inconsistent normal directions in Steiner subdivisions)
        plotter.add_mesh(
            region_mesh, 
            color=color, 
            opacity=opacity, 
            show_edges=True,           # Show mesh edges on region
            edge_color='gray',         # Match reference scripts
            line_width=0.5             # Match reference scripts
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
    highlight_vp_indices: list = None,  # Only highlight specific VPs
    camera_focus: np.ndarray = None,   # Point to focus camera on
    camera_zoom: float = None,         # Zoom level (distance from focus point)
    current_edge: Tuple[int, int] = None,  # Current edge (before migration)
    target_edge: Tuple[int, int] = None,  # Target edge (after migration)
    verbose: bool = False,  # Control debug output
):
    """
    Render ALL regions with precise boundaries.
    
    Args:
        target_only: If True, only render target region (much faster, ~1 minute)
        highlight_vp_indices: List of VP indices to highlight (for migrations)
        camera_focus: Point to focus camera on (for zoom)
        camera_zoom: Zoom level (smaller = more zoomed in)
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
        if verbose:
            print(f"  Adding {n_vps} highlighted variable points (migration-related)...")
        
        # Color scheme:
        # - First VP (index 0): YELLOW = migrated VP
        # - Second VP (index 1): BLUE = neighbor 1
        # - Third VP (index 2): MAGENTA = neighbor 2
        # - Others: CYAN
        neighbor_colors = ['yellow', 'blue', 'magenta', 'cyan', 'orange', 'lime']
        if verbose and n_vps > 0:
            print(f"    VP color scheme:")
            for i, vp_i in enumerate(highlight_vp_indices[:3]):
                color_name = neighbor_colors[i] if i < len(neighbor_colors) else 'cyan'
                print(f"      VP {vp_i}: {color_name.upper()}")
        
        if n_vps > 100:
            iterator = list(enumerate(tqdm(highlight_vp_indices, desc="  Drawing VPs", leave=False)))
        else:
            iterator = list(enumerate(highlight_vp_indices))
        
        for i, vp_idx in iterator:
            try:
                vp = partition.variable_points[vp_idx]
                pos = vp.evaluate(mesh.vertices)
                sphere = pv.Sphere(radius=vp_size, center=pos)
                # Color by position in list (migrated VP first, then neighbors)
                if i < len(neighbor_colors):
                    vp_color = neighbor_colors[i]
                else:
                    vp_color = 'cyan'
                plotter.add_mesh(sphere, color=vp_color, opacity=0.9)
                if verbose:
                    print(f"      Added VP {vp_idx} at position {pos} with color {vp_color}")
            except (IndexError, KeyError) as e:
                if verbose:
                    print(f"      WARNING: Could not render VP {vp_idx}: {e}")
                continue
    
    # Add Steiner points if requested
    if show_steiner:
        n_tps = len(steiner_handler.triple_points)
        if verbose:
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
    
    # Visualize current and target edges for Type 1 migration
    if current_edge is not None or target_edge is not None:
        if verbose:
            print(f"  Adding edge visualization for migration...")
        
        # Current edge (before migration) - RED
        if current_edge is not None:
            v1_idx, v2_idx = current_edge
            v1_pos = mesh.vertices[v1_idx]
            v2_pos = mesh.vertices[v2_idx]
            current_line = pv.Line(v1_pos, v2_pos)
            plotter.add_mesh(current_line, color='red', line_width=5, opacity=0.9, 
                           label='Current Edge (before migration)')
            if verbose:
                print(f"    Current edge: {current_edge} (RED)")
        
        # Target edge (after migration) - GREEN
        if target_edge is not None:
            v1_idx, v2_idx = target_edge
            v1_pos = mesh.vertices[v1_idx]
            v2_pos = mesh.vertices[v2_idx]
            target_line = pv.Line(v1_pos, v2_pos)
            plotter.add_mesh(target_line, color='lime', line_width=5, opacity=0.9,
                           label='Target Edge (after migration)')
            if verbose:
                print(f"    Target edge: {target_edge} (GREEN)")
    
    # NOTE: Mesh edges are already added at the beginning (line 617-632) efficiently
    # Don't duplicate here with individual line segments - it's VERY slow!
    
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
    if args.verbose:
        print("\nLoading partition data...")
    try:
        mesh, partition = load_partition_from_refined_file(args.solution, verbose=args.verbose)
    except Exception as e:
        print(f"ERROR: Failed to load refined contours file")
        print(f"       {e}")
        import traceback
        traceback.print_exc()
        return
    
    if args.verbose:
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
            highlight_vp_indices=None,  # No migration, no VPs to highlight
            camera_focus=None,          # No zoom for current state
            camera_zoom=None,
            verbose=args.verbose
        )
        plotter.show()
        
        return
    
    # Type 1 or Type 2 switch
    if args.switch_type == 'type1':
        print("\nAnalyzing Type 1 migration...")
        
        # NEW: Component-based migration (vertex-collapse strategy)
        # Initialize variables that will be used later (for scope)
        migrating_vp_idx = None
        left_neighbor = None
        right_neighbor = None
        component_idx = None
        selected_component = None
        
        if args.use_vertex_collapse:
            print("Using vertex-collapse strategy (component-based migration)")
            
            # Initialize SteinerHandler (needed for visualization)
            steiner_handler = SteinerHandler(mesh, partition)
            
            # Step 1: Get boundary VPs (excluding triple point VPs)
            boundary_vps = switcher.get_non_triple_point_boundary_vps(boundary_tol=args.boundary_tol)
            boundary_vps_set = set(boundary_vps)
            
            if not boundary_vps:
                print("ERROR: No boundary VPs found for Type 1 migration")
                return
            
            # Step 2: Find connected components
            components = switcher.find_connected_components(boundary_vps_set)
            print(f"\n✓ Found {len(components)} connected component(s)")
            
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
            
            # Step 6: Show all components and let user select
            print("\n" + "="*80)
            print("AVAILABLE COMPONENTS FOR MIGRATION")
            print("="*80)
            print(f"{'Idx':<5} {'Size':<6} {'Dist':<10} {'Status':<15} {'VPs':<30}")
            print("-" * 80)
            
            for i, comp in enumerate(component_info):
                status = "TO MIGRATE" if comp in to_migrate else "DEFERRED" if comp in deferred else "UNKNOWN"
                vp_list = str(comp['vp_indices'][:5]) + ("..." if len(comp['vp_indices']) > 5 else "")
                print(f"{i:<5} {comp['size']:<6} {comp['min_distance']:<10.6f} {status:<15} {vp_list:<30}")
            
            if chain_warnings:
                print(f"\n⚠️  Found {len(chain_warnings)} component chain(s):")
                for warning in chain_warnings:
                    print(f"  {warning['warning']}")
            
            # Step 7: Select component to migrate
            component_idx = args.component_index if args.component_index is not None else 0
            if component_idx >= len(component_info):
                print(f"ERROR: Component index {component_idx} out of range (max: {len(component_info)-1})")
                return
            
            selected_component = component_info[component_idx]
            print(f"\n✓ Selected Component {component_idx} for migration:")
            print(f"  Size: {selected_component['size']} VPs")
            print(f"  VPs: {selected_component['vp_indices']}")
            print(f"  Target vertex: {selected_component['target_vertex']}")
            print(f"  Min distance: {selected_component['min_distance']:.6f}")
            print(f"  Status: {'TO MIGRATE' if selected_component in to_migrate else 'DEFERRED' if selected_component in deferred else 'UNKNOWN'}")
            
            # Find migrating VP (closest to target vertex)
            migrating_vp_idx = min(selected_component['vp_indices'],
                                  key=lambda vp: switcher.compute_boundary_distance(vp))
            migrating_vp = partition.variable_points[migrating_vp_idx]
            
            # Get neighbors
            left_neighbor, right_neighbor = switcher._get_two_neighbors(migrating_vp_idx)
            
            print(f"\n  Migrating VP: {migrating_vp_idx}")
            print(f"  Neighbors: {left_neighbor}, {right_neighbor}")
            print(f"  Current edge: {migrating_vp.edge}")
            print(f"  Lambda: {migrating_vp.lambda_param:.6f}")
            
            # Store for visualization
            vp_idx = migrating_vp_idx  # For compatibility with existing visualization code
            vp = migrating_vp
            highlight_vp_indices = [migrating_vp_idx, left_neighbor, right_neighbor]
            
        else:
            # OLD: VP-based migration (existing code)
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
            
            # Find index for VP 765
            if 765 in filtered_vps_sorted:
                idx_765 = filtered_vps_sorted.index(765)
                print(f"\nVP 765 is at index {idx_765} in filtered_vps_sorted")
            else:
                print(f"\nVP 765 is NOT in filtered_vps_sorted")
            
            # Select closest VP
            vp_idx = filtered_vps_sorted[43] #63 and 70 are the examples that display two VPs in the component. 68 Also
            vp = partition.variable_points[vp_idx]
            
            print(f"\nSelected VP {vp_idx}:")
            print(f"  Edge: {vp.edge}")
            print(f"  Lambda: {vp.lambda_param:.6f}")
            print(f"  Distance to vertex: {compute_boundary_distance(partition, vp_idx):.6f}")
            
            highlight_vp_indices = None  # Will be set later
        
        # DEBUG SECTION 7: Analyze VP convergence BEFORE migration (only for old method)
        if not args.use_vertex_collapse:
            convergence_info = debug_type1_convergence_analysis(
                mesh, partition, mesh_topology, vp_idx, boundary_tol=args.boundary_tol, verbose=args.verbose
            )

            # DEBUG SECTION 8: Test ALL Type 1 candidates for convergence pattern
            all_convergence_stats = debug_all_type1_candidates_convergence(
                mesh, partition, boundary_tol=args.boundary_tol, verbose=args.verbose
            )

            # DEBUG SECTION 9: Component Proximity Analysis (only if verbose)
            if args.verbose:
                from component_proximity_debug import debug_component_proximity_analysis
                proximity_results = debug_component_proximity_analysis(
                    mesh, partition, boundary_tol=args.boundary_tol,
                    filtered_vps_sorted=filtered_vps_sorted
                )
            else:
                proximity_results = {}
        else:
            # For vertex-collapse, we already have component info
            convergence_info = {}
            all_convergence_stats = {}
            proximity_results = {}
        
        # Show ALL filtered VPs with their component sizes (only if verbose)
        if args.verbose:
            print("\n" + "="*80)
            print("ALL VPs IN filtered_vps_sorted WITH COMPONENT SIZES")
            print("="*80)
        
        # Build map of VP -> component size from convergence stats
        vp_to_component_size = {}
        if 'details' in all_convergence_stats:
            for detail in all_convergence_stats['details']:
                for vp_i in detail['vp_indices']:
                    vp_to_component_size[vp_i] = detail['n_vps']
        
        # Show summary (always) - only for old method
        if not args.use_vertex_collapse:
            print(f"\nSelected VP {vp_idx} for migration")
            print(f"  Component size: {vp_to_component_size.get(vp_idx, 'unknown')}")
            print(f"  Distance to vertex: {compute_boundary_distance(partition, vp_idx):.6f}")
        
        # Show detailed output only if verbose (only for old method)
        if args.verbose and not args.use_vertex_collapse:
            single_count = 0
            two_count = 0
            three_count = 0
            other_count = 0
            
            print(f"\nShowing all {len(filtered_vps_sorted)} VPs:")
            print(f"  {'Index':<7} {'VP':<6} {'CompSize':<10} {'Lambda':<10} {'Distance':<10} {'Edge':<20}")
            print(f"  {'-'*7} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")
            
            for idx, vp_i in enumerate(filtered_vps_sorted):
                vp_temp = partition.variable_points[vp_i]
                dist = compute_boundary_distance(partition, vp_i)
                comp_size = vp_to_component_size.get(vp_i, 0)
                
                # Count by size
                if comp_size == 1:
                    size_label = "1-VP"
                    single_count += 1
                elif comp_size == 2:
                    size_label = "2-VP"
                    two_count += 1
                elif comp_size == 3:
                    size_label = "3-VP"
                    three_count += 1
                else:
                    size_label = f"{comp_size}-VP"
                    other_count += 1
            
                edge_str = f"{vp_temp.edge}"
                print(f"  [{idx:3d}]   {vp_i:<6d} {size_label:<10} {vp_temp.lambda_param:<10.6f} {dist:<10.6f} {edge_str:<20}")
            
            print(f"\n  Summary:")
            print(f"    1-VP components: {single_count}")
            print(f"    2-VP components: {two_count}")
            print(f"    3-VP components: {three_count}")
            if other_count > 0:
                print(f"    Other: {other_count}")
            
            print(f"\n  To visualize any VP, use: filtered_vps_sorted[INDEX]")
            print(f"  where INDEX is the value in brackets above.")
            print("="*80)
        
        # Get target edge BEFORE migration (for visualization)
        current_edge = vp.edge
        if args.use_vertex_collapse:
            # For vertex-collapse, find target edge using _find_opposite_edge
            target_vertex = switcher._identify_target_vertex(vp)
            if target_vertex is not None:
                target_edge = switcher._find_opposite_edge(current_edge, target_vertex)
            else:
                target_edge = None
        else:
            # Old method
            target_edge = switcher.get_best_target_edge_for_type1(vp_idx, tol=args.boundary_tol)
        
        if target_edge is None:
            print("WARNING: Could not determine target edge for visualization")
            target_edge = None
        
        # BEFORE state
        # Always initialize area_calc_before for comparison (even if only showing 'after')
        if args.verbose:
            print("\n  Initializing AreaCalculator BEFORE migration...")
        area_calc_before = AreaCalculator(mesh, partition, use_vp_based=True)
        
        if args.state in ['before', 'both']:
            print("\n" + "="*60)
            print("BEFORE Type 1 Migration")
            print("="*60)
            
            area_calc = area_calc_before  # Use the same calculator
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
            if args.use_vertex_collapse and 'highlight_vp_indices' in locals():
                # Use component VPs (migrating VP + 2 neighbors)
                highlight_vps_before = highlight_vp_indices
            else:
                # Old method: get neighbors from triangle segments
                neighbors_before = get_neighbors_from_triangle_segments(partition, vp_idx)
                highlight_vps_before = [vp_idx] + neighbors_before
            
            # Calculate camera focus for Type 1 (VP position)
            vp_pos = partition.evaluate_variable_point(vp_idx)
            
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
                highlight_vp_indices=highlight_vps_before if args.show_vps else None,
                camera_focus=vp_pos if args.apply_zoom else None,
                camera_zoom=args.zoom_factor if args.apply_zoom else None,
                current_edge=current_edge,
                target_edge=target_edge,
                verbose=args.verbose
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
            if args.use_vertex_collapse:
                # Use new vertex-collapse strategy (component-based)
                print(f"  Using vertex-collapse strategy for Component {component_idx}")
                success = switcher.apply_type1_switch_v2(selected_component)
                if not success:
                    print("ERROR: Component migration failed!")
                    return
                
                # Get new edge from migrated VP
                new_edge = partition.variable_points[migrating_vp_idx].edge
                print(f"✓ Component {component_idx} migrated successfully")
                print(f"  Migrating VP {migrating_vp_idx} moved: {old_edge} → {new_edge}")
            else:
                # Old VP-based strategy
                success = switcher.apply_type1_switch(vp_idx, tol=args.boundary_tol)
                if not success:
                    print("ERROR: Type 1 switch failed!")
                    return
                
                new_edge = vp.edge
                print(f"✓ VP {vp_idx} moved: {old_edge} → {new_edge}")
            
            # Update edges for AFTER state visualization
            # After migration, the old edge is now "current" (where it was)
            # and the new edge is where it moved to
            current_edge_after = old_edge  # Where it was before
            target_edge_after = new_edge   # Where it is now
            
            # Rebuild
            if args.verbose:
                print("  Rebuilding triangle segments...")
            partition.rebuild_triangle_segments_from_current_vps()
            if args.verbose:
                print("  Classifying segments...")
            switcher.classify_all_segments()
            
            # Re-initialize with VP-based categorization
            if args.verbose:
                print("  Re-initializing AreaCalculator (VP-based)...")
            area_calc = AreaCalculator(mesh, partition, use_vp_based=True)
            area_calc_after = area_calc  # Alias for clarity
            if args.verbose:
                print("  Re-initializing SteinerHandler...")
            steiner_handler = SteinerHandler(mesh, partition)
            
            # DEBUG SECTION 1: Compare categorization BEFORE vs AFTER
            debug_categorization_comparison(
                mesh, partition, area_calc_before, area_calc_after, args.region, verbose=args.verbose
            )

            # DEBUG SECTION 2: Find orphaned triangles
            orphan_info = debug_orphaned_triangles(mesh, partition, area_calc, args.region, verbose=args.verbose)
            
            # Handle different return formats (verbose vs non-verbose)
            if isinstance(orphan_info.get('orphaned'), list):
                # Verbose mode: 'orphaned' is a list
                orphan_count = len(orphan_info['orphaned'])
                has_orphaned = orphan_count > 0
            else:
                # Non-verbose mode: 'orphaned' is a boolean, 'count' exists
                has_orphaned = orphan_info.get('orphaned', False)
                orphan_count = orphan_info.get('count', 0)
            
            if has_orphaned:
                print(f"⚠ Found {orphan_count} orphaned triangle(s)")
                if args.verbose and convergence_info.get('target_pos') is not None:
                    target_pos = convergence_info['target_pos']
                    orphan_centroid = orphan_info.get('orphaned_centroid')
                    if orphan_centroid is not None:
                        dist = np.linalg.norm(orphan_centroid - target_pos)
                        print(f"  Distance from orphaned centroid to convergence target vertex: {dist:.6f}")
                        if dist < 0.05:
                            print(f"  → Orphaned triangles ARE near the target vertex!")

            # DEBUG: Analyze what's happening around the migrated VP (existing debug)
            debug_migrated_vp(mesh, partition, mesh_topology, vp_idx, args.region, verbose=args.verbose)
            
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
            if args.use_vertex_collapse:
                # For vertex-collapse, use the same VPs we identified before migration
                # (migrating VP + 2 neighbors) - they're still the same VPs, just moved
                if migrating_vp_idx is not None and left_neighbor is not None and right_neighbor is not None:
                    highlight_vps_after = [migrating_vp_idx, left_neighbor, right_neighbor]
                elif 'highlight_vp_indices' in locals():
                    highlight_vps_after = highlight_vp_indices
                else:
                    # Last resort: try to get from component
                    if selected_component is not None:
                        highlight_vps_after = selected_component['vp_indices']
                    else:
                        highlight_vps_after = [vp_idx]
                
                print(f"  Highlighting VPs after migration: {highlight_vps_after}")
                print(f"    Expected: [migrating VP {migrating_vp_idx}, neighbor 1 {left_neighbor}, neighbor 2 {right_neighbor}]")
            else:
                # Old method: get neighbors from triangle segments
                neighbors_after = get_neighbors_from_triangle_segments(partition, vp_idx)
                highlight_vps_after = [vp_idx] + neighbors_after
                print(f"  Highlighting VPs after migration: {highlight_vps_after}")
            
            # Calculate new VP position for camera focus
            new_vp_pos = partition.evaluate_variable_point(vp_idx)
            
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
                highlight_vp_indices=highlight_vps_after if args.show_vps else None,
                camera_focus=new_vp_pos if args.apply_zoom else None,
                camera_zoom=args.zoom_factor if args.apply_zoom else None,
                current_edge=current_edge_after,  # Old edge (where VP was)
                target_edge=target_edge_after,    # New edge (where VP is now)
                verbose=args.verbose
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
        
        tp = boundary_tps[3]
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
            
            # Calculate camera focus for Type 2 (Steiner point position)
            steiner_pos = tp.compute_steiner_point()
            
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
                highlight_vp_indices=highlight_vps_before if args.show_vps else None,
                camera_focus=steiner_pos if args.apply_zoom else None,
                camera_zoom=args.zoom_factor if args.apply_zoom else None,
                verbose=args.verbose
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
            new_steiner_pos = None
            for tp_new in steiner_handler.triple_points:
                if any(vp_idx in tp.var_point_indices for vp_idx in tp_new.var_point_indices):
                    new_tp_vps = list(tp_new.var_point_indices)
                    new_steiner_pos = tp_new.compute_steiner_point()
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
                highlight_vp_indices=new_tp_vps if args.show_vps else None,
                camera_focus=new_steiner_pos if args.apply_zoom and new_steiner_pos is not None else None,
                camera_zoom=args.zoom_factor if args.apply_zoom else None,
                verbose=args.verbose
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
    parser.add_argument('--apply-zoom', action='store_true',
                       help='Automatically zoom and focus camera on the migration region')
    parser.add_argument('--zoom-factor', type=float, default=0.05,
                       help='Zoom level (default: 0.05, smaller = more zoomed in)')
    parser.add_argument('--save-before', type=str,
                       help='Path to save BEFORE state image')
    parser.add_argument('--save-after', type=str,
                       help='Path to save AFTER state image')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed debug output (default: False)')
    parser.add_argument('--use-vertex-collapse', action='store_true',
                       help='Use new vertex-collapse strategy for Type 1 migration (default: False)')
    parser.add_argument('--component-index', type=int, default=None,
                       help='Index of component to migrate (only used with --use-vertex-collapse). '
                            'If not specified, shows all components and uses component 0.')
    
    args = parser.parse_args()
    
    # Validate region index
    if args.region < 0:
        print(f"ERROR: Region index must be >= 0")
        return 1
    
    run_visualization(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

