#!/usr/bin/env python3
"""
Component Proximity Analysis for Proposed Type 1 Migration Strategy

This script analyzes whether boundary VP components are close enough
to interfere with each other during migration.

Author: Esteban Velez
Date: January 2026
"""

import numpy as np
from collections import defaultdict
from typing import List, Set, Tuple, Dict


def debug_component_proximity_analysis(mesh, partition, boundary_tol=0.01, filtered_vps_sorted=None):
    """
    Analyze proximity between connected components of boundary VPs.
    
    Identifies cases where two components might interfere:
    1. Share a common non-boundary VP neighbor
    2. Are spatially close (< threshold distance)
    3. Share target vertex with different approach angles
    
    Args:
        mesh: TriMesh object
        partition: PartitionContour object
        boundary_tol: Tolerance for boundary VP detection
        filtered_vps_sorted: Optional list of VPs sorted by distance (for index lookup)
        
    Returns:
        dict with analysis results
    """
    print("\n" + "="*80)
    print("COMPONENT PROXIMITY ANALYSIS")
    print("="*80)
    
    # Step 1: Get boundary VPs and build components
    boundary_vps = partition.get_boundary_variable_points(tol=boundary_tol)
    boundary_vps_set = set(boundary_vps)
    
    print(f"\n  Boundary tolerance: {boundary_tol}")
    print(f"  Total boundary VPs: {len(boundary_vps)}")
    
    # Build adjacency from boundary_segments
    adjacency = defaultdict(set)
    for seg in partition.boundary_segments:
        vp1, vp2 = seg.vp_idx_1, seg.vp_idx_2
        adjacency[vp1].add(vp2)
        adjacency[vp2].add(vp1)
    
    # Find connected components using DFS
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
            
            # Only follow edges through other boundary VPs
            for neighbor in adjacency[current]:
                if neighbor in boundary_vps_set and neighbor not in visited:
                    stack.append(neighbor)
        
        components.append(component)
    
    print(f"  Connected components found: {len(components)}")
    
    # Step 2: For each component, find its neighbors (including non-boundary)
    component_info = []
    
    for comp_idx, comp_vps in enumerate(components):
        # Get all neighbors of this component
        all_neighbors = set()
        for vp_idx in comp_vps:
            all_neighbors.update(adjacency[vp_idx])
        
        # External neighbors (not in this component)
        external_neighbors = all_neighbors - comp_vps
        
        # Separate boundary and non-boundary external neighbors
        boundary_neighbors = external_neighbors & boundary_vps_set
        non_boundary_neighbors = external_neighbors - boundary_vps_set
        
        # Find target vertex (common vertex all VPs share)
        all_edges = []
        all_vertices = set()
        for vp_idx in comp_vps:
            vp = partition.variable_points[vp_idx]
            all_edges.append(tuple(sorted(vp.edge)))
            all_vertices.update(vp.edge)
        
        target_vertex = None
        for v in all_vertices:
            if all(v in edge for edge in all_edges):
                target_vertex = v
                break
        
        # Compute centroid
        positions = []
        for vp_idx in comp_vps:
            vp = partition.variable_points[vp_idx]
            positions.append(vp.evaluate(mesh.vertices))
        centroid = np.mean(positions, axis=0) if positions else None
        
        component_info.append({
            'index': comp_idx,
            'vp_indices': list(comp_vps),
            'size': len(comp_vps),
            'target_vertex': target_vertex,
            'centroid': centroid,
            'boundary_neighbors': list(boundary_neighbors),
            'non_boundary_neighbors': list(non_boundary_neighbors)
        })
    
    # Step 3: Find proximity conflicts
    print(f"\n━━━ PROXIMITY CONFLICT DETECTION ━━━\n")
    
    conflicts = []
    
    # Type 1: Shared non-boundary neighbors with proximity analysis
    print(f"  Checking for shared non-boundary neighbors...")
    
    for i in range(len(component_info)):
        for j in range(i + 1, len(component_info)):
            comp_i = component_info[i]
            comp_j = component_info[j]
            
            shared_non_boundary = set(comp_i['non_boundary_neighbors']) & set(comp_j['non_boundary_neighbors'])
            
            if shared_non_boundary:
                # Calculate minimum distances in each component (closest VP to target vertex)
                min_dist_i = min(min(partition.variable_points[vp].lambda_param,
                                     1.0 - partition.variable_points[vp].lambda_param)
                                 for vp in comp_i['vp_indices'])
                min_dist_j = min(min(partition.variable_points[vp].lambda_param,
                                     1.0 - partition.variable_points[vp].lambda_param)
                                 for vp in comp_j['vp_indices'])
                
                # Determine if both components are near convergence
                proximity_threshold = 0.01
                both_near = min_dist_i < proximity_threshold and min_dist_j < proximity_threshold
                
                conflicts.append({
                    'type': 'shared_non_boundary_neighbor',
                    'component_i': i,
                    'component_j': j,
                    'shared_vps': list(shared_non_boundary),
                    'min_dist_i': min_dist_i,
                    'min_dist_j': min_dist_j,
                    'both_near_convergence': both_near,
                    'details': f"Components {i} and {j} share non-boundary neighbor(s): {list(shared_non_boundary)}"
                })
    
    print(f"    Found {sum(1 for c in conflicts if c['type'] == 'shared_non_boundary_neighbor')} shared neighbor conflicts (with proximity analysis)")
    
    # Type 2: Same target vertex with connected boundary neighbors
    print(f"  Checking shared target vertex with boundary connections...")
    
    for i in range(len(component_info)):
        for j in range(i + 1, len(component_info)):
            comp_i = component_info[i]
            comp_j = component_info[j]
            
            # Check if they share target vertex
            if comp_i['target_vertex'] == comp_j['target_vertex'] and comp_i['target_vertex'] is not None:
                # Check if one's boundary neighbors include the other's VPs
                shared_boundary = (set(comp_i['boundary_neighbors']) & set(comp_j['vp_indices'])) | \
                                (set(comp_j['boundary_neighbors']) & set(comp_i['vp_indices']))
                
                if shared_boundary:
                    # Analyze migration directions
                    target_pos = mesh.vertices[comp_i['target_vertex']]
                    dir_i = comp_i['centroid'] - target_pos if comp_i['centroid'] is not None else None
                    dir_j = comp_j['centroid'] - target_pos if comp_j['centroid'] is not None else None
                    
                    # Compute angle between directions
                    if dir_i is not None and dir_j is not None:
                        dot_product = np.dot(dir_i, dir_j) / (np.linalg.norm(dir_i) * np.linalg.norm(dir_j))
                        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
                        opposite_direction = angle_deg > 90
                    else:
                        angle_deg = None
                        opposite_direction = False
                    
                    conflicts.append({
                        'type': 'shared_target_vertex',
                        'component_i': i,
                        'component_j': j,
                        'target_vertex': comp_i['target_vertex'],
                        'shared_boundary_vps': list(shared_boundary),
                        'angle_deg': angle_deg,
                        'opposite_direction': opposite_direction,
                        'details': f"Components {i} and {j} share target vertex {comp_i['target_vertex']}, angle = {angle_deg:.1f}°"
                    })
    
    print(f"    Found {sum(1 for c in conflicts if c['type'] == 'shared_target_vertex')} shared target vertex conflicts")
    
    # Step 4: Detailed conflict reports
    if conflicts:
        print(f"\n━━━ DETAILED CONFLICT REPORT ━━━\n")
        print(f"  Total conflicts detected: {len(conflicts)}\n")
        
        # Build VP -> filtered index map if available
        vp_to_filtered_idx = {}
        if filtered_vps_sorted is not None:
            for idx, vp_idx in enumerate(filtered_vps_sorted):
                vp_to_filtered_idx[vp_idx] = idx
        
        for idx, conflict in enumerate(conflicts):
            print(f"  Conflict {idx + 1}: {conflict['type'].upper()}")
            print(f"    {conflict['details']}")
            
            # Show VPs from each component
            comp_i_idx = conflict['component_i']
            comp_j_idx = conflict['component_j']
            comp_i = component_info[comp_i_idx]
            comp_j = component_info[comp_j_idx]
            
            # Helper function to format VP info
            def format_vp_info(vp_idx):
                vp = partition.variable_points[vp_idx]
                dist = min(vp.lambda_param, 1.0 - vp.lambda_param)
                filtered_idx = vp_to_filtered_idx.get(vp_idx, '?')
                if filtered_idx != '?':
                    return f"VP {vp_idx:4d} [idx={filtered_idx:3d}], λ={vp.lambda_param:.6f}, dist={dist:.6f}, edge={vp.edge}"
                else:
                    return f"VP {vp_idx:4d} [not in filtered], λ={vp.lambda_param:.6f}, dist={dist:.6f}, edge={vp.edge}"
            
            print(f"\n    Component {comp_i_idx} VPs ({comp_i['size']} total):")
            for vp_idx in sorted(comp_i['vp_indices'], 
                                key=lambda v: min(partition.variable_points[v].lambda_param, 
                                                 1.0 - partition.variable_points[v].lambda_param)):
                print(f"      {format_vp_info(vp_idx)}")
            
            print(f"\n    Component {comp_j_idx} VPs ({comp_j['size']} total):")
            for vp_idx in sorted(comp_j['vp_indices'],
                                key=lambda v: min(partition.variable_points[v].lambda_param,
                                                 1.0 - partition.variable_points[v].lambda_param)):
                print(f"      {format_vp_info(vp_idx)}")
            
            # Show shared neighbors if applicable
            if conflict['type'] == 'shared_non_boundary_neighbor':
                shared_vps = conflict['shared_vps']
                print(f"\n    Shared non-boundary neighbor(s):")
                for shared_vp_idx in shared_vps:
                    print(f"      {format_vp_info(shared_vp_idx)}")
                
                # Show proximity analysis
                min_dist_i = conflict.get('min_dist_i')
                min_dist_j = conflict.get('min_dist_j')
                if min_dist_i is not None and min_dist_j is not None:
                    print(f"\n    Proximity analysis:")
                    print(f"      Component {comp_i_idx} closest VP distance: {min_dist_i:.6f}")
                    print(f"      Component {comp_j_idx} closest VP distance: {min_dist_j:.6f}")
                    if conflict.get('both_near_convergence', False):
                        print(f"      ⚠ Both components near convergence (< 0.01) - high interference risk")
                    else:
                        print(f"      ✓ At least one component still far from vertex - lower risk")
            
            if conflict['type'] == 'shared_target_vertex' and 'angle_deg' in conflict:
                angle = conflict['angle_deg']
                if angle is not None:
                    direction = "OPPOSITE" if conflict['opposite_direction'] else "SAME"
                    print(f"\n    Migration direction: {direction} (angle = {angle:.1f}°)")
                    if conflict['opposite_direction']:
                        print(f"    ✓ Should naturally resolve (opposite directions)")
                    else:
                        print(f"    ⚠ May create conflict (same direction)")
            
            print()
    else:
        print(f"\n  ✓ No proximity conflicts detected!")
        print(f"    All components are sufficiently isolated.")
    
    # Step 5: Statistics and recommendations
    print(f"\n━━━ SUMMARY AND RECOMMENDATIONS ━━━\n")
    
    total_components = len(components)
    conflicted_components = set()
    for c in conflicts:
        conflicted_components.add(c['component_i'])
        conflicted_components.add(c['component_j'])
    
    conflict_rate = 100 * len(conflicted_components) / total_components if total_components > 0 else 0
    
    print(f"  Total components: {total_components}")
    print(f"  Components involved in conflicts: {len(conflicted_components)} ({conflict_rate:.1f}%)")
    print(f"  Total conflict pairs: {len(conflicts)}")
    
    # Analyze opposite direction hypothesis
    shared_vertex_conflicts = [c for c in conflicts if c['type'] == 'shared_target_vertex']
    if shared_vertex_conflicts:
        opposite_count = sum(1 for c in shared_vertex_conflicts if c.get('opposite_direction', False))
        opposite_pct = 100 * opposite_count / len(shared_vertex_conflicts)
        print(f"\n  Shared target vertex conflicts: {len(shared_vertex_conflicts)}")
        print(f"    Moving in opposite directions: {opposite_count} ({opposite_pct:.1f}%)")
        print(f"    Moving in same direction: {len(shared_vertex_conflicts) - opposite_count} ({100-opposite_pct:.1f}%)")
        
        if opposite_pct >= 90:
            print(f"\n  ✓ HYPOTHESIS VALIDATED: Components move in opposite directions")
            print(f"    Recommendation: Migrate closest component first, defer the other")
        elif opposite_pct >= 70:
            print(f"\n  ⚠ HYPOTHESIS PARTIALLY VALIDATED")
            print(f"    Recommendation: Use distance-based priority, monitor exceptions")
        else:
            print(f"\n  ✗ HYPOTHESIS NOT VALIDATED: Many same-direction cases")
            print(f"    Recommendation: Develop more sophisticated conflict resolution")
    
    # Final recommendation
    print(f"\n━━━ GO/NO-GO DECISION ━━━\n")
    
    if conflict_rate < 10:
        print(f"  ✅ PROCEED: Low conflict rate ({conflict_rate:.1f}% < 10%)")
    elif conflict_rate < 20:
        print(f"  ⚠ CAUTION: Moderate conflict rate ({conflict_rate:.1f}%)")
        print(f"     Proceed but implement robust conflict resolution")
    else:
        print(f"  ❌ RECONSIDER: High conflict rate ({conflict_rate:.1f}% >= 20%)")
        print(f"     Strategy may need revision")
    
    print("\n" + "="*80)
    
    return {
        'components': component_info,
        'conflicts': conflicts,
        'total_components': total_components,
        'conflicted_components': len(conflicted_components),
        'conflict_rate': conflict_rate
    }


if __name__ == "__main__":
    print("This is a debugging module for component proximity analysis.")
    print("Import and call debug_component_proximity_analysis(mesh, partition)")

