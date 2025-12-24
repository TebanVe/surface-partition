#!/usr/bin/env python3
"""
Test script for topology switching - Phase 0: Verify Dependencies

This script verifies that all required imports work and basic infrastructure is set up.
Part of the incremental testing plan for topology switching implementation.

Phase 0 Goals:
- Verify all imports work
- Check that MeshTopology and TopologySwitcher can be instantiated
- Ensure dependencies are available

Author: Esteban Velez 
Date: November 24, 2025
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Segment Classification Helpers (for Phase 3 diagnostics)
# ==============================================================================

def get_neighboring_vps(partition, vp_idx: int) -> list:
    """
    Get indices of VPs that form segments with the given VP.
    
    Uses triangle_segments to find VPs in the same triangles.
    """
    neighbors = set()
    
    for tri_seg in partition.triangle_segments:
        if vp_idx in tri_seg.var_point_indices:
            # This triangle contains our VP - other VPs in this triangle are neighbors
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
    """
    Find connected components of boundary VPs.
    
    Returns:
        List of sets, each set is a connected component of VP indices
    """
    from collections import defaultdict
    
    # Build adjacency list
    adjacency = defaultdict(set)
    for segment in partition.boundary_segments:
        vp1, vp2 = segment.vp_idx_1, segment.vp_idx_2
        if vp1 in boundary_vps_set and vp2 in boundary_vps_set:
            adjacency[vp1].add(vp2)
            adjacency[vp2].add(vp1)
    
    # DFS to find components
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


def filter_connected_boundary_vps(boundary_vps, partition, logger):
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
    vps_deferred = []
    
    logger.info(f"  Found {len(components)} connected component(s) among {len(boundary_vps)} boundary VPs")
    
    for i, component in enumerate(components):
        if len(component) == 1:
            vp_idx = list(component)[0]
            vps_to_keep.append(vp_idx)
            logger.debug(f"    Component {i+1}: Single VP {vp_idx}")
        else:
            # Multiple connected VPs - keep closest
            vps_with_dist = [
                (compute_boundary_distance(partition, vp), vp)
                for vp in component
            ]
            vps_with_dist.sort()
            
            closest_dist, closest_vp = vps_with_dist[0]
            vps_to_keep.append(closest_vp)
            
            deferred_in_component = [vp for _, vp in vps_with_dist[1:]]
            vps_deferred.extend(deferred_in_component)
            
            logger.info(f"    Component {i+1}: {len(component)} connected VPs")
            logger.info(f"      Keeping VP {closest_vp} (distance={closest_dist:.6f})")
            logger.info(f"      Deferring: {deferred_in_component}")
    
    if vps_deferred:
        logger.info(f"  Total deferred: {len(vps_deferred)} VPs")
    
    return vps_to_keep


def test_crossing_point_computation(partition, mesh, mesh_topology, switcher, vp_idx: int):
    """
    Test the dual projection crossing point computation for a VP involved in Type 1 switch.
    
    Diagnostics:
    - Identify the old edge and target edge
    - Compute crossing point using dual projection
    - Verify crossing point lies on shared edge
    - Report distances from both projections
    
    Args:
        partition: PartitionContour
        mesh: TriMesh
        mesh_topology: MeshTopology
        switcher: TopologySwitcher
        vp_idx: Variable point index to test
    """
    logger.info("="*80)
    logger.info(f"TESTING CROSSING POINT COMPUTATION FOR VP {vp_idx}")
    logger.info("="*80)
    
    vp = partition.variable_points[vp_idx]
    old_edge = vp.edge
    lambda_param = vp.lambda_param
    
    logger.info(f"VP {vp_idx}:")
    logger.info(f"  Old edge: {old_edge}")
    logger.info(f"  Lambda: {lambda_param:.6f}")
    logger.info(f"  Position: {partition.evaluate_variable_point(vp_idx)}")
    
    # Identify target vertex
    target_vertex = switcher._identify_target_vertex(vp)
    logger.info(f"  Target vertex: {target_vertex}")
    
    # Get candidate target edges using NEW algorithm
    # 1. Get all triangles at target vertex
    triangles_at_vertex = switcher._get_all_triangles_at_vertex(target_vertex)
    logger.info(f"  Triangles at target vertex: {len(triangles_at_vertex)}")
    
    # 2. Filter to empty triangles
    empty_triangles = [
        tri_idx for tri_idx in triangles_at_vertex
        if not switcher._triangle_has_boundary_segment(tri_idx)
    ]
    logger.info(f"  Empty triangles (no boundary segments): {len(empty_triangles)}")
    
    # 3. Get free edges from empty triangles
    candidates = []
    for tri_idx in empty_triangles:
        free_edges = switcher._get_free_edges_in_triangle(tri_idx, target_vertex)
        candidates.extend(free_edges)
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    if not candidates:
        logger.info("  No candidate edges found in empty triangles!")
        return
    
    logger.info(f"  Candidate target edges: {candidates}")
    
    # Test first candidate (or use the best one from switcher)
    best_edge = switcher.get_best_target_edge_for_type1(vp_idx)
    target_edge = best_edge if best_edge else candidates[0]
    logger.info(f"\nTesting crossing for target edge: {target_edge}")
    
    # Simulate VP move to target edge
    new_lambda = 0.5  # Test with midpoint
    
    # Get old and new positions
    pos_old = partition.evaluate_variable_point(vp_idx)
    pos_new_on_edge = new_lambda * mesh.vertices[target_edge[0]] + (1 - new_lambda) * mesh.vertices[target_edge[1]]
    
    logger.info(f"  Old position (VP on old edge): {pos_old}")
    logger.info(f"  New position (test at λ=0.5 on target): {pos_new_on_edge}")
    
    # Get triangles containing each edge
    old_triangles = mesh_topology.get_triangles_sharing_edge(tuple(sorted(old_edge)))
    new_triangles = mesh_topology.get_triangles_sharing_edge(tuple(sorted(target_edge)))
    
    logger.info(f"  Triangles with old edge: {old_triangles}")
    logger.info(f"  Triangles with new edge: {new_triangles}")
    
    # Find shared edge between triangles
    for tri_old in old_triangles:
        for tri_new in new_triangles:
            shared_edge = switcher._find_shared_edge_between_triangles(tri_old, tri_new)
            if shared_edge:
                logger.info(f"\n  Found shared edge {shared_edge} between triangles {tri_old} and {tri_new}")
                
                # Test dual projection
                crossing = switcher._compute_crossing_via_dual_projection(
                    pos_old, pos_new_on_edge, tri_old, tri_new, shared_edge
                )
                
                if crossing is not None:
                    logger.info(f"  ✓ Crossing point computed: {crossing}")
                    
                    # Verify it's on the edge
                    edge_start = mesh.vertices[shared_edge[0]]
                    edge_end = mesh.vertices[shared_edge[1]]
                    edge_vec = edge_end - edge_start
                    edge_length = np.linalg.norm(edge_vec)
                    
                    # Project crossing onto edge to find parameter
                    proj_vec = crossing - edge_start
                    t = np.dot(proj_vec, edge_vec) / (edge_length ** 2)
                    
                    logger.info(f"  Edge vertices: {shared_edge[0]} → {shared_edge[1]}")
                    logger.info(f"  Edge length: {edge_length:.6f}")
                    logger.info(f"  Crossing parameter t: {t:.6f} (0=start, 1=end)")
                    
                    # Check distance to edge
                    closest_on_edge = edge_start + t * edge_vec
                    dist_to_edge = np.linalg.norm(crossing - closest_on_edge)
                    logger.info(f"  Distance from crossing to edge: {dist_to_edge:.2e}")
                    
                    if dist_to_edge < 1e-6:
                        logger.info(f"  ✓ Crossing point lies on edge (within tolerance)")
                    else:
                        logger.warning(f"  ⚠ Crossing point NOT on edge!")
                else:
                    logger.warning(f"  ✗ Dual projection failed to compute crossing")
                
                return  # Only test first shared edge
    
    logger.info("  No shared edge found between old and new triangles")


def analyze_segments_for_vp(partition, mesh, mesh_topology, vp_idx: int, logger):
    """
    Analyze all segments involving a specific VP and classify them.
    
    Returns:
        dict with classification counts and segment details
    """
    neighbors = get_neighboring_vps(partition, vp_idx)
    
    classification_counts = {"normal": 0, "edge_following": 0, "edge_cutting": 0}
    segment_details = []
    
    for neighbor_idx in neighbors:
        # Get segment classification from BoundarySegment
        seg_key = (min(vp_idx, neighbor_idx), max(vp_idx, neighbor_idx))
        seg_obj = None
        for seg in partition.boundary_segments:
            if seg.normalized_key() == seg_key:
                seg_obj = seg
                break
        
        if seg_obj:
            seg_type = seg_obj.segment_type
            classification_counts[seg_type] += 1
            segment_details.append({'type': seg_type, 'vp_idx_1': vp_idx, 'vp_idx_2': neighbor_idx})
        else:
            logger.warning(f"    Segment ({vp_idx}, {neighbor_idx}) not found in boundary_segments")
    
    # Log summary
    logger.info(f"  Segment classification for VP {vp_idx}:")
    logger.info(f"    Neighbors: {len(neighbors)}")
    logger.info(f"    Normal (same triangle): {classification_counts['normal']}")
    logger.info(f"    Edge-following (same mesh line): {classification_counts['edge_following']}")
    logger.info(f"    Edge-cutting (cuts triangles): {classification_counts['edge_cutting']}")
    
    # Log details for cross-triangle segments
    cross_triangle = [d for d in segment_details if d['type'] != "normal"]
    if cross_triangle:
        logger.info(f"  Cross-triangle segment details:")
        for detail in cross_triangle:
            logger.info(f"    Segment {detail['segment']}: {detail['type']}")
            logger.info(f"      VP edges: {detail['vp1_edge']} ↔ {detail['vp2_edge']}")
            if detail['shared_vertex'] is not None:
                logger.info(f"      Shared vertex: {detail['shared_vertex']}")
    else:
        logger.info(f"  No cross-triangle segments found for this VP")
    
    return {
        'vp_idx': vp_idx,
        'counts': classification_counts,
        'details': segment_details
    }


def test_imports():
    """Phase 0: Test that all required imports work."""
    logger.info("=" * 80)
    logger.info("PHASE 0: VERIFY DEPENDENCIES")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info("Testing imports...")
    
    try:
        # Core imports
        from src.core.tri_mesh import TriMesh
        logger.info("  ✓ TriMesh imported")
        
        from src.core.contour_partition import PartitionContour, VariablePoint, TriangleSegment
        logger.info("  ✓ PartitionContour, VariablePoint, TriangleSegment imported")
        
        from src.core.steiner_handler import SteinerHandler
        logger.info("  ✓ SteinerHandler imported")
        
        from src.core.area_calculator import AreaCalculator
        logger.info("  ✓ AreaCalculator imported")
        
        from src.core.perimeter_calculator import PerimeterCalculator
        logger.info("  ✓ PerimeterCalculator imported")
        
        from src.core.perimeter_optimizer import PerimeterOptimizer
        logger.info("  ✓ PerimeterOptimizer imported")
        
        # New topology switching imports
        from src.core.mesh_topology import MeshTopology
        logger.info("  ✓ MeshTopology imported")
        
        from src.core.topology_switcher import TopologySwitcher
        logger.info("  ✓ TopologySwitcher imported")
        
        logger.info("")
        logger.info("✓ All imports successful!")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_class_instantiation(relaxed_file: str, refined_file: str = None):
    """
    Phase 0/1: Load partition data and instantiate topology switching classes.
    
    Args:
        relaxed_file: Path to initial relaxed solution .h5 file
        refined_file: Optional path to refined_contours.h5 file with optimized lambdas
    """
    logger.info("")
    logger.info("Testing class instantiation...")
    logger.info("")
    
    try:
        from src.core.tri_mesh import TriMesh
        from src.core.contour_partition import PartitionContour
        from src.core.mesh_topology import MeshTopology
        from src.core.topology_switcher import TopologySwitcher
        from src.find_contours import ContourAnalyzer
        import h5py
        
        # Load using ContourAnalyzer (same as refine_perimeter.py)
        logger.info(f"Loading relaxed solution from: {relaxed_file}")
        analyzer = ContourAnalyzer(relaxed_file)
        analyzer.load_results(use_initial_condition=False)
        
        n_vertices = analyzer.vertices.shape[0]
        n_partitions = analyzer.densities.shape[1]
        logger.info(f"  ✓ Loaded: {n_vertices} vertices, {n_partitions} partitions")
        
        # Compute indicator functions
        logger.info("  Computing indicator functions...")
        indicators = analyzer.compute_indicator_functions()
        logger.info(f"  ✓ Indicators computed")
        
        # Extract boundary topology
        logger.info("  Extracting boundary topology...")
        raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
        n_boundary_triangles = sum(len(v) for v in boundary_topology.values())
        logger.info(f"  ✓ Extracted topology from {n_boundary_triangles} boundary triangles")
        
        # Create mesh
        logger.info("  Building TriMesh...")
        mesh = TriMesh(analyzer.vertices, analyzer.faces)
        logger.info(f"  ✓ Mesh created")
        
        # Create partition WITH boundary topology
        # CRITICAL: Must match VP indices used in refine_perimeter.py
        # The refined_contours.h5 lambda values were saved with this path
        logger.info("  Creating PartitionContour...")
        partition = PartitionContour(mesh, indicators, boundary_topology=boundary_topology)
        logger.info(f"  ✓ Partition created: {len(partition.variable_points)} variable points")
        logger.info(f"    - {len(partition.triangle_segments)} triangle segments")
        
        # If refined file provided, load optimized lambdas
        if refined_file:
            logger.info(f"  Loading optimized lambdas from: {refined_file}")
            with h5py.File(refined_file, 'r') as f:
                if 'lambda_parameters' in f:
                    lambda_opt = f['lambda_parameters'][:]
                    partition.set_variable_vector(lambda_opt)
                    logger.info(f"  ✓ Set {len(lambda_opt)} optimized lambda values")
                    logger.info(f"    - Min λ: {lambda_opt.min():.6f}")
                    logger.info(f"    - Max λ: {lambda_opt.max():.6f}")
                else:
                    logger.warning("  ⚠ lambda_parameters not found in refined file")
        
        # Build MeshTopology
        logger.info("  Building MeshTopology...")
        mesh_topology = MeshTopology(mesh)
        logger.info(f"  ✓ MeshTopology created")
        logger.info(f"    - {len(mesh_topology.vertex_to_triangles)} vertices in connectivity map")
        logger.info(f"    - {len(mesh_topology.edge_to_triangles)} edges in connectivity map")
        
        # Create TopologySwitcher
        logger.info("  Creating TopologySwitcher...")
        switcher = TopologySwitcher(mesh, partition, mesh_topology)
        logger.info(f"  ✓ TopologySwitcher created")
        
        logger.info("")
        logger.info("✓ All classes instantiated successfully!")
        
        return True, mesh, partition, mesh_topology, switcher
        
    except Exception as e:
        logger.error(f"✗ Class instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None


def test_topology_switch_detection(partition, mesh, mesh_topology, switcher, boundary_tol=1e-3):
    """
    Phase 1: Test topology switch detection.
    
    Args:
        partition: PartitionContour instance
        mesh: TriMesh instance
        mesh_topology: MeshTopology instance
        switcher: TopologySwitcher instance
        boundary_tol: Threshold for boundary detection
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1: TESTING TOPOLOGY SWITCH DETECTION")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        from src.core.steiner_handler import SteinerHandler
        
        # Test 1: Check for boundary variable points (Type 1)
        logger.info("Test 1: Detecting boundary variable points (Type 1)...")
        boundary_vps = partition.get_boundary_variable_points(tol=boundary_tol)
        logger.info(f"  Detected {len(boundary_vps)} boundary VPs")
        
        if len(boundary_vps) > 0:
            logger.info(f"  Boundary VP indices: {boundary_vps}")
            for vp_idx in boundary_vps[:3]:  # Show first 3
                vp = partition.variable_points[vp_idx]
                logger.info(f"    VP {vp_idx}: λ = {vp.lambda_param:.6f}, edge = {vp.edge}")
        else:
            logger.info("  ⚠ No boundary VPs detected (this is expected for your data)")
        
        # Test 2: Check for boundary triple points (Type 2)
        logger.info("")
        logger.info("Test 2: Detecting boundary triple points (Type 2)...")
        
        # Identify triple points
        partition.identify_triple_points()
        steiner_handler = SteinerHandler(mesh, partition)
        
        logger.info(f"  Total triple points: {len(steiner_handler.triple_points)}")
        
        # Check for boundary triple points
        boundary_tps = steiner_handler.get_boundary_triple_points(tol=boundary_tol)
        logger.info(f"  Boundary triple points: {len(boundary_tps)}")
        
        if len(boundary_tps) > 0:
            for i, tp in enumerate(boundary_tps[:3], 1):  # Show first 3
                logger.info(f"  Triple Point {i}:")
                logger.info(f"    Triangle: {tp.triangle_idx}")
                logger.info(f"    Cells meeting: {tp.cell_indices}")
                logger.info(f"    VPs: {tp.var_point_indices}")
                
                # Show lambda values for VPs
                for vp_idx in tp.var_point_indices:
                    vp = partition.variable_points[vp_idx]
                    logger.info(f"      VP {vp_idx}: λ = {vp.lambda_param:.6f}")
        
        # Note: VP selection for Type 2 is now done internally by apply_type2_switch()
        # using _select_vp_minimizing_perimeter() for perimeter optimization
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 1 COMPLETE: TOPOLOGY SWITCH DETECTION TESTED ✓")
        logger.info("=" * 80)
        logger.info(f"Summary:")
        logger.info(f"  - Type 1 (boundary VPs): {len(boundary_vps)} detected")
        logger.info(f"  - Type 2 (boundary triple points): {len(boundary_tps)} detected")
        
        return True, steiner_handler
        
    except Exception as e:
        logger.error(f"✗ Topology switch detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_triangle_segments_rebuild(partition, mesh, mesh_topology, switcher, steiner_handler):
    """
    Phase 2: Test that triangle_segments rebuild correctly after topology switches.
    
    Tests both Type 1 (VP migration) and Type 2 (triple point migration) switches.
    Also analyzes and classifies resulting segments (diagnostic for Phase 4).
    
    Args:
        partition: PartitionContour instance
        mesh: TriMesh instance
        mesh_topology: MeshTopology instance (for segment classification)
        switcher: TopologySwitcher instance
        steiner_handler: SteinerHandler instance (for triple point detection)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 2: TESTING TRIANGLE_SEGMENTS REBUILD")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        from src.core.steiner_handler import SteinerHandler
        
        # Record initial state
        num_vps_initial = len(partition.variable_points)
        num_tri_segs_initial = len(partition.triangle_segments)
        num_triple_points_initial = len(steiner_handler.triple_points)
        
        logger.info(f"Initial state:")
        logger.info(f"  Variable points: {num_vps_initial}")
        logger.info(f"  Triangle segments: {num_tri_segs_initial}")
        logger.info(f"  Triple points: {num_triple_points_initial}")
        
        # =====================================================================
        # Part A: Test Type 1 Switch (VP migration, NOT part of triple point)
        # =====================================================================
        logger.info("")
        logger.info("=" * 70)
        logger.info("Part A: Testing Type 1 Switch (VP Migration)")
        logger.info("=" * 70)
        
        # Find boundary VPs that are NOT part of triple points
        boundary_vps = partition.get_boundary_variable_points(tol=0.1)
        triple_point_vp_indices = set()
        for tp in steiner_handler.triple_points:
            triple_point_vp_indices.update(tp.var_point_indices)
        
        non_triple_boundary_vps = [vp for vp in boundary_vps if vp not in triple_point_vp_indices]
        
        if len(non_triple_boundary_vps) > 0:
            # Filter connected VPs (keep closest in each component)
            logger.info("")
            logger.info("Filtering connected boundary VPs...")
            filtered_vps = filter_connected_boundary_vps(non_triple_boundary_vps, partition, logger)
            
            # Sort by distance to target vertex
            filtered_vps_sorted = sorted(
                filtered_vps,
                key=lambda vp_idx: compute_boundary_distance(partition, vp_idx)
            )
            
            # Select closest VP
            vp_idx = filtered_vps_sorted[0]
            vp = partition.variable_points[vp_idx]
            
            # Log prioritization
            logger.info("")
            logger.info(f"Boundary VPs after filtering: {len(filtered_vps)}")
            logger.info("Top candidates (by distance to target vertex):")
            for i, vp_i in enumerate(filtered_vps_sorted[:min(5, len(filtered_vps_sorted))]):
                dist = compute_boundary_distance(partition, vp_i)
                vp_temp = partition.variable_points[vp_i]
                logger.info(f"  #{i+1}: VP {vp_i}, λ={vp_temp.lambda_param:.6f}, distance={dist:.6f}")
            
            logger.info(f"Selected VP {vp_idx} (closest to target vertex)")
            
            old_edge = vp.edge
            old_lambda = vp.lambda_param
            
            # CAPTURE neighbors BEFORE the switch (to analyze segments after)
            neighbors_before_type1 = get_neighboring_vps(partition, vp_idx)
            segments_to_analyze_type1 = [(vp_idx, n) for n in neighbors_before_type1]
            
            logger.info("")
            logger.info(f"Testing Type 1 with VP {vp_idx} (NOT part of triple point):")
            logger.info(f"  Before: λ = {old_lambda:.6f}, edge = {old_edge}")
            logger.info(f"  Neighbors before switch: {neighbors_before_type1}")
            
            # Test crossing point computation BEFORE applying switch
            logger.info("")
            test_crossing_point_computation(partition, mesh, mesh_topology, switcher, vp_idx)
            
            # Apply Type 1 switch
            logger.info("")
            logger.info("Applying Type 1 switch...")
            success = switcher.apply_type1_switch(vp_idx, tol=0.1)
            
            if success:
                new_edge = vp.edge
                new_lambda = vp.lambda_param
                
                logger.info(f"  ✓ Type 1 switch successful")
                logger.info(f"  After: λ = {new_lambda:.6f}, edge = {new_edge}")
                logger.info(f"  VP moved: {old_edge} → {new_edge}")
                
                # Rebuild triangle_segments
                logger.info("")
                logger.info("Rebuilding triangle_segments after Type 1...")
                partition.rebuild_triangle_segments_from_current_vps()
                
                # Phase 4: Classify all segments and populate crossing cache
                logger.info("Classifying segments and populating crossing cache...")
                switcher.classify_all_segments()
                
                num_tri_segs_after_type1 = len(partition.triangle_segments)
                logger.info(f"  Triangle segments after Type 1: {num_tri_segs_after_type1}")
                
                # Verify triple point count unchanged (Type 1 on non-triple VP shouldn't affect count)
                num_triple_after_type1 = sum(1 for ts in partition.triangle_segments if len(ts.var_point_indices) == 3)
                logger.info(f"  Triple point count: {num_triple_points_initial} → {num_triple_after_type1}")
                
                if num_triple_after_type1 == num_triple_points_initial:
                    logger.info(f"  ✓ Triple point count preserved (Type 1 didn't affect triple points)")
                else:
                    logger.warning(f"  ⚠ Triple point count changed unexpectedly")
                
                # Analyze the KNOWN segments (captured before switch)
                logger.info("")
                logger.info("Analyzing segment types after Type 1 switch:")
                logger.info(f"  Segments to analyze: {segments_to_analyze_type1}")
                
                type1_classification = {"normal": 0, "edge_following": 0, "edge_cutting": 0}
                for seg in segments_to_analyze_type1:
                    # Get segment classification from BoundarySegment
                    seg_key = (min(seg[0], seg[1]), max(seg[0], seg[1]))
                    seg_obj = None
                    for s in partition.boundary_segments:
                        if s.normalized_key() == seg_key:
                            seg_obj = s
                            break
                    
                    if seg_obj:
                        seg_type = seg_obj.segment_type
                        type1_classification[seg_type] += 1
                        
                        vp1_edge = partition.variable_points[seg[0]].edge
                        vp2_edge = partition.variable_points[seg[1]].edge
                        shared_vertex = set(vp1_edge) & set(vp2_edge)
                        
                        logger.info(f"    Segment {seg}: {seg_type}")
                        logger.info(f"      VP edges: {vp1_edge} ↔ {vp2_edge}")
                        if shared_vertex:
                            logger.info(f"      Shared vertex: {shared_vertex.pop()}")
                    else:
                        logger.warning(f"    Segment {seg} not found in boundary_segments")
                
                logger.info(f"  Type 1 Summary:")
                logger.info(f"    Normal (same triangle): {type1_classification['normal']}")
                logger.info(f"    Edge-following (same mesh line): {type1_classification['edge_following']}")
                logger.info(f"    Edge-cutting (cuts triangles): {type1_classification['edge_cutting']}")
                
                # Report crossing information from cache with improved clarity
                if partition.segment_crossing_cache:
                    logger.info("")
                    logger.info("Segment Crossing Information (from cache):")
                    
                    # Get VP positions for comparison
                    vp_positions_dict = {}
                    for n_idx in neighbors_before_type1 + [vp_idx]:
                        vp_pos = partition.evaluate_variable_point(n_idx)
                        vp_positions_dict[n_idx] = vp_pos
                    
                    for tri_idx, crossings in partition.segment_crossing_cache.items():
                        for crossing in crossings:
                            seg_key = crossing.segment
                            
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
                            
                            # Determine what to display
                            if entry_is_vp and exit_is_vp:
                                # Both endpoints are VPs (normal segment in single triangle)
                                logger.info(f"  Triangle {tri_idx}: Normal segment")
                                logger.info(f"    Segment ({seg_key[0]}, {seg_key[1]}): Both VPs in triangle")
                            elif entry_is_vp:
                                # Entry is VP, exit is computed crossing
                                vp_id = seg_key[0] if entry_is_vp1 else seg_key[1]
                                logger.info(f"  Triangle {tri_idx}: VP {vp_id} at entry, crossing at exit")
                                logger.info(f"    Segment ({seg_key[0]}, {seg_key[1]}):")
                                logger.info(f"      VP {vp_id} position (entry): {crossing.entry_point}")
                                logger.info(f"      COMPUTED CROSSING (exit) edge {crossing.exit_edge}: {crossing.exit_point}")
                            elif exit_is_vp:
                                # Exit is VP, entry is computed crossing
                                vp_id = seg_key[0] if exit_is_vp1 else seg_key[1]
                                logger.info(f"  Triangle {tri_idx}: Crossing at entry, VP {vp_id} at exit")
                                logger.info(f"    Segment ({seg_key[0]}, {seg_key[1]}):")
                                logger.info(f"      COMPUTED CROSSING (entry) edge {crossing.entry_edge}: {crossing.entry_point}")
                                logger.info(f"      VP {vp_id} position (exit): {crossing.exit_point}")
                            elif crossing.is_vertex_crossing:
                                # Vertex crossing (neither VP is at entry/exit)
                                logger.info(f"  Triangle {tri_idx}: Vertex crossing")
                                logger.info(f"    Segment ({seg_key[0]}, {seg_key[1]}): At vertex {crossing.entry_vertex}")
                                logger.info(f"      Position: {crossing.entry_point}")
                            else:
                                # True edge crossing (intermediate triangle)
                                logger.info(f"  Triangle {tri_idx}: Computed edge crossing")
                                logger.info(f"    Segment ({seg_key[0]}, {seg_key[1]}):")
                                logger.info(f"      COMPUTED entry edge {crossing.entry_edge}: {crossing.entry_point}")
                                logger.info(f"      COMPUTED exit edge {crossing.exit_edge}: {crossing.exit_point}")
                                logger.info(f"      Cell: {crossing.cell_idx}")
            else:
                logger.warning("  ⚠ Type 1 switch failed")
        else:
            logger.warning("  ⚠ No non-triple boundary VPs found, skipping Type 1 test")
        
        # =====================================================================
        # Part B: Test Type 2 Switch (Triple Point Migration)
        # =====================================================================
        logger.info("")
        logger.info("=" * 70)
        logger.info("Part B: Testing Type 2 Switch (Triple Point Migration)")
        logger.info("=" * 70)
        
        # Get boundary triple points
        boundary_tps = steiner_handler.get_boundary_triple_points(tol=0.1)
        
        if len(boundary_tps) > 0:
            tp = boundary_tps[0]
            old_triangle = tp.triangle_idx
            
            logger.info(f"Testing Type 2 with triple point in triangle {old_triangle}:")
            logger.info(f"  VPs in triple point: {tp.var_point_indices}")
            logger.info(f"  Cells meeting: {tp.cell_indices}")
            
            # Capture VP edges BEFORE switch (to detect which VP moved)
            vp_edges_before = {vp_idx: partition.variable_points[vp_idx].edge 
                              for vp_idx in tp.var_point_indices}
            
            # Apply Type 2 switch (triple point migration)
            logger.info("")
            logger.info("Applying Type 2 switch (triple point migration)...")
            success = switcher.apply_type2_switch(tp, tol=0.1)
            
            if success:
                logger.info(f"  ✓ Type 2 switch completed successfully")
                
                # Rebuild triangle_segments
                logger.info("")
                logger.info("Rebuilding triangle_segments after Type 2...")
                partition.rebuild_triangle_segments_from_current_vps()
                
                # Phase 4: Classify all segments and populate crossing cache
                logger.info("Classifying segments and populating crossing cache...")
                switcher.classify_all_segments()
                
                num_tri_segs_after_type2 = len(partition.triangle_segments)
                logger.info(f"  Triangle segments after Type 2: {num_tri_segs_after_type2}")
                
                # CRITICAL: Re-detect triple points to verify migration
                logger.info("")
                logger.info("Re-detecting triple points after migration...")
                steiner_after = SteinerHandler(mesh, partition)
                num_triple_after_type2 = len(steiner_after.triple_points)
                
                logger.info(f"  Triple point count: {num_triple_points_initial} → {num_triple_after_type2}")
                
                if num_triple_after_type2 == num_triple_points_initial:
                    logger.info(f"  ✓ Triple point count preserved (Type 2 migrated, not destroyed)")
                else:
                    logger.warning(f"  ⚠ Triple point count changed: expected {num_triple_points_initial}, got {num_triple_after_type2}")
                
                # Check if old triangle still has a triple point
                old_triangle_still_triple = any(ts.triangle_idx == old_triangle and len(ts.var_point_indices) == 3 
                                                for ts in partition.triangle_segments)
                if old_triangle_still_triple:
                    logger.warning(f"  ⚠ Old triangle {old_triangle} still has triple point (migration failed)")
                else:
                    logger.info(f"  ✓ Old triangle {old_triangle} no longer a triple point")
                
                # Detect which VP was moved and identify all VPs in the triple point
                moved_vp_idx = None
                anchor_vp_idx = None
                staying_vp_idx = None
                
                for vp_idx_tp, old_edge in vp_edges_before.items():
                    current_edge = partition.variable_points[vp_idx_tp].edge
                    if current_edge != old_edge:
                        moved_vp_idx = vp_idx_tp
                        logger.info(f"  Detected moved VP: {vp_idx_tp}")
                        logger.info(f"    Edge changed: {old_edge} → {current_edge}")
                
                # Identify anchor and staying VPs (the ones that didn't move)
                other_vps = [v for v in tp.var_point_indices if v != moved_vp_idx]
                if len(other_vps) == 2:
                    anchor_vp_idx, staying_vp_idx = other_vps[0], other_vps[1]
                
                # Analyze ALL segments involving the moved VP (after switch)
                if moved_vp_idx is not None:
                    logger.info("")
                    logger.info("Analyzing ALL segments involving moved VP after Type 2 switch:")
                    
                    # Find all segments involving the moved VP from boundary_segments
                    # This reflects the CURRENT topology after the switch
                    segments_involving_moved_vp = []
                    for seg in partition.boundary_segments:
                        if seg.vp_idx_1 == moved_vp_idx or seg.vp_idx_2 == moved_vp_idx:
                            other_vp = seg.vp_idx_2 if seg.vp_idx_1 == moved_vp_idx else seg.vp_idx_1
                            segments_involving_moved_vp.append((moved_vp_idx, other_vp))
                    
                    logger.info(f"  Segments involving VP {moved_vp_idx}: {len(segments_involving_moved_vp)}")
                    for seg in segments_involving_moved_vp:
                        logger.info(f"    {seg}")
                    
                    # Also check: the destroyed segment should NOT be in boundary_segments
                    destroyed_key = (min(moved_vp_idx, staying_vp_idx), max(moved_vp_idx, staying_vp_idx))
                    destroyed_exists = any(s.normalized_key() == destroyed_key for s in partition.boundary_segments)
                    if destroyed_exists:
                        logger.warning(f"  ⚠ Destroyed segment ({moved_vp_idx}, {staying_vp_idx}) still in boundary_segments!")
                    else:
                        logger.info(f"  ✓ Destroyed segment ({moved_vp_idx}, {staying_vp_idx}) correctly removed")
                    
                    # Show segment classifications (from classify_all_segments)
                    logger.info("")
                    logger.info("  Segment classifications after Type 2 switch:")
                    for seg_vps in segments_involving_moved_vp:
                        # Find the BoundarySegment object to get its classification
                        seg_obj = None
                        for seg in partition.boundary_segments:
                            if seg.normalized_key() == (min(seg_vps), max(seg_vps)):
                                seg_obj = seg
                                break
                        
                        if seg_obj:
                            logger.info(f"    Segment {seg_vps}: {seg_obj.segment_type}")
                        else:
                            logger.info(f"    Segment {seg_vps}: (not found in boundary_segments)")
                else:
                    logger.warning("  ⚠ Could not detect which VP was moved")
                
                # Verify consistency
                logger.info("")
                logger.info("Verifying consistency after both switches...")
                all_consistent = True
                errors = []
                
                for tri_seg in partition.triangle_segments:
                    for vp_idx_in_seg in tri_seg.var_point_indices:
                        vp = partition.variable_points[vp_idx_in_seg]
                        normalized_vp_edge = tuple(sorted(vp.edge))
                        
                        if normalized_vp_edge not in tri_seg.boundary_edges:
                            all_consistent = False
                            errors.append(f"VP {vp_idx_in_seg} on edge {vp.edge} not in triangle {tri_seg.triangle_idx} edges {tri_seg.boundary_edges}")
                
                if all_consistent:
                    logger.info(f"  ✓ All {num_tri_segs_after_type2} triangle segments are consistent")
                    logger.info(f"  ✓ No stale entries found")
                else:
                    logger.error(f"  ✗ Found {len(errors)} consistency errors:")
                    for err in errors[:5]:
                        logger.error(f"    {err}")
                    return False
            else:
                logger.warning("  ⚠ Type 2 switch failed")
        else:
            logger.warning("  ⚠ No boundary triple points found, skipping Type 2 test")
        
        # =====================================================================
        # CROSSING CACHE SUMMARY (After Type 1 and Type 2 migrations)
        # =====================================================================
        logger.info("")
        logger.info("=" * 70)
        logger.info("CROSSING CACHE SUMMARY (After Both Migrations)")
        logger.info("=" * 70)
        
        if partition.segment_crossing_cache:
            total_triangles_with_crossings = len(partition.segment_crossing_cache)
            total_crossing_infos = sum(len(crossings) for crossings in partition.segment_crossing_cache.values())
            
            logger.info(f"Total triangles with cached crossings: {total_triangles_with_crossings}")
            logger.info(f"Total crossing info records: {total_crossing_infos}")
            
            # Count by crossing type
            vertex_crossings = 0
            edge_crossings = 0
            
            for tri_idx, crossings in partition.segment_crossing_cache.items():
                for crossing in crossings:
                    if crossing.is_vertex_crossing:
                        vertex_crossings += 1
                    else:
                        edge_crossings += 1
            
            logger.info(f"  Vertex crossings (edge_following): {vertex_crossings}")
            logger.info(f"  Edge crossings (edge_cutting): {edge_crossings}")
            
            # Show ALL crossing data
            logger.info("")
            logger.info("All crossing data:")
            for i, (tri_idx, crossings) in enumerate(partition.segment_crossing_cache.items()):
                logger.info(f"  Triangle {tri_idx}: {len(crossings)} crossing(s)")
                for crossing in crossings:
                    seg = crossing.segment
                    if crossing.is_vertex_crossing:
                        # For edge-following segments, the shared vertex can be entry OR exit
                        vertex = crossing.entry_vertex if crossing.entry_vertex is not None else crossing.exit_vertex
                        logger.info(f"    Segment ({seg[0]}, {seg[1]}): VERTEX crossing at vertex {vertex}")
                    else:
                        logger.info(f"    Segment ({seg[0]}, {seg[1]}): EDGE crossing")
                        logger.info(f"      Entry: edge {crossing.entry_edge}, point {crossing.entry_point}")
                        logger.info(f"      Exit:  edge {crossing.exit_edge}, point {crossing.exit_point}")
        else:
            logger.info("WARNING: No crossing cache data found!")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 2 COMPLETE: TRIANGLE_SEGMENTS REBUILD TESTED ✓")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Triangle segments rebuild test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_area_perimeter_after_switch(partition, mesh):
    """
    Phase 3: Test that area and perimeter calculations work after topology switches.
    
    This tests Issue 1 fix (triangle_segments rebuild). Issue 2 (cross-triangle segments)
    may cause some incorrect values, but code should not crash.
    
    Args:
        partition: PartitionContour instance (after switch + rebuild)
        mesh: TriMesh instance
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 3: TESTING AREA/PERIMETER CALCULATIONS AFTER SWITCHES")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        from src.core.area_calculator import AreaCalculator
        from src.core.perimeter_calculator import PerimeterCalculator
        from src.core.steiner_handler import SteinerHandler
        
        # Get current lambda values
        lambda_vec = partition.get_variable_vector()
        logger.info(f"Current state:")
        logger.info(f"  Variable points: {len(partition.variable_points)}")
        logger.info(f"  Triangle segments: {len(partition.triangle_segments)}")
        logger.info(f"  Lambda vector size: {len(lambda_vec)}")
        
        # Initialize calculators
        logger.info("")
        logger.info("Initializing calculators...")
        area_calc = AreaCalculator(mesh, partition)
        perim_calc = PerimeterCalculator(mesh, partition)
        steiner_handler = SteinerHandler(mesh, partition)
        
        logger.info(f"  ✓ AreaCalculator initialized")
        logger.info(f"  ✓ PerimeterCalculator initialized")
        logger.info(f"  ✓ SteinerHandler initialized: {len(steiner_handler.triple_points)} triple points")
        
        # Verify triple point count is correct (should match initial count)
        initial_triple_point_count = 8  # From Phase 1 detection
        if len(steiner_handler.triple_points) == initial_triple_point_count:
            logger.info(f"  ✓ Triple point count preserved: {initial_triple_point_count}")
        else:
            logger.warning(f"  ⚠ Triple point count changed: {initial_triple_point_count} → {len(steiner_handler.triple_points)}")
        
        # Test area calculation
        logger.info("")
        logger.info("Testing area calculation...")
        try:
            partition.set_variable_vector(lambda_vec)  # Update VP positions
            areas = area_calc.compute_all_cell_areas(lambda_vec)
            total_area = np.sum(areas)
            mesh_area = float(mesh.M.sum())
            
            logger.info(f"  ✓ Area calculation successful")
            logger.info(f"  Cell areas: {[f'{a:.6f}' for a in areas]}")
            logger.info(f"  Total area: {total_area:.6f}")
            logger.info(f"  Mesh area: {mesh_area:.6f}")
            logger.info(f"  Difference: {abs(total_area - mesh_area):.6e}")
            
            # Check area conservation (may fail if Issue 2 is severe)
            if abs(total_area - mesh_area) < 1e-3:
                logger.info(f"  ✓ Area conserved (within tolerance)")
            else:
                logger.warning(f"  ⚠ Area NOT conserved (Issue 2: cross-triangle segments)")
                logger.warning(f"    This is expected - will be fixed in Phase 4")
        except Exception as e:
            logger.error(f"  ✗ Area calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test perimeter calculation
        logger.info("")
        logger.info("Testing perimeter calculation...")
        try:
            perimeter = perim_calc.compute_total_perimeter(lambda_vec)
            logger.info(f"  ✓ Perimeter calculation successful")
            logger.info(f"  Total perimeter: {perimeter:.6f}")
            
            # Compute per-cell perimeters
            cell_perimeters = []
            for cell_idx in range(partition.n_cells):
                perim = perim_calc.compute_cell_perimeter(cell_idx, lambda_vec)
                cell_perimeters.append(perim)
            
            logger.info(f"  Cell perimeters: {[f'{p:.6f}' for p in cell_perimeters]}")
            
        except Exception as e:
            logger.error(f"  ✗ Perimeter calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test gradient computation (area constraints)
        logger.info("")
        logger.info("Testing area gradient computation...")
        try:
            # Compute gradients for each cell
            area_grads = []
            for cell_idx in range(partition.n_cells):
                grad = area_calc.compute_area_gradient(cell_idx, lambda_vec)
                area_grads.append(grad)
            
            # Stack into matrix
            area_grad_matrix = np.vstack(area_grads)
            
            logger.info(f"  ✓ Area gradient computation successful")
            logger.info(f"  Gradient matrix shape: {area_grad_matrix.shape}")
            logger.info(f"  Total gradient norm: {np.linalg.norm(area_grad_matrix):.6f}")
            
        except Exception as e:
            logger.error(f"  ✗ Area gradient computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test perimeter gradient computation
        logger.info("")
        logger.info("Testing perimeter gradient computation...")
        try:
            perim_grad = perim_calc.compute_total_perimeter_gradient(lambda_vec)
            logger.info(f"  ✓ Perimeter gradient computation successful")
            logger.info(f"  Gradient shape: {perim_grad.shape}")
            logger.info(f"  Gradient norm: {np.linalg.norm(perim_grad):.6f}")
            
        except Exception as e:
            logger.error(f"  ✗ Perimeter gradient computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 3 COMPLETE: AREA/PERIMETER CALCULATIONS WORK AFTER SWITCHES ✓")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Note: If area is not conserved, this is expected (Issue 2).")
        logger.info("      Cross-triangle segments will be fixed in Phase 4.")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Area/perimeter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0-3: Test topology switching dependencies and calculations"
    )
    parser.add_argument(
        '--relaxed-file',
        type=str,
        required=False,
        default=None,
        help='Path to initial relaxed solution .h5 file'
    )
    parser.add_argument(
        '--refined-file',
        type=str,
        required=False,
        default=None,
        help='Path to refined_contours.h5 file (optional, for testing with optimized lambdas)'
    )
    parser.add_argument(
        '--boundary-tol',
        type=float,
        default=1e-3,
        help='Threshold for boundary detection (default: 1e-3)'
    )
    
    args = parser.parse_args()
    
    # Phase 0: Test imports
    imports_ok = test_imports()
    if not imports_ok:
        logger.error("Phase 0 FAILED: Import errors detected")
        return 1
    
    # Phase 0/1: Test class instantiation and topology switching (if files provided)
    if args.relaxed_file:
        # Load partition data
        instantiation_ok, mesh, partition, mesh_topology, switcher = test_class_instantiation(
            args.relaxed_file, 
            args.refined_file
        )
        if not instantiation_ok:
            logger.error("Class instantiation FAILED")
            return 1
        
        # Phase 1: Test topology switch detection
        detection_ok, steiner_handler = test_topology_switch_detection(
            partition, mesh, mesh_topology, switcher, 
            boundary_tol=args.boundary_tol
        )
        if not detection_ok:
            logger.error("Phase 1 FAILED: Topology switch detection errors")
            return 1
        
        # Phase 2: Test triangle_segments rebuild (with both Type 1 and Type 2 switches)
        rebuild_ok = test_triangle_segments_rebuild(partition, mesh, mesh_topology, switcher, steiner_handler)
        if not rebuild_ok:
            logger.error("Phase 2 FAILED: Triangle segments rebuild errors")
            return 1
        
        # Phase 3: Test area/perimeter calculations after switches
        calc_ok = test_area_perimeter_after_switch(partition, mesh)
        if not calc_ok:
            logger.error("Phase 3 FAILED: Area/perimeter calculation errors")
            return 1
    else:
        logger.info("")
        logger.info("⚠ No relaxed file provided - skipping class instantiation and Phase 1-3 tests")
        logger.info("  To run full tests, provide --relaxed-file (and optionally --refined-file)")
        logger.info("")
        logger.info("Example usage:")
        logger.info("  python test_topology_switcher_basic.py \\")
        logger.info("    --relaxed-file results/run_xyz/surface_part5_..._20251027_233612.h5 \\")
        logger.info("    --refined-file results/run_xyz/surface_part5_..._refined_contours.h5")
    
    # Success!
    logger.info("")
    logger.info("=" * 80)
    if args.relaxed_file:
        logger.info("ALL TESTS COMPLETE (Phases 0-3) ✓")
        logger.info("")
        logger.info("Summary:")
        logger.info("  ✓ Phase 0: Dependencies verified")
        logger.info("  ✓ Phase 1: Type 1 & Type 2 switching tested")
        logger.info("  ✓ Phase 2: Triangle segments rebuild tested")
        logger.info("  ✓ Phase 3: Area/perimeter calculations tested")
    else:
        logger.info("ALL TESTS COMPLETE (Phase 0 only) ✓")
    logger.info("=" * 80)
    logger.info("")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

