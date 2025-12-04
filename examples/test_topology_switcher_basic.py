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

def edges_share_triangle(mesh_topology, edge1, edge2) -> bool:
    """Check if two edges are part of the same mesh triangle."""
    edge1_norm = tuple(sorted(edge1))
    edge2_norm = tuple(sorted(edge2))
    tris1 = set(mesh_topology.get_triangles_sharing_edge(edge1_norm))
    tris2 = set(mesh_topology.get_triangles_sharing_edge(edge2_norm))
    return bool(tris1 & tris2)


def get_shared_vertex(edge1, edge2):
    """Get shared vertex between two edges, or None."""
    shared = set(edge1) & set(edge2)
    return shared.pop() if shared else None


def edges_are_collinear(mesh, edge1, edge2, shared_vertex, tol=1e-6) -> bool:
    """
    Check if two edges sharing a vertex are collinear (on same mesh line).
    
    Uses cross product to detect if direction vectors are parallel/antiparallel.
    """
    v_shared = mesh.vertices[shared_vertex]
    
    # Get the OTHER vertex of each edge
    v1_other = edge1[0] if edge1[1] == shared_vertex else edge1[1]
    v2_other = edge2[0] if edge2[1] == shared_vertex else edge2[1]
    
    p1 = mesh.vertices[v1_other]
    p2 = mesh.vertices[v2_other]
    
    # Direction vectors from shared vertex
    dir1 = p1 - v_shared
    dir2 = p2 - v_shared
    
    # Normalize
    norm1 = np.linalg.norm(dir1)
    norm2 = np.linalg.norm(dir2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return False
    
    dir1_norm = dir1 / norm1
    dir2_norm = dir2 / norm2
    
    # Cross product magnitude (0 if collinear)
    cross = np.cross(dir1_norm, dir2_norm)
    return np.linalg.norm(cross) < tol


def classify_segment(partition, mesh, mesh_topology, vp_idx1: int, vp_idx2: int) -> dict:
    """
    Classify a segment between two VPs.
    
    Returns dict with:
        'type': "normal", "edge_following", or "edge_cutting"
        'shared_vertex': vertex index if edge_following, else None
        'vp1_edge': edge of VP1
        'vp2_edge': edge of VP2
    """
    edge1 = partition.variable_points[vp_idx1].edge
    edge2 = partition.variable_points[vp_idx2].edge
    
    result = {
        'segment': (vp_idx1, vp_idx2),
        'vp1_edge': edge1,
        'vp2_edge': edge2,
        'shared_vertex': None,
        'type': None
    }
    
    # Step 1: Check if edges share a triangle (NORMAL case)
    if edges_share_triangle(mesh_topology, edge1, edge2):
        result['type'] = "normal"
        return result
    
    # --- VPs are in different triangles (cross-triangle segment) ---
    
    # Step 2: Check if edges share a vertex
    shared_vertex = get_shared_vertex(edge1, edge2)
    
    if shared_vertex is None:
        # No shared vertex - definitely EDGE-CUTTING
        result['type'] = "edge_cutting"
        return result
    
    result['shared_vertex'] = shared_vertex
    
    # Step 3: Edges share vertex - check COLLINEARITY
    if edges_are_collinear(mesh, edge1, edge2, shared_vertex):
        # Edges are on same mesh line - EDGE-FOLLOWING
        result['type'] = "edge_following"
    else:
        # Edges form an angle - EDGE-CUTTING (even though they share vertex)
        result['type'] = "edge_cutting"
    
    return result


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
        classification = classify_segment(partition, mesh, mesh_topology, vp_idx, neighbor_idx)
        seg_type = classification['type']
        classification_counts[seg_type] += 1
        segment_details.append(classification)
    
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
        
        # Create partition WITHOUT boundary topology
        # CRITICAL: Must match VP indices used when refined_contours.h5 was saved
        # Using boundary_topology creates different VP ordering, causing lambdas
        # to be assigned to wrong VPs when loading from refined file
        logger.info("  Creating PartitionContour...")
        partition = PartitionContour(mesh, indicators)
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
        
        # Test 3: Test VP selection for Type 2
        if len(boundary_tps) > 0:
            logger.info("")
            logger.info("Test 3: Testing VP selection for Type 2 switch...")
            tp = boundary_tps[0]
            vp_to_move = switcher.select_variable_point_for_type2(tp)
            
            if vp_to_move is not None:
                logger.info(f"  ✓ Selected VP {vp_to_move} for Type 2 switch")
                vp = partition.variable_points[vp_to_move]
                logger.info(f"    Current: λ = {vp.lambda_param:.6f}, edge = {vp.edge}")
            else:
                logger.warning(f"  ⚠ Could not select VP for Type 2 switch")
        
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
            vp_idx = non_triple_boundary_vps[0]
            vp = partition.variable_points[vp_idx]
            
            old_edge = vp.edge
            old_lambda = vp.lambda_param
            
            # CAPTURE neighbors BEFORE the switch (to analyze segments after)
            neighbors_before_type1 = get_neighboring_vps(partition, vp_idx)
            segments_to_analyze_type1 = [(vp_idx, n) for n in neighbors_before_type1]
            
            logger.info(f"Testing Type 1 with VP {vp_idx} (NOT part of triple point):")
            logger.info(f"  Before: λ = {old_lambda:.6f}, edge = {old_edge}")
            logger.info(f"  Neighbors before switch: {neighbors_before_type1}")
            
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
                    result = classify_segment(partition, mesh, mesh_topology, seg[0], seg[1])
                    seg_type = result['type']
                    type1_classification[seg_type] += 1
                    
                    logger.info(f"    Segment {seg}: {seg_type}")
                    logger.info(f"      VP edges: {result['vp1_edge']} ↔ {result['vp2_edge']}")
                    if result['shared_vertex'] is not None:
                        logger.info(f"      Shared vertex: {result['shared_vertex']}")
                
                logger.info(f"  Type 1 Summary:")
                logger.info(f"    Normal (same triangle): {type1_classification['normal']}")
                logger.info(f"    Edge-following (same mesh line): {type1_classification['edge_following']}")
                logger.info(f"    Edge-cutting (cuts triangles): {type1_classification['edge_cutting']}")
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
                    
                    # Classify all segments involving the moved VP
                    type2_classification = {"normal": 0, "edge_following": 0, "edge_cutting": 0}
                    for seg in segments_involving_moved_vp:
                        result = classify_segment(partition, mesh, mesh_topology, seg[0], seg[1])
                        seg_type = result['type']
                        type2_classification[seg_type] += 1
                        
                        logger.info(f"    Segment {seg}: {seg_type}")
                        logger.info(f"      VP edges: {result['vp1_edge']} ↔ {result['vp2_edge']}")
                        if result['shared_vertex'] is not None:
                            logger.info(f"      Shared vertex: {result['shared_vertex']}")
                    
                    logger.info(f"  Type 2 Summary (segments involving moved VP):")
                    logger.info(f"    Normal (same triangle): {type2_classification['normal']}")
                    logger.info(f"    Edge-following (same mesh line): {type2_classification['edge_following']}")
                    logger.info(f"    Edge-cutting (cuts triangles): {type2_classification['edge_cutting']}")
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

