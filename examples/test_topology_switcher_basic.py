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
        
        # Create partition with boundary topology
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
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Topology switch detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_triangle_segments_rebuild(partition, mesh, switcher):
    """
    Phase 2: Test that triangle_segments rebuild correctly after topology switches.
    
    Args:
        partition: PartitionContour instance
        mesh: TriMesh instance
        switcher: TopologySwitcher instance
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 2: TESTING TRIANGLE_SEGMENTS REBUILD")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Record initial state
        num_vps_before = len(partition.variable_points)
        num_tri_segs_before = len(partition.triangle_segments)
        
        logger.info(f"Initial state:")
        logger.info(f"  Variable points: {num_vps_before}")
        logger.info(f"  Triangle segments: {num_tri_segs_before}")
        
        # Find a VP that's close to a vertex (if any)
        boundary_vps = partition.get_boundary_variable_points(tol=0.1)
        
        if len(boundary_vps) == 0:
            # Manually create a boundary condition for testing
            logger.info("")
            logger.info("No natural boundary VPs found - creating one manually for testing...")
            test_vp_idx = 10  # Arbitrary choice
            vp = partition.variable_points[test_vp_idx]
            old_lambda = vp.lambda_param
            old_edge = vp.edge
            
            # Set it very close to vertex
            vp.lambda_param = 0.001
            logger.info(f"  Set VP {test_vp_idx}: λ = {old_lambda:.3f} → 0.001")
            
            boundary_vps = [test_vp_idx]
        
        if len(boundary_vps) > 0:
            vp_idx = boundary_vps[0]
            vp = partition.variable_points[vp_idx]
            
            # Save old values BEFORE switch (not references!)
            old_edge = vp.edge
            old_lambda = vp.lambda_param
            
            logger.info("")
            logger.info(f"Testing with VP {vp_idx}:")
            logger.info(f"  Before: λ = {old_lambda:.6f}, edge = {old_edge}")
            
            # Apply Type 1 switch
            logger.info("")
            logger.info("Applying Type 1 switch...")
            success = switcher.apply_type1_switch(vp_idx, tol=0.1)
            
            if success:
                # Get new values after switch
                new_edge = vp.edge
                new_lambda = vp.lambda_param
                
                logger.info(f"  ✓ Switch successful")
                logger.info(f"  After: λ = {new_lambda:.6f}, edge = {new_edge}")
                
                if new_edge != old_edge:
                    logger.info(f"  ✓ VP moved to new edge: {old_edge} → {new_edge}")
                else:
                    logger.warning(f"  ⚠ VP stayed on same edge (no switch occurred)")
                
                # Rebuild triangle_segments
                logger.info("")
                logger.info("Rebuilding triangle_segments...")
                partition.rebuild_triangle_segments_from_current_vps()
                
                num_tri_segs_after = len(partition.triangle_segments)
                logger.info(f"  Triangle segments after rebuild: {num_tri_segs_after}")
                
                # Verify consistency: all VPs in triangle_segments should be on correct edges
                logger.info("")
                logger.info("Verifying consistency...")
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
                    logger.info(f"  ✓ All {num_tri_segs_after} triangle segments are consistent")
                    logger.info(f"  ✓ No stale entries found")
                else:
                    logger.error(f"  ✗ Found {len(errors)} consistency errors:")
                    for err in errors[:5]:  # Show first 5
                        logger.error(f"    {err}")
                    return False
                
            else:
                logger.warning("  ⚠ Type 1 switch failed (could not find suitable edge)")
        else:
            logger.warning("  ⚠ Could not test switch (no boundary VPs available)")
        
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


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0/1: Test topology switching dependencies and detection"
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
        detection_ok = test_topology_switch_detection(
            partition, mesh, mesh_topology, switcher, 
            boundary_tol=args.boundary_tol
        )
        if not detection_ok:
            logger.error("Phase 1 FAILED: Topology switch detection errors")
            return 1
        
        # Phase 2: Test triangle_segments rebuild
        rebuild_ok = test_triangle_segments_rebuild(partition, mesh, switcher)
        if not rebuild_ok:
            logger.error("Phase 2 FAILED: Triangle segments rebuild errors")
            return 1
    else:
        logger.info("")
        logger.info("⚠ No relaxed file provided - skipping class instantiation and Phase 1 tests")
        logger.info("  To run full tests, provide --relaxed-file (and optionally --refined-file)")
        logger.info("")
        logger.info("Example usage:")
        logger.info("  python test_topology_switcher_basic.py \\")
        logger.info("    --relaxed-file results/run_xyz/surface_part5_..._20251027_233612.h5 \\")
        logger.info("    --refined-file results/run_xyz/surface_part5_..._refined_contours.h5")
    
    # Success!
    logger.info("")
    logger.info("=" * 80)
    logger.info("ALL TESTS COMPLETE ✓")
    logger.info("=" * 80)
    logger.info("")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

