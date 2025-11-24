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

