#!/usr/bin/env python3
"""
Test script for topology switching - Phase 0: Verify Dependencies

This script verifies that all required imports work and basic infrastructure is set up.
Part of the incremental testing plan for topology switching implementation.

Phase 0 Goals:
- Verify all imports work
- Check that MeshTopology and TopologySwitcher can be instantiated
- Ensure dependencies are available

Author: Topology Switching Implementation Team
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


def test_class_instantiation(partition_file: str):
    """
    Phase 0: Test that we can load data and instantiate topology switching classes.
    
    Args:
        partition_file: Path to a saved partition from previous optimization
    """
    logger.info("")
    logger.info("Testing class instantiation...")
    logger.info("")
    
    try:
        from src.core.tri_mesh import TriMesh
        from src.core.contour_partition import PartitionContour
        from src.core.mesh_topology import MeshTopology
        from src.core.topology_switcher import TopologySwitcher
        
        # Load partition from file
        logger.info(f"Loading partition from: {partition_file}")
        partition_data = np.load(partition_file, allow_pickle=True)
        
        # Extract mesh
        logger.info("  Extracting mesh...")
        vertices = partition_data['vertices']
        faces = partition_data['faces']
        mesh = TriMesh(vertices, faces)
        logger.info(f"  ✓ Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
        
        # Extract partition
        logger.info("  Extracting partition...")
        indicator_functions = partition_data['indicator_functions']
        partition = PartitionContour(mesh, indicator_functions)
        logger.info(f"  ✓ Partition loaded: {len(partition.variable_points)} variable points")
        logger.info(f"    - {len(partition.triangle_segments)} triangle segments")
        
        # Test MeshTopology instantiation
        logger.info("  Building MeshTopology...")
        mesh_topology = MeshTopology(mesh)
        logger.info(f"  ✓ MeshTopology created")
        logger.info(f"    - {len(mesh_topology.vertex_to_triangles)} vertices in connectivity map")
        logger.info(f"    - {len(mesh_topology.edge_to_triangles)} edges in connectivity map")
        
        # Test TopologySwitcher instantiation
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


def test_basic_methods(partition, mesh_topology, switcher):
    """
    Phase 0: Test that basic methods can be called without errors.
    
    Args:
        partition: PartitionContour instance
        mesh_topology: MeshTopology instance
        switcher: TopologySwitcher instance
    """
    logger.info("")
    logger.info("Testing basic method calls...")
    logger.info("")
    
    try:
        # Test partition methods
        logger.info("  Testing PartitionContour methods...")
        boundary_vps = partition.get_boundary_variable_points(tol=0.1)
        logger.info(f"  ✓ get_boundary_variable_points() works: {len(boundary_vps)} boundary VPs detected")
        
        # Test mesh_topology methods
        logger.info("  Testing MeshTopology methods...")
        test_vertex = 0
        edges_at_vertex = mesh_topology.get_edges_at_vertex(test_vertex)
        logger.info(f"  ✓ get_edges_at_vertex({test_vertex}) works: {len(edges_at_vertex)} edges found")
        
        if len(edges_at_vertex) > 0:
            test_edge = edges_at_vertex[0]
            tris_sharing_edge = mesh_topology.get_triangles_sharing_edge(test_edge)
            logger.info(f"  ✓ get_triangles_sharing_edge({test_edge}) works: {len(tris_sharing_edge)} triangles found")
        
        # Test switcher methods exist (don't call them yet - that's Phase 1)
        logger.info("  Checking TopologySwitcher methods exist...")
        assert hasattr(switcher, 'apply_type1_switch'), "apply_type1_switch method missing"
        logger.info(f"  ✓ apply_type1_switch method exists")
        
        assert hasattr(switcher, 'select_variable_point_for_type2'), "select_variable_point_for_type2 method missing"
        logger.info(f"  ✓ select_variable_point_for_type2 method exists")
        
        logger.info("")
        logger.info("✓ All basic methods work!")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Method call failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Verify topology switching dependencies"
    )
    parser.add_argument(
        '--partition-file',
        type=str,
        required=False,
        default=None,
        help='Path to saved partition from previous optimization (.npz file). Optional for Phase 0.'
    )
    
    args = parser.parse_args()
    
    # Phase 0.1: Test imports
    imports_ok = test_imports()
    if not imports_ok:
        logger.error("Phase 0 FAILED: Import errors detected")
        return 1
    
    # Phase 0.2 & 0.3: Test class instantiation and methods (only if partition file provided)
    if args.partition_file:
        # Phase 0.2: Test class instantiation
        instantiation_ok, mesh, partition, mesh_topology, switcher = test_class_instantiation(
            args.partition_file
        )
        if not instantiation_ok:
            logger.error("Phase 0 FAILED: Class instantiation errors detected")
            return 1
        
        # Phase 0.3: Test basic methods
        methods_ok = test_basic_methods(partition, mesh_topology, switcher)
        if not methods_ok:
            logger.error("Phase 0 FAILED: Method call errors detected")
            return 1
    else:
        logger.info("")
        logger.info("⚠ No partition file provided - skipping class instantiation tests")
        logger.info("  To run full Phase 0 tests, provide --partition-file argument")
    
    # Success!
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 0 COMPLETE: ALL DEPENDENCIES VERIFIED ✓")
    logger.info("=" * 80)
    logger.info("")
    if args.partition_file:
        logger.info("Ready to proceed to Phase 1: Test Type 1 & Type 2 switching")
    else:
        logger.info("Run with --partition-file to test class instantiation before Phase 1")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

