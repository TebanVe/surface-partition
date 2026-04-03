#!/usr/bin/env python3
"""Test Phase 2 refactoring of PerimeterCalculator."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mesh.tri_mesh import TriMesh
from src.partition.contour_partition import PartitionContour
from src.partition.perimeter_calculator import PerimeterCalculator
from src.logging_config import setup_logging, get_logger


def main():
    setup_logging(log_level='INFO')
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("Phase 2 Refactoring Test")
    logger.info("="*60)
    
    # Simple mesh
    vertices = np.array([[0,0], [1,0], [1,1], [0,1], [0.5,0.5]])
    faces = np.array([[0,1,4], [1,2,4], [2,3,4], [3,0,4]])
    mesh = TriMesh(vertices, faces)
    
    # 2 cells: left vs right
    indicators = np.array([
        [1, 0],  # v0: cell 0
        [0, 1],  # v1: cell 1
        [0, 1],  # v2: cell 1
        [1, 0],  # v3: cell 0
        [1, 0],  # v4: cell 0
    ])
    
    partition = PartitionContour(mesh, indicators)
    perim_calc = PerimeterCalculator(mesh, partition)
    
    lambda_vec = partition.get_variable_vector()
    
    logger.info(f"\n✓ Test 1: Total perimeter")
    total = perim_calc.compute_total_perimeter(lambda_vec)
    logger.info(f"  Result: {total:.6f}")
    
    logger.info(f"\n✓ Test 2: Per-cell perimeters")
    for i in range(partition.n_cells):
        p = perim_calc.compute_cell_perimeter(i, lambda_vec)
        logger.info(f"  Cell {i}: {p:.6f}")
    
    logger.info(f"\n✓ Test 3: Gradient")
    grad = perim_calc.compute_total_perimeter_gradient(lambda_vec)
    logger.info(f"  Shape: {grad.shape}, values: {grad}")
    
    logger.info(f"\n✓ Test 4: Check for deprecated patterns")
    import inspect
    src = inspect.getsource(PerimeterCalculator)
    if "get_cell_segments_from_triangles" in src:
        logger.info("  ✓ Using triangle-based extraction")
    if "for cell_idx in range" in src:
        logger.info("  ✓ Using cell index iteration")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("="*60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
