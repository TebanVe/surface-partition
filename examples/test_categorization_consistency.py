#!/usr/bin/env python3
"""
Test script to validate that both triangle categorization methods give identical results.

This script compares:
- OLD METHOD: vertex_labels-based categorization (use_vp_based=False)
- NEW METHOD: VP-based categorization (use_vp_based=True)

For the initial iteration (before any topology switches), both methods should
produce identical categorization because VPs are placed based on indicator_functions.

Usage:
    python examples/test_categorization_consistency.py --solution results/your_solution.h5
"""

import os
import sys
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from find_contours import ContourAnalyzer
from core.tri_mesh import TriMesh
from core.contour_partition import PartitionContour
from core.area_calculator import AreaCalculator
from logging_config import get_logger, setup_logging


def compare_categorizations(old_calc: AreaCalculator, new_calc: AreaCalculator) -> bool:
    """
    Compare two AreaCalculator instances to verify they have identical categorizations.
    
    Args:
        old_calc: AreaCalculator with vertex_labels method (use_vp_based=False)
        new_calc: AreaCalculator with VP-based method (use_vp_based=True)
        
    Returns:
        True if categorizations match, False otherwise
    """
    logger = get_logger(__name__)
    n_cells = old_calc.partition.n_cells
    all_match = True
    
    logger.info("="*80)
    logger.info("COMPARING TRIANGLE CATEGORIZATIONS")
    logger.info("="*80)
    
    for cell_idx in range(n_cells):
        # Compare interior triangles
        old_interior = set(old_calc.cell_interior_triangles[cell_idx])
        new_interior = set(new_calc.cell_interior_triangles[cell_idx])
        
        # Compare boundary triangles
        old_boundary = set(old_calc.cell_boundary_triangles[cell_idx])
        new_boundary = set(new_calc.cell_boundary_triangles[cell_idx])
        
        # Compare interior areas
        old_area = old_calc.cell_interior_area[cell_idx]
        new_area = new_calc.cell_interior_area[cell_idx]
        
        # Check for differences
        interior_match = (old_interior == new_interior)
        boundary_match = (old_boundary == new_boundary)
        area_match = np.isclose(old_area, new_area, rtol=1e-12)
        
        cell_match = interior_match and boundary_match and area_match
        
        if not cell_match:
            all_match = False
            logger.error(f"\nCell {cell_idx}: MISMATCH DETECTED!")
            
            if not interior_match:
                only_in_old = old_interior - new_interior
                only_in_new = new_interior - old_interior
                logger.error(f"  Interior triangles differ:")
                logger.error(f"    Only in OLD: {sorted(list(only_in_old))[:10]} "
                           f"({'...' if len(only_in_old) > 10 else ''}total: {len(only_in_old)})")
                logger.error(f"    Only in NEW: {sorted(list(only_in_new))[:10]} "
                           f"({'...' if len(only_in_new) > 10 else ''}total: {len(only_in_new)})")
            
            if not boundary_match:
                only_in_old = old_boundary - new_boundary
                only_in_new = new_boundary - old_boundary
                logger.error(f"  Boundary triangles differ:")
                logger.error(f"    Only in OLD: {sorted(list(only_in_old))[:10]} "
                           f"({'...' if len(only_in_old) > 10 else ''}total: {len(only_in_old)})")
                logger.error(f"    Only in NEW: {sorted(list(only_in_new))[:10]} "
                           f"({'...' if len(only_in_new) > 10 else ''}total: {len(only_in_new)})")
            
            if not area_match:
                logger.error(f"  Interior area differs:")
                logger.error(f"    OLD: {old_area:.12f}")
                logger.error(f"    NEW: {new_area:.12f}")
                logger.error(f"    Diff: {abs(old_area - new_area):.2e}")
        else:
            logger.info(f"Cell {cell_idx}: ✓ MATCH "
                       f"(interior: {len(old_interior)}, boundary: {len(old_boundary)})")
    
    logger.info("="*80)
    if all_match:
        logger.info("✓ SUCCESS: All categorizations match!")
    else:
        logger.error("✗ FAILURE: Categorizations differ!")
    logger.info("="*80)
    
    return all_match


def main():
    parser = argparse.ArgumentParser(
        description='Test triangle categorization consistency between old and new methods'
    )
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input solution .h5 file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("Triangle Categorization Consistency Test")
    logger.info("="*80)
    logger.info(f"Solution file: {args.solution}")
    
    # Check input file exists
    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1
    
    # Load solution and extract contours
    logger.info("\nStep 1: Loading solution and extracting contours...")
    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    
    # Compute indicator functions
    indicators = analyzer.compute_indicator_functions()
    
    # Extract contours with topology
    raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
    
    # Build mesh and partition
    logger.info("\nStep 2: Building data structures...")
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    partition = PartitionContour(mesh, indicators, boundary_topology=boundary_topology)
    
    logger.info(f"Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} triangles")
    logger.info(f"Partition: {partition.n_cells} cells, {len(partition.variable_points)} VPs")
    
    # Create two AreaCalculators with different methods
    logger.info("\nStep 3: Creating AreaCalculators with both methods...")
    
    logger.info("  Creating OLD method (vertex_labels)...")
    old_calc = AreaCalculator(mesh, partition, use_vp_based=False)
    
    logger.info("  Creating NEW method (VP-based)...")
    new_calc = AreaCalculator(mesh, partition, use_vp_based=True)
    
    # Compare categorizations
    logger.info("\nStep 4: Comparing categorizations...")
    match = compare_categorizations(old_calc, new_calc)
    
    # Test area calculations
    logger.info("\nStep 5: Testing area calculations...")
    lambda_vec = partition.get_variable_vector()
    partition.set_variable_vector(lambda_vec)
    
    old_areas = old_calc.compute_all_cell_areas(lambda_vec)
    new_areas = new_calc.compute_all_cell_areas(lambda_vec)
    
    areas_match = np.allclose(old_areas, new_areas, rtol=1e-10)
    
    if areas_match:
        logger.info("✓ Area calculations match!")
        logger.info(f"  Max difference: {np.max(np.abs(old_areas - new_areas)):.2e}")
    else:
        logger.error("✗ Area calculations differ!")
        for i in range(len(old_areas)):
            diff = abs(old_areas[i] - new_areas[i])
            if diff > 1e-10:
                logger.error(f"  Cell {i}: OLD={old_areas[i]:.12f}, NEW={new_areas[i]:.12f}, "
                           f"DIFF={diff:.2e}")
    
    # Final result
    logger.info("\n" + "="*80)
    if match and areas_match:
        logger.info("✓✓✓ ALL TESTS PASSED ✓✓✓")
        logger.info("Both categorization methods produce identical results!")
        logger.info("Safe to use VP-based method after topology switches.")
        return 0
    else:
        logger.error("✗✗✗ TESTS FAILED ✗✗✗")
        if not match:
            logger.error("Categorizations differ - VP-based method has bugs!")
        if not areas_match:
            logger.error("Area calculations differ - numerical issues!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

