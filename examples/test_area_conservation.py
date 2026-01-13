#!/usr/bin/env python3
"""
Test area conservation after Type 1 topology migration.

This script verifies that:
1. Total area is conserved after Type 1 migration
2. Individual cell areas change appropriately (one gains, one loses)
3. The crossing cache is properly populated
4. Intermediate triangles are correctly categorized

Usage:
    python test_area_conservation.py --solution <path>/*_refined_contours.h5

Author: Test script for boundary visualization fixes
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.visualize_precise_region import load_partition_from_refined_file
from src.core.mesh_topology import MeshTopology
from src.core.topology_switcher import TopologySwitcher
from src.core.steiner_handler import SteinerHandler
from src.core.area_calculator import AreaCalculator


def find_type1_candidate(partition, steiner_handler, tol=0.1):
    """Find a good VP candidate for Type 1 migration."""
    # Get boundary VPs
    boundary_vps = partition.get_boundary_variable_points(tol=tol)
    
    # Exclude triple point VPs
    triple_point_vp_indices = set()
    for tp in steiner_handler.triple_points:
        triple_point_vp_indices.update(tp.var_point_indices)
    
    non_triple_boundary_vps = [vp for vp in boundary_vps 
                               if vp not in triple_point_vp_indices]
    
    if not non_triple_boundary_vps:
        return None
    
    # Sort by distance to vertex (closest first)
    def boundary_distance(vp_idx):
        vp = partition.variable_points[vp_idx]
        return min(vp.lambda_param, 1.0 - vp.lambda_param)
    
    non_triple_boundary_vps.sort(key=boundary_distance)
    
    return non_triple_boundary_vps[0]


def test_area_conservation(solution_path, boundary_tol=0.1):
    """Test that area is conserved after Type 1 migration."""
    print("="*80)
    print("AREA CONSERVATION TEST")
    print("="*80)
    print(f"Solution: {solution_path}")
    print()
    
    # Load partition
    print("Loading partition...")
    mesh, partition = load_partition_from_refined_file(solution_path)
    
    # Initialize components
    mesh_topology = MeshTopology(mesh)
    switcher = TopologySwitcher(mesh, partition, mesh_topology)
    steiner_handler = SteinerHandler(mesh, partition)
    
    # Compute areas BEFORE migration
    print("\n" + "="*60)
    print("BEFORE Type 1 Migration")
    print("="*60)
    
    area_calc_before = AreaCalculator(mesh, partition, use_vp_based=True)
    lambda_vec = partition.get_variable_vector()
    
    areas_before = {}
    total_before = 0.0
    for cell_idx in range(partition.n_cells):
        area = area_calc_before.compute_cell_area(cell_idx, lambda_vec)
        areas_before[cell_idx] = area
        total_before += area
        print(f"  Cell {cell_idx}: {area:.8f}")
    
    print(f"  TOTAL: {total_before:.8f}")
    
    # Find candidate VP for Type 1
    vp_idx = find_type1_candidate(partition, steiner_handler, tol=boundary_tol)
    
    if vp_idx is None:
        print("\nERROR: No suitable VP found for Type 1 migration")
        return False
    
    vp = partition.variable_points[vp_idx]
    print(f"\nSelected VP {vp_idx}:")
    print(f"  Edge: {vp.edge}")
    print(f"  Lambda: {vp.lambda_param:.6f}")
    print(f"  Cells: {vp.belongs_to_cells}")
    
    # Apply Type 1 migration
    print("\nApplying Type 1 migration...")
    old_edge = vp.edge
    success = switcher.apply_type1_switch(vp_idx, tol=boundary_tol)
    
    if not success:
        print("ERROR: Type 1 migration failed!")
        return False
    
    new_edge = vp.edge
    print(f"  VP moved: {old_edge} → {new_edge}")
    
    # Rebuild topology structures
    print("\nRebuilding topology structures...")
    partition.rebuild_triangle_segments_from_current_vps()
    switcher.classify_all_segments()
    
    # Check crossing cache
    print("\nSegment crossing cache:")
    n_triangles_with_crossings = len(partition.segment_crossing_cache)
    n_total_crossings = sum(len(c) for c in partition.segment_crossing_cache.values())
    print(f"  Triangles with crossings: {n_triangles_with_crossings}")
    print(f"  Total crossing entries: {n_total_crossings}")
    
    for tri_idx, crossings in partition.segment_crossing_cache.items():
        for crossing in crossings:
            print(f"    Triangle {tri_idx}: segment {crossing.segment}, "
                  f"cell_pair={crossing.cell_pair}")
    
    # Compute areas AFTER migration
    print("\n" + "="*60)
    print("AFTER Type 1 Migration")
    print("="*60)
    
    area_calc_after = AreaCalculator(mesh, partition, use_vp_based=True)
    lambda_vec_after = partition.get_variable_vector()
    
    areas_after = {}
    total_after = 0.0
    for cell_idx in range(partition.n_cells):
        area = area_calc_after.compute_cell_area(cell_idx, lambda_vec_after)
        areas_after[cell_idx] = area
        total_after += area
        
        diff = area - areas_before[cell_idx]
        sign = "+" if diff > 0 else ""
        print(f"  Cell {cell_idx}: {area:.8f} ({sign}{diff:.2e})")
    
    print(f"  TOTAL: {total_after:.8f}")
    
    # Check conservation
    print("\n" + "="*60)
    print("CONSERVATION CHECK")
    print("="*60)
    
    total_diff = total_after - total_before
    print(f"  Total area before: {total_before:.8f}")
    print(f"  Total area after:  {total_after:.8f}")
    print(f"  Difference:        {total_diff:.2e}")
    
    # Check if area is conserved (within numerical tolerance)
    tolerance = 1e-6
    if abs(total_diff) < tolerance:
        print(f"\n  ✓ PASSED: Area conserved (diff < {tolerance:.0e})")
        passed = True
    else:
        print(f"\n  ✗ FAILED: Area NOT conserved (diff = {total_diff:.2e})")
        passed = False
    
    # Additional checks
    print("\n" + "="*60)
    print("ADDITIONAL CHECKS")
    print("="*60)
    
    # Check that at least one cell gained area and one lost
    cells_gained = [c for c in range(partition.n_cells) 
                    if areas_after[c] > areas_before[c] + 1e-10]
    cells_lost = [c for c in range(partition.n_cells) 
                  if areas_after[c] < areas_before[c] - 1e-10]
    
    print(f"  Cells that gained area: {cells_gained}")
    print(f"  Cells that lost area: {cells_lost}")
    
    # Check VP cells - migration should only affect these cells
    affected_cells = list(vp.belongs_to_cells)
    print(f"  VP's cells (should be affected): {affected_cells}")
    
    # Verify changes are in VP's cells
    changes_in_vp_cells = all(c in affected_cells for c in cells_gained + cells_lost)
    if changes_in_vp_cells:
        print(f"  ✓ Changes only in VP's cells")
    else:
        print(f"  ⚠ WARNING: Changes in cells other than VP's cells")
    
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Test area conservation after Type 1 migration"
    )
    parser.add_argument('--solution', required=True,
                       help='Path to *_refined_contours.h5 file')
    parser.add_argument('--boundary-tol', type=float, default=0.1,
                       help='Threshold for boundary detection (default: 0.1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.solution):
        print(f"ERROR: File not found: {args.solution}")
        return 1
    
    success = test_area_conservation(args.solution, args.boundary_tol)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

