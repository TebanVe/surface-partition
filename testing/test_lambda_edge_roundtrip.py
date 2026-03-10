"""
Test 5: Lambda-Edge Roundtrip Consistency
=========================================
Status: 🔄 UNDER INVESTIGATION

Objective:
    Verify that lambda parameters saved to an .h5 file can be correctly reloaded
    and matched to the same edges they were associated with in memory.

    After topology migrations, VPs may be moved to different edges. The in-memory
    VP list order may diverge from the sorted-edge order used during reconstruction
    from indicator_functions. If lambda_parameters are saved by list index but
    reloaded by sorted-edge index, the lambdas get applied to wrong VPs.

What it validates:
    1. Whether the set of edges in the saved file matches the reconstructed set
    2. Whether the lambda-to-edge mapping is preserved through save/load
    3. The perimeter impact of any lambda misassignment

Usage:
    python testing/test_lambda_edge_roundtrip.py \\
        --solution results/run_xyz/*_iteration2_refined_contours.h5

Author: Perimeter Refinement Testing
Date: March 2026
"""

import os
import sys
import argparse
import numpy as np
import h5py

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

from examples.data_loader import load_partition_from_refined_file
from src.find_contours import ContourAnalyzer
from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour


def main():
    parser = argparse.ArgumentParser(
        description='Test lambda-edge roundtrip consistency for refined contours files'
    )
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to refined contours .h5 file')
    args = parser.parse_args()

    refined_path = args.solution
    print("=" * 80)
    print("TEST: Lambda-Edge Roundtrip Consistency")
    print("=" * 80)
    print(f"File: {refined_path}")
    print()

    # =========================================================================
    # Step 1: Read raw data from the .h5 file
    # =========================================================================
    print("STEP 1: Reading raw data from .h5 file...")
    with h5py.File(refined_path, 'r') as f:
        saved_lambdas = f['lambda_parameters'][:]
        has_indicators = 'indicator_functions' in f
        if has_indicators:
            saved_indicators = f['indicator_functions'][:]
        has_edges = 'variable_points' in f
        if has_edges:
            saved_edges = []
            vp_grp = f['variable_points']
            for i in range(len(saved_lambdas)):
                vp_sub = vp_grp[f'vp_{i}']
                edge = (int(vp_sub.attrs['edge_start']), int(vp_sub.attrs['edge_end']))
                saved_edges.append(tuple(sorted(edge)))

    print(f"  Saved lambda_parameters: {len(saved_lambdas)} values")
    print(f"  Has indicator_functions: {has_indicators}")
    print(f"  Has variable_points (edge info): {has_edges}")
    print(f"  Lambda range: [{saved_lambdas.min():.6f}, {saved_lambdas.max():.6f}]")
    print(f"  Lambdas at exactly 0.5: {np.sum(saved_lambdas == 0.5)}")
    print(f"  Lambdas within 0.01 of 0.5: {np.sum(np.abs(saved_lambdas - 0.5) < 0.01)}")
    print()

    if not has_indicators:
        print("ERROR: File does not contain indicator_functions. Cannot test roundtrip.")
        return 1

    # =========================================================================
    # Step 2: Reconstruct partition WITHOUT applying saved lambdas
    # =========================================================================
    print("STEP 2: Reconstructing partition from indicator_functions (no lambda application)...")
    print("  (Building partition from scratch to check VP edge ordering)")

    # Reconstruct the partition topology from indicator_functions only
    temp_analyzer = ContourAnalyzer.__new__(ContourAnalyzer)
    with h5py.File(refined_path, 'r') as f:
        base_path = None
        for attr in ['base_solution_path', 'source_file']:
            if attr in f.attrs:
                base_path = f.attrs[attr]
                break
    
    base_analyzer = ContourAnalyzer(base_path) if base_path else None
    mesh_data = load_partition_from_refined_file(refined_path, verbose=False)
    mesh = mesh_data[0]
    
    temp_analyzer.vertices = mesh.vertices
    temp_analyzer.faces = mesh.faces
    temp_analyzer.densities = saved_indicators
    temp_analyzer.h5_filename = refined_path
    temp_analyzer.logger = __import__('logging').getLogger('test_roundtrip')
    
    _, boundary_topology = temp_analyzer.extract_contours_with_topology()
    partition_fresh = PartitionContour(mesh, saved_indicators, boundary_topology=boundary_topology)
    
    reconstructed_edges = [vp.edge for vp in partition_fresh.variable_points]

    print(f"  Reconstructed partition: {len(partition_fresh.variable_points)} VPs")
    print(f"  These VPs are in sorted-edge order with default λ=0.5")
    print()

    # =========================================================================
    # Step 3: Check edge ordering — saved vs reconstructed
    # =========================================================================
    print("STEP 3: Checking edge ordering (saved vs sorted-edge)...")

    has_saved_edges = 'vp_edges' in h5py.File(refined_path, 'r')
    
    if has_saved_edges:
        with h5py.File(refined_path, 'r') as f:
            saved_edge_array = f['vp_edges'][:]
        saved_edge_list = [(int(saved_edge_array[i, 0]), int(saved_edge_array[i, 1])) 
                          for i in range(len(saved_edge_array))]
        
        n_order_match = 0
        n_order_mismatch = 0
        first_mismatch = None
        for i in range(min(len(saved_edge_list), len(reconstructed_edges))):
            if saved_edge_list[i] == reconstructed_edges[i]:
                n_order_match += 1
            else:
                n_order_mismatch += 1
                if first_mismatch is None:
                    first_mismatch = i
        
        print(f"  File contains vp_edges dataset.")
        print(f"  Edges in same position: {n_order_match}")
        print(f"  Edges in different position: {n_order_mismatch}")
        if first_mismatch is not None:
            print(f"  First mismatch at index {first_mismatch}:")
            print(f"    Saved edge:         {saved_edge_list[first_mismatch]}")
            print(f"    Reconstructed edge: {reconstructed_edges[first_mismatch]}")
        
        if n_order_mismatch > 0:
            print(f"\n  *** ORDERING DIVERGENCE: {n_order_mismatch}/{len(saved_edge_list)} VPs are at different indices ***")
            print(f"  Without edge-keyed loading, {n_order_mismatch} lambdas would be applied to wrong edges.")
    elif has_edges:
        print("  File contains variable_points group (legacy format).")
        n_order_match = 0
        n_order_mismatch = 0
        first_mismatch = None
        for i in range(min(len(saved_edges), len(reconstructed_edges))):
            if saved_edges[i] == reconstructed_edges[i]:
                n_order_match += 1
            else:
                n_order_mismatch += 1
                if first_mismatch is None:
                    first_mismatch = i
        print(f"  Edges in same position: {n_order_match}")
        print(f"  Edges in different position: {n_order_mismatch}")
    else:
        print("  File does NOT contain edge info.")
        print("  Cannot verify edge ordering — lambda assignment correctness is unknown.")
    print()

    # =========================================================================
    # Step 4: Verify edge-keyed loading works correctly
    # =========================================================================
    print("STEP 4: Testing edge-keyed lambda loading...")
    
    mesh_loaded, partition_loaded = load_partition_from_refined_file(refined_path, verbose=True)
    loaded_lambdas = partition_loaded.get_variable_vector()
    loaded_edges = [vp.edge for vp in partition_loaded.variable_points]
    
    if has_saved_edges:
        # Build edge-to-lambda from saved data for ground truth
        edge_to_saved_lambda = {}
        for i in range(len(saved_lambdas)):
            edge_key = (int(saved_edge_array[i, 0]), int(saved_edge_array[i, 1]))
            edge_to_saved_lambda[edge_key] = saved_lambdas[i]
        
        # Check each loaded VP has the correct lambda for its edge
        n_correct = 0
        n_wrong = 0
        for vp in partition_loaded.variable_points:
            expected = edge_to_saved_lambda.get(vp.edge)
            if expected is not None and abs(vp.lambda_param - expected) < 1e-15:
                n_correct += 1
            else:
                n_wrong += 1
        
        print(f"\n  Edge-keyed verification: {n_correct} correct, {n_wrong} wrong")
        lambda_match = (n_wrong == 0)
    else:
        print(f"\n  No vp_edges in file — cannot verify edge-keyed correctness")
        print(f"  Falling back to index-based comparison (always passes trivially)")
        lambda_match = True
    print()

    # =========================================================================
    # Step 5: Perimeter comparison
    # =========================================================================
    print("STEP 5: Perimeter analysis...")

    from src.core.perimeter_calculator import PerimeterCalculator
    perim_calc = PerimeterCalculator(mesh_loaded, partition_loaded)

    loaded_perimeter = perim_calc.compute_total_perimeter(loaded_lambdas)

    # What perimeter would be if all lambdas were 0.5 (default)?
    default_lambdas = np.full_like(loaded_lambdas, 0.5)
    partition_loaded.set_variable_vector(default_lambdas)
    default_perimeter = perim_calc.compute_total_perimeter(default_lambdas)

    # Restore loaded lambdas
    partition_loaded.set_variable_vector(loaded_lambdas)

    print(f"  Perimeter with loaded lambdas:  {loaded_perimeter:.6f}")
    print(f"  Perimeter with all λ=0.5:       {default_perimeter:.6f}")

    # Check a few VPs to see if lambda actually changes their position
    print()
    print("  VP position sensitivity check (first 5 non-0.5 VPs):")
    count = 0
    for i, vp in enumerate(partition_loaded.variable_points):
        lam = loaded_lambdas[i]
        if abs(lam - 0.5) > 0.01 and count < 5:
            v_start = mesh_loaded.vertices[vp.edge[0]]
            v_end = mesh_loaded.vertices[vp.edge[1]]
            pos_at_lambda = lam * v_start + (1 - lam) * v_end
            pos_at_half = 0.5 * v_start + 0.5 * v_end
            dist = np.linalg.norm(pos_at_lambda - pos_at_half)
            edge_len = np.linalg.norm(v_start - v_end)
            print(f"    VP {i}: edge={vp.edge}, λ={lam:.6f}, "
                  f"edge_len={edge_len:.6f}, pos_shift={dist:.6f}")
            count += 1

    # =========================================================================
    # Step 6: Boundary segments analysis
    # =========================================================================
    print()
    print("STEP 6: Boundary segments topology...")
    print(f"  boundary_segments count: {len(partition_loaded.boundary_segments)}")
    print(f"  triangle_segments count: {len(partition_loaded.triangle_segments)}")
    
    seg_per_cell = {}
    for seg in partition_loaded.boundary_segments:
        for c in seg.cell_pair:
            seg_per_cell[c] = seg_per_cell.get(c, 0) + 1
    print(f"  Segments per cell: {dict(sorted(seg_per_cell.items()))}")
    
    invalid_pairs = 0
    zero_pairs = 0
    for seg in partition_loaded.boundary_segments:
        if seg.cell_pair == (0, 0):
            zero_pairs += 1
        if seg.cell_pair[0] == seg.cell_pair[1]:
            invalid_pairs += 1
    print(f"  Segments with cell_pair (0,0): {zero_pairs}")
    print(f"  Segments with same-cell pair: {invalid_pairs}")
    
    print()
    print("  Per-cell perimeter breakdown:")
    for cell_idx in range(partition_loaded.n_cells):
        cell_perim = perim_calc.compute_cell_perimeter(cell_idx, loaded_lambdas)
        n_segs = seg_per_cell.get(cell_idx, 0)
        print(f"    Cell {cell_idx}: perimeter={cell_perim:.6f}, segments={n_segs}")
    
    if zero_pairs > 0:
        print()
        print(f"  WARNING: {zero_pairs} segments have cell_pair (0,0)!")
        print(f"  These are misattributed to cell 0 instead of the correct cells.")
        shown = 0
        for seg in partition_loaded.boundary_segments:
            if seg.cell_pair == (0, 0) and shown < 5:
                vp1 = partition_loaded.variable_points[seg.vp_idx_1]
                vp2 = partition_loaded.variable_points[seg.vp_idx_2]
                print(f"    Segment VP{seg.vp_idx_1}({vp1.edge}) ↔ VP{seg.vp_idx_2}({vp2.edge}): "
                      f"cell_pair={seg.cell_pair}")
                shown += 1

    print()
    print("=" * 80)
    if lambda_match:
        print("VERDICT: PASS - Lambda-edge roundtrip is consistent")
        print(f"  Loaded perimeter: {loaded_perimeter:.6f}")
    else:
        print("VERDICT: FAIL - Lambda-edge roundtrip BROKEN")
        print(f"  Loaded perimeter: {loaded_perimeter:.6f}")
        print("  Root cause: VP list order diverges from sorted-edge order after migrations")
        print("  Fix: save vp_edges alongside lambda_parameters, load by edge key")
    print("=" * 80)

    return 0 if lambda_match else 1


if __name__ == '__main__':
    sys.exit(main())
