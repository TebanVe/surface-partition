#!/usr/bin/env python3
"""
Numerical equivalence tests for the vectorized evaluation modules.

This script constructs a synthetic partition (no .h5 files needed) and verifies
that every vectorized function matches its original object-oriented counterpart
to floating-point tolerance.

Tests:
  1. compile_arrays() round-trips correctly
  2. Vectorized perimeter ≈ PerimeterCalculator.compute_total_perimeter()
  3. Vectorized perimeter gradient ≈ PerimeterCalculator.compute_total_perimeter_gradient()
  4. Vectorized area ≈ AreaCalculator.compute_all_cell_areas()
  5. Vectorized area Jacobian ≈ AreaCalculator.compute_area_jacobian()
  6. Analytical Steiner point ≈ BFGS Steiner point (TriplePoint.compute_steiner_point)
  7. Vectorized Steiner perimeter ≈ SteinerHandler.get_total_perimeter_contribution()
  8. Vectorized Steiner area ≈ SteinerHandler.get_total_area_contribution()
  9. Vectorized Steiner perim gradient ≈ SteinerHandler.compute_total_gradient_finite_difference()
 10. Vectorized Steiner area Jacobian ≈ SteinerHandler.compute_area_gradients_finite_difference()
 11. PerimeterOptimizer vectorized path vs original path (end-to-end objective + constraints)

Usage:
    python testing/test_vectorized_evaluation.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.core.area_calculator import AreaCalculator
from src.core.perimeter_calculator import PerimeterCalculator
from src.core.steiner_handler import SteinerHandler
from src.core.perimeter_optimizer import PerimeterOptimizer
from src.core import vectorized_perimeter, vectorized_area, vectorized_steiner
from src.logging_config import setup_logging, get_logger

setup_logging(log_level='WARNING')
logger = get_logger(__name__)

# =========================================================================
# Synthetic mesh and partition builder
# =========================================================================

def build_test_partition():
    """Build a small planar mesh with 3 cells and at least 1 triple point.

    Layout (schematic — a 2×2 square split into 8 triangles, 3 cells):

        (0,1)----(0.5,1)----(1,1)
          |  \\      |      /  |
          |    \\  4 | 5  /    |
          |  3   \\  |  /   6  |
          |        \\ |/        |
        (0,.5)--(0.5,.5)--(1,.5)
          |        /|\\        |
          |  0   /  |  \\   7  |
          |    / 1  | 2  \\    |
          |  /      |      \\  |
        (0,0)----(0.5,0)----(1,0)

    Vertex labels:
        Bottom-left block  → cell 0
        Top block          → cell 1
        Right block        → cell 2

    This guarantees at least one triple-point triangle (center vertex at (0.5, 0.5)
    touches all 3 cells).
    """
    vertices = np.array([
        [0.0, 0.0],   # 0
        [0.5, 0.0],   # 1
        [1.0, 0.0],   # 2
        [0.0, 0.5],   # 3
        [0.5, 0.5],   # 4  (center — triple point area)
        [1.0, 0.5],   # 5
        [0.0, 1.0],   # 6
        [0.5, 1.0],   # 7
        [1.0, 1.0],   # 8
    ], dtype=np.float64)

    faces = np.array([
        [0, 1, 4],  # 0
        [0, 4, 3],  # 1
        [1, 2, 4],  # 2
        [2, 5, 4],  # 3
        [3, 4, 6],  # 4
        [4, 7, 6],  # 5
        [4, 5, 8],  # 6
        [4, 8, 7],  # 7
    ], dtype=np.int64)

    # Cell assignments: cell 0 = {0,1,3}, cell 1 = {6,7,8}, cell 2 = {2,5}
    # Center vertex 4 must go to one cell — put it in cell 0 so that triangles
    # touching (0,1,2) have 3 different labels → triple point
    n_cells = 3
    indicator = np.zeros((9, n_cells))
    cell_map = {0: 0, 1: 0, 3: 0, 4: 0, 6: 1, 7: 1, 8: 1, 2: 2, 5: 2}
    for v, c in cell_map.items():
        indicator[v, c] = 1.0

    mesh = TriMesh(vertices, faces)
    partition = PartitionContour(mesh, indicator)

    # Perturb lambdas away from 0.5 so gradients are non-trivial
    rng = np.random.RandomState(42)
    for vp in partition.variable_points:
        vp.lambda_param = np.clip(0.5 + rng.uniform(-0.2, 0.2), 0.05, 0.95)
    partition.set_variable_vector(
        np.array([vp.lambda_param for vp in partition.variable_points])
    )

    return mesh, partition


# =========================================================================
# Test helpers
# =========================================================================

_passed = 0
_failed = 0


def check(name, val_vec, ref_vec, atol=1e-10, rtol=1e-10):
    """Compare two values/arrays and report pass/fail."""
    global _passed, _failed
    val = np.asarray(val_vec, dtype=np.float64)
    ref = np.asarray(ref_vec, dtype=np.float64)
    if val.shape != ref.shape:
        print(f"  FAIL  {name}: shape mismatch {val.shape} vs {ref.shape}")
        _failed += 1
        return
    abs_diff = np.max(np.abs(val - ref)) if val.size else 0.0
    denom = max(np.max(np.abs(ref)), 1e-30) if ref.size else 1.0
    rel_diff = abs_diff / denom
    ok = abs_diff <= atol or rel_diff <= rtol
    status = "PASS" if ok else "FAIL"
    if not ok:
        _failed += 1
    else:
        _passed += 1
    print(f"  {status}  {name}  (abs={abs_diff:.2e}, rel={rel_diff:.2e})")


# =========================================================================
# Main
# =========================================================================

def main():
    global _passed, _failed

    mesh, partition = build_test_partition()
    full_vec = partition.get_variable_vector()
    partition.set_variable_vector(full_vec)

    steiner_handler = SteinerHandler(mesh, partition)
    area_calc = AreaCalculator(mesh, partition)
    perim_calc = PerimeterCalculator(mesh, partition)

    # Compile arrays
    pa = partition.compile_arrays(steiner_handler)

    print(f"\nTest partition: {len(partition.variable_points)} VPs, "
          f"{partition.n_cells} cells, {len(steiner_handler.triple_points)} triple points")
    print(f"Compiled arrays: {pa.n_active_vp} active VPs, "
          f"{len(pa.seg_vp1)} segments, {pa.n_triple_points} TPs\n")

    # Sync lambda into arrays (they should already match, but be explicit)
    pa.vp_lambda[:] = partition.get_active_variable_vector()

    # ------------------------------------------------------------------
    # Test 1: compile_arrays round-trip (lambda values)
    # ------------------------------------------------------------------
    print("--- Test 1: compile_arrays round-trip ---")
    for i, abs_idx in enumerate(pa.active_to_absolute):
        check(f"lambda[active={i}]",
              pa.vp_lambda[i],
              partition.variable_points[abs_idx].lambda_param,
              atol=1e-15)

    # ------------------------------------------------------------------
    # Test 2: Total perimeter
    # ------------------------------------------------------------------
    print("\n--- Test 2: Vectorized perimeter ---")
    ref_perim = perim_calc.compute_total_perimeter(full_vec)
    vec_perim = vectorized_perimeter.compute_total_perimeter(pa)
    check("total_perimeter", vec_perim, ref_perim, atol=1e-12)

    # ------------------------------------------------------------------
    # Test 3: Perimeter gradient
    # ------------------------------------------------------------------
    print("\n--- Test 3: Vectorized perimeter gradient ---")
    ref_grad = perim_calc.compute_total_perimeter_gradient(full_vec)
    vec_grad = vectorized_perimeter.compute_perimeter_gradient(pa)
    # ref_grad is full-sized; vec_grad is active-only — compress ref
    ref_grad_active = ref_grad[partition._active_vp_indices]
    check("perimeter_gradient", vec_grad, ref_grad_active, atol=1e-10)

    # ------------------------------------------------------------------
    # Test 4: Cell areas (regular, no Steiner)
    # ------------------------------------------------------------------
    print("\n--- Test 4: Vectorized cell areas ---")
    ref_areas = area_calc.compute_all_cell_areas(full_vec)
    vec_areas = vectorized_area.compute_cell_areas(pa)
    check("cell_areas", vec_areas, ref_areas, atol=1e-12)

    # ------------------------------------------------------------------
    # Test 5: Area Jacobian (regular, no Steiner) — analytical default
    # ------------------------------------------------------------------
    print("\n--- Test 5: Vectorized area Jacobian (analytical default) ---")
    ref_jac = area_calc.compute_area_jacobian(full_vec)
    vec_jac = vectorized_area.compute_area_jacobian(pa)
    # ref_jac is (n_cells-1, n_total); vec_jac is (n_cells-1, n_active)
    ref_jac_active = ref_jac[:, partition._active_vp_indices]
    check("area_jacobian", vec_jac, ref_jac_active, atol=1e-6, rtol=1e-4)

    # ------------------------------------------------------------------
    # Test 5a: Sparse FD area Jacobian vs AreaCalculator (machine precision)
    # ------------------------------------------------------------------
    print("\n--- Test 5a: Sparse FD area Jacobian vs AreaCalculator ---")
    sparse_jac = vectorized_area.compute_area_jacobian_sparse_fd(pa)
    check("sparse_fd_area_jacobian", sparse_jac, ref_jac_active, atol=1e-12, rtol=1e-10)

    # ------------------------------------------------------------------
    # Test 5b: Analytical area Jacobian vs Sparse FD (~1e-6 FD truncation)
    # ------------------------------------------------------------------
    print("\n--- Test 5b: Analytical area Jacobian vs sparse FD ---")
    analytical_jac = vectorized_area.compute_area_jacobian_analytical(pa)
    check("analytical_vs_sparse_fd", analytical_jac, sparse_jac, atol=1e-6, rtol=1e-4)

    # ------------------------------------------------------------------
    # Test 6: Analytical Steiner points vs BFGS
    # ------------------------------------------------------------------
    print("\n--- Test 6: Analytical Steiner points ---")
    if pa.n_triple_points > 0:
        vec_steiner_pts = vectorized_steiner.compute_steiner_points(pa)
        for tp_i, tp in enumerate(steiner_handler.triple_points):
            vp_pos = [partition.evaluate_variable_point(vi)
                      for vi in tp.var_point_indices]
            tp.steiner_point = None
            bfgs_pt = tp.compute_steiner_point(vp_positions=vp_pos)
            check(f"steiner_point[{tp_i}]",
                  vec_steiner_pts[tp_i], bfgs_pt, atol=1e-6)
    else:
        print("  (no triple points — skipped)")

    # ------------------------------------------------------------------
    # Test 7: Steiner perimeter + area contributions
    # ------------------------------------------------------------------
    print("\n--- Test 7: Steiner perimeter & area contributions ---")
    if pa.n_triple_points > 0:
        vec_sp = vectorized_steiner.compute_steiner_points(pa)
        vec_sp_perim = vectorized_steiner.compute_steiner_perimeter(pa, vec_sp)
        ref_sp_perim = steiner_handler.get_total_perimeter_contribution()
        check("steiner_perimeter", vec_sp_perim, ref_sp_perim, atol=1e-6)

        vec_sp_areas = vectorized_steiner.compute_steiner_areas(pa, vec_sp)
        ref_sp_areas_dict = steiner_handler.get_total_area_contribution()
        ref_sp_areas = np.array([ref_sp_areas_dict.get(c, 0.0)
                                 for c in range(partition.n_cells)])
        check("steiner_areas", vec_sp_areas, ref_sp_areas, atol=1e-6)
    else:
        print("  (no triple points — skipped)")

    # ------------------------------------------------------------------
    # Test 8: Steiner perimeter gradient
    # ------------------------------------------------------------------
    print("\n--- Test 8: Steiner perimeter gradient ---")
    if pa.n_triple_points > 0:
        vec_sg = vectorized_steiner.compute_steiner_perimeter_gradient(pa)
        ref_sg_full = steiner_handler.compute_total_gradient_finite_difference()
        ref_sg_active = ref_sg_full[partition._active_vp_indices]
        check("steiner_perimeter_gradient", vec_sg, ref_sg_active,
              atol=1e-4, rtol=1e-3)
    else:
        print("  (no triple points — skipped)")

    # ------------------------------------------------------------------
    # Test 9: Steiner area Jacobian
    # ------------------------------------------------------------------
    print("\n--- Test 9: Steiner area Jacobian ---")
    if pa.n_triple_points > 0:
        vec_sj = vectorized_steiner.compute_steiner_area_jacobian(pa)
        ref_sj_dict = steiner_handler.compute_area_gradients_finite_difference(
            mesh, eps=1e-7
        )
        n_c = partition.n_cells - 1
        ref_sj = np.zeros((n_c, len(full_vec)))
        for c in range(n_c):
            ref_sj[c, :] = ref_sj_dict[c]
        ref_sj_active = ref_sj[:, partition._active_vp_indices]
        check("steiner_area_jacobian", vec_sj, ref_sj_active,
              atol=1e-4, rtol=1e-3)
    else:
        print("  (no triple points — skipped)")

    # ------------------------------------------------------------------
    # Test 10: End-to-end optimizer (vectorized vs original)
    # ------------------------------------------------------------------
    print("\n--- Test 10: End-to-end optimizer comparison ---")
    total_area = float(mesh.triangle_areas.sum())
    target_area = total_area / partition.n_cells

    # Restore partition state from the known full_vec
    partition.set_variable_vector(full_vec)

    opt_orig = PerimeterOptimizer(partition, mesh, target_area,
                                   use_vectorized=False)
    active_vec = partition.get_active_variable_vector()
    obj_orig = opt_orig.objective(active_vec)
    grad_orig = opt_orig.objective_gradient(active_vec)
    constr_orig = opt_orig.constraint_area_equality(active_vec)
    jac_orig = opt_orig.constraint_area_jacobian(active_vec)

    # Restore again (original path mutates VP objects)
    partition.set_variable_vector(full_vec)

    opt_vec = PerimeterOptimizer(partition, mesh, target_area,
                                  use_vectorized=True)
    opt_vec.compile()
    obj_vec = opt_vec.objective(active_vec)
    grad_vec = opt_vec.objective_gradient(active_vec)
    constr_vec = opt_vec.constraint_area_equality(active_vec)
    jac_vec = opt_vec.constraint_area_jacobian(active_vec)

    check("optimizer.objective", obj_vec, obj_orig, atol=1e-6)
    check("optimizer.objective_gradient", grad_vec, grad_orig, atol=1e-4, rtol=1e-3)
    check("optimizer.constraint_area_equality", constr_vec, constr_orig, atol=1e-6)
    check("optimizer.constraint_area_jacobian", jac_vec, jac_orig, atol=1e-4, rtol=1e-3)

    # ------------------------------------------------------------------
    # Test 11: Fallback (pre-compile objective call)
    # ------------------------------------------------------------------
    print("\n--- Test 11: Pre-compile fallback ---")
    partition.set_variable_vector(full_vec)
    opt_fallback = PerimeterOptimizer(partition, mesh, target_area,
                                       use_vectorized=True)
    # Do NOT call compile() — _arrays should be None, triggering fallback
    obj_fallback = opt_fallback.objective(active_vec)
    check("fallback_objective", obj_fallback, obj_orig, atol=1e-12)

    # ------------------------------------------------------------------
    # Test 12a: Phase 2 — Sparse Jacobian vs Dense Jacobian
    # ------------------------------------------------------------------
    print("\n--- Test 12a: Sparse Jacobian vs Dense Jacobian ---")
    pa.vp_lambda[:] = partition.get_active_variable_vector()
    jac_dense = vectorized_area.compute_area_jacobian(pa)
    jac_sparse_vals = vectorized_area.compute_area_jacobian_sparse(pa)
    steiner_dense = vectorized_steiner.compute_steiner_area_jacobian(pa)
    steiner_sparse_vals = vectorized_steiner.compute_steiner_area_jacobian_sparse(pa)

    total_sparse = jac_sparse_vals + steiner_sparse_vals
    total_dense = jac_dense + steiner_dense

    jac_reconstructed = np.zeros_like(total_dense)
    jac_reconstructed[pa.jac_row, pa.jac_col] = total_sparse

    check("sparse_jac_vs_dense (at nnz positions)",
          jac_reconstructed[pa.jac_row, pa.jac_col],
          total_dense[pa.jac_row, pa.jac_col],
          atol=1e-10)

    # ------------------------------------------------------------------
    # Test 12b: Phase 4 — Analytical Hessian vs FD Hessian
    # ------------------------------------------------------------------
    print("\n--- Test 12b: Analytical Hessian vs FD Hessian ---")
    pa.vp_lambda[:] = partition.get_active_variable_vector()
    x_test = pa.vp_lambda.copy()
    n = len(x_test)

    # Create fake multipliers and obj_factor for the test
    n_c = pa.n_cells - 1
    rng_h = np.random.RandomState(123)
    test_multipliers = rng_h.randn(n_c)
    test_obj_factor = 1.0

    # Analytical Hessian
    pa.vp_lambda[:] = x_test
    perim_h = vectorized_perimeter.compute_perimeter_hessian_sparse(pa)
    steiner_perim_h = vectorized_steiner.compute_steiner_perimeter_hessian_fd(pa)
    area_h = vectorized_area.compute_area_hessian_sparse(pa, test_multipliers)
    steiner_area_h = vectorized_steiner.compute_steiner_area_hessian_fd(pa, test_multipliers)
    analytical_hess_vals = (test_obj_factor * (perim_h + steiner_perim_h)
                            + area_h + steiner_area_h)

    # FD Hessian of the Lagrangian
    eps_fd = 1e-5
    H_fd = np.zeros((n, n))
    for i in range(n):
        x_plus = x_test.copy(); x_plus[i] += eps_fd
        x_minus = x_test.copy(); x_minus[i] -= eps_fd

        # Lagrangian gradient = obj_factor * nabla f + sum mu_k * nabla c_k
        pa.vp_lambda[:] = x_plus
        grad_obj_p = vectorized_perimeter.compute_perimeter_gradient(pa)
        grad_obj_p += vectorized_steiner.compute_steiner_perimeter_gradient(pa)
        jac_p = vectorized_area.compute_area_jacobian(pa)
        jac_p += vectorized_steiner.compute_steiner_area_jacobian(pa)
        grad_lag_p = test_obj_factor * grad_obj_p + test_multipliers @ jac_p

        pa.vp_lambda[:] = x_minus
        grad_obj_m = vectorized_perimeter.compute_perimeter_gradient(pa)
        grad_obj_m += vectorized_steiner.compute_steiner_perimeter_gradient(pa)
        jac_m = vectorized_area.compute_area_jacobian(pa)
        jac_m += vectorized_steiner.compute_steiner_area_jacobian(pa)
        grad_lag_m = test_obj_factor * grad_obj_m + test_multipliers @ jac_m

        H_fd[i, :] = (grad_lag_p - grad_lag_m) / (2 * eps_fd)

    H_fd = 0.5 * (H_fd + H_fd.T)

    # Reconstruct analytical Hessian into dense form for comparison
    H_analytical = np.zeros((n, n))
    for idx in range(len(pa.hess_row)):
        r, c = int(pa.hess_row[idx]), int(pa.hess_col[idx])
        H_analytical[r, c] = analytical_hess_vals[idx]
        if r != c:
            H_analytical[c, r] = analytical_hess_vals[idx]

    # Compare only at Hessian sparsity pattern positions
    hess_positions = list(zip(pa.hess_row, pa.hess_col))
    ana_at_pattern = np.array([H_analytical[r, c] for r, c in hess_positions])
    fd_at_pattern = np.array([H_fd[r, c] for r, c in hess_positions])

    check("hessian_analytical_vs_fd",
          ana_at_pattern, fd_at_pattern,
          atol=1e-3, rtol=1e-3)

    # Restore lambda
    pa.vp_lambda[:] = x_test

    # ------------------------------------------------------------------
    # Test 12: End-to-end short optimization (vectorized vs original)
    # ------------------------------------------------------------------
    print("\n--- Test 12: End-to-end optimization (10 SLSQP iterations) ---")
    partition.set_variable_vector(full_vec)
    opt_e2e_orig = PerimeterOptimizer(partition, mesh, target_area,
                                       use_vectorized=False)
    result_orig = opt_e2e_orig.optimize(max_iter=10, tol=1e-8, method='SLSQP')
    final_orig = opt_e2e_orig.objective(result_orig.x)
    x_orig = result_orig.x.copy()

    partition.set_variable_vector(full_vec)
    opt_e2e_vec = PerimeterOptimizer(partition, mesh, target_area,
                                      use_vectorized=True)
    result_vec = opt_e2e_vec.optimize(max_iter=10, tol=1e-8, method='SLSQP')
    final_vec = opt_e2e_vec.objective(result_vec.x)
    x_vec = result_vec.x.copy()

    check("e2e_final_objective", final_vec, final_orig, atol=5e-2, rtol=1e-2)
    check("e2e_result_x", x_vec, x_orig, atol=1e-1, rtol=1e-1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    total = _passed + _failed
    print(f"Results: {_passed}/{total} passed, {_failed}/{total} failed")
    if _failed > 0:
        print("SOME TESTS FAILED")
        return 1
    else:
        print("ALL TESTS PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
