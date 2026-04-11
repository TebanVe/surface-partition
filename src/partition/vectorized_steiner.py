"""
Vectorized Steiner computation using the analytical Fermat-Torricelli formula.

Replaces the per-triple-point BFGS solver with a closed-form formula that runs
in microseconds instead of milliseconds, making caching unnecessary.

All functions operate on :class:`~partition_arrays.PartitionArrays`.
"""

import numpy as np

from .partition_arrays import PartitionArrays
from .vectorized_perimeter import _compute_vp_positions, _triangle_areas_batch


# =========================================================================
# Steiner point computation (analytical Fermat-Torricelli)
# =========================================================================

def compute_steiner_points(pa: PartitionArrays) -> np.ndarray:
    """Compute all Steiner points using the analytical Fermat-Torricelli formula.

    Returns:
        (n_triple_points, dim) float64.  For triple points where any void
        angle >= 120 deg, returns the obtuse vertex position (degenerate case).
    """
    if pa.n_triple_points == 0:
        return np.empty((0, pa.vertices.shape[1]))

    pos = _compute_vp_positions(pa)

    # Gather the 3 VP positions for each triple point: (n_tp, 3, dim)
    p = pos[pa.tp_vp_indices]
    p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]

    # Side lengths opposite each vertex
    a = np.linalg.norm(p2 - p3, axis=1)  # opposite p1
    b = np.linalg.norm(p1 - p3, axis=1)  # opposite p2
    c = np.linalg.norm(p1 - p2, axis=1)  # opposite p3

    # Angles via law of cosines
    cos_A = np.clip((b**2 + c**2 - a**2) / (2.0 * b * c + 1e-30), -1.0, 1.0)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2.0 * a * c + 1e-30), -1.0, 1.0)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2.0 * a * b + 1e-30), -1.0, 1.0)

    A_ang = np.arccos(cos_A)
    B_ang = np.arccos(cos_B)
    C_ang = np.arccos(cos_C)

    # Barycentric weights: w_i = a_i / sin(A_i + pi/3)
    pi_over_3 = np.pi / 3.0
    w1 = a / np.maximum(np.sin(A_ang + pi_over_3), 1e-15)
    w2 = b / np.maximum(np.sin(B_ang + pi_over_3), 1e-15)
    w3 = c / np.maximum(np.sin(C_ang + pi_over_3), 1e-15)

    w_sum = w1 + w2 + w3
    steiner = (w1[:, None] * p1 + w2[:, None] * p2 + w3[:, None] * p3) / w_sum[:, None]

    # Degenerate case: any angle >= 120 deg → Steiner = obtuse vertex
    threshold = 2.0 * np.pi / 3.0  # 120 deg
    all_angles = np.stack([A_ang, B_ang, C_ang], axis=1)
    degen_mask = all_angles.max(axis=1) >= threshold
    if np.any(degen_mask):
        max_idx = np.argmax(all_angles[degen_mask], axis=1)
        steiner[degen_mask] = p[degen_mask, max_idx]

    return steiner


# =========================================================================
# Steiner perimeter contribution
# =========================================================================

def compute_steiner_perimeter(pa: PartitionArrays,
                              steiner_pts: np.ndarray) -> float:
    """Total Steiner perimeter contribution.

    Per cell per triple point: d(vp_a, S) + d(vp_b, S) - d(vp_a, vp_b).
    Summed over all 3 cells per triple point (consistent with double-counting
    convention — each Steiner edge appears in 2 cells' contributions).
    """
    if pa.n_triple_points == 0:
        return 0.0

    pos = _compute_vp_positions(pa)
    vp1_pos = pos[pa.tp_contrib_vp1]
    vp2_pos = pos[pa.tp_contrib_vp2]
    s_pos = steiner_pts[pa.tp_contrib_tp_idx]

    d_s_vp1 = np.linalg.norm(vp1_pos - s_pos, axis=1)
    d_s_vp2 = np.linalg.norm(vp2_pos - s_pos, axis=1)
    d_vp1_vp2 = np.linalg.norm(vp1_pos - vp2_pos, axis=1)

    return float(np.sum(d_s_vp1 + d_s_vp2 - d_vp1_vp2))


# =========================================================================
# Steiner area contribution
# =========================================================================

def compute_steiner_areas(pa: PartitionArrays,
                          steiner_pts: np.ndarray) -> np.ndarray:
    """Steiner area contributions per cell.

    Each cell at a triple point gets:
      void_area   = area(vp_a, vp_b, steiner_point)
      corner_area = area(mesh_vertex, vp_a, vp_b)

    Returns:
        (n_cells,) float64 with contributions scatter-added.
    """
    areas = np.zeros(pa.n_cells, dtype=np.float64)
    if pa.n_triple_points == 0:
        return areas

    pos = _compute_vp_positions(pa)
    vp1_pos = pos[pa.tp_contrib_vp1]
    vp2_pos = pos[pa.tp_contrib_vp2]
    s_pos = steiner_pts[pa.tp_contrib_tp_idx]
    mv_pos = pa.vertices[pa.tp_contrib_mesh_vertex]

    void_areas = _triangle_areas_batch(vp1_pos, vp2_pos, s_pos)
    corner_areas = _triangle_areas_batch(mv_pos, vp1_pos, vp2_pos)

    np.add.at(areas, pa.tp_contrib_cell, void_areas + corner_areas)
    return areas


# =========================================================================
# Steiner perimeter gradient (finite differences on the analytical formula)
# =========================================================================

def compute_steiner_perimeter_gradient(pa: PartitionArrays,
                                       eps: float = 1e-6) -> np.ndarray:
    """d(steiner_perimeter)/d(lambda) via finite differences."""
    gradient = np.zeros(pa.n_active_vp, dtype=np.float64)
    if pa.n_triple_points == 0:
        return gradient

    base_steiner = compute_steiner_points(pa)
    base_perim = compute_steiner_perimeter(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        pert_perim = compute_steiner_perimeter(pa, pert_steiner)
        gradient[vp_idx] = (pert_perim - base_perim) / eps
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return gradient


# =========================================================================
# Steiner area Jacobian (finite differences on the analytical formula)
# =========================================================================

def compute_steiner_area_jacobian_sparse(pa: PartitionArrays,
                                         eps: float = 1e-7) -> np.ndarray:
    """Steiner area Jacobian — sparse values in jac_row/jac_col order.

    Uses the same finite-difference approach as compute_steiner_area_jacobian()
    but writes into a sparse values array instead of a dense matrix.

    Returns:
        (nnz,) float64
    """
    nnz = len(pa.jac_row)
    values = np.zeros(nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    n_constraints = pa.n_cells - 1
    base_steiner = compute_steiner_points(pa)
    base_areas = compute_steiner_areas(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        pert_areas = compute_steiner_areas(pa, pert_steiner)
        dA = (pert_areas[:n_constraints] - base_areas[:n_constraints]) / eps

        for c in range(n_constraints):
            offset = pa.nnz_lookup[c, vp_idx]
            if offset >= 0:
                values[offset] += dA[c]

        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return values


def compute_steiner_perimeter_hessian_fd(pa: PartitionArrays,
                                         eps: float = 1e-5) -> np.ndarray:
    """Steiner perimeter Hessian via central FD on the gradient.

    Only tp_affected_vps have non-zero entries. The inner gradient FD uses
    eps_inner = eps * 0.1 to avoid nested FD at the same scale.

    Returns:
        (hess_nnz,) float64
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    original = pa.vp_lambda.copy()
    affected = pa.tp_affected_vps
    eps_inner = eps * 0.1

    for vp_i in affected:
        pa.vp_lambda[vp_i] = original[vp_i] + eps
        grad_plus = compute_steiner_perimeter_gradient(pa, eps=eps_inner)
        pa.vp_lambda[vp_i] = original[vp_i] - eps
        grad_minus = compute_steiner_perimeter_gradient(pa, eps=eps_inner)
        pa.vp_lambda[vp_i] = original[vp_i]

        hess_col_i = (grad_plus - grad_minus) / (2.0 * eps)

        for vp_j in affected:
            if vp_j > vp_i:
                continue
            key = (max(int(vp_i), int(vp_j)), min(int(vp_i), int(vp_j)))
            if key in pa.hess_offset_map:
                values[pa.hess_offset_map[key]] += hess_col_i[vp_j]

    return values


def compute_steiner_area_hessian_fd(pa: PartitionArrays,
                                    multipliers: np.ndarray,
                                    eps: float = 1e-5) -> np.ndarray:
    """Multiplier-weighted Steiner area Hessian via forward FD on the Jacobian.

    The inner Jacobian uses its own default eps (1e-7), much smaller than the
    outer eps (1e-5), so there is no nested-epsilon accuracy problem.

    Returns:
        (hess_nnz,) float64
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    n_c = pa.n_cells - 1
    original = pa.vp_lambda.copy()
    affected = pa.tp_affected_vps

    base_jac = compute_steiner_area_jacobian(pa)

    for vp_i in affected:
        pa.vp_lambda[vp_i] = original[vp_i] + eps
        jac_plus = compute_steiner_area_jacobian(pa)
        pa.vp_lambda[vp_i] = original[vp_i]

        djac = (jac_plus - base_jac) / eps

        for vp_j in affected:
            if vp_j > vp_i:
                continue
            h_val = 0.0
            for c in range(n_c):
                h_val += multipliers[c] * djac[c, vp_j]
            key = (max(int(vp_i), int(vp_j)), min(int(vp_i), int(vp_j)))
            if key in pa.hess_offset_map:
                values[pa.hess_offset_map[key]] += h_val

    return values


def compute_steiner_area_jacobian(pa: PartitionArrays,
                                  eps: float = 1e-7) -> np.ndarray:
    """d(steiner_areas)/d(lambda) via finite differences.

    Returns:
        (n_cells - 1, n_active_vp) float64
    """
    n_constraints = pa.n_cells - 1
    jacobian = np.zeros((n_constraints, pa.n_active_vp), dtype=np.float64)
    if pa.n_triple_points == 0:
        return jacobian

    base_steiner = compute_steiner_points(pa)
    base_areas = compute_steiner_areas(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        pert_areas = compute_steiner_areas(pa, pert_steiner)
        jacobian[:, vp_idx] = (pert_areas[:n_constraints] - base_areas[:n_constraints]) / eps
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return jacobian
