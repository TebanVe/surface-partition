"""
Vectorized area computation on flat PartitionArrays.

All functions operate on :class:`~partition_arrays.PartitionArrays`.

Triple-point triangles are NOT in the boundary triangle arrays — their area
is computed by :mod:`vectorized_steiner` and added by the caller.
"""

import numpy as np

from .partition_arrays import PartitionArrays
from .vectorized_perimeter import _compute_vp_positions, _triangle_areas_batch


# =========================================================================
# Cell areas (constraint function)
# =========================================================================

def compute_cell_areas(pa: PartitionArrays) -> np.ndarray:
    """Compute all cell areas excluding Steiner contributions.

    Matches AreaCalculator.compute_all_cell_areas().  Steiner contributions
    from vectorized_steiner must be added separately by the caller.

    Returns:
        (n_cells,) float64
    """
    areas = pa.cell_interior_area.copy()
    pos = _compute_vp_positions(pa)

    # --- 1-inside boundary triangles: area = triangle(p_in, p_cut1, p_cut2) ---
    mask1 = pa.btri_n_inside == 1
    if np.any(mask1):
        p_in = pa.vertices[pa.btri_v_in[mask1, 0]]
        p_cut1 = pos[pa.btri_vp1[mask1]]
        p_cut2 = pos[pa.btri_vp2[mask1]]
        tri_areas = _triangle_areas_batch(p_in, p_cut1, p_cut2)
        np.add.at(areas, pa.btri_cell[mask1], tri_areas)

    # --- 2-inside boundary triangles: area = quad(p_in1, p_cut1, p_cut2, p_in2) ---
    mask2 = pa.btri_n_inside == 2
    if np.any(mask2):
        p_in1 = pa.vertices[pa.btri_v_in[mask2, 0]]
        p_in2 = pa.vertices[pa.btri_v_in[mask2, 1]]
        p_cut1 = pos[pa.btri_vp1[mask2]]
        p_cut2 = pos[pa.btri_vp2[mask2]]
        quad_areas = (_triangle_areas_batch(p_in1, p_cut1, p_cut2)
                      + _triangle_areas_batch(p_in1, p_cut2, p_in2))
        np.add.at(areas, pa.btri_cell[mask2], quad_areas)

    return areas


# =========================================================================
# Area constraint Jacobian — sparse finite differences (reference/test)
# =========================================================================

def compute_area_jacobian_sparse_fd(pa: PartitionArrays,
                                    eps: float = 1e-7) -> np.ndarray:
    """Area Jacobian via per-triangle sparse finite differences.

    For each boundary triangle, perturbs only the 2 affected VPs and
    computes the area change.  Results are scatter-added into the Jacobian.

    Cost: O(n_btri) — one vectorized pass over boundary triangles, no VP loop.

    Kept as a reference/test utility.  The default ``compute_area_jacobian``
    uses the faster analytical path.

    Returns:
        (n_cells - 1, n_active_vp) float64
    """
    n_c = pa.n_cells - 1
    pos = _compute_vp_positions(pa)
    jacobian = np.zeros((n_c, pa.n_active_vp), dtype=np.float64)

    # --- 1-inside triangles: area = triangle(p_in, pc1, pc2) ---
    m1 = pa.btri_n_inside == 1
    if np.any(m1):
        p_in = pa.vertices[pa.btri_v_in[m1, 0]]
        pc1 = pos[pa.btri_vp1[m1]]
        pc2 = pos[pa.btri_vp2[m1]]

        vp1_idx = pa.btri_vp1[m1]
        vp2_idx = pa.btri_vp2[m1]

        base_area = _triangle_areas_batch(p_in, pc1, pc2)

        # Perturb VP1: recompute position from lambda+eps (matches AreaCalculator FD path)
        lam1 = pa.vp_lambda[vp1_idx] + eps
        pc1_pert = (lam1[:, None] * pa.vertices[pa.vp_edge_v1[vp1_idx]]
                    + (1.0 - lam1[:, None]) * pa.vertices[pa.vp_edge_v2[vp1_idx]])
        dA_dlam1 = (_triangle_areas_batch(p_in, pc1_pert, pc2) - base_area) / eps

        # Perturb VP2: recompute position from lambda+eps
        lam2 = pa.vp_lambda[vp2_idx] + eps
        pc2_pert = (lam2[:, None] * pa.vertices[pa.vp_edge_v1[vp2_idx]]
                    + (1.0 - lam2[:, None]) * pa.vertices[pa.vp_edge_v2[vp2_idx]])
        dA_dlam2 = (_triangle_areas_batch(p_in, pc1, pc2_pert) - base_area) / eps

        cells = pa.btri_cell[m1]
        for c in range(n_c):
            cmask = cells == c
            np.add.at(jacobian[c], vp1_idx[cmask], dA_dlam1[cmask])
            np.add.at(jacobian[c], vp2_idx[cmask], dA_dlam2[cmask])

    # --- 2-inside triangles: area = quad(p_in1, pc1, pc2, p_in2) ---
    m2 = pa.btri_n_inside == 2
    if np.any(m2):
        p_in1 = pa.vertices[pa.btri_v_in[m2, 0]]
        p_in2 = pa.vertices[pa.btri_v_in[m2, 1]]
        pc1 = pos[pa.btri_vp1[m2]]
        pc2 = pos[pa.btri_vp2[m2]]

        vp1_idx = pa.btri_vp1[m2]
        vp2_idx = pa.btri_vp2[m2]

        base_area = (_triangle_areas_batch(p_in1, pc1, pc2)
                     + _triangle_areas_batch(p_in1, pc2, p_in2))

        # Perturb VP1 (affects only first sub-triangle)
        lam1 = pa.vp_lambda[vp1_idx] + eps
        pc1_pert = (lam1[:, None] * pa.vertices[pa.vp_edge_v1[vp1_idx]]
                    + (1.0 - lam1[:, None]) * pa.vertices[pa.vp_edge_v2[vp1_idx]])
        area_p1 = (_triangle_areas_batch(p_in1, pc1_pert, pc2)
                   + _triangle_areas_batch(p_in1, pc2, p_in2))
        dA_dlam1 = (area_p1 - base_area) / eps

        # Perturb VP2 (affects both sub-triangles)
        lam2 = pa.vp_lambda[vp2_idx] + eps
        pc2_pert = (lam2[:, None] * pa.vertices[pa.vp_edge_v1[vp2_idx]]
                    + (1.0 - lam2[:, None]) * pa.vertices[pa.vp_edge_v2[vp2_idx]])
        area_p2 = (_triangle_areas_batch(p_in1, pc1, pc2_pert)
                   + _triangle_areas_batch(p_in1, pc2_pert, p_in2))
        dA_dlam2 = (area_p2 - base_area) / eps

        cells = pa.btri_cell[m2]
        for c in range(n_c):
            cmask = cells == c
            np.add.at(jacobian[c], vp1_idx[cmask], dA_dlam1[cmask])
            np.add.at(jacobian[c], vp2_idx[cmask], dA_dlam2[cmask])

    return jacobian


# =========================================================================
# Area constraint Jacobian — analytical (default, recommended)
# =========================================================================

def compute_area_jacobian_analytical(pa: PartitionArrays,
                                     eps: float = 1e-7) -> np.ndarray:
    """Area Jacobian via analytical chain-rule derivatives.

    Uses the scalar triple product (3D) or signed-area derivative (2D) to
    compute dA/dlambda exactly for each boundary triangle, with no finite
    differences.

    The *eps* parameter is unused but kept for API compatibility with
    ``compute_area_jacobian``.

    Cost: O(n_btri) — same as ``compute_cell_areas`` itself.

    Returns:
        (n_cells - 1, n_active_vp) float64
    """
    n_c = pa.n_cells - 1
    pos = _compute_vp_positions(pa)
    jacobian = np.zeros((n_c, pa.n_active_vp), dtype=np.float64)
    dim = pa.vertices.shape[1]

    # --- 1-inside triangles: A = area(p_in, pc1, pc2) ---
    m1 = pa.btri_n_inside == 1
    if np.any(m1):
        p_in = pa.vertices[pa.btri_v_in[m1, 0]]
        pc1 = pos[pa.btri_vp1[m1]]
        pc2 = pos[pa.btri_vp2[m1]]

        u = pc1 - p_in
        v = pc2 - p_in

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m1]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m1]]])

        if dim == 3:
            cross_uv = np.cross(u, v)
            norm_uv = np.linalg.norm(cross_uv, axis=1, keepdims=True)
            norm_uv = np.maximum(norm_uv, 1e-30)
            n_hat = cross_uv / norm_uv

            dA_dl1 = 0.5 * np.sum(n_hat * np.cross(d1, v), axis=1)
            dA_dl2 = 0.5 * np.sum(n_hat * np.cross(u, d2), axis=1)
        else:
            signed_area = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
            sign = np.sign(signed_area)
            sign[sign == 0] = 1.0
            dA_dl1 = sign * 0.5 * (d1[:, 0] * v[:, 1] - d1[:, 1] * v[:, 0])
            dA_dl2 = sign * 0.5 * (u[:, 0] * d2[:, 1] - u[:, 1] * d2[:, 0])

        cells = pa.btri_cell[m1]
        vp1 = pa.btri_vp1[m1]
        vp2 = pa.btri_vp2[m1]
        for c in range(n_c):
            cmask = cells == c
            np.add.at(jacobian[c], vp1[cmask], dA_dl1[cmask])
            np.add.at(jacobian[c], vp2[cmask], dA_dl2[cmask])

    # --- 2-inside triangles: A = area(p_in1, pc1, pc2) + area(p_in1, pc2, p_in2) ---
    m2 = pa.btri_n_inside == 2
    if np.any(m2):
        p_in1 = pa.vertices[pa.btri_v_in[m2, 0]]
        p_in2 = pa.vertices[pa.btri_v_in[m2, 1]]
        pc1 = pos[pa.btri_vp1[m2]]
        pc2 = pos[pa.btri_vp2[m2]]

        u1 = pc1 - p_in1
        v1 = pc2 - p_in1
        u2 = pc2 - p_in1
        v2 = p_in2 - p_in1

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m2]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m2]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m2]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m2]]])

        if dim == 3:
            cross1 = np.cross(u1, v1)
            norm1 = np.maximum(np.linalg.norm(cross1, axis=1, keepdims=True), 1e-30)
            n1 = cross1 / norm1

            cross2 = np.cross(u2, v2)
            norm2 = np.maximum(np.linalg.norm(cross2, axis=1, keepdims=True), 1e-30)
            n2 = cross2 / norm2

            dA_dl1 = 0.5 * np.sum(n1 * np.cross(d1, v1), axis=1)
            dA_dl2 = (0.5 * np.sum(n1 * np.cross(u1, d2), axis=1)
                      + 0.5 * np.sum(n2 * np.cross(d2, v2), axis=1))
        else:
            sa1 = 0.5 * (u1[:, 0] * v1[:, 1] - u1[:, 1] * v1[:, 0])
            s1 = np.sign(sa1)
            s1[s1 == 0] = 1.0
            sa2 = 0.5 * (u2[:, 0] * v2[:, 1] - u2[:, 1] * v2[:, 0])
            s2 = np.sign(sa2)
            s2[s2 == 0] = 1.0

            dA_dl1 = s1 * 0.5 * (d1[:, 0] * v1[:, 1] - d1[:, 1] * v1[:, 0])
            dA_dl2 = (s1 * 0.5 * (u1[:, 0] * d2[:, 1] - u1[:, 1] * d2[:, 0])
                      + s2 * 0.5 * (d2[:, 0] * v2[:, 1] - d2[:, 1] * v2[:, 0]))

        cells = pa.btri_cell[m2]
        vp1 = pa.btri_vp1[m2]
        vp2 = pa.btri_vp2[m2]
        for c in range(n_c):
            cmask = cells == c
            np.add.at(jacobian[c], vp1[cmask], dA_dl1[cmask])
            np.add.at(jacobian[c], vp2[cmask], dA_dl2[cmask])

    return jacobian


# =========================================================================
# Area constraint Jacobian — public API (delegates to analytical)
# =========================================================================

def compute_area_jacobian_sparse(pa: PartitionArrays) -> np.ndarray:
    """Compute area Jacobian non-zero values in jac_row/jac_col order.

    Direct sparse computation — no dense matrix allocation.

    Returns:
        (nnz,) float64 — values at positions (pa.jac_row, pa.jac_col).
    """
    nnz = len(pa.jac_row)
    values = np.zeros(nnz, dtype=np.float64)
    pos = _compute_vp_positions(pa)
    dim = pa.vertices.shape[1]

    # --- 1-inside triangles ---
    m1 = pa.btri_n_inside == 1
    if np.any(m1):
        p_in = pa.vertices[pa.btri_v_in[m1, 0]]
        pc1 = pos[pa.btri_vp1[m1]]
        pc2 = pos[pa.btri_vp2[m1]]

        u = pc1 - p_in
        v = pc2 - p_in

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m1]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m1]]])

        if dim == 3:
            cross_uv = np.cross(u, v)
            norm_uv = np.maximum(np.linalg.norm(cross_uv, axis=1, keepdims=True), 1e-30)
            n_hat = cross_uv / norm_uv
            dA_dl1 = 0.5 * np.sum(n_hat * np.cross(d1, v), axis=1)
            dA_dl2 = 0.5 * np.sum(n_hat * np.cross(u, d2), axis=1)
        else:
            signed_area = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
            sign = np.sign(signed_area)
            sign[sign == 0] = 1.0
            dA_dl1 = sign * 0.5 * (d1[:, 0] * v[:, 1] - d1[:, 1] * v[:, 0])
            dA_dl2 = sign * 0.5 * (u[:, 0] * d2[:, 1] - u[:, 1] * d2[:, 0])

        cells = pa.btri_cell[m1]
        vp1 = pa.btri_vp1[m1]
        vp2 = pa.btri_vp2[m1]
        mask = cells < (pa.n_cells - 1)

        offsets1 = pa.nnz_lookup[cells[mask], vp1[mask]]
        offsets2 = pa.nnz_lookup[cells[mask], vp2[mask]]
        np.add.at(values, offsets1, dA_dl1[mask])
        np.add.at(values, offsets2, dA_dl2[mask])

    # --- 2-inside triangles ---
    m2 = pa.btri_n_inside == 2
    if np.any(m2):
        p_in1 = pa.vertices[pa.btri_v_in[m2, 0]]
        p_in2 = pa.vertices[pa.btri_v_in[m2, 1]]
        pc1 = pos[pa.btri_vp1[m2]]
        pc2 = pos[pa.btri_vp2[m2]]

        u1 = pc1 - p_in1
        v1 = pc2 - p_in1
        u2 = pc2 - p_in1
        v2 = p_in2 - p_in1

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m2]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m2]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m2]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m2]]])

        if dim == 3:
            cross1 = np.cross(u1, v1)
            norm1 = np.maximum(np.linalg.norm(cross1, axis=1, keepdims=True), 1e-30)
            n1 = cross1 / norm1

            cross2 = np.cross(u2, v2)
            norm2 = np.maximum(np.linalg.norm(cross2, axis=1, keepdims=True), 1e-30)
            n2 = cross2 / norm2

            dA_dl1 = 0.5 * np.sum(n1 * np.cross(d1, v1), axis=1)
            dA_dl2 = (0.5 * np.sum(n1 * np.cross(u1, d2), axis=1)
                      + 0.5 * np.sum(n2 * np.cross(d2, v2), axis=1))
        else:
            sa1 = 0.5 * (u1[:, 0] * v1[:, 1] - u1[:, 1] * v1[:, 0])
            s1 = np.sign(sa1); s1[s1 == 0] = 1.0
            sa2 = 0.5 * (u2[:, 0] * v2[:, 1] - u2[:, 1] * v2[:, 0])
            s2 = np.sign(sa2); s2[s2 == 0] = 1.0
            dA_dl1 = s1 * 0.5 * (d1[:, 0] * v1[:, 1] - d1[:, 1] * v1[:, 0])
            dA_dl2 = (s1 * 0.5 * (u1[:, 0] * d2[:, 1] - u1[:, 1] * d2[:, 0])
                      + s2 * 0.5 * (d2[:, 0] * v2[:, 1] - d2[:, 1] * v2[:, 0]))

        cells = pa.btri_cell[m2]
        vp1 = pa.btri_vp1[m2]
        vp2 = pa.btri_vp2[m2]
        mask = cells < (pa.n_cells - 1)

        offsets1 = pa.nnz_lookup[cells[mask], vp1[mask]]
        offsets2 = pa.nnz_lookup[cells[mask], vp2[mask]]
        np.add.at(values, offsets1, dA_dl1[mask])
        np.add.at(values, offsets2, dA_dl2[mask])

    return values


def compute_area_hessian_sparse(pa: PartitionArrays,
                                multipliers: np.ndarray) -> np.ndarray:
    """Compute multiplier-weighted area constraint Hessian: sum_k mu_k * d2A_k/dlam_i dlam_j.

    Args:
        pa: PartitionArrays snapshot
        multipliers: (n_cells-1,) float64 — Lagrange multipliers from IPOPT

    Returns:
        (hess_nnz,) float64 — values at positions (pa.hess_row, pa.hess_col).
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    pos = _compute_vp_positions(pa)
    dim = pa.vertices.shape[1]
    n_c = pa.n_cells - 1

    # --- 1-inside triangles ---
    m1 = pa.btri_n_inside == 1
    if np.any(m1):
        p_in = pa.vertices[pa.btri_v_in[m1, 0]]
        pc1 = pos[pa.btri_vp1[m1]]
        pc2 = pos[pa.btri_vp2[m1]]

        u = pc1 - p_in
        v = pc2 - p_in

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m1]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m1]]])

        if dim == 3:
            C = np.cross(u, v)
            normC = np.maximum(np.linalg.norm(C, axis=1), 1e-30)
            n_hat = C / normC[:, None]

            g1 = np.cross(d1, v)
            g2 = np.cross(u, d2)
            d1xd2 = np.cross(d1, d2)

            n_dot_g1 = np.sum(n_hat * g1, axis=1)
            n_dot_g2 = np.sum(n_hat * g2, axis=1)
            g1_dot_g1 = np.sum(g1 * g1, axis=1)
            g2_dot_g2 = np.sum(g2 * g2, axis=1)
            g1_dot_g2 = np.sum(g1 * g2, axis=1)
            n_dot_d1xd2 = np.sum(n_hat * d1xd2, axis=1)

            H_11 = (g1_dot_g1 - n_dot_g1**2) / (2.0 * normC)
            H_22 = (g2_dot_g2 - n_dot_g2**2) / (2.0 * normC)
            H_12 = ((g1_dot_g2 - n_dot_g1 * n_dot_g2) / normC + n_dot_d1xd2) / 2.0
        else:
            C_scalar = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            normC = np.maximum(np.abs(C_scalar), 1e-30)
            sign = np.sign(C_scalar)
            sign[sign == 0] = 1.0

            g1_scalar = d1[:, 0] * v[:, 1] - d1[:, 1] * v[:, 0]
            g2_scalar = u[:, 0] * d2[:, 1] - u[:, 1] * d2[:, 0]
            d1xd2_scalar = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]

            H_11 = (g1_scalar**2 - (sign * g1_scalar)**2) / (2.0 * normC)
            H_22 = (g2_scalar**2 - (sign * g2_scalar)**2) / (2.0 * normC)
            H_12 = ((g1_scalar * g2_scalar - (sign * g1_scalar) * (sign * g2_scalar)) / normC
                     + sign * d1xd2_scalar) / 2.0

        # btri1_cell_active was built from np.flatnonzero(btri_n_inside == 1),
        # the same ordering as btri_cell[m1] here, so indexing aligns row-for-row.
        assert pa.btri1_hess_off_aa is not None, \
            "PartitionArrays missing Phase A Hessian offset fields — call compile_arrays() first"
        active = pa.btri1_cell_active
        if np.any(active):
            cells_active = pa.btri_cell[m1][active]
            mu = multipliers[cells_active]
            np.add.at(values, pa.btri1_hess_off_aa[active], mu * H_11[active])
            np.add.at(values, pa.btri1_hess_off_bb[active], mu * H_22[active])
            np.add.at(values, pa.btri1_hess_off_ab[active], mu * H_12[active])

    # --- 2-inside triangles ---
    m2 = pa.btri_n_inside == 2
    if np.any(m2):
        p_in1 = pa.vertices[pa.btri_v_in[m2, 0]]
        p_in2 = pa.vertices[pa.btri_v_in[m2, 1]]
        pc1 = pos[pa.btri_vp1[m2]]
        pc2 = pos[pa.btri_vp2[m2]]

        u1 = pc1 - p_in1;  v1 = pc2 - p_in1
        u2 = pc2 - p_in1;  v2 = p_in2 - p_in1

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m2]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m2]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m2]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m2]]])

        if dim == 3:
            # Sub-triangle 1: (p_in1, pc1, pc2)
            C1 = np.cross(u1, v1)
            normC1 = np.maximum(np.linalg.norm(C1, axis=1), 1e-30)
            n1 = C1 / normC1[:, None]

            g1_1 = np.cross(d1, v1)
            g2_1 = np.cross(u1, d2)
            d1xd2 = np.cross(d1, d2)

            n1_dot_g1_1 = np.sum(n1 * g1_1, axis=1)
            n1_dot_g2_1 = np.sum(n1 * g2_1, axis=1)

            H_11_sub1 = (np.sum(g1_1 * g1_1, axis=1) - n1_dot_g1_1**2) / (2.0 * normC1)
            H_22_sub1 = (np.sum(g2_1 * g2_1, axis=1) - n1_dot_g2_1**2) / (2.0 * normC1)
            H_12_sub1 = ((np.sum(g1_1 * g2_1, axis=1) - n1_dot_g1_1 * n1_dot_g2_1)
                         / normC1 + np.sum(n1 * d1xd2, axis=1)) / 2.0

            # Sub-triangle 2: (p_in1, pc2, p_in2) — only lambda_2 appears
            C2 = np.cross(u2, v2)
            normC2 = np.maximum(np.linalg.norm(C2, axis=1), 1e-30)
            n2 = C2 / normC2[:, None]

            g2_2 = np.cross(d2, v2)
            n2_dot_g2_2 = np.sum(n2 * g2_2, axis=1)

            H_22_sub2 = (np.sum(g2_2 * g2_2, axis=1) - n2_dot_g2_2**2) / (2.0 * normC2)

            H_11 = H_11_sub1
            H_22 = H_22_sub1 + H_22_sub2
            H_12 = H_12_sub1
        else:
            # Sub-triangle 1
            C1s = u1[:, 0] * v1[:, 1] - u1[:, 1] * v1[:, 0]
            normC1 = np.maximum(np.abs(C1s), 1e-30)
            s1 = np.sign(C1s); s1[s1 == 0] = 1.0

            g1_1s = d1[:, 0] * v1[:, 1] - d1[:, 1] * v1[:, 0]
            g2_1s = u1[:, 0] * d2[:, 1] - u1[:, 1] * d2[:, 0]
            d1xd2s = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]

            H_11_sub1 = (g1_1s**2 - (s1 * g1_1s)**2) / (2.0 * normC1)
            H_22_sub1 = (g2_1s**2 - (s1 * g2_1s)**2) / (2.0 * normC1)
            H_12_sub1 = ((g1_1s * g2_1s - (s1 * g1_1s) * (s1 * g2_1s)) / normC1
                         + s1 * d1xd2s) / 2.0

            # Sub-triangle 2
            C2s = u2[:, 0] * v2[:, 1] - u2[:, 1] * v2[:, 0]
            normC2 = np.maximum(np.abs(C2s), 1e-30)
            s2 = np.sign(C2s); s2[s2 == 0] = 1.0

            g2_2s = d2[:, 0] * v2[:, 1] - d2[:, 1] * v2[:, 0]

            H_22_sub2 = (g2_2s**2 - (s2 * g2_2s)**2) / (2.0 * normC2)

            H_11 = H_11_sub1
            H_22 = H_22_sub1 + H_22_sub2
            H_12 = H_12_sub1

        assert pa.btri2_hess_off_aa is not None, \
            "PartitionArrays missing Phase A Hessian offset fields — call compile_arrays() first"
        active = pa.btri2_cell_active
        if np.any(active):
            cells_active = pa.btri_cell[m2][active]
            mu = multipliers[cells_active]
            np.add.at(values, pa.btri2_hess_off_aa[active], mu * H_11[active])
            np.add.at(values, pa.btri2_hess_off_bb[active], mu * H_22[active])
            np.add.at(values, pa.btri2_hess_off_ab[active], mu * H_12[active])

    return values


def compute_area_jacobian(pa: PartitionArrays,
                          eps: float = 1e-7) -> np.ndarray:
    """Compute area constraint Jacobian via analytical derivatives.

    Matches AreaCalculator.compute_area_jacobian().  Steiner Jacobian
    contributions from vectorized_steiner must be added separately.

    The *eps* parameter is unused but kept for backward compatibility.

    Returns:
        (n_cells - 1, n_active_vp) float64
    """
    return compute_area_jacobian_analytical(pa, eps)
