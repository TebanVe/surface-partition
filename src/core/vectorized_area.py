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
