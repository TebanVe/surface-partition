"""
Vectorized perimeter computation on flat PartitionArrays.

All functions operate on :class:`~partition_arrays.PartitionArrays` — no
PartitionContour or VP-object access during the SLSQP evaluation loop.

Convention: total perimeter = 2 * sum(segment_lengths) because each segment
separates exactly two cells and the codebase sums per-cell perimeters.
"""

import numpy as np

from .partition_arrays import PartitionArrays


# =========================================================================
# Shared helpers (also imported by vectorized_area / vectorized_steiner)
# =========================================================================

def _compute_vp_positions(pa: PartitionArrays) -> np.ndarray:
    """Compute all active VP positions from lambdas.

    Returns:
        (n_active_vp, dim) float64 — matches VariablePoint.evaluate().
    """
    return (pa.vp_lambda[:, None] * pa.vertices[pa.vp_edge_v1]
            + (1.0 - pa.vp_lambda[:, None]) * pa.vertices[pa.vp_edge_v2])


def _triangle_areas_batch(p1: np.ndarray, p2: np.ndarray,
                          p3: np.ndarray) -> np.ndarray:
    """Batch triangle area for (N, dim) point arrays.

    Matches AreaCalculator._triangle_area_3d() semantics.
    """
    v1 = p2 - p1
    v2 = p3 - p1
    if p1.shape[1] == 2:
        return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    else:
        cross = np.cross(v1, v2)
        return 0.5 * np.linalg.norm(cross, axis=1)


# =========================================================================
# Perimeter
# =========================================================================

def compute_total_perimeter(pa: PartitionArrays) -> float:
    """Compute total perimeter matching PerimeterCalculator.compute_total_perimeter().

    Each proper segment (cell_a != cell_b) is counted twice (once per adjacent
    cell).  Triple-point void segments that received the fallback cell_pair
    ``(0, 0)`` in the original code are counted only once, matching the
    sum-of-cell-perimeters behaviour.
    """
    pos = _compute_vp_positions(pa)
    diff = pos[pa.seg_vp1] - pos[pa.seg_vp2]
    lengths = np.linalg.norm(diff, axis=1)
    weights = np.where(pa.seg_cell_a != pa.seg_cell_b, 2.0, 1.0)
    return float(np.sum(weights * lengths))


# =========================================================================
# Perimeter gradient
# =========================================================================

def compute_perimeter_hessian_sparse(pa: PartitionArrays) -> np.ndarray:
    """Compute d2P/dlam_i dlam_j for the regular perimeter.

    Returns:
        (hess_nnz,) float64 — values at positions (pa.hess_row, pa.hess_col).
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    pos = _compute_vp_positions(pa)

    p1 = pos[pa.seg_vp1]
    p2 = pos[pa.seg_vp2]
    delta = p1 - p2
    r = np.linalg.norm(delta, axis=1)
    r = np.maximum(r, 1e-12)
    delta_hat = delta / r[:, None]

    d_a = (pa.vertices[pa.vp_edge_v1[pa.seg_vp1]]
           - pa.vertices[pa.vp_edge_v2[pa.seg_vp1]])
    d_b = (pa.vertices[pa.vp_edge_v1[pa.seg_vp2]]
           - pa.vertices[pa.vp_edge_v2[pa.seg_vp2]])

    weights = np.where(pa.seg_cell_a != pa.seg_cell_b, 2.0, 1.0)

    da_dot_da = np.sum(d_a * d_a, axis=1)
    db_dot_db = np.sum(d_b * d_b, axis=1)
    da_dot_db = np.sum(d_a * d_b, axis=1)
    da_dot_dh = np.sum(d_a * delta_hat, axis=1)
    db_dot_dh = np.sum(d_b * delta_hat, axis=1)

    H_aa = weights * (da_dot_da - da_dot_dh**2) / r
    H_bb = weights * (db_dot_db - db_dot_dh**2) / r
    H_ab = -weights * (da_dot_db - da_dot_dh * db_dot_dh) / r

    vp1 = pa.seg_vp1
    vp2 = pa.seg_vp2

    for s in range(len(vp1)):
        a, b = int(vp1[s]), int(vp2[s])

        off_aa = pa.hess_offset_map[(a, a)]
        values[off_aa] += H_aa[s]
        off_bb = pa.hess_offset_map[(b, b)]
        values[off_bb] += H_bb[s]

        hi, lo = max(a, b), min(a, b)
        off_ab = pa.hess_offset_map[(hi, lo)]
        values[off_ab] += H_ab[s]

    return values


def compute_perimeter_gradient(pa: PartitionArrays) -> np.ndarray:
    """Compute d(total_perimeter)/d(lambda) for all active VPs.

    Weights match ``compute_total_perimeter``: proper segments (cell_a != cell_b)
    contribute 2×, triple-point fallback segments contribute 1×.
    """
    pos = _compute_vp_positions(pa)
    diff = pos[pa.seg_vp1] - pos[pa.seg_vp2]
    lengths = np.linalg.norm(diff, axis=1)
    lengths = np.maximum(lengths, 1e-12)

    # dx/dlambda = vertices[edge[0]] - vertices[edge[1]]
    dv1 = pa.vertices[pa.vp_edge_v1[pa.seg_vp1]] - pa.vertices[pa.vp_edge_v2[pa.seg_vp1]]
    dv2 = pa.vertices[pa.vp_edge_v1[pa.seg_vp2]] - pa.vertices[pa.vp_edge_v2[pa.seg_vp2]]

    grad_vp1 = np.sum(diff * dv1, axis=1) / lengths
    grad_vp2 = np.sum(-diff * dv2, axis=1) / lengths

    weights = np.where(pa.seg_cell_a != pa.seg_cell_b, 2.0, 1.0)
    grad_vp1 *= weights
    grad_vp2 *= weights

    gradient = np.zeros(pa.n_active_vp, dtype=np.float64)
    np.add.at(gradient, pa.seg_vp1, grad_vp1)
    np.add.at(gradient, pa.seg_vp2, grad_vp2)

    return gradient
