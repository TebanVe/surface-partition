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
