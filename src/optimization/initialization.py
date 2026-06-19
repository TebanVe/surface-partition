"""Phase 1 level-0 initial-condition strategies.

Provides the seeded (Voronoi) initialization, which hands every cell a
contiguous winning region before optimization begins. This is the root-cause
fix for the dormant-cell failure documented in
``docs/reference/phase1_dormant_cell_argmax_issue.md``: with a seeded layout
there is no symmetry-break "land grab" for a cell to lose.

The legacy uniform-random initialization lives in
``create_initial_condition_with_projection`` (``projection.py``); the
level-0 dispatch in ``src/pipeline/relaxation.py`` selects between them.
"""

import numpy as np
from typing import Optional

from scipy.spatial import cKDTree

from ..logging_config import get_logger
from .projection import orthogonal_projection_iterative


def farthest_point_sampling(vertices: np.ndarray, n_seeds: int,
                            seed: Optional[int] = None) -> np.ndarray:
    """Select ``n_seeds`` well-spread vertices by greedy farthest-point sampling.

    Maintains a single running ``(V,)`` min-distance array (no ``V x N``
    matrix). The first seed is drawn from a ``seed``-seeded RNG so the layout
    is deterministic; each subsequent seed is the vertex farthest from the
    current seed set.

    Args:
        vertices: ``(V, 3)`` embedded coordinates (Euclidean distance in R^3).
        n_seeds: number of seeds to select.
        seed: RNG seed for the first pick (deterministic).

    Returns:
        ``(n_seeds,)`` integer array of selected vertex indices.
    """
    n_vertices = vertices.shape[0]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n_vertices))
    seeds = np.empty(n_seeds, dtype=np.intp)
    seeds[0] = first
    d2 = np.sum((vertices - vertices[first]) ** 2, axis=1)
    for i in range(1, n_seeds):
        nxt = int(np.argmax(d2))
        seeds[i] = nxt
        d2 = np.minimum(d2, np.sum((vertices - vertices[nxt]) ** 2, axis=1))
    return seeds


def create_seeded_initial_condition(mesh, n_partitions: int, v: np.ndarray,
                                    seed: Optional[int] = None,
                                    logger=None,
                                    max_iter: int = 100,
                                    tol: float = 1e-8) -> np.ndarray:
    """Create a level-0 Voronoi initial condition projected onto the constraints.

    Each cell is assigned a contiguous Voronoi region around a farthest-point
    seed, so every cell wins a region from iteration 0. The one-hot density is
    then projected with ``orthogonal_projection_iterative`` to enforce
    sum-to-one and equal areas while preserving the dominant regions.

    Args:
        mesh: TriMesh providing ``vertices`` (embedded coordinates).
        n_partitions: number of cells N.
        v: ``(V,)`` lumped mass (row-sum of M).
        seed: RNG seed for deterministic seed-point selection.
        logger: optional logger; defaults to module logger.
        max_iter: projection iteration cap (matches the random init default).
        tol: projection tolerance (matches the random init default).

    Returns:
        Flattened ``(V * N,)`` initial-condition vector.
    """
    if logger is None:
        logger = get_logger(__name__)

    vertices = np.asarray(mesh.vertices)
    n_vertices = vertices.shape[0]

    seeds = farthest_point_sampling(vertices, n_partitions, seed=seed)
    labels = cKDTree(vertices[seeds]).query(vertices)[1]

    A = np.zeros((n_vertices, n_partitions))
    A[np.arange(n_vertices), labels] = 1.0

    c = np.ones(n_partitions)
    d = (np.sum(v) / n_partitions) * np.ones(n_partitions)
    A = orthogonal_projection_iterative(
        A, c, d, v, max_iter=max_iter, tol=tol, logger=logger
    )

    logger.info(
        f"Seeded initial condition: {n_vertices}x{n_partitions}, "
        f"{n_partitions} farthest-point seeds, projection converged"
    )
    return A.flatten()
