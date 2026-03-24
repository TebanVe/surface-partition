"""
Flat array representation of partition state for vectorized evaluation.

This module provides the PartitionArrays dataclass — a frozen snapshot of partition
geometry in contiguous NumPy arrays.  It enables Mode B (evaluation) to run entirely
on flat arrays with no Python-level per-element loops, while Mode A (mutation) continues
to operate on the original object-oriented PartitionContour structures.

Lifecycle:
    1. PartitionContour.compile_arrays(steiner_handler) → PartitionArrays
    2. SLSQP loop reads/writes only pa.vp_lambda; all evaluation code uses PartitionArrays
    3. After optimization, result.x is synced back to PartitionContour via set_variable_vector()
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PartitionArrays:
    """Flat array representation of partition state for vectorized evaluation.

    All VP indices stored here are in *active-index* space (i.e. consecutive
    0..n_active_vp-1).  The ``active_to_absolute`` array maps back to the
    absolute VP indices used by PartitionContour.
    """

    # --- Variable point arrays (length n_active_vp) ---
    vp_edge_v1: np.ndarray       # int32 — edge[0] vertex index
    vp_edge_v2: np.ndarray       # int32 — edge[1] vertex index
    vp_lambda: np.ndarray        # float64 — lambda parameters (mutated during optim)

    # --- Boundary segment arrays (length n_segments) ---
    seg_vp1: np.ndarray          # int32 — first VP (active index)
    seg_vp2: np.ndarray          # int32 — second VP (active index)
    seg_cell_a: np.ndarray       # int32 — first cell of the pair
    seg_cell_b: np.ndarray       # int32 — second cell of the pair

    # --- Boundary triangle arrays (length n_boundary_triangles) ---
    # EXCLUDES triple-point triangles (handled by vectorized_steiner).
    # One row per (triangle, cell) pair where cell has 1 or 2 vertices inside.
    btri_idx: np.ndarray         # int32 — original triangle index in mesh
    btri_cell: np.ndarray        # int32 — cell this row contributes to
    btri_n_inside: np.ndarray    # int32 — 1 or 2
    btri_v_in: np.ndarray        # int32 (n_btri, 2) — inside vertex indices (padded w/ -1)
    btri_v_out: np.ndarray       # int32 (n_btri, 2) — outside vertex indices (padded w/ -1)
    btri_vp1: np.ndarray         # int32 — VP on first cut edge (active index)
    btri_vp2: np.ndarray         # int32 — VP on second cut edge (active index)

    # --- Pre-computed constants ---
    cell_interior_area: np.ndarray  # float64 (n_cells,) — constant interior area per cell
    n_cells: int
    n_active_vp: int

    # --- Index mappings ---
    active_to_absolute: np.ndarray  # int32 (n_active_vp,) — active → absolute VP index
    area_affected_vps: np.ndarray   # int32 — unique VPs appearing in btri_vp1 ∪ btri_vp2

    # --- Mesh vertex coordinates (reference, not copied) ---
    vertices: np.ndarray         # float64 (N, dim)

    # --- Triple-point arrays ---
    tp_vp_indices: np.ndarray      # int32 (n_tp, 3) — active VP indices per triple point
    n_triple_points: int

    # Per (triple_point, cell) contribution rows — length 3 * n_triple_points
    tp_contrib_tp_idx: np.ndarray    # int32 — which triple point
    tp_contrib_cell: np.ndarray      # int32 — cell index
    tp_contrib_vp1: np.ndarray       # int32 — first VP of cell's pair (active)
    tp_contrib_vp2: np.ndarray       # int32 — second VP of cell's pair (active)
    tp_contrib_mesh_vertex: np.ndarray  # int32 — mesh vertex for corner triangle

    tp_affected_vps: np.ndarray    # int32 — unique active VPs in any triple point
