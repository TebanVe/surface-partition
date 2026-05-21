"""
Vectorized Steiner (triple-point) computation on flat PartitionArrays.

The Fermat-Torricelli point and the Steiner perimeter / area *values* are
closed-form.  The *derivatives* exist in two implementations:

* ``*_analytical`` — closed-form derivatives obtained by applying the
  implicit-function theorem to the Fermat-point optimality condition
  ``sum_i n_i = 0`` (see docs/math/03-analytical-steiner-derivatives).
* ``*_fd_reference`` — finite-difference derivatives, retained as the
  independent regression oracle for the analytical code.

The public names (``compute_steiner_perimeter_gradient``,
``compute_steiner_area_jacobian`` and its ``_sparse`` twin) dispatch on the
module flag ``USE_ANALYTICAL_STEINER``.

All functions operate on :class:`~partition_arrays.PartitionArrays`.
"""

from dataclasses import dataclass

import numpy as np

from .partition_arrays import PartitionArrays
from .vectorized_perimeter import _compute_vp_positions, _triangle_areas_batch


# Dispatch flag: when True the public derivative functions use the analytical
# closed-form code; when False they fall back to the finite-difference
# reference.  Exposed for A/B testing and for the regression tests.
USE_ANALYTICAL_STEINER = True

# When True, the analytical code runs cheap structural self-checks (e.g. the
# translation-invariance identity sum_k dS/dp_k = I).  Off by default to keep
# the optimizer hot path assertion-free; the regression tests turn it on.
_STEINER_SELF_CHECK = False

# Distances below this are treated as zero when normalising u_i = p_i - S.
# Only ever bites degenerate rows (S sits on a vertex), which are masked out
# of the M-inverse path anyway.
_R_FLOOR = 1e-12


# =========================================================================
# Triple-point geometry
# =========================================================================

@dataclass
class TPGeometry:
    """Per-triple-point geometry consumed by the analytical derivative code.

    Shapes use ``t`` = n_triple_points, ``dim`` = embedding dimension (3).
    Slot index ``k`` in {0,1,2} is the position within ``tp_vp_indices[t]``.
    """

    p: np.ndarray            # (t, 3, dim)      VP positions
    S: np.ndarray            # (t, dim)         Fermat point (obtuse vertex if degenerate)
    u: np.ndarray            # (t, 3, dim)      u_k = p_k - S
    r: np.ndarray            # (t, 3)           r_k = ||u_k||
    n: np.ndarray            # (t, 3, dim)      n_k = u_k / r_k (0 on a degenerate obtuse slot)
    K: np.ndarray            # (t, 3, dim, dim) K_k = (I - n_k n_k^T) / r_k
    M: np.ndarray            # (t, dim, dim)    M = K_0 + K_1 + K_2
    d: np.ndarray            # (t, 3, dim)      edge direction d_k = V[v1_k] - V[v2_k]
    degen_mask: np.ndarray   # (t,) bool        True where any triangle angle >= 120 deg
    obtuse_idx: np.ndarray   # (t,) int32       slot of the obtuse vertex, -1 if not degenerate


def _compute_tp_steiner(pa: PartitionArrays):
    """Core Fermat-Torricelli evaluation and degeneracy detection.

    This is the single source of the Steiner point *and* the degenerate-case
    branch: both :func:`compute_steiner_points` (forward value) and
    :func:`_compute_tp_geometry` (derivatives) consume it, so the forward
    point and its derivatives can never sit on inconsistent branches.

    Returns:
        p          (n_tp, 3, dim) — the three VP positions per triple point
        steiner    (n_tp, dim)    — Fermat point (obtuse vertex on degenerate rows)
        degen_mask (n_tp,) bool   — True where any triangle angle >= 120 deg
        obtuse_idx (n_tp,) int32  — slot {0,1,2} of the obtuse vertex, -1 otherwise
    """
    dim = pa.vertices.shape[1]
    n_tp = pa.n_triple_points
    if n_tp == 0:
        return (np.empty((0, 3, dim)), np.empty((0, dim)),
                np.empty(0, dtype=bool), np.empty(0, dtype=np.int32))

    pos = _compute_vp_positions(pa)
    p = pos[pa.tp_vp_indices]                    # (n_tp, 3, dim)
    p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]

    # Side lengths opposite each vertex
    a = np.linalg.norm(p2 - p3, axis=1)          # opposite p1
    b = np.linalg.norm(p1 - p3, axis=1)          # opposite p2
    c = np.linalg.norm(p1 - p2, axis=1)          # opposite p3

    # Angles via the law of cosines
    cos_A = np.clip((b**2 + c**2 - a**2) / (2.0 * b * c + 1e-30), -1.0, 1.0)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2.0 * a * c + 1e-30), -1.0, 1.0)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2.0 * a * b + 1e-30), -1.0, 1.0)
    A_ang = np.arccos(cos_A)
    B_ang = np.arccos(cos_B)
    C_ang = np.arccos(cos_C)

    # Barycentric weights: w_i = side_i / sin(angle_i + pi/3)
    pi_over_3 = np.pi / 3.0
    w1 = a / np.maximum(np.sin(A_ang + pi_over_3), 1e-15)
    w2 = b / np.maximum(np.sin(B_ang + pi_over_3), 1e-15)
    w3 = c / np.maximum(np.sin(C_ang + pi_over_3), 1e-15)
    w_sum = w1 + w2 + w3
    steiner = (w1[:, None] * p1 + w2[:, None] * p2
               + w3[:, None] * p3) / w_sum[:, None]

    # Degenerate case: any angle >= 120 deg => Steiner point = obtuse vertex
    threshold = 2.0 * np.pi / 3.0
    all_angles = np.stack([A_ang, B_ang, C_ang], axis=1)
    degen_mask = all_angles.max(axis=1) >= threshold
    obtuse_idx = np.full(n_tp, -1, dtype=np.int32)
    if np.any(degen_mask):
        obtuse_idx[degen_mask] = np.argmax(all_angles[degen_mask], axis=1)
        steiner[degen_mask] = p[degen_mask, obtuse_idx[degen_mask]]

    return p, steiner, degen_mask, obtuse_idx


def compute_steiner_points(pa: PartitionArrays) -> np.ndarray:
    """Compute all Steiner points via the analytical Fermat-Torricelli formula.

    Returns:
        (n_triple_points, dim) float64.  For triple points where any void
        angle >= 120 deg, returns the obtuse vertex position (degenerate case).
    """
    if pa.n_triple_points == 0:
        return np.empty((0, pa.vertices.shape[1]))
    _, steiner, _, _ = _compute_tp_steiner(pa)
    return steiner


def _compute_tp_geometry(pa: PartitionArrays) -> TPGeometry:
    """Per-triple-point geometry for the analytical derivative code.

    Reuses :func:`_compute_tp_steiner` so degeneracy detection is shared with
    the forward value.  On degenerate rows ``M`` is meaningless (one r_k is
    zero) — downstream code masks those rows before inverting ``M``.
    """
    p, S, degen_mask, obtuse_idx = _compute_tp_steiner(pa)
    n_tp = p.shape[0]
    dim = pa.vertices.shape[1]

    if n_tp == 0:
        empty = np.empty((0, 3, dim))
        return TPGeometry(
            p=empty, S=np.empty((0, dim)), u=empty,
            r=np.empty((0, 3)), n=empty,
            K=np.empty((0, 3, dim, dim)), M=np.empty((0, dim, dim)),
            d=empty, degen_mask=degen_mask, obtuse_idx=obtuse_idx)

    u = p - S[:, None, :]                            # (n_tp, 3, dim)
    r = np.linalg.norm(u, axis=2)                    # (n_tp, 3)
    r_safe = np.maximum(r, _R_FLOOR)
    n = u / r_safe[:, :, None]                       # 0 on a degenerate obtuse slot
    eye = np.eye(dim)
    nnT = np.einsum('tki,tkj->tkij', n, n)           # (n_tp, 3, dim, dim)
    K = (eye[None, None, :, :] - nnT) / r_safe[:, :, None, None]
    M = K.sum(axis=1)                                # (n_tp, dim, dim)
    d = (pa.vertices[pa.vp_edge_v1[pa.tp_vp_indices]]
         - pa.vertices[pa.vp_edge_v2[pa.tp_vp_indices]])   # (n_tp, 3, dim)

    return TPGeometry(p=p, S=S, u=u, r=r, n=n, K=K, M=M, d=d,
                      degen_mask=degen_mask, obtuse_idx=obtuse_idx)


# =========================================================================
# Steiner perimeter / area values
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
# Analytical derivative of the Steiner point
# =========================================================================

def _compute_dS_dp(geom: TPGeometry) -> np.ndarray:
    """First derivative of the Steiner point: dS/dp_k.

    Non-degenerate rows: dS/dp_k = M^{-1} K_k, obtained with np.linalg.solve
    (never an explicit inverse — better conditioning near the 120 deg case).
    Degenerate rows: S equals the obtuse vertex identically, so
    dS/dp_k = I if k == obtuse else 0.

    Returns:
        (n_tp, 3, dim, dim) — dS_dp[t, k, :, :] = d S_t / d p_{t,k}.
    """
    M = geom.M
    n_tp = M.shape[0]
    dim = M.shape[1] if n_tp else geom.p.shape[2]
    dS_dp = np.zeros((n_tp, 3, dim, dim), dtype=np.float64)
    if n_tp == 0:
        return dS_dp

    non_degen = ~geom.degen_mask
    if np.any(non_degen):
        M_nd = M[non_degen]                          # (n_nd, dim, dim)
        for k in range(3):
            dS_dp[non_degen, k] = np.linalg.solve(M_nd, geom.K[non_degen, k])

    eye = np.eye(dim)
    for row in np.where(geom.degen_mask)[0]:
        dS_dp[row, geom.obtuse_idx[row]] = eye

    if _STEINER_SELF_CHECK:
        # Translation invariance: translating all three p_k together
        # translates S identically, hence sum_k dS/dp_k = I (all rows).
        err = float(np.max(np.abs(dS_dp.sum(axis=1) - eye)))
        assert err < 1e-6, (
            "Steiner dS/dp translation-invariance check failed: "
            f"max|sum_k dS/dp_k - I| = {err:.3e}")

    return dS_dp


def _compute_dS_dlambda(geom: TPGeometry, dS_dp: np.ndarray) -> np.ndarray:
    """Lift dS/dp to lambda-space: dS/dlambda_k = (dS/dp_k) . d_k.

    Each variable point carries its own scalar lambda, so the chain rule
    contracts the input-dim axis of dS/dp_k with the edge direction d_k.

    Returns:
        (n_tp, 3, dim) — dS_dl[t, k, :] = d S_t / d lambda_{t,k}.
    """
    if dS_dp.shape[0] == 0:
        return np.empty((0, 3, geom.p.shape[2] if geom.p.ndim == 3 else 0))
    return np.einsum('tkij,tkj->tki', dS_dp, geom.d)


# =========================================================================
# Analytical second derivative of the Steiner point
# =========================================================================

def _batched_solve(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Solve M Y = X batched over axis 0.

    M has shape (m, d, d); X has shape (m, d, *extra).  The solve acts on
    axis 1 of X; the trailing axes are flattened and restored.
    """
    m, d = M.shape[0], M.shape[1]
    extra = X.shape[2:]
    Y = np.linalg.solve(M, X.reshape(m, d, -1))
    return Y.reshape(m, d, *extra)


def _compute_d2S_dp2(geom: TPGeometry, dS_dp: np.ndarray) -> np.ndarray:
    """Second derivative of the Steiner point: d2S/dp_k dp_l.

    Index convention for the returned tensor ``d2S[t, k, l, x, b, c]``:
        x = output component of S
        b = component of p_k
        c = component of p_l
    so ``d2S[t,k,l,x,b,c]`` = d^2 S_x / d p_{k,b} d p_{l,c}.

    Non-degenerate rows use the implicit-function-theorem formula
        d2S/dp_k dp_l = -M^{-1} (dM/dp_l) M^{-1} K_k + M^{-1} (dK_k/dp_l).
    Degenerate rows: S equals the obtuse vertex identically, so d2S = 0.

    Every contraction is an explicit np.einsum: the rank-3 dK/dp tensors are
    outer products over the matrix and derivative slots, NOT matrix products.

    Returns:
        (n_tp, 3, 3, dim, dim, dim) float64.
    """
    M = geom.M
    n_tp = M.shape[0]
    dim = M.shape[1] if n_tp else geom.p.shape[2]
    d2S = np.zeros((n_tp, 3, 3, dim, dim, dim), dtype=np.float64)
    if n_tp == 0:
        return d2S

    non_degen = ~geom.degen_mask
    if not np.any(non_degen):
        return d2S

    M_nd = M[non_degen]                          # (m, dim, dim)
    K_nd = geom.K[non_degen]                     # (m, 3, dim, dim)
    n_nd = geom.n[non_degen]                     # (m, 3, dim)
    r_nd = geom.r[non_degen]                     # (m, 3)
    dSdp = dS_dp[non_degen]                      # (m, 3, dim, dim) = dS/dp_k [x,b]
    m = M_nd.shape[0]
    eye = np.eye(dim)

    # T[j,l] = du_j/dp_l = delta_{jl} I - dS/dp_l ;  index [m,j,l,b,c]
    T = -np.broadcast_to(dSdp[:, None, :, :, :], (m, 3, 3, dim, dim)).copy()
    for j in range(3):
        T[:, j, j] += eye

    # KT[j,l] = K_j @ T[j,l] = dn_j/dp_l ;  index [m,j,l,a,c]
    KT = np.einsum('mjab,mjlbc->mjlac', K_nd, T)
    # dr[j,l] = n_j . T[j,l] = dr_j/dp_l ;  index [m,j,l,c]
    dr = np.einsum('mjb,mjlbc->mjlc', n_nd, T)

    # dK_j/dp_l = -(1/r_j)[ (K_j T)(x)n_j + n_j(x)(K_j T) + K_j(x)dr_j ]
    # — each (x) is an outer product over the (b) and (c) slots, index [m,j,l,a,b,c].
    inv_r = (1.0 / r_nd)[:, :, None, None, None, None]
    term1 = np.einsum('mjlac,mjb->mjlabc', KT, n_nd)
    term2 = np.einsum('mja,mjlbc->mjlabc', n_nd, KT)
    term3 = np.einsum('mjab,mjlc->mjlabc', K_nd, dr)
    dK = -inv_r * (term1 + term2 + term3)        # (m, 3, 3, dim, dim, dim)

    # dM/dp_l = sum_j dK_j/dp_l ;  index [m,l,a,b,c]
    dM = dK.sum(axis=1)

    # d2S/dp_k dp_l = -M^{-1} (dM/dp_l)(M^{-1}K_k) + M^{-1}(dK_k/dp_l),
    # with M^{-1}K_k = dS/dp_k already in hand.  solve(), never inv().
    d2S_nd = np.zeros((m, 3, 3, dim, dim, dim), dtype=np.float64)
    for k in range(3):
        for l in range(3):
            # MA[y,b,c] = sum_z (dM/dp_l)[y,z,c] (dS/dp_k)[z,b]
            MA = np.einsum('myzc,mzb->mybc', dM[:, l], dSdp[:, k])
            term_a = -_batched_solve(M_nd, MA)
            term_b = _batched_solve(M_nd, dK[:, k, l])
            d2S_nd[:, k, l] = term_a + term_b

    d2S[non_degen] = d2S_nd

    if _STEINER_SELF_CHECK:
        # Second-order translation invariance: sum_k d2S/dp_k dp_l = 0 per l.
        err = float(np.max(np.abs(d2S.sum(axis=1))))
        assert err < 1e-5, (
            "Steiner d2S/dp2 translation check failed: "
            f"max|sum_k d2S/dp_k dp_l| = {err:.3e}")
        # Symmetry of mixed partials: swapping k<->l also swaps the
        # p_k-component (b) and p_l-component (c) axes.  The threshold is
        # loose — this guards against gross O(1) einsum bugs, not roundoff.
        sym = float(np.max(np.abs(d2S - d2S.transpose(0, 2, 1, 3, 5, 4))))
        assert sym < 1e-4, f"d2S/dp2 mixed-partial symmetry violated: {sym:.3e}"

    return d2S


def _compute_d2S_dlambda(geom: TPGeometry, d2S_dp2: np.ndarray) -> np.ndarray:
    """Lift d2S/dp_k dp_l to lambda-space.

    d2S/dlambda_i dlambda_j = (d2S/dp_i dp_j) contracted with d_i and d_j
    (each variable point carries its own scalar lambda, so k(i)=i).

    Returns:
        (n_tp, 3, 3, dim) float64 — d2S_dl[t,i,j,:] = d^2 S / d lambda_i d lambda_j.
    """
    if d2S_dp2.shape[0] == 0:
        dim = geom.p.shape[2] if geom.p.ndim == 3 else 0
        return np.empty((0, 3, 3, dim))
    return np.einsum('tijxbc,tib,tjc->tijx', d2S_dp2, geom.d, geom.d)


# =========================================================================
# Analytical Steiner perimeter gradient
# =========================================================================

def compute_steiner_perimeter_gradient_analytical(pa: PartitionArrays) -> np.ndarray:
    """d(Steiner perimeter)/d(lambda) in closed form.

    Per (triple point, cell) row the contribution is l_a + l_b - l_ab with
    l_a = ||p_a - S||, l_b = ||p_b - S||, l_ab = ||p_a - p_b||.  The legs
    l_a, l_b depend on all three lambda of the triple point (through S); the
    chord l_ab depends only on lambda_a, lambda_b.

    Returns:
        (n_active_vp,) float64
    """
    gradient = np.zeros(pa.n_active_vp, dtype=np.float64)
    if pa.n_triple_points == 0:
        return gradient

    geom = _compute_tp_geometry(pa)
    dS_dp = _compute_dS_dp(geom)
    dS_dl = _compute_dS_dlambda(geom, dS_dp)          # (n_tp, 3, dim)

    tp = pa.tp_contrib_tp_idx                          # (R,)
    slot_a = pa.tp_contrib_slot1                       # (R,)
    slot_b = pa.tp_contrib_slot2                       # (R,)

    n_a = geom.n[tp, slot_a]                           # (R, dim) leg-a unit vector
    n_b = geom.n[tp, slot_b]                           # (R, dim) leg-b unit vector
    vp_slots = pa.tp_vp_indices[tp]                    # (R, 3) global VP indices
    d_rows = geom.d[tp]                                # (R, 3, dim)
    dSl_rows = dS_dl[tp]                               # (R, 3, dim)

    # Legs l_a, l_b: contribute to all three slots of the triple point.
    for s in range(3):
        d_s = d_rows[:, s]                             # (R, dim)
        dSl_s = dSl_rows[:, s]                         # (R, dim)
        dpa = np.where((slot_a == s)[:, None], d_s, 0.0)   # d p_a / d lambda_s
        dpb = np.where((slot_b == s)[:, None], d_s, 0.0)   # d p_b / d lambda_s
        dla = np.sum(n_a * (dpa - dSl_s), axis=1)      # d l_a / d lambda_s
        dlb = np.sum(n_b * (dpb - dSl_s), axis=1)      # d l_b / d lambda_s
        np.add.at(gradient, vp_slots[:, s], dla + dlb)

    # Chord l_ab: depends only on p_a, p_b — the regular perimeter-segment
    # gradient, entered with coefficient -1.
    p_a = geom.p[tp, slot_a]
    p_b = geom.p[tp, slot_b]
    delta = p_a - p_b
    length = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
    d_a = geom.d[tp, slot_a]
    d_b = geom.d[tp, slot_b]
    dlab_da = np.sum(delta * d_a, axis=1) / length
    dlab_db = -np.sum(delta * d_b, axis=1) / length
    np.add.at(gradient, pa.tp_contrib_vp1, -dlab_da)
    np.add.at(gradient, pa.tp_contrib_vp2, -dlab_db)

    return gradient


# =========================================================================
# Analytical Steiner area Jacobian
# =========================================================================

def _assert_dim3(pa: PartitionArrays) -> None:
    dim = pa.vertices.shape[1]
    if dim != 3:
        raise NotImplementedError(
            f"Analytical Steiner area derivatives require dim=3 (got dim={dim}). "
            "Set vectorized_steiner.USE_ANALYTICAL_STEINER=False for 2-D meshes.")


def _steiner_area_jac_entries(pa: PartitionArrays, geom: TPGeometry,
                              dS_dl: np.ndarray):
    """Per (triple point, cell, slot) area-derivative entries.

    The cell area at a triple point is void(p_a, p_b, S) + corner(mv, p_a, p_b).
    The corner triangle has a fixed mesh vertex, so it only depends on
    lambda_a, lambda_b; the void triangle has all three vertices moving (S
    depends on every lambda of the triple point).

    Returns:
        (cells, vps, values) flat int32/int32/float64 arrays of length 3*R.
    """
    tp = pa.tp_contrib_tp_idx                          # (R,)
    slot_a = pa.tp_contrib_slot1
    slot_b = pa.tp_contrib_slot2
    R = len(tp)

    p_a = geom.p[tp, slot_a]                           # (R, dim)
    p_b = geom.p[tp, slot_b]
    S = geom.S[tp]
    mv = pa.vertices[pa.tp_contrib_mesh_vertex]
    d_rows = geom.d[tp]                                # (R, 3, dim)
    dSl_rows = dS_dl[tp]                               # (R, 3, dim)
    vp_slots = pa.tp_vp_indices[tp]                    # (R, 3)

    # Fixed triangle geometry (same for all three slot derivatives).
    void_e1 = p_b - p_a
    void_e2 = S - p_a
    void_C = np.cross(void_e1, void_e2)
    void_nhat = void_C / np.maximum(
        np.linalg.norm(void_C, axis=1, keepdims=True), 1e-30)

    corner_e1 = p_a - mv
    corner_e2 = p_b - mv
    corner_C = np.cross(corner_e1, corner_e2)
    corner_nhat = corner_C / np.maximum(
        np.linalg.norm(corner_C, axis=1, keepdims=True), 1e-30)

    cells = np.empty(3 * R, dtype=np.int32)
    vps = np.empty(3 * R, dtype=np.int32)
    values = np.zeros(3 * R, dtype=np.float64)

    for s in range(3):
        d_s = d_rows[:, s]
        dSl_s = dSl_rows[:, s]
        dpa = np.where((slot_a == s)[:, None], d_s, 0.0)   # d p_a / d lambda_s
        dpb = np.where((slot_b == s)[:, None], d_s, 0.0)   # d p_b / d lambda_s

        # void area: dA = 0.5 nhat . (de1 x e2 + e1 x de2)
        d_e1 = dpb - dpa
        d_e2 = dSl_s - dpa
        d_voidC = np.cross(d_e1, void_e2) + np.cross(void_e1, d_e2)
        d_void = 0.5 * np.sum(void_nhat * d_voidC, axis=1)

        # corner area: fixed mesh vertex, de1 = d p_a, de2 = d p_b
        d_cornerC = np.cross(dpa, corner_e2) + np.cross(corner_e1, dpb)
        d_corner = 0.5 * np.sum(corner_nhat * d_cornerC, axis=1)

        seg = slice(s * R, (s + 1) * R)
        cells[seg] = pa.tp_contrib_cell
        vps[seg] = vp_slots[:, s]
        values[seg] = d_void + d_corner

    return cells, vps, values


def compute_steiner_area_jacobian_analytical(pa: PartitionArrays) -> np.ndarray:
    """Steiner area Jacobian in closed form — dense (n_cells-1, n_active_vp)."""
    n_constraints = pa.n_cells - 1
    jacobian = np.zeros((n_constraints, pa.n_active_vp), dtype=np.float64)
    if pa.n_triple_points == 0:
        return jacobian
    _assert_dim3(pa)

    geom = _compute_tp_geometry(pa)
    dS_dp = _compute_dS_dp(geom)
    dS_dl = _compute_dS_dlambda(geom, dS_dp)
    cells, vps, values = _steiner_area_jac_entries(pa, geom, dS_dl)

    mask = cells < n_constraints
    np.add.at(jacobian, (cells[mask], vps[mask]), values[mask])
    return jacobian


def compute_steiner_area_jacobian_sparse_analytical(pa: PartitionArrays) -> np.ndarray:
    """Steiner area Jacobian in closed form — sparse values in jac_row order."""
    nnz = len(pa.jac_row)
    out = np.zeros(nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return out
    _assert_dim3(pa)

    geom = _compute_tp_geometry(pa)
    dS_dp = _compute_dS_dp(geom)
    dS_dl = _compute_dS_dlambda(geom, dS_dp)
    cells, vps, values = _steiner_area_jac_entries(pa, geom, dS_dl)

    n_constraints = pa.n_cells - 1
    mask = cells < n_constraints
    offsets = pa.nnz_lookup[cells[mask], vps[mask]]
    valid = offsets >= 0
    np.add.at(out, offsets[valid], values[mask][valid])
    return out


# =========================================================================
# Analytical Steiner Hessians
# =========================================================================

def _scatter_tp_block(pa: PartitionArrays, values: np.ndarray,
                      tp: np.ndarray, h_block: np.ndarray) -> None:
    """Scatter a per-row symmetric 3x3 Hessian block into the flat values array.

    Only the lower triangle (i >= j in slot space) is scattered — tp_hess_off
    maps both (i,j) and (j,i) to the same flat offset, so scattering all nine
    entries would double-count the off-diagonals.
    """
    tp_off = pa.tp_hess_off[tp]                  # (R, 3, 3)
    for i in range(3):
        for j in range(i + 1):
            np.add.at(values, tp_off[:, i, j], h_block[:, i, j])


def compute_steiner_perimeter_hessian_analytical(pa: PartitionArrays) -> np.ndarray:
    """d2(Steiner perimeter)/d(lambda)^2 in closed form.

    Per (triple point, cell) row the contribution is l_a + l_b - l_ab.  Each
    leg l = ||p - S|| has Hessian  Delta_i^T K' Delta_j + n . Delta_ij  with
    Delta the first derivative of (p - S) and Delta_ij = -d2S/dlambda^2; the
    chord l_ab = ||p_a - p_b|| reuses the regular perimeter-segment Hessian.

    Returns:
        (hess_nnz,) float64
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    geom = _compute_tp_geometry(pa)
    dS_dp = _compute_dS_dp(geom)
    dS_dl = _compute_dS_dlambda(geom, dS_dp)
    d2S_dl = _compute_d2S_dlambda(geom, _compute_d2S_dp2(geom, dS_dp))

    tp = pa.tp_contrib_tp_idx
    slot_a = pa.tp_contrib_slot1
    slot_b = pa.tp_contrib_slot2
    dim = pa.vertices.shape[1]

    d_rows = geom.d[tp]                            # (R, 3, dim)
    dSl_rows = dS_dl[tp]                           # (R, 3, dim)
    d2Sl_rows = d2S_dl[tp]                         # (R, 3, 3, dim)
    slots = np.arange(3)
    slot_a_oh = (slots[None, :] == slot_a[:, None])[:, :, None]   # (R,3,1)
    slot_b_oh = (slots[None, :] == slot_b[:, None])[:, :, None]
    dp_a = slot_a_oh * d_rows                      # (R, 3, dim) d p_a / d lambda_s
    dp_b = slot_b_oh * d_rows
    eye = np.eye(dim)

    # Leg l_a = ||p_a - S||:  Delta_a = d(p_a - S)/dlambda,  K_a = geom.K[slot_a].
    Da = dp_a - dSl_rows
    Ka = geom.K[tp, slot_a]
    n_a = geom.n[tp, slot_a]
    H_a = (np.einsum('rid,rde,rje->rij', Da, Ka, Da)
           - np.einsum('rd,rijd->rij', n_a, d2Sl_rows))

    # Leg l_b.
    Db = dp_b - dSl_rows
    Kb = geom.K[tp, slot_b]
    n_b = geom.n[tp, slot_b]
    H_b = (np.einsum('rid,rde,rje->rij', Db, Kb, Db)
           - np.einsum('rd,rijd->rij', n_b, d2Sl_rows))

    # Chord l_ab = ||p_a - p_b||  (regular perimeter-segment Hessian).
    delta = geom.p[tp, slot_a] - geom.p[tp, slot_b]
    length = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
    dhat = delta / length[:, None]
    Kab = (eye - np.einsum('rd,re->rde', dhat, dhat)) / length[:, None, None]
    Dab = dp_a - dp_b
    H_ab = np.einsum('rid,rde,rje->rij', Dab, Kab, Dab)

    h_block = H_a + H_b - H_ab                     # (R, 3, 3)
    _scatter_tp_block(pa, values, tp, h_block)
    return values


def _tri_area_hessian(e1: np.ndarray, e2: np.ndarray,
                      de1: np.ndarray, de2: np.ndarray,
                      d2e2: np.ndarray) -> np.ndarray:
    """Hessian block of the triangle area 0.5*||e1 x e2|| (3-D).

    Args:
        e1, e2: (R, 3) the two triangle edge vectors from vertex 0.
        de1, de2: (R, 3, 3) per-slot first derivatives (slot axis = 1).
        d2e2: (R, 3, 3, 3) per-slot-pair second derivative of e2 (d2e1 == 0
            for both the void and corner triangles, so it is not an argument).

    Returns:
        (R, 3, 3) the per-slot-pair area Hessian.
    """
    C = np.cross(e1, e2)
    normC = np.maximum(np.linalg.norm(C, axis=1), 1e-30)
    nhat = C / normC[:, None]
    eye = np.eye(C.shape[1])
    KC = (eye - np.einsum('rd,re->rde', nhat, nhat)) / normC[:, None, None]

    # dC[s] = de1[s] x e2 + e1 x de2[s]
    dC = (np.cross(de1, e2[:, None, :])
          + np.cross(e1[:, None, :], de2))                       # (R, 3, dim)
    # d2C[i,j] = de1[i] x de2[j] + de1[j] x de2[i] + e1 x d2e2[i,j]
    d2C = (np.cross(de1[:, :, None, :], de2[:, None, :, :])
           + np.cross(de1[:, None, :, :], de2[:, :, None, :])
           + np.cross(e1[:, None, None, :], d2e2))               # (R, 3, 3, dim)

    return 0.5 * (np.einsum('rid,rde,rje->rij', dC, KC, dC)
                  + np.einsum('rd,rijd->rij', nhat, d2C))


def compute_steiner_area_hessian_analytical(pa: PartitionArrays,
                                            multipliers: np.ndarray) -> np.ndarray:
    """Multiplier-weighted d2(Steiner area)/d(lambda)^2 in closed form.

    Per (triple point, cell) row the area is void(p_a, p_b, S) +
    corner(mesh_vertex, p_a, p_b).  The void triangle has all three vertices
    moving (S depends on every lambda); the corner triangle has a fixed mesh
    vertex.  Each cell's block is weighted by multipliers[cell].

    Returns:
        (hess_nnz,) float64
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values
    _assert_dim3(pa)

    geom = _compute_tp_geometry(pa)
    dS_dp = _compute_dS_dp(geom)
    dS_dl = _compute_dS_dlambda(geom, dS_dp)
    d2S_dl = _compute_d2S_dlambda(geom, _compute_d2S_dp2(geom, dS_dp))

    tp = pa.tp_contrib_tp_idx
    slot_a = pa.tp_contrib_slot1
    slot_b = pa.tp_contrib_slot2
    cell = pa.tp_contrib_cell
    R = len(tp)

    p_a = geom.p[tp, slot_a]
    p_b = geom.p[tp, slot_b]
    S = geom.S[tp]
    mv = pa.vertices[pa.tp_contrib_mesh_vertex]
    d_rows = geom.d[tp]
    dSl_rows = dS_dl[tp]
    d2Sl_rows = d2S_dl[tp]
    slots = np.arange(3)
    slot_a_oh = (slots[None, :] == slot_a[:, None])[:, :, None]
    slot_b_oh = (slots[None, :] == slot_b[:, None])[:, :, None]
    dp_a = slot_a_oh * d_rows                      # (R, 3, dim)
    dp_b = slot_b_oh * d_rows
    zero_d2 = np.zeros_like(d2Sl_rows)

    # Void triangle (p_a, p_b, S): e1 = p_b - p_a, e2 = S - p_a.
    void_H = _tri_area_hessian(p_b - p_a, S - p_a,
                               dp_b - dp_a, dSl_rows - dp_a, d2Sl_rows)
    # Corner triangle (mesh_vertex, p_a, p_b): fixed vertex, no d2S term.
    corner_H = _tri_area_hessian(p_a - mv, p_b - mv, dp_a, dp_b, zero_d2)

    h_block = void_H + corner_H                    # (R, 3, 3)

    # Multiplier weight; only constrained cells (cell < n_cells-1) contribute.
    mu = np.zeros(R, dtype=np.float64)
    active = cell < (pa.n_cells - 1)
    mu[active] = multipliers[cell[active]]
    h_block = h_block * mu[:, None, None]

    _scatter_tp_block(pa, values, tp, h_block)
    return values


# =========================================================================
# Finite-difference reference derivatives (regression oracle)
# =========================================================================

def compute_steiner_perimeter_gradient_fd_reference(pa: PartitionArrays,
                                                    eps: float = 1e-6,
                                                    _prof=None,
                                                    _prof_key: str = 'gradient'
                                                    ) -> np.ndarray:
    """d(steiner_perimeter)/d(lambda) via forward finite differences."""
    gradient = np.zeros(pa.n_active_vp, dtype=np.float64)
    if pa.n_triple_points == 0:
        return gradient

    base_steiner = compute_steiner_points(pa)
    if _prof is not None:
        _prof.record_steiner(_prof_key, 1)
    base_perim = compute_steiner_perimeter(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        if _prof is not None:
            _prof.record_steiner(_prof_key, 1)
        pert_perim = compute_steiner_perimeter(pa, pert_steiner)
        gradient[vp_idx] = (pert_perim - base_perim) / eps
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return gradient


def compute_steiner_area_jacobian_fd_reference(pa: PartitionArrays,
                                               eps: float = 1e-7,
                                               _prof=None,
                                               _prof_key: str = 'jacobian'
                                               ) -> np.ndarray:
    """d(steiner_areas)/d(lambda) via finite differences — dense.

    Returns:
        (n_cells - 1, n_active_vp) float64
    """
    n_constraints = pa.n_cells - 1
    jacobian = np.zeros((n_constraints, pa.n_active_vp), dtype=np.float64)
    if pa.n_triple_points == 0:
        return jacobian

    base_steiner = compute_steiner_points(pa)
    if _prof is not None:
        _prof.record_steiner(_prof_key, 1)
    base_areas = compute_steiner_areas(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        if _prof is not None:
            _prof.record_steiner(_prof_key, 1)
        pert_areas = compute_steiner_areas(pa, pert_steiner)
        jacobian[:, vp_idx] = (
            pert_areas[:n_constraints] - base_areas[:n_constraints]) / eps
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return jacobian


def compute_steiner_area_jacobian_sparse_fd_reference(pa: PartitionArrays,
                                                      eps: float = 1e-7,
                                                      _prof=None,
                                                      _prof_key: str = 'jacobian'
                                                      ) -> np.ndarray:
    """Steiner area Jacobian via finite differences — sparse values array.

    Returns:
        (nnz,) float64
    """
    nnz = len(pa.jac_row)
    values = np.zeros(nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    n_constraints = pa.n_cells - 1
    base_steiner = compute_steiner_points(pa)
    if _prof is not None:
        _prof.record_steiner(_prof_key, 1)
    base_areas = compute_steiner_areas(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        if _prof is not None:
            _prof.record_steiner(_prof_key, 1)
        pert_areas = compute_steiner_areas(pa, pert_steiner)
        dA = (pert_areas[:n_constraints] - base_areas[:n_constraints]) / eps

        for c in range(n_constraints):
            offset = pa.nnz_lookup[c, vp_idx]
            if offset >= 0:
                values[offset] += dA[c]

        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return values


# =========================================================================
# Public dispatchers (analytical by default, FD reference as fallback)
# =========================================================================

def compute_steiner_perimeter_gradient(pa: PartitionArrays,
                                       eps: float = 1e-6,
                                       _prof=None,
                                       _prof_key: str = 'gradient') -> np.ndarray:
    """d(Steiner perimeter)/d(lambda).  Analytical unless USE_ANALYTICAL_STEINER."""
    if USE_ANALYTICAL_STEINER:
        return compute_steiner_perimeter_gradient_analytical(pa)
    return compute_steiner_perimeter_gradient_fd_reference(
        pa, eps=eps, _prof=_prof, _prof_key=_prof_key)


def compute_steiner_area_jacobian(pa: PartitionArrays,
                                  eps: float = 1e-7,
                                  _prof=None,
                                  _prof_key: str = 'jacobian') -> np.ndarray:
    """d(Steiner areas)/d(lambda), dense.  Analytical unless USE_ANALYTICAL_STEINER."""
    if USE_ANALYTICAL_STEINER:
        return compute_steiner_area_jacobian_analytical(pa)
    return compute_steiner_area_jacobian_fd_reference(
        pa, eps=eps, _prof=_prof, _prof_key=_prof_key)


def compute_steiner_area_jacobian_sparse(pa: PartitionArrays,
                                         eps: float = 1e-7,
                                         _prof=None,
                                         _prof_key: str = 'jacobian') -> np.ndarray:
    """d(Steiner areas)/d(lambda), sparse.  Analytical unless USE_ANALYTICAL_STEINER."""
    if USE_ANALYTICAL_STEINER:
        return compute_steiner_area_jacobian_sparse_analytical(pa)
    return compute_steiner_area_jacobian_sparse_fd_reference(
        pa, eps=eps, _prof=_prof, _prof_key=_prof_key)


# =========================================================================
# Steiner Hessians — finite-difference reference (regression oracle)
#
# These central-difference the *public* analytical first-derivative
# dispatchers, so they are single finite differences (no nested FD).  They
# are retained as the independent oracle for the analytical Hessians.
# =========================================================================

def compute_steiner_perimeter_hessian_fd_reference(pa: PartitionArrays,
                                                   eps: float = 1e-5,
                                                   _prof=None) -> np.ndarray:
    """Steiner perimeter Hessian via central FD on the analytical gradient.

    A single central finite difference (the inner gradient is exact).  Only
    tp_affected_vps have non-zero entries.

    Returns:
        (hess_nnz,) float64
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    original = pa.vp_lambda.copy()
    affected = pa.tp_affected_vps

    for vp_i in affected:
        pa.vp_lambda[vp_i] = original[vp_i] + eps
        grad_plus = compute_steiner_perimeter_gradient(
            pa, _prof=_prof, _prof_key='hessian')
        pa.vp_lambda[vp_i] = original[vp_i] - eps
        grad_minus = compute_steiner_perimeter_gradient(
            pa, _prof=_prof, _prof_key='hessian')
        pa.vp_lambda[vp_i] = original[vp_i]

        hess_col_i = (grad_plus - grad_minus) / (2.0 * eps)

        for vp_j in affected:
            if vp_j > vp_i:
                continue
            key = (max(int(vp_i), int(vp_j)), min(int(vp_i), int(vp_j)))
            if key in pa.hess_offset_map:
                values[pa.hess_offset_map[key]] += hess_col_i[vp_j]

    return values


def compute_steiner_area_hessian_fd_reference(pa: PartitionArrays,
                                              multipliers: np.ndarray,
                                              eps: float = 1e-5,
                                              _prof=None) -> np.ndarray:
    """Multiplier-weighted Steiner area Hessian via central FD on the Jacobian.

    A single central finite difference (the inner Jacobian is exact).
    Retained as the regression oracle for the analytical area Hessian.

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

    for vp_i in affected:
        pa.vp_lambda[vp_i] = original[vp_i] + eps
        jac_plus = compute_steiner_area_jacobian(pa, _prof=_prof, _prof_key='hessian')
        pa.vp_lambda[vp_i] = original[vp_i] - eps
        jac_minus = compute_steiner_area_jacobian(pa, _prof=_prof, _prof_key='hessian')
        pa.vp_lambda[vp_i] = original[vp_i]

        djac = (jac_plus - jac_minus) / (2.0 * eps)

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


# =========================================================================
# Public Hessian dispatchers (analytical by default, FD reference fallback)
# =========================================================================

def compute_steiner_perimeter_hessian(pa: PartitionArrays,
                                      eps: float = 1e-5,
                                      _prof=None) -> np.ndarray:
    """Steiner perimeter Hessian.  Analytical unless USE_ANALYTICAL_STEINER."""
    if USE_ANALYTICAL_STEINER:
        return compute_steiner_perimeter_hessian_analytical(pa)
    return compute_steiner_perimeter_hessian_fd_reference(pa, eps=eps, _prof=_prof)


def compute_steiner_area_hessian(pa: PartitionArrays,
                                 multipliers: np.ndarray,
                                 eps: float = 1e-5,
                                 _prof=None) -> np.ndarray:
    """Multiplier-weighted Steiner area Hessian.  Analytical unless flag off."""
    if USE_ANALYTICAL_STEINER:
        return compute_steiner_area_hessian_analytical(pa, multipliers)
    return compute_steiner_area_hessian_fd_reference(
        pa, multipliers, eps=eps, _prof=_prof)
