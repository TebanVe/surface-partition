"""Approach B -- auction-dynamics MBO, as a replacement for Phase 1.

Volume-constrained multiphase threshold dynamics. The state is a *hard* labelling
``omega: V -> {0..N-1}``; there is never a continuous field whose winner-take-all
readout can diverge from what was constrained, which is the gap
``docs/reference/winner_take_all_partition_gap.md`` describes. Per step:

1. **diffuse** each indicator for time tau by solving ``(M + tau K) y_k = M chi_k``,
   one prefactorized LU per level and N back-substitutions;
2. **balanced threshold** ``omega(i) = argmax_k [y_ik + psi_k]``, with psi from
   ``solve_dual_offsets`` -- the shared semi-discrete OT dual that approach A
   already ships.

Provenance: threshold dynamics is Merriman-Bence-Osher; the multiphase variational
footing is Esedoglu-Otto; the volume-constrained "auction dynamics" variant is
Jacobs-Kim-Leger (JCP 2018), and Hu-Liu-Wang 2024 (arXiv 2405.16040) apply
thresholding + auction to minimal-length partitions. **Every published
implementation of this machinery convolves on a uniform Cartesian grid via FFT, on
a flat domain, at small N, for Polya's *outer* problem.** This is the *inner*
problem on a closed genus-1 surface with an unstructured FEM discretization at N in
the hundreds, so the diffusion operator above is a different object with different
failure modes -- mesh pinning (van Gennip et al.) is ours to manage, and no
grid-derived tau heuristic is imported.

Two design points that are measurements, not preferences; see
``docs/plans/PHASE1_BC_REPLACEMENT_PLAN.md`` Phase A:

* **The initialization is a HARD REQUIREMENT.** B starts from ONE balanced
  C-scores assignment (farthest-point seeds -> geodesic -d^2 -> balanced
  assignment). A cold, unbalanced start leaves the first diffused assignment at
  68.16% worst area deviation against 1.97% from a pre-balanced one, and gate 2's
  PASS for B is conditional on the balanced start.
* **tau = min((c*h_mean)^2, R_cell^2) with c = 4**, not 2. c=2 is the freeze
  regime. The cap is the over-merge side of a two-sided window: at sqrt(tau) =
  R_cell the diffused indicator has spread over the whole cell. Both
  ``sqrt(tau)/h_max`` (this mesh is 1.81x anisotropic, so h_mean flatters it) and
  ``sqrt(tau)/R_cell`` are reported per level.

**There is no exactly-monotone energy here** -- see ``descent_slack``. Track
``E_tau`` as a descriptive trend; gate on the inequality that survives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import factorized

from ..logging_config import get_logger
from ..optimization.initialization import farthest_point_sampling
from .arm_harness import assignment_quality_bar
from .balanced_readout import BalancedReadoutConfig, solve_dual_offsets

logger = get_logger(__name__)


@dataclass
class MBOConfig:
    """Parameters of the MBO arm. Every numeric constant here is pre-registered."""

    # --- tau, the two-sided window (plan section 3) -------------------------
    tau_c: float = 4.0
    #: Ceiling on sqrt(tau)/R_cell. 1.0 is the loosest defensible value -- at
    #: sqrt(tau) = R_cell the diffused indicator has spread over the whole cell.
    #: Anything tighter is a preference, not a derived constant, so the over-merge
    #: INSTRUMENT (fragmentation / compactness / core loss) adjudicates instead.
    rho: float = 1.0

    # --- per-level convergence ---------------------------------------------
    churn_tol: float = 1e-4
    patience: int = 3
    max_iters: int = 200

    # --- pinning response ---------------------------------------------------
    anneal_factor: float = 1.5
    max_anneals: int = 3

    # --- assignment ---------------------------------------------------------
    #: Early stopping is a PRODUCTION speedup (B calls the solver every step) and
    #: is barred from the measurement path: with it on, the in-loop worst deviation
    #: hugs its bar from below while the full-budget solver reaches materially
    #: better on the same scores (report 07, artefact 5). Every level therefore
    #: ends with a full-budget re-assignment, and in-loop figures are tagged.
    early_stop_in_loop: bool = True
    finalize_full_budget: bool = True

    # --- NC3, the tau-continuation (pinning) probe --------------------------
    #: Geometric ladder: a single doubling has a false-negative regime when the
    #: freeze deficit exceeds 2x.
    probe_factors: Tuple[float, ...] = (2.0, 4.0)
    #: PINNED iff probe dE > nc3_K * median|dE| over the converged tail AND the
    #: moved fraction exceeds nc3_f_min. Both are None until NC3-CAL pins them;
    #: the probe still runs and records, but reports ``pinned=None``.
    nc3_K: Optional[float] = None
    nc3_f_min: Optional[float] = None

    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "tau_c": float(self.tau_c),
            "rho": float(self.rho),
            "churn_tol": float(self.churn_tol),
            "patience": int(self.patience),
            "max_iters": int(self.max_iters),
            "anneal_factor": float(self.anneal_factor),
            "max_anneals": int(self.max_anneals),
            "early_stop_in_loop": bool(self.early_stop_in_loop),
            "finalize_full_budget": bool(self.finalize_full_budget),
            "probe_factors": [float(f) for f in self.probe_factors],
            "nc3_K": None if self.nc3_K is None else float(self.nc3_K),
            "nc3_f_min": None if self.nc3_f_min is None else float(self.nc3_f_min),
            "seed": None if self.seed is None else int(self.seed),
        }


# --------------------------------------------------------------------------
# mesh quantities
# --------------------------------------------------------------------------


def edge_endpoints(faces: np.ndarray) -> np.ndarray:
    """(E, 2) array of triangle-side endpoint pairs (with duplicates)."""
    f = np.asarray(faces)
    return np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])


def edge_lengths(mesh) -> np.ndarray:
    e = edge_endpoints(mesh.faces)
    return np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)


def edge_graph(mesh) -> sp.csr_matrix:
    """Symmetric edge-length-weighted graph on the mesh's own edges.

    Uses ``faces`` directly, so it respects the true surface topology (the torus
    periodic wrap is encoded there), not a flat parametrization.
    """
    e = edge_endpoints(mesh.faces)
    w = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    n = len(mesh.vertices)
    g = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return g.maximum(g.T)


def cell_radius(total_area: float, n_cells: int) -> float:
    """Radius of a disc of the equal-area target. The over-merge length scale."""
    return float(np.sqrt(total_area / n_cells / np.pi))


def tau_diagnostics(
    mesh, n_cells: int, cfg: MBOConfig, tau_override: Optional[float] = None
) -> Dict[str, float]:
    """tau for one level, plus every ratio the plan requires reported.

    The rule is ``tau = min((c*h_mean)^2, (rho*R_cell)^2)``. h_mean is the single
    ruler for both the rule and the window; h_max ratios are reported alongside
    because this mesh is anisotropic and a mean-based figure flatters the coarse
    band. The two rulers can disagree about which levels are at risk -- that
    disagreement is reported, and the calibrated probe (NC3) is the arbiter.
    """
    L = edge_lengths(mesh)
    h_mean, h_max = float(L.mean()), float(L.max())
    total_area = float(mesh.M.sum())
    r_cell = cell_radius(total_area, n_cells)

    tau_uncapped = (cfg.tau_c * h_mean) ** 2
    tau_cap = (cfg.rho * r_cell) ** 2
    tau = min(tau_uncapped, tau_cap)
    # After an anneal the level runs at a DIFFERENT tau, and every ratio below has
    # to move with it -- reporting the un-annealed ratios beside an annealed run
    # would misdescribe exactly the situation the anneal exists to handle.
    if tau_override is not None:
        tau = float(tau_override)
    st = float(np.sqrt(tau))

    return {
        "tau": float(tau),
        "tau_uncapped": float(tau_uncapped),
        "cap_active": bool(tau_cap < tau_uncapped and tau_override is None),
        "annealed": bool(tau_override is not None),
        "c_eff": st / h_mean,
        "h_mean": h_mean,
        "h_max": h_max,
        "anisotropy": h_max / h_mean,
        "r_cell": r_cell,
        "sqrt_tau_over_h_mean": st / h_mean,
        "sqrt_tau_over_h_max": st / h_max,
        "sqrt_tau_over_r_cell": st / r_cell,
        # Two-sided window, on the h_mean ruler.
        "c_lo_hmean": float(np.sqrt(r_cell / h_mean)),
        "c_hi_hmean": float(cfg.rho * r_cell / h_mean),
        # The same non-freeze criterion judged in the coarse band. Reported
        # because it disagrees with the h_mean one, not because it overrides it.
        "c_lo_hmax_local": float(np.sqrt(r_cell / h_max)),
        "sqrt_tau_over_h_max_vs_c_lo_hmax": (st / h_max)
        / float(np.sqrt(r_cell / h_max)),
    }


# --------------------------------------------------------------------------
# diffusion
# --------------------------------------------------------------------------


def make_diffusion_solver(mesh, tau: float) -> Callable[[np.ndarray], np.ndarray]:
    """One prefactorized LU of ``(M + tau K)``, reused for all N solves.

    ``sksparse`` is absent and unnecessary: SuperLU factorizes this in 0.45 s at
    V=47,488 and 2.33 s at V=114,144 on the development machine.
    """
    return factorized((mesh.M + tau * mesh.K).tocsc())


def sparse_one_hot(labels: np.ndarray, n_cells: int) -> sp.csr_matrix:
    """(V, N) sparse indicator. Never materialize chi densely -- at the production
    pin a dense (114144, 300) float64 is 274 MB and two of them are avoidable."""
    v = len(labels)
    return sp.csr_matrix(
        (np.ones(v), (np.arange(v), np.asarray(labels).astype(int))), shape=(v, n_cells)
    )


def diffuse_indicators(
    solve: Callable[[np.ndarray], np.ndarray],
    mesh,
    labels: np.ndarray,
    n_cells: int,
) -> Tuple[np.ndarray, float]:
    """Solve ``(M + tau K) y_k = M chi_k`` for every cell.

    Returns ``(y, inner_consistent)`` where ``inner_consistent = sum_k
    (M chi_k)^T y_k = <chi, M y>`` -- accumulated column by column so the dense
    ``M @ y`` is never formed.
    """
    chi = sparse_one_hot(labels, n_cells)
    rhs = (mesh.M @ chi).tocsc()
    v = len(labels)
    y = np.empty((v, n_cells), dtype=np.float64)
    inner = 0.0
    for k in range(n_cells):
        b = rhs[:, k].toarray().ravel()
        yk = solve(b)
        y[:, k] = yk
        inner += float(b @ yk)
    return y, inner


def lyapunov_energy(
    y: np.ndarray,
    labels: np.ndarray,
    lumped_mass: np.ndarray,
    total_area: float,
    tau: float,
    inner_consistent: Optional[float] = None,
) -> Dict[str, float]:
    """Threshold-dynamics energy, in both discretizations.

    ``E_tau = (A_total - sum_k <chi_k, W y_k>) / sqrt(tau)``. The LUMPED form
    (``W = diag(v)``) is primary because it is what the assignment step actually
    maximizes; the consistent form (``W = M``) is recorded alongside. Neither is a
    gate -- see ``descent_slack``.

    E_tau depends on the LABELS ALONE (given tau), which is what makes the
    tau-continuation probe well posed.
    """
    idx = np.arange(len(labels))
    inner_lumped = float(lumped_mass @ y[idx, labels])
    out = {
        "E_lumped": (total_area - inner_lumped) / np.sqrt(tau),
        "inner_lumped": inner_lumped,
    }
    if inner_consistent is not None:
        out["E_consistent"] = (total_area - inner_consistent) / np.sqrt(tau)
        out["inner_consistent"] = float(inner_consistent)
    return out


# --------------------------------------------------------------------------
# the descent inequality -- what survives of the monotonicity theorem
# --------------------------------------------------------------------------


def descent_slack(
    y: np.ndarray,
    lumped_mass: np.ndarray,
    psi: np.ndarray,
    labels_old: np.ndarray,
    labels_new: np.ndarray,
    n_cells: int,
) -> Dict[str, float]:
    """Check ``<chi',Dy> - <chi,Dy>  >=  sum_k psi_k (T_k - T'_k)``.

    Esedoglu-Otto monotonicity does NOT transfer here, for two independent
    reasons, and it is worth knowing which:

    1. The theorem's object is ``M A_tau = M (M + tau K)^-1 M``, symmetric PSD, so
       ``-<chi_k, A_tau chi_k>`` is concave and threshold dynamics is a
       minorize-maximize step. Our assignment maximizes ``<chi', D y>`` with
       ``D = diag(v)`` LUMPED, and ``D A_tau`` is not symmetric. Tracking one form
       while maximizing the other is simply the wrong pairing.
    2. Jacobs-Kim-Leger get a theorem because Bertsekas' auction solves the
       assignment EXACTLY. ``solve_dual_offsets`` is subgradient ascent on the
       dual and returns its best iterate, with a residual area error.

    What survives needs neither concavity nor exactness. Since
    ``omega'(i) = argmax_k [y_ik + psi_k]``, for every vertex
    ``y_{i,omega'(i)} + psi_{omega'(i)} >= y_{i,omega(i)} + psi_{omega(i)}``;
    multiply by ``v_i > 0`` and sum. The right-hand side is exactly the slack that
    inexactness costs: it vanishes at perfect balance, recovering the textbook
    monotone step, and otherwise scales like ``||psi|| x area error`` -- measured
    at 1.7e-4 to 1.5e-3 relative on this problem, i.e. five to seven orders of
    magnitude above any 1e-9 tolerance. So a *violation* of the inequality beyond
    floating point is a genuine bug (labels not the argmax at the returned psi, or
    a psi/label unit mismatch), which is what makes it worth gating on.
    """
    idx = np.arange(len(labels_old))
    lhs = float(lumped_mass @ (y[idx, labels_new] - y[idx, labels_old]))
    t_old = np.bincount(labels_old, weights=lumped_mass, minlength=n_cells)
    t_new = np.bincount(labels_new, weights=lumped_mass, minlength=n_cells)
    rhs = float(psi @ (t_old - t_new))
    scale = max(abs(float(lumped_mass @ y[idx, labels_old])), 1e-30)
    return {
        "lhs": lhs,
        "rhs": rhs,
        "slack_rel": abs(rhs) / scale,
        "violation_rel": (rhs - lhs) / scale,
    }


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------


def assignment_config(bar: float, early_stop: bool) -> BalancedReadoutConfig:
    """The shared solver, configured for a foreign score scale.

    ``normalize_scores=True`` is the path Phase 0 qualified for B's diffused
    scores; approach A's un-normalized path is untouched and stays byte-identical.
    """
    return BalancedReadoutConfig(
        normalize_scores=True,
        dual_early_stop_rel=(bar if early_stop else None),
    )


def balanced_assign(
    scores: np.ndarray,
    lumped_mass: np.ndarray,
    target: float,
    bar: float,
    early_stop: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """``argmax_k [scores_ik + psi_k]`` with equal discrete areas."""
    cfg = assignment_config(bar, early_stop)
    return solve_dual_offsets(scores, lumped_mass, target, cfg)


def geodesic_balanced_init(
    mesh, n_cells: int, seed: Optional[int]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """The HARD-REQUIRED initialization: one balanced C-scores assignment.

    Farthest-point seeds (the project's own deterministic sampler, so the arm and
    the PGD control start from the same seed layout) -> multi-source Dijkstra on
    the mesh edge graph -> scores ``-d^2`` -> balanced assignment.

    This is also approach C's step 0, which is why the driver can run it alone as
    the attribution control: if MBO barely improves on it, the reported number is
    the init's, not threshold dynamics'.
    """
    t0 = time.perf_counter()
    v, target = mesh.v, float(mesh.v.sum()) / n_cells
    bar = assignment_quality_bar(v, n_cells)
    sites = farthest_point_sampling(mesh.vertices, n_cells, seed=seed)
    d = dijkstra(edge_graph(mesh), indices=sites, directed=False).T
    scores = -(d**2)
    # Full budget: this runs once and sets the state every later step inherits.
    psi, labels, worst = balanced_assign(scores, v, target, bar, early_stop=False)
    return labels, {
        "worst_rel_dev": float(worst),
        "quality_bar": float(bar),
        "wall_seconds": time.perf_counter() - t0,
        "n_sites": int(len(sites)),
        "early_stop_capped": False,
    }


# --------------------------------------------------------------------------
# geometry instruments (the over-merge detector)
# --------------------------------------------------------------------------


def per_cell_boundary_lengths(
    labels: np.ndarray, vertices: np.ndarray, faces: np.ndarray, n_cells: int
) -> np.ndarray:
    """Midpoint-cut label-boundary length attributed to each cell.

    Same construction as ``balanced_readout.label_boundary_length`` (a triangle
    with two labels contributes the segment joining its two bi-labeled edge
    midpoints; one with three contributes three segments meeting at the centroid),
    but each segment is credited to BOTH cells it separates, so the result is a
    per-cell perimeter usable in an isoperimetric ratio.
    """
    out = np.zeros(n_cells)
    L = labels[faces]
    mixed = ~((L[:, 0] == L[:, 1]) & (L[:, 1] == L[:, 2]))
    F = faces[mixed]
    if F.shape[0] == 0:
        return out
    P = vertices[F]
    Lm = labels[F]
    mid = np.stack(
        [(P[:, 0] + P[:, 1]) / 2, (P[:, 1] + P[:, 2]) / 2, (P[:, 2] + P[:, 0]) / 2],
        axis=1,
    )
    # Edge e joins local vertices (e, e+1); it is cut when their labels differ.
    cut = np.stack(
        [Lm[:, 0] != Lm[:, 1], Lm[:, 1] != Lm[:, 2], Lm[:, 2] != Lm[:, 0]], axis=1
    )
    ncut = cut.sum(axis=1)

    two = np.flatnonzero(ncut == 2)
    for t in two:
        e0, e1 = np.flatnonzero(cut[t])
        seg = float(np.linalg.norm(mid[t, e0] - mid[t, e1]))
        # The odd vertex out is the one shared by both cut edges.
        shared = {e0, (e0 + 1) % 3} & {e1, (e1 + 1) % 3}
        odd = shared.pop()
        a = Lm[t, odd]
        b = Lm[t, (odd + 1) % 3] if Lm[t, (odd + 1) % 3] != a else Lm[t, (odd + 2) % 3]
        out[a] += seg
        out[b] += seg

    three = np.flatnonzero(ncut == 3)
    for t in three:
        c = mid[t].mean(axis=0)
        for e in range(3):
            seg = float(np.linalg.norm(mid[t, e] - c))
            out[Lm[t, e]] += seg
            out[Lm[t, (e + 1) % 3]] += seg
    return out


def compactness(
    labels: np.ndarray,
    mesh,
    n_cells: int,
) -> Dict[str, float]:
    """Per-cell isoperimetric ratio ``Q_k = P_k^2 / (4 pi A_k)``; Q = 1 is a disc.

    Over-merging makes a cell's territory wander away from its core, which raises
    Q. Reported as mean and worst. This is instrument 2 of the over-merge detector;
    assignment quality cannot see over-merging at all, because the balanced
    assignment hits its area target by construction.
    """
    per = per_cell_boundary_lengths(labels, mesh.vertices, mesh.faces, n_cells)
    areas = np.bincount(labels, weights=mesh.v, minlength=n_cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = per**2 / (4.0 * np.pi * areas)
    q = q[np.isfinite(q)]
    if q.size == 0:
        return {"q_mean": float("nan"), "q_worst": float("nan")}
    return {"q_mean": float(q.mean()), "q_worst": float(q.max())}


def core_loss(y: np.ndarray, labels: np.ndarray) -> int:
    """Cells whose own diffused-score peak vertex is not inside their territory.

    The most direct over-merge symptom: it says the cell's own signal was swamped
    by its neighbours' before the threshold was taken.
    """
    peaks = np.argmax(y, axis=0)
    return int(np.sum(labels[peaks] != np.arange(y.shape[1])))


# --------------------------------------------------------------------------
# the level loop
# --------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    churn: int
    churn_frac: float
    worst_rel_dev: float
    early_stop_capped: bool
    E_lumped: float
    E_consistent: float
    slack_rel: float
    violation_rel: float
    psi_absmax: float
    wall_seconds: float

    def to_dict(self) -> dict:
        return {
            k: (
                bool(v)
                if isinstance(v, bool)
                else int(v) if isinstance(v, (int, np.integer)) else float(v)
            )
            for k, v in self.__dict__.items()
        }


@dataclass
class LevelReport:
    level: int
    n_vertices: int
    n_cells: int
    tau: Dict[str, float]
    steps: List[StepRecord] = field(default_factory=list)
    n_anneals: int = 0
    converged: bool = False
    hit_max_iters: bool = False
    active_steps: int = 0
    final_worst_rel_dev: float = float("nan")
    final_early_stop_capped: bool = True
    probe: Dict = field(default_factory=dict)
    geometry: Dict = field(default_factory=dict)
    #: All three validity gates on THIS level's labels. Without this the ladder
    #: reports only final-label gates, and a per-level claim can only be inferred.
    #: That is exactly how "0 fragmented at every level" got into the record as an
    #: unmeasured assertion -- and it was false (1 transient fragment at N=300 L0).
    gates: Dict = field(default_factory=dict)
    label_boundary_length: float = float("nan")
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "level": int(self.level),
            "n_vertices": int(self.n_vertices),
            "n_cells": int(self.n_cells),
            "tau": {
                k: (bool(v) if isinstance(v, bool) else float(v))
                for k, v in self.tau.items()
            },
            "n_anneals": int(self.n_anneals),
            "converged": bool(self.converged),
            "hit_max_iters": bool(self.hit_max_iters),
            "n_steps": len(self.steps),
            "active_steps": int(self.active_steps),
            "final_worst_rel_dev": float(self.final_worst_rel_dev),
            "final_early_stop_capped": bool(self.final_early_stop_capped),
            "probe": self.probe,
            "geometry": self.geometry,
            "gates": self.gates,
            "label_boundary_length": float(self.label_boundary_length),
            "wall_seconds": float(self.wall_seconds),
            "steps": [s.to_dict() for s in self.steps],
        }


def _run_steps(
    mesh,
    labels: np.ndarray,
    n_cells: int,
    tau: float,
    cfg: MBOConfig,
) -> Tuple[np.ndarray, List[StepRecord], np.ndarray, bool, bool]:
    """Inner MBO loop at a fixed tau. Returns (labels, records, last_y, converged, hit_cap)."""
    v = mesh.v
    total_area = float(mesh.M.sum())
    target = total_area / n_cells
    bar = assignment_quality_bar(v, n_cells)
    solve = make_diffusion_solver(mesh, tau)
    n_vertices = len(labels)

    records: List[StepRecord] = []
    quiet = 0
    converged = False
    y = None
    for step in range(cfg.max_iters):
        t0 = time.perf_counter()
        y, inner_consistent = diffuse_indicators(solve, mesh, labels, n_cells)
        energy = lyapunov_energy(y, labels, v, total_area, tau, inner_consistent)
        psi, new_labels, worst = balanced_assign(
            y, v, target, bar, early_stop=cfg.early_stop_in_loop
        )
        slack = descent_slack(y, v, psi, labels, new_labels, n_cells)
        churn = int(np.count_nonzero(new_labels != labels))
        records.append(
            StepRecord(
                step=step,
                churn=churn,
                churn_frac=churn / n_vertices,
                worst_rel_dev=float(worst),
                early_stop_capped=cfg.early_stop_in_loop,
                E_lumped=energy["E_lumped"],
                E_consistent=energy["E_consistent"],
                slack_rel=slack["slack_rel"],
                violation_rel=slack["violation_rel"],
                psi_absmax=float(np.abs(psi).max()),
                wall_seconds=time.perf_counter() - t0,
            )
        )
        labels = new_labels
        quiet = quiet + 1 if churn / n_vertices < cfg.churn_tol else 0
        if quiet >= cfg.patience:
            converged = True
            break
    return labels, records, y, converged, not converged


def tau_continuation_probe(
    mesh,
    labels: np.ndarray,
    n_cells: int,
    tau: float,
    cfg: MBOConfig,
    tail_median_dE: float,
) -> Dict:
    """NC3 -- distinguish a level that CONVERGED from one that FROZE.

    E_tau is a function of the labels alone, so a frozen level and a converged one
    both show a flat energy; the only way to tell them apart is to perturb the
    diffusion length and see whether a better labelling was reachable all along.

    A bare sign test ("labels moved AND E dropped") fires on essentially
    everything: a wider blur re-tiebreaks boundary vertices at any labelling and
    some retiling always shaves E_tau. Measured on this project, a genuinely frozen
    c=2 state and a healthy still-churning c=4 state both move tens of vertices,
    with the HEALTHY one showing the larger drop. So the verdict needs a
    calibrated threshold, and ``nc3_K``/``nc3_f_min`` are pinned by NC3-CAL against
    a frozen/converged pair BEFORE any scored run. Until they are set this reports
    ``pinned = None`` rather than a verdict.

    A single doubling has a false-negative regime when the freeze deficit exceeds
    2x, hence the geometric ladder of probe factors.
    """
    v = mesh.v
    total_area = float(mesh.M.sum())
    target = total_area / n_cells
    bar = assignment_quality_bar(v, n_cells)
    n_vertices = len(labels)

    # Baseline energy at the LEVEL's own tau -- the probe changes tau only to
    # generate a candidate labelling; the comparison is always at the level's tau.
    base_solve = make_diffusion_solver(mesh, tau)
    y_base, inner = diffuse_indicators(base_solve, mesh, labels, n_cells)
    e_base = lyapunov_energy(y_base, labels, v, total_area, tau, inner)["E_lumped"]

    results = []
    for factor in cfg.probe_factors:
        solve = make_diffusion_solver(mesh, factor * tau)
        y_p, _ = diffuse_indicators(solve, mesh, labels, n_cells)
        _, probe_labels, _ = balanced_assign(
            y_p, v, target, bar, early_stop=cfg.early_stop_in_loop
        )
        moved = int(np.count_nonzero(probe_labels != labels))
        y_eval, inner_e = diffuse_indicators(base_solve, mesh, probe_labels, n_cells)
        e_probe = lyapunov_energy(y_eval, probe_labels, v, total_area, tau, inner_e)[
            "E_lumped"
        ]
        results.append(
            {
                "factor": float(factor),
                "moved": moved,
                "moved_frac": moved / n_vertices,
                "dE": float(e_base - e_probe),
                "dE_over_tail": (
                    float((e_base - e_probe) / tail_median_dE)
                    if tail_median_dE > 0
                    else float("inf")
                ),
            }
        )

    pinned: Optional[bool] = None
    if cfg.nc3_K is not None and cfg.nc3_f_min is not None:
        pinned = any(
            r["dE"] > cfg.nc3_K * tail_median_dE and r["moved_frac"] > cfg.nc3_f_min
            for r in results
        )
    return {
        "E_base": float(e_base),
        "tail_median_dE": float(tail_median_dE),
        "probes": results,
        "pinned": pinned,
        "calibrated": cfg.nc3_K is not None and cfg.nc3_f_min is not None,
    }


def mbo_level(
    mesh,
    labels: np.ndarray,
    n_cells: int,
    cfg: MBOConfig,
    level: int = 0,
    run_probe: bool = True,
) -> Tuple[np.ndarray, LevelReport]:
    """One mesh level: MBO to convergence, pinning probe, anneal if pinned."""
    t0 = time.perf_counter()
    v = mesh.v
    total_area = float(mesh.M.sum())
    target = total_area / n_cells
    bar = assignment_quality_bar(v, n_cells)

    diag = tau_diagnostics(mesh, n_cells, cfg)
    tau = diag["tau"]
    report = LevelReport(
        level=level, n_vertices=len(labels), n_cells=n_cells, tau=dict(diag)
    )

    anneals = 0
    while True:
        labels, records, y, converged, hit_cap = _run_steps(
            mesh, labels, n_cells, tau, cfg
        )
        report.steps.extend(records)
        report.converged = converged
        report.hit_max_iters = hit_cap

        tail = [
            abs(records[i].E_lumped - records[i - 1].E_lumped)
            for i in range(max(1, len(records) - cfg.patience - 2), len(records))
        ]
        tail_median = float(np.median(tail)) if tail else 0.0

        probe = (
            tau_continuation_probe(mesh, labels, n_cells, tau, cfg, tail_median)
            if run_probe
            else {}
        )
        report.probe = probe
        if probe.get("pinned") and anneals < cfg.max_anneals:
            anneals += 1
            tau *= cfg.anneal_factor
            logger.warning(
                "level %d PINNED -- annealing tau x%.2f (anneal %d of %d)",
                level,
                cfg.anneal_factor,
                anneals,
                cfg.max_anneals,
            )
            report.tau = dict(tau_diagnostics(mesh, n_cells, cfg, tau_override=tau))
            continue
        break
    report.n_anneals = anneals

    # Full-budget final assignment: the in-loop numbers are early-stop-capped and
    # must never be quoted as assignment quality (report 07, artefact 5).
    final_capped = cfg.early_stop_in_loop
    if cfg.finalize_full_budget:
        solve = make_diffusion_solver(mesh, tau)
        y, _ = diffuse_indicators(solve, mesh, labels, n_cells)
        _, labels, worst = balanced_assign(y, v, target, bar, early_stop=False)
        final_capped = False
    else:
        worst = records[-1].worst_rel_dev if records else float("nan")
    report.final_worst_rel_dev = float(worst)
    report.final_early_stop_capped = final_capped

    report.active_steps = sum(1 for s in report.steps if s.churn > 0)
    # Per-level gates, through the SAME harness path used for the final labels.
    from .arm_harness import labels_to_one_hot, run_gates

    _g = run_gates(labels_to_one_hot(labels, n_cells), mesh, n_cells)
    report.gates = {
        "n_imbalanced": int(_g.n_imbalanced),
        "worst_rel_dev": float(_g.worst_rel_dev),
        "quality_bar": float(_g.quality_bar),
        "n_fragmented": int(_g.n_fragmented),
        "fragmented": [int(c) for c in _g.fragmented],
        "worst_stray_rel": float(_g.worst_stray_rel),
    }
    if _g.n_fragmented:
        logger.warning(
            "level %d: %d FRAGMENTED cell(s) %s on this level's labels",
            level,
            _g.n_fragmented,
            _g.fragmented[:10],
        )
    report.geometry = compactness(labels, mesh, n_cells)
    report.geometry["core_loss"] = core_loss(y, labels) if y is not None else -1
    from .balanced_readout import label_boundary_length

    report.label_boundary_length = label_boundary_length(
        labels, mesh.vertices, mesh.faces
    )
    report.wall_seconds = time.perf_counter() - t0
    return labels, report


def interpolate_labels(
    old_vertices: np.ndarray, new_vertices: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Carry a labelling to a finer mesh by nearest embedded vertex.

    ``mesh.interpolation.nearest_neighbor_interpolate`` is an O(V_new x V_old)
    Python loop -- 8.8e9 distance evaluations at the finest N=100 level -- so this
    uses a cKDTree, exactly as ``create_seeded_initial_condition`` does. Labels are
    interpolated directly rather than densities, which is both cheaper and the
    honest representation of a hard partition.
    """
    from scipy.spatial import cKDTree

    _, idx = cKDTree(np.asarray(old_vertices)).query(np.asarray(new_vertices), k=1)
    return np.asarray(labels)[idx]


def run_mbo_ladder(
    meshes: Sequence,
    n_cells: int,
    cfg: MBOConfig,
    init_only: bool = False,
) -> Tuple[np.ndarray, List[LevelReport], Dict]:
    """The full arm: balanced geodesic init on the coarsest mesh, then the ladder.

    ``init_only=True`` runs the initialization on the FINEST mesh and stops -- the
    attribution control. B's init is approach C's step 0, so if MBO barely improves
    on it then the arm's number is the init's and the descent claim is unearned;
    nothing else in the gate suite would detect that.
    """
    t0 = time.perf_counter()
    if init_only:
        mesh = meshes[-1]
        labels, info = geodesic_balanced_init(mesh, n_cells, cfg.seed)
        info["init_only"] = True
        info["total_wall_seconds"] = time.perf_counter() - t0
        return labels, [], info

    labels, info = geodesic_balanced_init(meshes[0], n_cells, cfg.seed)
    info["init_only"] = False
    reports: List[LevelReport] = []
    for level, mesh in enumerate(meshes):
        if level > 0:
            labels = interpolate_labels(
                meshes[level - 1].vertices, mesh.vertices, labels
            )
        labels, report = mbo_level(mesh, labels, n_cells, cfg, level=level)
        reports.append(report)
        logger.info(
            "level %d: V=%d steps=%d active=%d worst=%.4f%% Q=%.3f wall=%.1fs",
            level,
            report.n_vertices,
            len(report.steps),
            report.active_steps,
            report.final_worst_rel_dev * 100,
            report.geometry.get("q_mean", float("nan")),
            report.wall_seconds,
        )
    info["total_wall_seconds"] = time.perf_counter() - t0
    return labels, reports, info
