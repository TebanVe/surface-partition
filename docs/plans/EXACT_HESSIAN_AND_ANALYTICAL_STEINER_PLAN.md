---
name: Exact-Hessian Validation & Analytical Steiner Derivatives
overview: >
  Consolidated plan for completing the IPOPT exact-Hessian path.  The
  sparse area Jacobian, the exact Lagrangian Hessian, and the vectorised
  ``np.add.at`` Hessian accumulation are already implemented and merged.
  Three phases remain.  Phase 1 adds a validation harness proving the
  analytical derivatives agree with finite differences and with the
  dense reference path.  Phase 2 replaces the finite-difference Steiner /
  triple-point first derivatives with closed-form analytical expressions.
  Phase 3 does the same for the Steiner second derivatives.  After
  Phase 3, every derivative the exact-Hessian path sees is analytical and
  the analytical-vs-FD validation tolerance tightens from ~1e-4 to
  ~1e-10.  All mathematical derivations for the quantities that already
  exist (regular perimeter / area value, gradient, Jacobian, Hessian;
  Steiner FD schemes; Lagrangian assembly; sparsity structure) are in
  docs/math/01-phase2-derivatives; this plan only adds the analytical
  Steiner derivations, which are new.
todos:
  # --- Phase 1: validation harness ---
  - id: test-sparse-jac
    content: Write testing/test_sparse_jacobian_equivalence.py — sparse Jacobian == dense Jacobian at (jac_row, jac_col)
    status: pending
  - id: test-hess-vs-fd
    content: Write testing/test_exact_hessian_vs_fd.py — analytical Lagrangian Hessian agrees with central-FD to ~1e-4
    status: pending
  - id: test-hess-matvec
    content: Optional large-scale matvec variant of the Hessian FD test (avoids O(n^2) memory)
    status: pending
  - id: compare-lbfgs-vs-exact
    content: Write testing/compare_hessian_modes.py — run IPOPT twice on same problem, record perimeter/iters/time
    status: pending
  - id: compare-hessian-profiling
    content: Extend compare_hessian_modes.py with per-component profiling (perimeter-H / area-H / Steiner-H / IPOPT linear solve)
    status: pending
  # --- Phase 2: analytical first derivatives ---
  - id: steiner-math-utils
    content: Implement _compute_tp_geometry() helper — n_i, r_i, M_i matrices per triple point
    status: pending
  - id: steiner-dS-dp
    content: Implement _compute_dS_dp() — analytical ∂S/∂p_k per triple point via implicit-function theorem
    status: pending
  - id: steiner-dS-dlambda
    content: Implement _compute_dS_dlambda() — chain rule ∂S/∂λ_i from ∂S/∂p_k and edge directions
    status: pending
  - id: steiner-perim-grad-analytical
    content: Implement compute_steiner_perimeter_gradient_analytical() and wire into vectorized_steiner
    status: pending
  - id: steiner-area-jac-analytical
    content: Implement compute_steiner_area_jacobian_analytical() and _sparse variant
    status: pending
  - id: degenerate-branch
    content: Add explicit degenerate-triangle detection and branch-safe derivatives (any angle ≥ 120°)
    status: pending
  - id: validate-steiner-first-order
    content: Add testing/test_steiner_gradient_analytical.py — analytical vs FD Steiner gradients agree to 1e-6
    status: pending
  # --- Phase 3: analytical second derivatives ---
  - id: steiner-d2S-dp2
    content: Implement _compute_d2S_dp2() — analytical ∂²S/∂p_k∂p_l via implicit differentiation
    status: pending
  - id: steiner-perim-hess-analytical
    content: Implement compute_steiner_perimeter_hessian_analytical() and replace FD version
    status: pending
  - id: steiner-area-hess-analytical
    content: Implement compute_steiner_area_hessian_analytical() (multiplier-weighted) and replace FD version
    status: pending
  - id: validate-full-hessian
    content: Tighten testing/test_exact_hessian_vs_fd.py to ~1e-8 / 1e-10 tolerance on pure-analytical path
    status: pending
  - id: optional-matvec-benchmark
    content: Optional — measure per-iteration cost of analytical vs FD Steiner on a reference problem
    status: pending
isProject: false
---

# Exact-Hessian Validation & Analytical Steiner Derivatives Plan

## Table of Contents

1. [Context](#1-context)
2. [Mathematical foundations — analytical Steiner derivatives](#2-math)
3. [Degenerate case](#3-degen)
4. [Phase 1 — Validation harness](#4-phase1)
5. [Phase 2 — Analytical first derivatives](#5-phase2)
6. [Phase 3 — Analytical second derivatives](#6-phase3)
7. [Optional fallback: hybrid (analytical 1st, FD 2nd)](#7-hybrid)
8. [Validation matrix](#8-validate)
9. [Acceptance criteria](#9-acceptance)
10. [Reference file paths](#10-paths)

---

## 1. Context <a id="1-context"></a>

### 1.1 What is already implemented

The IPOPT exact-Hessian path is functional and merged.  It is exercised
by `--method ipopt --exact-hessian` on `scripts/refine_perimeter.py`.
The following are done and are **not** in scope for this plan:

- **Sparse area Jacobian.**  `compute_area_jacobian_sparse()` and
  `compute_steiner_area_jacobian_sparse()` return values directly at the
  `(jac_row, jac_col)` sparsity positions; no dense matrix is allocated
  in the IPOPT callbacks.
- **Exact Lagrangian Hessian.**  `IPOPTProblemAdapter` attaches
  `hessian` / `hessianstructure` when `exact_hessian=True`.  The Hessian
  is assembled from the analytical regular-perimeter and regular-area
  Hessians plus finite-difference Steiner contributions.
- **Vectorised Hessian accumulation.**  `compute_perimeter_hessian_sparse()`
  and `compute_area_hessian_sparse()` accumulate into the flat
  `(hess_nnz,)` values array with `np.add.at` on pre-computed offset
  arrays (`seg_hess_off_{aa,bb,ab}`, `btri{1,2}_hess_off_{aa,bb,ab}`,
  `btri{1,2}_cell_active` on `PartitionArrays`).  The Python per-segment
  accumulation loop is gone.

The mathematics behind every quantity above — regular perimeter value /
gradient / Hessian, regular cell-area value / Jacobian / Hessian (both
the 1-inside and 2-inside cases), the Steiner finite-difference schemes,
the Lagrangian assembly, the lower-triangle Hessian sparsity convention —
is derived in `docs/math/01-phase2-derivatives` (compiled PDF).  Read
that document for the derivations; this plan does not repeat them.

### 1.2 What remains — three phases

| Phase | Deliverable | Why |
|---|---|---|
| **Phase 1** | Validation harness (`testing/`) | Proves the analytical derivatives agree with FD and with the dense reference.  Provides the regression reference for Phases 2–3. |
| **Phase 2** | Analytical Steiner **first** derivatives | Replaces the FD Steiner perimeter gradient and area Jacobian with closed-form expressions. |
| **Phase 3** | Analytical Steiner **second** derivatives | Replaces the FD Steiner perimeter and area Hessians. |

The phases are ordered.  Phase 1 must land first: Phases 2 and 3 change
the derivative code, and the only safe way to verify they are correct is
to regress them against the harness built in Phase 1 while it still
exercises the current FD-based Steiner code.  Phase 3 depends on Phase 2
because it reuses the geometry helpers Phase 2 introduces.

### 1.3 Scale target and applicability

The codebase's long-term ambition is partitions of **thousands of
cells**.  The work in this plan is necessary but not sufficient for that
ambition.  `docs/reference/SCALABILITY_ANALYSIS.md` is the master
roadmap; this plan delivers its Tier 2A (analytical Steiner) and the
Tier 2B validation harness.

| Problem size | What this plan delivers | Still blocking |
|---|---|---|
| N ≲ 20 | Correctness / robustness win; negligible speedup.  `tp_affected_vps` is tiny. | Nothing — the pipeline already works at this scale. |
| N ≈ 20–100 | Analytical Steiner removes FD-epsilon fragility; Steiner Hessian goes from ~30% of `h()` time to ~3%. | KKT Schur complement, iteration count. |
| N ≈ 100–500 | Steiner cost no longer shows up in profiles.  The FD Steiner Hessian here is an **O(N²)** per-iteration cost (outer FD over O(N) affected VPs × an inner gradient that is itself O(N)); the analytical formulas are **O(N)**. | Dense (N−1)×(N−1) Schur-complement block in IPOPT's KKT system; iteration-count growth. |
| N ≈ 500–1000+ | Required, but the monolithic IPOPT NLP has hit its ceiling. | **Tier 3** of `docs/reference/SCALABILITY_ANALYSIS.md` §3 — Lloyd CVT initialiser, multigrid, augmented-Lagrangian decomposition, or curve-shortening flow. |

Do not oversell this plan as "the thing that unlocks thousands of cells."
It is one necessary step in a longer chain.  The Tier 3 interventions are
out of scope here.

### 1.4 Scope guards

- **Phase 1** only adds files under `testing/`.  It does not modify any
  derivative code.  If a validation test fails, **stop and report** — a
  failure indicates a real bug, and this plan does not authorise editing
  the derivative formulas to make a test pass.
- **Phases 2–3** touch only `src/partition/vectorized_steiner.py`, plus
  optionally `src/partition/partition_arrays.py` and
  `src/partition/contour_partition.py` for cached per-triple-point slot
  indices (see §5.4).  Do **not** change the boundary-triangle area
  Hessian, the regular-perimeter Hessian, the Jacobian or Hessian
  sparsity patterns, the IPOPT adapter, or the `--exact-hessian` wiring.
  Those are already exact and (after Phase 1) tested.
- Phases 2–3 do **not** change `compute_steiner_points`.  The forward
  Fermat-point evaluation stays as-is; only the derivatives change.
- Phases 2–3 do **not** attempt to smooth the degenerate-case
  discontinuity at exactly 120° (see §3).  The derivatives are genuinely
  non-smooth there; this plan documents that and provides branch-safe
  code.

### 1.5 Where the mathematics lives

| Quantity | Reference |
|---|---|
| Regular perimeter / area value, gradient, Jacobian, Hessian | `docs/math/01-phase2-derivatives`, *Regular Perimeter* and *Regular Cell Area* sections |
| Steiner forward formula (Fermat–Torricelli point) | `docs/math/01-phase2-derivatives`, *Steiner / Triple-Point Contributions* |
| Steiner **finite-difference** derivative schemes (current code) | `docs/math/01-phase2-derivatives`, *Steiner derivatives: finite-difference schemes* |
| Lagrangian assembly, lower-triangle sparsity | `docs/math/01-phase2-derivatives`, *Lagrangian Hessian Assembly* |
| Steiner **analytical** derivatives (new — this plan) | §2 below |
| Empirical FD Steiner timing breakdown | `docs/math/02-phase2-timing-profile` |

---

## 2. Mathematical Foundations — Analytical Steiner Derivatives <a id="2-math"></a>

This section is self-contained.  A fresh agent with undergraduate
multivariable calculus should be able to derive everything below from
the two starting points given.  It is the only mathematics in this plan
not already in `docs/math/01-phase2-derivatives` — that document covers
the Steiner contributions only by finite differences.

### 2.1 Notation

At one triple point, three variable points `p_1, p_2, p_3 ∈ ℝ^dim`
(dim = 3 for surface meshes, dim = 2 for planar meshes).  Each
`p_i` depends affinely on a scalar `λ_i ∈ [0,1]`:

    p_i(λ_i) = λ_i V[v1_i] + (1 − λ_i) V[v2_i]
    d_i := dp_i/dλ_i = V[v1_i] − V[v2_i]   (constant edge direction)

All second derivatives `d²p_i/dλ_i²` are zero.  This is the same
structure as the boundary-triangle case.

Let `S ∈ ℝ^dim` be the Fermat–Torricelli (Steiner) point of the triangle
`(p_1, p_2, p_3)`.  Define

    u_i := p_i − S
    r_i := ‖u_i‖
    n_i := u_i / r_i               (unit vector from S to p_i)
    K_i := (I − n_i n_iᵀ) / r_i    (dim × dim PSD matrix)
    M   := K_1 + K_2 + K_3         (dim × dim SPD matrix; see §3 for degeneracy)

### 2.2 The Fermat point as the solution of an implicit equation

`S` is the unique minimiser of the strictly convex function
`G(S) = ‖p_1 − S‖ + ‖p_2 − S‖ + ‖p_3 − S‖`.  Its first-order optimality
condition is

    F(S, p_1, p_2, p_3) := −∇_S G = Σ_i (p_i − S)/r_i = Σ_i n_i = 0.   (★)

This is the "three unit vectors from S to the p_i sum to zero" statement
— equivalently, they make 120° angles between each other, which is the
geometric characterisation of the Fermat point when every triangle angle
is strictly less than 120°.

### 2.3 First derivative of S via the implicit function theorem

Differentiate (★) with respect to `p_k`:

    ∂F/∂S · ∂S/∂p_k + ∂F/∂p_k = 0.

The derivative of `n_i = u_i / r_i` with respect to a free vector `x`
when `u_i` depends linearly on `x` as `∂u_i/∂x = A` is the standard
"derivative of a unit vector" identity:

    ∂n_i/∂x = (I − n_i n_iᵀ)/r_i · A = K_i · A.

Applied separately for variations in S and in p_k:

    ∂F/∂S  = Σ_i (∂n_i/∂S)  = Σ_i K_i · (−I) = −M.
    ∂F/∂p_k = ∂n_k/∂p_k      = K_k · (I)      = K_k.

Hence

    −M · ∂S/∂p_k + K_k = 0   ⇒   **∂S/∂p_k = M^{-1} K_k**.    (∘)

Sanity check: `Σ_k ∂S/∂p_k = M^{-1} (K_1 + K_2 + K_3) = M^{-1} M = I`.
That encodes the translation invariance of the Fermat construction
(translating all three vertices by the same vector translates S by the
same vector).  Use this identity as a built-in unit test of your
implementation.

### 2.4 First derivative of Steiner perimeter and area

The Steiner perimeter contribution (per cell per triple point) is
`ℓ_a + ℓ_b − ℓ_{ab}` where

    ℓ_a = ‖p_a − S‖,   ℓ_b = ‖p_b − S‖,   ℓ_{ab} = ‖p_a − p_b‖.

For `p_a, p_b ∈ {p_1, p_2, p_3}` determined by the cell.  Since
`dp_a/dλ_a = d_a` and `∂S/∂λ_i = ∂S/∂p_k · d_k` where `k` is the VP that
carries λ_i (trivially `k = i` because each VP has its own λ):

    ∂ℓ_a/∂λ_i
        = n_aᵀ · (δ_{ai} d_i − ∂S/∂λ_i)
        = n_aᵀ · (δ_{ai} d_i − (M^{-1} K_i) d_i).

with `n_a = (p_a − S)/ℓ_a`.  The `ℓ_{ab}` term only depends on `p_a, p_b`
(not on S) and already has an analytical gradient — it is the same
formula the regular-perimeter code uses.

Analogously the Steiner area contribution per cell is

    A_cell = ½ ‖(p_b − p_a) × (S − p_a)‖ + corner_term

where the corner term doesn't involve S (it's a fixed mesh triangle).
The derivative w.r.t. λ_i is computed by chain rule through `∂p_a/∂λ_a`,
`∂p_b/∂λ_b`, and `∂S/∂λ_i = (M^{-1} K_i) d_i`.  The scalar-triple-product
derivative identity is the same one already used in
`compute_area_jacobian_analytical` — reuse the pattern.

### 2.5 Second derivative of S

Differentiate (∘) with respect to `p_l`:

    ∂²S/∂p_k ∂p_l = (∂/∂p_l) [M^{-1} K_k]
                  = (∂M^{-1}/∂p_l) K_k + M^{-1} (∂K_k/∂p_l)
                  = −M^{-1} (∂M/∂p_l) M^{-1} K_k + M^{-1} (∂K_k/∂p_l).

The remaining ingredients are `∂K_j/∂p_l` for each `(j, l)` pair.  Since
`K_j = (I − n_j n_jᵀ)/r_j` and both `n_j` and `r_j` depend on `p_l`
both explicitly (via `u_j = p_j − S`, if `l = j`) and implicitly (via
`S` always), we need to carry around `∂u_j/∂p_l = δ_{jl} I − ∂S/∂p_l`
already computed in Phase 2 and combine it.

Let `T_{jl} := ∂u_j/∂p_l = δ_{jl} I − ∂S/∂p_l`.  Then

    ∂r_j/∂p_l      = n_jᵀ · T_{jl}    (a row vector, dim = 1 × dim)
    ∂n_j/∂p_l      = K_j · T_{jl}     (a dim × dim matrix)
    ∂(n_j n_jᵀ)/∂p_l = (K_j T_{jl}) n_jᵀ + n_j (K_j T_{jl})ᵀ
    ∂K_j/∂p_l = − (K_j T_{jl}) n_jᵀ / r_j  − n_j (K_j T_{jl})ᵀ / r_j
                   − K_j · (∂r_j/∂p_l) / r_j
              (one dim × dim × dim rank-3 tensor per (j, l))

Then

    ∂M/∂p_l = Σ_j ∂K_j/∂p_l.

Plug these into the formula above for `∂²S/∂p_k ∂p_l`.  The result is a
rank-3 tensor of shape `(dim, dim, dim)` per `(k, l)` pair — three
"matrix columns" of second derivatives.

**Sanity check #2:** `Σ_{k,l} ∂²S/∂p_k ∂p_l = 0`.  This is the
second-order version of translation invariance (translating all three
vertices together doesn't bend the Fermat-point trajectory).  Use this as
a unit test once you have a working implementation.

### 2.6 Chain rule to λ-space

Given `∂²S/∂p_k ∂p_l` and the linearity `p_k = λ_k V[v1_k] + (1 − λ_k) V[v2_k]`,

    ∂²S/∂λ_i ∂λ_j = Σ_k Σ_l (∂²S/∂p_k ∂p_l) · d_k · d_l   (restricted to k = i-th VP, l = j-th VP)

which since each λ corresponds to exactly one VP simplifies to

    ∂²S/∂λ_i ∂λ_j = (∂²S/∂p_{k(i)} ∂p_{k(j)}) · d_i · d_j,

where `k(i)` denotes which of the three triangle vertices VP i is.  (In
practice `k(i) ∈ {0, 1, 2}` via `pa.tp_vp_indices[tp, :]`.)

For Steiner perimeter `ℓ_a = ‖p_a − S‖`, the second derivative uses the
same template as the regular-perimeter Hessian (see
`docs/math/01-phase2-derivatives`, the *Perimeter Hessian* subsection):

    ∂²ℓ_a/∂λ_i ∂λ_j = (Δᵀ (I − n_a n_aᵀ)/ℓ_a Δ)_{λ_i, λ_j}

where `Δ = p_a − S` and the first derivatives `∂p_a/∂λ_i` and `∂S/∂λ_i`
are the ones computed in Phase 2.  The product rule also generates a
term involving `∂²S/∂λ_i ∂λ_j` (since `d²p_a/dλ² = 0`) — that's where
the hard part of §2.5 enters.

Concretely:

    Δ_i   = (δ_{a,i} d_i − ∂S/∂λ_i)              ← first derivative of Δ w.r.t. λ_i
    Δ_{ij} = − ∂²S/∂λ_i ∂λ_j                       ← second derivative of Δ w.r.t. (λ_i, λ_j)

    ∂²ℓ_a/∂λ_i ∂λ_j = Δ_iᵀ (I − n_a n_aᵀ)/ℓ_a Δ_j + n_aᵀ Δ_{ij}.

The first term is the "projected dot product" familiar from the
regular-perimeter Hessian.  The second term is the contribution of the
bent trajectory of S, and it is the reason you need `∂²S/∂λ_i ∂λ_j` —
if you skip it, your Hessian is wrong, not merely approximate.

For the Steiner area contribution, apply the same chain-rule template
used in `compute_area_hessian_sparse` (see `docs/math/01-phase2-derivatives`,
the *Regular Cell Area* Hessian subsections), substituting
`p_a → p_a`, `pc1 → p_b`, `p_in → S`, and carefully tracking that `S`
depends on all three λ's, not on just one.

---

## 3. Degenerate Case <a id="3-degen"></a>

`compute_steiner_points` already detects "any triangle angle ≥ 120°" and
replaces `S` by the obtuse vertex.  Mathematically this is correct:
`S → p_{obtuse}` continuously as the angle approaches 120°, and the
minimum of `G(S) = Σ ‖p_i − S‖` genuinely collapses to the obtuse
vertex once the angle exceeds 120°.

### 3.1 Derivative behaviour

At exactly 120°, the Fermat point is the obtuse vertex, and the
derivatives of `S` are **non-smooth**:

- In the "non-degenerate" regime (all angles strictly < 120°), use the
  implicit-function-theorem formula from §2.3.
- In the "degenerate" regime (some angle ≥ 120°), `S = p_{obtuse}`, so
  `∂S/∂p_k = δ_{k,obtuse} I` and `∂²S/∂p_k ∂p_l = 0`.

These two branches **do not agree** at the boundary.  The boundary is a
measure-zero set in parameter space, so IPOPT should essentially never
sit on it, but implementation must pick one branch per evaluation.

### 3.2 Detection (reusing existing code)

`compute_steiner_points` already computes `all_angles` and
`degen_mask = all_angles.max(axis=1) >= 2π/3`.  Factor that detection
out into a shared helper so the derivative code can use it:

```python
# In vectorized_steiner.py
def _compute_tp_geometry(pa):
    """Per triple point, return (p, S, r, n, K, M, angles, degen_mask, obtuse_idx).

    Reuses the angle calculation from compute_steiner_points but also
    exposes the intermediate quantities needed by downstream derivative
    functions.  Kept private to this module.
    """
    ...
```

Return a dataclass (or just a tuple) containing:

- `p`: `(n_tp, 3, dim)` — the three VP positions.
- `S`: `(n_tp, dim)` — Fermat point (obtuse-vertex on degenerate rows).
- `r`: `(n_tp, 3)` — the three distances `‖p_i − S‖`.
- `n`: `(n_tp, 3, dim)` — unit vectors from `S` to `p_i`.
- `K`: `(n_tp, 3, dim, dim)` — the three K_i matrices.
- `M`: `(n_tp, dim, dim)` — sum of the K_i's.
- `degen_mask`: `(n_tp,)` bool — True where any angle ≥ 120°.
- `obtuse_idx`: `(n_tp,)` int32 — index in {0,1,2} of the obtuse vertex
  on degenerate rows, -1 elsewhere.

This helper is evaluated once per call to any derivative function.
Caching it across calls within the same IPOPT iteration is a possible
future optimisation (keyed off `pa.vp_lambda.ctypes.data + generation`),
but out of scope here.

### 3.3 Degenerate-row handling in each derivative function

In every analytical derivative function below (§5 and §6), the
implementation pattern is:

```python
non_degen = ~geom.degen_mask
# Compute the analytical formula only on non-degenerate rows
...[non_degen] = analytical_formula(...)

# Degenerate rows: handled separately
for row_idx in np.where(geom.degen_mask)[0]:
    obtuse = geom.obtuse_idx[row_idx]
    # ∂S/∂p_k = I if k == obtuse, else 0
    # ∂²S/... = 0
    ...
```

Avoid computing `M^{-1}` on degenerate rows — `M` is singular there
because `r_{obtuse} = 0` ⇒ `K_{obtuse}` blows up.  Mask before inversion.

### 3.4 Robustness near 120°

Just below 120°, `r_obtuse` is small and `K_obtuse` has a large operator
norm, which makes `M^{-1}` ill-conditioned.  Add a numerical guard:

```python
EPS_DEGEN = 1e-10   # below this, treat as degenerate
r_min = geom.r.min(axis=1)
near_degen = (r_min < EPS_DEGEN) & ~geom.degen_mask
if np.any(near_degen):
    # Promote near-degenerate rows to degenerate: treat the closest-to-S
    # vertex as obtuse.
    ...
```

The threshold `EPS_DEGEN` should scale with the mesh edge length if
possible (e.g. `0.01 * median_edge_length`).  For a uniform mesh `1e-10`
is usually fine; adjust if validation on a coarse mesh shows
ill-conditioned `M^{-1}` dominating the Hessian norm.

---

## 4. Phase 1 — Validation Harness <a id="4-phase1"></a>

Add CLI scripts under `testing/`.  Do **not** add pytest fixtures — this
project has no pytest suite (see `CLAUDE.md`, "Gotchas and Known
Issues").  Each script should be runnable as
`python testing/<name>.py --solution <path.h5> [--config <yaml>]`
and should print a clear PASS / FAIL line plus numeric diagnostics, and
exit with code 0 on PASS / 1 on FAIL.

All scripts need a compiled `PartitionArrays`.  The shared helper below
should live at the top of each script (or be factored into a single
`testing/_hessian_test_utils.py`):

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.pipeline.io import (
    load_partition_from_base_file, load_partition_from_refined_file,
    detect_run_layout,
)
from src.pipeline.pipeline_orchestrator import detect_file_type
from src.partition.steiner_handler import SteinerHandler
from src.optimization.perimeter_optimizer import PerimeterOptimizer


def build_optimizer(solution_path: str):
    """Load a solution/checkpoint file and return a compiled PerimeterOptimizer."""
    file_type = detect_file_type(solution_path)
    if file_type == 'base':
        mesh, partition = load_partition_from_base_file(solution_path, verbose=False)
    else:
        mesh, partition = load_partition_from_refined_file(solution_path, verbose=False)

    total_area = mesh.v.sum()
    target_area = float(total_area) / partition.n_cells

    steiner = SteinerHandler(mesh, partition)
    optimizer = PerimeterOptimizer(
        partition, mesh, target_area, steiner_handler=steiner,
        use_vectorized=True,
    )
    optimizer.compile()       # builds optimizer._arrays
    return optimizer
```

### 4.1 `testing/test_sparse_jacobian_equivalence.py`

**Goal.** Prove `compute_area_jacobian_sparse + compute_steiner_area_jacobian_sparse`
returns exactly the same values at positions `(jac_row, jac_col)` as the
dense reference path.

```python
#!/usr/bin/env python3
"""Verify that the sparse area Jacobian equals the dense reference."""
import argparse
import numpy as np

from _hessian_test_utils import build_optimizer   # or inline the helper
from src.partition import vectorized_area, vectorized_steiner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--atol', type=float, default=1e-10)
    args = ap.parse_args()

    opt = build_optimizer(args.solution)
    pa = opt._arrays

    # Dense reference (analytical area + FD Steiner)
    jac_dense_area   = vectorized_area.compute_area_jacobian(pa)
    jac_dense_stein  = vectorized_steiner.compute_steiner_area_jacobian(pa)
    jac_dense = jac_dense_area + jac_dense_stein   # (n_c-1, n_active)

    # Sparse path
    jac_sparse_area  = vectorized_area.compute_area_jacobian_sparse(pa)
    jac_sparse_stein = vectorized_steiner.compute_steiner_area_jacobian_sparse(pa)
    jac_sparse = jac_sparse_area + jac_sparse_stein    # (nnz,)

    # Extract non-zeros from dense in jac_row/jac_col order
    dense_vals = jac_dense[pa.jac_row, pa.jac_col]

    abs_err = np.max(np.abs(dense_vals - jac_sparse))
    rel_err = abs_err / max(np.max(np.abs(dense_vals)), 1e-30)

    print(f"nnz              = {len(pa.jac_row)}")
    print(f"dense shape      = {jac_dense.shape}")
    print(f"max |Δ|          = {abs_err:.3e}")
    print(f"max rel err      = {rel_err:.3e}")

    ok = abs_err < args.atol
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
```

**Pass criterion:** `abs_err < 1e-10`.  The Steiner contributions on both
sides use the same finite-difference code, so the only difference between
the paths is the scatter pattern, which should be exact.

### 4.2 `testing/test_exact_hessian_vs_fd.py`

**Goal.** Prove the full Lagrangian Hessian returned by
`IPOPTProblemAdapter._hessian_impl` agrees with a finite-difference
reference computed by central differences on the Lagrangian gradient.

**Why this matters.** This is the single test that could unmask a wrong
sign or a wrong chain-rule factor in any of the analytical Hessian
derivations.  If it passes, the analytical and FD Steiner pieces combine
into something consistent with the gradient, which is itself analytical
and already tested by IPOPT's derivative checker.

```python
#!/usr/bin/env python3
"""Verify the analytical Lagrangian Hessian against central FD."""
import argparse
import numpy as np

from _hessian_test_utils import build_optimizer
from src.optimization.perimeter_optimizer import IPOPTProblemAdapter


def lagrangian_grad(optimizer, x, lagrange, obj_factor):
    g  = obj_factor * optimizer.objective_gradient(x)
    J  = optimizer.constraint_area_jacobian(x)     # (n_c-1, n_active)
    return g + lagrange @ J


def hessian_from_adapter(optimizer, x, lagrange, obj_factor):
    """Call the adapter's Hessian and pack into a dense symmetric matrix."""
    adapter = IPOPTProblemAdapter(optimizer, exact_hessian=True)
    vals = adapter._hessian_impl(x, lagrange, obj_factor)   # (hess_nnz,)
    n    = len(x)
    H    = np.zeros((n, n))
    pa   = optimizer._arrays
    H[pa.hess_row, pa.hess_col] = vals
    # Mirror the lower triangle to upper
    strict_lower = pa.hess_row > pa.hess_col
    H[pa.hess_col[strict_lower], pa.hess_row[strict_lower]] = vals[strict_lower]
    return H


def hessian_fd(optimizer, x, lagrange, obj_factor, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        H[i, :] = (lagrangian_grad(optimizer, xp, lagrange, obj_factor)
                   - lagrangian_grad(optimizer, xm, lagrange, obj_factor)) / (2 * eps)
    return 0.5 * (H + H.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--rtol', type=float, default=1e-3,
                    help='Relative tolerance (loose — Steiner FD contributes noise).')
    ap.add_argument('--atol', type=float, default=1e-4,
                    help='Absolute floor for the comparison.')
    ap.add_argument('--obj-factor', type=float, default=1.0)
    ap.add_argument('--lagrange-mode', choices=['zero', 'ones', 'random'],
                    default='random')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-n', type=int, default=400,
                    help='Skip test if n_active_vp > this (use matvec mode instead).')
    args = ap.parse_args()

    opt = build_optimizer(args.solution)
    pa  = opt._arrays
    n   = pa.n_active_vp
    m   = pa.n_cells - 1

    if n > args.max_n:
        print(f"SKIP: n_active_vp={n} > --max-n={args.max_n}. "
              f"Run testing/test_exact_hessian_matvec.py instead.")
        return 0

    rng = np.random.default_rng(args.seed)
    x   = pa.vp_lambda.copy()
    if args.lagrange_mode == 'zero':
        lam = np.zeros(m)
    elif args.lagrange_mode == 'ones':
        lam = np.ones(m)
    else:
        lam = rng.standard_normal(m)

    H_ana = hessian_from_adapter(opt, x, lam, args.obj_factor)
    H_fd  = hessian_fd(opt, x, lam, args.obj_factor)

    abs_err = np.max(np.abs(H_ana - H_fd))
    ref     = max(np.max(np.abs(H_fd)), 1e-30)
    rel_err = abs_err / ref

    print(f"n_active_vp  = {n}")
    print(f"hess_nnz     = {len(pa.hess_row)}")
    print(f"||H_ana||_∞  = {np.max(np.abs(H_ana)):.3e}")
    print(f"||H_fd||_∞   = {np.max(np.abs(H_fd)):.3e}")
    print(f"max |Δ|      = {abs_err:.3e}")
    print(f"rel err      = {rel_err:.3e}")

    i, j = np.unravel_index(np.argmax(np.abs(H_ana - H_fd)), H_ana.shape)
    print(f"worst entry  = ({i},{j}) ana={H_ana[i,j]:.3e} fd={H_fd[i,j]:.3e}")

    ok = abs_err < max(args.atol, args.rtol * ref)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
```

**Tolerance guidance.** Because the Steiner perimeter and area Hessians
currently use finite differences internally (central-FD on the gradient
for perimeter, forward-FD on the Jacobian for area), expect ~1e-4
absolute agreement, not machine precision — two nested finite
differences cannot do better.  Suggested tolerances:

- `--atol 1e-4 --rtol 1e-3` for meshes with triple points.
- `--atol 1e-6 --rtol 1e-5` when `--lagrange-mode zero` and the problem
  has no triple points (pure analytical Hessian, no FD noise).

After Phase 3 lands (analytical Steiner Hessian), these tolerances
tighten dramatically — see the validation matrix in §8.

**Diagnostic suggestion.** If the test fails, re-run with
`--lagrange-mode zero` to isolate the objective Hessian (perimeter only),
then with `--obj-factor 0 --lagrange-mode ones` to isolate the
constraint Hessian (area only).  Whichever branch fails identifies the
module with the bug (perimeter, area, or Steiner FD).  If this test
fails, **stop** and report — Phase 1 does not authorise editing the
derivative code.

### 4.3 `testing/test_exact_hessian_matvec.py` (optional, for large meshes)

For problems where `n_active_vp > ~400` the dense `(n, n)` FD build is
slow and memory-hungry.  Provide a Hessian-vector-product check instead:
for a random unit vector `v`, verify

    H_ana @ v  ≈  (L_grad(x + ε v) − L_grad(x − ε v)) / (2 ε)

Loop over ~5 random `v`'s with a `seed` CLI option.  Accept if every
matvec agrees to within the same tolerance as §4.2.

```python
def hessian_matvec(optimizer, x, lagrange, obj_factor, v, eps=1e-5):
    xp = x + eps * v
    xm = x - eps * v
    return (lagrangian_grad(optimizer, xp, lagrange, obj_factor)
            - lagrangian_grad(optimizer, xm, lagrange, obj_factor)) / (2 * eps)


def hessian_matvec_from_adapter(optimizer, x, lagrange, obj_factor, v):
    H = hessian_from_adapter(optimizer, x, lagrange, obj_factor)   # dense
    return H @ v
```

For truly large problems (`n > 10000`) build the matvec directly from
the sparse values instead:

```python
def hessian_matvec_sparse(pa, vals, v):
    out = np.zeros_like(v)
    # Lower triangle contribution
    np.add.at(out, pa.hess_row, vals * v[pa.hess_col])
    # Upper triangle contribution (from symmetry, excluding the diagonal)
    strict = pa.hess_row > pa.hess_col
    np.add.at(out, pa.hess_col[strict], vals[strict] * v[pa.hess_row[strict]])
    return out
```

This is a useful utility — consider adding it to
`src/partition/vectorized_perimeter.py` so other code can reuse it.  But
keep the scope of Phase 1 to the test only.

### 4.4 `testing/compare_hessian_modes.py`

**Goal.** Quantitatively decide whether `--exact-hessian` is worth using
by default, and — just as importantly — answer **where the per-iteration
time actually goes** when the exact-Hessian path is slower.  The script
produces two kinds of output on the same run:

1. **End-to-end totals** per mode: final perimeter, constraint violation,
   iteration count, wall-clock time, success flag.
2. **Per-component breakdown of one Hessian evaluation** on the
   exact-Hessian path: time spent in `compute_perimeter_hessian_sparse`,
   `compute_area_hessian_sparse`, the two Steiner FD helpers, and (via
   IPOPT's built-in timing statistics) the linear solver.

The second output directs all subsequent scaling work: if Steiner
dominates, Phases 2–3 are the next dollar; if the IPOPT linear solver
dominates, the next dollar is in Tier 3 of `docs/reference/SCALABILITY_ANALYSIS.md` or
in a linear-solver swap (MA57/MA97 over MUMPS).

**Implementation sketch.** Call `PerimeterOptimizer.optimize()` directly
from one Python process so both runs see the same initial state.  The
per-component profiler is a small monkey-patch that intercepts the four
Hessian-building functions and accumulates their wall-clock into a
counter dict.

```python
#!/usr/bin/env python3
"""Compare IPOPT L-BFGS vs. exact Hessian on the same problem.

Also emits a per-component profile of one Hessian evaluation on the
exact-Hessian path, so the user can see at a glance which piece is the
bottleneck (Python accumulation, Steiner FD, or IPOPT's linear solver).
"""
import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

from _hessian_test_utils import build_optimizer

from src.partition import vectorized_perimeter as _vperim
from src.partition import vectorized_area      as _varea
from src.partition import vectorized_steiner   as _vstein


# ---------------------------------------------------------------------
# Per-component profiler
# ---------------------------------------------------------------------

_PROFILE = defaultdict(lambda: {'calls': 0, 'time': 0.0})


def _timed(name, fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _PROFILE[name]['calls'] += 1
        _PROFILE[name]['time']  += time.perf_counter() - t0
        return out
    wrapper.__wrapped__ = fn
    return wrapper


@contextmanager
def profile_hessian_components():
    """Monkey-patch the four Hessian kernels to record cumulative time.

    Only active inside the `with` block.  Use around a single IPOPT
    run — not around two, or the accumulators would double-count.
    """
    originals = {
        ('_vperim', 'compute_perimeter_hessian_sparse'):
            _vperim.compute_perimeter_hessian_sparse,
        ('_varea',  'compute_area_hessian_sparse'):
            _varea.compute_area_hessian_sparse,
        ('_vstein', 'compute_steiner_perimeter_hessian_fd'):
            _vstein.compute_steiner_perimeter_hessian_fd,
        ('_vstein', 'compute_steiner_area_hessian_fd'):
            _vstein.compute_steiner_area_hessian_fd,
    }
    _PROFILE.clear()
    try:
        _vperim.compute_perimeter_hessian_sparse     = _timed(
            'perimeter_hess', _vperim.compute_perimeter_hessian_sparse)
        _varea.compute_area_hessian_sparse           = _timed(
            'area_hess',      _varea.compute_area_hessian_sparse)
        _vstein.compute_steiner_perimeter_hessian_fd = _timed(
            'steiner_perim_hess_fd',
            _vstein.compute_steiner_perimeter_hessian_fd)
        _vstein.compute_steiner_area_hessian_fd      = _timed(
            'steiner_area_hess_fd',
            _vstein.compute_steiner_area_hessian_fd)
        yield _PROFILE
    finally:
        for (mod_name, attr), orig in originals.items():
            mod = {'_vperim': _vperim, '_varea': _varea,
                   '_vstein': _vstein}[mod_name]
            setattr(mod, attr, orig)


# ---------------------------------------------------------------------
# IPOPT built-in timing statistics
# ---------------------------------------------------------------------
#
# `print_timing_statistics yes` makes IPOPT write a block like:
#
#   OverallAlgorithm....................:     12.345 (sys:  0.021 wall:  12.380)
#   PDSystemSolver.....................:      7.234 (sys:  0.012 wall:   7.250)
#   ...
#
# to stdout after the run.  Redirect stdout to a buffer, run IPOPT,
# then parse the "PDSystemSolver" line for its wall time.  This is the
# only reliable way to separate "Python callback time" from "IPOPT
# linear-solve time".

import io, contextlib, re

_PD_RE = re.compile(
    r'PDSystemSolver\.*:\s*\S+\s*\(sys:\s*\S+\s*wall:\s*(\S+)\)')
_OA_RE = re.compile(
    r'OverallAlgorithm\.*:\s*\S+\s*\(sys:\s*\S+\s*wall:\s*(\S+)\)')


def _extract_ipopt_timing(log_text):
    pd = _PD_RE.search(log_text)
    oa = _OA_RE.search(log_text)
    return {
        'pd_solver_time':      float(pd.group(1)) if pd else None,
        'overall_ipopt_time':  float(oa.group(1)) if oa else None,
    }


# ---------------------------------------------------------------------
# One-mode runner
# ---------------------------------------------------------------------

def run_once(solution_path, exact_hessian, max_iter, tol, do_profile):
    opt = build_optimizer(solution_path)

    # IPOPT options to get the PDSystemSolver wall time in stdout.
    opt._ipopt_extra_options = {'print_timing_statistics': 'yes'}

    stdout_capture = io.StringIO()
    profile = None
    t0 = time.perf_counter()
    if do_profile:
        with profile_hessian_components() as p, \
             contextlib.redirect_stdout(stdout_capture):
            result = opt.optimize(
                max_iter=max_iter, tol=tol, method='ipopt',
                exact_hessian=exact_hessian,
            )
        profile = {k: dict(v) for k, v in p.items()}
    else:
        with contextlib.redirect_stdout(stdout_capture):
            result = opt.optimize(
                max_iter=max_iter, tol=tol, method='ipopt',
                exact_hessian=exact_hessian,
            )
    elapsed = time.perf_counter() - t0
    ipopt_timing = _extract_ipopt_timing(stdout_capture.getvalue())
    final_viol = float(np.max(np.abs(opt.constraint_area_equality(result.x))))

    return {
        'final_perimeter': float(result.fun),
        'final_viol':      final_viol,
        'iters':           int(getattr(result, 'nit', -1)),
        'status':          int(getattr(result, 'status', -1)),
        'time':            elapsed,
        'success':         bool(result.success),
        'ipopt_timing':    ipopt_timing,
        'profile':         profile,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def _fmt_row(label, seconds, calls, total_wall):
    pct = 100.0 * seconds / total_wall if total_wall > 0 else 0.0
    per_call = 1000.0 * seconds / calls if calls > 0 else 0.0
    return f"  {label:32s} {seconds:8.3f} s  "\
           f"{pct:5.1f}%  {calls:6d} calls  "\
           f"{per_call:7.2f} ms/call"


def print_component_breakdown(result_exact):
    total = result_exact['time']
    prof = result_exact['profile']
    ipopt_t = result_exact['ipopt_timing']

    print(f"\n  --- Per-component breakdown (total run = {total:.3f} s) ---")
    for name in ('perimeter_hess', 'area_hess',
                 'steiner_perim_hess_fd', 'steiner_area_hess_fd'):
        info = prof.get(name, {'calls': 0, 'time': 0.0})
        print(_fmt_row(name, info['time'], info['calls'], total))

    py_hess = sum(prof[k]['time'] for k in prof
                  if k in ('perimeter_hess', 'area_hess',
                           'steiner_perim_hess_fd',
                           'steiner_area_hess_fd'))
    print(_fmt_row("  Σ Python Hessian kernels", py_hess, 0, total))

    if ipopt_t['pd_solver_time'] is not None:
        pd = ipopt_t['pd_solver_time']
        pct = 100.0 * pd / total
        print(f"  {'IPOPT PDSystemSolver (lin. alg.)':32s} "
              f"{pd:8.3f} s  {pct:5.1f}%   (from print_timing_statistics)")

        other = max(0.0, total - py_hess - pd)
        pct   = 100.0 * other / total
        print(f"  {'other (f, g, c, jac, overhead)':32s} "
              f"{other:8.3f} s  {pct:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--max-iter', type=int, default=200)
    ap.add_argument('--tol',      type=float, default=1e-7)
    ap.add_argument('--no-profile', action='store_true',
                    help='Skip the per-component profile (faster).')
    args = ap.parse_args()

    print("=== L-BFGS ===")
    r_lbfgs = run_once(args.solution, exact_hessian=False,
                       max_iter=args.max_iter, tol=args.tol,
                       do_profile=False)
    for k in ('final_perimeter', 'final_viol', 'iters', 'time', 'success'):
        print(f"  {k:17s} {r_lbfgs[k]}")
    if r_lbfgs['ipopt_timing']['pd_solver_time'] is not None:
        print(f"  IPOPT PDSystemSolver "
              f"{r_lbfgs['ipopt_timing']['pd_solver_time']:.3f} s")

    print("\n=== Exact Hessian ===")
    r_exact = run_once(args.solution, exact_hessian=True,
                       max_iter=args.max_iter, tol=args.tol,
                       do_profile=not args.no_profile)
    for k in ('final_perimeter', 'final_viol', 'iters', 'time', 'success'):
        print(f"  {k:17s} {r_exact[k]}")
    if r_exact['ipopt_timing']['pd_solver_time'] is not None:
        print(f"  IPOPT PDSystemSolver "
              f"{r_exact['ipopt_timing']['pd_solver_time']:.3f} s")

    if r_exact['profile'] is not None:
        print_component_breakdown(r_exact)

    print("\n=== Summary ===")
    dp = r_exact['final_perimeter'] - r_lbfgs['final_perimeter']
    dt = r_exact['time']            - r_lbfgs['time']
    di = r_exact['iters']           - r_lbfgs['iters']
    print(f"  Δ perimeter (exact − lbfgs) = {dp:+.6e}")
    print(f"  Δ iters     (exact − lbfgs) = {di:+d}")
    print(f"  Δ wall time (exact − lbfgs) = {dt:+.2f} s")
    print(f"  per-iter  exact  = {r_exact['time']/max(1, r_exact['iters']):.3f} s")
    print(f"  per-iter  lbfgs  = {r_lbfgs['time']/max(1, r_lbfgs['iters']):.3f} s")


if __name__ == '__main__':
    raise SystemExit(main())
```

**Plumbing note (monkey-patch robustness).** The profiler patches
module-level symbols in `vectorized_perimeter`, `vectorized_area`, and
`vectorized_steiner`.  That works **only** if the call sites inside
`PerimeterOptimizer` reach those functions via attribute access on the
module rather than via `from ... import compute_perimeter_hessian_sparse`
into the optimizer's own namespace.  Before writing this script, grep
`src/optimization/perimeter_optimizer.py` for the four function names
and confirm.  If they are bound directly, either switch to
module-qualified access in the optimizer (one-line change) or patch the
names in `perimeter_optimizer` too.  Document which you picked at the top
of the script.

**Plumbing note (IPOPT options hook).** The sketch assumes
`PerimeterOptimizer` exposes a way to pass extra IPOPT options through to
`cyipopt.Problem.add_option`.  If it does not, add a kwarg on
`optimize()`: `extra_ipopt_options: dict = None`, and have the
`IPOPTProblemAdapter` call `problem.add_option(k, v)` for each entry
(~5 lines).  This is generally useful (also for `linear_solver`,
`mu_strategy`, `hessian_approximation` overrides).

**Reading the breakdown.** The vectorised `np.add.at` accumulation is
already done, so the `perimeter_hess` / `area_hess` rows reflect genuine
arithmetic, not Python dict-lookup overhead.  Use the breakdown to
decide what to do next:

- **Steiner FD rows dominate** → Phases 2–3 (analytical Steiner) are the
  next win, both for wall-clock and for the validation tolerance.
- **`perimeter_hess` / `area_hess` dominate** → unexpected; profile with
  `cProfile` (see below) — the arithmetic should be cheap.
- **IPOPT `PDSystemSolver` dominates** → this is a linear-algebra
  problem, not a Python problem.  Try swapping MUMPS for MA57, or move to
  Tier 3 of `docs/reference/SCALABILITY_ANALYSIS.md`.

**What "success" looks like.** The exact Hessian should give equal or
lower final perimeter, fewer IPOPT iterations, and fewer restoration
phases (visible in the IPOPT log during the run).  Report both the
end-to-end comparison and the component breakdown; do **not** change any
defaults based on a single problem.  If exact-Hessian is convincingly
better, raise that as a separate follow-up — it is outside the scope of
this plan.

**Optional — cProfile dump.** If the four wrapped functions do not
account for most of the Python time (the "other" bucket is >40%), add
`--cprofile path.prof` to the CLI, wrap the exact-Hessian run in
`cProfile.Profile()`, and dump with
`pstats.Stats(prof).strip_dirs().sort_stats('cumulative').print_stats(25)`.
Keep it behind a flag — it adds ~30% overhead.

### 4.5 Hook tests into `testing/README_testing.md`

Add a short section listing the new CLIs and their purpose, in the same
style as the existing entries.  One paragraph each.  Include a
one-paragraph reading guide for the `compare_hessian_modes.py`
per-component breakdown (what "Steiner FD dominates" vs "linear solver
dominates" implies for further optimisation).

---

## 5. Phase 2 — Analytical First Derivatives <a id="5-phase2"></a>

Phase 2 replaces the finite-difference Steiner first derivatives with the
closed-form expressions of §2.  All new code lives in
`src/partition/vectorized_steiner.py`.

### 5.1 Shared geometry helper

Implement `_compute_tp_geometry(pa)` as described in §3.2.  Place it
near the top of `src/partition/vectorized_steiner.py`, just below
`compute_steiner_points()`.  Keep it private (underscore prefix).

### 5.2 `_compute_dS_dp(geom)` — ∂S/∂p_k per triple point

```python
def _compute_dS_dp(geom):
    """Return (n_tp, 3, dim, dim) array: dS/dp[tp, k, :, :].

    Non-degenerate rows: ∂S/∂p_k = M^{-1} K_k.
    Degenerate rows: ∂S/∂p_k = I if k == obtuse_idx else 0.
    """
    n_tp, _, dim, _ = geom.K.shape
    dS_dp = np.zeros((n_tp, 3, dim, dim), dtype=np.float64)

    non_degen = ~geom.degen_mask
    if np.any(non_degen):
        M_inv = np.linalg.inv(geom.M[non_degen])       # (n_nd, dim, dim)
        K_sub = geom.K[non_degen]                      # (n_nd, 3, dim, dim)
        # Batched matmul: dS_dp[row, k] = M_inv[row] @ K_sub[row, k]
        dS_dp[non_degen] = np.einsum('tij,tkjl->tkil', M_inv, K_sub)

    if np.any(geom.degen_mask):
        eye = np.eye(dim)
        for row in np.where(geom.degen_mask)[0]:
            k = geom.obtuse_idx[row]
            dS_dp[row, k] = eye

    return dS_dp
```

Add a unit test inside the module (or a debug-build assertion) that on
non-degenerate rows `np.allclose(dS_dp.sum(axis=1), np.eye(dim), atol=1e-8)`
— the translation-invariance identity from §2.3.  Cheap, catches formula
errors.

### 5.3 Lift to λ-space: `_compute_dS_dlambda(pa, geom, dS_dp)`

Each triple point has 3 VP indices `pa.tp_vp_indices[tp, :]`, and each
VP `i` has its own edge direction `d_i = V[v1_i] − V[v2_i]`.

```python
def _compute_dS_dlambda(pa, geom, dS_dp):
    """Return (n_tp, 3, dim) array: dS/dλ[tp, slot] = dS/dp[tp, slot] · d_slot.

    slot ∈ {0, 1, 2} corresponds to the three VPs of the triple point.
    """
    tp_vps = pa.tp_vp_indices                         # (n_tp, 3)
    d      = pa.vertices[pa.vp_edge_v1[tp_vps]] - pa.vertices[pa.vp_edge_v2[tp_vps]]
    # d has shape (n_tp, 3, dim)
    return np.einsum('tkij,tkj->tki', dS_dp, d)
```

### 5.4 `compute_steiner_perimeter_gradient_analytical(pa)`

Replace the FD function `compute_steiner_perimeter_gradient`.  Scope:
accumulate ∂P_steiner/∂λ into an array of length `pa.n_active_vp`,
matching the current FD output.

Per triple point, per cell (3 cells per TP — the existing `tp_contrib_*`
arrays enumerate them), the contribution is

    ∂(ℓ_a + ℓ_b − ℓ_{ab})/∂λ_i
       = n_aᵀ (∂p_a/∂λ_i − ∂S/∂λ_i)
         + n_bᵀ (∂p_b/∂λ_i − ∂S/∂λ_i)
         − [standard ‖p_a − p_b‖ gradient w.r.t. λ_i]

For λ_i corresponding to VP slot `s`: `∂p_a/∂λ_i = δ_{a,s} d_s`.

```python
def compute_steiner_perimeter_gradient_analytical(pa):
    gradient = np.zeros(pa.n_active_vp, dtype=np.float64)
    if pa.n_triple_points == 0:
        return gradient

    geom   = _compute_tp_geometry(pa)
    dS_dp  = _compute_dS_dp(geom)
    dS_dl  = _compute_dS_dlambda(pa, geom, dS_dp)    # (n_tp, 3, dim)

    # Per (tp, cell) row: identify which slot is a, which is b.
    # tp_contrib_vp1 / tp_contrib_vp2 index into the 3 slots of
    # pa.tp_vp_indices[tp, :].  Decode the slot indices with a small
    # helper, or pre-compute during compile_arrays() (faster).
    ...
    return gradient
```

If slot decoding is slow, add `tp_contrib_slot1`, `tp_contrib_slot2`
int32 arrays to `PartitionArrays`, populated in `compile_arrays()`
alongside the existing `tp_contrib_*` fields — a minor compatibility-safe
addition (`= None` default, like the other optional `PartitionArrays`
fields).

### 5.5 `compute_steiner_area_jacobian_analytical(pa)` and the sparse twin

Same template as §5.4, but the kernel is the scalar-triple-product
gradient pattern from `compute_area_jacobian_analytical`.  Reuse the
existing formulas in `vectorized_area.py` by passing in the analytical
`∂S/∂λ` wherever that code currently treats `S` as an implicit variable.

The sparse variant is identical modulo the output layout (`(nnz,)` at
positions `(pa.jac_row, pa.jac_col)` instead of a dense
`(n_cells-1, n_active)` matrix).  Mirror the structure of
`compute_steiner_area_jacobian_sparse` in the current code.

### 5.6 Wiring

In `src/optimization/perimeter_optimizer.py`, no changes are needed if
you preserve the original public function names.  The clean approach:

1. Rename the current FD functions to `*_fd_reference` (retain them for
   validation).
2. Publish new names `*_analytical` in the module.
3. Make the public `compute_steiner_perimeter_gradient` /
   `compute_steiner_area_jacobian` / `_sparse` dispatch to the analytical
   version by default, and expose a module-level flag
   `USE_ANALYTICAL_STEINER = True` to fall back to FD for A/B testing.

Do **not** delete the FD functions — `testing/test_exact_hessian_vs_fd.py`
needs them as the independent reference.

### 5.7 Phase 2 acceptance

- `testing/test_sparse_jacobian_equivalence.py`: still passes at 1e-10
  (unchanged — it does not involve the objective gradient).
- `testing/test_exact_hessian_vs_fd.py` with `--obj-factor 0
  --lagrange-mode ones`: should now pass at `--atol 1e-8 --rtol 1e-6`
  (the constraint Hessian is still partly FD at this stage — only the
  Jacobian is fully analytical — so the improvement comes from removing
  one nested FD).
- Add `testing/test_steiner_gradient_analytical.py` that calls
  `compute_steiner_perimeter_gradient_analytical` and the renamed
  `_fd_reference` version on the same `pa` and asserts agreement to
  1e-6.  Template identical to `test_sparse_jacobian_equivalence.py`.

---

## 6. Phase 3 — Analytical Second Derivatives <a id="6-phase3"></a>

Phase 3 depends on Phase 2 having shipped — it reuses
`_compute_tp_geometry`, `_compute_dS_dp`, `_compute_dS_dlambda` and
builds on them.

### 6.1 `_compute_d2S_dp2(geom, dS_dp)` — ∂²S/∂p_k ∂p_l per triple point

Implement the formula from §2.5.  Return shape
`(n_tp, 3, 3, dim, dim, dim)`: entry `[tp, k, l, :, :, :]` is the
rank-3 tensor `∂²S/∂p_k ∂p_l`.

```python
def _compute_d2S_dp2(geom, dS_dp):
    n_tp, _, dim, _ = geom.K.shape
    d2S = np.zeros((n_tp, 3, 3, dim, dim, dim), dtype=np.float64)

    non_degen = ~geom.degen_mask
    if not np.any(non_degen):
        return d2S

    # --- (A) T_{jl} = δ_{jl} I − dS/dp_l for (j, l) ∈ {0..2}² ---
    T = np.zeros((n_tp, 3, 3, dim, dim), dtype=np.float64)
    eye = np.eye(dim)
    for j in range(3):
        for l in range(3):
            T[:, j, l] = (eye if j == l else 0) - dS_dp[:, l]

    # --- (B) ∂K_j/∂p_l for each (j, l) ---
    # K_j = (I - n_j n_j^T) / r_j
    # ∂n_j/∂p_l = K_j · T_{jl}
    # ∂r_j/∂p_l = n_j^T · T_{jl}   (row vector)
    # ∂K_j/∂p_l = [term1 + term2] / r_j - K_j · ∂r_j/∂p_l / r_j
    # where  term1 = -(K_j T_{jl}) n_j^T,  term2 = -n_j (K_j T_{jl})^T
    # Uses einsum heavily; derive and validate on a 2-triple-point toy
    # case before trusting it at scale.
    ...

    # --- (C) ∂M/∂p_l = Σ_j ∂K_j/∂p_l ---
    ...

    # --- (D) ∂²S/∂p_k ∂p_l = -M^{-1} (∂M/∂p_l) M^{-1} K_k + M^{-1} (∂K_k/∂p_l) ---
    ...

    return d2S
```

Implement incrementally.  Between each stage, test the running result
against central finite differences on `dS_dp` (from §5.2):

```python
# Verify d2S via central FD on dS_dp
eps = 1e-5
for l in range(3):
    pa.vp_lambda[...] += eps * (some perturbation along p_l direction)
    dS_dp_plus  = _compute_dS_dp(_compute_tp_geometry(pa))
    pa.vp_lambda[...] -= 2 * eps
    dS_dp_minus = _compute_dS_dp(_compute_tp_geometry(pa))
    # (dS_dp_plus - dS_dp_minus)/(2 eps) should match d2S_dp2[..., l, :, :, :]
```

This is the single highest-leverage validation in Phase 3.  If
`d2S_dp2` matches a central-FD of `dS_dp` to 1e-8, the rest of Phase 3
is mechanical application of the chain rule.

### 6.2 Chain rule to λ-space

Add `_compute_d2S_dlambda(pa, geom, d2S_dp2)` returning
`(n_tp, 3, 3, dim)` — `∂²S/∂λ_i ∂λ_j` with `i, j ∈ {0,1,2}` (slots of
the triple point).  Since each λ has its own `d_k`,

    ∂²S/∂λ_i ∂λ_j = (∂²S/∂p_{slot(i)} ∂p_{slot(j)}) · d_i · d_j

(Apply the rank-3 tensor first along its "column" direction with `d_j`
then along its "row" direction with `d_i`; the remaining dim index is the
output axis of S.)

### 6.3 `compute_steiner_perimeter_hessian_analytical(pa)`

Replace `compute_steiner_perimeter_hessian_fd`.  For each cell
contribution `ℓ_a + ℓ_b − ℓ_{ab}`:

    ∂²ℓ_a/∂λ_i ∂λ_j = Δ_iᵀ K_a Δ_j + n_aᵀ (−∂²S/∂λ_i ∂λ_j)

where `Δ_i = δ_{a, slot(i)} d_i − ∂S/∂λ_i`, `K_a = (I − n_a n_aᵀ)/ℓ_a`,
and similarly for `ℓ_b`.  The `ℓ_{ab}` term uses the regular-perimeter
Hessian template (see `docs/math/01-phase2-derivatives`, *Perimeter
Hessian*).

Output is `(hess_nnz,)` accumulated into the lower-triangle Hessian
entries via the `hess_offset_map` / pre-computed offset arrays.  Only VP
pairs within the same triple point produce non-zero entries; no cross-TP
contributions exist.

### 6.4 `compute_steiner_area_hessian_analytical(pa, multipliers)`

Same pattern as §6.3 but the "area second derivative" template is the
scalar-triple-product Hessian already coded for boundary triangles in
`vectorized_area.py`.  Reuse the sub-expressions: `C = u × v`, `normC`,
`n̂`, `g1 = d1 × v`, `g2 = u × d2`, `d1xd2 = d1×d2`.

The difference from the boundary-triangle case is that here all three
vertices (`p_a`, `p_b`, `S`) can depend on any of the three triple-point
λ's, so the algebra is messier, but each block has the same template.
Multiply by `multipliers[c]` per cell, as in
`compute_area_hessian_sparse`.

### 6.5 Phase 3 acceptance

- `testing/test_exact_hessian_vs_fd.py` with `--lagrange-mode random`
  should now pass at `--atol 1e-8 --rtol 1e-6` on every reference
  problem.
- A new `testing/test_steiner_hessian_analytical.py` assembles the
  Hessian values and compares to `compute_steiner_perimeter_hessian_fd` /
  `compute_steiner_area_hessian_fd` on the same `pa`; tolerance ~1e-4
  (the FD references are the noisy ones).
- `testing/compare_hessian_modes.py`: re-run and record the new numbers.
  Expected: on problems with many triple points, analytical Steiner
  slightly improves wall-clock (no inner FD loop) and removes any
  residual restoration-phase noise from inaccurate curvature.

---

## 7. Optional Fallback: Hybrid (Phase 2 only + FD Hessian) <a id="7-hybrid"></a>

If Phase 3 turns out to be more algebra than the implementing agent is
comfortable committing to, Phase 2 alone is still a strict improvement:

- Phase 2 gives analytical first derivatives everywhere.
- Keep `compute_steiner_perimeter_hessian_fd` and
  `compute_steiner_area_hessian_fd`, but change their internal calls from
  `compute_steiner_perimeter_gradient` (FD) to
  `compute_steiner_perimeter_gradient_analytical`.  Because the inner
  gradient is now exact, the outer FD is **single**-FD rather than
  nested-FD, which pushes the achievable validation tolerance from ~1e-4
  to ~1e-7.

In changed lines this is a 5-line change to each FD Hessian function
(replace the inner call and remove the `eps_inner = eps * 0.1` trick,
which exists only to manage nested-FD noise).

This fallback should **not** be the default outcome of the plan — Phase 3
is the proper conclusion.  But if you are mid-way through Phase 3 and
need to ship a partial result, the hybrid is a safe stopping point.

---

## 8. Validation Matrix <a id="8-validate"></a>

### 8.1 Test matrix

| Test | Pre-Phase-2 | Post-Phase-2 | Post-Phase-3 |
|---|---|---|---|
| `test_sparse_jacobian_equivalence.py` | passes @ 1e-10 | passes @ 1e-10 | passes @ 1e-10 |
| `test_exact_hessian_vs_fd.py`, `--lagrange-mode zero`, `--obj-factor 1` | ~1e-4 | ~1e-7 | ~1e-10 |
| `test_exact_hessian_vs_fd.py`, `--lagrange-mode ones`, `--obj-factor 0` | ~1e-4 | ~1e-8 | ~1e-10 |
| `test_exact_hessian_vs_fd.py`, `--lagrange-mode random` | ~1e-4 | ~1e-7 | ~1e-8 |
| `test_steiner_gradient_analytical.py` (NEW, Phase 2) | — | passes @ 1e-6 | passes @ 1e-6 |
| `test_steiner_hessian_analytical.py` (NEW, Phase 3) | — | — | passes @ 1e-4 (vs FD) |

The `random`-mode post-Phase-3 row is looser than the others because
`_compute_d2S_dp2` involves several dim × dim matrix inversions and its
floating-point accuracy is limited by the conditioning of `M`.  Triangles
close to the 120° boundary spike the condition number; the test should
emit a warning with `np.linalg.cond(geom.M[tp])` on failure.

### 8.2 Degenerate-case test

Add `testing/test_steiner_degenerate_case.py`: construct by hand a triple
point whose triangle has a 125° angle.  Verify:

- `_compute_dS_dp` returns `I` on the obtuse slot and 0 on the others.
- `_compute_d2S_dp2` returns 0 everywhere.
- The Steiner perimeter is
  `‖p_a − p_{obtuse}‖ + ‖p_b − p_{obtuse}‖ − ‖p_a − p_b‖ = 0`
  (since `S = p_{obtuse}` and one of `{p_a, p_b}` equals `p_{obtuse}`).

One-screen test; the only systematic way to exercise the degenerate
branch, which almost never fires in real problems.

### 8.3 Performance sanity check

`testing/compare_hessian_modes.py` records wall-clock times.  After
Phase 3 the `ipopt + exact_hessian` path should be roughly as fast per
iteration as before (the analytical Steiner kernels are more arithmetic
but no inner FD loop).  If it is significantly slower (> 20%), profile
with `cProfile` on a single IPOPT iteration.  Likely culprits:
`np.linalg.inv` on the per-TP `M` (mitigation: `np.linalg.solve` with a
pre-broadcast RHS) or accidental Python loops in the chain-rule code
(mitigation: collapse into one `np.einsum`).

---

## 9. Acceptance Criteria <a id="9-acceptance"></a>

Do not ship partial deliveries within a phase.

### Phase 1 — Validation harness

- [ ] `testing/test_sparse_jacobian_equivalence.py` exits 0 on the
      reference problem with `--atol 1e-10`.
- [ ] `testing/test_exact_hessian_vs_fd.py` exits 0 on the reference
      problem with default tolerances (`--atol 1e-4 --rtol 1e-3`) in
      each of the three `--lagrange-mode` settings.
- [ ] (Optional) `testing/test_exact_hessian_matvec.py` exits 0 for a
      larger mesh where `test_exact_hessian_vs_fd.py` prints SKIP.
- [ ] `testing/compare_hessian_modes.py` runs end-to-end and prints a
      sensible summary block (informational — no assertion on which
      method wins).
- [ ] `compare_hessian_modes.py` without `--no-profile` additionally
      prints the per-component breakdown: `perimeter_hess`, `area_hess`,
      `steiner_perim_hess_fd`, `steiner_area_hess_fd`, the Python-side
      sum, IPOPT's `PDSystemSolver` wall time, and an "other" bucket —
      all non-negative, percentages add to ~100%.
- [ ] The per-component breakdown is documented in
      `testing/README_testing.md` with a one-paragraph reading guide.
- [ ] Each script starts with a usage comment and exits 0 on PASS / 1
      on FAIL.

### Phase 2 — Analytical first derivatives

- [ ] `_compute_tp_geometry`, `_compute_dS_dp`, `_compute_dS_dlambda`
      exist in `vectorized_steiner.py` with docstrings, no circular
      imports, no dependency outside numpy + `PartitionArrays`.
- [ ] `compute_steiner_perimeter_gradient_analytical` and
      `compute_steiner_area_jacobian_analytical` (plus `_sparse`) exist
      and are wired as the defaults.
- [ ] The old FD functions are renamed with `_fd_reference` suffix and
      still callable; `USE_ANALYTICAL_STEINER` module flag toggles
      between them.
- [ ] `testing/test_steiner_gradient_analytical.py` passes at 1e-6 on
      the reference problem.
- [ ] `testing/test_exact_hessian_vs_fd.py` hits the Post-Phase-2
      tolerances in the §8.1 matrix.
- [ ] Translation-invariance identity `dS_dp.sum(axis=1) ≈ I` verified
      as an internal assertion or unit test on non-degenerate rows.
- [ ] Degenerate rows never invoke `np.linalg.inv` on a singular `M`.

### Phase 3 — Analytical second derivatives

- [ ] `_compute_d2S_dp2`, `_compute_d2S_dlambda` exist, with the
      central-FD internal validation checked during development.
- [ ] `compute_steiner_perimeter_hessian_analytical` and
      `compute_steiner_area_hessian_analytical` exist and are the
      default.
- [ ] `compute_steiner_perimeter_hessian_fd` and
      `compute_steiner_area_hessian_fd` are preserved under the
      `_fd_reference` naming convention for validation.
- [ ] `testing/test_exact_hessian_vs_fd.py` passes at 1e-8 absolute /
      1e-6 relative on the full Lagrangian Hessian, random multipliers,
      random `obj_factor` in {0, 0.5, 1}.
- [ ] Second-order translation-invariance identity (`d2S_dp2` summed
      over `(k, l)` ≈ 0) verified on non-degenerate rows during
      development.
- [ ] `testing/test_steiner_degenerate_case.py` passes.
- [ ] Wall-clock regression vs the FD path within 20% on
      `compare_hessian_modes.py`.

### Global

- [ ] No change to any regular-perimeter or boundary-triangle
      derivative formula, the Jacobian/Hessian sparsity patterns, the
      IPOPT adapter, or the L-BFGS/exact-Hessian toggle.
- [ ] Phases 2–3 change only `vectorized_steiner.py` + `testing/` +
      optional `partition_arrays.py` / `contour_partition.py` for cached
      slot indices.
- [ ] `pip install -e ".[ipopt]"` followed by
      `python scripts/refine_perimeter.py --method ipopt --exact-hessian
      --solution <ref> --config <ref yaml>` still runs to completion.

---

## 10. Reference File Paths <a id="10-paths"></a>

```
src/partition/vectorized_steiner.py           # Phases 2–3: all new derivative code
src/partition/partition_arrays.py             # Phase 2: optional cached tp_contrib_slot{1,2}
src/partition/contour_partition.py            # Phase 2: optional — populate cached slots in compile_arrays()
src/partition/vectorized_perimeter.py         # Read-only (regular-perimeter Hessian reference)
src/partition/vectorized_area.py              # Read-only (regular-area Hessian reference)
src/optimization/perimeter_optimizer.py       # Read-only for this plan
scripts/refine_perimeter.py                   # Entry point for acceptance runs
src/pipeline/io.py                            # load_partition_from_{base,refined}_file
testing/_hessian_test_utils.py                # Phase 1 NEW (optional shared helper)
testing/test_sparse_jacobian_equivalence.py   # Phase 1 NEW
testing/test_exact_hessian_vs_fd.py           # Phase 1 NEW; tighten tolerances after Phases 2–3
testing/test_exact_hessian_matvec.py          # Phase 1 NEW (optional)
testing/compare_hessian_modes.py              # Phase 1 NEW
testing/test_steiner_gradient_analytical.py   # Phase 2 NEW
testing/test_steiner_hessian_analytical.py    # Phase 3 NEW
testing/test_steiner_degenerate_case.py       # Phase 3 NEW (§8.2)
testing/README_testing.md                     # Phases 1–3: add entries for the above
docs/math/01-phase2-derivatives/main.tex      # Derivations of all currently-implemented quantities
```

### Reference problems

- **Small:** any base solution from `parameters/torus_10part.yaml`.
  Quick iteration cycle.
- **With triple points:** a Phase-2 iteration-1 checkpoint on the torus
  10-partition problem.  After one refinement iteration several triple
  points exist; they exercise all of the new code.
- **Edge case:** a manually constructed `PartitionArrays` where one
  triangle has a 125° angle (§8.2).  Small, synthetic, deterministic.

If no torus solution exists, generate one with
`python scripts/find_surface_partition.py --config parameters/torus_10part.yaml`,
then a short `scripts/refine_perimeter.py --solution ... --max-iterations 1`
run to produce a checkpoint with triple points.

---

**End of plan.**  Estimated effort: Phase 1 ~1–1.5 days; Phase 2 ~2 days;
Phase 3 ~3–4 days (the `∂²S/∂p²` derivation and vectorisation is the
bulk).  Total ~1 week of focused work.  The returns are (a) a regression
harness for all future Hessian work, (b) a ~4–6 orders of magnitude
tighter validation bound on the exact-Hessian path, and (c) removal of
every FD-epsilon knob from the refinement pipeline.
