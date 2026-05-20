---
name: Analytical Steiner Derivatives
overview: >
  Follow-up to docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md and
  docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md.  Replace every
  finite-difference step currently used for Steiner / triple-point
  derivatives with closed-form analytical expressions.  The
  Fermat–Torricelli point is the minimiser of a smooth strictly-convex
  function on the plane of the three VPs, so the implicit-function
  theorem gives its first and second derivatives in closed form.  Chain
  rule through the linear map ``p_i(λ_i) = λ_i V[v1_i] + (1−λ_i) V[v2_i]``
  then yields analytical gradients of the Steiner perimeter and analytical
  Jacobians / Hessians of the Steiner area contributions, eliminating all
  FD epsilon tuning and tightening the validation tolerance of the
  exact-Hessian path from ~1e-4 to ~1e-10.
prerequisites: >
  docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md must be complete.  In
  particular, the validation harness from its Phase B (especially
  testing/test_exact_hessian_vs_fd.py) must be in place and passing on
  the current FD-based Steiner code — otherwise there is no reference to
  regress against.
todos:
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
    content: Extend prior-plan tests to verify analytical vs FD Steiner gradients agree to 1e-6
    status: pending
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

# Analytical Steiner Derivatives Plan

## Table of Contents

1. [Context](#1-context)
2. [Mathematical foundations](#2-math)
3. [Degenerate case](#3-degen)
4. [Phase A — Analytical first derivatives](#4-phase-a)
5. [Phase B — Analytical second derivatives](#5-phase-b)
6. [Optional fallback: hybrid (analytical 1st, FD 2nd)](#6-hybrid)
7. [Validation](#7-validate)
8. [Acceptance criteria](#8-acceptance)
9. [Reference file paths](#9-paths)

---

## 1. Context <a id="1-context"></a>

This plan is the **third** document in a series. Read in order:

1. `docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md` — original plan,
   introduces the sparse-Jacobian and exact-Hessian infrastructure. It
   deliberately uses finite differences on the Steiner / triple-point
   contributions because the closed-form derivatives are "extremely
   verbose" (see its §2.3).
2. `docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md` — follow-up that
   replaces Python accumulation loops with `np.add.at` and adds a
   validation harness (sparse-vs-dense Jacobian, analytical-vs-FD
   Hessian, L-BFGS-vs-exact comparison).
3. **This plan** — replaces the remaining finite-difference blocks in
   `src/partition/vectorized_steiner.py` with closed-form analytical
   expressions. After it lands, every derivative the IPOPT exact-Hessian
   path sees is analytical.

### 1.1 Why bother

With the current FD Steiner contributions:

- The analytical-vs-FD Hessian validation can only be tightened to
  `abs_err < ~1e-4` (§6.2 of the original plan) because two nested
  finite differences introduce noise. A fully analytical path would
  pass at ~1e-10, which is a much stronger correctness guarantee.
- FD epsilon choices (`eps=1e-5`, `eps=1e-6`, `eps=1e-7`) are currently
  hand-tuned. They happen to work, but they are fragile — on meshes
  with very small triangles the inner FD can drown in floating-point
  cancellation.
- **The wall-clock cost is scale-dependent.** At small partition counts
  (N ≲ 20) `tp_affected_vps` is tiny and this is essentially a
  correctness / robustness win, not a speedup. At the target scale of
  this project — **thousands of cells** — the picture is different: the
  number of triple points grows as T ≈ 2N, and the current FD Steiner
  Hessian is an **O(N²)** cost per IPOPT iteration (outer FD over
  O(N) affected VPs × inner gradient that is itself O(N)). By the table
  in `docs/SCALABILITY_ANALYSIS.md` §1, this is ~250× more expensive at
  N = 1000 than at N = 5. The analytical formulas in this plan are
  **O(N)** per iteration. At scale this is a real performance unlock,
  not merely a numerical-accuracy one.

### 1.1.1 Scale target and applicability

This plan is listed as **Tier 2A** in `docs/SCALABILITY_ANALYSIS.md`.
Concretely, it is necessary but not sufficient for the project's
thousand-cell ambition:

| Problem size | Analytical Steiner impact | Still blocking? |
|---|---|---|
| N ≲ 20 | negligible perf win; big correctness win | no — current pipeline already works |
| N ≈ 20–100 | meaningful perf win (Steiner Hessian goes from ~30% of `h()` time to ~3%) | KKT Schur complement, iteration count |
| N ≈ 100–500 | Steiner cost no longer shows up in profiles | **dense Schur O(N³), iteration-count growth** |
| N ≈ 500–1000+ | required, but IPOPT monolithic NLP has hit its ceiling | **Tier 3 is needed** — Lloyd CVT initialiser, multigrid, augmented-Lagrangian decomposition, or curve-shortening flow (see `SCALABILITY_ANALYSIS.md` §3) |

Do not oversell this plan as "the thing that unlocks thousands of cells."
It is one necessary step in a longer chain. The final Tier 3 interventions
are out of scope here — they are separate roadmap items in
`SCALABILITY_ANALYSIS.md` action-items table.

### 1.2 Scope guard

Do **not** touch anything outside `src/partition/vectorized_steiner.py`
except:

- Possibly `src/partition/partition_arrays.py` if you decide to cache
  geometry-dependent quantities per triple point (optional — see §4.4).
- The three validation scripts under `testing/` added in the prior plan.
- The frontmatter todo status in the prior plan.

Do not change any of the following: the boundary-triangle area Hessian,
the regular perimeter Hessian, the Jacobian sparsity pattern, the
Hessian sparsity pattern, the IPOPT adapter, the `--exact-hessian`
wiring. All those are already exact and tested.

### 1.3 What this plan does **not** do

- It does not change `compute_steiner_points`. The analytical formula
  (Fermat point from barycentric weights) stays as-is for the forward
  evaluation; only the derivatives change.
- It does not attempt to smooth the degenerate-case discontinuity at
  exactly 120°. The derivatives are genuinely non-smooth there (see §3);
  this plan documents that and provides branch-safe code but does not
  invent a regularised Steiner point. If a user's optimisation oscillates
  across the 120° boundary, that is a separate modelling issue.
- It does not replace the FD inside the validation harness. The
  FD-based Hessian in `testing/test_exact_hessian_vs_fd.py` stays —
  it is the reference, and with analytical Steiner it should now agree
  with the adapter to ~1e-8.

---

## 2. Mathematical Foundations <a id="2-math"></a>

This section is self-contained. A fresh agent with undergraduate
multivariable calculus should be able to derive everything below from
the two starting points given.

### 2.1 Notation

At one triple point, three variable points `p_1, p_2, p_3 ∈ ℝ^dim`
(dim = 3 for surface meshes, dim = 2 for planar meshes). Each
`p_i` depends affinely on a scalar `λ_i ∈ [0,1]`:

    p_i(λ_i) = λ_i V[v1_i] + (1 − λ_i) V[v2_i]
    d_i := dp_i/dλ_i = V[v1_i] − V[v2_i]   (constant edge direction)

All second derivatives `d²p_i/dλ_i²` are zero. This is the same
structure as the boundary-triangle case.

Let `S ∈ ℝ^dim` be the Fermat–Torricelli (Steiner) point of the triangle
`(p_1, p_2, p_3)`. Define

    u_i := p_i − S
    r_i := ‖u_i‖
    n_i := u_i / r_i               (unit vector from S to p_i)
    K_i := (I − n_i n_iᵀ) / r_i    (dim × dim PSD matrix)
    M   := K_1 + K_2 + K_3         (dim × dim SPD matrix; see §3 for degeneracy)

The codebase abbreviates `M_i = K_i` in the plan text; pick whichever name
you prefer — I use `K_i` here to free `M_*` for the boundary-triangle area
matrices that live next door in `vectorized_area.py`.

### 2.2 The Fermat point as the solution of an implicit equation

`S` is the unique minimiser of the strictly convex function
`G(S) = ‖p_1 − S‖ + ‖p_2 − S‖ + ‖p_3 − S‖`. Its first-order optimality
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
same vector). Use this identity as a built-in unit test of your
implementation.

### 2.4 First derivative of Steiner perimeter and area

The Steiner perimeter contribution (per cell per triple point) is
`ℓ_a + ℓ_b − ℓ_{ab}` where

    ℓ_a = ‖p_a − S‖,   ℓ_b = ‖p_b − S‖,   ℓ_{ab} = ‖p_a − p_b‖.

For `p_a, p_b ∈ {p_1, p_2, p_3}` determined by the cell. Since
`dp_a/dλ_a = d_a` and `∂S/∂λ_i = ∂S/∂p_k · d_k` where `k` is the VP that
carries λ_i (trivially `k = i` because each VP has its own λ):

    ∂ℓ_a/∂λ_i
        = n_aᵀ · (δ_{ai} d_i − ∂S/∂λ_i)
        = n_aᵀ · (δ_{ai} d_i − (M^{-1} K_i) d_i).

with `n_a = (p_a − S)/ℓ_a`. The `ℓ_{ab}` term only depends on `p_a, p_b`
(not on S) and already has an analytical gradient — it's the same
formula the regular-perimeter code uses.

Analogously the Steiner area contribution per cell is

    A_cell = ½ ‖(p_b − p_a) × (S − p_a)‖ + corner_term

where the corner term doesn't involve S (it's a fixed mesh triangle).
The derivative w.r.t. λ_i is computed by chain rule through `∂p_a/∂λ_a`,
`∂p_b/∂λ_b`, and `∂S/∂λ_i = (M^{-1} K_i) d_i`.

The scalar-triple-product derivative identity is the same one already
used in `compute_area_jacobian_analytical` — reuse the pattern.

### 2.5 Second derivative of S

Differentiate (∘) with respect to `p_l`:

    ∂²S/∂p_k ∂p_l = (∂/∂p_l) [M^{-1} K_k]
                  = (∂M^{-1}/∂p_l) K_k + M^{-1} (∂K_k/∂p_l)
                  = −M^{-1} (∂M/∂p_l) M^{-1} K_k + M^{-1} (∂K_k/∂p_l).

The remaining ingredients are `∂K_j/∂p_l` for each `(j, l)` pair. Since
`K_j = (I − n_j n_jᵀ)/r_j` and both `n_j` and `r_j` depend on `p_l`
both explicitly (via `u_j = p_j − S`, if `l = j`) and implicitly (via
`S` always), we need to carry around `∂u_j/∂p_l = δ_{jl} I − ∂S/∂p_l`
already computed in Phase A and combine it.

Let `T_{jl} := ∂u_j/∂p_l = δ_{jl} I − ∂S/∂p_l`. Then

    ∂r_j/∂p_l      = n_jᵀ · T_{jl}    (a row vector, dim = 1 × dim)
    ∂n_j/∂p_l      = K_j · T_{jl}     (a dim × dim matrix)
    ∂(n_j n_jᵀ)/∂p_l = (K_j T_{jl}) n_jᵀ + n_j (K_j T_{jl})ᵀ
    ∂K_j/∂p_l = − (K_j T_{jl}) n_jᵀ / r_j  − n_j (K_j T_{jl})ᵀ / r_j
                   − K_j · (∂r_j/∂p_l) / r_j
              (one dim × dim × dim rank-3 tensor per (j, l))

Then

    ∂M/∂p_l = Σ_j ∂K_j/∂p_l.

Plug these into the formula above for `∂²S/∂p_k ∂p_l`. The result is a
rank-3 tensor of shape `(dim, dim, dim)` per `(k, l)` pair — three
"matrix columns" of second derivatives.

**Sanity check #2:** `Σ_{k,l} ∂²S/∂p_k ∂p_l = 0`. This is the second-order
version of translation invariance (translating all three vertices
together doesn't bend the Fermat-point trajectory). Use this as a unit
test once you have a working implementation.

### 2.6 Chain rule to λ-space

Given `∂²S/∂p_k ∂p_l` and the linearity `p_k = λ_k V[v1_k] + (1 − λ_k) V[v2_k]`,

    ∂²S/∂λ_i ∂λ_j = Σ_k Σ_l (∂²S/∂p_k ∂p_l) · d_k · d_l   (restricted to k = i-th VP, l = j-th VP)

which since each λ corresponds to exactly one VP simplifies to

    ∂²S/∂λ_i ∂λ_j = (∂²S/∂p_{k(i)} ∂p_{k(j)}) · d_i · d_j,

where `k(i)` denotes which of the three triangle vertices VP i is. (In
practice `k(i) ∈ {0, 1, 2}` via `pa.tp_vp_indices[tp, :]`.)

For Steiner perimeter `ℓ_a = ‖p_a − S‖`, the second derivative uses the
same template as the regular-perimeter Hessian (§2.3 of the original
plan):

    ∂²ℓ_a/∂λ_i ∂λ_j = (Δᵀ (I − n_a n_aᵀ)/ℓ_a Δ)_{λ_i, λ_j}

where `Δ = p_a − S` and the first derivatives `∂p_a/∂λ_i` and `∂S/∂λ_i`
are the ones computed in Phase A. The product rule also generates a
term involving `∂²S/∂λ_i ∂λ_j` (since `d²p_a/dλ² = 0`) — that's where
the hard part of §2.5 enters.

Concretely:

    Δ_i   = (δ_{a,i} d_i − ∂S/∂λ_i)              ← first derivative of Δ w.r.t. λ_i
    Δ_{ij} = − ∂²S/∂λ_i ∂λ_j                       ← second derivative of Δ w.r.t. (λ_i, λ_j)

    ∂²ℓ_a/∂λ_i ∂λ_j = Δ_iᵀ (I − n_a n_aᵀ)/ℓ_a Δ_j + n_aᵀ Δ_{ij}.

The first term is the "projected dot product" familiar from §2.3 of the
original plan. The second term is the contribution of the bent trajectory
of S, and it is the reason you need `∂²S/∂λ_i ∂λ_j` — if you skip it,
your Hessian is wrong, not merely approximate.

For the Steiner area contribution, apply the same chain-rule template
used in `compute_area_hessian_sparse` (see §2.4 of the original plan),
substituting `p_a → p_a`, `pc1 → p_b`, `p_in → S`, and carefully
tracking that `S` depends on all three λ's, not on just one.

---

## 3. Degenerate Case <a id="3-degen"></a>

`compute_steiner_points` already detects "any triangle angle ≥ 120°" and
replaces `S` by the obtuse vertex. Mathematically this is correct:
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

These two branches **do not agree** at the boundary. The boundary is a
measure-zero set in parameter space, so IPOPT should essentially never
sit on it, but implementation must pick one branch per evaluation.

### 3.2 Detection (reusing existing code)

`compute_steiner_points` already computes `all_angles` and
`degen_mask = all_angles.max(axis=1) >= 2π/3`. Factor that detection
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

In every analytical derivative function below (§4 and §5), the
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
because `r_{obtuse} = 0` ⇒ `K_{obtuse}` blows up. Mask before inversion.

### 3.4 Robustness near 120°

Just below 120°, `r_obtuse` is small and `K_obtuse` has a large operator
norm, which makes `M^{-1}` ill-conditioned. Add a numerical guard:

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
possible (e.g. `0.01 * median_edge_length`). For a uniform mesh `1e-10`
is usually fine; adjust if validation on a coarse mesh shows ill-conditioned
`M^{-1}` dominating the Hessian norm.

---

## 4. Phase A — Analytical First Derivatives <a id="4-phase-a"></a>

### 4.1 Shared geometry helper

Implement `_compute_tp_geometry(pa)` as described in §3.2. Place it
near the top of `src/partition/vectorized_steiner.py`, just below
`compute_steiner_points()`. Keep it private (underscore prefix).

### 4.2 `_compute_dS_dp(geom)` — ∂S/∂p_k per triple point

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

Add a unit test inside the module (or as a side-effect assertion in
debug builds) that on non-degenerate rows,
`np.allclose(dS_dp.sum(axis=1), np.eye(dim), atol=1e-8)` — the translation-
invariance identity. This is cheap and catches formula errors.

### 4.3 Lift to λ-space: `_compute_dS_dlambda(pa, geom, dS_dp)`

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
    # Apply dS_dp[tp, slot, :, :] to d[tp, slot, :]
    return np.einsum('tkij,tkj->tki', dS_dp, d)
```

### 4.4 `compute_steiner_perimeter_gradient_analytical(pa)`

Replace the FD function `compute_steiner_perimeter_gradient`. Scope:
accumulate ∂P_steiner/∂λ into an array of length `pa.n_active_vp`,
matching the current FD output.

Per triple point, per cell (there are 3 cells per TP — the existing
`tp_contrib_*` arrays enumerate them), the contribution is

    ∂(ℓ_a + ℓ_b − ℓ_{ab})/∂λ_i
       = n_aᵀ (∂p_a/∂λ_i − ∂S/∂λ_i)
         + n_bᵀ (∂p_b/∂λ_i − ∂S/∂λ_i)
         − [standard ‖p_a − p_b‖ gradient w.r.t. λ_i]

For λ_i corresponding to VP slot `s` of this triple point:
`∂p_a/∂λ_i = δ_{a,s} d_s` (only the VP that owns λ_i has a non-zero
∂p/∂λ). The implementation is a few dozen lines of `np.einsum`.

Skeleton:

```python
def compute_steiner_perimeter_gradient_analytical(pa):
    gradient = np.zeros(pa.n_active_vp, dtype=np.float64)
    if pa.n_triple_points == 0:
        return gradient

    geom   = _compute_tp_geometry(pa)
    dS_dp  = _compute_dS_dp(geom)
    dS_dl  = _compute_dS_dlambda(pa, geom, dS_dp)    # (n_tp, 3, dim)

    # Per (tp, cell) row: identify which slot is a, which is b
    # Already tabulated: tp_contrib_vp1 / tp_contrib_vp2 index into the
    # 3 slots of pa.tp_vp_indices[tp, :].  Decode the slot indices with
    # a small helper, or pre-compute during compile_arrays() (faster,
    # less intrusive).
    ...
    return gradient
```

If the slot decoding is slow, add `tp_contrib_slot1`, `tp_contrib_slot2`
int32 arrays to `PartitionArrays`, populated in `compile_arrays()`
alongside the existing `tp_contrib_*` fields. That is a minor
compatibility-safe addition (`= None` default like in the prior plan).

### 4.5 `compute_steiner_area_jacobian_analytical(pa)` and the sparse twin

Same template as §4.4, but the kernel is the scalar-triple-product
gradient pattern from `compute_area_jacobian_analytical`. Reuse the
existing formulas in that file by passing in the analytical `∂S/∂λ`
wherever that code currently treats `S` as an implicit variable.

The sparse variant is identical modulo the output layout
(`(nnz,)` at positions `(pa.jac_row, pa.jac_col)` instead of a dense
`(n_cells-1, n_active)` matrix). Mirror the structure of
`compute_steiner_area_jacobian_sparse` in the current code.

### 4.6 Wiring

In `src/optimization/perimeter_optimizer.py`, no changes are needed if
you preserve the original function names (`compute_steiner_perimeter_gradient`,
`compute_steiner_area_jacobian`). The clean approach:

1. Rename the current FD functions to `*_fd_reference` (retain them for
   validation).
2. Publish new names `*_analytical` in the module.
3. Make the public `compute_steiner_perimeter_gradient` /
   `compute_steiner_area_jacobian` / `_sparse` dispatch to the analytical
   version by default and expose a module-level flag
   `USE_ANALYTICAL_STEINER = True` to fall back to FD for A/B testing.

Do **not** delete the FD functions — the validation harness
(`testing/test_exact_hessian_vs_fd.py`) needs them as the independent
reference.

### 4.7 Phase A acceptance

Run the validation harness from the prior plan:

- `testing/test_sparse_jacobian_equivalence.py`: still passes at 1e-10
  (unchanged, this doesn't involve the objective gradient at all).
- `testing/test_exact_hessian_vs_fd.py` with `--obj-factor 0
  --lagrange-mode ones`: should now pass at `--atol 1e-8 --rtol 1e-6`
  (the constraint Hessian is still partly FD at this stage — only the
  Jacobian is fully analytical — so the improvement comes from removing
  one nested FD).
- Add a new script `testing/test_steiner_gradient_analytical.py`
  that calls `compute_steiner_perimeter_gradient_analytical` and the
  renamed `_fd_reference` version on the same `pa` and asserts
  agreement to 1e-6. Template is identical to
  `test_sparse_jacobian_equivalence.py` — just two calls and one
  `np.allclose`.

---

## 5. Phase B — Analytical Second Derivatives <a id="5-phase-b"></a>

Phase B depends on Phase A having shipped, because it reuses the
`_compute_tp_geometry`, `_compute_dS_dp`, `_compute_dS_dlambda` helpers
and builds on them.

### 5.1 `_compute_d2S_dp2(geom, dS_dp)` — ∂²S/∂p_k ∂p_l per triple point

Implement the formula from §2.5. Return shape
`(n_tp, 3, 3, dim, dim, dim)`: entry `[tp, k, l, :, :, :]` is the
rank-3 tensor `∂²S/∂p_k ∂p_l` at triple point `tp`.

Proposed structure:

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
    # where
    #   term1 = -(K_j T_{jl}) n_j^T
    #   term2 = -n_j (K_j T_{jl})^T
    # Uses einsum heavily; derive and validate on a 2-triple-point toy case
    # before trusting it at scale.
    ...

    # --- (C) ∂M/∂p_l = Σ_j ∂K_j/∂p_l ---
    ...

    # --- (D) ∂²S/∂p_k ∂p_l = -M^{-1} (∂M/∂p_l) M^{-1} K_k + M^{-1} (∂K_k/∂p_l) ---
    ...

    return d2S
```

Implement incrementally. Between each stage, test the running result
against central finite differences on `dS_dp` (from §4.2):

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

This is the single highest-leverage validation in the whole plan. If
`d2S_dp2` matches a central-FD of `dS_dp` to 1e-8, the rest of Phase B
is mechanical application of the chain rule.

### 5.2 Chain rule to λ-space

Add `_compute_d2S_dlambda(pa, geom, d2S_dp2)` returning
`(n_tp, 3, 3, dim)` — `∂²S/∂λ_i ∂λ_j` with `i, j ∈ {0,1,2}` (slots of
the triple point). Since each λ has its own `d_k`,

    ∂²S/∂λ_i ∂λ_j = (∂²S/∂p_{slot(i)} ∂p_{slot(j)}) · d_i · d_j

(Apply the rank-3 tensor `∂²S/∂p_k ∂p_l` first along its "column"
direction with `d_j` then along its "row" direction with `d_i`; the
remaining dim index is the output axis of S.)

### 5.3 `compute_steiner_perimeter_hessian_analytical(pa)`

Replace `compute_steiner_perimeter_hessian_fd`. For each cell contribution
`ℓ_a + ℓ_b − ℓ_{ab}`:

    ∂²ℓ_a/∂λ_i ∂λ_j = Δ_iᵀ K_a Δ_j + n_aᵀ (−∂²S/∂λ_i ∂λ_j)

where `Δ_i = δ_{a, slot(i)} d_i − ∂S/∂λ_i`, `K_a = (I − n_a n_aᵀ)/ℓ_a`,
and similarly for `ℓ_b`. The `ℓ_{ab}` term uses the regular-perimeter
Hessian template from §2.3 of the original plan.

Output is `(hess_nnz,)` accumulated into the lower-triangle Hessian
entries via the `hess_offset_map` / pre-computed offset arrays from the
prior plan. Only VP pairs within the same triple point produce non-zero
entries; no cross-TP contributions exist.

### 5.4 `compute_steiner_area_hessian_analytical(pa, multipliers)`

Same pattern as §5.3 but the "area second derivative" template is the
scalar-triple-product Hessian already coded for boundary triangles in
`vectorized_area.py`. Reuse the sub-expressions:

- `C = u × v`, `normC`, `n̂`, `g1 = d1 × v`, `g2 = u × d2`, `d1xd2 = d1×d2`.

The difference from the boundary-triangle case is that here all three
vertices (`p_a`, `p_b`, `S`) can depend on any of the three triple-point
λ's, so the algebra is messier, but each block has the same template.
Multiply by `multipliers[c]` per cell, as in
`compute_area_hessian_sparse`.

### 5.5 Phase B acceptance

- `testing/test_exact_hessian_vs_fd.py` with `--lagrange-mode random`
  should now pass at `--atol 1e-8 --rtol 1e-6` on every reference
  problem.
- A new `testing/test_steiner_hessian_analytical.py` that assembles
  the Hessian values and compares to `compute_steiner_perimeter_hessian_fd`
  / `compute_steiner_area_hessian_fd` on the same `pa`; tolerance ~1e-4
  (the FD references are the noisy ones, so loose tolerance is expected).
- `testing/compare_hessian_modes.py` from the prior plan: re-run and
  record the new numbers. Expected: on problems with many triple points,
  analytical Steiner should slightly improve wall-clock (no inner FD loop)
  and clearly remove any residual restoration-phase noise attributable to
  inaccurate curvature.

---

## 6. Optional Fallback: Hybrid (Phase A only + FD Hessian) <a id="6-hybrid"></a>

If Phase B turns out to be more algebra than the agent is comfortable
committing to, Phase A alone is still a strict improvement. Specifically:

- Phase A gives analytical first derivatives everywhere.
- Keep `compute_steiner_perimeter_hessian_fd` and
  `compute_steiner_area_hessian_fd`, but change their internal calls
  from `compute_steiner_perimeter_gradient` (FD) to
  `compute_steiner_perimeter_gradient_analytical`. Because the inner
  gradient is now exact, the outer FD is **single**-FD rather than
  nested-FD, which pushes the achievable validation tolerance from ~1e-4
  to ~1e-7. That is already a big win.

In terms of changed lines this is a 5-line change to each FD Hessian
function (replace the inner call and remove the `eps_inner = eps * 0.1`
trick, which exists only to manage nested-FD noise).

This fallback should **not** be the default outcome of the plan — Phase B
is the proper conclusion. But if you are mid-way through Phase B and need
to ship a partial result, the hybrid is a safe stopping point.

---

## 7. Validation <a id="7-validate"></a>

### 7.1 Test matrix

| Test | Pre-Phase-A | Post-Phase-A | Post-Phase-B |
|---|---|---|---|
| `test_sparse_jacobian_equivalence.py` | passes @ 1e-10 | passes @ 1e-10 | passes @ 1e-10 |
| `test_exact_hessian_vs_fd.py`, `--lagrange-mode zero`, `--obj-factor 1` | ~1e-4 | ~1e-7 | ~1e-10 |
| `test_exact_hessian_vs_fd.py`, `--lagrange-mode ones`, `--obj-factor 0` | ~1e-4 | ~1e-8 | ~1e-10 |
| `test_exact_hessian_vs_fd.py`, `--lagrange-mode random` | ~1e-4 | ~1e-7 | ~1e-8 |
| `test_steiner_gradient_analytical.py` (NEW) | — | passes @ 1e-6 | passes @ 1e-6 |
| `test_steiner_hessian_analytical.py` (NEW) | — | — | passes @ 1e-4 (vs FD) |

The second-to-last row (`random` mode, post-Phase-B) is looser than the
others because `_compute_d2S_dp2` involves several dim × dim matrix
inversions and its floating-point accuracy is limited by the conditioning
of `M`. Rows with triangles close to the 120° boundary can spike the
condition number; the test should emit a warning with
`np.linalg.cond(geom.M[tp])` on failure.

### 7.2 Degenerate-case test

Add `testing/test_steiner_degenerate_case.py`: construct by hand a triple
point whose triangle has a 125° angle. Verify:

- `_compute_dS_dp` returns `I` on the obtuse slot and 0 on the others.
- `_compute_d2S_dp2` returns 0 everywhere.
- The Steiner perimeter is `‖p_a − p_{obtuse}‖ + ‖p_b − p_{obtuse}‖ − ‖p_a − p_b‖ = 0`
  (since `S = p_{obtuse}` and one of `{p_a, p_b}` equals `p_{obtuse}`).

This is a one-screen test; it is the only way to exercise the degenerate
branch systematically, because it almost never fires in real problems.

### 7.3 Performance sanity check

`testing/compare_hessian_modes.py` from the prior plan records wall-clock
times. After Phase B the `ipopt + exact_hessian` path should be roughly as
fast per iteration as before (the analytical Steiner kernels are more
arithmetic but no inner FD loop). If it is significantly slower (> 20%),
profile with `cProfile` on a single IPOPT iteration. The likely culprits
are `np.linalg.inv` on the per-TP `M` matrix (mitigation: use
`np.linalg.solve` with a pre-broadcast RHS instead) or accidental Python
loops in the chain-rule code (mitigation: collapse into one `np.einsum`).

---

## 8. Acceptance Criteria <a id="8-acceptance"></a>

Phase A (first derivatives) is complete when:

- [ ] `_compute_tp_geometry`, `_compute_dS_dp`, `_compute_dS_dlambda`
      exist in `src/partition/vectorized_steiner.py` with docstrings,
      no circular imports, and no dependency outside numpy + the local
      `PartitionArrays`.
- [ ] `compute_steiner_perimeter_gradient_analytical` and
      `compute_steiner_area_jacobian_analytical` (plus `_sparse`) exist
      and are wired as the defaults.
- [ ] The old FD functions are renamed with `_fd_reference` suffix and
      still callable; `USE_ANALYTICAL_STEINER` module flag toggles between
      them.
- [ ] `testing/test_steiner_gradient_analytical.py` passes at 1e-6 on
      the reference problem.
- [ ] `testing/test_exact_hessian_vs_fd.py` with the sub-cases above
      hits the tolerances in the table in §7.1.
- [ ] Translation-invariance identity
      `dS_dp.sum(axis=1) ≈ I` verified as an internal assertion or unit
      test on non-degenerate rows.
- [ ] Degenerate rows never invoke `np.linalg.inv` on a singular `M`.

Phase B (second derivatives) is complete when:

- [ ] `_compute_d2S_dp2`, `_compute_d2S_dlambda` exist, with the central-FD
      internal validation checked during development (doesn't need to be
      kept as a test).
- [ ] `compute_steiner_perimeter_hessian_analytical` and
      `compute_steiner_area_hessian_analytical` exist and are the default.
- [ ] `compute_steiner_perimeter_hessian_fd` and
      `compute_steiner_area_hessian_fd` are preserved under the
      `_fd_reference` naming convention for validation.
- [ ] `testing/test_exact_hessian_vs_fd.py` passes at 1e-8 absolute /
      1e-6 relative on the full Lagrangian Hessian, random multipliers,
      random `obj_factor` in {0, 0.5, 1}.
- [ ] Second-order translation-invariance identity
      `d2S_dp2.sum over (k, l)` is (approximately) zero on non-degenerate
      rows, verified during development.
- [ ] `testing/test_steiner_degenerate_case.py` passes.
- [ ] Wall-clock regression vs FD path within 20% on
      `compare_hessian_modes.py`.

Global:

- [ ] No change outside `vectorized_steiner.py` + `testing/` + optional
      `partition_arrays.py` / `contour_partition.py` for cached slot
      indices.
- [ ] No change to the IPOPT adapter, the boundary-triangle derivatives,
      the sparsity patterns, or the L-BFGS/exact-Hessian toggle.
- [ ] Parent plan (`docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md`) §2.3
      note about "use finite differences" is updated to reference this
      plan's analytical replacement.

---

## 9. Reference File Paths <a id="9-paths"></a>

```
src/partition/vectorized_steiner.py           # All new derivative code lives here
src/partition/partition_arrays.py             # Optional: cache tp_contrib_slot{1,2} (Phase A §4.4)
src/partition/contour_partition.py            # Optional: populate cached slot arrays in compile_arrays()
src/optimization/perimeter_optimizer.py       # READ-ONLY for this plan
testing/test_sparse_jacobian_equivalence.py   # Existing — still passes unchanged
testing/test_exact_hessian_vs_fd.py           # Existing — tighten tolerances post-Phase-A and -B
testing/test_steiner_gradient_analytical.py   # NEW, Phase A §4.7
testing/test_steiner_hessian_analytical.py    # NEW, Phase B §5.5
testing/test_steiner_degenerate_case.py       # NEW, §7.2
testing/compare_hessian_modes.py              # Existing — re-run to record new timings
docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md  # Update §2.3 note
docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md  # Update §1 "what's remaining"
```

### Reference problems

- Small: any base solution from `parameters/torus_10part.yaml`. Start
  here — quick iteration cycle.
- With triple points: a Phase-2 iteration-1 checkpoint on the torus
  10-partition problem. After one refinement iteration several triple
  points exist; they exercise all of the new code.
- Edge case: manually construct a `PartitionArrays` where one triangle
  has a 125° angle (§7.2). Small, synthetic, deterministic.

---

**End of plan.** Estimated effort:

- Phase A: ~2 days (geometry helper + dS/dp + chain rule + 2 derivative
  functions + new gradient test).
- Phase B: ~3–4 days (the d²S/dp² derivation and vectorisation is the
  bulk; expect a full day of debugging against the central-FD internal
  test before the chain-rule application).
- Validation + docs: ~0.5 days.

Total: ~1 week of focused work. The returns are (a) a ~4–6 orders of
magnitude tighter validation bound on the exact-Hessian path, and (b)
removal of every FD-epsilon knob from the refinement pipeline.
