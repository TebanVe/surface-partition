# Implementation Plan — Dual Semismooth-Newton Projection

**Replace the linearly-convergent iterative constraint projection with an exact,
quadratically-convergent dual semismooth-Newton solve — the highest-leverage single speedup
for Phase 1, and a foundational improvement independent of the (experimental) territory term.**

**Status:** SPECIFICATION — ready to implement. Not started.
**Target branch:** `feat/newton-projection` (branched from `main`; this document lives on
`main` and is inherited by that branch).
**Audience:** an implementing agent with NO prior conversation context. Self-contained for
implementation; where it says "read", read before writing code.
**Independence:** this is a **foundational** change to `src/optimization/projection.py` that
benefits *every* run (original energy, the territory-aware term, any N). It does **not** depend
on the territory-aware work on `feat/phase1-territory-aware-relaxation`. Validate and merge it
to `main` on its own; the territory branch will later rebase onto the faster projection.

---

## 0. Orientation — read before writing code

**Why this matters (motivation & measured evidence):**
- `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` §2.3 — the profiling shows the projection is
  **~86–91% of Phase-1 wall time**, converges only **linearly** (stalls at a ~1e-9 floor), and
  is **non-idempotent** (projecting an already-feasible point *moves* it and *costs* +9.2e-3
  energy — a noise floor that contributed to the frozen optimizer). This plan removes all three.
- `docs/plans/PHASE1_N1000_SCALING_PLAN.md` — lists the dual-Newton projection as a core
  performance item; this plan is the standalone, CPU-only realization of it (no GPU/sparse
  backend required — see §5).

**The code to analyze (this is the critical step — the Newton solve must reproduce the EXACT
same projection the current code defines):**
- `src/optimization/projection.py`:
  - `orthogonal_projection_iterative(A, c, d, v, ...)` — the current method to replace. Extract
    **exactly**: the objective/metric being minimized (Euclidean vs `v`-weighted — check the
    `C`/`v_norm_squared` construction at lines ~65–67), the constraint set (the row/partition
    constraint via `c`, the area constraint via `d`, the box), the argument shapes/orientation
    of `A` (V×N vs N×V), and the stall/convergence criteria.
  - `orthogonal_projection_direct(...)` — a pre-existing "direct" variant; read it first, it may
    already encode part of the dual/closed-form structure and is a useful reference/starting point.
  - The validation helper(s) that check feasibility (`row_error`, `area_error`, `non_neg_error`).
- `src/optimization/pgd_optimizer.py` — how/where the projection is called each iteration, and
  the P2 `_reduced_gradient` dual solve (α per vertex, β per cell via Gauss–Seidel): the Newton
  projection uses the *same dual variables*; reuse conventions and warm-start ideas.
- `testing/validate_pgd_optimizations.py` and `testing/test_steiner_gradient_analytical.py` —
  the equivalence/FD test patterns to mirror.
- `CLAUDE.md` — conventions (Black 88, relative imports, `get_logger`, no `print` in `src/`,
  vectorization, zero-overhead-when-off), and `docs/math/AUTHORING_GUIDE.md` (you will create
  `docs/math/08-dual-newton-projection/`).

---

## 1. The projection problem (verify the exact form against the code)

Per PGD iteration, given a candidate density `Y ∈ ℝ^{V×N}` (post gradient step, infeasible),
find the nearest feasible `U`. Template (Euclidean — **confirm the metric in the code and adjust
if it is `v`-weighted**):

    min_U  (1/2)‖U − Y‖²   s.t.
        (row/partition)  Σ_k u_{i,k} = c_i        ∀ vertices i     (c_i = 1 in the partition case)
        (area)           Σ_i v_i u_{i,k} = d_k     ∀ cells k         (d_k the per-cell area target)
        (box)            0 ≤ u_{i,k} ≤ 1

`v ∈ ℝ^V` is the lumped mass. `V` = vertices, `N` = cells. The Newton solver MUST solve the
*identical* problem (metric, constraints) as `orthogonal_projection_iterative`, so its output
matches to tight tolerance (the equivalence test, §8, is the guard).

---

## 2. The dual formulation

Introduce multipliers `α ∈ ℝ^V` (row constraint, per vertex) and `β ∈ ℝ^N` (area constraint,
per cell). KKT stationarity for the box-constrained QP gives the **primal-from-dual** map
(Euclidean template; adjust `v` factors for a weighted metric):

    u_{i,k}(α, β) = clip( y_{i,k} + α_i + v_i·β_k ,  0,  1 )                                   (2.1)

The duals are fixed by requiring the two equality constraints to hold:

    (row_i)   Σ_k u_{i,k}(α,β) = c_i          ∀ i                                              (2.2)
    (area_k)  Σ_i v_i·u_{i,k}(α,β) = d_k       ∀ k                                              (2.3)

Because of the clip, `u(α,β)` is **piecewise-linear** in `(α,β)`, so (2.2)–(2.3) is a
**semismooth** system — solve it by semismooth Newton (Newton generalized to the kinks where an
entry hits 0 or 1). Quadratic local convergence; the solution is **exact** (no 1e-9 floor, no
non-idempotency).

---

## 3. The efficient nested algorithm (eliminate α per-vertex, Newton on β)

Do **not** solve the full `(V+N)`-dim system directly. Exploit the structure:

**Inner (per vertex, given β) — a capped-simplex projection.** With
`c_{i,k} = y_{i,k} + v_i·β_k` known, (2.2) becomes, for each vertex i independently:

    Σ_k clip(c_{i,k} + α_i, 0, 1) = c_i                                                        (3.1)

The left side is **monotone non-decreasing in α_i**, so α_i has a unique root — solvable exactly
(sort the `2N` breakpoints, O(N log N)) or by a monotone 1D Newton/bisection. This is the
standard projection onto the capped simplex `{Σ = c_i, 0 ≤ · ≤ 1}` shifted by β. The V vertices
are **independent** (embarrassingly parallel; but cheap serial/vectorized too).

**Outer (semismooth Newton on β, dimension N).** Define the area residual
`R_k(β) = Σ_i v_i·u_{i,k}(α(β), β) − d_k`, where `α(β)` is the inner solution. Solve `R(β)=0`
by semismooth Newton:

    β ← β − J(β)^{-1} R(β),    J = ∂R/∂β  (N×N, dense, from the generalized Jacobian +
                                            the implicit sensitivity dα/dβ of the inner solve)

**Key consequence:** the only linear solve is the **N×N** system `J δβ = −R` — and `N` is the
*cell count* (≤ ~1000), so this is a **tiny dense solve** (Cholesky/LU), trivial even at N=1000.
This is why the projection needs **no large linear solve and no GPU** (see §5). The math doc
(§7) must derive `J` (including the active-set / generalized-Jacobian terms and the `dα/dβ`
implicit sensitivity).

**Globalization & warm-start (essential for robustness):**
- Warm-start `(α, β)` from the **previous PGD iteration's** duals — the candidate `Y` barely
  changes step to step, so the duals are nearly right and Newton converges in very few steps.
- Damp the Newton step (line search on `‖R‖`) if a full step increases the residual; fall back
  to a bounded step. Semismooth Newton can need this from a cold start.

---

## 4. Numerical & correctness requirements

- **Exactness:** on output, `row_error`, `area_error`, `non_neg_error` (the code's own feasibility
  metrics) must be at or below the current method's *best* (≤ ~1e-10), and **idempotent**:
  projecting an already-feasible `U` returns it with `‖P(U) − U‖ ≈ machine-eps` (the current
  method fails this — it is the explicit non-idempotency to eliminate).
- **Stability:** guard the inner monotone solve against degenerate rows; the outer `J` may be
  near-singular if constraints are nearly dependent — regularize minimally (mirror the code's
  existing `C_reg`/`v_norm_squared` regularization) and document it.
- **Determinism / vectorization:** pure NumPy/SciPy, no Python loops over V or N in the inner
  solve (vectorize the breakpoint/monotone solve across vertices); the outer is O(N²)–O(N³) but
  N is small.

---

## 5. Sparse-CG note (why it is a fallback, not the main path)

The nested algorithm's only linear solve is the **N×N** outer system (§3), which is dense and
tiny — **sparse CG is not needed** at N ≤ 1000. Sparse/iterative CG would only be relevant if one
instead solved the **full `(V+N)` coupled system without eliminating α** (V is large). Do **not**
do that as the primary path. Provide a sparse-CG solve **only** as an optional fallback for the
non-eliminated formulation, gated off, for future very-large-N stress tests. **No GPU or explicit
parallelization is required anywhere in this plan** (the per-vertex inner solves are independent
and could later be parallelized/GPU'd for throughput, but that is out of scope and not needed for
correctness or the target speedup).

---

## 6. Implementation

- Add `orthogonal_projection_newton(A, c, d, v, ...)` in `src/optimization/projection.py`
  implementing §2–§4, matching the existing signature and returning the same object shape.
- **Flag-gated with fallback:** a config/selector (e.g. `projection_method: iterative | newton`,
  default `iterative`) chosen in `RelaxationConfig` / `ProjectedGradientOptimizer`. Keep
  `orthogonal_projection_iterative` as the default and the equivalence reference. Zero behavior
  change when `newton` is not selected.
- Warm-start state: thread the previous iteration's `(α, β)` through the optimizer so the Newton
  solve starts warm each PGD step.
- Style/standards per `CLAUDE.md`.

---

## 7. Math document to create

`docs/math/08-dual-newton-projection/` (follow `AUTHORING_GUIDE.md`): the QP and its dual;
(2.1)–(2.3); the nested elimination; the inner capped-simplex monotone solve; the outer
generalized Jacobian `J` incl. the `dα/dβ` sensitivity; the semismooth-Newton convergence
statement; the exactness/idempotency argument. Build the PDF; update `docs/math/Makefile` and
`docs/math/shared/references.bib`.

---

## 8. Validation (fast — no multi-day runs)

The whole point of this being independent: it is validated by **equivalence + speed + exactness**,
on quick configs (small mesh, low N), in minutes.

1. **Equivalence:** for a batch of random feasible-and-infeasible candidates `Y` (fixed seeds,
   several V/N sizes), `orthogonal_projection_newton(Y)` must match `orthogonal_projection_iterative(Y)`
   to a tight tolerance (both solve the same QP). Gate: max abs difference ≤ 1e-8.
   `testing/test_newton_projection_equivalence.py` (mirror `validate_pgd_optimizations.py`).
2. **Exactness/idempotency:** `‖P_newton(U) − U‖ ≈ machine-eps` on an already-feasible `U`
   (the current method fails this); feasibility residuals ≤ 1e-10.
3. **Speed:** count inner iterations and wall time vs the iterative method on the same inputs —
   expect far fewer iterations and lower wall time; record the ratio.
4. **End-to-end smoke:** one short low-N relaxation (e.g. `torus_10part.yaml`) with
   `projection_method: newton` produces a valid result matching the `iterative` run within
   tolerance, at lower wall time.

---

## 9. Definition of done

- [ ] Exact projection form extracted from `projection.py` (metric + constraints) and reproduced.
- [ ] `docs/math/08-…/main.pdf` built; dual derivation + `J` + convergence/exactness proven.
- [ ] `orthogonal_projection_newton` implemented, vectorized, flag-gated (default iterative).
- [ ] Equivalence test passes (≤ 1e-8); idempotency holds (≈ machine-eps); feasibility ≤ 1e-10.
- [ ] Speed measured: fewer inner iterations + lower wall time vs iterative, recorded.
- [ ] Low-N end-to-end smoke matches the iterative run at lower wall time.
- [ ] `CLAUDE.md` updated (new projection option, flag, test). Merge to `main` on its own merits.

---

## 10. References

- J. Nocedal, S. J. Wright. *Numerical Optimization*, 2nd ed., Springer, 2006 — KKT for QP
  (Ch. 16), semismooth/active-set methods, finite-difference & equivalence checking.
- L. Qi, J. Sun. *A nonsmooth version of Newton's method.* Mathematical Programming 58, 1993 —
  semismooth Newton and its quadratic convergence.
- M. Hintermüller, K. Ito, K. Kunisch. *The primal-dual active set strategy as a semismooth
  Newton method.* SIAM J. Optim. 13(3), 2002 — box-constrained QP as semismooth Newton (the
  active-set structure used here).
- Projection onto the (capped) simplex with an equality: e.g. L. Condat, *Fast projection onto
  the simplex and the ℓ1 ball*, Math. Programming 158, 2016 — the inner per-vertex solve.
- Internal: `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` §2.3 (the measured motivation),
  `docs/plans/PHASE1_N1000_SCALING_PLAN.md`, `src/optimization/projection.py` (the exact problem).
