# Implementation Plan — Dual Semismooth-Newton Projection

**Replace the linearly-convergent iterative constraint projection with an exact,
quadratically-convergent dual semismooth-Newton solve — the highest-leverage single speedup
for Phase 1, and a foundational improvement independent of the (experimental) territory term.
The replacement is also a *correctness* upgrade: the current iterative method does not compute
the true Euclidean projection at all (see "Empirical finding" below) — only a suboptimal,
non-idempotent feasible point.**

**Status:** SPECIFICATION — ready to implement. Not started. *Revised 2026-07-18 after empirical
verification of the current projection's behavior (see "Empirical finding" below). The core dual
algorithm (§2–§3) is unchanged — it already targets the true projection — but the correctness
target and validation strategy (§0, §1, §8, §9) were corrected: the original "match the iterative
output to ≤1e-8" guard rested on a false premise (that the iterative method computes the
projection) and has been replaced by validation against the true QP.*
**Target branch:** `feat/newton-projection` (branched from `main`; this document lives on
`main` and is inherited by that branch).
**Audience:** an implementing agent with NO prior conversation context. Self-contained for
implementation; where it says "read", read before writing code.
**Independence:** this is a **foundational** change to `src/optimization/projection.py` that
benefits *every* run (original energy, the territory-aware term, any N). It does **not** depend
on the territory-aware work on `feat/phase1-territory-aware-relaxation`. Validate and merge it
to `main` on its own; the territory branch will later rebase onto the faster projection.

---

## Empirical finding that reshapes the correctness target (measured 2026-07-18)

Before implementation, the current projection was measured against a trusted **independent**
reference — Dykstra's algorithm onto `{row-sum=1} ∩ {area hyperplane} ∩ {box}`, which the
Boyle–Dykstra theorem guarantees converges to the *exact* Euclidean projection — and certified
by a **solver-free KKT stationarity test**. Result:

> **`orthogonal_projection_iterative` does NOT compute the Euclidean projection.** It returns a
> feasible but objective-suboptimal, path-dependent point.

- The exact projection must satisfy, on box-inactive entries, `(U−Y)[i,k] = α_i + v_i·β_k` for
  some duals (α per vertex, β per cell). The Dykstra reference's KKT residual is machine-eps
  (~1e-15) in every test; the iterative method's is **0.17–0.31** — it does not lie in the dual
  span, so it is not the projection. Both are feasible, but Dykstra attains strictly lower
  `½‖U−Y‖²`.
- `max |iterative − true projection|` grows with N and interface crispness:

  | mesh  | V   | N  | regime   | frac clipped→0 | max‖iter−true‖ | KKT resid (Dykstra / iter) |
  |-------|-----|----|----------|----------------|----------------|----------------------------|
  | torus | 128 | 4  | interior | 0.00           | 0.039          | 1e-16 / 0.039              |
  | torus | 128 | 4  | binding  | 0.40           | 0.081          | 3e-15 / 0.17               |
  | torus | 128 | 10 | binding  | 0.64           | 0.12           | 2e-15 / 0.29               |

  The gap is ~0.01–0.04 even in the box-inactive *interior* (the method's pre-loop row
  renormalization and steps 10–12 — clip, then *multiplicative* row-renorm and column-rescale —
  are not Euclidean projections) and rises to ~0.12 in the crisp/**binding** regime, which is the
  actual partition operating point.
- Idempotency: re-projecting an *exactly* feasible point barely moves (7e-12), but the PGD-level
  operation `clip([1e-8,1−1e-8]) ∘ project` (what the optimizer runs each step) moves ~6.9e-8 —
  the real per-step noise floor. Asking the iterative method for `tol=1e-12` makes it *raise*
  (it floors at ~1e-8 area error).

**Two consequences, folded into §1 and §8:**

1. The dual-Newton method computes the **true** Euclidean projection, so it will *differ* from
   the iterative method by up to ~0.12 — "match the iterative output to ≤1e-8" (the original
   §8.1 guard) is both unachievable and the wrong target. The correctness reference is the
   **true QP** (independent Dykstra solver + KKT conditions), not the incumbent.
2. Because the per-step projection changes, the PGD *trajectory* changes: `newton` is a
   deliberate behavioral upgrade (exact, idempotent, no noise floor), **not** a bit-for-bit
   drop-in. Validate it end-to-end for partition *validity*, keeping `iterative` byte-identical
   as the default. (Reproduce via the KKT-certificate validation script committed under
   `testing/`; the same certificate is the primary §8 guard.)

---

## 0. Orientation — read before writing code

**Why this matters (motivation & measured evidence):**
- `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` §2.3 — the profiling shows the projection is
  **~86–91% of Phase-1 wall time**, converges only **linearly** (stalls at a ~1e-9 floor), and
  is **non-idempotent** (projecting an already-feasible point *moves* it and *costs* +9.2e-3
  energy — a noise floor that contributed to the frozen optimizer). This plan removes all three.
  (Refined by the direct measurement in "Empirical finding": the *bare* map is near-idempotent on
  *exactly* feasible points, but the PGD-level `clip ∘ project` step carries the ~6.9e-8 per-step
  floor — and, more fundamentally, the method is not the true projection at all.)
- `docs/plans/PHASE1_N1000_SCALING_PLAN.md` — lists the dual-Newton projection as a core
  performance item; this plan is the standalone, CPU-only realization of it (no GPU/sparse
  backend required — see §5).

**The code to analyze (this is the critical step — but note the current code does NOT define the
exact Euclidean projection; it only approximates it, see "Empirical finding". The Newton solve
targets the *true* projection. Extract the metric and constraints so the true QP is set up
identically to what the iterative method was *trying* to solve):**
- `src/optimization/projection.py`:
  - `orthogonal_projection_iterative(A, c, d, v, ...)` — the current method to replace. Extract
    **exactly**: the objective/metric being minimized (Euclidean vs `v`-weighted — check the
    `C`/`v_norm_squared` construction at lines ~65–67), the constraint set (the row/partition
    constraint via `c`, the area constraint via `d`, the box), the argument shapes/orientation
    of `A` (V×N vs N×V), and the stall/convergence criteria.
    **Confirmed by measurement:** the metric is plain **Euclidean** `½‖U−Y‖²` (the correction
    `η_i + v_i·λ_k` at line ~106 carries unit weight on the row term; it is *not* `v`-weighted).
    With `row-sum=1 ∧ U≥0` the upper bound `U≤1` is automatically satisfied, so the box reduces to
    `U≥0` and the per-vertex inner solve is a **probability-simplex** projection. `A` is `V×N`
    (rows = vertices, columns = cells); the row constraint targets `1` (the arg `c` is validated
    but the code hardcodes the row target to ones). Steps 1–9 are a genuine Euclidean *affine*
    projection onto `{row=1, area=d}`, but the pre-loop row renormalization (lines ~54–57) and
    steps 10–12 are multiplicative repairs, not Euclidean projections — this is precisely why the
    iterative fixed point is not the true projection.
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
find the nearest feasible `U`. The metric is **plain Euclidean** (confirmed against the code —
the correction `η_i + v_i·λ_k` carries unit weight on the row term; it is not `v`-weighted):

    min_U  (1/2)‖U − Y‖²   s.t.
        (row/partition)  Σ_k u_{i,k} = c_i        ∀ vertices i     (c_i = 1 in the partition case)
        (area)           Σ_i v_i u_{i,k} = d_k     ∀ cells k         (d_k the per-cell area target)
        (box)            0 ≤ u_{i,k} ≤ 1           (upper bound implied by row=1 ∧ U≥0)

`v ∈ ℝ^V` is the lumped mass. `V` = vertices, `N` = cells. The Newton solver solves **this exact
QP** — the true Euclidean projection. Because `u ≤ 1` is automatically satisfied whenever the row
constraint holds with `U ≥ 0`, the effective box is `U ≥ 0` and the per-vertex inner solve (§3)
is a projection onto the **probability simplex** (the cap-at-1 never binds).

**Crucial correction (see "Empirical finding").** `orthogonal_projection_iterative` does *not*
solve this QP — it returns a suboptimal feasible point that differs from the true projection by
up to ~0.12 in the crisp/binding regime (and ~0.01–0.04 even in the interior). So the correctness
reference for the Newton solve is an **independent QP solver (Dykstra) plus the KKT conditions**,
**not** the incumbent iterative output. Do not gate on equivalence to `iterative` (§8 revised
accordingly).

---

## 2. The dual formulation

Introduce multipliers `α ∈ ℝ^V` (row constraint, per vertex) and `β ∈ ℝ^N` (area constraint,
per cell). KKT stationarity for the box-constrained QP gives the **primal-from-dual** map
(Euclidean metric, confirmed in §1 — the row dual `α_i` enters with unit weight, the area dual
`β_k` with the mass weight `v_i`):

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
standard projection onto the capped simplex `{Σ = c_i, 0 ≤ · ≤ 1}` shifted by β. **With
`c_i = 1` the cap at 1 never binds** (a nonneg row summing to 1 has every entry ≤ 1), so the
inner solve is the plain probability-simplex projection — only the lower breakpoints `0` are
active, which halves the breakpoint bookkeeping. The V vertices are **independent**
(embarrassingly parallel; but cheap serial/vectorized too).

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
  metrics) must be **≤ 1e-10** (the iterative method floors at ~1e-8 area error and *raises* if
  asked for 1e-12), and **idempotent**: projecting an already-feasible `U` returns it with
  `‖P(U) − U‖ ≈ machine-eps`. (Measured nuance: the *bare* iterative map is already near-idempotent
  on *exactly* feasible points — any exactly feasible point is a fixed point — but the PGD-level
  `clip([1e-8,1−1e-8]) ∘ project` step moves ~6.9e-8 each iteration; that is the noise floor to
  eliminate. Under the exact projection the pre-clip is unnecessary — see §6.)
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
  `orthogonal_projection_iterative` as the default: **byte-for-byte unchanged when `newton` is not
  selected** (zero behavior change for existing runs). Note `iterative` is *not* an equivalence
  reference for `newton` (it is not the true projection) — it is only the legacy default.
  Selecting `newton` is a deliberate correctness/behavioral upgrade (exact, idempotent
  projection) that changes the PGD trajectory; validate it end-to-end (§8) rather than expecting
  bit-equality with `iterative` runs.
- **Pre-clip interaction:** `optimize()` currently does `A = np.clip(A, 1e-8, 1−1e-8)` before
  projecting (`pgd_optimizer.py` ~254 and ~317). That clip is a safeguard for the iterative
  method; under the exact `newton` projection it is unnecessary (the projection already returns
  exact, feasible entries, including hard 0s). Leave it untouched on the `iterative` path; for
  `newton` it is harmless-but-redundant — document the choice, and do not silently alter the
  `iterative` path while wiring the flag.
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

The whole point of this being independent: it is validated by **correctness (vs the true QP) +
exactness + speed + end-to-end validity**, on quick configs (small mesh, low N), in minutes.
**Critical change from the original spec:** do **NOT** validate by equivalence to
`orthogonal_projection_iterative` — that method is not the true projection ("Empirical finding"),
so the correct reference is an independent QP solver plus the KKT conditions.

1. **Correctness vs the true projection (the primary guard).** For a batch of random
   feasible-and-infeasible candidates `Y` — **including a crisp / near-one-hot "binding" regime**,
   not just interior inputs — at several V/N sizes and with realistic torus lumped mass `v`,
   check `orthogonal_projection_newton(Y)` against an **independent** exact projection:
   - a **KKT certificate** (solver-free): on box-inactive entries `(U−Y)[i,k] = α_i + v_i·β_k`
     (stationarity residual ≤ 1e-9), with the sign conditions holding on active entries;
   - agreement with a **Dykstra** reference projection: max abs difference ≤ 1e-8.

   `testing/test_newton_projection_equivalence.py` (name kept for continuity; it now tests
   equivalence to the *true projection*, not to the iterative method — mirror the certificate
   used in the pre-implementation check).
2. **Exactness / idempotency:** feasibility residuals (`row_error`, `area_error`, `non_neg_error`)
   ≤ 1e-10, and `‖P_newton(U) − U‖ ≈ machine-eps` on an already-feasible `U` (idempotent — the
   PGD clip-then-project ~6.9e-8 floor is eliminated).
3. **Characterize the difference from `iterative` (do not null it out):** on the same inputs,
   report `max|newton − iterative|` (expected up to ~0.12 in the binding regime) **and** show
   `newton` is strictly better — lower `½‖U−Y‖²`, exact feasibility, idempotent. This documents
   that the change is a correctness upgrade, not a regression.
4. **Speed:** count inner (Newton / simplex) iterations and wall time vs the iterative method on
   the same inputs — expect far fewer iterations and lower wall time (warm-started); record the
   ratio.
5. **End-to-end validity smoke:** one short low-N relaxation (e.g. `torus_10part.yaml`) with
   `projection_method: newton` must produce a **valid partition** — pass the dormant-cell and
   area-imbalance gates (`detect_dormant_cells` / `detect_area_imbalance`), with comparable or
   better final perimeter — at lower wall time. Because the projection semantics change, do
   **not** require the iterate-by-iterate trajectory or final density to match the `iterative`
   run bit-for-bit; require partition *validity / quality* and a sane, monotone-ish energy
   history.

---

## 9. Definition of done

- [ ] Projection QP (Euclidean metric + constraints) extracted from `projection.py`; confirmed
      the iterative method only *approximates* it ("Empirical finding" recorded).
- [ ] `docs/math/08-…/main.pdf` built; dual derivation + `J` + convergence/exactness proven.
- [ ] `orthogonal_projection_newton` implemented, vectorized, flag-gated (default iterative,
      byte-identical when off).
- [ ] Correctness guard passes: KKT certificate ≤ 1e-9 and agreement with an independent Dykstra
      projection ≤ 1e-8, on interior **and** binding inputs; idempotency ≈ machine-eps;
      feasibility ≤ 1e-10.
- [ ] Difference from `iterative` characterized (≈ up to 0.1 in the binding regime), with
      `newton` shown strictly better (lower objective, exact feasibility, idempotent).
- [ ] Speed measured: fewer inner iterations + lower wall time vs iterative, recorded.
- [ ] Low-N end-to-end smoke: `newton` yields a VALID partition (dormant + area-imbalance gates
      pass, perimeter comparable/better) at lower wall time — not required to match `iterative`
      bit-for-bit.
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
- J. P. Boyle, R. L. Dykstra. *A method for finding projections onto the intersection of convex
  sets in Hilbert spaces.* Lecture Notes in Statistics 37, 1986 — the reference projection used to
  validate exactness (converges to the *exact* Euclidean projection onto the intersection; used as
  ground truth in §8 in place of the non-projection incumbent).
- Internal: `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` §2.3 (the measured wall-time motivation),
  `docs/plans/PHASE1_N1000_SCALING_PLAN.md`, `src/optimization/projection.py` (the QP the
  iterative method *approximates*; the Newton solve computes it exactly).
