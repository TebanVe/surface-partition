# Implementation Plan — Exact Dual Projection for Phase 1

**Replace the linearly-convergent iterative constraint projection with the *exact* Euclidean
projection, computed by maximizing the constraint problem's **concave dual** `q(β)` — **L-BFGS-B
on `β` as the robust primary solver, with an optional semismooth-Newton polish tail** for the last
digits of feasibility. This is the highest-leverage single Phase-1 speedup and a foundational,
CPU-only improvement independent of the (experimental) territory term. It is also a *correctness*
upgrade: the current iterative method does not compute the true Euclidean projection at all (see
"Empirical finding" below) — only a suboptimal, non-idempotent feasible point.**

**Status:** SPECIFICATION — ready to implement. Not started. Revised twice:
- *2026-07-18* — after empirical verification that the incumbent iterative method is **not** the
  true projection: corrected the correctness target and validation strategy (validate against the
  true QP, not equivalence-to-iterative).
- *2026-07-19* — after an **independent adversarial review** (which reproduced the empirical
  finding with three unrelated reference solvers — gap even larger than first measured — and
  stress-tested the algorithm). The outer solve is reframed around the **concave dual `q(β)`**
  with **L-BFGS-B primary + semismooth-Newton polish**: the originally-specified "semismooth-Newton
  primary with an ε-ridge" path provably **stalls or explodes on a reachable empty-cell input**.
  Also folded in: cap-free inner solve mandated for *well-posedness* (not just efficiency); the
  explicit Jacobian and its **structural** singularity `J·1=0`; the KKT certificate fixed to use
  the solver's **own** duals (an LS-fit false-negatives on saturated rows); the pre-clip treated as
  a real algorithmic choice (not "harmless"); the phantom `_reduced_gradient` reference removed; and
  the difference-from-iterative magnitude re-scoped.

**Naming note:** the config flag value, the new function, and the test keep the label **`newton`**
(`projection_method: newton`, `orthogonal_projection_newton`, `test_newton_projection_equivalence.py`)
for branch/file continuity (`feat/newton-projection`, this plan's filename). The `newton` path
implements the **exact dual projection** — L-BFGS-B ascent on the concave dual with an optional
semismooth-Newton polish; "Newton" survives in the name as the polish. Do not read the name as
"semismooth-Newton is the primary solver".

**Target branch:** `feat/newton-projection` (branched from `main`; this document lives on
`main` and is inherited by that branch).
**Audience:** an implementing agent with NO prior conversation context. Self-contained for
implementation; where it says "read", read before writing code.
**Independence:** this is a **foundational** change to `src/optimization/projection.py` that
benefits *every* run (original energy, the territory-aware term, any N). It does **not** depend
on the territory-aware work on `feat/phase1-territory-aware-relaxation`. Validate and merge it
to `main` on its own; the territory branch will later rebase onto the faster projection.

---

## Empirical finding that reshapes the correctness target (measured 2026-07-18, re-confirmed 2026-07-19)

Before implementation, the current projection was measured against a trusted **independent**
reference — Dykstra's algorithm onto `{row-sum=1} ∩ {area hyperplane} ∩ {box}`, which the
Boyle–Dykstra theorem guarantees converges to the *exact* Euclidean projection — and certified
by a KKT stationarity test. The 2026-07-19 adversarial review reproduced this from scratch with
**three algorithmically unrelated references** (closed-form affine projection, Dykstra, and dual
L-BFGS with an exact simplex oracle), which agree pairwise to **≤ 4e-9** and are cross-checked
against `scipy` trust-constr. Result:

> **`orthogonal_projection_iterative` does NOT compute the Euclidean projection.** It returns a
> feasible but objective-suboptimal, path-dependent point.

- The exact projection must satisfy, on box-inactive entries, `(U−Y)[i,k] = α_i + v_i·β_k` for
  some duals (α per vertex, β per cell). The reference's KKT residual is machine-eps (~1e-15) in
  every test; the iterative method's is **0.17–0.31** — it does not lie in the dual span, so it is
  not the projection. Both are feasible, but the true projection attains strictly lower `½‖U−Y‖²`
  (the iterative point's objective excess is **+26% to +75%** in the binding regime).
- `max |iterative − true projection|` is **input-dependent** and grows with N and interface
  crispness:

  | mesh  | V   | N  | regime   | frac clipped→0 | max‖iter−true‖ |
  |-------|-----|----|----------|----------------|----------------|
  | torus | 128 | 4  | interior | 0.00           | 0.008–0.04     |
  | torus | 128 | 4  | binding  | ~0.4           | 0.08–0.35      |
  | torus | 128 | 10 | binding  | ~0.7           | 0.12–0.9       |
  | torus | 5280 (real level-0) | 10 | crisp | — | 0.40 |

  At the **realistic crisp PGD operating point** (pre-clipped near-one-hot densities, as fed
  through `pgd_optimizer.py:317`) the gap is **0.146** with **+34%** objective excess. Do not
  quote a single number: it ranges ~0.01 (strictly interior) to ~0.9 (harsh inputs).
- **Sharper mechanism:** the *interior* gap is entirely the incumbent's pre-loop **multiplicative
  row-renormalization** (`projection.py:54-57`): `|iterative − P_affine(rownorm(Y))| ≈ 6e-12`, i.e.
  the incumbent returns the exact affine projection *of the row-renormalized input*. Steps 1–9
  (`projection.py:81-109`) are the exact Euclidean affine projection (the `C` matrix at
  `projection.py:65-68` verified term-for-term); steps 10–12 (clip, then multiplicative row-renorm
  and column-rescale) are the non-projection repairs.
- **Idempotency:** re-projecting an *exactly* feasible point barely moves (~1e-11), but the
  PGD-level operation `clip([1e-8,1−1e-8]) ∘ project` (what the optimizer runs each step) moves
  ~3–7e-8 — the real per-step noise floor. Asking the iterative method for `tol=1e-12` makes it
  *raise* (it floors at ~1e-8…1e-10 area error, input-dependent).

**Two consequences, folded into §1 and §8:**

1. The dual projection computes the **true** Euclidean projection, so it will *differ* from the
   iterative method by an input-dependent amount up to ~0.9 — "match the iterative output to
   ≤1e-8" (the original guard) is both unachievable and the wrong target. The correctness
   reference is the **true QP** (independent Dykstra solver + the solver's own KKT duals), not
   the incumbent.
2. Because the per-step projection changes, the PGD *trajectory* changes: the `newton` path is a
   deliberate behavioral upgrade (exact, idempotent, no noise floor), **not** a bit-for-bit
   drop-in. Validate it end-to-end for partition *validity*, keeping `iterative` byte-identical
   as the default.

**Reproducible evidence.** This whole section is documented as a measured study in
`docs/experiments/03-dual-projection-verification/` (`main.pdf` + `make_figures.py`, seeded
synthetic inputs — no `results/` run) with the anchors above (gap up to 0.86 / +64% objective;
KKT residual 1e-15 vs 0.01–1.16; interior-gap mechanism 6e-12).

---

## 0. Orientation — read before writing code

**Why this matters (motivation & measured evidence):**
- `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` §2.3 — the profiling shows the projection is
  **~86–91% of Phase-1 wall time** (this measurement is at **fine-level** meshes, V ≈ 1.7e5;
  see §8.4), converges only **linearly** (stalls at a ~1e-9 floor), and is **non-idempotent**
  (projecting an already-feasible point *moves* it and *costs* +9.2e-3 energy — a noise floor
  that contributed to the frozen optimizer). This plan removes all three. (Refined by the direct
  measurement in "Empirical finding": the *bare* map is near-idempotent on *exactly* feasible
  points, but the PGD-level `clip ∘ project` step carries the per-step floor — and, more
  fundamentally, the method is not the true projection at all.)
- `docs/plans/PHASE1_N1000_SCALING_PLAN.md` — lists the dual projection as a core performance
  item; this plan is the standalone, CPU-only realization of it (no GPU/sparse backend required
  — see §5).

**The code to analyze (this is the critical step — but note the current code does NOT define the
exact Euclidean projection; it only approximates it, see "Empirical finding". The dual solve
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
- `src/optimization/pgd_optimizer.py` — how/where the projection is called each iteration
  (`optimize()`, the `orthogonal_projection_iterative(...)` calls at ~234, ~257, ~318, and the
  pre-clip `np.clip(A, 1e-8, 1−1e-8)` at ~254 and ~317). **Note:** an earlier draft of this plan
  pointed here for a "`_reduced_gradient` dual solve (α/β)" — *that code does not exist* (`grep -rn
  _reduced_gradient src/` → no hits). It is an **unimplemented future item** (shared duals, P2 of
  `PHASE1_N1000_VALIDITY_PLAN.md`); do not go looking for it. The dual variables here (α per
  vertex, β per cell) are introduced fresh by §2.
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

`v ∈ ℝ^V` is the lumped mass. `V` = vertices, `N` = cells. The dual solver computes **this exact
QP** — the true Euclidean projection. Because `u ≤ 1` is automatically satisfied whenever the row
constraint holds with `U ≥ 0`, the effective box is `U ≥ 0` and the per-vertex inner solve (§3)
is a projection onto the **probability simplex** (the cap-at-1 never binds).

**Feasibility of the QP.** It is feasible iff `Σ_k d_k = Σ_i v_i` (the row constraints force
`Σ_k Σ_i v_i u_{i,k} = Σ_i v_i`); with `d_k = Σv/N` and `d ≥ 0` the constant-column point
`u_{i,k} = d_k/Σv` is always feasible. The production code guarantees this (`d = Σv/N`,
`pgd_optimizer.py:256`), but `orthogonal_projection_newton` is a public function — **assert**
`|Σd − Σv| ≤ tol·Σv` on entry with a clear error (§4).

**Crucial correction (see "Empirical finding").** `orthogonal_projection_iterative` does *not*
solve this QP — it returns a suboptimal feasible point that differs from the true projection by an
input-dependent amount (~0.01 interior, ~0.1–0.2 at the realistic crisp operating point, up to
~0.9 on harsh inputs). So the correctness reference is an **independent QP solver (Dykstra) plus
the KKT conditions checked with the solver's own duals**, **not** the incumbent iterative output.
Do not gate on equivalence to `iterative` (§8 revised accordingly).

---

## 2. The dual formulation and its concave dual function

Introduce multipliers `α ∈ ℝ^V` (row constraint, per vertex) and `β ∈ ℝ^N` (area constraint,
per cell). KKT stationarity for the box-constrained QP gives the **primal-from-dual** map
(Euclidean metric, confirmed in §1 — the row dual `α_i` enters with unit weight, the area dual
`β_k` with the mass weight `v_i`):

    u_{i,k}(α, β) = clip( y_{i,k} + α_i + v_i·β_k ,  0,  1 )                                   (2.1)

The duals are fixed by requiring the two equality constraints to hold:

    (row_i)   Σ_k u_{i,k}(α,β) = c_i          ∀ i                                              (2.2)
    (area_k)  Σ_i v_i·u_{i,k}(α,β) = d_k       ∀ k                                              (2.3)

**The backbone of the algorithm is the *partial dual function* over β.** For fixed β, the inner
problem (2.2) — projecting each row onto the probability simplex shifted by `v_i·β` — is solved in
closed form (§3), which *eliminates both α and the primal*. Substituting back gives the concave,
`C¹`, piecewise-quadratic partial dual

    q(β) = Σ_i [ min_{u_i ∈ Δ} ( ½‖u_i − y_i‖² − v_i·β·u_i ) ] + β·d ,     Δ = probability simplex

whose gradient is exactly the (negated) area residual (Danskin / envelope theorem):

    ∇q(β) = d − Σ_i v_i·u_i(β) = − R(β) ,    R_k(β) := Σ_i v_i·u_{i,k}(β) − d_k .              (2.4)

The true projection is `u(β★)` at the **unique maximizer** `β★` of the concave `q` (unique up to
the gauge below). This reframing is what makes the outer solve both robust and simple: **maximize
a concave `C¹` function**, for which globally-convergent first-order methods (L-BFGS-B) apply
directly, with semismooth Newton available as an optional accelerated polish (§3).

**Gauge (structural, always present).** The simplex projection is shift-invariant per row, so
`(α, β) → (α − t·v, β + t·1)` leaves every `u_{i,k}` unchanged. Hence `q` is constant along `1`,
`∇q·1 = −1ᵀR ≡ 0` (verified numerically to 1e-14), and the maximizer is a 1-D ridge — the *primal*
`u(β)` is nonetheless unique. Fix the gauge (e.g. `Σβ = 0`, or pin one component) for any
Newton-style linear solve; L-BFGS simply drifts harmlessly along the ridge.

---

## 3. The efficient nested algorithm (inner simplex per vertex, outer concave-dual ascent on β)

Do **not** solve the full `(V+N)`-dim system directly. Exploit the structure:

**Inner (per vertex, given β) — a probability-simplex projection. CAP-FREE IS MANDATORY.** With
`z_{i,k} = y_{i,k} + v_i·β_k` known, (2.2) for vertex i is: find `α_i` with
`Σ_k clip(z_{i,k} + α_i, 0, 1) = 1`, i.e. project `z_i` onto the probability simplex. Solve it
**cap-free** — project onto `{Σ = 1, · ≥ 0}` (Condat/Michelot: sort `z_i`, find the threshold τ,
`u_{i,k} = max(z_{i,k} − τ, 0)`), which has `N` (not `2N`) breakpoints. **Keeping the `clip(·,0,1)`
cap in the inner solve is a well-posedness bug, not merely slower:** whenever a row's top entry
exceeds its runner-up by more than 1, the capped equation has a *flat interval* of solutions in
`α_i` (e.g. `z_i=[2.5,−0.3,−0.8] → every α ∈ [−1.5, 0.3]` satisfies `Σ clip=1`), so `α_i(β)` is
non-unique and `dα/dβ` is undefined — which breaks the outer Jacobian. The cap-free simplex solve
gives the unique `α_i` (and the identical primal `u_i`, since `U ≤ 1` is implied). The V vertices
are **independent** and the whole inner pass is a single vectorized simplex projection
(O(V·N log N)); it is **stateless** (recomputed from β each outer step — never warm-start the inner
active sets, that reintroduces the degeneracies).

**Outer — maximize the concave dual `q(β)` (2.4). Primary: L-BFGS-B; polish: semismooth Newton.**

*Primary solver — L-BFGS-B on `−q(β)` (dimension N).* Value and gradient (`∇q = −R`, eq. 2.4)
each cost one vectorized inner pass. Because `q` is concave and `C¹`, L-BFGS-B with a standard
line search is **globally convergent** — no Jacobian assembly, no singular linear solve, no
active-set/generalized-Jacobian selection, and it **handles the empty-cell pathology natively**
(the gradient component `−R_k = +d_k > 0` drives `β_k` up until a dead cell re-enters some active
set). Measured across every case (incl. the pathologies of §3-failure-modes): **5–14 iterations
cold, 1–2 warm-started**, feasibility ~1e-9. Warm-start β from the previous PGD step (§6).

*Optional polish — 1–2 semismooth-Newton steps* to drive feasibility from ~1e-9 to the §4 target
(≤1e-10) and shave warm-start iterations. The area residual `R(β)` is semismooth; its generalized
Jacobian, derived through the inner solve (with active set `A_i = {k : z_{i,k}+α_i > 0}`,
`m_i = |A_i| ≥ 1`, and `∂α_i/∂β_l = −v_i·[l∈A_i]/m_i`), is

    J_{kl} = ∂R_k/∂β_l = δ_{kl} · Σ_{i: k∈A_i} v_i²  −  Σ_{i: k,l ∈ A_i} v_i²/m_i             (3.1)

i.e. `J = Σ_i v_i²·(diag(a_i) − a_i a_iᵀ/m_i)` with `a_i` the 0/1 indicator of `A_i`. **`J` is
symmetric positive-semidefinite** (each term is `v_i²` times a PSD "centering" matrix; it equals
`−∇²q`) and **always exactly singular** with `J·1 = 0` (the §2 gauge; measured ‖J·1‖∞ ≈ 5e-15).
The Newton step solves `J δβ = −R` on the **gauge-fixed reduced space** (`Σβ = 0`, or drop one
row/column) — never a bare ε-ridge (see §4). Use `q` itself as the line-search merit: since `J`
is PSD, `δβ = −J⁺R` is a **provable ascent direction** for `q` (`∇q·δβ = RᵀJ⁺R ≥ 0`), giving a
sound globalization — unlike a `‖R‖` merit, which has no such guarantee and cannot revive a dead
cell (the Newton direction has zero component in that coordinate). **Dead-cell safeguard:** if `R`
has a component outside `range(J)` (an empty cell → a zero row of `J` with `R_k ≠ 0`), do not take
a Newton step in that coordinate; fall back to the L-BFGS/gradient ascent on `q`.

**Key consequence for §5:** the *primary* (L-BFGS) path has **no linear solve at all**. The only
linear solve is the **tiny gauge-fixed `(N−1)×(N−1)`** system inside the optional Newton polish —
`N` is the cell count (≤ ~1000), a trivial dense LDLᵀ/Cholesky-of-reduced solve. No large linear
solve, no GPU (§5). The math doc (§7) derives `q`, `∇q`, and `J`.

**Warm-start (essential for the 1–2-iteration steady state).** Thread only **β** (N floats) from
the previous PGD iteration — the candidate `Y` barely changes step to step, so `β` is nearly
right. **α is not persistent state**: the inner solve is direct and recomputes α exactly from β;
threading α is unnecessary and wrong across mesh refinement (V and α's dimension change; β's does
not).

---

## 4. Numerical & correctness requirements

- **Exactness:** on output, `row_error`, `area_error`, `non_neg_error` (the code's own feasibility
  metrics) must be **≤ 1e-10** (the iterative method floors at ~1e-8 area error and *raises* if
  asked for 1e-12), and **idempotent**: projecting an already-feasible `U` returns it with
  `‖P(U) − U‖ ≈ machine-eps` (trivially achieved — a feasible point has `R = 0`, so the warm dual
  solve terminates immediately). Under the exact projection the PGD `clip([1e-8,1−1e-8]) ∘ project`
  noise floor is eliminated; the pre-clip becomes an explicit choice (§6).
- **Input assertion:** require `|Σd − Σv| ≤ tol·Σv` on entry (else the QP is infeasible, §1) and
  raise a clear error — the public function must not silently return garbage.
- **Stability / regularization — do NOT "mirror the code's `C_reg`" ε-ridge.** The concave dual is
  globally solvable by L-BFGS **without any regularization**. For the optional Newton polish, `J`'s
  *structural* kernel `span{1}` is handled by the **gauge-fixed reduced solve**, not an ε-ridge:
  an ε-ridge is actively dangerous because `J` can have an *additional* zero row when a cell is
  empty (no active entry at any vertex) while `R_k = −d_k ≠ 0` — the system is then **inconsistent**
  and the ridge produces `δβ_k = d_k/ε ≈ 1e10` (measured 3.7e10 at ε=1e-10), an explosion. Detect
  dead cells (empty active column) and route them through the L-BFGS/gradient fallback (§3), which
  revives them. Assert `1ᵀR ≈ 0` as a runtime invariant.
- **Determinism / vectorization:** pure NumPy/SciPy, no Python loops over V or N in the inner
  solve (vectorize the sort-based simplex projection across vertices); the outer L-BFGS is a few
  N-vector operations per iteration, the optional polish is O(N²)–O(N³) but N is small. The inner
  solve is stateless (recomputed from β) — see §3.

---

## 5. Sparse-CG note (why it is irrelevant on the chosen path)

The **primary** (L-BFGS) outer solve has **no linear system at all**. The only linear solve is the
tiny gauge-fixed `(N−1)×(N−1)` dense system in the *optional* Newton polish (§3), trivial even at
N=1000 — **sparse CG is not needed**. Sparse/iterative CG would only be relevant if one instead
solved the **full `(V+N)` coupled system without eliminating α** (V is large). Do **not** do that.
Provide a sparse-CG solve **only** as an optional fallback for the non-eliminated formulation,
gated off, for future very-large-N stress tests. **No GPU or explicit parallelization is required
anywhere in this plan** (the per-vertex inner solves are independent and could later be
parallelized/GPU'd for throughput, but that is out of scope and not needed for correctness or the
target speedup).

---

## 6. Implementation

- Add `orthogonal_projection_newton(A, c, d, v, ...)` in `src/optimization/projection.py`
  implementing §2–§4 (L-BFGS-B ascent on the concave dual `q(β)`, optional Newton polish),
  matching the existing signature and returning the same object shape. (Name retained for
  continuity — see the naming note in the header; the method is dual/L-BFGS-based.)
- **Flag-gated with fallback:** a config/selector (`projection_method: iterative | newton`,
  default `iterative`) in `RelaxationConfig` / `ProjectedGradientOptimizer`. Keep
  `orthogonal_projection_iterative` as the default: **byte-for-byte unchanged when `newton` is not
  selected** (zero behavior change for existing runs). `iterative` is *not* an equivalence
  reference for `newton` (it is not the true projection) — it is only the legacy default.
  Selecting `newton` is a deliberate correctness/behavioral upgrade that changes the PGD
  trajectory; validate it end-to-end (§8), do not expect bit-equality with `iterative` runs.
- **Pre-clip is a real algorithmic choice, NOT "harmless".** `optimize()` clips
  `A = np.clip(A, 1e-8, 1−1e-8)` before projecting (`pgd_optimizer.py` ~254 and ~317). The exact
  projection does **not** commute with clipping: `max|P(Y) − P(clip(Y))|` measures **0.07–0.47** on
  realistic post-gradient inputs (~50% of entries outside [0,1]). Keeping the clip on the `newton`
  path silently defines a *third* operator (exact projection of the *clipped* candidate), which
  this plan never analyzed. **Decision: remove the pre-clip on the `newton` path** (flag-gated) and
  project the raw PGD candidate — the mathematically correct projected-gradient step. Leave the
  `iterative` path's clip untouched. Run the §8.5 smoke without the pre-clip.
- **Warm-start state:** thread only **β** (N floats) through the optimizer; α is recomputed by the
  stateless inner solve and its dimension changes across refinement levels (§3).
- Style/standards per `CLAUDE.md`.

---

## 7. Math document to create

`docs/math/08-dual-newton-projection/` (follow `AUTHORING_GUIDE.md`): the QP and its dual;
(2.1)–(2.3); the **concave partial dual `q(β)`** (2.4), its concavity, `C¹`-ness, and
`∇q = −R` via Danskin; the inner cap-free simplex solve and the well-posedness of `α(β)`; the
outer Jacobian `J = −∇²q` (3.1) with its symmetry, PSD-ness, and **structural kernel `span{1}`**
(`J·1 = 0`, `1ᵀR = 0`); global convergence of L-BFGS on the concave dual; the semismooth-Newton
polish and its BD-regularity on the **gauge-fixed quotient** `Σβ = 0` (Qi–Sun applies there, not
on the full space); the dead-cell/empty-column degeneracy; and the exactness/idempotency argument.
Build the PDF; update `docs/math/Makefile` and `docs/math/shared/references.bib`.

---

## 8. Validation (fast — no multi-day runs)

Validated by **correctness (vs the true QP) + exactness + speed + end-to-end validity**, on quick
configs (small mesh, low N), in minutes. **Do NOT validate by equivalence to
`orthogonal_projection_iterative`** — it is not the true projection ("Empirical finding").

1. **Correctness vs the true projection (the primary guard).** For a batch of random
   feasible-and-infeasible candidates `Y` — **including a crisp / near-one-hot "binding" regime**,
   not just interior inputs — at several V/N sizes and with realistic torus lumped mass `v`,
   check `orthogonal_projection_newton(Y)` two ways:
   - **KKT certificate using the solver's OWN duals** (α, β returned by the solve): stationarity
     `(U−Y)[i,k] = α_i + v_i·β_k` on box-inactive entries, the sign conditions on active entries,
     and feasibility — all ≤ 1e-9. **Do not fit the duals by least squares on inactive entries:**
     that construction false-negatives (reports KKT ≈ 0.13 on the *true* projection) on
     fully-saturated one-hot rows, which have no inactive entry to pin α_i — and such rows are
     generic late in Phase 1. (Certifying an *external* point, if ever needed, is an LP feasibility
     check, not least squares.)
   - **Agreement with a Dykstra reference projection: max abs difference ≤ 1e-8.** Dykstra
     converges only linearly — specify its stop (successive-iterate delta ≤ 1e-12; 22–314
     iterations at V=128–5280) and verify the reference's own feasibility + objective before
     trusting it as truth.

   `testing/test_newton_projection_equivalence.py` (name kept for continuity; it tests equivalence
   to the *true projection*, not to the iterative method).
2. **Exactness / idempotency:** feasibility residuals (`row_error`, `area_error`, `non_neg_error`)
   ≤ 1e-10, and `‖P_newton(U) − U‖ ≈ machine-eps` on an already-feasible `U`.
3. **Characterize the difference from `iterative` (do not null it out):** report
   `max|newton − iterative|` — **input-dependent: ~0.01–1.0, ≈ 0.1–0.2 at a realistic crisp
   operating point** — **and** show `newton` is strictly better (lower `½‖U−Y‖²`, exact
   feasibility, idempotent). This documents a correctness upgrade, not a regression.
4. **Speed — measure per refinement level.** Count outer iterations and wall time vs the iterative
   method **at each level, including a fine level (V ≈ 1.7e5) where the 86–91% wall-time motivation
   lives** — not only coarse levels (at V≈5k the prototype is only ~2× warm and can be *slower*
   cold, so a coarse-only measurement understates the win or misleads). Record the ratios. Phrase
   the DoD so a null speedup at coarse levels does **not** block merge — correctness alone
   justifies it — but the "highest-leverage speedup" headline must be *demonstrated* at fine level,
   not assumed.
5. **End-to-end validity smoke (pre-clip removed on the `newton` path, per §6).** Short low-N
   relaxations with `projection_method: newton` must produce a **valid partition** — pass the
   dormant-cell and area-imbalance gates (`detect_dormant_cells` / `detect_area_imbalance`), with
   comparable or better final perimeter — at lower wall time. Use **both** `torus_10part.yaml`
   (N=10 gates are nearly always passed, so low information) **and** the **N=30 seeded config**
   (which has a measured ~0.7% worst-cell-area baseline to compare against). Do **not** require the
   trajectory or final density to match the `iterative` run bit-for-bit; require partition
   *validity / quality* and a sane energy history.
6. **Adversarial / pathological-input tests (new file).** Exercise exactly where a naive solver
   fails: **empty cell at cold start** (must revive, not stall/explode), **fully-saturated one-hot
   rows**, **deficient rows** (row-sum < 1 inputs), and **tie/breakpoint** inputs. If the Newton
   polish is implemented, add a **finite-difference validation of `J`** against central differences
   of `R(β)` (reference check ≈ 3.6e-10 relative).

---

## 9. Definition of done

- [ ] Projection QP (Euclidean metric + constraints) extracted from `projection.py`; confirmed
      the iterative method only *approximates* it ("Empirical finding" recorded); input assertion
      `Σd = Σv` in place.
- [ ] `docs/math/08-…/main.pdf` built; concave dual `q(β)` + `∇q=−R` + Jacobian `J` (PSD,
      structural kernel `span{1}`) + L-BFGS/Newton convergence + exactness proven.
- [ ] `orthogonal_projection_newton` implemented (L-BFGS-B on the concave dual, cap-free inner
      simplex solve, optional Newton polish with gauge-fixed reduced solve + dead-cell safeguard),
      vectorized, flag-gated (default iterative, byte-identical when off).
- [ ] Correctness guard passes: KKT certificate (solver's **own** duals) ≤ 1e-9 and agreement with
      an independent Dykstra projection ≤ 1e-8, on interior **and** binding inputs; idempotency ≈
      machine-eps; feasibility ≤ 1e-10.
- [ ] Pathological inputs pass: empty-cell cold start, saturated/deficient rows, breakpoint ties;
      `J` FD-validated if the polish is implemented.
- [ ] Difference from `iterative` characterized (input-dependent, ≈ 0.1–0.2 at a crisp operating
      point), with `newton` shown strictly better (lower objective, exact feasibility, idempotent).
- [ ] Speed measured **per level incl. a fine level**; recorded. (A null coarse-level result does
      not block merge; the headline speedup is shown at fine level.)
- [ ] Low-N end-to-end smoke (`torus_10part.yaml` **and** the N=30 seeded config, pre-clip removed
      on the `newton` path): valid partitions (dormant + area-imbalance gates pass, perimeter
      comparable/better) at lower wall time — not required to match `iterative` bit-for-bit.
- [ ] `CLAUDE.md` updated (new projection option, flag, test). Merge to `main` on its own merits.

---

## 10. References

- J. Nocedal, S. J. Wright. *Numerical Optimization*, 2nd ed., Springer, 2006 — KKT for QP
  (Ch. 16), semismooth/active-set methods, finite-difference & equivalence checking.
- L. Qi, J. Sun. *A nonsmooth version of Newton's method.* Mathematical Programming 58, 1993 —
  semismooth Newton and its quadratic convergence (applies on the gauge-fixed quotient here).
- M. Hintermüller, K. Ito, K. Kunisch. *The primal-dual active set strategy as a semismooth
  Newton method.* SIAM J. Optim. 13(3), 2002 — box-constrained QP as semismooth Newton (the
  active-set structure used in the polish).
- R. H. Byrd, P. Lu, J. Nocedal, C. Zhu. *A limited memory algorithm for bound constrained
  optimization.* SIAM J. Sci. Comput. 16(5), 1995 — L-BFGS-B, the primary outer solver on the
  concave dual (available as `scipy.optimize.minimize(method="L-BFGS-B")`).
- D. P. Bertsekas. *Nonlinear Programming*, 2nd ed., Athena Scientific, 1999 — duality for
  box/equality QP, Danskin's theorem (`∇q = −R`), concavity of the partial dual.
- Projection onto the (capped) simplex with an equality: e.g. L. Condat, *Fast projection onto
  the simplex and the ℓ1 ball*, Math. Programming 158, 2016 — the inner per-vertex solve.
- J. P. Boyle, R. L. Dykstra. *A method for finding projections onto the intersection of convex
  sets in Hilbert spaces.* Lecture Notes in Statistics 37, 1986 — the reference projection used to
  validate exactness (converges to the *exact* Euclidean projection onto the intersection; used as
  ground truth in §8 in place of the non-projection incumbent).
- Internal: `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` §2.3 (the measured wall-time motivation)
  and its item P2 (future shared-dual work — *not yet implemented*),
  `docs/plans/PHASE1_N1000_SCALING_PLAN.md`, `src/optimization/projection.py` (the QP the
  iterative method *approximates*; the dual solve computes it exactly).
