# Scalability Analysis: Perimeter Optimization with IPOPT

## Problem statement

The perimeter refinement step — minimising total partition boundary length
subject to equal-area constraints via IPOPT — becomes the computational
bottleneck as the number of partition cells N grows.  Empirical evidence shows
that the cost per iteration, the number of iterations to convergence, and the
overall problem conditioning all degrade faster than linearly with N, even when
the number of variable points (VPs) is comparable or smaller.

This document records the observed behaviour, analyses the underlying causes,
and catalogues mitigation strategies ranging from near-term parameter tuning to
longer-term algorithmic redesigns.

---

## Empirical evidence

All runs below were executed on the same torus surface (total area ≈ 23.68)
using `method=ipopt`, `best_iterate=True`.  Runs 1 and 2 use
`exact_hessian=True`; Run 3 uses L-BFGS (`exact_hessian=False`).

### Run 1 — 5 partitions, fresh from base solution

| Property | Value |
|---|---|
| Log | `logs/ring_partition_20260331_122948.log` |
| Mesh vertices | 141,120 |
| Active VPs | 3,462 |
| Triple points | 8 |
| Area constraints | 4 |
| Jacobian nnz / density | 5,698 / 41.1% |
| Hessian nnz (lower tri) | 6,936 |
| Initial constraint violation | 7.56e-03 |
| Perimeter (start → end) | 40.68 → 37.43 (8.0%) |
| Iterations to convergence | ~55 |
| Per-iteration time | ~2–5 s |
| **Total optimisation time** | **192 s (≈ 3 min)** |

### Run 2 — 10 partitions, fresh from base solution

| Property | Value |
|---|---|
| Log | `logs/ring_partition_20260402_181839.log` |
| Mesh vertices | 17,608 |
| Active VPs | 1,942 |
| Triple points | 20 |
| Area constraints | 9 |
| Jacobian nnz / density | 3,567 / 20.4% |
| Hessian nnz (lower tri) | 3,914 |
| Initial constraint violation | 1.20e-02 |
| Perimeter (start → end) | 70.12 → 58.60 (16.4%) |
| Iterations to convergence | ~525 (acceptable tol) |
| Per-iteration time | ~10–27 s (early), ~70 s (mid) |
| **Total optimisation time** | **7,245 s (≈ 2 h)** |

### Key observation

The 10-partition problem has **fewer VPs** yet takes **~38× longer** than the
5-partition problem.  The bottleneck is not the number of optimisation
variables; it is the problem structure imposed by more partitions.

### Run 3 — 10 partitions, L-BFGS (no exact Hessian), same base solution as Run 2

| Property | Value |
|---|---|
| Log | `logs/ring_partition_20260402_222701.log` |
| Mesh vertices | 17,608 |
| Active VPs | 1,942 |
| Triple points | 20 |
| Area constraints | 9 |
| L-BFGS memory | 20 |
| `exact_hessian` | **False** |
| Initial constraint violation | 1.20e-02 |
| Perimeter (start → end) | 70.12 → 58.61 (16.4%) |
| Iterations | 1000 (hit max_iter) |
| Per-iteration time | **~0.3 s (constant)** |
| **Total optimisation time** | **304 s (≈ 5 min)** |

### Head-to-head: exact Hessian vs L-BFGS (10 partitions)

| Metric | Run 2: Exact Hessian | Run 3: L-BFGS (m=20) |
|---|---|---|
| Per-iter time (iter 0–10) | 9.5 s | 0.3 s |
| Per-iter time (iter 200–300) | 4.3 s | 0.3 s |
| Per-iter time (iter 400+) | 1.2 s | 0.3 s |
| Total IPOPT iterations | ~525 | 1000 (max_iter) |
| **Total optimisation time** | **7,245 s (2.0 h)** | **304 s (5.1 min)** |
| **Wall-clock speedup** | — | **23.8×** |
| Final perimeter | 58.6012 | 58.6059 |
| Perimeter difference | — | +0.0048 (0.008%) |
| Max constraint violation | 1.08e-07 | 1.06e-06 |
| Converged? | Yes (acceptable tol) | No (max_iter reached) |
| Migrations: Type 1 | 77 | **0** |
| Migrations: Type 2 | 11 | 12 |
| **Total migrations** | **88** | **12** |

The L-BFGS run is **23.8× faster** with a perimeter difference of only
0.008%.  However, the migration asymmetry reveals a critical quality concern
discussed below.

### L-BFGS quality trade-off: migration deficit and boundary jaggedness

The L-BFGS run triggered **zero Type 1 migrations** compared to the exact
Hessian's 77.  This reproduces a known issue documented in detail in
`docs/reference/IPOPT_REFINEMENT_QUALITY.md` (Sections 2.2, 3.5):

- **Insufficient VP saturation.** The exact Hessian pushes VPs to triangle
  edge endpoints (min_dist ≈ 0.0000), triggering Type 1 migrations that
  enable large topology changes.  L-BFGS lacks the curvature information to
  coordinate neighbouring VPs to this degree, so VPs stop short of the
  boundary and Type 1 triggers never fire.

- **Jagged boundaries.** Without exact second-order coupling between
  neighbouring VPs, L-BFGS moves each VP semi-independently.  On coarser
  meshes (where each VP controls a long boundary segment), this produces
  visibly jagged partition boundaries.  The issue is mesh-resolution-dependent:
  on finer meshes (>100k vertices) the coupling becomes more local and L-BFGS
  produces smoother results (see `docs/reference/IPOPT_REFINEMENT_QUALITY.md` Section 3.4).

- **Oscillatory convergence.**  The L-BFGS objective trace shows non-monotone
  behaviour (e.g., obj bounces between 58.606 and 58.614 around iterations
  480–640), whereas the exact Hessian monotonically converges after the initial
  descent.  This is consistent with restoration-phase losses amplified by the
  crude curvature approximation.

- **Reduced topology exploration.**  With 88 migrations (exact) vs 12
  (L-BFGS), the exact Hessian explores a much richer set of topological
  configurations across outer iterations.  Over multiple optimize → detect →
  migrate cycles, this compounding difference can lead to significantly
  different final partition quality.

These observations confirm that L-BFGS is not a drop-in replacement for the
exact Hessian when solution quality and topology exploration matter.  It is
best understood as a **fast rough solver** whose results require either:
(a) polishing with the exact Hessian, (b) polishing with SLSQP, or
(c) running on sufficiently fine meshes where the quality gap narrows.

---

## Root-cause analysis

### 1. Finite-difference Steiner Hessian: O(T) per iteration

The current exact Hessian implementation computes Steiner-point contributions
via finite differences (`compute_steiner_perimeter_hessian_fd`,
`compute_steiner_area_hessian_fd` in `vectorized_steiner.py`).  Each triple
point requires multiple function evaluations per Hessian call.

| N (partitions) | Triple points T | Relative FD cost |
|---|---|---|
| 5 | 8 | 1× |
| 10 | 20 | 2.5× |
| 100 | ~200 | 25× |
| 1,000 | ~2,000 | 250× |

For an optimal partition on a genus-1 surface, the Euler relation for the
cell complex gives T ≈ 2N triple points (vertices), E ≈ 3N boundary arcs
(edges), and the growth is exactly linear in N.

### 2. KKT system conditioning

IPOPT solves the augmented KKT system at each iteration:

```
[H + Σ + D   Jᵀ ] [Δx]   [rₓ ]
[J          −δI ] [Δλ] = [r_λ]
```

where H is the Hessian of the Lagrangian, Σ is a regularisation term, D is the
barrier diagonal, and J is the (N−1) × n_vp constraint Jacobian.  As N grows:

- The Schur complement S = J (H + Σ + D)⁻¹ Jᵀ is a dense (N−1) × (N−1) block.
  Factoring it costs O(N³).  At N = 1,000, this alone is ≈ 10⁹ operations per
  iteration.
- Each triple-point VP appears in up to 3 constraint rows simultaneously,
  increasing the coupling and worsening the condition number.
- The barrier perturbation interacts with the constraint coupling, producing
  ill-conditioned iterates that trigger restoration phases and smaller step
  sizes.

### 3. Number of iterations

With more constraints and worse conditioning, IPOPT needs more iterations:

| N | Observed / estimated iters | Phase behaviour |
|---|---|---|
| 5 | ~55 | Clean quadratic convergence |
| 10 | ~525 | Long plateau (iter 200–520 with obj stalled) |
| 100 | ~1,000+ (est.) | Extended restoration phases expected |
| 1,000 | likely does not converge | — |

The 10-partition log shows IPOPT reaching obj = 58.601162 by iteration 60, then
spending 460 more iterations trying to satisfy constraint feasibility to tight
tolerance — a clear sign of poor conditioning in the constraint subspace.

### 4. Per-iteration cost summary

| Component | Cost scaling (exact Hessian) | Cost scaling (L-BFGS) |
|---|---|---|
| Objective + gradient | O(n_vp + T) | O(n_vp + T) |
| Constraint + Jacobian | O(n_vp · N) | O(n_vp · N) |
| Hessian computation | O(n_vp + T · fd_evals) | **0** (built internally) |
| KKT factorisation | O(n_vp · bw² + N³) | O(n_vp · m_lbfgs) |
| **Total per iteration** | **O(n_vp · bw² + T · fd + N³)** | **O(n_vp · N)** |

where bw = bandwidth of the reordered Hessian, fd = FD evals per triple point,
m_lbfgs = L-BFGS memory size (typically 6–50).

---

## Projected scaling

Rough estimates for a torus mesh fine enough to resolve boundaries
(n_vp ∝ √N for fixed surface area, assuming mesh refinement follows boundary
density).  Times assume the same hardware as the empirical runs above.

| N | VPs (est.) | Triple pts | Constraints | Exact Hessian (est.) | L-BFGS (est.)† |
|---|---|---|---|---|---|
| 5 | 3,500 | 8 | 4 | **3 min** (measured) | ~1–2 min |
| 10 | 2,000* | 20 | 9 | **2 h** (measured) | **5 min** (measured) |
| 30 | ~5,000 | ~60 | 29 | ~1 day | ~30 min–1 h |
| 100 | ~10,000 | ~200 | 99 | weeks (impractical) | ~4–12 h |
| 500 | ~25,000 | ~1,000 | 499 | impractical | ~days |
| 1,000 | ~50,000 | ~2,000 | 999 | impractical | ~weeks |

*The 10-partition run used a coarser mesh than the 5-partition run; VPs would
be higher on the same mesh.

†L-BFGS times represent optimisation wall time only.  Quality caveats apply:
L-BFGS may require a subsequent exact Hessian or SLSQP polishing pass to
achieve equivalent boundary smoothness and trigger necessary topology
migrations.  The combined time (L-BFGS + polish) should be used when
comparing against pure exact Hessian runs for quality-equivalent results.

---

## Mitigation strategies

Strategies are ordered by implementation effort (low → high) and expected
impact.

### Tier 1 — Parameter tuning and existing code paths (no code changes)

#### 1A. Switch to L-BFGS approximation

**Already implemented.**  Drop the `--exact-hessian` flag:

```bash
python testing/refine_perimeter_iterative.py \
  --method ipopt --lbfgs-memory 20 --best-iterate \
  --solution <file.h5> --max-iterations 1
```

**Measured impact (10-partition benchmark):** per-iteration cost dropped from
1.2–9.5 s to 0.3 s; total wall time dropped from 2 h to 5 min (**23.8×
speedup**).  The initial scalability estimates (3–5× speedup, 15–30 min) were
conservative; the FD Steiner Hessian dominated per-iteration cost more than
expected.

**Quality trade-off — significant.** The benchmark revealed that L-BFGS
triggered **zero Type 1 migrations** versus 77 for the exact Hessian (see
the head-to-head comparison above and `docs/reference/IPOPT_REFINEMENT_QUALITY.md`).
L-BFGS does not push VPs to triangle edge endpoints, so the topology
exploration that Type 1 migrations enable is effectively disabled.  On
coarser meshes the boundaries also appear jagged due to the lack of
curvature coupling between neighbouring VPs.

**When to use**: as a **fast initial solver** for rapid descent from contour
extraction, followed by an exact Hessian or SLSQP polishing pass if
boundary quality and topology exploration matter.  Also suitable as a
standalone solver on fine meshes (>100k vertices) where the quality gap
narrows (see `docs/reference/IPOPT_REFINEMENT_QUALITY.md` Section 3.4).

**Not recommended** as the sole solver when topology migrations are important
or when visual boundary smoothness is required at coarse mesh resolutions.

#### 1B. Loosen acceptable tolerance

IPOPT's `acceptable_tol` and `acceptable_iter` control early termination.  For
the 10-partition run, the objective converged by iteration 60 but feasibility
took 460 more iterations.  Relaxing tolerances avoids this tail:

```python
problem.add_option('acceptable_tol', 1e-4)      # currently tol * 100
problem.add_option('acceptable_iter', 5)         # stop after 5 acceptable
problem.add_option('acceptable_constr_viol_tol', 1e-6)
```

**Impact**: can cut iteration count by 50–80% when the solver is in a
feasibility-polishing phase that does not improve the objective.

#### 1C. Warm-start from coarser mesh or previous solution

When resuming from a checkpoint (`--solution <iteration_N.h5>`), IPOPT starts
near the optimum and converges in very few iterations.  A deliberate two-phase
strategy:

1. Solve on a coarse mesh (few VPs) to get a good starting topology.
2. Interpolate to the fine mesh and re-solve (fast convergence).

This is already partially supported by the pipeline but could be made
systematic.

### Tier 2 — Targeted code improvements (moderate effort)

#### 2A. Analytical Steiner Hessian

Replace the finite-difference Steiner Hessian (`compute_steiner_*_hessian_fd`)
with closed-form derivatives.  The Steiner tree in each triple-point triangle
is a function of 3 VP λ-parameters; the analytical Hessian is a 3×3 block
that can be derived explicitly.

**Impact**: eliminates FD overhead entirely.  Makes the exact Hessian cost
independent of the number of triple points (up to the O(T) cost of evaluating
T blocks).  Empirically removes the O(T) per-iteration cost that scales
with N.

**Status**: **planned in detail** as Phases 2–3 of
`docs/plans/EXACT_HESSIAN_AND_ANALYTICAL_STEINER_PLAN.md`.  That plan
documents the implicit-function-theorem derivation (∂S/∂p_k = M⁻¹ K_k and
the chain rule to ∂²S/∂p_k∂p_l), analytical first derivatives (Phase 2)
and second derivatives (Phase 3), degenerate-case handling, and a
validation harness tightening the exact-Hessian-vs-FD tolerance to ~1e-10.

#### 2B. Sparse Jacobian assembly + vectorised Hessian accumulation

Replace the current dense-then-extract Jacobian path with direct sparse
assembly.  This avoids allocating an (N−1) × n_vp dense matrix at every
callback.  Vectorise the Python-level accumulation into the sparse Hessian
`values` array using `np.add.at` and pre-computed offset arrays.

**Impact**: reduces memory from O(N · n_vp) to O(nnz_jac) and avoids
unnecessary zero-filling.  The sparse-Jacobian piece is already in the
codebase; the vectorised Hessian accumulation is the remaining win and is
most impactful for mid-to-large problems (many segments → many dict
lookups in the current Python loop).

**Status**:
- Sparse Jacobian — **implemented**.
- Vectorised Hessian accumulation (`np.add.at` on pre-computed offset
  arrays) — **implemented**.
- Validation harness — **planned in detail** as Phase 1 of
  `docs/plans/EXACT_HESSIAN_AND_ANALYTICAL_STEINER_PLAN.md`.  That plan
  specifies `testing/compare_hessian_modes.py` with a per-component
  profile breakdown (perimeter-H / area-H / Steiner-H / IPOPT
  linear-solve time) — the primary instrument for deciding when Tier 2
  has hit its ceiling and Tier 3 is required.

#### 2C. Hybrid solver strategy (L-BFGS → exact Hessian, or L-BFGS → SLSQP)

The benchmark confirms that L-BFGS and exact Hessian have complementary
strengths.  A two-phase workflow exploits both:

1. **Phase 1 (L-BFGS):** rapid descent from contour extraction.  At 0.3 s
   per iteration, 1000–2000 iterations take 5–10 minutes and capture >95%
   of the perimeter reduction.
2. **Phase 2 (exact Hessian or SLSQP):** polishing pass starting from the
   L-BFGS solution.  Pushes VPs to edge endpoints (triggering Type 1
   migrations), smooths boundaries, and tightens constraint feasibility.

This is already possible with the current CLI:

```bash
# Phase 1: fast L-BFGS descent (5 min)
python testing/refine_perimeter_iterative.py \
  --solution base.h5 --method ipopt --lbfgs-memory 20 \
  --best-iterate --max-opt-iter 2000 --max-iterations 1 \
  --allow-partial-convergence

# Phase 2: exact Hessian polish (from checkpoint)
python testing/refine_perimeter_iterative.py \
  --solution iteration1_refined_contours.h5 --method ipopt \
  --exact-hessian --best-iterate --max-iterations 1
```

For the 10-partition problem, Phase 1 (5 min) + Phase 2 (starting near the
optimum, expect ~10–30 min) would likely total ~15–35 min — significantly
faster than pure exact Hessian (2 h) while preserving quality.

A `--method auto` or `--hybrid` flag could automate this in a future
refactoring pass.  See also `docs/reference/IPOPT_REFINEMENT_QUALITY.md` Option C for
the analogous IPOPT → SLSQP hybrid strategy.

### Tier 3 — Algorithmic redesigns (significant effort)

#### 3A. Augmented Lagrangian / ADMM decomposition

Instead of solving one monolithic NLP, reformulate as:

```
min  Σ_k  perimeter(cell_k)
s.t. area(cell_k) = A/N   for each k
```

using an augmented Lagrangian:

```
L_ρ = Σ_k [ perimeter(cell_k) + λ_k (area_k - A/N) + (ρ/2)(area_k - A/N)² ]
```

Each subproblem (minimising L_ρ over the boundary VPs of cell k) involves only
the VPs on cell k's boundary.  The multipliers λ_k and penalty ρ are updated
in an outer loop.

**Impact**: per-iteration cost becomes O(N × n_vp_per_cell) instead of
O(n_vp_total × N).  Subproblems are independent and can be parallelised across
cores.

**Trade-off**: convergence of the outer loop can be slow; requires careful
tuning of ρ schedule.  Coupling between adjacent cells (shared boundary
segments and Steiner points) must be handled at the interface.

**Suitability**: N = 50–500.

#### 3B. Multigrid / hierarchical optimisation

Solve the partition problem at multiple scales:

1. Start with a coarse partition (e.g., N = 10).
2. Subdivide each cell (N = 40).
3. Re-optimise locally (only boundary VPs near the new subdivisions).
4. Repeat until target N is reached.

Each level inherits a good starting point from the previous level, so IPOPT
converges quickly at each step.

**Impact**: total work is O(N log N) instead of O(N^α) with α > 2.

**Trade-off**: requires a cell subdivision operator and local re-meshing.  Not
guaranteed to find the global optimum (but neither is the current approach).

**Suitability**: N = 100–1,000.

#### 3C. Lloyd-type iteration on surfaces (CVT approach)

Centroidal Voronoi Tessellation (CVT) algorithms generalised to surfaces:

1. Place N seed points on the surface.
2. Compute the geodesic Voronoi diagram (restricted to the mesh).
3. Move each seed to the centroid of its Voronoi cell.
4. Repeat until convergence.

Each iteration costs O(M log N) where M is the number of mesh vertices (using
a priority-queue geodesic distance computation).  The algorithm is simple,
robust, and scales to N = 10,000+ on GPU.

**Impact**: per-iteration cost is independent of the optimisation variable
count (there are none — only N seed positions).  Converges to equal-area
partitions with low perimeter.

**Trade-off**:
- Produces Voronoi-type partitions, which are geodesically convex.  This may
  not capture the true perimeter-minimising partition (which has 120° triple
  points, not Voronoi-type angles).
- Can be used as an excellent initialiser for the IPOPT-based refinement: run
  Lloyd to get a good topology, then polish with IPOPT.

**Suitability**: N = 100–10,000 (as initialiser), or standalone when Voronoi
quality is sufficient.

#### 3D. Curve-shortening flow with area constraints

Evolve the partition boundaries directly as curves on the surface:

```
∂γ/∂t = κ_g · n  −  Σ_k μ_k · n
```

where κ_g is the geodesic curvature, n is the boundary normal, and μ_k are
Lagrange multipliers enforcing area constraints.  This is a PDE approach:

- Naturally handles topology changes (curves can merge/split).
- Each time step is a local operation on each boundary segment.
- Parallelises across boundaries and across GPUs.
- Converges to 120° triple-point junctions (Plateau's laws) by construction.

**Impact**: per-time-step cost is O(total boundary length) ≈ O(√N · mesh_res).
For N = 1,000, this could be seconds per step.

**Trade-off**: requires careful numerical handling of triple-point junctions
and topology changes.  Time-step stability limits may require many steps.
Significant implementation effort.

**Suitability**: N = 500–100,000 (research frontier).

---

## Recommended strategy by partition count

| N range | Recommended approach |
|---|---|
| 1–7 | Current pipeline with exact Hessian. Fast enough (minutes). |
| 8–30 | **Hybrid: L-BFGS rapid descent → exact Hessian polish (Tier 2C).** L-BFGS alone is fast but misses Type 1 migrations and produces jagged boundaries; the polishing pass recovers quality. Looser tolerances (1B) for the L-BFGS phase. |
| 30–100 | Hybrid (2C) + analytical Steiner Hessian (2A) to make the polishing pass affordable + coarse-to-fine warm start (1C). |
| 100–500 | Augmented Lagrangian decomposition (3A) or multigrid (3B). |
| 500+ | Lloyd initialiser (3C) + local IPOPT polish, or curve-shortening flow (3D). |

---

## Action items

| Priority | Item | Effort | Impact | Status |
|---|---|---|---|---|
| ~~Immediate~~ | ~~Benchmark L-BFGS vs exact Hessian on 10-partition~~ | ~~1 hour~~ | ~~Validates Tier 1A~~ | **Done** — 23.8× speedup confirmed; quality trade-offs documented |
| **Immediate** | Test hybrid L-BFGS → exact Hessian workflow (Tier 2C) on 10-partition | 1–2 hours | Validates combined speed + quality | Pending |
| **Immediate** | Increase `max_opt_iter` for L-BFGS runs to 2000–3000 | trivial | L-BFGS did not converge in 1000 iters | Pending |
| **Short-term** | Validation harness for the exact-Hessian path — Phase 1 of `docs/plans/EXACT_HESSIAN_AND_ANALYTICAL_STEINER_PLAN.md` | 1–2 days | Adds `compare_hessian_modes.py` with per-component profile breakdown (the primary instrument for deciding when Tier 2 is exhausted and Tier 3 is needed) | Pending |
| **Short-term** | Implement analytical Steiner derivatives (Tier 2A) — Phases 2–3 of `docs/plans/EXACT_HESSIAN_AND_ANALYTICAL_STEINER_PLAN.md` | 1–2 weeks | At N ~1000 the FD Steiner Hessian is ~O(N²) per iter; analytical Steiner is O(N). Becomes a real perf win (not just accuracy) at scale. Required for the exact-Hessian path at N ≳ 100 | Pending |
| **Short-term** | Experiment with `acceptable_tol` tuning (Tier 1B) | 1 hour | May cut exact Hessian polishing time by 50%+ | Pending |
| ~~**Medium-term**~~ | ~~Direct sparse Jacobian (Tier 2B)~~ | ~~1 week~~ | ~~Needed for N > 50~~ | **Done** — implemented |
| **Medium-term** | Swap IPOPT linear solver (MUMPS → MA57 / MA97 via HSL) | 1–3 days once Tier 2 profile identifies it as the bottleneck | Attacks the dense Schur complement cost directly; `compare_hessian_modes.py` tells you when this is worthwhile | Pending |
| **Medium-term** | Augmented Lagrangian prototype (Tier 3A) | 2–4 weeks | Enables N = 100–500 by decomposing the monolithic NLP; per-iter cost becomes independent of N | Pending |
| **Long-term** | Multigrid / hierarchical coarse-to-fine optimisation (Tier 3B) | 3–6 weeks | Enables N = 100–1,000 by re-using warm starts from coarser partitions | Pending |
| **Long-term** | Lloyd CVT initialiser (Tier 3C) | 2–4 weeks | Unlocks N = 1,000+. Best used as an initialiser for a Tier 2 polishing pass, not a replacement | Pending |
| **Research** | Curve-shortening flow with area constraints (Tier 3D) | months | Theoretically optimal; scales to N = 1,000–100,000 in principle; significant implementation effort | Pending |

---

## Related documents

- **`docs/reference/IPOPT_REFINEMENT_QUALITY.md`** — Detailed analysis of
  L-BFGS quality issues (jagged boundaries, migration deficit,
  restoration-phase losses), including the mesh-resolution dependence
  that makes L-BFGS adequate on fine meshes but insufficient on coarser
  ones.  The findings in that document directly explain the migration
  asymmetry observed in the benchmark above.
- **`docs/reference/OPTIMIZATION_METHODS_PRIMER.md`** — Conceptual primer
  on SLSQP, IPOPT, L-BFGS, and the exact Hessian, assuming no prior
  optimisation background.  The non-technical companion to this
  document.
- **`docs/plans/EXACT_HESSIAN_AND_ANALYTICAL_STEINER_PLAN.md`** — The
  consolidated plan for completing the exact-Hessian path.  Phase 1 is a
  validation harness (`test_sparse_jacobian_equivalence.py`,
  `test_exact_hessian_vs_fd.py`, `compare_hessian_modes.py` with the
  per-component profile breakdown that gates Tier 2 vs Tier 3
  decisions).  Phases 2–3 replace the FD Steiner derivatives with
  closed-form analytical formulas derived via the implicit-function
  theorem from the Fermat-point optimality condition — at N ≈ 1000 an
  O(N²) → O(N) performance unlock on the Steiner Hessian, not merely an
  accuracy improvement.  The sparse Jacobian, exact Hessian, and
  vectorised `np.add.at` accumulation are already implemented; the
  mathematical derivations are in `docs/math/01-phase2-derivatives`.

---

## References

- Wächter, A. and Biegler, L.T. (2006). "On the implementation of an
  interior-point filter line-search algorithm for large-scale nonlinear
  programming." *Mathematical Programming*, 106(1), 25–57.
- Du, Q., Faber, V., and Gunzburger, M. (1999). "Centroidal Voronoi
  tessellations: applications and algorithms." *SIAM Review*, 41(4), 637–676.
- Mantegazza, C. (2011). *Lecture Notes on Mean Curvature Flow.* Progress in
  Mathematics, Birkhäuser.
- Morgan, F. (2009). "Soap bubbles in R² and in surfaces." *Pacific Journal of
  Mathematics*, 165(2), 347–361.
- Cox, S.J. and Flikkema, E. (2010). "The minimal perimeter for N confined
  deformable bubbles of equal area." *Electronic Journal of Combinatorics*,
  17, #R45.
