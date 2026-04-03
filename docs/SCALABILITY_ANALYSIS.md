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
using `method=ipopt`, `best_iterate=True`, `exact_hessian=True`.

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

| N | VPs (est.) | Triple pts | Constraints | Exact Hessian (est.) | L-BFGS (est.) |
|---|---|---|---|---|---|
| 5 | 3,500 | 8 | 4 | **3 min** (measured) | ~1–2 min |
| 10 | 2,000* | 20 | 9 | **2 h** (measured) | ~15–30 min |
| 30 | ~5,000 | ~60 | 29 | ~1 day | ~2–4 h |
| 100 | ~10,000 | ~200 | 99 | weeks (impractical) | ~1–3 days |
| 500 | ~25,000 | ~1,000 | 499 | impractical | ~weeks |
| 1,000 | ~50,000 | ~2,000 | 999 | impractical | impractical |

*The 10-partition run used a coarser mesh than the 5-partition run; VPs would
be higher on the same mesh.

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

**Impact**: eliminates Hessian computation entirely.  Per-iteration cost drops
from ~15–25 s to ~2–5 s for the 10-partition case.  Trades quadratic
convergence rate for superlinear, but total wall-clock time is expected to be
3–5× shorter for N ≥ 10.

**Trade-off**: may need more iterations to reach the same tight tolerance.
Final perimeter is expected to be identical for practical purposes.

**When to use**: N ≥ 10, or whenever per-iteration cost exceeds ~5 s.

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
T blocks).

**Status**: planned in `SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md`.

#### 2B. Sparse Jacobian assembly

Replace the current dense-then-extract Jacobian path with direct sparse
assembly.  This avoids allocating an (N−1) × n_vp dense matrix at every
callback.

**Impact**: reduces memory from O(N · n_vp) to O(nnz_jac) and avoids
unnecessary zero-filling.  Most impactful for N > 50.

**Status**: planned in `SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md`.

#### 2C. Adaptive Hessian strategy

Use L-BFGS for the early iterations (when far from the optimum and exact
curvature is wasted) and switch to exact Hessian for the final polish:

```python
if iter_count < warmup_iters:
    use L-BFGS
else:
    switch to exact Hessian
```

This requires restarting IPOPT mid-solve (or using IPOPT's warm-start
mechanism), but combines the best of both approaches.

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
| 8–30 | Switch to L-BFGS (Tier 1A). Consider looser tolerances (1B). |
| 30–100 | L-BFGS + analytical Steiner Hessian (2A) + coarse-to-fine warm start (1C). |
| 100–500 | Augmented Lagrangian decomposition (3A) or multigrid (3B). |
| 500+ | Lloyd initialiser (3C) + local IPOPT polish, or curve-shortening flow (3D). |

---

## Action items

| Priority | Item | Effort | Impact |
|---|---|---|---|
| **Immediate** | Benchmark L-BFGS vs exact Hessian on the 10-partition problem | 1 hour | Validates Tier 1A estimate |
| **Short-term** | Implement analytical Steiner Hessian (Tier 2A) | 1–2 weeks | Removes dominant per-iter cost |
| **Short-term** | Experiment with `acceptable_tol` tuning (Tier 1B) | 1 hour | May cut 10-partition time by 50%+ |
| **Medium-term** | Direct sparse Jacobian (Tier 2B) | 1 week | Needed for N > 50 |
| **Medium-term** | Augmented Lagrangian prototype (Tier 3A) | 2–4 weeks | Enables N = 100+ |
| **Long-term** | Lloyd CVT initialiser (Tier 3C) | 2–4 weeks | Unlocks N = 1,000+ |
| **Research** | Curve-shortening flow (Tier 3D) | months | Theoretically optimal |

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
