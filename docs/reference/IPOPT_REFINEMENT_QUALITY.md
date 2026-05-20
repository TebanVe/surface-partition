# IPOPT Refinement Quality: Diagnosis and Remediation Options

## 1. Problem Statement

After the initial IPOPT integration,
empirical testing on the 5-partition torus problem reveals that IPOPT delivers a
**~130× speed-up per optimization cycle** but **fails to match SLSQP's solution
quality in later iterations**. The perimeter gap widens over successive
optimize → migrate cycles, and the resulting boundaries are visibly less smooth.

## 2. Empirical Evidence

### 2.1 Per-iteration perimeter comparison

All runs use the same 5-partition torus (46 080 vertices, ~1 977 VPs, 8 triple
points) starting from identical initial conditions (perimeter = 40.6376).

| Iteration | SLSQP final | IPOPT final | Gap    | SLSQP time | IPOPT time |
|-----------|-------------|-------------|--------|------------|------------|
| 1         | 37.4355     | 37.4394     | 0.004  | 4 h 13 m   | 2 min      |
| 2         | 37.4210     | 37.4399     | 0.019  | 4 h 43 m   | 52 s       |
| 4         | 37.4138     | 37.4371     | 0.023  | 1 h 33 m   | 52 s       |

SLSQP steadily reduces the perimeter across iterations (37.44 → 37.42 → 37.41).
IPOPT stalls around 37.44, barely improving after the first cycle.

### 2.2 Migration count asymmetry

| Iteration | SLSQP migrations | IPOPT migrations |
|-----------|------------------|------------------|
| 1         | 66 (62 T1 + 4 T2) | 6 (3 T1 + 3 T2) |
| 2         | 66 (62 T1 + 4 T2) | 4 (1 T1 + 3 T2) |
| 4         | 5 (2 T1 + 3 T2)   | 3 (0 T1 + 3 T2) |

SLSQP pushes VPs to edge endpoints (min_dist ≈ 0.0000), triggering many Type 1
migrations that enable large topology changes. IPOPT leaves VPs short of the
boundary, so most Type 1 triggers never fire.

### 2.3 Non-monotone IPOPT convergence (iteration 4 trace)

```
Iter 340: 37.4380       ← good progress
Iter 500: 37.4376       ← better than final result
Iter 560: 37.4369       ← BEST VALUE FOUND
Iter 610: 37.4368       ← absolute minimum touched
Iter 620: 37.4377       ← restoration phase, jumps UP
Iter 700: 37.4393       ← significantly worse
  ... slow recovery ...
Iter 880: 37.4367       ← almost recovers
Iter 900: 37.4372       ← bounces again
Iter 1000: 37.4371      ← RETURNED VALUE (0.0003 worse than best)
```

IPOPT repeatedly finds good values but cannot hold them due to restoration
phases. It returns the **last** iterate, not the best, so accumulated losses
compound across outer iterations.

### 2.4 Visual quality

SLSQP produces smooth partition boundaries. IPOPT boundaries appear jagged and
irregular on the 46 080-vertex test problem, particularly along long boundary
arcs where neighboring VPs should move in a coordinated fashion.

### 2.5 Contrasting observation: smooth IPOPT boundaries on a finer mesh

A separate run on a 7-partition problem with **>100 000 mesh vertices** (more
than 2× the VP count of the 5-cell test case) produced **visually smooth
boundaries** with IPOPT, despite a similar `max_iter`-exceeded exit status and
few migration triggers. This is a significant counterpoint to the 46k-vertex
results above and suggests the quality issue is not intrinsic to IPOPT but is
instead **mesh-resolution-dependent** (see Section 3.5).

## 3. Root Cause Analysis

### 3.1 L-BFGS Hessian approximation is too crude (at coarse resolution)

The Phase 1 integration uses `hessian_approximation='limited-memory'`, which
builds a low-rank (typically rank 6) approximation of the Hessian. On the
46 080-vertex problem with ~2 000 VPs, the boundary arcs are relatively coarse —
each VP controls a long segment — so neighboring VPs are strongly coupled over
long distances. The rank-6 approximation misses this long-range coupling:

- **Cannot capture curvature coupling** between adjacent VPs — each VP is
  effectively optimized semi-independently, producing jagged boundaries.
- **Causes overshooting** — without accurate curvature, IPOPT takes steps that
  are too aggressive, triggering restoration phases.
- **Prevents fine polishing** — the late-stage micro-adjustments that smooth
  boundaries require coordinated VP movement that L-BFGS cannot orchestrate.

SLSQP uses a dense BFGS update that fully models the coupling between all
variables. At ~2 000 VPs this remains tractable and gives far superior curvature
information.

### 3.2 Restoration-phase losses are not recovered

IPOPT's interior-point method periodically enters a **restoration phase** when
the current iterate drifts too close to the feasible region boundary. During
restoration, IPOPT prioritizes constraint satisfaction over objective reduction,
often jumping to a significantly worse objective value. Recovery from these jumps
consumes hundreds of iterations that would otherwise be spent refining the
solution.

### 3.3 Last-iterate return policy

When `max_iter` is hit, IPOPT returns the iterate at termination — not the best
objective value encountered during the solve. In the iteration 4 trace above,
IPOPT touched 37.4368 but returned 37.4371. This small loss (0.0003) accumulates
over multiple outer iterations.

### 3.4 Scale-dependence of the L-BFGS quality issue

The contrasting observation from Section 2.5 — IPOPT producing smooth results on
the 100k+-vertex / 7-cell problem — points to an important nuance: **the L-BFGS
approximation quality is mesh-resolution-dependent**.

On a finer mesh:
- Each VP controls a **shorter boundary segment**, so the relevant curvature
  coupling between two neighbors is more local and more easily captured by the
  rank-6 approximation.
- **Required step sizes are smaller** in absolute terms (VP positions change by
  smaller fractions of the edge length), so overshooting is less severe and
  restoration phases are triggered less frequently.
- **Better problem conditioning** — more triangles per boundary segment leads to
  a more well-conditioned area Jacobian, resulting in more stable Newton steps.

At 46 080 vertices with ~2 000 VPs the per-VP boundary segment is long enough
that long-range VP coupling dominates and the rank-6 approximation fails.
At 100 000+ vertices with proportionally more VPs, the coupling is sufficiently
local that L-BFGS captures the essential curvature and boundary smoothness is
preserved.

This means the quality issue observed in the 46k benchmark may **not manifest
in production runs at target mesh resolutions**, and Phase 4 (exact Hessian) may
only be necessary if high-quality results are also required at coarse resolutions.

### 3.5 Insufficient VP saturation limits topology exploration

Because IPOPT doesn't push VPs as close to edge endpoints as SLSQP, fewer
Type 1 migrations are triggered. This limits the topology search: SLSQP's
aggressive endpoint saturation (62 Type 1 triggers in iteration 1 vs. IPOPT's 3)
explores a much larger set of topological configurations, finding lower-energy
arrangements that IPOPT never discovers.

## 4. Remediation Options

### Option A: Increase L-BFGS memory (quick win, moderate impact)

**Change:** Set `limited_memory_max_history` to 20–50 (default is 6).

```python
problem.add_option('limited_memory_max_history', 30)
```

**Rationale:** A higher-rank Hessian approximation captures more curvature
information, reducing oscillations and improving coordination between neighboring
VPs. Memory cost is negligible at this problem size (30 × 2000 × 8 bytes ≈ 0.5 MB).

**Expected impact:** Moderate — reduces oscillation amplitude and improves
boundary smoothness, but still cannot match a full Hessian. Worth trying first
as it requires a single line change.

### Option B: Track and return the best iterate

**Change:** Record the best (objective, x) pair seen during the solve via the
`intermediate()` callback, and substitute it into the `OptimizeResult` when
IPOPT terminates with `max_iter_exceeded`.

```python
def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, ...):
    if obj_value < self._best_obj and inf_pr < 1e-6:
        self._best_obj = obj_value
        self._best_x = self._opt._get_current_lambdas()  # snapshot
    return True
```

**Rationale:** Eliminates the compounding loss from restoration-phase regressions.
In the iteration 4 example, this would recover 0.0003 per cycle.

**Expected impact:** Small but consistent — prevents quality regression across
outer iterations. Very low implementation cost.

**Caveat:** The `intermediate()` callback does not receive `x` directly.
Capturing the current iterate requires either (a) storing it from the last
`objective()` call, or (b) using IPOPT's `output_file` and parsing it. Option
(a) is simpler and recommended.

### Option C: Hybrid solver strategy (IPOPT → SLSQP)

**Change:** Use IPOPT for the first N outer iterations (rapid convergence from
the initial state), then switch to SLSQP for polishing.

```bash
# Phase 1: rapid descent with IPOPT (iterations 1-2)
python testing/refine_perimeter_iterative.py \
  --solution base.h5 --method ipopt --max-iterations 2 \
  --max-opt-iter 2000 --allow-partial-convergence

# Phase 2: polish with SLSQP (remaining iterations)
python testing/refine_perimeter_iterative.py \
  --solution iteration2_checkpoint.h5 --method SLSQP --max-iterations 5
```

**Rationale:** Exploits the complementary strengths of both solvers:
- IPOPT: fast initial convergence (40.64 → 37.44 in 2 minutes)
- SLSQP: fine polishing and smooth boundaries (37.44 → 37.41 with full
  curvature information)

**Expected impact:** High — captures the speed advantage of IPOPT where it
matters most (early iterations) and the quality of SLSQP where it matters most
(later iterations). Requires no code changes, only a workflow adjustment.

This could also be automated with a `--method auto` flag that switches from
IPOPT to SLSQP when the per-iteration perimeter improvement drops below a
threshold.

### Option D: Sparse Jacobian (implemented)

**Change:** Replace the dense Jacobian computation in `IPOPTProblemAdapter.jacobian()`
with a direct sparse computation.  **This has since been implemented**, together
with the exact Hessian — see `docs/plans/EXACT_HESSIAN_AND_ANALYTICAL_STEINER_PLAN.md`
§1.1.

**Rationale:** While this doesn't directly fix the Hessian approximation issue,
it removes the 50 MB dense matrix warning and enables scaling to larger problems.
It was a prerequisite for the exact Hessian.

**Expected impact on quality:** Minimal — the Jacobian is already correct; this
is a performance optimization. However, it unblocks Phase 4.

### Option E: Implement Phase 4 — exact Hessian (high effort, high impact)

**Change:** Provide IPOPT with an exact (or analytical) Hessian of the Lagrangian
via the `hessian()` and `hessianstructure()` callbacks, removing the need for
L-BFGS approximation entirely.

The Lagrangian Hessian is:

    H(x, σ, λ) = σ · ∇²f(x) + Σᵢ λᵢ · ∇²cᵢ(x)

where f is the perimeter objective and cᵢ are the area constraints. Both
∇²f and ∇²cᵢ can be derived analytically from the vectorized evaluation
pipeline.

**Rationale:** This is the definitive fix. With exact curvature information,
IPOPT gains the same advantage that makes SLSQP effective — full knowledge of
how VPs couple — while retaining its superior scaling and interior-point
convergence properties.

**Expected impact:** High — should match or exceed SLSQP quality while
maintaining IPOPT's per-iteration speed advantage. The Hessian is sparse (same
sparsity pattern as the Jacobian, roughly), so computation cost is manageable.

**Complexity:** Significant. Requires deriving and implementing second derivatives
of both the perimeter and area functions with respect to the λ parameters. The
perimeter Hessian involves second derivatives of edge lengths; the area Hessian
involves second derivatives of sub-triangle areas with respect to VP positions
along edges.

### Option F: Tune IPOPT barrier and step parameters

**Change:** Adjust IPOPT options to reduce restoration-phase frequency:

```python
problem.add_option('mu_strategy', 'monotone')       # less aggressive barrier
problem.add_option('mu_init', 1e-2)                  # start with smaller barrier
problem.add_option('alpha_for_y', 'min')             # conservative dual updates
problem.add_option('recalc_y', 'yes')                # recompute multipliers
problem.add_option('max_resto_iter', 50)             # limit restoration iterations
```

**Rationale:** The default `adaptive` mu strategy can be overly aggressive for
this problem's geometry, triggering unnecessary restoration phases. A monotone
strategy with conservative step acceptance may reduce oscillation.

**Expected impact:** Uncertain — highly problem-dependent. Worth experimenting
with alongside Option A.

## 5. Recommended Strategy

The mesh-resolution dependence (Section 3.4) is the most important factor
shaping the recommendations. IPOPT appears adequate for production-scale meshes
(>100k vertices); the quality issues are most pronounced on coarser test problems.

A phased approach, ordered by effort-to-impact ratio:

1. **Immediate (Options A + B):** Increase L-BFGS memory to 30 and implement
   best-iterate tracking. These are small code changes that should reduce
   oscillation and prevent regression at all resolutions. ~1 hour of work.

2. **Validate at scale:** Run the full optimize → migrate loop on the 100k+
   vertex problem for 5–10 outer iterations with IPOPT only. If boundaries
   remain smooth and perimeter continues improving, the quality issue may be
   limited to the coarse-mesh test case and options C–E can be deferred.

3. **Short-term if scale validation shows quality issues (Option C):** Adopt
   the hybrid IPOPT → SLSQP workflow for production runs. This requires no
   code changes and immediately captures the best of both solvers: IPOPT for
   the rapid initial descent, SLSQP for fine polishing.

4. **Medium-term (Option D → E):** Implement sparse Jacobian (Phase 2), then
   exact Hessian (Phase 4). This is the definitive long-term solution and
   makes IPOPT quality-competitive with SLSQP at all mesh resolutions while
   preserving its speed advantage. Priority rises if coarse-resolution
   experiments or multi-resolution workflows are required.

5. **Exploratory (Option F):** Experiment with barrier parameter tuning in
   parallel with the above. Results may inform whether the exact Hessian is
   truly necessary or whether a well-tuned L-BFGS is sufficient at all scales.

## 6. Reference Logs

| Log file | Date | Solver | Description |
|---|---|---|---|
| `ring_partition_20260323_221825.log` | Mar 23 | SLSQP | Iteration 1 from base |
| `ring_partition_20260324_115108.log` | Mar 24 | SLSQP | Iteration 2 (migrate + optimize) |
| `ring_partition_20260324_183200.log` | Mar 24 | SLSQP | Iteration 4 (migrate + optimize) |
| `ring_partition_20260330_113634.log` | Mar 30 | IPOPT | Iteration 1 from base |
| `ring_partition_20260330_112215.log` | Mar 30 | IPOPT | Iteration 2 (resume + optimize) |
| `ring_partition_20260330_113124.log` | Mar 30 | IPOPT | Iteration 4 (resume + optimize) |
