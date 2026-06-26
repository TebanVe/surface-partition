# Phase 1 PGD Serial Optimization Audit

This document records a read-only audit of the **serial** optimization
opportunities in the Phase 1 relaxation hot path (Projected Gradient Descent with
per-step orthogonal projection, in `src/optimization/`). It was produced before a
multi-day N=50 production relaxation, to decide what to optimize without
parallelization. It assigns each opportunity a **stable ID (#1–#13)** that other
documents reference (notably
`docs/plans/PHASE1_PGD_SERIAL_OPTIMIZATIONS_PLAN.md`, which implements #1, #4, #6).
Findings are grounded in the empirical timing breakdown
(`docs/math/04-phase1-timing-profile/`,
`docs/plans/PHASE1_RELAXATION_TIMING_PROFILE.md`) and corroborated against a live
N=50 run. Line numbers are as of the audit; locate by surrounding context if they
have drifted.

> **Naming.** This document uses **V = vertices**, **N = regions** (the framing of
> the analysis). In the *code* the convention is the opposite (`N` = vertices,
> `n` = partitions). Quoted code reads in code convention.

## Method

The audit fanned six independent readers across the hot-path subsystems
(`projection.py`, `pgd_optimizer.py` energy/gradient, backtracking/triggers,
`relaxation.py`/interpolation/schedule, `tri_mesh.py` FEM, initialization + timing
profile), adversarially verified each reader's claims against the actual code, then
synthesized and globally ranked the results. Three reader claims were corrected on
verification and the corrections are folded in below: opportunity #2 was demoted
HIGH→MEDIUM (the cap-burn premise belonged to the old tolerance regime), the
projection *warm-start* idea was reclassified as a negative finding with a
correctness pitfall (#13), and the "wrong sparse format" hypothesis was found false
(M/K are already CSR, #13).

## Empirical basis and live-run findings

- The orthogonal projection is **86–90 % of wall time** at fine levels; energy +
  gradient is the secondary bucket (~8.5 %); everything else (constraints,
  triggers, I/O, FEM assembly, init) is single-digit-percent or less.
- The projection is re-invoked **on every backtracking trial step**, and `step0` is
  hard-reset to `1.0` at the top of every PGD iteration. Cost scales as
  *(backtracks) × (projection inner-iters) × O(V·N)*.
- **Live N=50 run** (`results/run_20260625_113015_surftorus_npart50_..._seed84172851`):
  - **The projection inner-loop cap is NOT being burned.** With
    `pgd_projection_tol = 1e-9` (`parameters/torus_50part.yaml`) the relaxation log
    has **zero** `"Projection did not converge after 300 iterations"` warnings. The
    residual floor (~1–2×10⁻¹⁰) is set by the hard-coded `epsilon = 1e-10` in
    `projection.py:60`, which is below `1e-9`, so the inner loop converges in
    ~20–40 iterations. **This makes opportunity #2 a non-issue for current runs.**
  - **The backtracking pathology is real and measured.** In
    `traces/pgd_part50_..._level1_summary.out` the accepted `STEP` column decays
    from `1.0` to `3.9e-3 = 2⁻⁸` near convergence — i.e. **~8 rejected trials per
    iteration** at the tail, each a full projection + full energy, with `step0`
    re-walked from `1.0` every iteration. This is exactly what opportunity #1 fixes.
  - That run launched **without `--profile`**, so it produced no
    `timing_profile.yaml`; future runs should pass `--profile`.

## Ranked opportunities

| # | Opportunity | Location | Impact | Fix type | Status |
|---|---|---|---|---|---|
| 1 | Warm-start / adapt the initial backtracking step | `pgd_optimizer.py:294`, loop `299–322` | **HIGH** | both | Planned (Change A) |
| 2 | Make projection tolerance reachable / residual-plateau stall break | `projection.py:138,144,151–154,60` | MEDIUM* | both | Out of scope (already fixed by `tol=1e-9`) |
| 3 | Dual-Newton over closed-form per-vertex simplex projections | `projection.py:70–122` | HIGH (merit) | new-algorithm | Deferred (not pre-run; durable fix for N≥100) |
| 4 | Trim per-inner-iteration waste in `orthogonal_projection_iterative` | `projection.py:71,144,63–68,129–132,80–82,90` | MEDIUM | code-fix | Planned (Change B) |
| 5 | Batch per-region energy/gradient loops into CSR @ dense matmuls | `pgd_optimizer.py:92–98,135–140,100–115,141–164` | MEDIUM | code-fix | Optional follow-up |
| 6 | Reuse post-step gradient as next iteration's step gradient | `pgd_optimizer.py:290,331` | LOW (free, exact) | code-fix | Planned (Change C) |
| 7 | Skip dead `var_w` in gradient penalty loop (fixed-target mode) | `pgd_optimizer.py:147` | LOW | code-fix | Optional follow-up |
| 8 | Hoist invariant `‖g‖²` out of the backtracking loop | `pgd_optimizer.py:314` | LOW | code-fix | Optional (foldable into Change A) |
| 9 | Defer per-iteration bookkeeping to a summary stride | `pgd_optimizer.py:336,339–341,363,380` | LOW | code-fix | Optional follow-up |
| 10 | Stop flushing summary + HDF5 to disk every iteration | `pgd_optimizer.py:356–357` | LOW (matters on networked FS) | code-fix | Optional follow-up |
| 11 | Capture energy components from the accepted trial on save strides | `pgd_optimizer.py:349` | LOW | code-fix | Optional follow-up |
| 12 | Vectorize FEM assembly / triangle-area loop; cache lumped-mass `v` | `tri_mesh.py:71–129,38–55,147–150` | LOW (one-time per level) | code-fix | Optional follow-up |
| 13 | Negative findings — do **not** spend effort here | see below | — | — | Informational |

\* #2 was rated HIGH by the reader on the premise the 300-cap is burned every
iteration; demoted to MEDIUM on verification because the live `tol=1e-9` already
clears the `epsilon=1e-10` residual floor. It is now a cheap safety net, not a
fresh win.

### Detail

**#1 — Warm-start / adapt the initial backtracking step (HIGH).**
`step0` is hard-reset to `1.0` every PGD iteration (`pgd_optimizer.py:294`), and
each rejected backtrack runs a full `orthogonal_projection_iterative` (line 304)
plus full `compute_energy` (line 310). With ~5.4 backtracks/iter at the finest
level (~4.4 rejected), ~95 % of wall is per-trial cost × backtrack count.
- *Tier 1 (low risk, keeps Armijo exactly):* carry the last accepted step and start
  the next search from `min(step_max, last_step/backtrack_rho)`.
- *Tier 2 (new step rule):* Barzilai–Borwein `step0` from `s = x_k − x_{k−1}` and
  `y = g_k − g_{k−1}`, clamped, still Armijo-backtracked.
- *Impact:* cutting trials toward ~1.5–2 is roughly a **2× speedup**; thrash worsens
  with N. *Risk:* BB is non-monotone and needs a Grippo-type safeguard; large steps
  hit the `[1e-8, 1−1e-8]` clip. Tier 1 is the safe choice. **Tier 1 is implemented
  by the plan (Change A); Tier 2 is not.**

**#2 — Projection tolerance / residual-plateau stall break (MEDIUM, already fixed).**
The inner loop converges only on the absolute residual test (`projection.py:138`);
the floor is set by `epsilon=1e-10` (line 60). The live `tol=1e-9` already clears
it, so the cap is not burned. A relative-stall break on the scalar `max_error`
(line 126), gated on `max_error < ~10·tol`, remains a cheap safety net against the
floor rising as V grows. *Out of scope for the plan* except that Change B replaces
the `allclose` stall with exactly this scalar-residual test as a refactor.

**#3 — Dual-Newton over closed-form simplex projections (HIGH merit, deferred).**
The coupled sum-to-one + equal-area projection is already closed-form
(`projection.py:79–107`); only `A ≥ 0` forces iteration. Drop the area constraint →
each vertex row projects onto the probability simplex in closed form (O(N log N));
re-introduce the N equal-area couplings via a dual `y ∈ ℝ^N` solved with semismooth
Newton (~5–15 iters). Replaces the 17–300 clamp+Sinkhorn sweeps with ~10 Newton
steps and gives exact feasibility (removes the ε-floor/cap pathology). *Risk:* needs
a correct simplex projection and dual Jacobian (semismooth at active-set changes);
**not landable before a multi-day run** — the durable fix for N≥100. Belongs in its
own plan.

**#4 — Trim per-inner-iteration waste in the projection (MEDIUM).**
The inner loop runs ~12–14 dense V×N passes/iter. Remove the O(V·N)
`A_prev = A.copy()` (line 71) and `np.allclose` stall check (line 144) — replace
with the scalar `max_error` history; delete the dead `convergence_history` dict
(lines 63–68, 129–132; never returned — the function returns only `A` at line 170);
hoist the constant `C = ‖v‖²(I − J/n)` and its regularized solve out of the loop
(lines 80–82, 90). ~12–22 % of total wall at zero accuracy cost. *Risk:* very low;
the `C` hoist must reproduce the ε-regularized solve exactly. **Implemented by the
plan (Change B).**

**#5 — Batch per-region energy/gradient loops into CSR @ dense matmuls (MEDIUM).**
`compute_energy`/`compute_gradient` loop over the N regions issuing ~2N CSR matvecs
per evaluation. Replace with whole-matrix ops on `Phi = x.reshape(V,N)`:
`KP = K @ Phi`, `Interface = Phi²(1−Phi)²`, `MI = M @ Interface`; reductions and
`(1−2Phi)` applied after. Identical flops; removes ~200 dispatches/eval and the
strided-column copies. Bounded by the ~8.5 % energy+gradient share. *Risk:*
float reduction-order drift (~1e-12…1e-15) can shift accept/plateau decisions by a
few iterations; preserve the penalty math and apply `(1−2u)` after the M matmat.
*Optional follow-up.*

**#6 — Reuse the post-step gradient (LOW, free, exact).**
`g_post = compute_gradient(x)` (line 331) is bit-identical to the next iteration's
`g = compute_gradient(x)` (line 290): `x` is unmodified between them (lines 336–410
only read `x`). Carry `g_post` forward → halves gradient evals; saves ~1.6 % wall;
bit-identical. **Implemented by the plan (Change C).**

**#7 — Skip dead `var_w` in the gradient penalty loop (LOW).**
`var_w` (`pgd_optimizer.py:147`) is unused in the active `'fixed'` penalty mode
(the fixed branch at line 164 uses only `grad_var`/`T_eff`). Guard it behind
`if self.penalty_target_mode == 'adaptive'`. **Do not** touch the `var_w` in
`compute_energy:106` — that one is used (line 114). *Optional follow-up.*

**#8 — Hoist invariant `‖g‖²` out of the backtracking loop (LOW).**
`float(np.dot(g, g))` is recomputed every trial (line 314) though `g` is
loop-invariant; compute once before the `while`. Bit-identical. **Sits inside the
loop edited by Change A and may be folded in for free** (see the plan's Phase 1
optional sub-item).

**#9 — Defer per-iteration bookkeeping to a summary stride (LOW).**
`constraint_fun`, gnorm/cnorm/feas, and a duplicate `v@x` (lines 336, 339–341, 363,
380) are computed every iteration but feed only the summary line and the secondary
plateau checks; the energy-stability gate needs only `E` (free from the line
search). Stride these. <0.5 % wall; must preserve plateau-window history semantics.
*Optional follow-up.*

**#10 — Stop flushing to disk every iteration (LOW).**
`summary_fh.flush()` + `h5f.flush()` run every iteration (lines 356–357); the HDF5
flush fires even on the 499/500 iterations that wrote nothing. Move it inside the
save-stride gate, keep a final flush. ~0 % locally; real only on a high-latency
networked filesystem (e.g. Pelle `/proj`). *Optional follow-up.*

**#11 — Capture energy components from the accepted trial on save strides (LOW).**
`compute_energy(x, return_components=True)` (line 349) re-evaluates a full energy
pass at the already-evaluated accepted point, only every `h5_save_stride` (500)
iterations. Negligible; listed for completeness. *Optional follow-up.*

**#12 — Vectorize FEM assembly / triangle-area loop; cache `v` (LOW).**
`compute_matrices` (`tri_mesh.py:71–129`) assembles M/K via a Python triangle loop
into `lil_matrix` (slowest scipy build); `_compute_triangle_areas` (38–55) loops per
face; the `v` property (147–150) recomputes `M.sum(axis=0)` per access. Vectorize to
COO triplets / batched cross products, memoize `v`. All run **once per level**
(~5–6×), fully amortized — minutes against a multi-day run. Storage is **already
CSR** — no per-matvec win here. *Optional follow-up.*

**#13 — Negative findings (do not pursue).**
- *Projection warm-starting across PGD steps:* **no valid serial win.** The affine
  multipliers are solved exactly each inner iteration (no Dykstra dual state to
  reuse — saves only the negligible O(N³) solve), and the projection's fixed point
  is **start-dependent**: seeding the primal `A` from the previous projected point
  lands on a *different* feasible point and **silently corrupts every PGD step**. A
  warm-startable quantity only exists after the #3 reformulation.
- *Sparse matrix format:* M and K are **already CSR** (`tri_mesh.py:128–129`); all
  hot-path matvecs run on CSR — no re-conversion, no `SparseEfficiencyWarning`, no
  densification. The "wrong format" win does not exist.
- *Seeded initialization* (`initialization.py`): runs once at level 0, sub-second,
  ~0.001–0.01 % of work. Leave it.
- *Interpolation between levels* (#6 question below): runs ~once per transition +
  one warm-start projection; amortized to negligible. Leave it.
- *Refinement triggers:* gated by `patience=30`, then O(30) Python work; measured
  0.0 % of wall; triggers already fire early. Nothing to do.

## The eight audit questions

1. **Projection algorithm / complexity / faster alternatives.** A hybrid — *not*
   Dykstra, *not* textbook alternating projection: each inner iter does an exact
   closed-form coupled affine projection (n×n Lagrange solve,
   `C = ‖v‖²(I − J/n) + εI`, `projection.py:79–107`), a non-negativity clamp (110),
   then multiplicative row + column Sinkhorn renormalizations (113–121). Per inner
   iter **O(V·N)** (~12–14 dense passes); the n×n solve is O(N³) but negligible
   (1.25×10⁵ vs V·N=5.7×10⁶ at L4, V=114,144). Per call O(inner·V·N). Faster =
   the dual-Newton/simplex rewrite (#3).
2. **Projection warm-starting.** Cold-restart every call (no warm-start arg,
   `A.copy()` at line 51, called fresh per backtrack trial at 304). **Negative
   finding** — no correct beneficial warm-start without restructuring (multipliers
   solved exactly; fixed point start-dependent → warm-starting corrupts the
   projection). See #13.
3. **Energy/gradient redundancy & allocation.** `K@u`, `interface_vec=u²(1−u)²`,
   `M@interface_vec` recomputed independently in energy (94–96) and gradient
   (137–139) with no caching; `g_post` (331) duplicates the next iteration's
   gradient (290) — #6; `‖g‖²` recomputed per trial (314) — #8; `var_w` dead in
   fixed mode (147) — #7. No preallocation / `out=` / in-place anywhere — ~300–450
   fresh length-V arrays per call, ×backtracks, ×2 for the gradient; strided
   `phi[:,i]` views force contiguity copies. Fixes: #5, #6, #7, #8.
4. **Backtracking & adaptive step.** Armijo step-halving with a `‖g‖²` surrogate;
   `step0` hard-reset to `1.0` every iteration (294); `ρ=0.5`, `c=1e-4`, no
   max-trial count (stops at `step<1e-12`). ~5.40 backtracks/iter at finest, 6.26
   aggregate, 15.8 at L0. No BB/spectral. **#1 is the top lever.**
5. **Refinement-trigger frequency.** No opportunity — 0.0 % of wall, gated by
   `patience=30`, fires early/beneficially. Only related micro-win is striding the
   secondary bookkeeping (#9).
6. **Interpolation cost.** Negligible — once per level transition, not in the hot
   path. Leave it.
7. **Sparse format & numpy anti-patterns.** Format already optimal (CSR) — no win
   (#13). Real anti-patterns: per-region matvec loops (#5), per-inner-iter
   copies/temporaries/redundant reductions in the projection (#4), strided-column
   contiguity copies. FEM assembly is a slow LIL Python loop but one-time per level
   (#12).
8. **Level schedule.** 5-level coarse→fine to V=114,144 at L4; matrices built once
   per level and dropped between (`relaxation.py:293–299`), fully amortized; seeded
   init at L0, interpolation warm-start at finer levels. The schedule itself is
   sound — no confirmed serial code lever lives in it. Schedule *tuning* (fewer
   intermediate-level iters, earlier triggers, fewer levels) is plausible
   algorithm-tuning but unaudited and would change solution quality — not
   recommended without measurement.

## Key takeaway

The serial wins are concentrated in the projection step (86–90 % of wall) and the
backtracking line search that re-invokes it. The single highest-confidence lever
(**#1**) is warm-starting/adapting the initial backtracking step — `step0` is reset
to `1.0` every iteration, forcing ~5.4 trials/iter (~4.4 rejected), and since both
the full projection and full energy run per trial, ~95 % of wall is per-trial cost;
cutting trials toward ~1.5–2 is roughly a 2× speedup. The projection
tolerance/stall concern (**#2**) is already resolved by the live `tol=1e-9`. The
dual-Newton projection rewrite (**#3**) is the durable structural win but is not
landable before a multi-day run. Everything in energy/gradient and the mesh/FEM
layer is secondary, bounded by the ~10–15 % non-projection share. The actionable,
safe, pre-run set is **#1 + #4 + #6** (with #8 folded in for free).

## Related documents

- Plan implementing #1, #4, #6:
  `docs/plans/PHASE1_PGD_SERIAL_OPTIMIZATIONS_PLAN.md`
- Empirical timing basis:
  `docs/plans/PHASE1_RELAXATION_TIMING_PROFILE.md`,
  `docs/math/04-phase1-timing-profile/`
- Code: `src/optimization/projection.py`, `src/optimization/pgd_optimizer.py`,
  `src/mesh/tri_mesh.py`, `src/pipeline/relaxation.py`
