# Phase 1 PGD Serial Optimizations + Validation Harness

**Status:** Not Started

> Implement this on a dedicated branch (e.g. `feat/pgd-serial-optimizations`). Do
> **not** apply to `main` until the validation harness in Phase 4 passes. A long
> N=50 production relaxation may be running off `main`; these changes are for the
> *next* runs (an N=50 confirmation rerun, N=100, sweeps), not for hot-swapping
> into an in-flight job.

## Background

Phase 1 relaxation (`run_relaxation` in `src/pipeline/relaxation.py`) minimizes a
Γ-convergence (Modica–Mortola) energy over `N` region density functions via
Projected Gradient Descent (PGD) with per-step orthogonal projection onto the
partition (sum-to-one) and equal-area constraints, across multi-level mesh
refinement. A read-only audit of the serial hot path —
`docs/reference/PHASE1_PGD_SERIAL_OPTIMIZATION_AUDIT.md`, which assigns the stable
opportunity IDs (#1–#13) referenced throughout this plan, grounded in the empirical
timing in `docs/plans/PHASE1_RELAXATION_TIMING_PROFILE.md` and
`docs/math/04-phase1-timing-profile/` — established:

- The orthogonal projection is **86–90 % of wall time** at fine levels.
- The projection is re-run **on every backtracking trial step** inside the Armijo
  line search, and `step0` is hard-reset to `1.0` at the top of *every* PGD
  iteration.

Live measurement on the in-flight N=50 run
(`results/run_20260625_113015_surftorus_npart50_..._seed84172851`) confirmed:

- **The projection inner-loop cap is NOT being burned.** With
  `pgd_projection_tol = 1e-9` (already set in `parameters/torus_50part.yaml`) the
  log contains **zero** `"Projection did not converge after 300 iterations"`
  warnings. The residual floor (~1–2×10⁻¹⁰) is set by the hard-coded
  `epsilon = 1e-10` in `src/optimization/projection.py`, which is below the
  `1e-9` tolerance, so the inner loop converges. **Therefore the projection
  tolerance/stall problem is already solved and is explicitly out of scope here.**
- **The backtracking pathology is real and measured.** In
  `traces/pgd_part50_..._level1_summary.out` the accepted `STEP` column starts at
  `1.0` and decays to `3.9e-3 = 2⁻⁸` near convergence — i.e. **~8 rejected
  backtracking trials per iteration** at the tail, each a full projection + full
  energy evaluation, with `step0` re-walked from `1.0` every iteration.

This plan implements the three **safe, high-value, solution-preserving** serial
optimizations identified by the audit, plus a validation harness that proves they
do not change the computed minimizer while measuring the speedup.

### Scope

**In scope (three code changes):**

- **Change A — backtracking step warm-start** (audit **#1**; the headline ~2×
  lever — Tier 1 only). Carry the last accepted step into the next iteration's
  initial step instead of resetting to `1.0`. Keeps the Armijo acceptance test
  exactly, so the descent guarantee and the converged minimizer are preserved; only
  the per-iteration step *trajectory* changes. (Audit #1 "Tier 2", the
  Barzilai–Borwein step rule, is **not** part of this plan.)
- **Change B — projection inner-loop cleanup** (audit **#4**). Remove
  per-inner-iteration waste in `orthogonal_projection_iterative`: the
  `A_prev = A.copy()` + `np.allclose` stall check (replaced by a scalar-residual
  stall test), the dead `convergence_history` dict, and the per-iteration
  rebuild/re-solve of the constant matrix `C`. Result-preserving.
- **Change C — gradient reuse** (audit **#6**). The post-step gradient `g_post`
  computed at the accepted iterate equals the next iteration's pre-step gradient `g`
  (same `x`, same function). Carry it forward and delete one of the two gradient
  evaluations per iteration. Bit-identical (exact).

**Explicitly OUT of scope:**

- **Projection tolerance / residual-stall stopping** ("opportunity #2" in the
  audit) — already resolved by the live `tol = 1e-9` setting; do not change
  `epsilon` or the convergence test.
- **Dual-Newton / simplex-projection rewrite** ("opportunity #3") — the durable
  structural win for `N ≥ 100`, but a new algorithm with real correctness risk;
  defer to a separate plan.
- The energy/gradient batching rewrite, dead-`var_w` removal, `‖g‖²` hoist, FEM
  assembly vectorization, energy-component reuse, and bookkeeping/flush striding
  (audit opportunities #5, #7, #8, #9, #10, #11, #12) — optional follow-ups, not
  part of this plan. (One trivial exception: the `‖g‖²` hoist, audit #8, sits inside
  the loop edited by Change A and may be folded in for free — see Phase 1, optional
  sub-item.)

### Key facts for the implementer

- `optimize()` (`src/optimization/pgd_optimizer.py:199`) runs the **entire PGD loop
  for one mesh level** in a single call; it is invoked once per level from
  `_optimize_level` (`src/pipeline/relaxation.py:571`). Change A's warm-start state
  is therefore **loop-scoped within one `optimize()` call** — each level correctly
  restarts from `step0`. No cross-call/cross-level state is needed.
- **No new config field is required.** The warm-start cap is the existing `step0`
  argument and the growth factor is `1 / backtrack_rho`, both already threaded from
  `RelaxationConfig` (`pgd_step0`, `pgd_backtrack_rho`) through `_optimize_level`.
- Line numbers below are accurate as of writing; if they have drifted, locate the
  edit by the quoted anchor text.

---

## Phase 1 — Change A: backtracking step warm-start

**Status:** Not Started
**File:** `src/optimization/pgd_optimizer.py`, inside `optimize()`, the per-iteration
loop and Armijo line search (currently lines ~286–326).

### Current code (anchors)

```python
for k in range(maxiter):
    # Gradient at current x
    ...
    g = self.compute_gradient(x)                      # line ~290
    ...
    # Backtracking line search
    step = float(step0)                               # line ~294  <-- hard reset
    accepted = False
    n_backtracks = 0
    ...
    while True:
        ...
        A_trial = x.reshape(N, n) - step * g.reshape(N, n)
        A_trial = np.clip(A_trial, 1e-8, 1 - 1e-8)
        A_trial = orthogonal_projection_iterative(A_trial, c, d, self.v, ...)
        x_trial = A_trial.flatten()
        E_trial = self.compute_energy(x_trial)
        # Armijo condition with ||g||^2 surrogate
        if E_trial <= E - armijo_c * step * float(np.dot(g, g)):   # line ~314
            accepted = True
            x = x_trial
            E = E_trial
            break
        step *= backtrack_rho                          # line ~319
        if step < 1e-12:
            break
```

### Change

1. **Before the loop** (just before `for k in range(maxiter):`, after the initial
   `E = self.compute_energy(x)` / `curr_x` setup near line ~264), initialize the
   warm-start state:

   ```python
   prev_step = float(step0)   # last accepted step; seeds the next line search
   ```

2. **Replace** the hard reset at line ~294:

   ```python
   step = float(step0)
   ```

   with a warm start one notch **above** the last accepted step, capped at `step0`:

   ```python
   # Warm-start the line search from just above the last accepted step so the
   # search converges in ~1-2 trials instead of re-walking from step0 each time.
   # Cap at step0; backtrack_rho in (0,1) so prev_step/backtrack_rho > prev_step.
   step = min(float(step0), prev_step / backtrack_rho)
   ```

   On the first iteration `prev_step == step0`, and `step0 / backtrack_rho > step0`,
   so the `min` selects `step0` — **iteration 0 is identical to the current
   behaviour.**

3. **On acceptance**, record the accepted step. In the `if E_trial <= ...:` block,
   after `E = E_trial`:

   ```python
   prev_step = step
   ```

4. **On a failed search** (step underflow), reset so the next iteration recovers
   from `step0`. In the `if step < 1e-12:` block, before `break`:

   ```python
   prev_step = float(step0)
   ```

### Optional free co-optimization (audit #8, same loop)

`float(np.dot(g, g))` is recomputed on every trial inside the `while` loop although
`g` is invariant across the line search. Hoist it just before the loop:

```python
gg = float(np.dot(g, g))
```

and use `... - armijo_c * step * gg` in the Armijo test. Bit-identical; saves one
`O(V·N)` dot per rejected trial. Include only if convenient.

### Correctness

- The Armijo acceptance test (line ~314) is **unchanged**, so every accepted step
  satisfies sufficient decrease — monotone energy descent and the converged
  minimizer basin are preserved.
- Only the per-iteration step *size trajectory* changes, so the iterate path will
  differ from `main` and the final per-level solution will match **within optimizer
  tolerance, not bit-for-bit** (see Phase 4 acceptance criteria).
- Because refinement triggers fire on energy plateaus
  (`refine_delta_energy`), a different trajectory may trigger refinement at a
  slightly different iteration and change total iteration counts; the final
  per-level minimizer must still be equivalent.

### Risk

Low. Assumes `backtrack_rho ∈ (0, 1)` (default `0.5`; always true). The cap at
`step0` prevents unbounded step growth. This is the standard "grow one notch, then
backtrack" adaptive step rule with the Armijo guarantee intact. (A Barzilai–Borwein
step rule — audit opportunity #1 "Tier 2" — is explicitly *not* part of this plan because
it is non-monotone and needs a Grippo-type safeguard.)

---

## Phase 2 — Change B: projection inner-loop cleanup

**Status:** Not Started
**File:** `src/optimization/projection.py`, function
`orthogonal_projection_iterative` (currently lines 8–170). **Do not touch**
`orthogonal_projection_direct` or the hard-coded `epsilon = 1e-10` (it sets the
residual floor that the live `tol = 1e-9` deliberately clears).

### Edits

1. **Hoist the constant matrix `C` and its regularized form out of the loop.**
   `v_norm_squared`, `C` (lines ~80–82), and `np.eye(n)` (line ~90) depend only on
   `v` and `n`, which are fixed for the whole call. Compute once before the
   `for iter in range(max_iter):` loop:

   ```python
   v_norm_squared = np.sum(v**2)
   C = np.full((n, n), -v_norm_squared / n)
   np.fill_diagonal(C, v_norm_squared - v_norm_squared / n)
   C_reg = C + epsilon * np.eye(n)
   ```

   Inside the loop, replace the per-iteration rebuild + `np.linalg.solve(C + epsilon*np.eye(n), q)`
   with a solve against the precomputed `C_reg`:

   ```python
   lambda_vec = np.linalg.solve(C_reg, q)
   ```

   Preserve the existing singular-matrix fallback (the `except np.linalg.LinAlgError`
   branch that solves the reduced system) — precompute the reduced
   `C_reg[:-1, :-1]` once as well if you keep the fallback path. `C_reg` is
   `epsilon`-regularized and effectively never singular, so the fallback rarely
   fires. **The numerical result of the solve must be unchanged** (same matrix,
   same `q`).

2. **Remove the `A_prev` copy and `np.allclose` stall check.** Delete
   `A_prev = A.copy()` (line ~71). Replace the iterate-delta stall test (line ~144,
   `if iter > 0 and np.allclose(A, A_prev, rtol=tol, atol=tol):`) with a
   **scalar-residual** stall test on the already-computed `max_error` (line ~126):

   ```python
   # near the top, before the loop:
   prev_max_error = np.inf
   ...
   # replace the np.allclose stall block (after the convergence test, ~line 144):
   if iter > 0 and (prev_max_error - max_error) < 1e-2 * tol and max_error < 10 * tol:
       # residual has plateaued and is already near tolerance: stop.
       break
   prev_max_error = max_error
   ```

   The `max_error < 10 * tol` gate ensures the loop **never returns an infeasible
   iterate** — it only stalls out when already close to the constraint tolerance.
   This removes two `O(V·N)` passes per inner iteration (the copy and the
   `allclose`).

3. **Remove the dead `convergence_history` dict.** Delete its initialization
   (lines ~63–68) and the four `convergence_history[...].append(...)` calls
   (lines ~129–132). The function returns only `A` (line ~170); the history is
   never read or returned.

4. **Keep** the primary convergence test (line ~138,
   `if row_sum_error < tol and area_error < tol:`), the non-negativity clamp, the
   row/column renormalizations, the `_prof` profiling hooks, and the final
   validation (`if final_row_error > 10*tol ...: raise RuntimeError`).

### Correctness

- Edits 1 and 3 are pure refactors — bit-identical converged `A`.
- Edit 2 changes only the *stall stopping rule*. Because the projection now
  converges via the residual test (line ~138) at `tol = 1e-9` in practice, the
  stall break is a rarely-exercised safety net; when it does fire it returns an
  iterate already within `10*tol`. The converged `A` matches `main` to within the
  projection tolerance (validated in Phase 4 with `atol = max(1e-10, 10*tol)`).

### Risk

Low. Edit 2 is the only behavioural change; it is gated to never weaken
feasibility. Edit 1 must reproduce the regularized solve exactly (same `C_reg`).

---

## Phase 3 — Change C: gradient reuse

**Status:** Not Started
**File:** `src/optimization/pgd_optimizer.py`, `optimize()` (lines ~286–333,
~350, ~366).

### Rationale (verified)

`g_post = self.compute_gradient(x)` (line ~331) is evaluated at the accepted
iterate. `x` is **not mutated** anywhere between line ~331 and the next iteration's
`g = self.compute_gradient(x)` (line ~290): lines ~336–410 only read `x`
(`constraint_fun`, norms, HDF5 save, area logging, `curr_x = x.copy()`,
`best_x = x.copy()`, refinement-trigger check, which only raises). Therefore
`g(k+1) == g_post(k)` exactly (same function, same input).

### Change

1. **Compute the initial gradient once before the loop**, after `x` is set (the
   projected `x0`, line ~258). Profile-wrap it the same way the per-iteration
   gradient is wrapped:

   ```python
   if profile is not None:
       _t_g = time.perf_counter()
   g = self.compute_gradient(x)
   if profile is not None:
       profile.record('gradient', time.perf_counter() - _t_g)
   ```

2. **Delete the per-iteration pre-step gradient recompute** at line ~290 (the
   `g = self.compute_gradient(x)` and its profiling wrapper at lines ~288–292).
   `g` now holds the gradient at the current `x` on loop entry.

3. **Keep** `g_post = self.compute_gradient(x)` at line ~331 (still needed for
   `gnorm_post` and the HDF5 save), and **at the end of each iteration carry it
   forward** as the next iteration's `g`. Add, after `g_post` is finished being used
   (e.g. just before or alongside `self.curr_x = x.copy()` at line ~369):

   ```python
   g = g_post   # reuse: next iteration's pre-step gradient == g_post at this x
   ```

   On a non-accepted iteration `x` is unchanged, so `g_post == g` and the carry-
   forward remains correct.

### Correctness

Bit-identical to `main` in exact arithmetic (the gradient *values* are unchanged;
only the number of evaluations drops from 2 to 1 per iteration). Across two
separate processes, BLAS reduction-order may introduce ~1e-14 noise — validate with
a `< 1e-12` threshold, not strict equality.

### Risk

None functionally while `x` is unmodified between line ~331 and the next ~290 (a
verified invariant). If a future edit mutates `x` in the logging/trigger region,
this invariant breaks — add a brief comment at the `g = g_post` line noting the
dependency.

---

## Phase 4 — Validation harness

**Status:** Not Started
**New file:** `testing/validate_pgd_optimizations.py` (CLI diagnostic, consistent
with the `testing/` convention — argparse, no pytest, uses `print` for user-facing
output).

The harness has **two modes**. Mode 1 (in-process, fast) proves the
result-preserving changes B/C are numerically faithful. Mode 2 (compare two
completed runs) proves the end-to-end equivalence and measures the speedup.

### Mode 1 — `--equivalence` (in-process numerical proof)

**1a. Projection equivalence (Change B).** Freeze a verbatim copy of the *original*
`orthogonal_projection_iterative` body into the harness as
`reference_projection_iterative` (so the test does not depend on git history). For
a grid of sizes `(V, N) ∈ {(200,5), (500,10), (1000,30), (2000,50)}` and several
fixed seeds:

- Build a random `A = rng.random((V, N))`, a positive lumped-mass-like vector
  `v = rng.random(V) + 0.1`, and `d = (v.sum() / N) * np.ones(N)`, `c = np.ones(N)`.
- Run both the **new** `orthogonal_projection_iterative` and
  `reference_projection_iterative` with identical `max_iter` and `tol` (use
  `tol = 1e-9` to match production).
- Assert `np.max(np.abs(A_new - A_ref)) < max(1e-10, 10 * tol)` and that both
  results satisfy the constraints (`max row-sum error < 10*tol`,
  `max area error < 10*tol`). Report the actual deltas.

Rationale for the tolerance (not strict equality): Change B's stall rule may stop
at a different inner iteration, so the two converged matrices agree to ~`tol`, not
bit-for-bit. Pure-refactor edits (hoisting `C`, removing dead history) do not move
the result.

**1b. Gradient-reuse identity (Change C).** Build a small
`ProjectedGradientOptimizer` (tiny synthetic `K`, `M`, `v`, e.g. from
`TorusMeshProvider` at minimal resolution). Pick a random feasible `x`, take one
manual PGD step to an accepted `x'`, and assert
`compute_gradient(x')` is reproducible (call it twice, assert
`max|Δ| < 1e-12`). This is a sanity check; the load-bearing proof for C is the
bit-identical end-to-end gate in Mode 2.

### Mode 2 — `--compare --baseline <run_dir> --candidate <run_dir>`

Compares two completed Phase 1 run directories produced by
`scripts/find_surface_partition.py` on the **same config and seed** — one built
from `main`, one from the branch. Load each run's base solution via the existing
helpers in `src/pipeline/io.py` (`detect_run_layout`, `find_base_solution_path`) or
by globbing `solution/*.h5` (datasets `x_opt`, `vertices`, `faces`; attr
`n_partitions`), plus `solution/metadata.yaml` and, when present,
`solution/timing_profile.yaml`.

Compute and report (assert where noted):

1. **Final energy.** From the finest-level trace summary
   (`traces/*_level{max}_summary.out`, last row, `OBJFUN` column) or
   `metadata.yaml`. Same config ⇒ comparable (energy is ε- and resolution-
   dependent, so only compare same-config runs). Assert relative difference
   `|E_b − E_c| / |E_b|` below the stage threshold (see below).
2. **Solution vector.** Reshape `x_opt` to `(V, N)`. For exact stages assert
   `max|x_b − x_c| < 1e-12`. (Same mesh + same seed ⇒ aligned vertex indices.)
3. **Partition equivalence (permutation-invariant).** Per-vertex labels via
   `argmax` over regions. Region indices may be permuted between runs, so build the
   `N×N` overlap matrix and find the optimal relabeling with
   `scipy.optimize.linear_sum_assignment`; report the percentage of vertices that
   agree after optimal relabel. Assert `≥ 99.5 %` for behavioural stages.
4. **Region areas.** `areas = v @ Phi`, sorted; assert relative max difference
   `< 1e-3` (permutation-invariant geometric check).
5. **Speedup (report, and assert improvement).** Mean backtracks per iteration from
   `timing_profile.yaml` (`backtracks_per_iter_total / major_iterations`,
   populated only with `--profile`) and total relaxation wall time. Assert mean
   backtracks(candidate) `<` mean backtracks(baseline); report the wall-time ratio.

#### Stage thresholds

The three changes are committed and validated **in order C → B → A** so each gate
has a known expected delta:

| After commit | Expected vs `main` | Energy / `x_opt` threshold | Partition |
|---|---|---|---|
| **C** (gradient reuse) | exact | `max|Δx| < 1e-12`, energy rel `< 1e-10` | 100 % |
| **B** (projection cleanup) | result-preserving | `max|Δx| < 1e-9`, energy rel `< 1e-8` | 100 % |
| **A** (step warm-start) | within tolerance | energy rel `< 1e-6`, areas rel `< 1e-3` | `≥ 99.5 %` |

Validating in this order means any unexpected divergence is caught at an *exact*
stage (C or B) before the *behavioural* stage (A), isolating the cause.

### A/B run protocol (document inside the harness `--help` and here)

1. Choose a **small, fast, deterministic** config — `parameters/torus_10part.yaml`
   with a **fixed `seed`**. Optionally reduce `refinement_levels` / `max_iter` for
   speed, but use the **identical** config file for both branches. Run with
   `--profile` so backtrack counts and timing are recorded.
2. `git checkout main` →
   `python scripts/find_surface_partition.py --config parameters/torus_10part.yaml --profile`
   → note the `results/run_.../` directory (this is `--baseline`).
3. `git checkout feat/pgd-serial-optimizations` (after the commit under test) →
   run the **same** command → note the new `results/run_.../` directory (this is
   `--candidate`).
4. `python testing/validate_pgd_optimizations.py --compare --baseline <baseline_run_dir> --candidate <candidate_run_dir>`.
5. Also run `python testing/validate_pgd_optimizations.py --equivalence` (mode 1)
   after the B and C commits.

Both runs must execute on the same machine (BLAS reduction order affects the
`1e-12` exact thresholds).

---

## Phase 5 — Documentation sync & sign-off

**Status:** Not Started

Per the standing rules in `CLAUDE.md` ("Keeping Documentation in Sync"):

1. **Register the harness.** Add `testing/validate_pgd_optimizations.py` to the
   `testing/` list in `CLAUDE.md` and to `testing/README_testing.md`.
2. **No config/CLAUDE.md schema change** is expected (no new `RelaxationConfig`
   field, no new CLI flag). If the implementer chooses to expose the warm-start as
   a toggle, that *would* require a `CLAUDE.md` + config-doc update — but the
   default plan adds no such flag.
3. **Record the measured speedup** (mean backtracks before/after, wall-time ratio)
   in the commit message and, if it is a durable finding, append a short note to
   `docs/plans/PHASE1_RELAXATION_TIMING_PROFILE.md` (or its reference successor).
4. **Retire this plan.** Once all three changes are merged and validated, this plan
   is fully implemented: delete it, or move any lasting explanation (e.g. the
   warm-start step rule and the gradient-reuse invariant) into a
   `docs/reference/` note and cross-reference it.

### Acceptance criteria (all must hold on the branch before merge)

- [ ] Mode 1 `--equivalence` passes (projection within `max(1e-10,10·tol)`;
      gradient identity within `1e-12`).
- [ ] Mode 2 after the **C** commit: solution bit-identical to `main` (`< 1e-12`).
- [ ] Mode 2 after the **B** commit: solution within `1e-9`, partition 100 %.
- [ ] Mode 2 after the **A** commit (and combined): energy rel `< 1e-6`, region
      areas rel `< 1e-3`, partition agreement `≥ 99.5 %`.
- [ ] Mode 2 reports a **reduction in mean backtracks per iteration** (target: from
      ~5–8 down to ~1.5–2) and a material total wall-time reduction.

---

## Related documents

- Code (edit sites):
  - `src/optimization/pgd_optimizer.py` — `optimize()` loop (Changes A, C)
  - `src/optimization/projection.py` — `orthogonal_projection_iterative` (Change B)
  - `src/pipeline/relaxation.py` — `_optimize_level` call site (`:571`); confirms
    per-level, loop-scoped warm-start and existing `step0`/`backtrack_rho` plumbing
  - `testing/validate_pgd_optimizations.py` — new validation harness (Phase 4)
- Reference / empirical basis:
  - `docs/reference/PHASE1_PGD_SERIAL_OPTIMIZATION_AUDIT.md` — the full ranked audit
    and the source of the stable opportunity IDs (#1–#13) used in this plan
  - `docs/plans/PHASE1_RELAXATION_TIMING_PROFILE.md`
  - `docs/math/04-phase1-timing-profile/` — per-callback timing breakdown
- Out-of-scope follow-ups (future plans): dual-Newton/simplex projection rewrite
  (audit opportunity #3); energy/gradient CSR-batching and other micro-opts
  (audit #5, #7, #9, #10, #11, #12).
