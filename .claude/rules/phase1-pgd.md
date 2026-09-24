---
paths:
  - "src/optimization/pgd_optimizer.py"
  - "src/optimization/projection.py"
  - "src/optimization/initialization.py"
  - "src/pipeline/relaxation.py"
  - "scripts/find_surface_partition.py"
  - "testing/validate_pgd_optimizations.py"
  - "testing/test_soft_area_constraint.py"
  - "testing/calibrate_soft_area_mu.py"
  - "testing/validate_struct_trigger.py"
---

# Phase 1 — Gamma-convergence relaxation (PGD)

Energy = `eps*u^T*K*u + (1/eps)*q^T*M*q` with `q = u(1-u)` (the double well
`int u^2(1-u)^2`) + `lambda*penalty`. Constraints: partition sum-to-one, equal
areas. Energy and gradient are in `compute_energy()` / `compute_gradient()`; the
penalty term is modular, controlled by `penalty_target_mode` and
`lambda_penalty`.

The interface term was previously mis-discretized as `u^2(1-u)^2` (a typo copied
from the paper, making the coded well `int u^4(1-u)^4` with an inconsistent
gradient); **fixed** in `6ff71a0`. See
`docs/reference/phase1_energy_discretization_bug.md`,
`docs/math/06-phase1-energy-discretization/`,
`docs/experiments/02-corrected-energy-highn-validation/`.

## `init_method` — seeded is effectively mandatory

`random` (dataclass default, kept for backward compatibility) or `seeded`
(farthest-point Voronoi seeds, deterministic given `seed`, via
`create_seeded_initial_condition`). Dispatch is `_create_initial_condition` in
`relaxation.py`, **level-0 branch only** — finer levels warm-start by
interpolation.

The corrected (steeper) well makes the symmetric diffuse state a local minimum,
so random init now **traps**: N=30 gives 43% worst-cell area error / 23
imbalanced cells against 0.7% seeded. Every config at N >= 30 sets
`init_method: seeded`. The `torus_30part_random_s*.yaml` configs are the
exception — random init is their experimental point, do not "fix" them.

## `lambda_penalty` has a working WINDOW at high N — over-raising backfires

The crispness penalty is the main lever against the winner-take-all runt, but it
has an upper **ceiling**, not just a lower bound.

- Too low -> diffuse runts (`docs/reference/winner_take_all_partition_gap.md`).
- Too high -> the penalty dominates, the multi-level refinement triggers misfire
  (finer levels fire after *tens* of iterations instead of thousands), and PGD
  stops before crisping the interfaces — leaving a diffuse
  `min peak density ~ 0.7` mush with most cells area-imbalanced, and the run
  finishes suspiciously fast.

At N=300: `lambda_penalty: 12` relaxes properly (min peak ~0.98, finest level
~7.7k iterations); `15` collapses to mush (min peak 0.71, 234/300 imbalanced).
The needed lambda grows with N (~5 at N=100, ~11 at N=200) but stays under the
ceiling; some high-N failures are also seed-specific.

**Diagnostic:** if a high-N run looks wrong, check the final min peak density
(`dormant_cells.max_density_per_cell` in `metadata.yaml`) and the per-level
`Refinement triggered at iteration N` counts in the log — a fast run with low
peak density means lambda is over the ceiling; lower it.

⚠ This window does **not** transfer to another surface, and lambda cannot be
calibrated from `||g||_inf` on a marching-cubes mesh — see `surfaces.md`.

## A floored line search is NORMAL convergence

Every healthy level ends with the Armijo step at the backtracking floor
`9.0949470177292824e-13` (= `pgd_step0 * pgd_backtrack_rho^40`) and the energy
frozen for exactly `refine_patience` (30) iterations, at which point the plateau
trigger fires. The validated N=100 deliverable does this on **four of its five
levels**. An earlier `LINE SEARCH STALLED` warning fired on any floored step and
therefore fired on every healthy termination; two commits reasoned from it
before it was checked against a known-good run.

**What is diagnostic is *when* the floor is reached, not that it is.** The guard
(`_STALL_EARLY_ITERS = 200`) warns `UNDER-RESOLVED LEVEL` only when the step
floors inside the first 200 iterations. That threshold sits in an empty gap —
but a **narrow one**: on the full tree (350 summary traces, including the 185 in
sweep directories that an earlier replay's glob could not see) three
`torus_npart10` levels floor at **227 / 280 / 303**, so the real gap is
**(100, 227)**. 200 is correct on all 350 levels, but the margin is **1.13x, not
6.7x**, and a threshold of 250 *would* misfire.

Two blind spots: it does not detect the A2 soft-area pathology (levels floored
*late*, collapse still real), nor levels that die **without ever flooring**
(three N=30 sweep levels end at 34-36 iterations with no floored step at all).

## A level can die doing no work — but "~90 verts/cell" is NOT established

The *phenomenon* is real: levels at 31/32/48/64/83 v/cell reach the floor at
iteration **18-47** and stop, so the cheap equalization phase never happens and
the first level that runs starts from a badly imbalanced field.

**The threshold's location is a different claim and is refuted as a constant.**
v/cell is causally implicated by exactly **one controlled pair** (N=100
lambda=5.1: 31 v/cell dies at onset 30, 96 v/cell never floors — report 06).
Every other point pooled into "~90" varies in lambda-window membership and
straddles the 2026-07-06 energy fix: `run_20260629_141012` is 96 v/cell and dies
at onset 32, because its lambda=2.1 is below the N=100 window. Separated by
start type the brackets are cold (64, 92) and warm (83, 124) — wide and
overlapping, and no cold N=300 point above 83 v/cell exists on disk.

Treat ~90 as a rough prior, never a design constant. **Do not confuse it with
the ~250-300 v/cell gate floor** of `winner_take_all_partition_gap.md` §4b (the
resolution at which a level can itself clear the 5% gate); they differ by ~3x in
mesh budget at N=1000.

Caveat that stops it being a rule: a dying level 0 is not by itself fatal —
N=200 reaches 0 imbalanced with a level 0 at 48 v/cell that dies at iteration 21,
because its level 1 (124 v/cell) then runs 12,155 iterations. The statement
consistent with every run is weaker: *some early, cheap level must run long
enough to equalize before the expensive levels begin.*

## A level below the resolution floor permanently damages the partition

Measured at N=100 (`docs/experiments/06-subfloor-ladder/`): dropping only the
coarsest level from 96 to 31 verts/cell — everything else, including the finest
mesh, identical to the validated control — yields a cell 15.92% off the
equal-area target where the control gets 0.78%. The defect is born at level 0
(46.80%) and is **not healed** by 37x more vertices downstream. It did *not*
produce fragmentation.

Note the sub-floor ladder is also *cheaper* (~24,300 s vs 48,132 s), so cost
alone never indicates a healthy ladder — judge validity per unit compute.

The N=300 ladder's own reconstruction — decelerating returns, level 2 doing all
the work, and why "the ladder is exhausted" is **not** established — is
`docs/reference/winner_take_all_partition_gap.md` §4b.

## Structure-based refinement trigger (`struct_trigger_enabled`) — experimental

A **stuck detector**, not a second convergence test. The energy-plateau trigger
compares `|dE|` against an *absolute* `refine_delta_energy`, so a level whose
partition has stopped changing keeps running as long as the energy creeps. On
the N=100 deliverable, level 0's labels are frozen from ~iteration 6,000 yet it
runs all 30,000 — **15,340 s, 31.9% of that run's whole Phase 1 ladder**, spent
after the answer stopped moving.

It tracks `argmax_k u_ik` and fires when fewer than
`struct_rate_tol * V * struct_window` labels have flipped over the last
`struct_window` iterations (defaults `1e-6`, `2000`).

**The window is long rather than the rate tight, and that is the whole design.**
The frozen churn rate at N=100 level 0 (4.2e-7) is indistinguishable from the
rate at N=300 level 4 shortly before its energy trigger legitimately fires
(4.6e-7). Rate alone cannot separate "frozen for good" from "nearly done" —
duration can.

Both structure rules are **sampled** every `struct_sample_stride` (default 500),
deliberately the granularity the offline replay uses: a per-iteration rule is a
*different* rule, because a vertex that flips and flips back is invisible to a
sampled one.

`struct_gate_enabled` is the same signal used the other way — it *blocks*
refinement while the label field is still moving (>1e-5 flips/iter/vertex over
500). The two Phase 1 pathologies are mirror images of one mis-calibrated
signal: the exact-projection run overruns a frozen structure, while the
soft-area run triggers at iteration 2,575 with 3.7% of vertices still changing
cell, and refinement then froze the pinched configuration that became its 3
disconnected cells.

Measured end to end at N=100 against the 13.4 h control `run_20260709_081548`
(live trigger fired at iteration 6,499, reproducing the offline replay exactly):

| | control | + structure trigger |
|---|---|---|
| Phase 1 wall | 48,132 s | **34,967 s** — 1.38x, saves 3.66 h |
| level 0 iterations | 30,000 (cap) | 6,500 |
| levels 1-4 iterations | 1372/986/1197/1279 | 1462/956/1192/1284 |
| dead / weak | 0 / 0 | 0 / 0 |
| imbalanced (worst) | 0 (0.78%) | 0 (0.97%) |
| fragmented | 0 | 0 |
| Phase 2 perimeter | 185.2546 | **185.2096** |

**1.38x, not the 1.47x predicted from iteration counts** — level 0's first 6,500
iterations cost 0.93 s each against the 30,000-iteration average of 0.653 s, so
the tail this trigger removes was the *cheap* part. Predicting a saving from
iteration proportions overstates it; 27.4% of Phase 1 is the measured figure.

Replay the rule offline against completed runs with
`testing/validate_struct_trigger.py`.

## Soft continuous equal-area constraint (`soft_area_constraint`) — experimental

The exact equal-area alternating projection is **93.3% of Phase 1 wall time**
(`docs/math/04-phase1-timing-profile/`), and its purpose — an equal-area
winner-take-all partition — is now met independently, exactly and at any N, by
the balanced readout at extraction time.

`soft_area_constraint: true` (default `false`) moves equal area into the
objective: energy `+= (mu/2)*sum_k r_k^2` with `r_k = (v.u_k - Abar)/Abar`;
gradient `+= (mu/Abar^2)*v_i*(v.u_k - Abar)`; and the per-trial line-search
projection becomes `project_rows_onto_simplex` — sum-to-one + box in closed
form, one sort, no inner loop.

1. **The entry projection stays exact** (`orthogonal_projection_iterative`), so
   an A/B starts each level from the identical iterate.
2. **`constraint_fun` drops the area block when the flag is on**, so `FEAS`
   keeps meaning "the constraints actually enforced". Area drift is reported by
   `area_deviation()`.
3. **`soft_area_adaptive` is the interface that transfers.** A hand-set
   `soft_area_from_level` has to be read off a *finished* run. Adaptive mode
   starts every level EXACT and switches inside the level once its own label
   churn drops below `soft_switch_rate_tol` (1e-5) over `soft_switch_window`
   (500). It **must** be within-level: every level starts churning and decays,
   so a "was the previous level settled?" test answers yes at the end of every
   level and turns the whole ladder soft. Leave `soft_area_mu: 0` and it is
   calibrated at the switch. In practice the soft phase collapses within ~50-70
   iterations of the switch, so the saving comes from stopping early.
4. **`mu` set by hand is per-config and not N-invariant.** Use
   `testing/calibrate_soft_area_mu.py --config <cfg>`: N=100 -> 245.9,
   N=30 -> 895.3.

**The falsifier is dormant cells, not runts.** Runts are expected here and the
readout repairs them; the hard mass constraint may be what keeps a cell *alive*
early in the flow, so any dead or weak cell fails the approach regardless of
speedup. Measured verdict at N=100: **rejected** — 32x faster, 0 dead, but 3-4
of 100 fragmented and a collapsed line search at every level. Its
**exact-at-level-0 hybrid survives** (6.83x, all gates passing, Phase 2 within
+0.246%) but requires the balanced readout. See
`docs/experiments/05-soft-area-constraint/`.

## Territory-aware relaxation — tried, measured, removed. Do not re-add.

A flag-gated WTA balance term, discrete-area trim, P2 reduced-gradient fix and
adaptive `wta_schedule` were implemented, merged for the record at `14e0518`,
and **removed** immediately after. It worked on the axis it targeted (N=200 bad
seed: worst runt -34% -> 0 imbalanced) and was a **net regression overall**: 14
of 200 disconnected cells against a matched 0/200 control, and a ladder
extrapolating to ~20 days at N=200. Root cause: it enforces *total* area through
a **nonlocal** operator (the projection), which buys a runt's area with far-away
territory.

Before proposing anything in this space, read
`docs/reference/winner_take_all_partition_gap.md` **§4b** (why it failed) and
**§4c** (the local-operator test any replacement must pass). Derivation kept at
`docs/math/07-phase1-wta-balance/` (marked not-adopted); measurement at
`docs/experiments/04-territory-aware-highn-validation/`.
