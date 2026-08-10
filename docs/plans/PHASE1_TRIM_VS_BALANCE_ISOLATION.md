# Phase 1 — Isolating the cause of split cells: trim vs balance term

**Status:** planned (configs written, not yet run). 2026-08-03.

## Why

The N=200 reseed run on the *proven-clean* seed 61803399
(`run_20260730_211516`, `wta_schedule: adaptive`, λ=11) produced **14 of 200
fragmented cells** by internal level 1 — *worse* than the 3 fragmented cells on
the bad seed that motivated the reseed, and far worse than the matched control
`run_20260722_175451` (same seed/λ/mesh, **machinery OFF**, 0 fragmented,
finalised deliverable). So at N=200 the territory-aware machinery is a net
regression: the original energy already produces a clean partition, and the
machinery both is far slower (never triggers refinement) and *manufactures*
disconnected cells.

The machinery is a bundle. Two pieces actively push cell areas toward equal and
are the fragmentation suspects — they have **never been run separately**:

- **balance term** (`wta_balance_enabled`) — an extra energy penalty pulling each
  cell's *total* territory toward the equal-area target.
- **discrete-area trim** (`wta_trim_enabled`) — periodically retargets the
  projection's per-cell area goals toward exact winner-take-all equality.

The leading suspect is the **trim** (in the failed run it was clamp-saturated for
most of the run, and its persistent worst-deviation cells at level 0 — 198, 16,
37 — are exactly the worst fragmented cells at level 2). But that is not proven:
balance and trim were always on together (the `adaptive`/`all_levels` schedule
forces balance+trim+reduced-gradient on as one unit). This experiment turns them
on one at a time.

## The knob that makes isolation possible

`ProjectedGradientOptimizer` has independent switches for the three machinery
pieces. `_setup_level` (`src/pipeline/relaxation.py`) reads them **verbatim only
when `wta_schedule: off`**:

```
if wta_active is None:                # wta_schedule: off
    eff_balance = config.wta_balance_enabled
    eff_trim    = config.wta_trim_enabled
    eff_reduced = config.pgd_reduced_gradient
else:                                 # adaptive / all_levels
    eff_balance = eff_trim = eff_reduced = wta_active   # forced together
```

So every config here sets **`wta_schedule: off`** and selects the pieces with the
individual flags. (`adaptive`/`all_levels` would defeat the isolation.)

## The four experiments (2×2 on balance × trim, reduced-gradient held as control)

The reduced gradient is *not* an area-forcing mechanism (it only changes the step
direction), so it is unlikely to fragment on its own — but it was ON in the failed
run, so we hold it ON for the two machinery cells and add a `reduced-only` cell to
prove it is inert. All four run the **same seed (61803399), λ=11, 2 levels**
(final mesh 162×154 = 24,948 vertices = the exact `checkpoint_level02` point where
the failed run showed 14 fragments).

| config file | balance | trim | reduced | role / question |
|---|---|---|---|---|
| `..._iso_baseline.yaml`    | off | off | off | pure original energy = the `main` control. Must be clean. |
| `..._iso_reducedonly.yaml` | off | off | **on** | does the reduced-gradient step alone fragment? (expected: no) |
| `..._iso_trimonly.yaml`    | off | **on** | **on** | **is the trim the culprit?** (prime suspect) |
| `..._iso_balanceonly.yaml` | **on** | off | **on** | or is the balance term the culprit? |

Reference points already known: **both on** (the failed run) = 14 fragmented;
**machinery off, 5 levels** (`run_20260722_175451`) = 0 fragmented.

### How to read the result
- `trimonly` fragments **and** `balanceonly` clean → **trim is the cause.** Fix or
  drop the trim; the balance term may be salvageable.
- `balanceonly` fragments too → the problem is inherent to the area-forcing
  approach, not just the trim.
- `baseline`/`reducedonly` also fragment at 162×154 → fragmentation at intermediate
  levels is *normal* and heals by the final level (as the main deliverable did), so
  the machinery's real problem is only its cost (never triggering). Different
  conclusion entirely.

## Run (cluster)

Profiling is on via `profile: true` in each YAML (the submit script does not pass
`--profile`). Per-level checkpoints are on, so fragmentation can be checked at
level 0 *and* the final level 1, and a slow run can be killed early once its answer
is clear.

```bash
for cfg in baseline reducedonly trimonly balanceonly; do
  bash cluster/submit_relaxation.sh \
    --config parameters/torus_200part_s61803399_lam11_iso_${cfg}.yaml \
    --time 48:00:00 --cpus 8
done
```

Expected cost (2 coarse levels, 30000-iter cap): `baseline` and `reducedonly`
finish in hours (they trigger refinement early); `balanceonly` is moderate;
`trimonly` is the slow one (~1.5–2 days — the trim breaks the refinement trigger,
so it runs the full cap per level). Kill `trimonly` early if its `checkpoint_level01`
already answers the question.

## Check (after each run, or on live checkpoints)

`run_relaxation` already writes all three gates into `solution/metadata.yaml`, but
`check_fragmentation.py` prints them for any solution *or* per-level checkpoint:

```bash
# final level-1 solution (162x154):
python testing/check_fragmentation.py results/<run>/solution/surface_*.h5
# intermediate level-0 checkpoint (100x96):
python testing/check_fragmentation.py results/<run>/solution/checkpoint_level01.h5
```

The line that matters is `n_fragmented` (and `VERDICT`). Compare across the four
runs at the same level.

## Identifying which run is which
All four runs share seed/λ/mesh, so their result-directory names differ only by
timestamp. Identify each by its copied `experiment.yaml` (the `wta_balance_enabled`
/ `wta_trim_enabled` / `pgd_reduced_gradient` flags) or by `experiment.name`, or
submit them one at a time and record each job's run directory.
