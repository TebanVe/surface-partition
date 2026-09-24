---
paths:
  - "sweep/**"
---

# Parameter sweeps

`sweep/` is independent from the core scripts. Runs are grouped by experiment
identity (`{surface}_npart{N}`) rather than by sweep invocation — see
`run-layouts.md` layout 3.

```bash
python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml
python sweep/parameter_sweep.py --sweep <spec> --mode generate-only      # preview
python sweep/parameter_sweep.py --sweep <spec> --mode local-parallel --workers 4
python sweep/parameter_sweep.py --sweep <spec> --resume                  # skip completed
python sweep/parameter_sweep.py --sweep <spec> --mode collect            # rescan & index

python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/
python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/ --metric final_energy

python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/
python sweep/timing_analyzer.py --experiment-dir <dir> --phase relaxation
```

## Combination strategies

- `strategy: grid` — Cartesian product of all parameter lists.
- `strategy: paired` — zip together (all lists must have equal length).

Parameters that must scale together (e.g. `n_grid_x` and `n_grid_y`) go in a
**named group**: the group's parameters are zipped internally, then the group
participates in the cross-strategy as a unit.

## `experiment_index.yaml`

The central index: every run with its parameters, status, and key metrics
(perimeter, final_energy, initial_N, final_N, converged, total_iterations).
**Perimeter is the primary comparison metric** because it is
resolution-independent, unlike energy, which is eps-dependent.

With `--profile` runs, `--mode collect` also extracts `n_cells`,
`n_active_vps`, `n_triple_points`, per-campaign `timing_*` fields, and Phase 1
`relax_timing_*` fields read from `solution/timing_profile.yaml`. Those are what
`sweep/timing_analyzer.py` consumes.

⚠ The 185 summary traces living under sweep directories are invisible to a glob
of `results/run_*/traces/` — an offline replay that used one under-counted the
corpus by more than half. Enumerate both.

`cluster/cleanup_sweep_results.py` prunes the worst runs, keeping the N best by
perimeter.
