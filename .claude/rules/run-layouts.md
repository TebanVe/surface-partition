---
paths:
  - "scripts/**"
  - "src/pipeline/io.py"
  - "src/export/**"
  - "src/h5util.py"
  - "src/profiling.py"
  - "testing/check_fragmentation.py"
  - "testing/watch_level_gates.py"
---

# Run output layouts, export, and how to measure a run

`detect_run_layout()` (`src/pipeline/io.py`) supports the structured layout below
and the legacy flat layout.

## Layout 1 — `run_*` (PGD Phase 1)

```
results/run_{timestamp}_surf{surface}_npart{N}_v1..._v2..._lam{lam}_seed{S}/
├── experiment.yaml               # verbatim copy of the input config
├── solution/
│   ├── surface_part{N}_..._{timestamp}.h5
│   ├── checkpoint_level{L}.h5    # newest only; deleted once the run finishes
│   ├── metadata.yaml             # derived runtime results
│   └── timing_profile.yaml       # only with --profile
├── traces/                       # per-level summary .out + internal_data.hdf5
├── readout/{campaign}/           # optional balanced-readout bridge
│   └── refinement/{campaign}/    # Phase 2 FROM a readout nests here, not at run root
├── refinement/{campaign}/        # iteration_NNN_*.h5, refinement.yaml, .log
├── analysis/
└── logs/relaxation.log
```

Campaign names come from `build_campaign_name()`: `{method}_btol{boundary_tol}`
plus non-default IPOPT extras (`_lbfgs{N}`, `_hess`, `_bestiter`, `_partial`) and
distance (`_midpoint` / `_dist{value}`).

**Per-level checkpoints (`checkpoint_per_level`, default `True`) are what make a
multi-day run survivable.** Writes go to
a `.tmp` and are moved into place, so a kill mid-write leaves the previous
checkpoint intact. `completed_levels` is the *absolute* ladder position, so
resuming a resumed run works.

## Layout 2 — `arm_*` (approach B) — a SECOND, DIFFERENT genre

```
results/arm_{mbo|init}_{timestamp}_npart{N}_V{V}_seed{S}/
├── experiment.yaml               # the config that set the ladder
├── arm_report.yaml               # THE results record
├── solution/arm_{mode}_part{N}_V{V}_seed{S}.h5   # Phase 1 SCHEMA
├── refinement/{campaign}/
└── partition/                    # only after scripts/export_partition.py
```

| | `run_*` | `arm_*` |
|---|---|---|
| name encodes | surface, mesh schedule, **lambda**, seed | npart, **V**, seed |
| | | **no lambda** (MBO has no crispness penalty), no surface |
| results record | `solution/metadata.yaml` | **`arm_report.yaml`** at the run root |
| also present | `traces/`, `logs/`, `analysis/`, `readout/` | none of them |

`arm_report.yaml` holds the `config` block, `init`, per-`levels` entries with the
full tau diagnostics (`cap_active`, `sqrt_tau_over_h_max`, `c_lo_hmax_local`),
`gates_raw` (with `vacuous_for_arms` on dormant), `label_boundary_length`, and
`phase2` with the whole perimeter trajectory, `best_iteration` and `censored`.

The solution is written in the **Phase 1 schema on purpose** so
`refine_perimeter.py`, `visualize_partition_fast.py`, `export_partition.py` and
`testing/check_fragmentation.py` consume it unchanged. That schema includes
`var1`/`var2` and `completed_levels`, which `export_partition.py` requires;
`run_mbo_arm.py` supplies them and asserts `var1*var2 == V` **only for structured
surfaces**. ⚠ **Arm runs produced before 2026-08-22 lack them and fail export
with `KeyError: 'var1'`** — backfill the attrs rather than re-running. An `arm_*`
directory has no `run_*` ancestor, so `source_run_id` falls back to the arm
directory name, which is correct and needs no flag.

## Layout 3 — experiment directories (parameter sweeps)

`results/{surface}_npart{N}/` groups runs by experiment identity rather than by
sweep invocation, with `experiment_index.yaml` as the central auto-maintained
index and `sweeps/` holding each sweep spec + summary CSV. See `sweeps.md`.

## Export

```bash
python scripts/export_partition.py \
  --solution results/<run>/refinement/<campaign>/iteration_NNN_*.h5 \
  --config parameters/<cfg>.yaml \
  --output results/<run>/partition/<name>.h5 [--force-finalised]
```

`finalised = not pending_migration`, and downstream repos gate on
`finalised == True`. **Every high-N export needs `--force-finalised`** because
the migration-cycling plateau leaves `pending_migration=True` on the best
iterate; the flag writes `finalised=True` plus an explanatory `finalised_note`,
and is mutually exclusive with `--strict`.

`source_run_id` is taken from the nearest `run_*` ancestor of the checkpoint, not
from `run_dir` — a readout-derived checkpoint sits one level deeper
(`readout/{campaign}/refinement/{campaign}/`), so the old `parent.parent.parent`
landed on the readout campaign name.

The record of what has been exported is `docs/reference/deliverables.yaml`,
verified by `python scripts/check_deliverables.py`.

## Measuring a run: four traps

1. **`run_time_seconds` in `solution/metadata.yaml` is NOT the run's wall time.**
   It is the *last level's* PGD elapsed time only. Sum `level_wall_s` over
   `levels` in `solution/timing_profile.yaml` instead (requires `--profile`).
2. **`timing_profile.yaml`'s `summary.total_wall_s` is NOT the Phase 2 campaign
   wall time** — it is solver time only, a 19x under-report at N=500. Take
   campaign wall from the `iteration_*.h5` timestamps.
3. **Exclude aborted launches.** The N=750 mesh-matched campaign was started
   twice, leaving 21 iterate files with two bit-identical copies of iteration 1;
   measuring from the directory spans both and reads 14,021 s against a real
   12,908 s. Check `grep -c "STARTING ITERATIVE REFINEMENT" refinement.log`
   before quoting any campaign wall — it is 1 for nine of the ten arm campaigns.
4. **Read the run's own `experiment.yaml`, never `parameters/`.** Configs drift:
   `torus_300part_seeded_lam11p5_original_energy.yaml:45` says
   `refinement_levels: 5` (raised post-hoc to drive a resume) while the anchor
   run `run_20260806_123326` was produced with **3** and its solution is
   V=47,488. Reconstructing a mesh from the live `parameters/` file lands on a
   *finer* mesh than the run being compared against — a ~0.11% perimeter effect
   pointing in the newcomer's favour.

⚠ `results/` is gitignored, so **each git worktree has its own and none of it is
in version control** — enumerate *both* working directories before claiming a run
does not exist.

## HDF5 compression (`src/h5util.py`)

`create_dataset()` applies transparent gzip **only to one-hot payloads** —
`arm_harness` (`x_opt`/`x0`) and `pipeline_orchestrator`
(`indicator_functions`). Measured: gzip=4 shrinks such a block **459x** (457 MB
-> 1.0 MB) and is *faster* end to end.

The ratio comes entirely from the data being almost all zeros and does **not**
generalise: on Phase 1's *continuous* PGD density the same setting gives **1.21x**
and makes the write **42.6x slower** (0.04 s -> 1.64 s on 91 MB). So
`relaxation.py` and `balanced_readout.py` deliberately do **not** use it — and
relaxation's write is the per-level checkpoint that makes multi-day runs
survivable. Not applied to `src/export/writer.py` (a downstream-consumed format,
not ours to change unilaterally) or to PGD traces. Datasets under 1 MiB and
scalars are left alone. gzip is a standard HDF5 filter, so h5py decompresses
transparently: no reader changed, and pre-compression files still open. Gates:
`testing/test_h5_compression.py`.
