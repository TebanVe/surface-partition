# CLAUDE.md — surface-partition

> **This file is capped at 350 lines.** It holds only what is needed in *every*
> session. Topic knowledge lives in `.claude/rules/` (loaded when you touch the
> matching files) and findings live in `docs/`. Before adding anything here, read
> **Maintaining this file** at the bottom.

## What this is

A Python framework computing **minimal-perimeter partitions on closed
triangulated surfaces**, implementing Bogosel & Oudet, *Partitions of Minimal
Length on Manifolds*, in two phases:

1. **Phase 1** — produce a valid labelling of the surface into `N` equal-area
   cells. Two implementations: **PGD** (the original Γ-convergence relaxation)
   and **approach B** (auction-dynamics MBO), which replaces it.
2. **Phase 2** — direct constrained perimeter minimization on contour variable
   points, with Type 1 / Type 2 topology migrations.

Surfaces: **torus**, **ellipsoid**, **double torus**, **Banchoff-Chmutov order 4**.

**State of play.** The torus is the maintained surface. **Approach B is the
working Phase 1** — valid, connected, equal-area partitions to **N = 1000**, in
minutes where PGD took days, and it is the only route that works on the implicit
surfaces (PGD results there are *invalid*, not merely unvalidated). PGD remains
the incumbent baseline up to N = 300 and is what the comparisons are anchored on.
**Phase 2 is now the bottleneck** — 83–89% of end-to-end cost at high N.

## Build & run

```bash
pyenv activate ringtest-3.9      # see .python-version
pip install -e ".[all]"          # core + PyVista + IPOPT + scikit-image

# --- Phase 1, approach B (START HERE) -------------------------------------
# Against a PGD anchor: ladder/N/seed come from the anchor's experiment.yaml,
# and the driver refuses to score unless the final vertex counts match.
python scripts/run_mbo_arm.py --anchor-run 'results/run_20260709_081548*' --phase2
python scripts/run_mbo_arm.py --anchor-run 'results/run_20260709_081548*' --init-only --phase2  # attribution control
# EXPLORATORY: no PGD baseline exists (e.g. a new N). No mesh assertion, no
# scored perimeter. --levels truncates for smoke tests.
python scripts/run_mbo_arm.py --config parameters/torus_400part_mbo.yaml --levels 2
python scripts/run_mbo_arm.py --config parameters/torus_400part_mbo.yaml --phase2

# --- Phase 1, PGD (the incumbent) -----------------------------------------
# THE representative config: N=100, λ=5.1, seed 84172851, 5 levels
# (finest 348×328 = 114,144 verts). Produced the validated deliverable
# run_20260709_081548: all gates pass, worst cell 0.78%, perimeter 185.2546.
# Phase 1 wall ≈ 13.4 h.
python scripts/find_surface_partition.py --config parameters/torus_100part_coarse_seeded.yaml
python scripts/find_surface_partition.py --config parameters/torus_100part_coarse_seeded_3lvl.yaml  # 3 levels
python scripts/find_surface_partition.py --config <cfg> --profile   # writes solution/timing_profile.yaml
python scripts/find_surface_partition.py --config <cfg> --resume-from <checkpoint_levelNN.h5>

# --- Balanced readout (optional Phase 1 → Phase 2 bridge, PGD at high N) ---
# Writes <run_root>/readout/<campaign>/solution_balanced.h5; source untouched.
python scripts/balanced_readout.py --solution <solution.h5>
python scripts/balanced_readout.py --solution <solution.h5> --no-repair   # dual shifts only

# --- Phase 2 --------------------------------------------------------------
python scripts/refine_perimeter.py --solution <solution.h5> --config <cfg> [--profile]
python scripts/refine_perimeter.py --solution <solution.h5> --max-iterations 10 --method ipopt

# --- Export (what downstream repos consume) -------------------------------
python scripts/export_partition.py --solution <iteration_NNN_*.h5> --config <cfg> \
    --output results/<run>/partition/<name>.h5 --force-finalised
python scripts/check_deliverables.py          # verify docs/reference/deliverables.yaml

# --- Visualization and analysis -------------------------------------------
python scripts/visualize_partition_fast.py --solution <solution.h5>   # production viewer
python scripts/visualize_partition.py --solution <solution.h5>        # debugging viewer
python scripts/optimization_analyzer.py --results-dir results/<run_dir>
```

Parameter sweeps (`sweep/`) and the UPPMAX Pelle cluster scripts (`cluster/`)
have their own rule files; both are independent of the core pipeline.

## Testing

**There are no pytest unit tests.** `testing/` holds CLI diagnostics — `pytest`
will discover the `test_*.py` files and collect zero test functions.

```bash
python testing/check_fragmentation.py <solution_or_checkpoint.h5>   # all 3 validity gates
python testing/test_mbo_auction.py [--negative-controls]            # approach B, G1-G8
python testing/test_arm_harness.py                                  # the evaluation harness
python testing/watch_level_gates.py <run_dir>                       # grade a LIVE run per level
```

The three gates run **only on the final solution**, so a run still on the mesh
ladder reports none of them — `check_fragmentation.py` is how you grade a
per-level checkpoint. `scripts/debug_archive/` holds retired diagnostics.

## Repository map

```
src/
├── mesh/            TriMesh (P1 FEM M, K, lumped v), topology, interpolation
├── surfaces/        SurfaceProvider ABC, torus, ellipsoid, implicit (marching
│                    cubes), double_torus, banchoff_chmutov, factory
├── optimization/    pgd_optimizer (Phase 1), perimeter_optimizer (Phase 2),
│                    projection, initialization, exceptions
├── partition/       mbo_auction (approach B), arm_harness, balanced_readout,
│                    find_contours (the 3 gates), contour_partition, perimeter /
│                    area / steiner calculators + vectorized counterparts
├── migration/       Type 1 and Type 2 topology switches
├── pipeline/        relaxation (Phase 1), pipeline_orchestrator (Phase 2), io
├── export/          rep3_builder, writer  (schema 1.1 torus / 2.0 general)
├── visualization/   plot utils, cell colouring, offscreen screenshots
├── h5util.py        gzip for one-hot payloads ONLY — see run-layouts rule
├── profiling.py     opt-in timing accumulators (Phase 1 and Phase 2)
└── logging_config.py
scripts/    CLI entry points (see Build & run)
testing/    CLI diagnostics and gates
parameters/ experiment configs (51) — each carries its own header comment
sweep/      parameter sweep orchestrator and analyzers
cluster/    UPPMAX Pelle submission scripts
docs/       see docs/README.md
```

## Documentation

`docs/README.md` is the index — the folder taxonomy and a where-to-look table.
Eight folders: `math/` (derivations), `experiments/` (measured studies),
`reference/` (permanent explanations), `plans/`, `explanations/` (plain
language), `presentations/`, `guides/`, `papers/`.

Four to know by name:

| | |
|---|---|
| `docs/reference/winner_take_all_partition_gap.md` | The central high-N document: why PGD's readout fails, and §4c's locality criterion any replacement must pass |
| `docs/reference/PHASE1_HIGHN_APPROACHES_ABCDE.md` | The standing taxonomy A/A2/B/C/D/E — **read before proposing anything in this space** |
| `docs/experiments/08-mbo-auction-dynamics/` | Approach B's measured results |
| `docs/reference/deliverables.yaml` | Every exported partition, verified by script |
| `docs/reference/DOWNSTREAM_CONSUMERS.md` | Who consumes the exports and what must not change |

To create a document, use the **`/new-doc`** skill. To write one, see
`.claude/rules/docs-authoring.md`.

## Key classes

| Class | Module | Role |
|---|---|---|
| `TriMesh` | `src/mesh/tri_mesh.py` | Mesh with P1 FEM mass `.M`, stiffness `.K`, lumped mass `.v` |
| `SurfaceProvider` | `src/surfaces/base.py` | ABC for all surfaces; `build()` returns a `TriMesh` |
| `ProjectedGradientOptimizer` | `src/optimization/pgd_optimizer.py` | Phase 1 PGD: Γ-convergence energy under sum-to-one + equal-area |
| `run_mbo_ladder` / `MBOConfig` | `src/partition/mbo_auction.py` | Approach B: volume-constrained threshold dynamics (module-level functions, not a class) |
| `apply_balanced_readout` | `src/partition/balanced_readout.py` | Equal-area, connected extraction; `solve_dual_offsets` is the shared assignment primitive for A, B and C |
| `PerimeterOptimizer` | `src/optimization/perimeter_optimizer.py` | Phase 2: perimeter under equal-area (SLSQP / trust-constr / IPOPT) |
| `ContourAnalyzer` | `src/partition/find_contours.py` | HDF5 → indicators → boundary topology; hosts the three gates |
| `PartitionContour` / `VariablePoint` | `src/partition/contour_partition.py` | Phase 2 state: VPs on mesh edges with a λ parameter |
| `MigrationOrchestrator` | `src/migration/migration_orchestrator.py` | Detect → execute topology switches |
| `RelaxationConfig` / `RefinementConfig` | `src/pipeline/` | Config dataclasses; `from_yaml_dict()` reads sectioned or flat YAML |

## Data flow

1. **Phase 1** → a labelling, written to `solution/` in the Phase 1 HDF5 schema,
   config copied verbatim to `experiment.yaml` at the run root. `run_relaxation`
   then evaluates **three validity gates** on the final solution — `detect_dormant_cells`
   (dead: wins no vertex; weak: peak density < 0.5), `detect_area_imbalance`
   (> 5% off the equal-area target), `detect_disconnected_cells` (territory splits
   into ≥ 2 components on the surface, stray pieces > 1% of target). Each logs a
   warning and lands in `solution/metadata.yaml`. **A fragmented cell must not be
   handed to Phase 2** — it yields multi-loop contours.
2. **Bridge** — `ContourAnalyzer` turns densities into indicators (argmax) and
   places variable points on crossed edges. At high N under PGD, insert the
   **balanced readout** first and point Phase 2 at its output.
3. **Phase 2** — optimize → detect migrations → export checkpoint → migrate, per
   campaign under `refinement/{campaign}/`.
4. **Export** → the link-list schema consumed by downstream repos.

Run directory layouts (three distinct genres) are in `.claude/rules/run-layouts.md`.

## HDF5 formats

**Phase 1 solution** — datasets `x_opt`, `x0`, `vertices`, `faces`; attrs
`n_partitions`, `surface`, `completed_levels`, `lambda_penalty`, `seed`, and
`var1`/`var2` (final level resolution, required by the exporter).

**Phase 2 checkpoint** (`iteration_NNN_YYYYMMDD_HHMMSS.h5`) — datasets
`lambda_parameters`, `vp_edges`, `indicator_functions`; attrs
`n_variable_points`, `n_cells`, `final_perimeter`, `iteration_number`,
`pending_migration`, `base_solution_path`; group `optimization_info/`. The
filename encodes only the iteration and its creation time — all experiment
context comes from the parent run and campaign directories.

## The λ convention (critical)

A variable point sits on a mesh edge at `x = λ·v[e0] + (1−λ)·v[e1]`, with edges
normalized so `e0 < e1`. So **λ = 1 is at the SMALLER vertex index**, λ = 0 at
the larger. λ within `boundary_tol` of 0 or 1 triggers a Type 1 migration.

## Style and conventions

- **Black**, line length 88, target Python 3.9.
- Relative imports inside `src/` (`from ..mesh.tri_mesh import TriMesh`);
  scripts add the repo root to `sys.path`.
- snake_case functions, PascalCase classes; mathematical variables keep paper
  notation (ε, λ, M, K, v).
- **No print statements in library code** — use `get_logger(__name__)` from
  `src/logging_config.py`. Scripts may print for user-facing messages.
- Dataclasses for config and result types.
- Comments only for non-obvious logic, mathematical references, or critical
  conventions. No narration comments.
- **PyVista is not in `requirements`** — install it separately for the viewers.

## Standing rules

- **Read a run's own `experiment.yaml`, never `parameters/`.** Configs drift
  after the fact; reconstructing a mesh from the live config lands on a
  different mesh than the run you are comparing against.
- **Perimeter is `Σ_k Per(cell_k)`, so every interface is counted twice.**
  `label_boundary_length` counts each once, so the two differ by a factor of ~2
  *by definition* — never compare them directly.
- **`run_time_seconds` and `summary.total_wall_s` are not wall times.** Both
  under-report by 3–19×. See `.claude/rules/run-layouts.md` before quoting any
  timing.
- **Exported partitions are consumed by external repos.** Never change the
  export schema, rename a deliverable or delete its run without checking
  `docs/reference/DOWNSTREAM_CONSUMERS.md` first.
- **`results/` is gitignored and per-worktree** — enumerate every working
  directory before claiming a run does not exist.
- **A high-N export needs `--force-finalised`**, because the Phase 2 plateau
  leaves `pending_migration=True` on the best iterate.
- **Any number in a document must be regenerable by a committed script.**
  Reviewer-supplied figures and inline session Python are not sources.
- **Do not re-add territory-aware relaxation.** Implemented, measured, removed —
  a net regression. `.claude/rules/phase1-pgd.md` has the reason.
- Both sectioned and flat experiment YAML are accepted by `from_yaml_dict()`.
- Do not confuse the A–E taxonomy with the P1–P5 enumeration in
  `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md`.

## Topic rules (`.claude/rules/`)

Each loads only when you read a file it matches.

| Rule | Covers |
|---|---|
| `approach-b-mbo.md` | The τ window, freeze ratio, cost scaling, over-merge instruments, `mbo:` config block, G4 |
| `phase1-pgd.md` | λ window, `init_method`, the line-search floor, dying levels, structure trigger, soft-area constraint, territory-aware rejection |
| `balanced-readout.md` | The dual shifts and connectivity repair, `solve_dual_offsets` as the shared A/B/C primitive, `max_repair_sweeps` reproduction |
| `phase2-refinement.md` | λ convention, migrations, VP soft deletion, the migration-cycling plateau, why the IPOPT solver is the wrong target |
| `surfaces.md` | Adding a provider, marching-cubes stiffness conditioning, why λ does not transfer between surfaces |
| `run-layouts.md` | The three run genres, export, the four measurement traps, HDF5 compression |
| `sweeps.md` / `cluster.md` | Sweep orchestration; UPPMAX Pelle |
| `docs-authoring.md` | The reproducibility rule, LaTeX and Marp gotchas, doc sync |

## Maintaining this file

**Budget: 350 lines.** A change that pushes CLAUDE.md over the cap must evict
something in the same commit. `wc -l CLAUDE.md` is the check; a `PostToolUse`
hook (`.claude/hooks/claude-md-budget.sh`) reports it automatically.

### Route it before writing it

| What you learned | Where it goes |
|---|---|
| Changes what to do in **every** session | **this file**, as one imperative line |
| Changes what to do only when touching specific files | `.claude/rules/<topic>.md` with a `paths:` glob |
| A multi-step procedure done on request | a skill under `.claude/skills/` |
| A measurement, a result, a finding | `docs/experiments/` or `docs/reference/` — here, at most one imperative line and a pointer |
| A list of files, runs or numbers | a data file with a committed checker |
| Already in auto-memory | nowhere — memory loads every session too; do not duplicate it |

**The default is not this file.** Most things route elsewhere.

### Write the rule, not the evidence

An entry here is an **imperative**: what to do or not do, in one or two lines,
with a pointer to where the justification lives. If an entry needs a table, a run
ID, or a caveat paragraph, it is evidence — and evidence does not live here.

### Evict on these triggers

| When | This file keeps | The rest goes to |
|---|---|---|
| A plan is fully implemented | nothing | delete the plan, or move its lasting explanation to `docs/reference/` |
| An experiment report is written | ≤ 2 lines + pointer | the report |
| An approach is tried and rejected | the prohibition only | the report |
| A gotcha's evidence lands in `docs/` | the imperative + pointer | that doc |
| A rule fires only on specific files | its row in the rules table | `.claude/rules/` |
| A section becomes a list of files or numbers | a pointer | a data file + checker |

### Keeping documents in sync

When a code change adds, removes, renames or relocates anything described in
this file, a rule file, or a document under `docs/plans/` or `docs/reference/`,
**update that file in the same change** — routed by the table above, so accuracy
does not cost size. A fully-implemented plan should be deleted or folded into
`docs/reference/`. The one drift case this does not reliably catch is analysed
in `docs/reference/DOCUMENTATION_SYNC_RULE.md`.

Rule files carry a soft **250-line** budget and the same routing test.

## Dependencies

Core (`pip install -e .`): `numpy`, `scipy`, `pyyaml`, `matplotlib`, `h5py`, `tqdm`.
Optional groups: `[ipopt]` (cyipopt), `[viz]` (pyvista, colorcet), `[implicit]`
(scikit-image), `[all]`, `[dev]` (pytest, black, flake8).
