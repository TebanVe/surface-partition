# CLAUDE.md — Project Context for Claude Code

## Project Overview

**surface-partition** is a Python framework for computing minimal-perimeter partitions on closed triangulated surfaces. It implements the method from "Partitions of Minimal Length on Manifolds" (Bogosel & Oudet) using two phases:

1. **Phase 1 (Relaxation):** Γ-convergence energy minimization via Projected Gradient Descent (PGD) on nodal density functions, with multi-level mesh refinement.
2. **Phase 2 (Refinement):** Direct constrained perimeter minimization on extracted contour variable points, with automatic topology migrations (Type 1 and Type 2).

Surfaces implemented: **torus** (`TorusMeshProvider`), **ellipsoid** (`EllipsoidMeshProvider`), **double torus** (`DoubleTorusMeshProvider`), and **Banchoff-Chmutov order 4** (`BanchoffChmutovMeshProvider`).

**Surface scope — this project is torus-focused.** It began aiming at arbitrary
closed surfaces; it has since narrowed to the torus, and the work that matters
here (high-N validity, the λ window, the balanced readout, Phase 1 cost) is all
torus work. The ellipsoid, double-torus, and Banchoff-Chmutov providers and
their configs are **retained but unmaintained** — kept to seed a follow-on
project, not exercised since the energy-discretization fix (`6ff71a0`), and so
of unverified convergence. Treat any result from them as unvalidated until
re-run. Representative config: `parameters/torus_100part_coarse_seeded.yaml`.

## Build & Run

```bash
# Setup (uses pyenv — see .python-version for the environment name)
pyenv activate ringtest-3.9   # or: pyenv activate surface-partition
pip install -e .               # core only
pip install -e ".[all]"        # or: core + PyVista + IPOPT + scikit-image

# Phase 1: Γ-convergence relaxation
# START HERE. The representative configuration — N=100, λ=5.1, seed 84172851,
# seeded init, 5 levels (finest 348×328 = 114,144 vertices). This is the config
# that produced the validated N=100 deliverable run_20260709_081548:
# 0 dead / 0 weak / 0 imbalanced / 0 fragmented, worst cell 0.78% off target,
# exported perimeter 185.2546. Phase 1 wall ≈ 13.4 h (48,132 s, Mac mini M-series).
python scripts/find_surface_partition.py --config parameters/torus_100part_coarse_seeded.yaml

# Fast variant for smoke-testing — same λ and seed, 3 levels instead of 5
# (finest 224×212 = 47,488 vertices). Phase 1 wall ≈ 7.3 h (26,214 s).
# It is levels 0-2 of the run above, bit-for-bit (deterministic given the seed),
# so "fast" is relative: level 0 alone is 75% of it, because level 0 runs the
# full 30,000-iteration cap. For a quick pipeline check use torus_10part.yaml.
python scripts/find_surface_partition.py --config parameters/torus_100part_coarse_seeded_3lvl.yaml

# Enable timing profiling (writes solution/timing_profile.yaml with per-level breakdown):
python scripts/find_surface_partition.py --config parameters/torus_100part_coarse_seeded_3lvl.yaml --profile

# Minimal legacy smoke test (N=10 — well below the cell counts this project
# targets; use it to check the pipeline runs, not to judge quality). Note its
# finest level runs to the 25,000-iteration cap (~6 h); lower refinement_levels
# or max_iter for a quick check — the partition is valid several levels earlier.
python scripts/find_surface_partition.py --config parameters/torus_10part.yaml

# Other surfaces — LEGACY / UNMAINTAINED, unverified since the energy fix
# (6ff71a0). See "Surface scope" below before relying on these.
python scripts/find_surface_partition.py --config parameters/ellipsoid_6part.yaml
python scripts/find_surface_partition.py --config parameters/double_torus_10part.yaml      # requires .[implicit]
python scripts/find_surface_partition.py --config parameters/banchoff_chmutov_12part.yaml  # requires .[implicit]

# Balanced readout (optional Phase 1 → Phase 2 bridge; fixes the winner-take-all gap)
# Writes <run_root>/readout/<campaign>/solution_balanced.h5 in the Phase 1 schema.
# The source solution is never modified.
python scripts/balanced_readout.py --solution <path_to_solution.h5>
python scripts/balanced_readout.py --solution <path_to_solution.h5> --no-repair  # dual shifts only

# Approach B (Phase A) -- auction-dynamics MBO as a Phase 1 REPLACEMENT (experimental)
# The ladder, N and seed come from the ANCHOR RUN's experiment.yaml, so the arm and
# the control are provably on the same meshes; the driver refuses to score if the
# arm's final vertex count does not match the anchor's.
python scripts/run_mbo_arm.py --anchor-run 'results/run_20260709_081548*' --phase2
python scripts/run_mbo_arm.py --anchor-run 'results/run_20260709_081548*' --init-only --phase2  # attribution control
python testing/test_mbo_auction.py                      # gates G1-G8
python testing/test_mbo_auction.py --negative-controls   # NC1/NC4/NC5 + NC3-CAL

# Phase 2: Perimeter refinement (requires Phase 1 output)
python scripts/refine_perimeter.py --solution <path_to_solution.h5> --config parameters/torus_100part_coarse_seeded.yaml
# Or with CLI overrides:
python scripts/refine_perimeter.py --solution <path_to_solution.h5> --max-iterations 10 --method ipopt
# Enable timing profiling (writes timing_profile.yaml per campaign):
python scripts/refine_perimeter.py --solution <path_to_solution.h5> --config parameters/torus_100part_coarse_seeded.yaml --profile

# Visualization (all require pyvista)
# Production viewer — vectorized, handles fine meshes efficiently:
python scripts/visualize_partition_fast.py --solution <path_to_solution.h5>
# Original viewer — slower on fine meshes, useful for debugging:
python scripts/visualize_partition.py --solution <path_to_solution.h5>
# Migration debugging viewers (Type 1 / Type 2 topology switches):
python scripts/visualize_type1_vertex_collapse.py --solution <path_to_refined.h5> --region 2
python scripts/visualize_type2_triple_point.py --solution <path_to_refined.h5> --region 2

# Export finalised partition to link-list-torus HDF5 schema
python scripts/export_partition.py \
 --solution results/<run>/refinement/<campaign>/iteration_NNN_*.h5 \
 --config parameters/torus_100part_coarse_seeded.yaml \
 --output results/<run>/partition/torus_partition_<run-id>.h5
# If Phase 2 stalled in the migration-cycling plateau (pending_migration never
# clears), add --force-finalised to write finalised=True on the best iterate:
python scripts/export_partition.py \
 --solution results/<run>/refinement/<campaign>/iteration_NNN_*.h5 \
 --config parameters/torus_100part_coarse_seeded.yaml \
 --output results/<run>/partition/torus_partition_<run-id>.h5 --force-finalised

# Analysis (auto-includes relaxation_timing_profile.png when --profile was used)
python scripts/optimization_analyzer.py --results-dir results/<run_dir>

# Parameter sweeps (sweep/ directory — independent from core scripts)
python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml                      # local-sequential
python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode local-parallel --workers 4
python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode generate-only  # configs only
python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode collect        # scan & index

# Sweep analysis (reads experiment_index.yaml)
python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/
python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/ --metric final_energy

# Timing analysis (reads experiment_index.yaml timing fields; requires --profile runs)
python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/
python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/ --campaign ipopt_btol0.001_lbfgs30_hess
python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/ --phase relaxation  # Phase 1 PGD breakdown
```

## Testing

There are no pytest unit tests. The `testing/` directory contains CLI diagnostic tools:
```bash
python testing/test_migrations_debug.py --solution <path_to_refined.h5>

# Run all 3 Phase 1 validity gates (dormant / area-imbalance / connectivity) on a
# solution OR a per-level checkpoint. run_relaxation evaluates these only on the
# FINAL solution, so this is the way to check a run still on the mesh ladder — or
# one that died before the last level and never wrote metadata.yaml:
python testing/check_fragmentation.py <run_dir>/solution/checkpoint_level02.h5

# Validate the Phase 1 PGD serial optimizations (Changes A/B/C):
# Mode 1 — in-process projection-equivalence + gradient-reuse identity:
python testing/validate_pgd_optimizations.py --equivalence
# Mode 2 — compare two completed runs (same config+seed; one from main, one from branch):
python testing/validate_pgd_optimizations.py --compare \
    --baseline <main_run_dir> --candidate <branch_run_dir> --stage A
```
The `scripts/debug_archive/` directory contains archived diagnostic scripts.

## Code Architecture

### Directory Layout

```
src/
├── mesh/
│   ├── tri_mesh.py               # TriMesh: vertices, faces, P1 FEM mass (M) and stiffness (K) matrices, lumped mass (v)
│   ├── mesh_topology.py          # MeshTopology: edge-triangle adjacency for migration subsystem
│   └── interpolation.py          # Nearest-neighbor interpolation between refinement levels
├── surfaces/
│   ├── base.py                   # SurfaceProvider ABC: interface for all surface providers
│   ├── torus.py                  # TorusMeshProvider: structured torus mesh from (n_theta, n_phi, R, r)
│   ├── ellipsoid.py              # EllipsoidMeshProvider: parametric spherical-coord mesh
│   ├── implicit.py               # ImplicitSurfaceProvider: marching-cubes base class
│   ├── double_torus.py           # DoubleTorusMeshProvider: Bogosel & Oudet Figure 3
│   └── banchoff_chmutov.py       # BanchoffChmutovMeshProvider: Bogosel & Oudet Figure 4
├── optimization/
│   ├── pgd_optimizer.py          # ProjectedGradientOptimizer: Phase 1 PGD with Γ-convergence energy
│   ├── perimeter_optimizer.py    # PerimeterOptimizer + IPOPTProblemAdapter: Phase 2 constrained minimization
│   ├── projection.py             # Iterative constraint projection (sum-to-one, equal areas); random level-0 init
│   ├── initialization.py         # Seeded (Voronoi/farthest-point) level-0 initial condition
│   └── exceptions.py             # RefinementTriggered exception
├── partition/
│   ├── find_contours.py          # ContourAnalyzer: HDF5 → indicator functions → boundary topology
│   ├── balanced_readout.py       # Equal-area + connected extraction: OT dual shifts + connectivity repair
│   ├── arm_harness.py            # THE single evaluation harness for Phase 1 replacement arms (B/C) and the PGD control: writes arm labels in the Phase 1 schema, runs all 3 gates, runs Phase 2 as a subprocess against a PINNED campaign refinement.yaml, reports best-iterate perimeter + trajectory + censoring + wall time
│   ├── mbo_auction.py            # Approach B (Phase A): auction-dynamics MBO -- volume-constrained multiphase threshold dynamics as a Phase 1 REPLACEMENT. Diffuse `(M + tau K) y_k = M chi_k` (one prefactorized LU per level), then reassign by balanced thresholding `argmax_k [y_ik + psi_k]` via `solve_dual_offsets`. Hard-required init: one balanced C-scores assignment. `tau = min((4*h_mean)^2, (rho*R_cell)^2)`, rho=1.0 -- see the tau-window gotcha below
│   ├── contour_partition.py      # PartitionContour, VariablePoint, TriangleSegment
│   ├── perimeter_calculator.py   # Per-segment perimeter with analytical gradients
│   ├── area_calculator.py        # Per-cell FEM area with analytical gradients
│   ├── steiner_handler.py        # Steiner/triple-point perimeter + area contributions
│   ├── partition_arrays.py       # PartitionArrays: sparse Jacobian/Hessian sparsity for IPOPT
│   ├── vectorized_perimeter.py   # Fast vectorized perimeter evaluation
│   ├── vectorized_area.py        # Fast vectorized area evaluation
│   └── vectorized_steiner.py     # Steiner forward values + analytical first/second derivatives (FD reference retained)
├── migration/
│   ├── migration_orchestrator.py # MigrationOrchestrator: top-level detect → execute loop
│   ├── migration_detector.py     # Type 1 + Type 2 trigger detection
│   ├── migration_executor.py     # Execute migrations on partition state
│   ├── migration_types.py        # DetectionResult, MigrationResult, MigrationConfig dataclasses
│   ├── migration_utils.py        # Shared helpers (edge utilities, geometry)
│   ├── type2_migration_io.py     # Type 2 snapshot save/restore
│   ├── type2_migration_history.py   # Type 2 rollback history tracking
│   └── one_ring_rebuilder.py     # One-ring mesh topology rebuilding after Type 1 migration
├── pipeline/
│   ├── relaxation.py             # run_relaxation(): multi-level PGD pipeline (Phase 1)
│   ├── pipeline_orchestrator.py  # PipelineOrchestrator, RefinementConfig, derive_output_paths (Phase 2)
│   └── io.py                     # HDF5 loaders, detect_run_layout(), find_base_solution_path()
├── export/
│   ├── __init__.py              # public API: export_partition()
│   ├── rep3_builder.py          # builds subdivided mesh (Representation 3)
│   └── writer.py                # assembles and writes the export HDF5
├── visualization/
│   ├── plot_utils.py             # Matplotlib utilities
│   ├── partition_helpers.py      # Partition-specific viz helpers (cell coloring, VP/Steiner markers)
│   ├── cell_coloring.py          # Neighbour-distinct cell colors (graph coloring; glasbey/HSV palette)
│   └── partition_screenshots.py  # Offscreen multi-angle partition rendering (PyVista, optional)
├── profiling.py                  # ProfilingState (Phase 2) + RelaxationProfilingState (Phase 1): opt-in timing accumulators (stdlib only)
└── logging_config.py             # Logging setup, get_logger(), @log_performance decorator
scripts/
├── find_surface_partition.py     # Phase 1 CLI: Γ-convergence relaxation
├── refine_perimeter.py           # Phase 2 CLI: iterative perimeter refinement
├── balanced_readout.py           # Bridge CLI: balanced, connected readout of a Phase 1 solution
├── run_mbo_arm.py                # Phase A CLI: run approach B and score it through arm_harness. Reads the mesh ladder from the ANCHOR RUN's experiment.yaml (never `parameters/`) and asserts `arm_final_V == anchor_V` before scoring
├── optimization_analyzer.py      # Per-run analysis and plotting
├── visualize_partition_fast.py   # Fast partition viewer — production (PyVista, vectorized, neighbour-distinct cell colors)
├── visualize_partition.py        # Original partition viewer — debugging (PyVista)
├── visualize_type1_vertex_collapse.py  # Type 1 migration debugging viewer
├── visualize_type2_triple_point.py     # Type 2 migration debugging viewer
├── export_partition.py           # Export finalised partition to link-list-torus schema. `source_run_id` is taken from the nearest `run_*` ancestor of the checkpoint, not from `run_dir` — a readout-derived checkpoint sits one level deeper (`readout/{campaign}/refinement/{campaign}/`), so the old `parent.parent.parent` landed on the readout campaign name. `run_dir` itself is unchanged, so the default `--output` path still nests with the refinement output
└── debug_archive/                # Archived diagnostic scripts
testing/
├── README_testing.md                    # Test registry documentation
├── _hessian_test_utils.py               # Shared build_optimizer() helper for the harness below
├── test_sparse_jacobian_equivalence.py  # Sparse vs dense area-Jacobian equivalence
├── test_exact_hessian_vs_fd.py          # Analytical Lagrangian Hessian vs Richardson FD
├── test_exact_hessian_matvec.py         # Hessian-vector-product check (large meshes)
├── compare_hessian_modes.py             # L-BFGS vs exact-Hessian comparison + breakdown
├── test_steiner_gradient_analytical.py  # Analytical Steiner first derivatives vs FD
├── test_steiner_hessian_analytical.py   # Analytical Steiner second derivatives vs FD
├── test_steiner_degenerate_case.py      # Degenerate (>=120 deg) Steiner branch
├── test_migrations_debug.py             # Migration debug CLI
├── test_type1_triple_point_guard.py     # Type 1 triple-point safety-guard smoke test
├── test_type1_triple_point_overlap.py   # Type 1 one-ring / Steiner overlap smoke test
├── test_white_triangle_fix.py           # Zero-length-boundary rendering-fix smoke test
├── validate_pgd_optimizations.py        # Phase 1 PGD serial-opt (Changes A/B/C) equivalence + A/B speedup
├── test_soft_area_constraint.py         # A2 gates: simplex projection vs KKT, penalty gradient vs FD, flag-off bit-identity
├── calibrate_soft_area_mu.py            # A2: pick soft_area_mu for a config (force parity at the 5% area gate)
├── validate_struct_trigger.py           # Replay the structure-based refinement trigger offline against completed runs
├── test_disconnected_cells_detection.py # Phase 1 connectivity gate: detect_disconnected_cells split/speckle logic
├── check_fragmentation.py               # Run all 3 Phase 1 validity gates on any solution OR per-level checkpoint (mid-ladder)
├── test_balanced_assignment_solver.py   # Phase 0a gates: (1) approach A byte-identical after generalizing the solver; (2) foreign score scales (B's diffused y, C's -d^2) converge on the arms' ITERATED state. Bar = max(2x granularity, A's own dual stall at EQUAL budget). The 1.25x strong-reference term was removed: it was computed with the solver under test, so six crippled solvers passed it, including one returning psi=0. Gate 3 now asserts a null solver FAILS; gate 4 checks early stopping is safe inside an iterated loop
├── test_arm_harness.py                  # Phase 0b: harness correctness -- one-hot round trip, hardening a continuous field does not move the gates, granularity floor, censoring detection, area-gate-as-solver-fault
├── test_mbo_auction.py                  # Phase A gates G1-G8 + negative controls NC1/NC2/NC3-CAL/NC4/NC5 for approach B. Every gate has a failing counterpart: G1 mutates its own input, G7 rejects a 2x tau error that mass conservation is blind to, G4 refuses to certify on a frozen tail, NC5 must separate c=8 from c=4 or the finding is recorded as "no instrument here sees over-merge"
├── test_phase0d_shakedown.py            # Phase 0d: pushes the N=100 control's OWN labels through the whole harness and requires 185.2546144718457 back (bitwise on this machine, else <=0.01%). ~34 min; writes only to a scratch root
├── watch_level_gates.py                 # Poll a LIVE run and grade every per-level checkpoint as it lands (the pipeline keeps only the newest and deletes them all at the end, so this is the only way to see WHERE a defect first appears)
├── diagnose_neighbor_triggers.py        # Neighbor-trigger diagnostic
└── diagnose_white_triangles.py          # White-triangle diagnostic
parameters/                       # (selected — see the directory for the full set)
├── torus_100part_coarse_seeded.yaml      # ★ THE representative config: N=100, λ=5.1, seed 84172851, seeded, 5 levels (finest 348×328). Produced the validated deliverable run_20260709_081548 (all 3 gates pass, worst cell 0.78%, perimeter 185.2546). Phase 1 ≈ 13.4 h
├── torus_100part_coarse_seeded_3lvl.yaml # ★ Fast variant of the above for smoke tests: same λ/seed, 3 levels (finest 224×212)
├── torus_100part_coarse_seeded_softarea.yaml # A2 experiment arm: the ★ config + soft_area_constraint (control = run_20260709_081548, not re-run)
├── torus_30part_softarea.yaml    # A2 pre-flight at N=30 (cheap shakedown of the soft-area path)
├── torus_100part.yaml            # ⚠ KNOWN-BAD — λ=2.1 is below the N=100 working window, and its 142×136 mesh is the run_20260701_143238 "finer mesh made the runt worse" case. Use torus_100part_coarse_seeded.yaml
├── torus_10part.yaml             # Torus, 10 partitions — minimal legacy smoke test only (N=10 is unrepresentative); seeded init since 2026-08-10
├── torus_30part.yaml             # Torus, 30 partitions (parametric mesh; seeded init)
├── torus_30part_random_s*.yaml   # Torus N=30 random-init studies — random init is their experimental point; do not "fix"
├── torus_50part.yaml             # Torus, 50 partitions (parametric mesh; seeded init, 6 levels)
├── torus_150part / 200part / 300part *.yaml  # High-N production and probe configs (seeded; λ from the window — see the λ-window gotcha)
├── torus_300part_seeded_lam11p5_original_energy.yaml  # N=300 λ-window probe; λ=11.5 (between proven N=200 λ=11 and N=300 λ=12→9/300 imbalanced), seed 61803399, 5 levels (levels 4-5 via --resume-from) — the last low-λ test after all prior N=300 (λ 12/13/15) failed the equal-area check
├── torus_100part_subfloor_ladder.yaml   # Ladder experiment Stage 1 (RAN, hypothesis refuted): the ★ config with only the base mesh dropped to 60×52 (31 v/cell, matching N=300's 32). Result in docs/experiments/06-subfloor-ladder/
├── torus_300part_above_floor_ladder.yaml # Ladder experiment Stage 2 — **NOT run and not to be run**: it was gated on Stage 1 confirming fragmentation, which it refuted
├── torus_300part_rebased_ladder.yaml    # ⚠ **Probably under-corrected — do not run without re-deciding.** Would have re-based the N=300 ladder, but puts level 0 at 51 v/cell. That was dismissed as "still inside the dying regime" on a ~90 v/cell boundary that is **not established** (see the runs-or-dies gotcha below), so the dismissal is weaker than it was written: 51 v/cell is below every *controlled* dying point measured, but the target "~92 v/cell ⇒ V₀ ≈ 27,600 (base ~176×164)" inherits the same unestablished number. Kept as the record — see its header and the two ladder gotchas below
├── ellipsoid_6part.yaml          # Ellipsoid, 6 partitions — LEGACY/UNMAINTAINED (see Surface scope)
├── double_torus_10part.yaml      # Double torus, 10 partitions — LEGACY/UNMAINTAINED (implicit / marching cubes)
└── banchoff_chmutov_12part.yaml  # Banchoff-Chmutov order 4, 12 partitions — LEGACY/UNMAINTAINED (implicit / marching cubes)
sweep/                              # Parameter sweep tool (independent from core pipeline)
├── parameter_sweep.py            # Sweep orchestrator (grid/paired, local/parallel/generate/collect)
├── sweep_analyzer.py             # Experiment-wide analysis (heatmaps, line plots, convergence overlays)
├── timing_analyzer.py            # Scaling figures from timing_profile.yaml data (requires --profile runs)
└── parameters/
    ├── sweep_torus_lambda.yaml       # Sweep: lambda × seed for torus (grid strategy)
    └── sweep_double_torus_lambda.yaml  # Sweep: lambda × resolution for double torus (grouped grid)
cluster/
├── pelle_config.sh              # Shared Pelle configuration (project, venv, SLURM defaults)
├── submit_relaxation.sh         # Submit Phase 1 job to Pelle
├── submit_refinement.sh         # Submit Phase 2 job to Pelle
├── submit_sweep.sh              # Submit parameter sweep to Pelle (one job per combination)
└── cleanup_sweep_results.py     # Prune worst sweep runs, keeping the N best by perimeter
```

### Documentation (`docs/`)

The `docs/` tree is version-controlled and has five parts:

```
docs/
├── math/        ← LaTeX derivations of the quantities computed in the code
├── guides/      ← LaTeX user guides and professional documents (compiled PDFs)
├── experiments/ ← LaTeX measured studies (question→method→measurement→conclusion)
├── plans/       ← design plans for not-yet-implemented work
└── reference/   ← permanent explanatory docs (methodology, known issues, primers)
```

**`docs/math/`** — mathematical derivations written as LaTeX, compiled to PDF:

```
docs/math/
├── AUTHORING_GUIDE.md          ← how to add a new document (read this first)
├── Makefile                    ← master build: `make all`
├── shared/
│   ├── macros.tex              ← shared notation for all documents
│   └── references.bib          ← shared bibliography
├── 01-phase2-derivatives/      ← Phase 2 regular perimeter/area derivatives; Steiner forward values
├── 02-phase2-timing-profile/   ← empirical IPOPT callback timing profile
├── 03-analytical-steiner-derivatives/  ← analytical Steiner first/second derivatives
├── 04-phase1-timing-profile/   ← empirical Phase 1 PGD timing profile (projection bottleneck)
├── 05-phase1-nregion-scaling/  ← empirical wall-time scaling with number of regions
├── 06-phase1-energy-discretization/  ← Phase 1 Γ-convergence energy: Dirichlet term, corrected double well (q=u(1-u)), Modica–Mortola limit, crispness penalty
└── 07-phase1-wta-balance/  ← Phase 1 winner-take-all balance term: soft territory, balance penalty + gradient, discrete-area trim, six structural properties, Γ-consistency, γ calibration. **NOT ADOPTED** — the term was implemented, measured, and removed (see "Territory-Aware Relaxation" below); the derivation is kept so the rejection is checkable
```

Each `NN-slug/` directory holds `main.tex` and the compiled `main.pdf`.
LaTeX build artifacts (`*.aux`, `*.bbl`, …) are ignored via
`docs/math/.gitignore`; `*.tex`, `*.bib`, `Makefile`, `*.md`, and the
`main.pdf` outputs are tracked.

**`docs/guides/`** — user guides, onboarding documents, and professional
technical documents that are not purely mathematical derivations:

```
docs/guides/
├── Makefile                    ← master build: `make all`
├── shared/
│   └── preamble.tex            ← shared packages and styles (listings, tcolorbox, etc.)
└── 01-pelle-user-guide/        ← step-by-step guide for running on UPPMAX Pelle
    ├── main.tex
    └── main.pdf
```

Guides use the same `NN-slug/` numbering convention as math documents.
`docs/guides/.gitignore` suppresses LaTeX build artifacts; sources and PDFs
are tracked. Build with: `make -C docs/guides/NN-slug` or `make -C docs/guides all`.

**`docs/experiments/`** — LaTeX **measured studies**: the empirical results of a
`question → method → measurement → conclusion` study whose numbers come from
running the code on specific inputs (convergence behaviour, error distributions,
failure-mode forensics). Distinct genre from a math derivation (`docs/math/`) or a
standing explanation (`docs/reference/`) — a measured study often *pairs with* a
reference doc. Same `NN-slug/` LaTeX system, reusing the math shared macros; every
report **must** open with a provenance block (date, source run(s) under `results/`,
producing script, library versions, a numerical anchor) and carry a status label
(**measured** / **partial** / **planned**). Figures are vector `fig_*.pdf` produced
by a committed `make_figures.py` beside the report; `docs/experiments/.gitignore`
suppresses build artifacts but tracks `main.pdf` + `fig_*.pdf`. Build with
`make -C docs/experiments/NN-slug` or `make -C docs/experiments all` (needs
`latexmk`; LaTeX at `/Library/TeX/texbin`). Reports:
`01-winner-take-all-partition-gap/` (the high-N runt failure, measured under the
buggy energy) and `02-corrected-energy-highn-validation/` (its post-fix resolution:
runt 22.5%→0.8%, Phase 2 −13.6%, random-init trap) — both pair with
`docs/reference/winner_take_all_partition_gap.md`; and
`05-soft-area-constraint/` (measured: the soft equal-area constraint rejected at N=100 — 32× faster and 0 dead cells, but 3–4/100 fragmented cells and a collapsed line search at every level; confirms §9b's locality criterion as a *predictive* screening test. Its **exact-at-level-0 hybrid survives**: 6.83× with all gates passing and Phase 2 within +0.246% of the control — but only via the balanced readout, which it requires), and
`04-territory-aware-highn-validation/` (measured/partial: the WTA balance term drives the
N=200 bad-seed runt from −34% to 0 imbalanced cells by level 1, `run_20260717_102306`; plus
the no-refinement-trigger diagnosis — the trim removes the energy plateau — and the coarse-level
resolution-floor finding. **The approach was rejected on its side effects and the code
removed** — the report stands as the measurement behind that call); and
`06-subfloor-ladder/` (measured: reproduces N=300's 32-verts/cell level-0 condition
at N=100 by dropping only the base mesh to 60×52. **The pre-registered hypothesis is
refuted** — 0 fragmented at every level, so N=300's cells 274/290 have another cause
and the prepared Stage-2 config was *not* run. But a starved level 0 manufactures a
**permanent** runt: worst cell 46.80% → 15.92% across 37× more vertices, never
crossing the 5% gate, vs the control's 0.78%. Also corrects the stall guard — see the
"under-resolved level" gotcha below). and
`07-phase0-shared-harness/` (measured: Phase 0 of the B/C replacement programme -- qualifying the measurement instrument. The harness reproduces the validated N=100 Phase 2 perimeter 185.2546144718457 **bitwise**, which simultaneously proves hardening an arm's labels costs nothing and that Phase 2 is bit-deterministic here; the shared assignment solver needed **no new algorithm** (the auction/Sinkhorn conclusion was a fixture artefact); gate 2 passes 18/18 and is shown to discriminate (the pre-0a solver scores identically to doing nothing); and at equal footing the arms sit at **parity +/-10%** with the incumbent, not clear of it. Also catalogues **five** occasions on which a measurement artefact was read as a result about the method, and the two practices that caught them). See
`docs/experiments/README.md`. (Slot 03 is
`03-dual-projection-verification`, on `feat/newton-projection`.)

**`docs/plans/`** — design plans for work not yet implemented (e.g. the
mesh-cleanup tool). The **active forward plan** is
`PHASE1_BC_REPLACEMENT_PLAN.md` — the shared balanced-assignment solver and
evaluation harness (Phase 0), then approach C, then approach B, with
pre-registered falsifiers anchored on the N=100 deliverable (Phase 2 perimeter
185.2546144718457). Read it together with
`docs/reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`.

**The two valid N=300 partitions** — the only ones that pass all three gates —
are `run_20260806_123326` (V=47,488, worst cell 1.63%, Phase 2 **323.3192**) and
`run_20260808_191030` (V=114,144, worst cell 0.64%, Phase 2 **322.9622**), both
λ=11.5 seed 61803399, both via `readout/dualshift_gate0.05_repair/`. They were
produced in the sibling worktree and **copied into this checkout on 2026-08-15**
(12.6 GB, verified byte-identical). ⚠ **`results/` is gitignored, so each git
worktree has its own and none of it is in version control** — enumerate *both*
working directories before claiming a run does not exist. The plan's v1 got this
wrong and built its central section on the mistake.

**`docs/reference/`** — permanent explanatory documents: topology-switch
methodology, scalability analysis, the optimization-methods primer, and
recorded known-issue investigations. Includes
`PHASE1_HIGHN_APPROACHES_ABCDE.md` — **the standing taxonomy of high-N
approaches: A / A2 / B / C / D / E**, with status and what each still needs; the
source proposal is archived verbatim beside it as
`phase1_highn_proposal_ABCDE_original.md`. Read it before proposing anything in
this space. Three things it fixes: **A2 is a sub-branch of A, not a peer of B and
C** (so A2's rejection says nothing about them); **B (auction-dynamics MBO, rank
2) and C (capacity-constrained geodesic Lloyd, rank 4) were fully specified,
falsifiers included, and are gated-open** — A2's rejection met their gate on
2026-08-12; and **everything done on 2026-08-14 sits inside D(iii)**, which that
taxonomy ranks last of five and calls "a supporting lever, never the fix" — a
judgement the measurements since have corroborated. Do not confuse A–E with the
separate P1–P5 enumeration in `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md`.

**Adding a new math document**: follow `docs/math/AUTHORING_GUIDE.md`.  It
specifies the directory naming convention, the `main.tex` template, all
available macros from `shared/macros.tex`, bibliography keys, and the scope
policy (only derive what is currently implemented — not planned features).

**Adding a new guide**: create `docs/guides/NN-slug/` with a `main.tex`
that begins with `\input{../shared/preamble}` and a `Makefile` copied from
an existing sibling.  Update `docs/guides/Makefile` to add the slug to the
`DOCS` variable.

To rebuild any PDF: `make -C docs/math/NN-slug` or `make -C docs/guides/NN-slug`.

**Creating a new document**: use the `/new-doc` skill — it classifies the
document (plan / reference / math / guide) and supplies the correct template.

### Keeping Documentation in Sync

Documentation must track the codebase. Two standing rules:

- **`docs/` sync** — When a code change is motivated by, or invalidates, a
  document under `docs/plans/` or `docs/reference/`, update that document in
  the same change. For a plan: advance its phase status and fold in findings
  from implementation (constraints, performance results, design decisions not
  in the original plan); a fully-implemented plan should be deleted or have its
  lasting explanation moved to `docs/reference/`. For a reference doc: correct
  whatever the change made inaccurate.
- **CLAUDE.md sync** — When a change adds, removes, renames, or relocates
  anything CLAUDE.md describes — a script, a config file, a public class, a
  directory, a CLI flag, a convention, a dependency — or resolves a documented
  gotcha, update CLAUDE.md in the same change so it never drifts.

### Run Output Layout (Structured)

Each Phase 1 run creates a structured directory under `results/`:

```
results/run_{timestamp}_surf{surface}_npart{N}_v1..._v2..._lam{λ}_seed{S}/
├── experiment.yaml               # Verbatim copy of the input config (reproduction recipe)
├── solution/
│   ├── surface_part{N}_surf{surface}_v1..._v2..._lam{λ}_seed{S}_{timestamp}.h5
│   ├── checkpoint_level{L}.h5    # Per-level resume checkpoint (newest only; deleted once the run finishes)
│   └── metadata.yaml             # Derived runtime results (mesh stats, timings, file paths)
├── traces/
│   ├── pgd_part{N}_v1{label}{n1}_v2{label}{n2}_level{L}_summary.out
│   └── pgd_part{N}_v1{label}{n1}_v2{label}{n2}_level{L}_internal_data.hdf5
├── readout/                      # Balanced-readout campaigns (optional bridge stage)
│   └── dualshift_gate0.05_repair/
│       ├── solution_balanced.h5  # Phase 1 schema + psi, labels_final, labels_source
│       ├── readout.yaml          # Config snapshot (reproduction recipe)
│       ├── readout.log
│       ├── metadata.yaml         # All three gates at each stage (source / shifted / repaired)
│       └── refinement/           # Phase 2 campaigns refined FROM this readout nest here,
│           └── {campaign}/       # not in the run-root refinement/ (same campaign name would
│                                 # otherwise interleave iterates from two different inputs)
├── refinement/
│   ├── slsqp_btol0.001/
│   │   ├── iteration_001_20260410_120523.h5
│   │   ├── refinement.yaml
│   │   └── refinement.log
│   └── ipopt_btol0.001_lbfgs20_hess/
│       ├── iteration_001_20260410_131042.h5
│       ├── iteration_002_20260410_131215.h5
│       ├── refinement.yaml
│       ├── refinement.log
│       └── timing_profile.yaml   # written only when --profile is passed
├── analysis/
│   ├── refinement_optimization_metrics.png
│   ├── constraint_evolution.png
│   └── energy_components.png
└── logs/
    └── relaxation.log
```

Each refinement campaign directory under `refinement/` is named by its
differentiating parameters via `build_campaign_name()`:
- Base: `{method}_btol{boundary_tol}`
- IPOPT extras (non-default only): `_lbfgs{N}`, `_hess`, `_bestiter`, `_partial`
- Distance (non-default only): `_midpoint` or `_dist{value}`

Each campaign contains a `refinement.yaml` config snapshot for reproducibility
and a `refinement.log` with full Phase 2 logs.

Layout detection (`detect_run_layout()` in `src/pipeline/io.py`) supports
both this structured layout and the legacy flat layout for backward
compatibility with older result directories.

### Experiment Directory Layout (Parameter Sweeps)

When using `parameter_sweep.py`, runs are grouped by experiment identity
(`{surface}_npart{N}`) rather than by sweep invocation:

```
results/torus_npart10/
├── experiment_index.yaml                      # auto-maintained index of all runs
├── run_20260413_120100_..._lam0.5_seed42/     # from sweep "lambda-coarse"
│   ├── experiment.yaml
│   ├── solution/
│   ├── traces/
│   ├── analysis/                              # auto-generated plots + screenshots
│   └── logs/
├── run_20260413_120200_..._lam1.0_seed42/
│   └── ...
├── sweeps/                                    # provenance: which sweeps ran here
│   ├── 20260413_120000_lambda-coarse.yaml     # copy of sweep spec
│   ├── 20260413_120000_lambda-coarse_summary.csv
│   └── 20260413_120000_lambda-coarse_run001.yaml  # per-run generated configs
└── analysis/                                  # experiment-wide analysis plots
    ├── heatmap_perimeter.png
    ├── line_perimeter.png
    ├── convergence_overlay.png
    └── sensitivity_perimeter.png
```

`experiment_index.yaml` is the central index listing every run with its
parameters, status, and key metrics (perimeter, final_energy, initial_N,
final_N, converged, total_iterations). Perimeter is the primary comparison
metric because it is resolution-independent (unlike energy, which is
ε-dependent).

When runs have been profiled with `--profile`, `--mode collect` also extracts
timing scalars into each run entry: `n_cells`, `n_active_vps`, `n_triple_points`,
and per-campaign `timing_*` fields (total wall time, IPOPT iter count, per-callback
% breakdown, Steiner recomputation totals). Phase 1 `--profile` runs additionally
yield `relax_timing_*` fields (total wall time, per-callback % breakdown, mean
backtracks / projection inner iters, and the per-level list) read from
`solution/timing_profile.yaml`. These fields are consumed by
`sweep/timing_analyzer.py` to produce scaling figures (`--phase relaxation` for
the Phase 1 breakdown).

### Key Classes and Their Roles

| Class | Module | Purpose |
|-------|--------|---------|
| `TriMesh` | `src/mesh/tri_mesh.py` | Triangle mesh with P1 FEM mass (M) and stiffness (K) matrices. Properties: `.M`, `.K`, `.v` (lumped mass = row-sum of M). Supports R2 and R3. |
| `MeshTopology` | `src/mesh/mesh_topology.py` | Edge-triangle adjacency structures needed by migration subsystem. |
| `TorusMeshProvider` | `src/surfaces/torus.py` | Builds structured torus TriMesh from (n_theta, n_phi, R, r). Supports refinement increments. |
| `EllipsoidMeshProvider` | `src/surfaces/ellipsoid.py` | Parametric ellipsoid via spherical-coord grid with polar cap triangles. |
| `ImplicitSurfaceProvider` | `src/surfaces/implicit.py` | Abstract base for zero-level-set surfaces; uses `skimage.measure.marching_cubes`. |
| `DoubleTorusMeshProvider` | `src/surfaces/double_torus.py` | Double torus: `(x(x-1)²(x-2)+y²)²+z²=0.03` (Bogosel & Oudet Figure 3). |
| `BanchoffChmutovMeshProvider` | `src/surfaces/banchoff_chmutov.py` | Banchoff-Chmutov order 4: `T4(x)+T4(y)+T4(z)=0` (Bogosel & Oudet Figure 4). Keeps largest connected component. |
| `ProjectedGradientOptimizer` | `src/optimization/pgd_optimizer.py` | Phase 1 PGD. Energy = ε·u^T·K·u + (1/ε)·q^T·M·q with q=u(1-u) (the double-well ∫u²(1-u)²) + λ·penalty. Constraints: partition sum-to-one, equal areas. The interface term was previously mis-discretized as `u²(1-u)²` (a typo copied from the paper, making the coded well ∫u⁴(1-u)⁴ with an inconsistent gradient); **fixed** in commit `6ff71a0` and validated at N=30/N=100. The corrected (steeper) well requires `init_method: seeded` — random init now traps in the symmetric state. See `docs/reference/phase1_energy_discretization_bug.md`, `docs/math/06-phase1-energy-discretization/`, `docs/experiments/02-corrected-energy-highn-validation/`. **Soft continuous equal-area constraint (`soft_area_constraint`, `soft_area_mu`; default OFF, byte-for-byte backward compatible):** replaces the exact equal-area constraint with a quadratic penalty `P = (μ/2)·Σ_k((v·u_k−Ā)/Ā)²` (gradient `(μ/Ā²)·v_i·(v·u_k−Ā)`), leaving sum-to-one + box — a closed-form per-vertex simplex projection instead of the iterative alternating projection that is 93.3% of Phase 1 wall time. See "Soft Continuous Equal-Area Constraint" below. |
| `PerimeterOptimizer` | `src/optimization/perimeter_optimizer.py` | Phase 2. Minimizes total perimeter (regular + Steiner) subject to equal cell areas. Supports SLSQP, trust-constr, IPOPT. |
| `IPOPTProblemAdapter` | `src/optimization/perimeter_optimizer.py` | Adapts PerimeterOptimizer for cyipopt interface. Optional best-iterate tracking and exact Hessian. |
| `ContourAnalyzer` | `src/partition/find_contours.py` | Loads HDF5 solution, computes indicator functions (winner-take-all), extracts boundary triangles and topology. |
| `BalancedReadoutConfig` / `apply_balanced_readout` | `src/partition/balanced_readout.py` | Optional Phase 1 → Phase 2 bridge closing the winner-take-all gap **at extraction time**, in two stages. (1) **Dual shifts:** replaces `argmax_k u_ik` with `argmax_k [log u_ik + ψ_k]`, the N per-cell offsets ψ solving the semi-discrete OT dual by subgradient ascent so every cell's discrete area hits target — balanced to one-vertex granularity **at any N** (tail-immune, unlike any soft/variance-reducing mechanism). Raising ψ_k grows cell k outward from its core, so the correction is *local* — the property the discrete-area trim lacked, which is why the trim manufactured islands and this does not. (2) **Connectivity repair:** stray components are absorbed by the neighbour sharing the longest boundary, then equal areas are restored by single-vertex boundary transfers (improving iff `T_donor − T_receiver > v_i`, ranked by `log u_ir − log u_id`), each gated by an exact articulation check so no move disconnects a donor. Both invariants hold by construction on exit. Reports its own strain (`n_moves`, `n_blocked_by_connectivity`, `sweeps_used`, `hit_sweep_cap`) so a run beyond what repair can fix says so. **`solve_dual_offsets` is the shared balanced-assignment primitive for approaches A, B and C** — it takes any additive per-vertex score matrix (A: `log u`; B: diffused `y`; C: `−d²`) and touches it only through `argmax(scores + psi)`. Correctness is scale-free but *convergence* is not, so `normalize_scores` (default **off**, keeping A byte-identical) divides by a robust assignment-margin scale, applies `normalized_eta0` (2.0, vs `dual_eta0` 0.5 which was tuned for log-density scale) and rescales ψ back. Measured at torus N=300 (`run_20260806_123326`, λ=11.5): **10 imbalanced / worst 36.15% / 2 fragmented → 0 / 1.63% / 0**, +1.88% label-boundary length, 1840 vertices relabeled, ~17 s. Re-verified 2026-08-15 from `readout/dualshift_gate0.05_repair/solution_balanced.h5`: **all three gates pass, worst cell 1.63%** — matching the figure above exactly. Its 19-iteration Phase 2 reaches perimeter **323.3192**. **`max_repair_sweeps` default is 200** (raised from 50 on 2026-08-15). It is a ceiling, never a schedule — the sweep loop breaks as soon as no candidate move improves — so any readout that converged in under 50 sweeps is byte-identical across the change; only readouts that reported `hit_sweep_cap: true` differ, and for those the extra sweeps can only move further toward equal area. **Reproducing an archived capped readout requires `--max-repair-sweeps 50`**; each campaign's `readout.yaml` records the value actually used. |
| `PartitionContour` | `src/partition/contour_partition.py` | Central data structure: list of `VariablePoint`s (edge + λ parameter), `TriangleSegment`s, indicator arrays, Steiner bookkeeping. |
| `VariablePoint` | `src/partition/contour_partition.py` | Point on mesh edge at position x = λ·v_start + (1-λ)·v_end. λ∈[0,1]. λ=1 → at smaller vertex index. Has `active` flag for soft deletion. |
| `PerimeterCalculator` | `src/partition/perimeter_calculator.py` | Computes per-segment perimeter contributions with analytical gradients. |
| `AreaCalculator` | `src/partition/area_calculator.py` | Computes per-cell FEM area with analytical gradients. |
| `SteinerHandler` | `src/partition/steiner_handler.py` | Manages Steiner/triple-point perimeter and area contributions for triangles where 3+ cells meet. |
| `PartitionArrays` | `src/partition/partition_arrays.py` | Pre-computes sparse Jacobian/Hessian sparsity structure for IPOPT. |
| `MigrationOrchestrator` | `src/migration/migration_orchestrator.py` | Detects Type 1 (vertex collapse: VP λ→0 or λ→1) and Type 2 (triple-point) triggers, executes migrations on partition state. |
| `ProfilingState` | `src/profiling.py` | Opt-in timing accumulator for Phase 2 IPOPT callbacks. Tracks wall-clock time and Steiner recomputation counts per callback type. `finalize()` computes means and % breakdown; `to_yaml_dict()` writes `timing_profile.yaml`. Zero overhead when `--profile` is absent (all guards are `if _prof is not None:`). |
| `RelaxationProfilingState` | `src/profiling.py` | Opt-in per-level + aggregate timing accumulator for Phase 1 PGD. Per-level lifecycle: `start_level()` → `set_level_mesh_stats()` → PGD → `finalize_level()`; `finalize()` partitions `total_wall_s` (backtrack reported net of nested energy/projection); `to_yaml_dict()` writes `solution/timing_profile.yaml`. Same zero-overhead contract (`if profile is not None:`). |
| `RelaxationConfig` | `src/pipeline/relaxation.py` | Dataclass for Phase 1 config. `from_yaml_dict()` reads sectioned or flat YAML. `init_method` (`'random'` default \| `'seeded'`) selects the level-0 initial condition. `struct_trigger_enabled` / `struct_window` / `struct_rate_tol` add the structure-based refinement trigger (default off); `soft_area_constraint` (default `False`) + `soft_area_mu` select the A2 soft equal-area penalty — see the `ProjectedGradientOptimizer` row. **Per-level checkpointing:** `checkpoint_per_level` (default `True`) writes `solution/checkpoint_level{L}.h5` after every completed level — see the Phase 1 data-flow section. |
| `RefinementConfig` | `src/pipeline/pipeline_orchestrator.py` | Dataclass for Phase 2 config. `from_yaml_dict()` reads sectioned or flat YAML. CLI flags override. |
| `PipelineOrchestrator` | `src/pipeline/pipeline_orchestrator.py` | Phase 2 loop: optimize → detect → export checkpoint → migrate. Auto-detects base vs checkpoint files. Creates campaign directories under `refinement/`. |

### Data Flow

1. **Phase 1:** `find_surface_partition.py --config <experiment.yaml>` → reads `relaxation` + `surface` sections → `run_relaxation()` → builds provider → PGD loop → saves solution to `solution/`, traces to `traces/`, log to `logs/relaxation.log`, copies config to `experiment.yaml` at run root.

   **Phase 1 timing profile:** `--profile` on `scripts/find_surface_partition.py` writes `<run_dir>/solution/timing_profile.yaml` with a per-level wall-clock breakdown by callback (`matrix_assembly`, `projection`, `energy`, `gradient`, `backtrack`, `h5_save`, …). Zero overhead when omitted. Parallels the Phase 2 `--profile` campaign profile. When the file is present, `optimization_analyzer.py` automatically produces `analysis/relaxation_timing_profile.png` (stacked wall-time bars, per-call scaling, projection inner-iter growth, backtrack rate — all across the 5 refinement levels).

   **Per-level checkpoint / resume:** with `checkpoint_per_level: true` (the default), `run_relaxation` writes `solution/checkpoint_level{L}.h5` after every completed level except the last (the final solution follows immediately). The file has the same schema as a solution file — including `completed_levels` — so `--resume-from <checkpoint>` restarts the ladder at the next level instead of from level 0. Writes go to a `.tmp` file and are moved into place, so a kill mid-write leaves the previous checkpoint intact; only the newest is kept, and all are deleted once the final solution is saved. **This is what makes a multi-day cluster run survivable** — the report-04 N=200 run died mid-level-2 and lost every completed level. Note that `completed_levels` is the *absolute* ladder position (`start_level + levels run this invocation`), so resuming a resumed run works.

   **Dormant-cell detection:** `run_relaxation` calls `detect_dormant_cells()` (`src/partition/find_contours.py`) on the final solution. A cell is *dead* if it wins no vertex under winner-take-all argmax, or *weak* if its peak density stays below `WEAK_CELL_DENSITY_THRESHOLD = 0.5`. When any are found, a prominent warning is logged (console + `logs/relaxation.log`) and printed by the CLI — the solution is a consistent continuous minimizer but **not** a valid N-region partition (an under-resolved coarsest mesh is the usual cause; see `docs/reference/winner_take_all_partition_gap.md`). The full result is persisted as the `dormant_cells` block in `solution/metadata.yaml`.

   **Discrete-area-imbalance gate:** `run_relaxation` also calls `detect_area_imbalance()` (`src/partition/find_contours.py`) on the final solution. It computes the winner-take-all discrete cell areas (lumped mass assigned to each vertex's argmax cell) and flags cells whose area deviates from the equal-area target by more than `AREA_IMBALANCE_REL_THRESHOLD = 0.05`. This catches diffuse "runt" cells that pass the dormant check (peak density 1.0) but hold most of their mass outside their argmax territory — the worst cell's absolute deviation equals the Phase 2 equal-area constraint violation at iteration 0, so a large value predicts a Phase 2 run that *raises* perimeter and stalls at local infeasibility. Warning is logged + printed by the CLI (same pattern as dormant cells); the full result is the `area_imbalance` block in `solution/metadata.yaml`. This is a high-N failure distinct from dormant cells — a finer mesh does not reliably help; see `docs/reference/winner_take_all_partition_gap.md`.

   **Disconnected-cell (connectivity) gate:** `run_relaxation` also calls `detect_disconnected_cells()` (`src/partition/find_contours.py`) on the final solution. It builds the induced subgraph of mesh edges (triangle sides) whose two endpoints share the same winner-take-all argmax cell and runs `scipy.sparse.csgraph.connected_components`; a cell is *fragmented* when its territory splits into ≥2 components on the surface — so it respects the true topology (e.g. torus periodic wrap encoded in `faces`), not a flat parametrization. A non-largest component counts as a genuine stray piece only if its area exceeds `DISCONNECTED_FRAGMENT_REL_THRESHOLD = 0.01` of the equal-area target (smaller = argmax speckle, ignored). **This is a third validity dimension the other two gates are blind to:** a fragmented cell's pieces sum to the target area and each is crisp, so it passes *both* `detect_dormant_cells` (peak density ~1) and `detect_area_imbalance` (total area correct) — yet a minimal-perimeter cell is connected, so a split cell is a non-physical relaxation local minimum (nothing in the energy penalizes disconnection). Observed at torus N=200 (`run_20260722_121925`, seed 84172851): 3/200 cells split into 3–4 islands while the run reported 0 dead / 0 imbalanced. Warning is logged + printed by the CLI (same pattern as the other two gates); the full result is the `disconnected_cells` block in `solution/metadata.yaml` (`fragmented`, `n_fragmented`, per-cell `details` with `component_areas`, `worst_cell`/`worst_stray_rel`). A fragmented cell should not be handed to Phase 2 (it yields multi-loop contours). **This is NOT seed-specific — that hypothesis is falsified:** the proven-clean seed 61803399 with the (since-removed) territory-aware machinery ON gave **14/200** fragmented at level 2 (`run_20260730_211516`), worse than the 3/200 that motivated the reseed. That machinery is gone; fragmentation on the plain energy is rare (2/300 at N=300) and is repaired by the balanced readout. **Note all three gates run only on the FINAL solution**, so a run still on the mesh ladder reports none of them — use `testing/check_fragmentation.py <solution_or_checkpoint.h5>` to check a per-level checkpoint mid-ladder. FD/logic gate: `testing/test_disconnected_cells_detection.py`. See `docs/reference/winner_take_all_partition_gap.md` §4b.

2. **Phase 1 → Phase 2 bridge:** `ContourAnalyzer` loads HDF5, computes indicator functions, extracts boundary topology → `PartitionContour` is created with `VariablePoint`s on crossed edges.

   **Balanced readout (optional, high N):** when the Phase 1 solution fails the area-imbalance or connectivity gate, `scripts/balanced_readout.py --solution <solution.h5>` writes a gate-passing replacement to `<run_root>/readout/<campaign>/solution_balanced.h5` and Phase 2 is pointed at *that* file instead. Output uses the **Phase 1 solution schema**, so `refine_perimeter.py`, `visualize_partition_fast.py`, `export_partition.py` and `testing/check_fragmentation.py` consume it unchanged, with no new flags. The source run is never modified (the stage only ever creates a new campaign directory, mirroring `refinement/`). Downstream consumers read densities **only through `argmax`** (`compute_indicator_functions` builds a hard 0/1 indicator), so the stage encodes its relabelings by *swapping* the winner's and target cell's density values at each relabeled vertex: row sums — and hence partition-of-unity — are untouched, and the source densities are **exactly** recoverable by swapping back using the stored `labels_source`/`labels_final` (verified: max abs diff 0.0). `psi` is stored for provenance. See the `BalancedReadoutConfig` row above and `docs/reference/winner_take_all_partition_gap.md`.

3. **Phase 2:** `refine_perimeter.py --solution <base.h5> --config <experiment.yaml>` → reads `refinement` section (CLI flags override) → `PipelineOrchestrator.run_refinement_loop()`:
   - Creates campaign directory under `refinement/{method}_btol{tol}/` with `refinement.yaml` config snapshot.
   - **Optimize:** `PerimeterOptimizer.optimize()` adjusts λ values.
   - **Detect:** `MigrationOrchestrator.detect_all_triggers()` finds VPs near vertices (Type 1) or triple-point geometry changes (Type 2).
   - **Export:** Saves checkpoint HDF5 with `lambda_parameters`, `vp_edges`, `indicator_functions`, `pending_migration` flag, and `base_solution_path`.
   - **Migrate:** `MigrationOrchestrator.execute_migrations()` applies topology changes.
   - Loop until no migrations needed (converged) or max iterations reached.

### HDF5 File Formats

**Base solution** (Phase 1 output):
- Datasets: `x_opt`, `x0`, `vertices`, `faces`
- Attrs: `n_partitions`, `surface`, `completed_levels`, `lambda_penalty`, `seed`

**Refined contours** (Phase 2 checkpoints — `iteration_NNN_YYYYMMDD_HHMMSS.h5`):
- Datasets: `lambda_parameters`, `vp_edges`, `indicator_functions`
- Attrs: `n_variable_points`, `n_cells`, `final_perimeter`, `iteration_number`, `timestamp`, `pending_migration`, `base_solution_path` (relative path to the Phase 1 solution)
- Group `optimization_info/`: `initial_perimeter`, `perimeter_reduction`, `percent_reduction`, `constraint_violations`
- Filename encodes only the iteration number and the checkpoint's own creation time; all experiment context (surface, mesh, optimizer, tolerances) is captured by the parent run and campaign directories.

### The λ Convention (Critical)

Variable points sit on mesh edges. Position: `x = λ * vertices[edge[0]] + (1-λ) * vertices[edge[1]]`. Edges are normalized with `edge[0] < edge[1]`. So:
- λ = 1 → at the vertex with the smaller index
- λ = 0 → at the vertex with the larger index
- When λ approaches 0 or 1 (within `boundary_tol`), a Type 1 migration is triggered.

### Migration Types

- **Type 1 (Vertex Collapse):** A VP's λ is near 0 or 1, meaning it has migrated to a mesh vertex. Trigger detection (`migration_detector.py`) requires ≥3 incident boundary VPs all approaching the same vertex (with a triple-point safety guard rejecting candidates whose 1-ring intersects an existing Steiner triangle). The vertex is then flipped and its 1-ring is rebuilt edge-by-edge by `one_ring_rebuilder.py` (valence-agnostic).
- **Type 2 (Triple-Point):** Changes to which cells meet at a Steiner/triple point. Can be a forward migration (new triple-point structure) or a rollback (revert to a prior snapshot). History tracked in `type2_migration_history.py`.

## Style & Conventions

- **Formatter:** Black, line length 88, target Python 3.9
- **Imports:** Relative imports within `src/` (e.g., `from ..mesh.tri_mesh import TriMesh`). Scripts add repo root to `sys.path`.
- **Naming:** snake_case for functions/variables, PascalCase for classes. Mathematical variables keep paper notation where applicable (ε, λ, M, K, v).
- **Logging:** Use `get_logger(__name__)` from `src/logging_config.py`. Performance-sensitive functions use `@log_performance` decorator.
- **No print statements** in library code — use logger. Scripts may use print for user-facing messages.
- **Dataclasses** are preferred for config and result types.
- **Comments:** Only for non-obvious logic, mathematical references, or critical conventions (like the λ convention). No narration comments.

## Common Patterns

### Adding a New Surface Provider

**Parametric surfaces:** Subclass `SurfaceProvider` (in `src/surfaces/base.py`) and implement:
- `surface_name() → str`
- `resolution_labels() → Tuple[str, str]`
- `get_resolution() → Tuple[int, int]`
- `set_resolution(n1, n2)`
- `get_initial_resolution() → Tuple[int, int]`
- `get_resolution_increment() → Tuple[int, int]`
- `resolution_summary(refinement_levels) → Tuple[str, str]`
- `build() → TriMesh`
- `theoretical_total_area() → Optional[float]` (return `None` if no closed form)

See `EllipsoidMeshProvider` for a parametric example with polar cap handling.

**Implicit surfaces (zero level sets):** Subclass `ImplicitSurfaceProvider` (in `src/surfaces/implicit.py`) and implement only:
- `surface_name() → str`
- `implicit_function(x, y, z)` — vectorized function, surface is where `f = 0`
- `bounding_box()` — returns `((xmin,xmax), (ymin,ymax), (zmin,zmax))`

The base class handles marching cubes meshing, resolution tracking, and refinement scaling. Override `build()` if post-processing is needed (e.g., `BanchoffChmutovMeshProvider` filters to the largest connected component).

Then: add the provider to `src/surfaces/__init__.py`, add a branch in `scripts/find_surface_partition.py`, and create a YAML config under `parameters/`.

### Modifying the PGD Energy

Energy and gradient are in `ProjectedGradientOptimizer.compute_energy()` and `.compute_gradient()`. The penalty term is modular — controlled by `penalty_target_mode` and `lambda_penalty`.

### Territory-Aware Relaxation — tried, measured, removed (do not re-add)

A flag-gated "territory-aware" energy (a **WTA balance term** `P_bal = (γ/2)Σr_k²`
over a soft argmax-surrogate territory, a **discrete-area trim** retargeting the
projection toward exact argmax equality, a **P2 reduced-gradient** step/acceptance/
trigger fix, and an adaptive `wta_schedule`) was implemented on
`feat/phase1-territory-aware-relaxation`, merged for the record at commit `14e0518`,
and **removed** immediately after. It worked on the axis it targeted (N=200 bad seed:
worst runt −34% → 0 imbalanced) and was a **net regression overall**: it manufactured
14/200 disconnected cells against a matched 0/200 control, and its ladder extrapolated
to ~20 days at N=200. Root cause: it enforces *total* area through a **nonlocal**
operator (the projection), which buys a runt's area with far-away territory.

What replaced it is `src/partition/balanced_readout.py` — the same balance enforced
**exactly, at extraction, in ~17 s** (see the `BalancedReadoutConfig` row and the
"Balanced readout" paragraph of the Phase 1 data flow). Before proposing anything in this
space again, read `docs/reference/winner_take_all_partition_gap.md` **§4b** (why it
failed) and **§9b** (the local-operator test any replacement must pass). The
derivation is kept at `docs/math/07-phase1-wta-balance/` (marked not-adopted) and the
measurement at `docs/experiments/04-territory-aware-highn-validation/`.

### Approach B — auction-dynamics MBO — BUILT AND MEASURED (2026-08-19)

`src/partition/mbo_auction.py` + `scripts/run_mbo_arm.py`. A **replacement for
Phase 1**, not an adjustment to it: the state is a hard labelling throughout, so
there is never a continuous field whose winner-take-all readout can diverge.
Measured on `feat/phase1-mbo-auction-dynamics` through the Phase 0 harness, against
pre-registered thresholds committed before any run:

| | Phase 1 wall | speedup | gates on RAW labels (see caveat 1) | Phase 2 best | vs anchor |
|---|---|---|---|---|---|
| N=100 s84172851 | 228 s | **210.9×** | dormant vacuous; area 0.1504% (solver check); **0 frag** | 184.4118 (it 18/20) | **−0.455%** |
| N=100 s61803399 | 214 s | **225.2×** | vacuous; 0.1493%; **0 frag** | 184.1615 (it 19/20) | **−0.590%** |
| N=300 s61803399 | 248 s | **318.8×** | vacuous; 1.4172%; **0 frag** | 319.9428 (it 16/19) | **−1.044%** |

**Four things that matter more than the perimeter:**

1. **B produces 0 FRAGMENTED cells at both N — and connectivity is the only one of
   the three gates that has content for B.** ⚠ *"All three gates pass" is true but
   misleading, and an earlier version of this section said it.* Per Phase 0's own
   lane definitions: **dormant is VACUOUS** (one-hot labels give peak density 1.0
   by construction — `arm_harness.py` already flags `vacuous_for_arms: True`);
   **area is near-expected**, since equal area is precisely what B's assignment
   step optimizes, so passing means the solver converged — Phase 0 established it
   as the *solver-failure detector*, not a validity test. **Connectivity is the
   only gate B could genuinely fail**: nothing in B enforces it and the source
   proposal states it is *"not per-step guaranteed"*. The defensible claim is
   therefore narrower and still strong — **0 fragmented at N=300 where raw PGD
   gives 2** (plus 10 imbalanced, worst 36.15%), which is why PGD's baseline needed
   the balanced readout (A + E) and B needed nothing. The "compose with E" lane
   went unused at both N.
2. **The win is attributable to MBO, not to its init.** B is initialized from one
   balanced C-scores assignment — literally approach C's step 0 — so the attribution
   control was run: init alone + Phase 2 = **189.65**, against B's 184.41.
   MBO contributes **+2.761% / +2.892%** against a 0.3% threshold. (That gap is an
   *upper* bound: the init arm is censored.)
3. **Seed spread is 0.136%**, 3.3× smaller than the margin over the control, so the
   result is not seed noise.
4. **The comparison is "B beats the best available PGD pipeline at each N, at
   equal Phase 2 budget" — never "B beats PGD", and the two N must NOT be pooled.**
   The N=100 anchor is *raw* PGD; the N=300 anchor is PGD **+ balanced readout**,
   because raw PGD is invalid there. Different baselines, so the −0.455% and
   −1.044% margins are not one trend.
   ⚠ *An earlier version of this line said "both PGD anchors are censored, so they
   were still improving". That is **false for N=300**, whose best is iterate 18 of
   19 — plateaued, not censored.* For N=100, review 6 resumed the control's own
   Phase 2 for **20 further iterations (21–40) and found no improvement**: best over
   all 40 remains 185.2546 at iterate 20, the extension drifting up to ~185.30. So
   both trajectories sit at the migration-cycling plateau and B's plateau is simply
   0.455% lower — **do not say "B converged where the control had not"; both are
   plateaued.** (That extension is the review's measurement, **not self-reproduced**;
   report 08 must either reproduce it or label it as such.)
   ⚠ **PGD's own seed-to-seed spread at these configs is UNMEASURED** — every N=100
   PGD run on both worktrees is seed 84172851 and every N=300 λ=11.5 run is seed
   61803399, so each anchor is a single trajectory. B's margin exceeds *B's* seed
   spread by 3.3×, but "B beats PGD by more than PGD's own seed lottery" is not
   established.
5. **The 211–319× is Phase-1-replacement wall only.** End-to-end, including the
   shared ~30–40 min Phase 2, it is ≈23× (N=100) and ≈31× (N=300). And the 48,132 s
   incumbent contains the diagnosed 15,340 s frozen level-0 tail: against the
   structure-trigger variant (34,967 s) the incumbent-best figure is ≈153×.

**Free datum on C:** its step 0 lands at **+2.38% above the PGD control** (PARTIAL)
for 67 s. Real C iterates Lloyd, so that is a *lower bound* — but C starts behind
both B and PGD. Relevant to whether C is worth building.

⚠ **Do not use `E_τ` monotonicity as a gate.** The Esedoğlu–Otto theorem does not
transfer: the step maximizes `⟨χ',Dy⟩` with **lumped** `D`, while the theorem's
object `M A_τ` is the symmetric one, *and* Jacobs–Kim–Léger need an **exact**
auction where `solve_dual_offsets` is an inexact subgradient dual. Real `E_τ`
increases are observed (4 of 96 active steps at N=300). What holds by construction,
and what `testing/test_mbo_auction.py` G4 gates, is
`⟨χ',Dy⟩ − ⟨χ,Dy⟩ ≥ Σ_k ψ_k(T_k − T'_k)` — measured violation ≤ 6.1e-08.

Full pre-registration, every prediction (three missed: P1, P6, P8′; one untestable:
P9), and the negative controls are in `docs/plans/PHASE1_BC_REPLACEMENT_PLAN.md`
Phase A.

### Structure-Based Refinement Trigger (`struct_trigger_enabled`) — experimental

A **stuck detector**, not a second convergence test. The energy-plateau trigger
compares `|dE|` against an *absolute* `refine_delta_energy`, so a level whose
partition has stopped changing keeps running as long as the energy creeps. On the
N=100 deliverable `run_20260709_081548`, level 0's winner-take-all labels are
frozen from ~iteration 6,000 (≤2 of 9,600 vertices flip per 500 iterations) and
its discrete areas are unchanged from 10,000 on — yet it runs all 30,000
iterations. That is **15,340 s, 31.9% of that run's whole Phase 1 ladder**, spent
after the answer stopped moving.

`struct_trigger_enabled: true` (default `false`) tracks `argmax_k u_ik` each
iteration and fires when fewer than `struct_rate_tol · V · struct_window` labels
have flipped over the last `struct_window` iterations (defaults `1e-6` and
`2000`). Cost: one O(V·N) argmax per iteration, ~0.1%.

**The window is long rather than the rate tight, and that is the whole design.**
The frozen churn rate at N=100 level 0 (4.2e-7 flips/iter/vertex) is
indistinguishable from the rate at N=300 level 4 *shortly before its energy
trigger legitimately fires* (4.6e-7 at iteration 2,000; fired at 2,246). Rate
alone cannot separate "frozen for good" from "nearly done" — duration can.

Validated offline against both runs with `testing/validate_struct_trigger.py`,
which replays the rule over saved traces:

| run / level | iterations | structure rule |
|---|---|---|
| N=100 level 0 | 30,000 (cap) | **fires ~6,500** (78% of the level reclaimed) |
| N=100 levels 1–4 | 986–1,372 | silent (no full window) — unchanged |
| N=300 levels 0–1 | 47, 57 | silent (too short to evaluate) — unchanged |
| N=300 level 2 | 6,658 | **silent** — 228 flips vs budget 95 (thinnest margin of the set) |
| N=300 level 3 | 3,838 | silent — unchanged |
| N=300 level 4 | 2,246 | silent — unchanged |

Both structure rules are **sampled** every `struct_sample_stride` (default 500)
rather than evaluated per iteration — deliberately the granularity the offline
replay uses, because a per-iteration rule is a *different* rule: it counts every
flip, while a sampled one counts vertices whose label differs between samples,
so a vertex that flips and flips back is invisible to it. Sampling also makes the
signal free (one argmax per stride: 17.2 ms at V=114k/N=300 against a 43.6 s
iteration, ~0.04% even unsampled).

**The gate — the same signal used the other way.** `struct_gate_enabled` *blocks*
refinement while the label field is still moving (default: >1e-5 flips per
iteration per vertex over 500 iterations). The two Phase 1 pathologies are mirror
images of one mis-calibrated signal: the exact-projection run's energy creeps so
it never triggers and overruns a frozen structure, while the soft-area run's
energy plateaus so it triggers at iteration 2,575 with 3.7% of vertices still
changing cell every 500 iterations — and refinement then froze the pinched
configuration that became its 3 disconnected cells.

So it reclaims the pathology and provably does not touch the runs that already
behave.

**Measured end to end at N=100** (full ladder with the trigger on, against the
13.4 h control `run_20260709_081548`; live trigger fired at iteration 6,499 with
"13 label flips over the last 2000 iterations, budget 19.2" — reproducing the
offline replay exactly):

| | control | + structure trigger |
|---|---|---|
| Phase 1 wall | 48,132 s (13.37 h) | **34,967 s (9.71 h)** — 1.38×, saves 3.66 h |
| level 0 iterations | 30,000 (cap) | 6,500 |
| levels 1–4 iterations | 1372 / 986 / 1197 / 1279 | 1462 / 956 / 1192 / 1284 |
| dead / weak | 0 / 0 | 0 / 0 |
| imbalanced (worst) | 0 (0.78%) | 0 (0.97%) |
| fragmented | 0 | 0 |
| readout strain | 51 moves, 0 blocked, +0.13% | 76 moves, 0 blocked, +0.22% |
| Phase 2 perimeter | 185.2546 (20 iters) | **185.2096** (20 iters, no abort) |

The downstream levels are undisturbed (iteration counts within ±7%) and the final
partition is materially identical — Phase 2 lands 0.024% *better*, i.e. inside
run-to-run noise. So the saving is real and free.

**1.38×, not the 1.47× predicted from iteration counts.** Level 0's first 6,500
iterations cost 0.93 s each against the 30,000-iteration average of 0.653 s: the
projection needs more inner iterations while the field is still moving, so the
tail this trigger removes was the *cheap* part of the level. Predicting a saving
from iteration proportions overstates it; 27.4% of Phase 1 is the measured
figure.

### Soft Continuous Equal-Area Constraint (`soft_area_constraint`) — experimental

Phase 1 enforces equal *continuous* cell mass `v·u_k = Ā` exactly, via an
iterative alternating projection. That projection is **93.3% of Phase 1 wall
time** (measured at N=300, λ=11.5: 21.96 h total, 6,762 major iterations, mean
46.3 projection inner iterations; energy + gradient + backtrack together are
6.49% — `docs/math/04-phase1-timing-profile/`). Its purpose — delivering an
equal-area *winner-take-all* partition — is now met independently, exactly and
at any N, by the **balanced readout** at extraction time.

`soft_area_constraint: true` (default `false`) therefore moves equal area out of
the feasible set and into the objective:

- energy `+= (μ/2)·Σ_k r_k²`, `r_k = (v·u_k − Ā)/Ā`, `μ = soft_area_mu`;
- gradient `+= (μ/Ā²)·v_i·(v·u_k − Ā)` (rank-one, O(V·N), no extra K/M passes);
- the per-trial line-search projection becomes `project_rows_onto_simplex`
  (`src/optimization/projection.py`) — sum-to-one + box in **closed form**, one
  sort, no inner loop.

Three things to know before using it:

1. **The entry projection stays exact.** The once-per-level projection of the
   initial iterate still uses `orthogonal_projection_iterative`; it is not on
   the hot path and it makes an A/B start each level from the identical iterate.
2. **`constraint_fun` drops the area block when the flag is on**, so `FEAS`
   keeps meaning "the constraints actually enforced". Otherwise FEAS would sit
   permanently above `refine_constraint_tol` and silently change what the
   refinement trigger tests. Area drift is reported by
   `ProjectedGradientOptimizer.area_deviation()` and in each progress log line.
3. **Adaptive switching (`soft_area_adaptive`) is the interface that transfers.**
   A hand-set `soft_area_from_level` has to be read off a *finished* run, which
   makes it useless at the problem size you actually care about. Adaptive mode
   starts every level EXACT and switches to SOFT inside the level once its own
   label churn drops below `soft_switch_rate_tol` (default 1e-5) over
   `soft_switch_window` (500) — fixed rule, data-driven switch point, the same
   contract as the structure trigger. It **must** be within-level: every level
   starts churning (interpolation onto a finer mesh unsettles the field) and
   decays, so a "was the previous level settled?" test answers yes at the end of
   every level and turns the whole ladder soft. Leave `soft_area_mu: 0` and it is
   calibrated at the switch from the iterate in hand, since μ is neither N- nor
   level-invariant. **In practice the soft phase collapses within ~50–70
   iterations of the switch**, so the rule behaves as "end the level once its
   structure has settled" and the saving comes from stopping early rather than
   from cheap iterations. Validation config:
   `parameters/torus_300part_adaptive_resume.yaml`.
4. **`μ` set by hand is calibrated per config, and is not N-invariant** (it scales with
   vertices-per-cell). Use `testing/calibrate_soft_area_mu.py --config <cfg>`:
   it picks the μ at which the penalty force reaches parity with the base
   energy's `‖g‖_∞` at a 5% relative area deviation (the discrete-area gate
   threshold), at the level-0 seeded init. N=100 → 245.9; N=30 → 895.3.

**The falsifier is dormant cells, not runts.** Runt (area-imbalanced) cells are
expected here and the readout repairs them. The hard mass constraint may however
be what keeps a cell *alive* early in the flow — so any dead or weak cell in a
soft-area run fails the approach regardless of the speedup. Correctness gates:
`python testing/test_soft_area_constraint.py`.

### Phase 1 Initial Condition (`init_method`)

`relaxation.init_method` selects the level-0 initial condition: `random` (default; legacy uniform-random densities then projected, via `create_initial_condition_with_projection` in `src/optimization/projection.py`) or `seeded` (Voronoi seed regions via `create_seeded_initial_condition` in `src/optimization/initialization.py`). The seeded path picks `N` well-spread seed vertices by farthest-point sampling (deterministic given `seed`), labels every vertex by nearest seed (`scipy.spatial.cKDTree`), and projects the one-hot density with `orthogonal_projection_iterative`. It hands every cell a contiguous winning region from iteration 0, eliminating the dormant-cell symmetry-break failure at higher `N` (see `docs/reference/winner_take_all_partition_gap.md`). Dispatch is in `_create_initial_condition` (`src/pipeline/relaxation.py`), level-0 branch only; finer levels still warm-start by interpolation. The dataclass default stays `random` for backward compatibility, but **`seeded` is effectively mandatory on the corrected double-well energy** (see the `ProjectedGradientOptimizer` row above): the steeper corrected well makes the symmetric diffuse state a local minimum, so random init now *traps* (N=30: 43% worst-cell area error / 23 imbalanced cells vs 0.7% seeded). Every `parameters/*.yaml` config we run at N ≥ 30 sets `init_method: seeded`.

### Modifying Perimeter Optimization

`PerimeterOptimizer` delegates to `PerimeterCalculator`, `AreaCalculator`, and `SteinerHandler` (or their vectorized counterparts). To change the objective or constraints, modify these calculators. The `PartitionArrays` class pre-computes sparsity patterns for IPOPT.

### Running a Parameter Sweep

1. Create a sweep YAML spec (see `sweep/parameters/sweep_torus_lambda.yaml` for examples).
2. Generate configs and run:
   ```bash
   # Preview what will run
   python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode generate-only

   # Execute sequentially
   python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml

   # Execute in parallel (4 workers)
   python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode local-parallel --workers 4

   # Resume an interrupted sweep (skips completed runs)
   python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --resume
   ```
3. After runs complete, `experiment_index.yaml` is updated automatically. To rescan manually:
   ```bash
   python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode collect
   ```
4. Generate experiment-wide analysis plots:
   ```bash
   python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/
   ```

Sweep specs support two combination strategies:
- `strategy: grid` — Cartesian product of all parameter lists
- `strategy: paired` — zip together (all lists must have equal length)

Parameters that must scale together (e.g., `n_grid_x` and `n_grid_y`) are placed in a named group — the group's parameters are zipped internally, then the group participates in the cross-strategy as a unit.

### Running on Pelle (UPPMAX Cluster)

**First-time setup:**
1. Clone the repo to `$HOME`: `git clone <url> ~/surface-partition`
2. Edit `cluster/pelle_config.sh` — set `PROJECT_ID`, `PROJECT_BASE`, and verify `PYTHON_MODULE`
3. Create venv on Pelle: see setup instructions in `pelle_config.sh`

Code and scripts live in `$HOME` (small, backed up). Large output data (results, HDF5) goes under `/proj/<allocation>/`.

**Single relaxation job:**
```bash
bash cluster/submit_relaxation.sh --config parameters/torus_100part_coarse_seeded.yaml
bash cluster/submit_relaxation.sh --config parameters/torus_100part_coarse_seeded.yaml --time 24:00:00 --cpus 8
bash cluster/submit_relaxation.sh --config parameters/torus_100part_coarse_seeded.yaml --resume-from results/run_.../solution/surface_....h5
bash cluster/submit_relaxation.sh --config parameters/torus_100part_coarse_seeded.yaml --dry-run
```

**Single refinement job:**
```bash
bash cluster/submit_refinement.sh --solution results/run_.../solution/surface_....h5 --config parameters/torus_100part_coarse_seeded.yaml
bash cluster/submit_refinement.sh --solution results/run_.../solution/surface_....h5 --config parameters/torus_100part_coarse_seeded.yaml --method ipopt --exact-hessian
```

**Parameter sweep (submits one job per combination):**
```bash
bash cluster/submit_sweep.sh --sweep sweep/parameters/sweep_torus_lambda.yaml
bash cluster/submit_sweep.sh --sweep sweep/parameters/sweep_torus_lambda.yaml --dry-run
bash cluster/submit_sweep.sh --sweep sweep/parameters/sweep_torus_lambda.yaml --auto-collect
```

**Collect results after sweep jobs finish:**
```bash
python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml --mode collect
```

**Analyze sweep results:**
```bash
python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/
```

## Gotchas and Known Issues

- **No automated tests.** `testing/` contains only manual CLI diagnostics (smoke tests run from the command line). `pytest` will discover the `test_*.py` files but collect zero test functions from them.
- **PyVista is not in requirements.** It must be installed separately for 3D visualization scripts.
- **Experiment YAML format:** Both scripts accept sectioned YAML (`experiment`/`relaxation`/`surface`/`refinement` keys) and legacy flat YAML (all keys at top level). `from_yaml_dict()` on both config dataclasses handles both formats.
- **Cluster scripts** target UPPMAX Pelle. Edit `cluster/pelle_config.sh` to set your project ID and paths before first use. Verify the Python module version with `module spider python` on Pelle.
- **VariablePoint soft deletion:** Destroyed VPs are marked `active=False` but never removed from the list. This preserves index stability for snapshot rollback but means you must always filter on `vp.active`.
- **Consistency checks:** `PipelineOrchestrator.export_checkpoint()` runs roundtrip perimeter verification after saving. If this fails with a warning, the indicator functions may be out of sync with the live VP state.
- **Phase 2 migration-cycling plateau (high N).** At higher region counts (observed at N=100 and again at N=150), Phase 2 does not reach a clean convergence. After the large first-iteration perimeter drop, per-iteration gains decay to noise (~0.003%) and the topology *oscillates*: migrations (Type 1/2) periodically raise the perimeter by a hair and the next optimize step claws it back, so `pending_migration` never clears and `optimization_success` stays `False`. It runs to the iteration cap without converging — this is a **plateau, not a failure or a bug**. The exported geometry at the best iterate is complete and valid; it just wasn't topologically frozen. **Standard workflow:** pick the minimum-`final_perimeter` iteration across the campaign (scan `final_perimeter` on every `iteration_*.h5`) and export it. Because that iterate carries `pending_migration=True`, `scripts/export_partition.py` writes `finalised=False` by default (`finalised = not pending_migration` in `src/export/writer.py`); for the accepted final deliverable, pass **`--force-finalised`** — it writes `finalised=True` plus an explanatory `finalised_note` (best iterate at the plateau) in one reproducible step, so external repos that gate on `finalised==True` accept it. `--force-finalised` is mutually exclusive with `--strict`. The N=100 deliverables were finalised by hand-patching the attr (before the flag existed); the N=150 deliverable uses `--force-finalised`.
- **`run_time_seconds` in `solution/metadata.yaml` is NOT the run's wall time.** It is `float(results[-1]['elapsed'])` — the **last level's** PGD elapsed time only (`src/pipeline/relaxation.py`). On the N=100 deliverable `run_20260709_081548` it reads **13,416 s** while the run actually took **48,132 s** (13.4 h) — a 3.6× under-report, because the last level is not the expensive one (level 0 is: it runs the 30,000-iteration cap and is 41% of that ladder). **For a real total, sum `level_wall_s` over `levels` in `solution/timing_profile.yaml`** (requires `--profile`), or read `summary.total_wall_s` on runs new enough to populate it. Anything comparing Phase 1 cost across configs — scaling studies, optimizer A/B arms — must use the profile, not `run_time_seconds`.
- **Phase 1 `lambda_penalty` has a working *window* at high N — over-raising it backfires.** The crispness penalty is the main lever against the winner-take-all runt at high N, but it has an upper *ceiling*, not just a lower bound. Too low → diffuse runts (see `docs/reference/winner_take_all_partition_gap.md`). **Too high → the penalty dominates the energy, the multi-level refinement triggers misfire (finer levels fire after *tens* of iterations instead of thousands), and PGD stops before crisping the interfaces — leaving a diffuse `min peak density ≈ 0.7` mush with most cells area-imbalanced, and the run finishes suspiciously fast.** Concretely at N=300: `lambda_penalty: 12` relaxes properly (min peak ~0.98, finest level ~7.7k iterations); `15` collapses to mush (min peak 0.71, 234/300 imbalanced). The needed λ grows with N (~5 at N=100, ~11 at N=200) but stays under the ceiling; some high-N failures are also seed-specific (a different `seed` can resolve a runt — this unblocked N=200). **Diagnostic:** if a high-N run looks wrong, check the final min peak density (`dormant_cells.max_density_per_cell` in `metadata.yaml`) and the per-level `Refinement triggered at iteration N` counts in the log — a fast run with low peak density means λ is over the ceiling; lower it.
- **A floored line search is NORMAL convergence — do not read it as failure.** Every healthy Phase 1 level ends with the Armijo step at the backtracking floor `9.0949470177292824e-13` (= `pgd_step0 · pgd_backtrack_rho^40`) and the energy frozen for exactly `refine_patience` (30) iterations, at which point the plateau trigger fires. The validated N=100 deliverable `run_20260709_081548` does this on **four of its five levels**. An earlier `LINE SEARCH STALLED` warning fired on any floored step and therefore fired on every healthy level termination; two commits reasoned from it before it was checked against a known-good run. **What is diagnostic is *when* the floor is reached, not that it is.** The guard (`_STALL_EARLY_ITERS = 200` in `src/optimization/pgd_optimizer.py`) now warns `UNDER-RESOLVED LEVEL` only when the step floors inside the first 200 iterations. That threshold sits in an empty gap — but a **narrower one than first reported**: the replay behind it saw 130 of the **350** summary traces in `results/`, because its glob (`results/run_*/traces/`) could not see the 185 in the sweep directories. On the full tree three `torus_npart10` levels floor at **227 / 280 / 303**, so the real gap is **(100, 227)**. 200 is still correct on all 350 levels (nothing floors in [100, 226]), but the margin is **1.13×, not 6.7×**, and a threshold of 250 *would* misfire. Firing means the level died before doing any work — check `lambda_penalty` and verts/cell. Two blind spots: it does **not** detect the A2 soft-area pathology (levels floored *late*, collapse still real), nor levels that die **without ever flooring** (three N=30 sweep levels end at 34–36 iterations with no floored step at all). See `docs/experiments/06-subfloor-ladder/`.
- **A Phase 1 level can die doing no work — but the "~90 verts/cell boundary" is NOT established.** The *phenomenon* is real and matters: levels at 31/32/48/64/83 v/cell reach the backtracking floor at iteration **18–47** and stop, so the cheap equalization phase never happens there and the first level that runs starts from a badly imbalanced field. **The threshold's location is a different claim, and it is refuted as previously stated here.** v/cell is causally implicated by exactly **one controlled pair** (N=100 λ=5.1: 31 v/cell dies at onset 30, 96 v/cell never floors — the subfloor experiment itself). Every other point pooled into "~90" varies in λ-window membership and straddles the 2026-07-06 energy fix: **`run_20260629_141012` is 96 v/cell and dies at onset 32**, because its λ=2.1 is below the N=100 window. Separated by start type the brackets are cold (64, 92) and warm (83, 124) — wide and overlapping, and **no cold N=300 point above 83 v/cell exists on disk**. Treat ~90 as a rough prior, never a design constant: the N=1000 mesh-budget arithmetic and the "51 v/cell is still in the dying regime" dismissal of `torus_300part_rebased_ladder.yaml` both inherit this uncertainty. **Do not confuse this with the ~250–300 v/cell gate floor** of `winner_take_all_partition_gap.md` §4b (the resolution at which a level can itself clear the 5% gate); they differ by ~3× in mesh budget at N=1000. Caveat that stops it being a rule: a dying level 0 is not by itself fatal — N=200 reaches 0 imbalanced with a level 0 at 48 v/cell that dies at iteration 21, because its level 1 (124 v/cell) then runs 12,155 iterations. The statement consistent with every run is weaker: *some early, cheap level must run long enough to equalize before the expensive levels begin.*
- **N=300's ladder shows decelerating returns — but "exhausted" is NOT established.** Reconstructed for free from the λ=11.5 control's saved trace iterates (`traces/*_internal_data.hdf5` hold `x`; rebuild the mesh with `TorusMeshProvider` and run the gates): seeded init 230/300 imbalanced / 44.02%; after L0 (32 v/cell) 234 / 45.57%; after L1 (83 v/cell) 234 / 49.26% — both coarse levels die and change nothing (**stronger than it reads**: L0's 47 iterations flipped *zero* argmax labels, and interp+project of the seeded init reproduces L1's `iter_0` labels 100.000%, 0 of 24,948 — the +4 is pure resampling); **L2 (158 v/cell) does all the work** (→ 10 imbalanced / 36.15%); then L3 and L4, decelerating. **Three caveats the numbers above hide.** (1) *L2 did not demonstrably converge*: its line search floored at iteration 6,628 with the energy **still falling** (−0.667 over the last 1,000) and ‖g‖ = 31.11 — a 40-deep Armijo failure mid-descent, i.e. an optimizer limit, not a shown stationary point; the flat window is ~658 iterations, not 1,000. (2) *The table mixes three undisclosed estimators* for "after level L" — L0/L1 rows are the next level's `iter_0` (all those traces hold), the "after L3" row is the **same** level's iter_3500 of a 3,838-iteration level, and L2/L4 are true finals; L4's actual `iter_0` is **6 cells / 25.94%**, not 4 / 27.04%, and where both estimators exist resampling *triples* the imbalanced count (10 → 31). (3) *The per-level gains are partly mesh resampling, not optimization*: L4's −2.23 pts is −1.13 optimization + −1.10 resampling; L3's −9.11 is −8.65 + −0.46. So "the residual imbalance is a property of the energy, not incomplete descent" is **consistent with these data, not established by them** — no stationarity or KKT measure exists anywhere in the traces. **Untested**, and the one ladder question still open: whether giving N=300 an *earlier* level that actually runs (~92 v/cell ⇒ V₀ ≈ 27,600, base ~176×164) moves the converged point — report 06 shows that point is start-dependent. `parameters/torus_300part_rebased_ladder.yaml` does **not** test it (it lifts level 0 only to 51 v/cell, still in the dying regime). **Also note fragmentation is born mid-level-2** (iterations 2,500 and 4,500) while the imbalanced count falls 234 → 10 — balance is bought with disconnection on the *plain* energy too, the same mechanism report 05 measured for A2, milder. See `docs/reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`.
- **⚠ `parameters/*.yaml` can DRIFT from the run that was actually produced with it — read the run's own `experiment.yaml`.** `parameters/torus_300part_seeded_lam11p5_original_energy.yaml:45` says `refinement_levels: 5` (finest 348×328, V=114,144), raised post-hoc to drive a resume, while the anchor run `run_20260806_123326` was produced with **3** (its own `experiment.yaml:45`) and its solution is **V=47,488** — verified on disk. Anything that reconstructs a run's mesh from the live `parameters/` file therefore lands on a *finer* mesh than the run it is being compared against, which is a ~0.11% perimeter effect pointing in the newcomer's favour. `scripts/run_mbo_arm.py` reads the ladder from the anchor's `experiment.yaml` and hard-asserts `arm_final_V == anchor_V`. The N=100 control did not drift; only N=300 did.
- **MBO time step `tau` is a TWO-SIDED window, and a fixed `c` does not hold it.** For approach B, `tau = min((c·h_mean)², (ρ·R_cell)²)` with `c = 4`, `ρ = 1.0`. Below `c ≈ √(R_cell/h)` the scheme *freezes* (per-step motion ≈ τκ falls under one edge length); above `√τ ≈ R_cell` it *over-merges* (the diffused indicator has spread over the whole cell). The trap: a fixed `c` holds `√τ/h_max` constant at 2.20 while `√τ/R_cell` varies **6×** across the ladders in use (0.250 at N=100 V=114,144 up to 1.495 at N=300 V=9,600), so the two sides pull in opposite directions as the mesh coarsens or N rises. The cap binds on exactly one level in current use, N=300 level 0. Report **`√τ/h_max`, never `h_mean` alone** — this mesh is **1.81× anisotropic**, so a mean-based figure shows a comfortable 2.20 while the coarse-triangle band sits below its own local non-freeze threshold of 2.97. Note the two rulers disagree about which levels are at risk, so neither settles it by algebra; the calibrated pinning probe (NC3) is the arbiter.
- **Assignment quality cannot detect over-merging** — the balanced assignment hits its area target *by construction*, so a smeared, non-local cell scores just as well as a compact one. Anything reasoning about MBO's `tau` being too large must use a geometry instrument: fragmented-cell count, per-cell isoperimetric ratio `Q_k = P_k²/(4πA_k)`, or core loss (cells whose own diffused-score peak vertex lies outside their territory). Citing an assignment-quality result against an over-merge claim is the same class of error as the theatre bar and the early-stop mask (`docs/experiments/07-phase0-shared-harness/`).
- **A level below the resolution floor permanently damages the partition.** Measured at N=100 (`docs/experiments/06-subfloor-ladder/`): dropping only the coarsest level from 96 to 31 verts/cell — everything else, including the finest mesh, identical to the validated control — yields a cell 15.92% off the equal-area target where the control gets 0.78%. The defect is born at level 0 (46.80%) and is **not healed** by 37× more vertices downstream. It did *not* produce fragmentation. Note the sub-floor ladder is also *cheaper* (≈24,300 s vs 48,132 s), so cost alone never indicates a healthy ladder — judge validity per unit compute.

## Dependencies

Core (`pip install -e .`): `numpy`, `scipy`, `pyyaml`, `matplotlib`, `h5py`, `tqdm`
Optional groups (defined in `pyproject.toml`):
- `pip install -e ".[ipopt]"` — adds `cyipopt` (IPOPT solver for Phase 2)
- `pip install -e ".[viz]"` — adds `pyvista` (3D visualization) and `colorcet` (optional; glasbey palette for neighbour-distinct cell colors — falls back to a built-in HSV palette if absent)
- `pip install -e ".[implicit]"` — adds `scikit-image` (marching cubes for implicit surfaces)
- `pip install -e ".[all]"` — all optional deps
- `pip install -e ".[dev]"` — adds `pytest`, `black`, `flake8`
