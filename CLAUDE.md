# CLAUDE.md — Project Context for Claude Code

## Project Overview

**surface-partition** is a Python framework for computing minimal-perimeter partitions on closed triangulated surfaces. It implements the method from "Partitions of Minimal Length on Manifolds" (Bogosel & Oudet) using two phases:

1. **Phase 1 (Relaxation):** Γ-convergence energy minimization via Projected Gradient Descent (PGD) on nodal density functions, with multi-level mesh refinement.
2. **Phase 2 (Refinement):** Direct constrained perimeter minimization on extracted contour variable points, with automatic topology migrations (Type 1 and Type 2).

Surfaces currently implemented: **torus** (`TorusMeshProvider`), **ellipsoid** (`EllipsoidMeshProvider`), **double torus** (`DoubleTorusMeshProvider`), and **Banchoff-Chmutov order 4** (`BanchoffChmutovMeshProvider`).

## Build & Run

```bash
# Setup (uses pyenv — see .python-version for the environment name)
pyenv activate ringtest-3.9   # or: pyenv activate surface-partition
pip install -e .               # core only
pip install -e ".[all]"        # or: core + PyVista + IPOPT + scikit-image

# Phase 1: Γ-convergence relaxation
python scripts/find_surface_partition.py --config parameters/torus_10part.yaml
python scripts/find_surface_partition.py --config parameters/ellipsoid_6part.yaml
python scripts/find_surface_partition.py --config parameters/double_torus_4part.yaml      # requires .[implicit]
python scripts/find_surface_partition.py --config parameters/banchoff_chmutov_4part.yaml  # requires .[implicit]

# Phase 2: Perimeter refinement (requires Phase 1 output)
python scripts/refine_perimeter.py --solution <path_to_solution.h5> --config parameters/torus_10part.yaml
# Or with CLI overrides:
python scripts/refine_perimeter.py --solution <path_to_solution.h5> --max-iterations 10 --method ipopt

# Visualization (all require pyvista)
# Production viewer — vectorized, handles fine meshes efficiently:
python scripts/visualize_partition_fast.py --solution <path_to_solution.h5>
# Original viewer — slower on fine meshes, useful for debugging:
python scripts/visualize_partition.py --solution <path_to_solution.h5>
# Migration debugging viewers (Type 1 / Type 2 topology switches):
python scripts/visualize_type1_vertex_collapse.py --solution <path_to_refined.h5> --region 2
python scripts/visualize_type2_triple_point.py --solution <path_to_refined.h5> --region 2

# Analysis
python scripts/optimization_analyzer.py --results-dir results/<run_dir>
```

## Testing

There are no pytest unit tests. The `testing/` directory contains CLI diagnostic tools:
```bash
python testing/test_migrations_debug.py --solution <path_to_refined.h5>
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
│   ├── projection.py             # Iterative constraint projection (sum-to-one, equal areas)
│   └── exceptions.py             # RefinementTriggered exception
├── partition/
│   ├── find_contours.py          # ContourAnalyzer: HDF5 → indicator functions → boundary topology
│   ├── contour_partition.py      # PartitionContour, VariablePoint, TriangleSegment
│   ├── perimeter_calculator.py   # Per-segment perimeter with analytical gradients
│   ├── area_calculator.py        # Per-cell FEM area with analytical gradients
│   ├── steiner_handler.py        # Steiner/triple-point perimeter + area contributions
│   ├── partition_arrays.py       # PartitionArrays: sparse Jacobian/Hessian sparsity for IPOPT
│   ├── vectorized_perimeter.py   # Fast vectorized perimeter evaluation
│   ├── vectorized_area.py        # Fast vectorized area evaluation
│   └── vectorized_steiner.py     # Fast vectorized Steiner evaluation
├── migration/
│   ├── migration_orchestrator.py # MigrationOrchestrator: top-level detect → execute loop
│   ├── migration_detector.py     # Type 1 + Type 2 trigger detection
│   ├── migration_executor.py     # Execute migrations on partition state
│   ├── migration_types.py        # DetectionResult, MigrationResult, MigrationConfig dataclasses
│   ├── migration_utils.py        # Shared helpers (edge utilities, geometry)
│   ├── type1_component_analyzer.py  # Connected-component analysis for Type 1 vertex collapse
│   ├── type2_migration_io.py     # Type 2 snapshot save/restore
│   ├── type2_migration_history.py   # Type 2 rollback history tracking
│   └── one_ring_rebuilder.py     # One-ring mesh topology rebuilding after Type 1 migration
├── pipeline/
│   ├── relaxation.py             # run_relaxation(): multi-level PGD pipeline (Phase 1)
│   ├── pipeline_orchestrator.py  # PipelineOrchestrator, RefinementConfig, derive_output_paths (Phase 2)
│   └── io.py                     # HDF5 loaders, detect_run_layout(), find_base_solution_path()
├── visualization/
│   ├── plot_utils.py             # Matplotlib utilities
│   └── partition_helpers.py      # Partition-specific viz helpers (cell coloring, VP/Steiner markers)
└── logging_config.py             # Logging setup, get_logger(), @log_performance decorator
scripts/
├── find_surface_partition.py     # Phase 1 CLI: Γ-convergence relaxation
├── refine_perimeter.py           # Phase 2 CLI: iterative perimeter refinement
├── optimization_analyzer.py      # Result analysis and plotting
├── visualize_partition_fast.py   # Fast partition viewer — production (PyVista, vectorized)
├── visualize_partition.py        # Original partition viewer — debugging (PyVista)
├── visualize_type1_vertex_collapse.py  # Type 1 migration debugging viewer
├── visualize_type2_triple_point.py     # Type 2 migration debugging viewer
└── debug_archive/                # Archived diagnostic scripts
testing/
├── README_testing.md             # Test registry documentation
└── test_migrations_debug.py      # Migration debug CLI
parameters/
├── torus_10part.yaml             # Torus, 10 partitions (parametric mesh)
├── ellipsoid_6part.yaml          # Ellipsoid, 6 partitions (parametric mesh)
├── double_torus_4part.yaml       # Double torus, 4 partitions (implicit / marching cubes)
└── banchoff_chmutov_4part.yaml   # Banchoff-Chmutov order 4, 4 partitions (implicit / marching cubes)
cluster/submit.sh                 # SLURM submission for UPPMAX (Rackham)
```

### Run Output Layout (Structured)

Each Phase 1 run creates a structured directory under `results/`:

```
results/run_{timestamp}_surf{surface}_npart{N}_v1..._v2..._lam{λ}_seed{S}/
├── experiment.yaml               # Verbatim copy of the input config (reproduction recipe)
├── solution/
│   ├── surface_part{N}_surf{surface}_v1..._v2..._lam{λ}_seed{S}_{timestamp}.h5
│   └── metadata.yaml             # Derived runtime results (mesh stats, timings, file paths)
├── traces/
│   ├── pgd_part{N}_v1{label}{n1}_v2{label}{n2}_level{L}_summary.out
│   └── pgd_part{N}_v1{label}{n1}_v2{label}{n2}_level{L}_internal_data.hdf5
├── refinement/
│   ├── slsqp_btol0.001/
│   │   ├── iteration_001_20260410_120523.h5
│   │   ├── refinement.yaml
│   │   └── refinement.log
│   └── ipopt_btol0.001_lbfgs20_hess/
│       ├── iteration_001_20260410_131042.h5
│       ├── iteration_002_20260410_131215.h5
│       ├── refinement.yaml
│       └── refinement.log
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
| `ProjectedGradientOptimizer` | `src/optimization/pgd_optimizer.py` | Phase 1 PGD. Energy = ε·u^T·K·u + (1/ε)·(u²(1-u)²)^T·M·(u²(1-u)²) + penalty. Constraints: partition sum-to-one, equal areas. |
| `PerimeterOptimizer` | `src/optimization/perimeter_optimizer.py` | Phase 2. Minimizes total perimeter (regular + Steiner) subject to equal cell areas. Supports SLSQP, trust-constr, IPOPT. |
| `IPOPTProblemAdapter` | `src/optimization/perimeter_optimizer.py` | Adapts PerimeterOptimizer for cyipopt interface. Optional best-iterate tracking and exact Hessian. |
| `ContourAnalyzer` | `src/partition/find_contours.py` | Loads HDF5 solution, computes indicator functions (winner-take-all), extracts boundary triangles and topology. |
| `PartitionContour` | `src/partition/contour_partition.py` | Central data structure: list of `VariablePoint`s (edge + λ parameter), `TriangleSegment`s, indicator arrays, Steiner bookkeeping. |
| `VariablePoint` | `src/partition/contour_partition.py` | Point on mesh edge at position x = λ·v_start + (1-λ)·v_end. λ∈[0,1]. λ=1 → at smaller vertex index. Has `active` flag for soft deletion. |
| `PerimeterCalculator` | `src/partition/perimeter_calculator.py` | Computes per-segment perimeter contributions with analytical gradients. |
| `AreaCalculator` | `src/partition/area_calculator.py` | Computes per-cell FEM area with analytical gradients. |
| `SteinerHandler` | `src/partition/steiner_handler.py` | Manages Steiner/triple-point perimeter and area contributions for triangles where 3+ cells meet. |
| `PartitionArrays` | `src/partition/partition_arrays.py` | Pre-computes sparse Jacobian/Hessian sparsity structure for IPOPT. |
| `MigrationOrchestrator` | `src/migration/migration_orchestrator.py` | Detects Type 1 (vertex collapse: VP λ→0 or λ→1) and Type 2 (triple-point) triggers, executes migrations on partition state. |
| `RelaxationConfig` | `src/pipeline/relaxation.py` | Dataclass for Phase 1 config. `from_yaml_dict()` reads sectioned or flat YAML. |
| `RefinementConfig` | `src/pipeline/pipeline_orchestrator.py` | Dataclass for Phase 2 config. `from_yaml_dict()` reads sectioned or flat YAML. CLI flags override. |
| `PipelineOrchestrator` | `src/pipeline/pipeline_orchestrator.py` | Phase 2 loop: optimize → detect → export checkpoint → migrate. Auto-detects base vs checkpoint files. Creates campaign directories under `refinement/`. |

### Data Flow

1. **Phase 1:** `find_surface_partition.py --config <experiment.yaml>` → reads `relaxation` + `surface` sections → `run_relaxation()` → builds provider → PGD loop → saves solution to `solution/`, traces to `traces/`, log to `logs/relaxation.log`, copies config to `experiment.yaml` at run root.

2. **Phase 1 → Phase 2 bridge:** `ContourAnalyzer` loads HDF5, computes indicator functions, extracts boundary topology → `PartitionContour` is created with `VariablePoint`s on crossed edges.

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

- **Type 1 (Vertex Collapse):** A VP's λ is near 0 or 1, meaning it has migrated to a mesh vertex. The VP is absorbed and the topology around that vertex is rebuilt via connected-component analysis (`type1_component_analyzer.py`, `one_ring_rebuilder.py`).
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

### Modifying Perimeter Optimization

`PerimeterOptimizer` delegates to `PerimeterCalculator`, `AreaCalculator`, and `SteinerHandler` (or their vectorized counterparts). To change the objective or constraints, modify these calculators. The `PartitionArrays` class pre-computes sparsity patterns for IPOPT.

## Gotchas and Known Issues

- **No automated tests.** `testing/` contains only manual CLI diagnostics. `pytest` will find `test_migrations_debug.py` but collect zero test functions.
- **`docs/` is gitignored.** README references to `docs/PERIMETER_REFINEMENT.md` will be broken for fresh clones.
- **PyVista is not in requirements.** It must be installed separately for 3D visualization scripts.
- **Experiment YAML format:** Both scripts accept sectioned YAML (`experiment`/`relaxation`/`surface`/`refinement` keys) and legacy flat YAML (all keys at top level). `from_yaml_dict()` on both config dataclasses handles both formats.
- **`cluster/submit.sh`** defaults surface to `ring` in the YAML fallback; always pass `--config` with a valid experiment YAML to select the desired surface.
- **VariablePoint soft deletion:** Destroyed VPs are marked `active=False` but never removed from the list. This preserves index stability for snapshot rollback but means you must always filter on `vp.active`.
- **Consistency checks:** `PipelineOrchestrator.export_checkpoint()` runs roundtrip perimeter verification after saving. If this fails with a warning, the indicator functions may be out of sync with the live VP state.

## Dependencies

Core (`pip install -e .`): `numpy`, `scipy`, `pyyaml`, `matplotlib`, `h5py`, `tqdm`
Optional groups (defined in `pyproject.toml`):
- `pip install -e ".[ipopt]"` — adds `cyipopt` (IPOPT solver for Phase 2)
- `pip install -e ".[viz]"` — adds `pyvista` (3D visualization)
- `pip install -e ".[implicit]"` — adds `scikit-image` (marching cubes for implicit surfaces)
- `pip install -e ".[all]"` — all optional deps
- `pip install -e ".[dev]"` — adds `pytest`, `black`, `flake8`
