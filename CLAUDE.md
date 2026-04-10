# CLAUDE.md — Project Context for Claude Code

## Project Overview

**surface-partition** is a Python framework for computing minimal-perimeter partitions on closed triangulated surfaces. It implements the method from "Partitions of Minimal Length on Manifolds" (Bogosel & Oudet) using two phases:

1. **Phase 1 (Relaxation):** Γ-convergence energy minimization via Projected Gradient Descent (PGD) on nodal density functions, with multi-level mesh refinement.
2. **Phase 2 (Refinement):** Direct constrained perimeter minimization on extracted contour variable points, with automatic topology migrations (Type 1 and Type 2).

The only surface currently implemented is the **torus** (`TorusMeshProvider`). The architecture is designed for adding new surfaces by implementing a provider class.

## Build & Run

```bash
# Setup (uses pyenv — see .python-version for the environment name)
pyenv activate ringtest-3.9   # or whichever pyenv virtualenv is configured
pip install -e .               # core only
pip install -e ".[all]"        # or: core + PyVista + IPOPT

# Phase 1: Γ-convergence relaxation
python scripts/find_surface_partition.py --input parameters/input.yaml --surface torus

# Phase 2: Perimeter refinement (requires Phase 1 output)
python scripts/refine_perimeter.py --solution <path_to_solution.h5> --max-iterations 10

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
├── mesh/               # TriMesh, FEM assembly, topology, interpolation
├── surfaces/           # Surface providers (torus.py)
├── optimization/       # PGD optimizer (Phase 1), perimeter optimizer (Phase 2)
├── partition/          # Contour extraction, PartitionContour, calculators, vectorized kernels
├── migration/          # Topology migration detection and execution
├── pipeline/           # Workflow orchestration (relaxation, refinement, I/O)
├── visualization/      # Matplotlib plotting helpers
├── config.py           # Legacy Config class (mostly superseded by RelaxationConfig)
└── logging_config.py   # Logging with @log_performance decorator
scripts/                # CLI entry points
parameters/input.yaml   # Default run configuration
cluster/submit.sh       # SLURM submission for UPPMAX
```

### Key Classes and Their Roles

| Class | Module | Purpose |
|-------|--------|---------|
| `TriMesh` | `src/mesh/tri_mesh.py` | Triangle mesh with P1 FEM mass (M) and stiffness (K) matrices. Properties: `.M`, `.K`, `.v` (lumped mass = row-sum of M). Supports R2 and R3. |
| `MeshTopology` | `src/mesh/mesh_topology.py` | Edge-triangle adjacency structures needed by migration subsystem. |
| `TorusMeshProvider` | `src/surfaces/torus.py` | Builds structured torus TriMesh from (n_theta, n_phi, R, r). Supports refinement increments. |
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
| `RelaxationConfig` | `src/pipeline/relaxation.py` | Dataclass for Phase 1 config. `from_yaml_dict()` factory. |
| `RefinementConfig` | `src/pipeline/pipeline_orchestrator.py` | Dataclass for Phase 2 config (max_iterations, method, tolerances, etc.). |
| `PipelineOrchestrator` | `src/pipeline/pipeline_orchestrator.py` | Phase 2 loop: optimize → detect → export checkpoint → migrate. Auto-detects base vs checkpoint files. |

### Data Flow

1. **Phase 1:** `find_surface_partition.py` → `run_relaxation()` → builds `TorusMeshProvider` → for each refinement level: `_setup_level()` → `ProjectedGradientOptimizer.optimize()` → saves solution HDF5 with `x_opt`, `x0`, `vertices`, `faces`.

2. **Phase 1 → Phase 2 bridge:** `ContourAnalyzer` loads HDF5, computes indicator functions, extracts boundary topology → `PartitionContour` is created with `VariablePoint`s on crossed edges.

3. **Phase 2:** `refine_perimeter.py` → `PipelineOrchestrator.run_refinement_loop()`:
   - **Optimize:** `PerimeterOptimizer.optimize()` adjusts λ values.
   - **Detect:** `MigrationOrchestrator.detect_all_triggers()` finds VPs near vertices (Type 1) or triple-point geometry changes (Type 2).
   - **Export:** Saves checkpoint HDF5 with `lambda_parameters`, `vp_edges`, `indicator_functions`, `pending_migration` flag.
   - **Migrate:** `MigrationOrchestrator.execute_migrations()` applies topology changes.
   - Loop until no migrations needed (converged) or max iterations reached.

### HDF5 File Formats

**Base solution** (Phase 1 output):
- Datasets: `x_opt`, `x0`, `vertices`, `faces`
- Attrs: `n_partitions`, `surface`, `completed_levels`, `lambda_penalty`, `seed`

**Refined contours** (Phase 2 checkpoints):
- Datasets: `lambda_parameters`, `vp_edges`, `indicator_functions`
- Attrs: `n_variable_points`, `n_cells`, `final_perimeter`, `iteration_number`, `pending_migration`
- Group `optimization_info/`: `initial_perimeter`, `perimeter_reduction`, `percent_reduction`, `constraint_violations`

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

Create `src/surfaces/my_surface.py` with a class implementing:
- `surface_name() → str`
- `resolution_labels() → Tuple[str, str]`
- `get_resolution() → Tuple[int, int]`
- `set_resolution(n1, n2)`
- `resolution_summary(refinement_levels) → Tuple[str, str]`
- `build() → TriMesh`
- `theoretical_total_area() → float` (optional)

Then add a branch in `scripts/find_surface_partition.py` for the new `--surface` value.

### Modifying the PGD Energy

Energy and gradient are in `ProjectedGradientOptimizer.compute_energy()` and `.compute_gradient()`. The penalty term is modular — controlled by `penalty_target_mode` and `lambda_penalty`.

### Modifying Perimeter Optimization

`PerimeterOptimizer` delegates to `PerimeterCalculator`, `AreaCalculator`, and `SteinerHandler` (or their vectorized counterparts). To change the objective or constraints, modify these calculators. The `PartitionArrays` class pre-computes sparsity patterns for IPOPT.

## Gotchas and Known Issues

- **No automated tests.** `testing/` contains only manual CLI diagnostics. `pytest` will find `test_migrations_debug.py` but collect zero test functions.
- **`docs/` is gitignored.** README references to `docs/PERIMETER_REFINEMENT.md` will be broken for fresh clones.
- **PyVista is not in requirements.** It must be installed separately for 3D visualization scripts.
- **`src/config.py`** is a legacy Config class, mostly superseded by `RelaxationConfig` and `RefinementConfig` dataclasses. It still exists but is not used by the main pipeline.
- **`cluster/submit.sh`** defaults surface to `ring` in the YAML fallback, but only `torus` is implemented.
- **VariablePoint soft deletion:** Destroyed VPs are marked `active=False` but never removed from the list. This preserves index stability for snapshot rollback but means you must always filter on `vp.active`.
- **Consistency checks:** `PipelineOrchestrator.export_checkpoint()` runs roundtrip perimeter verification after saving. If this fails with a warning, the indicator functions may be out of sync with the live VP state.

## Dependencies

Core (`pip install -e .`): `numpy`, `scipy`, `pyyaml`, `matplotlib`, `h5py`, `tqdm`
Optional groups (defined in `pyproject.toml`):
- `pip install -e ".[ipopt]"` — adds `cyipopt` (IPOPT solver for Phase 2)
- `pip install -e ".[viz]"` — adds `pyvista` (3D visualization)
- `pip install -e ".[all]"` — all optional deps
- `pip install -e ".[dev]"` — adds `pytest`, `black`, `flake8`
