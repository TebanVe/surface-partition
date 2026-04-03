# Codebase Restructure Plan

This document describes a proposed restructuring of the `src/` and `examples/` directories to improve organization, readability, and maintainability. The goal is to make the codebase easy to navigate for both human developers and AI agents that may interact with it programmatically.

The restructure is organized into three phases:

- **Phase A** — Reorganize `src/core/` into focused subpackages.
- **Phase B** — Move `examples/data_loader.py` into `src/` as proper library code; clarify the role of `examples/` and `testing/`.
- **Phase C** — Implement a `src/pipeline/` orchestrator that chains all stages programmatically, reducing entry-point scripts to thin wrappers.

---

## Problem

The current layout has three issues:

1. **`core/` is overloaded** — 25 files mixing five unrelated domains: mesh infrastructure, partition data structures, vectorized evaluation, optimization algorithms, and topology migration. A developer looking for "the optimizer" must scan past mesh connectivity, contour partitions, migration history, and vectorized calculators.

2. **`src/` root is a grab-bag** — config, logging, plotting, contour analysis, and projection sit at the top level with no grouping. These files have very different audiences (plotting is for visualization scripts; contour analysis and projection are part of the computational pipeline).

3. **No programmatic pipeline** — the four computational stages (Γ-convergence relaxation → contour extraction → perimeter refinement → topology migration) can only be run by executing scripts in sequence. There is no single entry point that chains them, and shared loading logic (`data_loader.py`) lives in `examples/` instead of the library.

## Current Structure

### `src/` (31 files)

```
src/
├── __init__.py                     (37 lines — package exports, __version__ = "0.2.0")
├── config.py                       (118 lines — configuration)
├── logging_config.py               (133 lines — logging setup)
├── find_contours.py                (276 lines — contour extraction from HDF5 solutions)
├── projection_iterative.py         (363 lines — projection onto constraints)
├── plot_utils.py                   (232 lines — 2D plotting)
├── core/                           (NO __init__.py — 22 files)
│   ├── tri_mesh.py                 (156 lines — mesh container + FEM)
│   ├── mesh_topology.py            (156 lines — mesh connectivity)
│   ├── interpolation.py            (23 lines — nearest-neighbor interp)
│   ├── contour_partition.py        (1187 lines — VP, segments, partition)
│   ├── area_calculator.py          (493 lines — area computation)
│   ├── perimeter_calculator.py     (252 lines — perimeter computation)
│   ├── steiner_handler.py          (429 lines — triple point Steiner trees)
│   ├── partition_arrays.py         (86 lines — flat NumPy snapshot for vectorized eval)
│   ├── vectorized_area.py          (560 lines — vectorized area + Jacobian + Hessian)
│   ├── vectorized_perimeter.py     (146 lines — vectorized perimeter + gradient + Hessian)
│   ├── vectorized_steiner.py       (297 lines — analytical Steiner vectorized + FD Hessian)
│   ├── perimeter_optimizer.py      (719 lines — SLSQP/IPOPT perimeter optimization)
│   ├── pgd_optimizer.py            (394 lines — projected gradient descent)
│   ├── pyslsqp_optimizer.py        (295 lines — PySLSQP wrapper)
│   ├── migration_types.py          (108 lines — migration dataclasses)
│   ├── migration_utils.py          (94 lines — migration helpers)
│   ├── migration_detector.py       (311 lines — Type 1/2 trigger detection)
│   ├── migration_executor.py       (531 lines — snapshot + T1/T2 execution)
│   ├── migration_orchestrator.py   (264 lines — optimize–detect–migrate loop)
│   ├── one_ring_rebuilder.py       (199 lines — 1-ring rebuild after migration)
│   ├── type1_component_analyzer.py (1434 lines — Type 1 component analysis)
│   ├── type2_migration_history.py  (110 lines — in-memory migration history)
│   └── type2_migration_io.py       (198 lines — HDF5 I/O for migration history)
└── surfaces/                       (NO __init__.py)
    ├── torus.py                    (106 lines — torus mesh provider)
    └── ring.py                     (89 lines — ring mesh provider)
```

### `examples/` (current state)

```
examples/
├── data_loader.py                  (shared loading utility — imported by multiple scripts)
├── find_surface_partition.py       (Phase 1 entry point — Γ-convergence relaxation)
├── optimization_analyzer.py        (diagnostic/analysis tool for optimizer output)
├── visualize_partition.py          (PyVista visualization of solutions)
├── visualize_type1_vertex_collapse.py (Type 1 migration visualization)
├── visualize_type2_triple_point.py (Type 2 migration visualization)
├── visualize_partition_fast.py    (fast matplotlib partition visualization — added during IPOPT work)
└── debug_archive/                  (7 archived debug scripts — low priority)
    ├── README.md
    ├── test_initial_partitions.py
    ├── test_lambda_values.py
    ├── test_phase1_triangle_segments.py
    ├── test_phase2_perimeter_calculator.py
    ├── test_refined_contour_extraction.py
    └── test_variable_points.py
```

**Problem:** `data_loader.py` is library code masquerading as an example. It is imported by `testing/test_migrations_debug.py`, `testing/test_migration_and_continue.py`, and `testing/refine_perimeter_iterative.py` — it belongs in `src/`, not `examples/`.

**Boundary blurring with `testing/`:** `testing/refine_perimeter_iterative.py` and `testing/test_migration_and_continue.py` are pipeline scripts, not unit tests. The distinction between `examples/` (runnable science scripts) and `testing/` (integration/validation scripts) should be clarified.

## Proposed Structure

Group files by domain. Each subpackage has a clear purpose, an `__init__.py` with public exports, and a focused set of files.

### `src/` (Phase A + B)

```
src/
├── __init__.py              (top-level public API)
├── config.py                (stays — global config)
├── logging_config.py        (stays — used everywhere)
│
├── mesh/                    (mesh infrastructure)
│   ├── __init__.py
│   ├── tri_mesh.py          (TriMesh, FEM assembly)
│   ├── mesh_topology.py     (vertex/edge/triangle connectivity)
│   └── interpolation.py     (mesh-to-mesh interpolation)
│
├── partition/               (contour partition, calculators, vectorized evaluation)
│   ├── __init__.py
│   ├── find_contours.py     (ContourAnalyzer — HDF5 → indicator → level-set segments)
│   ├── contour_partition.py (VariablePoint, TriangleSegment, PartitionContour)
│   ├── area_calculator.py   (AreaCalculator)
│   ├── perimeter_calculator.py (PerimeterCalculator)
│   ├── steiner_handler.py   (TriplePoint, SteinerHandler)
│   ├── partition_arrays.py  (PartitionArrays — flat NumPy snapshot)
│   ├── vectorized_area.py   (vectorized area + area Jacobian)
│   ├── vectorized_perimeter.py (vectorized perimeter + gradient)
│   └── vectorized_steiner.py   (analytical Steiner, Fermat–Torricelli)
│
├── optimization/            (all optimizers)
│   ├── __init__.py
│   ├── perimeter_optimizer.py (constrained perimeter minimization via SLSQP/IPOPT
│   │                           + IPOPTProblemAdapter with sparse Jacobian,
│   │                           exact Hessian, and best-iterate tracking)
│   ├── pgd_optimizer.py     (projected gradient descent — Γ-convergence)
│   ├── pyslsqp_optimizer.py (PySLSQP wrapper — Γ-convergence)
│   └── projection.py        (orthogonal projection onto constraints)
│
├── migration/               (topology switch system)
│   ├── __init__.py
│   ├── migration_types.py   (dataclasses: triggers, snapshots, results)
│   ├── migration_utils.py   (low-level helpers, e.g. target vertex from λ)
│   ├── migration_detector.py (Type 1/2 trigger detection + conflict resolution)
│   ├── migration_executor.py (snapshot + Type 1/2 execution + rollback)
│   ├── migration_orchestrator.py (top-level optimize–detect–migrate loop)
│   ├── one_ring_rebuilder.py (1-ring rebuild after vertex relabel)
│   ├── type1_component_analyzer.py (connected components + conflict selection)
│   ├── type2_migration_history.py (in-memory migration history)
│   └── type2_migration_io.py (HDF5 save/load for migration history)
│
├── pipeline/                (Phase C — high-level pipeline orchestrator)
│   ├── __init__.py
│   ├── io.py                (data loading/saving — absorbs examples/data_loader.py)
│   └── pipeline_orchestrator.py (chains all stages: relax → refine → migrate → repeat)
│
├── surfaces/                (surface mesh providers)
│   ├── __init__.py
│   ├── torus.py
│   └── ring.py
│
└── visualization/           (all plotting)
    ├── __init__.py
    └── plot_utils.py        (2D matplotlib plots)
```

**Total: 31 active files.**

### `examples/` (after Phase B + C)

Once `data_loader.py` moves to `src/pipeline/io.py` and the pipeline orchestrator exists, `examples/` becomes a thin layer of entry-point scripts that call into the library. Each script becomes a short CLI wrapper with argument parsing and a single call into `src/`.

```
examples/
├── find_surface_partition.py       (CLI: Phase 1 relaxation — calls pipeline/pipeline_orchestrator.py)
├── refine_perimeter.py             (CLI: Phase 2/3 refinement — NEW thin wrapper, calls pipeline)
├── run_full_pipeline.py            (CLI: full end-to-end run — NEW, wraps all stages)
├── optimization_analyzer.py        (diagnostic tool — reads HDF5 results)
├── visualize_partition.py          (PyVista visualization)
├── visualize_type1_vertex_collapse.py
├── visualize_type2_triple_point.py
├── visualize_partition_fast.py          (fast matplotlib partition visualization)
└── debug_archive/                  (unchanged — archived debug scripts)
```

## Rationale for Each Subpackage

### `mesh/` — Mesh Infrastructure
Contains the foundational mesh data structure (`TriMesh`), its topology queries (`MeshTopology`), and interpolation between meshes. These are surface-agnostic and have no dependency on partition or optimization concepts.

### `partition/` — Partition Data Structures and Evaluation
Everything that defines and measures a partition on a mesh. This includes:
- **Contour extraction** (`find_contours.py`): the bridge between Phase 1 relaxation output and the `PartitionContour` structure. Placed here because its `BoundaryTriangleInfo` is imported by `contour_partition.py` and its output feeds directly into `PartitionContour` construction.
- **Core representation** (`contour_partition.py`): `VariablePoint`, `TriangleSegment`, `PartitionContour`.
- **OO calculators** (`area_calculator.py`, `perimeter_calculator.py`, `steiner_handler.py`): measure partition properties.
- **Vectorized evaluation** (`partition_arrays.py`, `vectorized_area.py`, `vectorized_perimeter.py`, `vectorized_steiner.py`): flat NumPy fast-path for the same area/perimeter/Steiner computations, used by the optimizer for performance. `partition_arrays.py` also stores Jacobian sparsity (`jac_row`, `jac_col`, `nnz_lookup`) and Hessian sparsity (`hess_row`, `hess_col`, `hess_offset_map`) structures for IPOPT. The vectorized modules provide both dense-returning functions (for SLSQP) and sparse-returning functions (for IPOPT's `jacobian()` / `hessian()` callbacks).

### `optimization/` — Optimizers
All optimization algorithms: the two Γ-convergence optimizers (`pgd_optimizer`, `pyslsqp_optimizer`), the perimeter optimizer (Section 5 of the paper), and the projection onto constraints. These share a common pattern (minimize a functional subject to constraints) but operate at different stages of the pipeline. `projection_iterative.py` is renamed to `projection.py` for clarity.

`perimeter_optimizer.py` now contains both the `PerimeterOptimizer` class and the `IPOPTProblemAdapter` class. The adapter wraps optimizer callbacks into the cyipopt problem interface, providing sparse Jacobian evaluation (via `compute_area_jacobian_sparse()` and `compute_steiner_area_jacobian_sparse()`), exact Hessian of the Lagrangian (via `compute_perimeter_hessian_sparse()`, `compute_area_hessian_sparse()`, plus finite-difference Steiner Hessian contributions), and best-iterate tracking. The `optimize()` method accepts `method='ipopt'` with options `exact_hessian`, `best_iterate`, and `lbfgs_memory`.

**Future split (not blocking):** Consider extracting `IPOPTProblemAdapter` into a dedicated `optimization/ipopt_adapter.py`. This would isolate the `cyipopt` dependency from the SLSQP path and keep `perimeter_optimizer.py` focused on the solver-agnostic `PerimeterOptimizer` class.

### `migration/` — Topology Switches
The modular migration system. Fully self-contained with clear internal layering:
- **Types** → **Detection** → **Rebuild** → **Execution** → **Orchestration**
- Includes the Type 1 component analyzer and Type 2 history tracking/persistence, which are tightly coupled to migration logic.

### `surfaces/` — Surface Providers
The extension point for closed surfaces (surfaces without boundary). The torus is the primary surface for the project. The ring (`ring.py`) was used during early planar tests but will be **removed** — the mathematical setting requires surfaces without boundary, and the ring (an annulus) does not satisfy this. After removal, `surfaces/` will contain only `torus.py`, but the subpackage is designed to accommodate future closed surfaces (e.g. sphere, genus-2 surface) as the research expands.

### `visualization/` — Plotting
Matplotlib plotting utilities. Used only by example/testing scripts, never by the computational core. The 3D visualization scripts use PyVista directly.

### `pipeline/` — Pipeline Orchestration (Phase C)
Two new files that serve different roles:

- **`io.py`**: absorbs `examples/data_loader.py`. Loading and saving HDF5 results, constructing `PartitionContour` and `TriMesh` objects from disk — this is library code, not example code. It belongs here so any script or agent can load data without depending on `examples/`.

- **`pipeline_orchestrator.py`**: chains the four computational stages into a single programmatic API:
  1. **Phase 1** — Γ-convergence relaxation (calls `optimization/pgd_optimizer` or `pyslsqp_optimizer`)
  2. **Phase 2** — Contour extraction (calls `partition/find_contours`)
  3. **Phase 3** — Perimeter refinement (calls `optimization/perimeter_optimizer`)
  4. **Phase 4** — Topology migration (calls `migration/migration_orchestrator`)
  5. Repeat phases 2–4 until convergence or max iterations

  This makes the full pipeline callable from a single entry point, testable as a unit, and accessible to external tools and AI agents without requiring script execution.

## Legacy Files

### Ring surface

The ring (planar annulus) was used for early testing but is mathematically out of scope — the project requires surfaces **without boundary**, and the ring does not satisfy this condition. The torus is the primary surface.

| File / location | Action |
|-----------------|--------|
| `src/surfaces/ring.py` | **Remove.** |
| `parameters/input.yaml` — `n_radial`, `n_angular`, `r_inner`, `r_outer`, `n_radial_increment`, `n_angular_increment` | **Remove** ring-specific keys; keep only torus parameters. |
| `examples/find_surface_partition.py` — `--surface ring` branch + `RingMeshProvider` import | **Remove.** |
| Metadata serialization block — `r_inner`, `r_outer` surface params | **Remove** ring branch; keep torus branch only. |
| `src/surfaces/__init__.py` (new) | Export only `TorusMeshProvider`. |
| `README.md`, documentation | Remove ring references; clarify that the codebase targets closed surfaces. |

## Import Path Changes

The restructure changes import paths. Examples of the most common transitions:

| Before | After |
|--------|-------|
| `from src.core.tri_mesh import TriMesh` | `from src.mesh.tri_mesh import TriMesh` |
| `from src.core.contour_partition import PartitionContour` | `from src.partition.contour_partition import PartitionContour` |
| `from src.find_contours import ContourAnalyzer` | `from src.partition.find_contours import ContourAnalyzer` |
| `from src.core.perimeter_optimizer import PerimeterOptimizer` | `from src.optimization.perimeter_optimizer import PerimeterOptimizer` |
| `from src.core.pgd_optimizer import ProjectedGradientOptimizer` | `from src.optimization.pgd_optimizer import ProjectedGradientOptimizer` |
| `from src.core.pyslsqp_optimizer import PySLSQPOptimizer` | `from src.optimization.pyslsqp_optimizer import PySLSQPOptimizer` |
| `from src.projection_iterative import ...` | `from src.optimization.projection import ...` |
| `from src.core.migration_orchestrator import MigrationOrchestrator` | `from src.migration.migration_orchestrator import MigrationOrchestrator` |
| `from src.plot_utils import ...` | `from src.visualization.plot_utils import ...` |

To minimize disruption, `src/__init__.py` can re-export key symbols at the old paths during a transition period.

## Consumer Scripts That Need Import Updates

### `examples/`

| Script | Imports affected |
|--------|-----------------|
| `find_surface_partition.py` | Main driver — config, optimizers, mesh, surfaces |
| `data_loader.py` | `ContourAnalyzer`, `TriMesh`, `PartitionContour` |
| `optimization_analyzer.py` | Diagnostic tool — may import config, plotting |
| `visualize_partition.py` | Plotting, `TriMesh`, `PartitionContour` |
| `visualize_type1_vertex_collapse.py` | Migration + plotting |
| `visualize_type2_triple_point.py` | Migration + plotting |

### `testing/`

| Script | Imports affected |
|--------|-----------------|
| `refine_perimeter_iterative.py` | `ContourAnalyzer`, `TriMesh`, partition, migration, optimization |
| `test_migration_and_continue.py` | `ContourAnalyzer`, migration, partition |
| `test_migrations_debug.py` | `data_loader`, migration, partition |
| `test_vectorized_evaluation.py` | Vectorized modules, `PartitionArrays`, calculators |

### `examples/debug_archive/`

Seven archived debug scripts with legacy import paths. Update as a batch; these are low priority since they are archived.

## Execution Strategy

The restructure is organized into three sequential phases.

### Phase A — Reorganize `src/core/` into subpackages

1. **Add `__init__.py` files** to all new subpackages.
2. **Move files one subpackage at a time**, in dependency order:
   - `mesh/` first (no internal deps)
   - `partition/` second (depends on `mesh/`)
   - `optimization/` third (depends on `partition/`)
   - `migration/` fourth (depends on `partition/` and `optimization/`)
   - `surfaces/` (depends on `mesh/`)
   - `visualization/` last (standalone)
3. **Update imports** in all consumer scripts after each subpackage move. Use `git grep` to find all import statements.
4. **Run all tests** after each subpackage move to catch broken imports immediately.
5. **Update `src/__init__.py`** to re-export from new locations for backward compatibility.

### Phase B — Elevate `examples/data_loader.py` into `src/`

1. Move `data_loader.py` to `src/pipeline/io.py`.
2. Update all importers (`testing/test_migrations_debug.py`, `testing/test_migration_and_continue.py`, `testing/refine_perimeter_iterative.py`, and any `examples/` scripts that use it).
3. Clarify `examples/` vs `testing/`: scripts that are integration/validation pipelines (`refine_perimeter_iterative.py`, `test_migration_and_continue.py`) may belong in `testing/` or should be clearly labeled as pipeline scripts rather than unit tests.

### Phase C — Implement the pipeline orchestrator

**Source of truth:** `testing/refine_perimeter_iterative.py` is the current production entry point for phases 2–4 (contour extraction → perimeter refinement → topology migration). Its `main()` function contains the complete iterate-refine loop: optimize → detect → export → migrate → repeat with checkpointing and resume. The pipeline orchestrator will be built by extracting this logic into a library API.

1. Design the `PipelineOrchestrator` API: inputs (config, surface, mesh resolutions, seeds), outputs (HDF5 paths, final partition state, per-stage logs).
2. Implement `src/pipeline/pipeline_orchestrator.py` by extracting the iterate-refine loop from `testing/refine_perimeter_iterative.py` into a callable library class.
3. The orchestrator API must expose IPOPT-specific options that are currently CLI flags in `testing/refine_perimeter_iterative.py` and parameters on `PerimeterOptimizer.optimize()`: `method` (`'SLSQP'` vs `'ipopt'`), `exact_hessian`, `best_iterate`, `lbfgs_memory`, and `allow_partial_convergence`.
4. Refactor `examples/find_surface_partition.py` into a thin CLI wrapper that calls `PipelineOrchestrator`. Create `examples/refine_perimeter.py` as a thin CLI wrapper for the refinement stages.
5. Optionally add `examples/run_full_pipeline.py` as a single end-to-end entry point.
6. After extraction, `testing/refine_perimeter_iterative.py` can be reduced to a thin integration test that calls the pipeline orchestrator with a known input and asserts convergence.

### Cleanup pass (any phase)

- Remove ring surface files and references (see §Legacy Files — Ring surface).
- Update `README.md` to remove stale references (ring surface, etc.).
- Fix `pyproject.toml` version mismatch (`0.1.0` → `0.2.0`) and `testpaths` (points to nonexistent `tests/`; should be `testing/`).
- Remove the broken link to `docs/PERIMETER_REFINEMENT.md` from `README.md`.
- Fix the typo on line 23 of `parameters/input.yaml` (`# Optimization parametersJeez`).

### Dependencies

`cyipopt` is an optional dependency required for `--method ipopt`. The restructure is a good time to formalize this in `pyproject.toml` as an optional extra (e.g. `pip install .[ipopt]`). Core functionality (SLSQP path, Γ-convergence, migration) must remain usable without `cyipopt` installed. The `IPOPTProblemAdapter` import is already guarded by a `try/except ImportError` in `perimeter_optimizer.py`.

## Dependency Flow Between Subpackages

```
surfaces/ ──→ mesh/
partition/ ──→ mesh/
optimization/ ──→ partition/ ──→ mesh/
migration/ ──→ partition/ ──→ mesh/
           ──→ optimization/
pipeline/ ──→ optimization/
          ──→ migration/
          ──→ partition/
          ──→ surfaces/
visualization/ (standalone, no src/ deps)
```

No circular dependencies. Each arrow means "imports from." `pipeline/` sits at the top of the dependency tree and is the only layer that `examples/` scripts need to import directly.

**Note on `optimization/ → partition/` coupling:** `IPOPTProblemAdapter.jacobian()` and `_hessian_impl()` call directly into `partition/vectorized_area`, `partition/vectorized_perimeter`, and `partition/vectorized_steiner` (not just indirectly through `PerimeterOptimizer`'s existing methods). This is intentional — the adapter needs the sparse-output variants (`compute_area_jacobian_sparse`, `compute_steiner_area_jacobian_sparse`, `compute_perimeter_hessian_sparse`, etc.) that write directly into flat `(nnz,)` / `(hess_nnz,)` value arrays using the `nnz_lookup` and `hess_offset_map` tables. These sparse functions are distinct from the dense-returning public APIs used by the SLSQP path. The dependency is `optimization/ → partition/` (same direction as before), so no new circular risk.
