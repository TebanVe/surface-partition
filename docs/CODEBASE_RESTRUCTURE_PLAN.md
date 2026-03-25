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

### `src/` (34 files)

```
src/
├── __init__.py                     (package exports, __version__ = "0.2.0")
├── config.py                       (131 lines — configuration)
├── logging_config.py               (284 lines — logging setup)
├── find_contours.py                (296 lines — contour extraction from HDF5 solutions)
├── projection_iterative.py         (363 lines — projection onto constraints)
├── plot_utils.py                   (411 lines — 2D plotting)
├── plot_utils_3d.py                (160 lines — 3D PyVista plotting)
├── core/                           (NO __init__.py — 25 files)
│   ├── tri_mesh.py                 (311 lines — mesh container + FEM)
│   ├── mesh_topology.py            (204 lines — mesh connectivity)
│   ├── interpolation.py            (23 lines — nearest-neighbor interp)
│   ├── contour_partition.py        (1305 lines — VP, segments, partition)
│   ├── area_calculator.py          (511 lines — area computation)
│   ├── perimeter_calculator.py     (274 lines — perimeter computation)
│   ├── steiner_handler.py          (457 lines — triple point Steiner trees)
│   ├── partition_arrays.py         (73 lines — flat NumPy snapshot for vectorized eval)
│   ├── vectorized_area.py          (279 lines — vectorized area + Jacobian)
│   ├── vectorized_perimeter.py     (94 lines — vectorized perimeter + gradient)
│   ├── vectorized_steiner.py       (181 lines — analytical Steiner vectorized)
│   ├── perimeter_optimizer.py      (846 lines — SLSQP perimeter optimization)
│   ├── pgd_optimizer.py            (394 lines — projected gradient descent)
│   ├── pyslsqp_optimizer.py        (306 lines — PySLSQP wrapper)
│   ├── migration_types.py          (108 lines — migration dataclasses)
│   ├── migration_utils.py          (94 lines — migration helpers)
│   ├── migration_detector.py       (328 lines — Type 1/2 trigger detection)
│   ├── migration_executor.py       (531 lines — snapshot + T1/T2 execution)
│   ├── migration_orchestrator.py   (308 lines — optimize–detect–migrate loop)
│   ├── one_ring_rebuilder.py       (199 lines — 1-ring rebuild after migration)
│   ├── type1_component_analyzer.py (1434 lines — Type 1 component analysis)
│   ├── type2_migration_history.py  (203 lines — in-memory migration history)
│   ├── type2_migration_io.py       (198 lines — HDF5 I/O for migration history)
│   ├── topology_switcher.py        (3824 lines — LEGACY monolithic migration)
│   └── topology_switcher_legacy.py (3824 lines — LEGACY duplicate)
└── surfaces/                       (NO __init__.py)
    ├── torus.py                    (106 lines — torus mesh provider)
    └── ring.py                     (89 lines — ring mesh provider)
```

### `examples/` (current state)

```
examples/
├── data_loader.py                  (shared loading utility — imported by multiple scripts)
├── find_surface_partition.py       (Phase 1 entry point — Γ-convergence relaxation)
├── refine_perimeter.py             (Phase 2/3 entry point — perimeter refinement)
├── optimization_analyzer.py        (diagnostic/analysis tool for optimizer output)
├── visualize_partition.py          (PyVista visualization of solutions)
├── visualize_type1_vertex_collapse.py (Type 1 migration visualization)
├── visualize_type2_triple_point.py (Type 2 migration visualization)
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
│   ├── perimeter_optimizer.py (constrained perimeter minimization via SLSQP)
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
    ├── plot_utils.py        (2D matplotlib plots)
    └── plot_utils_3d.py     (3D PyVista plots)
```

**Total: 33 active files** (two legacy `topology_switcher*.py` files removed; see §Legacy Files below).

### `examples/` (after Phase B + C)

Once `data_loader.py` moves to `src/pipeline/io.py` and the pipeline orchestrator exists, `examples/` becomes a thin layer of entry-point scripts that call into the library. Each script becomes a short CLI wrapper with argument parsing and a single call into `src/`.

```
examples/
├── find_surface_partition.py       (CLI: Phase 1 relaxation — calls pipeline/pipeline_orchestrator.py)
├── refine_perimeter.py             (CLI: Phase 2/3 refinement — calls pipeline/pipeline_orchestrator.py)
├── run_full_pipeline.py            (CLI: full end-to-end run — NEW, wraps all stages)
├── optimization_analyzer.py        (diagnostic tool — reads HDF5 results)
├── visualize_partition.py          (PyVista visualization)
├── visualize_type1_vertex_collapse.py
├── visualize_type2_triple_point.py
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
- **Vectorized evaluation** (`partition_arrays.py`, `vectorized_area.py`, `vectorized_perimeter.py`, `vectorized_steiner.py`): flat NumPy fast-path for the same area/perimeter/Steiner computations, used by the SLSQP optimizer for performance.

### `optimization/` — Optimizers
All optimization algorithms: the two Γ-convergence optimizers (`pgd_optimizer`, `pyslsqp_optimizer`), the perimeter optimizer (Section 5 of the paper), and the projection onto constraints. These share a common pattern (minimize a functional subject to constraints) but operate at different stages of the pipeline. `projection_iterative.py` is renamed to `projection.py` for clarity.

### `migration/` — Topology Switches
The modular migration system that replaces the legacy monolithic `topology_switcher.py`. Fully self-contained with clear internal layering:
- **Types** → **Detection** → **Rebuild** → **Execution** → **Orchestration**
- Includes the Type 1 component analyzer and Type 2 history tracking/persistence, which are tightly coupled to migration logic.

### `surfaces/` — Surface Providers
The extension point for closed surfaces (surfaces without boundary). The torus is the primary surface for the project. The ring (`ring.py`) was used during early planar tests but will be **removed** — the mathematical setting requires surfaces without boundary, and the ring (an annulus) does not satisfy this. After removal, `surfaces/` will contain only `torus.py`, but the subpackage is designed to accommodate future closed surfaces (e.g. sphere, genus-2 surface) as the research expands.

### `visualization/` — Plotting
All matplotlib and PyVista plotting utilities. Used only by example/testing scripts, never by the computational core.

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

The following files should be removed as part of the restructure.

### Monolithic topology switcher

The modular migration system (`migration_detector.py`, `migration_executor.py`, `migration_orchestrator.py`, etc.) is its replacement and is already in use.

| File | Action |
|------|--------|
| `topology_switcher.py` (3824 lines) | **Remove.** |
| `topology_switcher_legacy.py` (3824 lines) | **Remove.** Duplicate of `topology_switcher.py`, kept as a safety copy. No longer needed once the modular system is validated. |

If a fallback is desired during transition, move both files to a `_legacy/` directory outside `src/` (or a `legacy/` subdirectory within `migration/`) rather than keeping them as active code.

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
| `from src.core.topology_switcher import TopologySwitcher` | **Removed** (use modular migration system) |
| `from src.plot_utils import ...` | `from src.visualization.plot_utils import ...` |
| `from src.plot_utils_3d import ...` | `from src.visualization.plot_utils_3d import ...` |

To minimize disruption, `src/__init__.py` can re-export key symbols at the old paths during a transition period.

## Consumer Scripts That Need Import Updates

### `examples/`

| Script | Imports affected |
|--------|-----------------|
| `find_surface_partition.py` | Main driver — config, optimizers, mesh, surfaces |
| `data_loader.py` | `ContourAnalyzer`, `TriMesh`, `PartitionContour` |
| `refine_perimeter.py` | `ContourAnalyzer`, `TriMesh`, partition + optimization |
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

The topology switch system is implemented and under active testing. The restructure can proceed once the current test run completes. It is organized into three sequential phases.

### Phase A — Reorganize `src/core/` into subpackages

1. **Validate the modular migration system** passes all current tests before removing legacy files.
2. **Add `__init__.py` files** to all new subpackages.
3. **Move files one subpackage at a time**, in dependency order:
   - `mesh/` first (no internal deps)
   - `partition/` second (depends on `mesh/`)
   - `optimization/` third (depends on `partition/`)
   - `migration/` fourth (depends on `partition/` and `optimization/`)
   - `surfaces/` (depends on `mesh/`)
   - `visualization/` last (standalone)
4. **Update imports** in all consumer scripts after each subpackage move. Use `git grep` to find all import statements.
5. **Run all tests** after each subpackage move to catch broken imports immediately.
6. **Update `src/__init__.py`** to re-export from new locations for backward compatibility.
7. **Remove legacy files** (`topology_switcher.py`, `topology_switcher_legacy.py`) after confirming the modular migration system covers all use cases.

### Phase B — Elevate `examples/data_loader.py` into `src/`

1. Move `data_loader.py` to `src/pipeline/io.py`.
2. Update all importers (`testing/test_migrations_debug.py`, `testing/test_migration_and_continue.py`, `testing/refine_perimeter_iterative.py`, and any `examples/` scripts that use it).
3. Clarify `examples/` vs `testing/`: scripts that are integration/validation pipelines (`refine_perimeter_iterative.py`, `test_migration_and_continue.py`) may belong in `testing/` or should be clearly labeled as pipeline scripts rather than unit tests.

### Phase C — Implement the pipeline orchestrator

1. Design the `PipelineOrchestrator` API: inputs (config, surface, mesh resolutions, seeds), outputs (HDF5 paths, final partition state, per-stage logs).
2. Implement `src/pipeline/pipeline_orchestrator.py` to chain all four stages.
3. Refactor `examples/find_surface_partition.py` and `examples/refine_perimeter.py` into thin CLI wrappers that call `PipelineOrchestrator`.
4. Optionally add `examples/run_full_pipeline.py` as a single end-to-end entry point.

### Cleanup pass (any phase)

- Remove ring surface files and references (see §Legacy Files — Ring surface).
- Update `README.md` (still references deleted files: `island_analysis.py`, `ring_visualization.py`, ring surface, etc.).
- Fix `pyproject.toml` version mismatch (`0.1.0` → `0.2.0`) and `testpaths` (points to nonexistent `tests/`; should be `testing/`).
- Remove the broken link to `docs/PERIMETER_REFINEMENT.md` from `README.md`.
- Fix the typo on line 23 of `parameters/input.yaml` (`# Optimization parametersJeez`).

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
