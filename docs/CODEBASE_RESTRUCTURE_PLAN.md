# Codebase Restructure Plan

This document describes a proposed restructuring of the `src/` and `examples/` directories to improve organization, readability, and maintainability. The goal is to make the codebase easy to navigate for both human developers and AI agents that may interact with it programmatically.

The restructure is organized into a prerequisite step and three phases:

- **Prerequisite** — Standardize the import mechanism across the entire codebase (remove dual-import `try/except` blocks, unify `sys.path` usage).
- **Phase A** — Reorganize `src/core/` into focused subpackages.
- **Phase B** — Move `examples/data_loader.py` into `src/` as proper library code; extract shared visualization helpers; clarify the role of `examples/` and `testing/`.
- **Phase C** — Implement a `src/pipeline/` orchestrator that chains all stages programmatically, reducing entry-point scripts to thin wrappers.

---

## Problem

The current layout has three issues:

1. **`core/` is overloaded** — 23 files mixing five unrelated domains: mesh infrastructure, partition data structures, vectorized evaluation, optimization algorithms, and topology migration. A developer looking for "the optimizer" must scan past mesh connectivity, contour partitions, migration history, and vectorized calculators.

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
├── core/                           (NO __init__.py — 23 files)
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

**Problem:** `data_loader.py` is library code masquerading as an example. It is imported by **6 consumer scripts**:
- `testing/refine_perimeter_iterative.py`
- `testing/test_migrations_debug.py`
- `examples/visualize_partition.py`
- `examples/visualize_partition_fast.py`
- `examples/visualize_type1_vertex_collapse.py`
- `examples/visualize_type2_triple_point.py`

It belongs in `src/`, not `examples/`.

**Cross-imports between example scripts:** Two scripts import helpers from `examples/visualize_type2_triple_point.py`:

- `examples/visualize_partition_fast.py` imports: `compute_cell_portion_in_triangle_simple`, `compute_triple_point_cell_portion`, `add_steiner_visualization`, `add_vp_visualization`.
- `examples/visualize_partition.py` imports: `render_single_region_simple`, `add_steiner_visualization`, `add_vp_visualization`.

The full set of shared helpers to extract is therefore five functions: `compute_cell_portion_in_triangle_simple`, `compute_triple_point_cell_portion`, `add_steiner_visualization`, `add_vp_visualization`, and `render_single_region_simple`. These are library code inside an example script. During Phase B they should be extracted into `src/visualization/partition_helpers.py` so that example scripts do not import from each other.

**Boundary blurring with `testing/`:** `testing/refine_perimeter_iterative.py` is a pipeline script, not a unit test. The distinction between `examples/` (runnable science scripts) and `testing/` (integration/validation scripts) should be clarified. `testing/README_testing.md` tracks tests that were developed before moving scripts to the production directory (`examples/`); it should be updated during the restructure to reflect the final layout.

**`scripts/` directory (out of scope):** `scripts/submit.sh` is a cluster job submission script used when running the project on supercomputers. It is not part of the restructure and should be left as-is. It does contain ring-related defaults (`--surface ring`, ring parameter extraction) that may need a future update once ring removal is complete, but this is non-blocking.

## Proposed Structure

Group files by domain. Each subpackage has a clear purpose, an `__init__.py` with public exports, and a focused set of files.

### `src/` (Phase A + B)

```
src/
├── __init__.py              (minimal — just __version__)
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
│   └── torus.py             (ring.py removed — see §Legacy Files)
│
└── visualization/           (all plotting)
    ├── __init__.py
    ├── plot_utils.py        (2D matplotlib plots)
    └── partition_helpers.py (shared PyVista helpers for partition visualization —
                              extracted from examples/visualize_type2_triple_point.py)
```

**Total: 40 files (32 implementation modules + 8 `__init__.py`).**

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
Matplotlib plotting utilities and shared PyVista helpers. Used only by example/testing scripts, never by the computational core. `plot_utils.py` provides 2D matplotlib plots. `partition_helpers.py` (extracted from `examples/visualize_type2_triple_point.py` during Phase B) provides shared PyVista helper functions for partition rendering — cell portion computation, region rendering, Steiner point visualization, and VP visualization. These helpers are imported by `visualize_partition.py`, `visualize_type2_triple_point.py`, and `visualize_partition_fast.py`.

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
| `examples/find_surface_partition.py` — `--surface ring` branch + `RingMeshProvider` import | **Remove.** Also change the `--surface` default from `'ring'` to `'torus'` (ring is currently the default). |
| `examples/optimization_analyzer.py` — `from surfaces.ring import RingMeshProvider` | **Remove** ring import and any ring-specific branches. |
| Metadata serialization block — `r_inner`, `r_outer` surface params | **Remove** ring branch; keep torus branch only. |
| `src/surfaces/__init__.py` (new) | Export only `TorusMeshProvider`. |
| `src/plot_utils.py` — `plot_ring_mesh` function, `RingMesh` docstrings, `r_inner`/`r_outer`/`n_radial`/`n_angular` parameters | **Remove** ring-specific plotting function and parameters; keep only torus/generic plotting. |
| `src/__init__.py` — `plot_ring_mesh` export | **Remove** ring-related re-exports. |
| `src/config.py` — "ring partition" docstring, "Ring parameters" section, `n_radial`, `n_angular`, `r_inner`, `r_outer`, increments | **Remove** ring parameter definitions and ring references from docstrings. |
| `src/logging_config.py` — log filenames `ring_partition_{timestamp}.log`, `ring_partition.log` | **Rename** log files to a surface-agnostic name (e.g. `surface_partition_*.log`). |
| `src/surfaces/torus.py` — comment "mirrors RingMeshProvider" | **Remove** the ring reference from the comment. |
| `pyproject.toml` — project name `ring-partition` | **Rename** to `surface-partition` (or similar) to reflect the codebase scope. |
| `README.md`, documentation | Remove ring references; clarify that the codebase targets closed surfaces. Also remove the stale reference to `examples/ring_visualization.py` (file does not exist). |

### `testing/test_vectorized_evaluation.py`

This script was a one-time verification that the vectorized evaluation modules produce results equivalent to the object-oriented implementations. That verification is complete and the results are trusted. **Delete `testing/test_vectorized_evaluation.py`** during the cleanup pass.

## Import Path Changes

The restructure changes import paths. The table below covers **both**
import styles found in the codebase (the `src.xxx` style used by most
scripts, and the flat `xxx` style used by `find_surface_partition.py` and
`optimization_analyzer.py` — see §Prerequisite for why both exist).

| Before (repo-root style) | Before (flat style, `sys.path → src/`) | After |
|--------|--------|-------|
| `from src.core.tri_mesh import TriMesh` | `from core.tri_mesh import TriMesh` | `from src.mesh.tri_mesh import TriMesh` |
| `from src.core.contour_partition import PartitionContour` | `from core.contour_partition import PartitionContour` | `from src.partition.contour_partition import PartitionContour` |
| `from src.find_contours import ContourAnalyzer` | `from find_contours import ContourAnalyzer` | `from src.partition.find_contours import ContourAnalyzer` |
| `from src.core.perimeter_optimizer import PerimeterOptimizer` | `from core.perimeter_optimizer import PerimeterOptimizer` | `from src.optimization.perimeter_optimizer import PerimeterOptimizer` |
| `from src.core.pgd_optimizer import ProjectedGradientOptimizer` | `from core.pgd_optimizer import ProjectedGradientOptimizer` | `from src.optimization.pgd_optimizer import ProjectedGradientOptimizer` |
| `from src.core.pyslsqp_optimizer import PySLSQPOptimizer` | `from core.pyslsqp_optimizer import PySLSQPOptimizer` | `from src.optimization.pyslsqp_optimizer import PySLSQPOptimizer` |
| `from ..projection_iterative import ...` (relative, in `src/core/pgd_optimizer.py`) | `from projection_iterative import ...` | `from src.optimization.projection import ...` |
| `from src.core.migration_orchestrator import MigrationOrchestrator` | `from core.migration_orchestrator import MigrationOrchestrator` | `from src.migration.migration_orchestrator import MigrationOrchestrator` |
| `from src.plot_utils import ...` | — | `from src.visualization.plot_utils import ...` |
| `from src.logging_config import ...` | `from logging_config import ...` | `from src.logging_config import ...` (unchanged) |
| `from examples.data_loader import ...` | — | `from src.pipeline.io import ...` |
| `from examples.visualize_type2_triple_point import ...` | — | `from src.visualization.partition_helpers import ...` |

After the Prerequisite step, only the "After" column will exist. All
consumer scripts will use `sys.path → repo_root` and `from src.xxx`
imports exclusively.

## Consumer Scripts That Need Import Updates

### `examples/`

| Script | Imports affected | Notes |
|--------|-----------------|-------|
| `find_surface_partition.py` | `core.*` (flat), `logging_config`, `surfaces.*`, `find_contours` | Uses `sys.path → src/`; must switch to repo-root style |
| `data_loader.py` | `src.core.contour_partition`, `src.core.tri_mesh`, `src.find_contours` | Moves to `src/pipeline/io.py` in Phase B |
| `optimization_analyzer.py` | `logging_config`, `surfaces.ring`, `surfaces.torus` | Uses `sys.path → src/`; ring import must be removed |
| `visualize_partition.py` | `src.core.*`, `src.find_contours`, `examples.data_loader`, **`examples.visualize_type2_triple_point`** | Uses `sys.path → repo_root`; cross-import from another example |
| `visualize_type1_vertex_collapse.py` | `src.core.*`, `examples.data_loader` | Uses `sys.path → repo_root` |
| `visualize_type2_triple_point.py` | `src.core.*`, `examples.data_loader` | Uses `sys.path → repo_root`; exports helpers imported by `visualize_partition.py` and `visualize_partition_fast.py` |
| `visualize_partition_fast.py` | `src.core.*`, `examples.data_loader`, **`examples.visualize_type2_triple_point`** | Uses `sys.path → repo_root`; cross-import from another example |

### `testing/`

| Script | Imports affected | Notes |
|--------|-----------------|-------|
| `refine_perimeter_iterative.py` | `src.core.*`, `src.find_contours`, `src.logging_config`, `examples.data_loader` | Uses `sys.path → repo_root` |
| `test_migrations_debug.py` | `src.core.*`, `src.logging_config`, `examples.data_loader` | Uses **double** `sys.path`: repo_root + `examples/` |

### `examples/debug_archive/`

Seven archived debug scripts with legacy import paths. These use the flat
`core.*` import style (e.g. `from core.tri_mesh import TriMesh`) with no
`sys.path` pointing to repo root — they expect `sys.path` to contain `src/`.
Update as a batch to repo-root `from src.xxx` style; low priority since
they are archived.

## Execution Strategy

The restructure is organized into a prerequisite step followed by three
sequential phases.

### Prerequisite — Standardize the import mechanism

The codebase currently uses **two incompatible import mechanisms** that must
be unified before any files are moved.

**Current state:** many files inside `src/core/` contain a `try/except`
block that attempts relative package imports first, then falls back to
`sys.path` manipulation. **Not all files use this pattern** — the split is:

- **Files WITH dual-import `try/except`** (16 of 23): `tri_mesh.py`,
  `mesh_topology.py`, `contour_partition.py`, `area_calculator.py`,
  `perimeter_calculator.py`, `steiner_handler.py`, `perimeter_optimizer.py`,
  `pgd_optimizer.py`, `pyslsqp_optimizer.py`, `migration_types.py`,
  `migration_utils.py`, `migration_detector.py`, `migration_executor.py`,
  `migration_orchestrator.py`, `one_ring_rebuilder.py`,
  `type1_component_analyzer.py`.
- **Files WITHOUT dual-import** (7 of 23): `interpolation.py`,
  `partition_arrays.py`, `vectorized_area.py`, `vectorized_perimeter.py`,
  `vectorized_steiner.py`, `type2_migration_history.py`,
  `type2_migration_io.py`. These already use clean relative imports only.

The typical dual-import pattern looks like this (note: some files use
`except Exception` instead of `except ImportError`):

```python
# Pattern found in ~16 src/core/*.py files:
try:
    from ..logging_config import get_logger      # ← relative (package mode)
    from .tri_mesh import TriMesh
except ImportError:                               # (or except Exception)
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger         # ← flat (standalone mode)
    from core.tri_mesh import TriMesh             # ← uses old 'core.' prefix
```

Consumer scripts also disagree on how to reach `src/`:

| Script | `sys.path` target | Import style |
|--------|-------------------|--------------|
| `examples/find_surface_partition.py` | `sys.path.append(src/)` | `from core.xxx`, `from logging_config` |
| `examples/visualize_partition.py` | `sys.path.insert(0, repo_root)` | `from src.core.xxx`, `from src.logging_config` |
| `examples/optimization_analyzer.py` | `sys.path.append(src/)` | `from surfaces.ring`, `from logging_config` |
| `testing/refine_perimeter_iterative.py` | `sys.path.append(repo_root)` | `from src.core.xxx`, `from examples.data_loader` |

**Decision: standardize on repo-root `sys.path` + `from src.xxx` imports.**

Steps:

1. **In every `src/core/*.py` that has a dual-import block (see list
   above), `src/find_contours.py`, and `src/projection_iterative.py`:**
   remove the `try/except` dual-import blocks and any `sys.path`
   manipulation entirely.  Replace them with **relative imports only**
   (e.g. `from ..logging_config import get_logger`, `from .tri_mesh import
   TriMesh`).  This makes `src/` a proper Python package.  Files inside
   `src/` should never manipulate `sys.path`.

   **Note:** `src/find_contours.py` does not have a `try/except` block; it
   uses an unconditional `sys.path.append` + flat import.  Remove the
   `sys.path` line and convert to relative imports.
   `src/projection_iterative.py` has a minimal `try/except` that only
   covers `logging_config` (using `from .logging_config`, not `from
   ..logging_config`).  Convert it to a clean relative import.
   The 7 `src/core/` files that already use clean relative imports need no
   changes for this step.

2. **In every `examples/*.py` and `testing/*.py` consumer script:**
   standardize on adding the **repo root** to `sys.path` (not `src/`):
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   ```
   Then import using `from src.xxx` (e.g. `from src.mesh.tri_mesh import
   TriMesh`).  This is already used by `visualize_partition.py` and
   `refine_perimeter_iterative.py`.

3. **Rewrite `src/__init__.py`** from scratch.  The current file exports
   symbols from old paths (`from .core.pgd_optimizer import ...`,
   `from .projection_iterative import ...`).  After Phase A, all of these
   paths change.  Replace with a minimal file:
   ```python
   __version__ = "0.2.0"
   ```
   Consumers import directly from subpackages (e.g. `from src.mesh.tri_mesh
   import TriMesh`).  This avoids maintaining a fragile re-export list and
   makes import paths explicit.

4. **Clean `__pycache__`** after each subpackage move:
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   ```
   Stale `.pyc` files from old paths will cause ghost imports.

**This step must be completed before Phase A** because moving files from
`core/` to `mesh/`, `partition/`, etc. will break both branches of the
`try/except` blocks simultaneously, making failures hard to diagnose.

### Phase A — Reorganize `src/core/` into subpackages

1. **Add `__init__.py` files** to all new subpackages.
2. **Move files one subpackage at a time**, in dependency order:
   - `mesh/` first (no internal deps)
   - `partition/` second (depends on `mesh/`)
   - `optimization/` third (depends on `partition/`)
   - `migration/` fourth (depends on `partition/` and `mesh/`)
   - `surfaces/` (depends on `mesh/`)
   - `visualization/` last (standalone)
3. **Update imports** in all consumer scripts after each subpackage move. Use `git grep` to find all import statements referencing the moved module names.
4. **Validate** after each subpackage move (see §Validation below).

### Phase B — Elevate `examples/data_loader.py` into `src/`

1. Move `data_loader.py` to `src/pipeline/io.py`.
2. Update **all 6 importers** (listed in §Current Structure above):
   - `testing/refine_perimeter_iterative.py`
   - `testing/test_migrations_debug.py`
   - `examples/visualize_partition.py`
   - `examples/visualize_partition_fast.py`
   - `examples/visualize_type1_vertex_collapse.py`
   - `examples/visualize_type2_triple_point.py`
3. Extract the five shared visualization helpers from
   `examples/visualize_type2_triple_point.py` into
   `src/visualization/partition_helpers.py`:
   `compute_cell_portion_in_triangle_simple`, `compute_triple_point_cell_portion`,
   `add_steiner_visualization`, `add_vp_visualization`, and
   `render_single_region_simple`. Then update both
   `visualize_partition_fast.py` and `visualize_partition.py` to import
   from `src.visualization.partition_helpers` instead of from another
   example script.
4. `testing/refine_perimeter_iterative.py` remains in `testing/` as an
   integration/pipeline script. It is not a unit test; its name reflects
   its role as an integration validation script.

### Phase C — Implement the pipeline orchestrator

The pipeline orchestrator must combine logic from **two separate scripts**
that currently implement different stages of the pipeline:

| Script | Stages covered | Key logic to extract |
|--------|---------------|---------------------|
| `examples/find_surface_partition.py` | Phase 1 (Γ-convergence relaxation) + contour extraction + initial perimeter logging | Surface/mesh construction, PGD/PySLSQP optimizer invocation, HDF5 solution save, contour extraction, initial perimeter calculation and metadata logging |
| `testing/refine_perimeter_iterative.py` | Phases 2–4 (perimeter refinement → topology migration → repeat) | The iterate-refine loop: optimize → detect → export → migrate → repeat, with checkpointing, resume, and IPOPT option handling |

`testing/refine_perimeter_iterative.py` is the more complex of the two. Its
`main()` function contains the complete iterate-refine loop with
checkpointing and resume. The pipeline orchestrator will be built primarily
by extracting this logic, while also absorbing the Phase 1 logic from
`find_surface_partition.py`.

**Phase numbering caveat:** The phase labels in this plan (Phase 1 =
relaxation, Phase 2 = contour extraction, Phase 3 = perimeter refinement,
Phase 4 = topology migration) do **not** match the internal labels inside
`refine_perimeter_iterative.py`, which uses its own numbering within the
iterate-refine loop: "Phase 1" = optimize perimeter, "Phase 2" = detect
topology switches, "Phase 3" = export checkpoint, "Phase 4" = execute
migration. When extracting logic, refer to the **functionality** (optimize,
detect, export, migrate), not the phase numbers.

1. Design the `PipelineOrchestrator` API: inputs (config, surface, mesh resolutions, seeds), outputs (HDF5 paths, final partition state, per-stage logs).
2. Implement `src/pipeline/pipeline_orchestrator.py` by extracting:
   - Phase 1 (relaxation + contour extraction) from `examples/find_surface_partition.py`
   - Phases 2–4 (refinement + migration loop) from `testing/refine_perimeter_iterative.py`
3. The orchestrator API must expose IPOPT-specific options that are currently CLI flags in `testing/refine_perimeter_iterative.py` and parameters on `PerimeterOptimizer.optimize()`: `method` (`'SLSQP'` vs `'ipopt'`), `exact_hessian`, `best_iterate`, `lbfgs_memory`, and `allow_partial_convergence`.
4. Preserve the initial perimeter calculation and `metadata.yaml` logging that was recently added to `find_surface_partition.py` — this must be part of the orchestrator's Phase 1 → Phase 2 handoff.
5. Refactor `examples/find_surface_partition.py` into a thin CLI wrapper that calls `PipelineOrchestrator`. Create `examples/refine_perimeter.py` as a thin CLI wrapper for the refinement stages.
6. Optionally add `examples/run_full_pipeline.py` as a single end-to-end entry point.
7. After extraction, `testing/refine_perimeter_iterative.py` can be reduced to a thin integration test that calls the pipeline orchestrator with a known input and asserts convergence.

### Cleanup pass — run during Phase A, before Phase B

The cleanup should run **during Phase A** (after the Prerequisite step) so
that ring code is already gone before Phases B and C modify the same files
(e.g. `find_surface_partition.py`, `config.py`, `src/__init__.py`). This
avoids having to work around dead ring branches during later phases.

- Remove ring surface files and references (see §Legacy Files — Ring surface).
- Delete `testing/test_vectorized_evaluation.py` (see §Legacy Files — test_vectorized_evaluation).
- Update `README.md` to remove stale references (ring surface, nonexistent `examples/ring_visualization.py`, etc.).
- Fix `pyproject.toml` version mismatch (`0.1.0` → `0.2.0`) and `testpaths` (points to nonexistent `tests/`; should be `testing/`).
- Remove the broken links to `docs/PERIMETER_REFINEMENT.md` from `README.md` (appears on two lines).
- Fix the typo on line 23 of `parameters/input.yaml` (`# Optimization parametersJeez`).
- Update `testing/README_testing.md` to reflect the final directory layout.

### Validation

There is no `pytest` test suite — the `testing/` directory contains
integration scripts, not unit tests.  Use the following checks after each
phase to verify nothing is broken.

**Quick import check (after every subpackage move):**

```bash
# Verify that every Python file under src/ can be imported without error.
# Run from the repo root:
python -c "
import importlib, pathlib, sys
sys.path.insert(0, '.')
failed = []
for p in sorted(pathlib.Path('src').rglob('*.py')):
    if p.name == '__init__.py':
        continue
    mod = str(p.with_suffix('')).replace('/', '.')
    try:
        importlib.import_module(mod)
    except Exception as e:
        failed.append((mod, e))
for m, e in failed:
    print(f'FAIL: {m}: {e}')
if not failed:
    print('All imports OK')
"
```

**Functional validation (after each phase is fully complete):**

```bash
# 1. Phase 1 relaxation (short run — just verify it starts and produces output):
python examples/find_surface_partition.py \
  --input parameters/input.yaml --surface torus

# 2. Perimeter refinement (1 iteration, 10 optimizer steps — verify pipeline runs):
python testing/refine_perimeter_iterative.py \
  --solution <path-to-latest-h5> --method ipopt --lbfgs-memory 6 \
  --max-iterations 1 --max-opt-iter 10 --best-iterate \
  --allow-partial-convergence

# 3. Visualization scripts (verify they load data without crashing):
python examples/visualize_partition_fast.py --solution <path-to-latest-h5>
```

Replace `<path-to-latest-h5>` with a result file from a previous run, e.g.
`results/run_*/iteration*_refined_contours.h5` or a base solution.  There
should be existing result files in the `results/` directory from prior runs.

### Dependencies

`cyipopt` is an optional dependency required for `--method ipopt`. The restructure is a good time to formalize this in `pyproject.toml` as an optional extra (e.g. `pip install .[ipopt]`). Core functionality (SLSQP path, Γ-convergence, migration) must remain usable without `cyipopt` installed. The `IPOPTProblemAdapter` import is already guarded by a `try/except ImportError` in `perimeter_optimizer.py`.

## Dependency Flow Between Subpackages

```
surfaces/ ──→ mesh/
partition/ ──→ mesh/
optimization/ ──→ partition/ ──→ mesh/
migration/ ──→ partition/ ──→ mesh/
pipeline/ ──→ optimization/
          ──→ migration/
          ──→ partition/
          ──→ surfaces/
visualization/ (standalone, no src/ deps)
```

No circular dependencies. Each arrow means "imports from." `pipeline/` sits at the top of the dependency tree and is the only layer that `examples/` scripts need to import directly.

**Note:** `migration/` does **not** import from `optimization/` at the library level. The coupling between migration and optimization happens at the **pipeline** level (in `testing/refine_perimeter_iterative.py`, which calls both), not inside the migration modules themselves.

**Note on `optimization/ → partition/` coupling:** `IPOPTProblemAdapter.jacobian()` and `_hessian_impl()` call directly into `partition/vectorized_area`, `partition/vectorized_perimeter`, and `partition/vectorized_steiner` (not just indirectly through `PerimeterOptimizer`'s existing methods). This is intentional — the adapter needs the sparse-output variants (`compute_area_jacobian_sparse`, `compute_steiner_area_jacobian_sparse`, `compute_perimeter_hessian_sparse`, etc.) that write directly into flat `(nnz,)` / `(hess_nnz,)` value arrays using the `nnz_lookup` and `hess_offset_map` tables. These sparse functions are distinct from the dense-returning public APIs used by the SLSQP path. The dependency is `optimization/ → partition/` (same direction as before), so no new circular risk.
