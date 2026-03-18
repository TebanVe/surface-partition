# Codebase Restructure Plan

This document describes a proposed restructuring of the `src/` directory to improve organization, readability, and maintainability. This restructure is **not required** for the topology switch implementation — it can be done independently, after the migration system is working.

---

## Problem

The current `src/` layout has two issues:

1. **`core/` is overloaded** — 17 files mixing four unrelated domains: mesh infrastructure, partition data structures, optimization algorithms, and topology migration. A developer looking for "the optimizer" must scan past mesh connectivity, contour partitions, and migration history files.

2. **`src/` root is a grab-bag** — config, logging, plotting, contour analysis, projection, and island analysis sit at the top level with no grouping. These files have very different audiences (plotting is for visualization scripts; projection is for the optimization pipeline).

## Current Structure (25 files)

```
src/
├── __init__.py              (package exports)
├── config.py                (131 lines — configuration)
├── logging_config.py        (284 lines — logging setup)
├── find_contours.py         (296 lines — contour extraction)
├── island_analysis.py       (87 lines — density field analysis)
├── projection_iterative.py  (363 lines — projection onto constraints)
├── plot_utils.py            (411 lines — 2D plotting)
├── plot_utils_3d.py         (160 lines — 3D PyVista plotting)
├── core/                    (NO __init__.py)
│   ├── tri_mesh.py          (311 lines — mesh container + FEM)
│   ├── mesh_topology.py     (193 lines — mesh connectivity)
│   ├── interpolation.py     (23 lines — nearest-neighbor interp)
│   ├── contour_partition.py (1119 lines — VP, segments, partition)
│   ├── area_calculator.py   (513 lines — area computation)
│   ├── perimeter_calculator.py (274 lines — perimeter computation)
│   ├── perimeter_optimizer.py (780 lines — SLSQP perimeter opt)
│   ├── steiner_handler.py   (657 lines — triple point Steiner trees)
│   ├── pyslsqp_optimizer.py (306 lines — PySLSQP wrapper)
│   ├── pgd_optimizer.py     (394 lines — projected gradient descent)
│   ├── migration_utils.py   (94 lines — migration helpers)
│   ├── topology_switcher.py (3824 lines — monolithic migration)
│   ├── type1_component_analyzer.py (1434 lines — Type 1 analysis)
│   ├── type2_migration_history.py (203 lines — migration history)
│   └── type2_migration_io.py (198 lines — HDF5 I/O for history)
└── surfaces/                (NO __init__.py)
    ├── torus.py             (106 lines — torus mesh provider)
    └── ring.py              (89 lines — ring mesh provider)
```

## Proposed Structure

Group files by domain. Each subpackage has a clear purpose, an `__init__.py` with public exports, and 2-5 files.

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
├── partition/               (contour partition and calculators)
│   ├── __init__.py
│   ├── contour_partition.py (VariablePoint, TriangleSegment, PartitionContour)
│   ├── area_calculator.py   (AreaCalculator)
│   ├── perimeter_calculator.py (PerimeterCalculator)
│   └── steiner_handler.py   (TriplePoint, SteinerHandler)
│
├── optimization/            (all optimizers)
│   ├── __init__.py
│   ├── perimeter_optimizer.py (constrained perimeter minimization)
│   ├── pgd_optimizer.py     (projected gradient descent)
│   ├── pyslsqp_optimizer.py (PySLSQP wrapper)
│   └── projection.py        (orthogonal projection onto constraints)
│
├── migration/               (topology switch system — NEW)
│   ├── __init__.py
│   ├── migration_types.py   (dataclasses)
│   ├── migration_detector.py (detection + conflict resolution)
│   ├── one_ring_rebuilder.py (1-ring rebuild)
│   ├── migration_executor.py (snapshot + T1/T2 execution)
│   └── migration_orchestrator.py (top-level loop)
│
├── surfaces/                (surface providers)
│   ├── __init__.py
│   ├── torus.py
│   └── ring.py
│
├── analysis/                (post-processing analysis)
│   ├── __init__.py
│   ├── find_contours.py     (contour extraction from density fields)
│   └── island_analysis.py   (density field metrics)
│
└── visualization/           (all plotting)
    ├── __init__.py
    ├── plot_utils.py        (2D matplotlib plots)
    └── plot_utils_3d.py     (3D PyVista plots)
```

## Rationale for Each Subpackage

### `mesh/` — Mesh Infrastructure
Contains the foundational mesh data structure (`TriMesh`), its topology queries (`MeshTopology`), and interpolation between meshes. These are surface-agnostic and have no dependency on partition or optimization concepts.

### `partition/` — Partition Data Structures
Contains everything that defines a partition on a mesh: the contour representation (`PartitionContour`, `VariablePoint`, `TriangleSegment`), the area and perimeter calculators that measure properties of the partition, and the Steiner tree handler for triple points. These all share the same core abstraction (a partition contour on a mesh) and have tight interdependencies.

### `optimization/` — Optimizers
Contains all optimization algorithms: the two Gamma-convergence optimizers (`pgd_optimizer`, `pyslsqp_optimizer`), the perimeter optimizer (Section 5 of the paper), and the projection onto constraints. These share a common pattern (minimize a functional subject to constraints) but operate at different stages of the pipeline.

### `migration/` — Topology Switches
The new modular migration system. Fully self-contained with clear internal layering (types → detection → rebuild → execution → orchestration).

### `surfaces/` — Surface Providers
Already a separate subpackage. Just needs an `__init__.py`.

### `analysis/` — Post-Processing
Contour extraction and density field analysis. Used by visualization scripts but not by the optimization pipeline itself.

### `visualization/` — Plotting
All matplotlib and PyVista plotting utilities. Used only by example/testing scripts, never by the computational core.

## Import Path Changes

The restructure changes import paths. Examples:

| Before | After |
|--------|-------|
| `from src.core.tri_mesh import TriMesh` | `from src.mesh.tri_mesh import TriMesh` |
| `from src.core.contour_partition import PartitionContour` | `from src.partition.contour_partition import PartitionContour` |
| `from src.core.perimeter_optimizer import PerimeterOptimizer` | `from src.optimization.perimeter_optimizer import PerimeterOptimizer` |
| `from src.plot_utils import plot_ring_mesh` | `from src.visualization.plot_utils import plot_ring_mesh` |
| `from src.find_contours import ContourAnalyzer` | `from src.analysis.find_contours import ContourAnalyzer` |
| `from src.projection_iterative import ...` | `from src.optimization.projection import ...` |

To minimize disruption, `__init__.py` files can re-export symbols at the old paths during transition.

## Execution Strategy

1. **Do this AFTER the topology switch implementation is complete and tested.** The migration system is the priority; restructuring is a quality-of-life improvement.
2. Add `__init__.py` files to all subpackages first (currently missing in `core/` and `surfaces/`).
3. Move files one subpackage at a time, updating imports in all consumers.
4. Use `git grep` to find all import statements that need updating.
5. Run all tests after each subpackage move to catch broken imports immediately.
6. Update `src/__init__.py` to re-export from new locations.

## Files Affected by the Restructure

- **Moved files**: All 25 files in `src/` are reorganized (though `config.py` and `logging_config.py` stay in place).
- **Import updates needed in**: all files under `testing/`, `examples/`, and cross-references within `src/` itself.
- **New files**: 7 `__init__.py` files (one per subpackage).
- **Renamed**: `projection_iterative.py` → `optimization/projection.py` (cleaner name).

## Dependency Flow Between Subpackages

```
surfaces/ ──→ mesh/
partition/ ──→ mesh/
optimization/ ──→ partition/ ──→ mesh/
migration/ ──→ partition/ ──→ mesh/
           ──→ optimization/
analysis/ ──→ mesh/ (standalone)
visualization/ (standalone, no src/ deps)
```

No circular dependencies. Each arrow means "imports from."
