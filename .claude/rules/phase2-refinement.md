---
paths:
  - "src/optimization/perimeter_optimizer.py"
  - "src/partition/perimeter_calculator.py"
  - "src/partition/area_calculator.py"
  - "src/partition/steiner_handler.py"
  - "src/partition/contour_partition.py"
  - "src/partition/partition_arrays.py"
  - "src/partition/vectorized_*.py"
  - "src/partition/find_contours.py"
  - "src/migration/**"
  - "src/pipeline/pipeline_orchestrator.py"
  - "scripts/refine_perimeter.py"
  - "testing/test_*hessian*.py"
  - "testing/test_steiner_*.py"
  - "testing/test_migrations_debug.py"
  - "testing/test_type1_*.py"
---

# Phase 2 — perimeter refinement

`PerimeterOptimizer` minimizes total perimeter (regular + Steiner) subject to
equal cell areas. ⚠ `PerimeterCalculator.compute_total_perimeter` returns the
**sum of all cell perimeters** `sum_k Per(cell_k)`, so every interface is counted
**twice**; `label_boundary_length` counts each once. Measured ratio 1.999 on both
implicit surfaces before Phase 2 does anything. It delegates to
`PerimeterCalculator`, `AreaCalculator` and `SteinerHandler` (or their
vectorized counterparts). To change the objective or
constraints, modify those calculators. `PartitionArrays` pre-computes the sparse
Jacobian/Hessian patterns for IPOPT.

The loop (`PipelineOrchestrator.run_refinement_loop()`): optimize -> detect
migrations -> export checkpoint -> migrate, until no migrations are needed or
the iteration cap is hit.

## The lambda convention

Variable points sit on mesh edges: `x = lambda*vertices[e0] + (1-lambda)*vertices[e1]`,
with edges normalized so `e0 < e1`. So **lambda = 1 is at the SMALLER vertex
index** and lambda = 0 at the larger. When lambda approaches 0 or 1 within
`boundary_tol`, a Type 1 migration triggers.

## Migration types

- **Type 1 (vertex collapse).** A VP's lambda is near 0 or 1. Detection requires
  **>= 3 incident boundary VPs all approaching the same vertex**, with a
  triple-point safety guard rejecting candidates whose 1-ring intersects an
  existing Steiner triangle. The vertex is flipped and its 1-ring rebuilt
  edge-by-edge by `one_ring_rebuilder.py` (valence-agnostic).
- **Type 2 (triple point).** Changes to which cells meet at a Steiner point —
  either a forward migration or a rollback to a prior snapshot. History in
  `type2_migration_history.py`.

## VariablePoint soft deletion

Destroyed VPs are marked `active=False` but **never removed from the list**.
This preserves index stability for snapshot rollback, so **always filter on
`vp.active`**.

## Consistency checks

`PipelineOrchestrator.export_checkpoint()` runs a roundtrip perimeter
verification after saving. If it warns, the indicator functions may be out of
sync with the live VP state.

## The migration-cycling plateau at high N — a plateau, not a bug

Observed from N=100 upward. After the large first-iteration perimeter drop,
per-iteration gains decay to noise (~0.003%) and the topology **oscillates**:
migrations periodically raise the perimeter by a hair and the next optimize step
claws it back, so `pending_migration` never clears and `optimization_success`
stays `False`. It runs to the iteration cap without converging.

The exported geometry at the best iterate is complete and valid; it just was not
topologically frozen. **Standard workflow:** scan `final_perimeter` across every
`iteration_*.h5` in the campaign, pick the minimum, and export it with
`--force-finalised` (see `run-layouts.md`).

## The IPOPT solver is NOT the thing to optimise

`timing_profile.yaml`'s `summary.total_wall_s` counts only time inside
`optimizer.optimize()`, so it reads **459 s against an 8,829 s campaign at
N=500** — a 19x under-report. Taken correctly, **the solver is 5.2% of Phase 2
at N=500 and 3.6% at N=750** — a *shrinking* share — running 201 solver
iterations per topology iteration at both, i.e. hitting `max_opt_iter: 200`
every time.

Phase 2 also scales superlinearly in variable points (0.250 / 0.270 / 0.389 s
per VP at N=400/500/750; exponents 1.69 then 2.00), which makes it the binding
constraint on N, and it is now ~85-89% of end-to-end cost. So optimising the
solver buys <= 5% and the exact-Hessian path is the wrong target — the ~95%
outside `optimize()` (contour rebuild, Steiner setup, migration detection,
checkpoint export with roundtrip verification) is **unmeasured**, and
decomposing it needs instrumentation that does not exist yet. See
`docs/plans/PUBLICATION_READINESS_PLAN.md` Phase 4.
