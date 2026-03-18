# Implementation Prompt: Topology Switch System v2

Use this prompt verbatim when starting a new Cursor agent session in the `feature/topology-switch-v2` worktree.

---

## PROMPT START

You are implementing a new topology switch system for a manifold partitioning library. This is a Python scientific computing project that partitions surfaces (torus, ring) into equal-area cells by minimizing geodesic perimeter using finite elements on triangulated meshes.

### Your task

Implement the plan described in `.cursor/plans/topology_switch_implementation_bf20b2fb.plan.md`. This is your primary specification. Read it fully before writing any code.

The algorithmic details for the topology switches are in `docs/TOPOLOGY_SWITCH_METHODOLOGY.md`. This is your algorithmic reference — it contains concrete examples with full 1-ring configurations, VP fate tables, segment accounting, and step-by-step execution for both Type 1 and Type 2 migrations. Read it fully before writing any code.

### Context: what this codebase does

This library implements the numerical method from a research paper on partitioning manifolds into regions of minimal perimeter. The pipeline has two stages:

1. **Gamma-convergence relaxation** (already implemented, working): Uses phase-field approximation + L-BFGS/PGD to find an approximate partition. This produces density functions on the mesh.

2. **Perimeter refinement** (partially implemented): Extracts sharp contours from the density functions, places Variable Points (VPs) on mesh edges, and optimizes their positions to minimize total perimeter subject to area constraints. This stage includes topology switches when VPs approach mesh vertices — that is what you are implementing.

The first stage is upstream and **must not be broken**. All data structure changes include backward-compatible properties to protect it.

### Critical design decisions (do NOT deviate from these)

1. **VP index stability**: Destroyed VPs are marked `active = False` but NEVER removed from the `variable_points` list. Their list index is their permanent identity. This is critical because `TriplePointHistory` snapshots store VP indices across iterations. If you compact/reindex, all stored snapshots become invalid.

2. **Type 2 migration creates +1 VP, +1 segment**: The Steiner point S converts into a new VP on the collapsed edge. VP A is NOT "kept" — it is destroyed by Type 1 mechanics and recreated on a different edge. The Steiner-converted VP fills the slot left on the collapsed edge. This is the single most important accounting fact in the entire system. See methodology doc Section 2.10 for the complete VP fate table.

3. **Type 2 reversal uses snapshot-based rollback, NOT forward migration**: Applying forward migration mechanics in reverse would accumulate +1 VP per round trip. The `LocalStateSnapshot` captures the full local state before migration; reversal restores it exactly, deactivating the Steiner-converted VP. See methodology doc Part 3.

4. **The collapsed edge gets NO VP from the OneRingRebuilder**: During a Type 2 migration, the rebuilder skips VP creation on the collapsed edge (via `collapsed_edge` parameter). The `Type2Executor` creates the Steiner-converted VP there in a separate step (step 3b in the plan).

5. **Three-VP validation for Type 1**: A vertex flip requires at least 3 VPs on edges emanating from that vertex to be approaching it. If fewer than 3, the trigger is rejected.

6. **Pure Type 1 must not create triple-point triangles**: If a Type 1 flip (not part of a Type 2) results in a triangle with 3 different cell labels, it is an error. Only Type 2 (which flips to the third cell) is expected to create new triple-point triangles.

### Execution order

Follow this strict order. Do NOT skip ahead.

**Phase 0 — Data structure refactoring (do these first, all must pass before Phase 1):**
1. `refactor-partition-contour` — This is the largest and most important refactoring. Read `src/core/contour_partition.py` (1119 lines) thoroughly first. Key changes: `_vertex_labels` replaces `indicator_functions`, `_triangle_segments` dict replaces list, `_vp_adjacency` replaces `boundary_segments`, add `VariablePoint.active` flag, add `rebuild_active_vp_indices()` / `get_variable_vector()` / `set_variable_vector()`. Add backward-compat `@property` for each replaced field.
2. `refactor-steiner-handler` — Read `src/core/steiner_handler.py` (657 lines). Remove `self.partition` back-reference from `TriplePoint`. Methods receive VP positions as arguments. Add `compute_void_angles()`.
3. `refactor-perimeter-calculator` — Read `src/core/perimeter_calculator.py` (274 lines). Iterate `triangle_segments` dict instead of `boundary_segments`.
4. `refactor-area-calculator` — Read `src/core/area_calculator.py` (513 lines). Replace `np.argmax(indicator_functions)` with `partition.vertex_labels`.
5. `update-mesh-topology` — Read `src/core/mesh_topology.py` (193 lines). Add `get_vertex_neighbors()` and `are_neighbors()` with caching.

After each Phase 0 change, verify that the existing tests and scripts still work. The backward-compat properties must be transparent.

**Phase 1 — New modules:**
6. `create-migration-types` — Create `src/core/migration_types.py`. Pure dataclasses, no logic. This must exist before anything else in Phase 1.
7. `create-migration-detector` — Create `src/core/migration_detector.py`. Three internal sections: Type 1 detection, Type 2 detection, conflict resolution.
8. `create-one-ring-rebuilder` — Create `src/core/one_ring_rebuilder.py`. The core algorithm. Follow the methodology doc Section 1.7-1.8 for Type 1 and Section 2.6 for Type 2.

**Phase 2 — Executor:**
9. `create-migration-executor` — Create `src/core/migration_executor.py`. Three internal sections: snapshot capture/restore, Type 1 execution, Type 2 execution (forward + rollback).

**Phase 3 — Orchestration and integration:**
10. `create-migration-orchestrator` — Create `src/core/migration_orchestrator.py`. The `MigrationOrchestrator` class with the public API described in the plan.
11. `update-test-scripts` — Update `testing/refine_perimeter_iterative.py`, `testing/test_migration_and_continue.py`, `testing/test_migrations_debug.py`.
12. `update-visualization-scripts` — Update `examples/visualize_type1_vertex_collapse.py`, `examples/visualize_type2_triple_point.py`, `examples/visualize_partition.py`.
13. `integration-testing` — Run the test scripts on existing data and verify correctness.

### Coding standards

- Python 3.10+. Use type hints on all function signatures.
- Use `@dataclass` for data structures. Use `Optional`, `Dict`, `List`, `Tuple`, `Set` from `typing`.
- Use `numpy` for numerical arrays. No pandas.
- Logging: use `from src.logging_config import get_logger` and `logger = get_logger(__name__)`. Log at INFO level for migration events, DEBUG for VP-level details.
- No comments that narrate what code does. Comments only for non-obvious intent or algorithmic notes.
- Each new file should have a module docstring explaining its role in 2-3 sentences.
- Functions should be short (< 50 lines preferred). Extract helpers with clear names.
- Preserve the old `topology_switcher.py` as `topology_switcher_legacy.py` — do NOT delete it.

### Files you must read before starting

Read these files in full before writing any code:

1. `.cursor/plans/topology_switch_implementation_bf20b2fb.plan.md` — The complete implementation plan with data structures, module specs, VP accounting, and rollback procedures.
2. `docs/TOPOLOGY_SWITCH_METHODOLOGY.md` — Algorithmic specification with concrete examples, VP fate tables, segment accounting, and the reversal undo-stack.
3. `src/core/contour_partition.py` — Current data structures (`VariablePoint`, `TriangleSegment`, `PartitionContour`). You are refactoring this.
4. `src/core/steiner_handler.py` — Current `TriplePoint` and `SteinerHandler`. You are refactoring this.
5. `src/core/mesh_topology.py` — Current mesh connectivity. You are extending this.
6. `src/core/topology_switcher.py` — The OLD monolithic implementation (3824 lines). Read it to understand the existing approach, but do NOT base your implementation on it. The new system replaces it entirely. Rename it to `topology_switcher_legacy.py`.
7. `src/core/migration_utils.py` — Existing helpers (`identify_target_vertex`, `compute_boundary_distance`). Reuse where applicable.
8. `testing/refine_perimeter_iterative.py` — Primary test script. Understand its current workflow before modifying.
9. `testing/test_migrations_debug.py` — Debug test script. Understand its `--watch-vps` mechanism.

### How to verify your work

After Phase 0:
- Run `python testing/refine_perimeter_iterative.py` on an existing solution file and verify it produces the same output as before the refactoring. The backward-compat properties must be transparent.

After Phase 1-2:
- Write a small test that creates a known 1-ring configuration (matching the Type 1 example in the methodology doc: v1 center, w2, w3, w4, v5, v6, v7), runs `rebuild_one_ring()`, and verifies the VP create/destroy/keep counts match the methodology doc tables exactly.
- Write a small test for `capture_snapshot` + `restore_snapshot` round-trip.

After Phase 3:
- Run `python testing/test_migrations_debug.py --solution-file <path>` and verify migration detection and execution.
- Run `python examples/visualize_type1_vertex_collapse.py` and verify the visualization renders correctly with the new APIs.

### What NOT to do

- Do NOT modify the Gamma-convergence optimizers (`pyslsqp_optimizer.py`, `pgd_optimizer.py`). They are upstream and working.
- Do NOT modify `src/core/tri_mesh.py`. The mesh container is stable.
- Do NOT compact or reindex the `variable_points` list. Ever.
- Do NOT implement Type 2 reversal using forward migration mechanics. Use snapshot-based rollback only.
- Do NOT create more than 5 new files for the migration system. The architecture is: `migration_types.py`, `migration_detector.py`, `one_ring_rebuilder.py`, `migration_executor.py`, `migration_orchestrator.py`.
- Do NOT delete `topology_switcher.py` — rename it to `topology_switcher_legacy.py`.
- Do NOT modify files outside the plan without explicit reason.
- Do NOT create documentation files unless explicitly instructed.

### Summary of new files to create

| File | ~Lines | Contents |
|------|--------|----------|
| `src/core/migration_types.py` | 150 | Dataclasses: Type1Trigger, Type2Trigger, LocalStateSnapshot, SteinerSnapshot, TriplePointHistory, RebuildResult |
| `src/core/migration_detector.py` | 500 | Type 1 detection + Type 2 detection + conflict resolution |
| `src/core/one_ring_rebuilder.py` | 400 | 1-ring rebuild after vertex flip (VP create/destroy/keep) |
| `src/core/migration_executor.py` | 600 | Snapshot capture/restore + Type 1 exec + Type 2 forward/rollback |
| `src/core/migration_orchestrator.py` | 300 | MigrationOrchestrator class with optimize-detect-switch loop |

### Begin

Start by reading the plan file and methodology document in full. Then proceed with Phase 0, step 1 (`refactor-partition-contour`). Work methodically, one todo at a time. After completing each todo, verify nothing is broken before moving to the next.

## PROMPT END
