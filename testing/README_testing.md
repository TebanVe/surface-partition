# Test Registry and Tracking

**Purpose**: This document tracks all tests implemented for the surface partition project, providing a centralized registry for test status, objectives, results, and maintenance.

**Last Updated**: 2026-05-05 (May 5, 2026) — Test 5 retired after validation

---

## Test Status Definitions

- **✅ APPROVED**: Test passed validation, confirmed working as expected
- **🔄 UNDER INVESTIGATION**: Test implemented but results under review
- **❌ FAILED**: Test revealed issues, requires fixes
- **📝 PLANNED**: Test designed but not yet implemented
- **🚧 IN PROGRESS**: Test currently being developed

---

## Test Execution Guidelines

### Running Tests

All tests should be run from the project root:
```bash
cd /path/to/RingTest
python testing/test_name.py [arguments]
```


### Logging

- Tests should produce clear, structured output
- Use section separators (=== or ---) for readability
- Include timestamps for long-running tests
- Save detailed logs to `testing/logs/` directory (if needed)

### Validation Criteria

A test is considered **APPROVED** when:
1. All assertions pass
2. Results match expected behavior
3. No unexpected warnings or errors
4. Performance is acceptable
5. Reviewed and confirmed by team

---

## Adding New Tests

When adding a new test:

1. **Create test file**: `testing/test_<descriptive_name>.py`
2. **Update this registry**: Add entry to Test Inventory section
3. **Include**:
   - Clear test objective
   - What is being validated
   - Usage instructions
   - Expected output
   - Dependencies
4. **Set initial status**: 🚧 IN PROGRESS or 📝 PLANNED
5. **Update status** as test progresses through development and validation
6. **Add timestamp** when test is created and when approved

---

## Maintenance Notes

**Review Frequency**: Tests should be re-run after:
- Significant changes to key parts of the codebase
- Architectural changes or refactoring
- Bug fixes that affect core functionality
- Before merging changes to main branch

**Regression Testing**: All approved tests should pass before merging changes to main branch.

**Documentation**: Keep this registry up to date with test status changes and new findings.

---

## Test Inventory

### Test 0: Lambda-Edge Roundtrip Consistency

**File**: ~~`test_lambda_edge_roundtrip.py`~~ _(deleted 2026-03-24)_

**Status**: ✅ APPROVED → 🗑️ DELETED

**Created**: 2026-03-05

**Approved and deleted**: 2026-03-24

**Objective**:
Verified that lambda parameters saved to an `.h5` file are correctly reloaded and
matched to the same edges they were associated with in memory — specifically after
topology migrations where the in-memory VP list order diverges from the sorted-edge
order used during reconstruction from `indicator_functions`.

**What it validated**:
1. Whether the set of edges in the saved file matched the reconstructed set
2. Whether the lambda-to-edge mapping was preserved through save/load
3. The perimeter impact of any lambda misassignment

**Key finding**:
Post-migration VP list order diverges from sorted-edge reconstruction order. Without
edge-keyed loading, lambdas get applied to the wrong VPs. Fix: save `vp_edges`
alongside `lambda_parameters` in the HDF5 file; load by edge key in
`scripts/data_loader.py::load_partition_from_refined_file`. **Confirmed working.**

**Why deleted**:
Diagnostic script requiring an external `--solution` file; not runnable as an
automated regression test. Historical findings are preserved in this entry and in
`scripts/debug_archive/`. The underlying fix lives in `data_loader.py`.

---

### Test 1: Self-Healing Component Selection

**File**: `test_self_healing_selection.py`

**Status**: ✅ APPROVED ⚠️ BROKEN IMPORT (needs update)

**Created**: 2026-01-26

**Approved**: 2026-01-26

**Objective**: 
Validate that the component selection architecture exhibits self-healing behavior without requiring cross-iteration tracking. Specifically, test that components excluded due to conflicts in one iteration are correctly re-evaluated in the next iteration after migrations have occurred.

**What it validates**:
- Pre-filter correctly identifies components that can form valid auxiliary components
- Components excluded due to conflicts are not tracked across iterations
- After migration, previously excluded components are re-evaluated from scratch
- If neighbors moved during migration, pre-filter correctly excludes components with broken auxiliaries
- System automatically handles component validity without manual tracking

**Test scenario**:
1. **Iteration 1**: 
   - Components 44 and 45 selected for migration (closer to target vertices)
   - Components 49 and 50 excluded due to conflicts with 44 and 45
   - Migrate components 44 and 45
2. **Iteration 2**:
   - Re-analyze all components (no tracking from iteration 1)
   - Components 49 and 50 cannot form valid auxiliary (neighbors moved)
   - Pre-filter excludes 49 and 50 (no tracking needed)

**Key findings**:
- Component 49 shared VP 1371 with Component 44's auxiliary
- When Component 44 migrated, VP 1371 moved to a different edge
- VP 1371 now approaches vertex 32275 (Component 44's target) instead of 32276 (Component 49's target)
- Iteration 2 pre-filter automatically detected broken auxiliary for Component 49
- Self-healing confirmed: No cross-iteration tracking required

**Usage**:
```bash
python testing/test_self_healing_selection.py \
    --solution output/ring_lambda0.5_refined_contours.h5
```
Solution file used for the test:
results/run_20251027_233612_surftorus_npart5_v1nt60-240_incr20_v2np48-192_incr16_lam3.0_seed163498/surface_part5_surftorus_v1nt60-240_incr20_v2np48-192_incr16_lam3.0_seed163498_20251027_233612_refined_contours.h5

⚠️ **Note**: Update the `--solution` path to point to the specific refined contours file you want to analyze. The test results depend on the solution file being used.

**Expected output**:
```
✓ Test 1 PASSED: Components 44 and 45 were migrated in iteration 1
✓ Test 2 PASSED: Components 49 and 50 were excluded in iteration 1 (conflict)
✓ Test 3 PASSED: Algorithm ran without cross-iteration tracking
✓ Test 4 PASSED: Components 49/50 re-evaluated in iteration 2 without tracking
✓✓✓ ALL TESTS PASSED - SELF-HEALING VALIDATED ✓✓✓
```

**Dependencies**:
- `src/core/topology_switcher.py` (select_components_for_migration with pre-filter)
- `src/core/tri_mesh.py`
- `src/core/contour_partition.py`
- `src/core/mesh_topology.py`
- ~~`scripts/visualize_precise_region.py`~~ → replaced by `scripts/data_loader.py::load_partition_from_refined_file`

**Notes**:
- Test uses deep copy of partition to compare states across iterations
- Tracks VP movements to explain why auxiliaries break
- Provides detailed logging for debugging

⚠️ **Known issue (2026-03-24)**: Script imports `from examples.visualize_precise_region import load_partition_from_refined_file` which no longer exists. The function now lives in `scripts/data_loader.py`. The import line must be updated before this test can be run.

---

### Test 3: Iterative Perimeter Refinement with Automatic Migrations

**File**: `refine_perimeter_iterative.py`

**Status**: ✅ APPROVED (Production-validated 2026-03-23)

**Created**: 2026-02-08

**Last Updated**: 2026-03-23 (Promoted to APPROVED after successful production run)

**Approved**: 2026-03-23

**Objective**: 
Implement a complete production-ready iterative refinement workflow that automatically applies topology migrations and continues optimization until convergence (no switches needed) or maximum iterations reached. This script serves as the main workflow orchestrator for perimeter refinement in production use.

**What it validates**:
- Loading initial solution from PGD/SLSQP relaxation optimization
- Iterative loop: optimize → detect switches → apply migrations (Type 2 first, then Type 1) → repeat
- Proper calculator reuse within iterations (efficiency pattern from Test 2)
- Exporting intermediate states at each iteration (before migrations)
- Exporting final converged state
- Exit-on-first-failure migration handling with detailed diagnostics
- Indicator function persistence across save/reload cycles
- Complete convergence tracking across multiple iterations

**Key features**:
- Production-ready iterative workflow: Loop(optimize → detect → migrate)
- Three-way entry point: base solution, checkpoint with pending migration, converged checkpoint
- Supports IPOPT with exact Hessian, best-iterate tracking, and partial convergence
- Checkpoint semantics with `pending_migration` flag for unambiguous resume

**Usage**:
```bash
# Basic usage
python testing/refine_perimeter_iterative.py \
    --solution results/run_xyz/solution_level0.h5 \
    --max-iterations 10

# Custom settings with all options
python testing/refine_perimeter_iterative.py \
    --solution results/run_xyz/solution_level0.h5 \
    --max-iterations 20 \
    --tolerance 1e-8 \
    --boundary-tol 0.005 \
    --distance-preservation midpoint \
    --method trust-constr \
    --log-level DEBUG
```

**Command-line arguments**:
- `--solution`: Path to initial solution HDF5 file from PGD/SLSQP (required)
- `--max-iterations`: Maximum topology iteration cycles (required)
- `--output`: Output path for final state (default: auto-generated)
- `--max-opt-iter`: Max optimization iterations per topology (default: 1000)
- `--tolerance`: Optimization convergence tolerance (default: 1e-7)
- `--boundary-tol`: Boundary point detection threshold (default: 1e-3)
- `--distance-preservation`: VP placement strategy for Type 1: `preserve` (default), `midpoint`, or float 0.0-1.0
- `--method`: Optimization method: `SLSQP` (default) or `trust-constr`
- `--log-level`: Logging verbosity: DEBUG, INFO (default), WARNING, ERROR

**Expected output files**:
For a run with 3 iterations before convergence:
```
results/run_xyz/
├── solution_level0.h5                              # Input (initial relaxation)
├── solution_level0_iteration1_refined_contours.h5  # After iteration 1 opt, before switches
├── solution_level0_iteration2_refined_contours.h5  # After iteration 2 opt, before switches
├── solution_level0_iteration3_refined_contours.h5  # After iteration 3 opt, before switches
└── solution_level0_refined_contours.h5             # Final converged state
```

**Each iteration file contains**:
- Lambda parameters (VP positions)
- **Indicator functions** (updated after migrations - critical for visualization)
- Optimization metadata (perimeter, areas, violations)
- Iteration number and timestamp

**Success criteria**:
```
✓ Initial solution loaded
✓ Each iteration:
  - Optimization converged
  - State exported before migrations
  - Switches detected
  - Migrations applied successfully
  - Calculators reinitialized
✓ Convergence achieved (no switches needed) OR max iterations reached
✓ Final state exported
✓ Summary shows total migrations and perimeter reduction
```

**Failure handling**:
Script exits immediately with detailed diagnostics if:
- **Optimization fails**: Exports current state, logs error message
- **Migration fails** (first failure): 
  - Logs detailed component/triple-point information (VPs, triangles, cells, target)
  - Provides exact visualization command to investigate
  - Exports state before failed migration
  - Example output:
    ```
    ================================================================================
    MIGRATION FAILURE - TYPE 1
    ================================================================================
    Component 44 migration FAILED
    
    Component details:
      VPs involved: [1260, 1261]
      Target vertex: 32275
      Cells involved: [1, 4]
      Distance to target: 0.007175
    
    Error: Edge validation failed
    
    ================================================================================
    DIAGNOSTIC RECOMMENDATION
    ================================================================================
    Visualize this component to investigate:
    
      python scripts/visualize_type1_vertex_collapse.py \
        --solution results/.../iteration2_refined_contours.h5 \
        --component-index 44 \
        --boundary-tol 0.01 \
        --state before \
        --show-vps \
        --show-steiner
    
    State before failure saved to: iteration2_refined_contours.h5
    ================================================================================
    ```
- **Zero migrations applied**: Logs warning, exports current state, exits

**Expected workflow**:
```
Iteration 1:
  Optimize → Export state → Detect (182 Type 1, 8 Type 2) → Migrate (85 Type 1, 7 Type 2) → Reinitialize
  
Iteration 2:
  Optimize → Export state → Detect (155 Type 1, 7 Type 2) → Migrate (72 Type 1, 6 Type 2) → Reinitialize
  
Iteration 3:
  Optimize → Export state → Detect (98 Type 1, 5 Type 2) → Migrate (45 Type 1, 4 Type 2) → Reinitialize
  
... continues until ...

Iteration N:
  Optimize → Export state → Detect (0 switches) → CONVERGED!
  Export final state
```

**Dependencies**:
- `src/find_contours.py` (ContourAnalyzer)
- `src/core/tri_mesh.py`
- `src/core/contour_partition.py`
- `src/core/perimeter_optimizer.py`
- `src/core/mesh_topology.py`
- `src/core/topology_switcher.py` (v2/v4 methods)
- `src/core/type1_component_analyzer.py`
- `src/logging_config.py`
- `h5py`, `numpy`

**Visualization integration**:
All intermediate and final files can be visualized:
```bash
# View any iteration state
python scripts/visualize_partition.py \
    --solution results/run_xyz/*_iteration2_refined_contours.h5 \
    --region 2 --show-steiner

# View final converged state
python scripts/visualize_partition.py \
    --solution results/run_xyz/*_refined_contours.h5 \
    --region 2 --show-steiner

# Debug specific component
python scripts/visualize_type1_vertex_collapse.py \
    --solution results/run_xyz/*_iteration3_refined_contours.h5 \
    --component-index 44 --state before
```

**Notes**:
- **Migration order**: Type 2 migrations always applied before Type 1 (proven optimal order)
- **Calculator efficiency**: Reuses optimizer instance within each iteration, reinitializes after migrations
- **Automatic progress logging**: Optimization automatically logs every 10 iterations
- **File overwrites**: Always overwrites existing iteration files without warning
- **Convergence criteria**: No topology switches detected (all VPs and triple points away from boundaries)
- **Typical convergence**: 5-15 iterations for production meshes
- **Runtime**: Few hours to overnight depending on mesh size and max-opt-iter settings

**Design documentation**:
See `docs/REFINE_PERIMETER_ITERATIVE_PLAN.md` for complete implementation plan including:
- Detailed workflow diagrams
- Code examples for all components
- Safety analysis and validation strategy
- Testing checklist

**Relation to other tests**:
- Uses component analysis architecture validated in Test 1 (`test_self_healing_selection.py`)
- Serves as the production workflow for iterative perimeter refinement

---

### Test 4: Migration Debug (Type 2 then Type 1, no optimization, no export)

**File**: `test_migrations_debug.py`

**Status**: 🚧 IN PROGRESS

**Created**: 2026-03-03

**Objective**:
Isolate and debug the migrate→migrate workflow (Type 2 first, then Type 1) without
any optimization iterations or file exports. Designed to diagnose cases where VP
lambda values are inadvertently altered by Type 2 migrations, causing incorrect
component grouping in the subsequent Type 1 analysis.

**What it validates**:
- Lambda and edge values of watched VPs before and after *each* Type 2 migration
- Whether `identify_target_vertex` classification flips for any watched VP after a Type 2 migration
- Which components the watched VPs end up in after the post-Type2 Type 1 analysis
- Whether watched VPs appear in each other's auxiliary components (spurious grouping)

**Key differences from `refine_perimeter_iterative.py`**:
- **No optimization**: skips the optimize step entirely
- **No export**: no HDF5 output written
- **Per-migration VP tracking**: logs edge/λ/target state of watched VPs before and after *every* individual migration
- **Diff logging**: clearly marks TARGET FLIP when `identify_target_vertex` changes for a watched VP

**Usage**:
```bash
# Investigate VPs 1621, 1624, 1625 across Type 2 then Type 1 migrations
python testing/test_migrations_debug.py \
    --solution results/run_xyz/*_iterationN_refined_contours.h5 \
    --watch-vps 1621 1624 1625 \
    --boundary-tol 0.001

# Only run Type 2 migrations
python testing/test_migrations_debug.py \
    --solution results/run_xyz/*_iterationN_refined_contours.h5 \
    --migration-type type2 \
    --watch-vps 1621 1624 1625
```

**Command-line arguments**:
- `--solution`: Path to refined contours HDF5 file (required)
- `--migration-type`: `type1`, `type2`, or `both` (default: `both`)
- `--distance-preservation`: VP placement strategy (default: `preserve`)
- `--boundary-tol`: Boundary detection threshold (default: `1e-3`)
- `--watch-vps`: Space-separated VP indices to track (e.g. `--watch-vps 1621 1624 1625`)
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

**Expected output sections**:
1. `INITIAL STATE OF WATCHED VPs` — edge/λ/target before any migration
2. Per-Type2-migration diff — shows changes (with `*** TARGET FLIP ***` marker)
3. `WATCHED VP STATE AFTER ALL TYPE 2 MIGRATIONS` — consolidated post-Type2 view
4. `TYPE 1 COMPONENT ANALYSIS` — which components contain the watched VPs
5. Per-Type1-migration diff — shows any further changes
6. `FINAL STATE OF WATCHED VPs`

**Dependencies**:
- `scripts/data_loader.py`
- `src/core/topology_switcher.py`
- `src/core/type1_component_analyzer.py`
- `src/core/steiner_handler.py`
- `src/core/perimeter_optimizer.py`
- `src/core/type2_migration_history.py`
- `src/core/migration_utils.py`

---

### Test 5: Phase A Vectorised Hessian Equivalence

**File**: ~~`test_phase_a_vectorised_hessian.py`~~ _(deleted 2026-05-05)_

**Status**: ✅ APPROVED → 🗑️ DELETED

**Created**: 2026-05-05

**Approved and deleted**: 2026-05-05

**Objective**:
Lock in the Phase A refactor described in `docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md` §3 — replacing per-row Python loops in `compute_perimeter_hessian_sparse` and `compute_area_hessian_sparse` with vectorised `np.add.at` calls on pre-computed offset arrays. Guards against any future regression where the new offset arrays drift out of sync with `hess_offset_map`.

**What it validates**:
- The new vectorised path (using `seg_hess_off_*`, `btri{1,2}_hess_off_*`) returns numerically identical Hessian values (to `atol=1e-12, rtol=0`) as the legacy per-row Python loop.
- Both kernels agree across three regimes: regular perimeter Hessian, area Hessian with zero multipliers (no contribution), and area Hessian with a deterministic random multiplier vector (`np.random.default_rng(42)`).
- All nine Phase A offset fields on `PartitionArrays` are populated (non-`None`) after `compile_arrays()`.

**Implementation**:
The same process is used to compare both paths — no git stashing needed. The legacy fallback `else:` branches in `compute_perimeter_hessian_sparse` and `compute_area_hessian_sparse` are byte-for-byte the original code; the test forces them by monkey-patching the new offset fields to `None` on the live `PartitionArrays` instance, runs the kernel, then restores the offsets and re-runs.

**Usage**:
```bash
# Any base solution or refined-contours iteration file
python testing/test_phase_a_vectorised_hessian.py \
    --solution results/run_xyz/solution_level0.h5

# Tighten tolerance (default already 1e-12)
python testing/test_phase_a_vectorised_hessian.py \
    --solution results/run_xyz/iteration_001_*.h5 \
    --atol 1e-14
```

**Command-line arguments**:
- `--solution`: Path to a Phase 1 base solution or Phase 2 refined-contours HDF5 file (required)
- `--atol`: Absolute tolerance for `np.allclose` (default `1e-12`)
- `--seed`: RNG seed for the random-multiplier area Hessian case (default `42`)

**Expected output**:
```
Phase A vectorised-Hessian regression test
  solution = results/.../surface_part3_..._20260405_121807.h5
  atol     = 1e-12

  n_active_vp = 122
  n_cells     = 3  (constrained: 2)
  hess_nnz    = 250
  segments    = 128
  btri rows   = 232

[1/3] compute_perimeter_hessian_sparse
  [PASS] perimeter Hessian                ||H_new||_∞=1.322e+01  max|Δ|=5.551e-17
[2/3] compute_area_hessian_sparse  (multipliers = 0)
  [PASS] area Hessian (mu=0)              ||H_new||_∞=0.000e+00  max|Δ|=0.000e+00
[3/3] compute_area_hessian_sparse  (multipliers = N(0,1), seed=42)
  [PASS] area Hessian (random mu)         ||H_new||_∞=5.289e-02  max|Δ|=3.081e-33

  worst max|Δ| over all cases = 5.551e-17
RESULT: PASS
```

**Dependencies**:
- `src/partition/partition_arrays.py` (Phase A offset fields)
- `src/partition/contour_partition.py` (`compile_arrays()` populates them)
- `src/partition/vectorized_perimeter.py` (`compute_perimeter_hessian_sparse`)
- `src/partition/vectorized_area.py` (`compute_area_hessian_sparse`)
- `src/optimization/perimeter_optimizer.py`, `src/pipeline/io.py`

**Why deleted**:
The test was written as a one-time refactor validator; its only mechanism was forcing the legacy Python-loop fallback via monkey-patching and comparing against `np.add.at` output. Once the refactor was confirmed correct (worst `max|Δ| = 5.6e-17`) and the fallback branches were removed from the source, the test had no further comparison reference and no ongoing regression value. Future Hessian correctness is covered by the Phase B harness (`test_exact_hessian_vs_fd.py`), which tests against mathematical ground truth (finite differences) rather than against a deleted code path. Historical validation results are recorded in this entry.

---

## Planned Tests

_This section tracks future tests that need to be implemented. Add new test proposals here._

---

## Contact

For questions about tests or to report issues:
- Update test status in this document
- Add notes section to relevant test entry
- Include date and description of investigation

---

## Version History

| Date | Change | Author |
|------|--------|--------|
| 2026-05-05 | Added Test 5 (Phase A vectorised Hessian); validated and deleted after removing fallback branches | System |
| 2026-03-24 | Deleted test_lambda_edge_roundtrip.py (resolved diagnostic); noted broken import in test_self_healing_selection.py; promoted refine_perimeter_iterative.py to APPROVED; added Test 5 (vectorized evaluation) | System |
| 2026-03-03 | Added Test 4 (migration debug, no opt/export) | System |
| 2026-02-08 | Added Test 3 (iterative perimeter refinement) | System |
| 2026-02-06 | Updated Test 2 (efficiency refactoring) | System |
| 2026-02-04 | Added Test 2 (migration and continue optimization) | System |
| 2026-01-26 | Created test registry, added Test 1 (self-healing) | System |
