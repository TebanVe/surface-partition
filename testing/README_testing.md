# Test Registry and Tracking

**Purpose**: This document tracks all tests implemented for the RingTest project, providing a centralized registry for test status, objectives, results, and maintenance.

**Last Updated**: 2026-01-26 (January 26, 2026)

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

### Test 1: Self-Healing Component Selection

**File**: `test_self_healing_selection.py`

**Status**: ✅ APPROVED

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
- `examples/visualize_precise_region.py` (for load_partition_from_refined_file)

**Notes**:
- Test uses deep copy of partition to compare states across iterations
- Tracks VP movements to explain why auxiliaries break
- Provides detailed logging for debugging

---

### Test 2: Migration and Continue Optimization (with Iterative Support)

**File**: `test_migration_and_continue.py`

**Status**: ✅ APPROVED (Iterative functionality added 2026-02-09)

**Created**: 2026-02-04

**Last Updated**: 2026-02-09 (Added iterative refinement capability)

**Approved**: 2026-02-09

**Objective**: 
Test the migrate→optimize workflow starting from a refined partition file. This script can now operate in two modes:
1. **Single-cycle mode** (default, `--max-iterations 1`): Apply migrations once, optimize, export
2. **Iterative mode** (`--max-iterations > 1`): Repeatedly apply migrations and optimize until convergence or max iterations

**New Features (2026-02-09)**:
- **Iterative refinement support**: Can run multiple migrate→optimize cycles until convergence
- **Smart iteration detection**: Automatically detects starting iteration number from input filename
- **Sequential numbering**: If input is `iteration5`, outputs are `iteration6`, `iteration7`, etc.
- **Convergence detection**: Stops when no more topology switches are detected
- **Backward compatible**: Default `--max-iterations 1` preserves original single-cycle behavior

**What it validates**:
- Loading refined contours from HDF5 file (iteration files or first refined file)
- Detection of topology switches (Type 1 and Type 2)
- Application of migrations with proper indicator_functions updates
- **Efficient calculator reuse** (pre-migration and post-migration phases)
- Optimization convergence after migrations
- Export in same HDF5 format as input (compatible with visualization scripts)
- **Iterative workflow**: migrate → optimize → detect → [repeat until convergence]

**Key differences from other scripts**:
- **vs `refine_perimeter.py`**: Uses new v2/v4 migration methods, opposite workflow order
- **vs `refine_perimeter_iterative.py`**: Different workflow (MIGRATE first vs OPTIMIZE first), different starting point (iteration file vs base solution)
- **Complementary roles**: 
  - `refine_perimeter_iterative.py`: Start from BASE solution, optimize→migrate loop
  - `test_migration_and_continue.py`: Start from ITERATION file, migrate→optimize loop

**Usage**:
```bash
# Single cycle (backward compatible - default)
python testing/test_migration_and_continue.py \
    --solution results/run_xyz/*_refined_contours.h5 \
    --migration-type both

# Iterative refinement from iteration 1 (up to 10 cycles)
python testing/test_migration_and_continue.py \
    --solution results/run_xyz/*_iteration1_refined_contours.h5 \
    --migration-type both \
    --max-iterations 10

# Resume from iteration 5 (continue for 5 more iterations)
python testing/test_migration_and_continue.py \
    --solution results/run_xyz/*_iteration5_refined_contours.h5 \
    --migration-type both \
    --max-iterations 5

# Apply only Type 1 migrations
python testing/test_migration_and_continue.py \
    --solution results/run_xyz/*_refined_contours.h5 \
    --migration-type type1

# Apply only Type 2 migrations with custom output
python testing/test_migration_and_continue.py \
    --solution results/run_xyz/*_refined_contours.h5 \
    --migration-type type2 \
    --output results/run_xyz/*_custom_output.h5
```

**Command-line arguments**:
- `--solution`: Path to refined contours HDF5 file (required)
- `--migration-type`: Which migrations to apply: `type1`, `type2`, or `both` (default: `both`)
- `--max-iterations`: Maximum refinement cycles (default: 1 for backward compatibility)
- `--output`: Output path (default: auto-generated with correct iteration number)
- `--max-opt-iter`: Maximum optimization iterations per cycle (default: 1000)
- `--tolerance`: Optimization tolerance (default: 1e-7)
- `--boundary-tol`: Boundary detection threshold (default: 1e-3)
- `--method`: Optimization method: `SLSQP` or `trust-constr` (default: SLSQP)
- `--log-level`: Logging verbosity (default: INFO)

**Expected output**:
The script performs these stages with detailed logging:
1. **Load refined data**: Mesh, partition, indicator functions, λ parameters
2. **Initial diagnostics**: Current perimeter, areas, VPs, triple points
3. **Detect switches**: Identify Type 1 and Type 2 candidates
4. **Apply migrations**: Execute selected migration types with per-migration logging
5. **Rebuild calculators**: Recreate AreaCalculator, PerimeterCalculator, SteinerHandler
6. **Run optimization**: One optimization iteration with convergence tracking
7. **Export results**: Save to HDF5 with metadata
8. **Post-optimization analysis**: Detect if new switches are needed

**Success criteria**:
```
✓ Data loaded successfully
✓ Switches detected
✓ Migrations applied (> 0 successful)
✓ Calculators rebuilt
✓ Optimization converged
✓ Results exported
✓ Post-analysis completed
TEST COMPLETED SUCCESSFULLY
```

**Failure handling**:
- Script aborts with clear error messages if:
  - Input file not found
  - No switches detected
  - No migrations applied
  - Optimization fails to converge
  - Export fails

**Dependencies**:
- `examples/data_loader.py` (load_partition_from_refined_file)
- `src/core/tri_mesh.py`
- `src/core/contour_partition.py`
- `src/core/area_calculator.py`
- `src/core/perimeter_calculator.py`
- `src/core/steiner_handler.py`
- `src/core/perimeter_optimizer.py`
- `src/core/mesh_topology.py`
- `src/core/topology_switcher.py`
- `src/logging_config.py`

**Notes**:
- Output file format matches input format (visualization scripts can directly read it)
- Log file created in `logs/` directory with timestamp
- Default output location: same directory as input file
- If no switches detected, script exits gracefully (partition already converged)
- Maximum one complete cycle per execution

**Integration**:
After successful test, results can be visualized with:
```bash
python examples/visualize_type1_vertex_collapse.py \
    --solution results/run_xyz/*_iteration2_refined_contours.h5 \
    --region X --component-index Y --state before
```

---

### Test 3: Iterative Perimeter Refinement with Automatic Migrations

**File**: `refine_perimeter_iterative.py`

**Status**: 🚧 IN PROGRESS (Implementation complete, awaiting runtime testing)

**Created**: 2026-02-08

**Last Updated**: 2026-02-08

**Approved**: _Pending testing and validation_

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

**Key differences from other scripts**:
| Feature | `refine_perimeter.py` | `test_migration_and_continue.py` | `refine_perimeter_iterative.py` |
|---------|----------------------|----------------------------------|----------------------------------|
| **Input** | Initial solution | Refined contours | Initial solution |
| **Migrations** | Old methods (broken) | New v2/v4 methods | New v2/v4 methods |
| **Workflow** | Opt → detect → stop | Migrate → optimize (1 cycle) | Loop(optimize → detect → migrate) |
| **Iterations** | Stops at first switch | Single cycle | Multiple until convergence |
| **Output** | Single refined file | Single iteration file | Multiple iteration files + final |
| **Purpose** | Legacy (outdated) | Testing/debugging | **Production workflow** |

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
    
      python examples/visualize_type1_vertex_collapse.py \
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
python examples/visualize_partition.py \
    --solution results/run_xyz/*_iteration2_refined_contours.h5 \
    --region 2 --show-steiner

# View final converged state
python examples/visualize_partition.py \
    --solution results/run_xyz/*_refined_contours.h5 \
    --region 2 --show-steiner

# Debug specific component
python examples/visualize_type1_vertex_collapse.py \
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
- Builds on efficiency patterns validated in Test 2 (`test_migration_and_continue.py`)
- Uses component analysis architecture validated in Test 1 (`test_self_healing_selection.py`)
- Serves as the production workflow that Test 2 prepares for

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

**Key differences from `test_migration_and_continue.py`**:
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
- `examples/data_loader.py`
- `src/core/topology_switcher.py`
- `src/core/type1_component_analyzer.py`
- `src/core/steiner_handler.py`
- `src/core/perimeter_optimizer.py`
- `src/core/type2_migration_history.py`
- `src/core/migration_utils.py`

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
| 2026-03-03 | Added Test 4 (migration debug, no opt/export) | System |
| 2026-02-08 | Added Test 3 (iterative perimeter refinement) | System |
| 2026-02-06 | Updated Test 2 (efficiency refactoring) | System |
| 2026-02-04 | Added Test 2 (migration and continue optimization) | System |
| 2026-01-26 | Created test registry, added Test 1 (self-healing) | System |
