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

### Test 2: Migration and Continue Optimization

**File**: `test_migration_and_continue.py`

**Status**: 🚧 IN PROGRESS

**Created**: 2026-02-04

**Approved**: _Pending testing and validation_

**Objective**: 
Test the complete workflow cycle after topology switches: Load refined partition → Apply migrations → Optimize → Export. This validates that the optimization can successfully continue after migrations are applied, with proper calculator rebuilding and indicator function updates.

**What it validates**:
- Loading refined contours from HDF5 file (same format as visualization scripts use)
- Detection of topology switches (Type 1 and Type 2)
- Application of migrations with proper indicator_functions updates
- Rebuilding of calculators (AreaCalculator, PerimeterCalculator, SteinerHandler) after migrations
- Optimization convergence after migrations
- Export in same HDF5 format as input (compatible with visualization scripts)
- One complete cycle: migrate → optimize → export

**Key differences from `refine_perimeter.py`**:
- **Input**: Reads refined contours `*_refined_contours.h5` (not initial solution)
- **Workflow**: Migrations FIRST, then optimization (inverse of refine_perimeter.py)
- **Output**: New iteration file `*_iteration2_refined_contours.h5` in same format as input
- **Scope**: One cycle only (not iterative loop)

**Usage**:
```bash
# Apply both types of migrations
python testing/test_migration_and_continue.py \
    --solution results/run_xyz/*_refined_contours.h5 \
    --migration-type both

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
- `--output`: Output path (default: auto-generated in same directory as input)
- `--max-opt-iter`: Maximum optimization iterations (default: 1000)
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
| 2026-02-04 | Added Test 2 (migration and continue optimization) | System |
| 2026-01-26 | Created test registry, added Test 1 (self-healing) | System |
