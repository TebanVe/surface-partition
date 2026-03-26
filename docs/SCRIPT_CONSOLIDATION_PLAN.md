# Script Consolidation Plan: Unified Iterative Refinement Script

## Goal

Consolidate two testing scripts into one unified script:

- **Keep and extend**: `testing/refine_perimeter_iterative.py`
- **Absorb and delete**: `testing/test_migration_and_continue.py`

The result is a single script that handles both fresh runs (from base relaxation output)
and resumed runs (from a saved iteration checkpoint), with a clean and consistent
Optimize → Export → Detect → Migrate loop order.

---

## Background: What the Two Scripts Currently Do

### `refine_perimeter_iterative.py` — the survivor

**Loop order:** Optimize → Export → Detect → Migrate

**Loads from:** Base solution HDF5 (`x_opt`, `vertices`, `faces`) via `ContourAnalyzer`.

**Per-iteration structure:**
1. Optimize current topology (`PerimeterOptimizer`)
2. Export pre-migration snapshot → `_btol{tol}_iteration{N}_refined_contours.h5`
3. Detect topology switches (boundary VPs / triple points)
4. If none: converged, exit
5. Apply migrations (Type 2 first, then Type 1)
6. Loop

**What it lacks:**
- Cannot load from an iteration checkpoint file (no resume)
- Does not detect the starting iteration number from the filename
- Does not run `_check_indicator_vp_consistency` after migration
- Does not run a perimeter roundtrip check after export
- Does not call `rebuild_triangle_segments_from_current_vps()` after Type 2 (legacy path)
- Does not do pre-execution VP revalidation before Type 1 (legacy path)
- Does not load migration history when resuming (legacy path)
- All iteration files are always saved (no way to suppress them)

### `test_migration_and_continue.py` — to be deleted

**Loop order:** Detect → Migrate → Optimize → Export

**Loads from:** Refined/iteration checkpoint HDF5 (`lambda_parameters`, `vp_edges`,
`indicator_functions`) via `examples/data_loader.load_partition_from_refined_file`.

**Key features to extract and carry over:**
1. `load_partition_from_refined_file` loading path
2. Regex-based starting iteration detection from filename
3. Migration history load from HDF5 when resuming (legacy path)
4. `_check_indicator_vp_consistency` after migration
5. Perimeter roundtrip check after export (reload and compare)
6. `rebuild_triangle_segments_from_current_vps()` after Type 2 migrations (legacy path)
7. `boundary_segments` drift check after each Type 2 (legacy path)
8. Pre-execution VP revalidation before each Type 1 migration (legacy path)

**What must NOT be carried over:**
- The Detect → Migrate → Optimize loop order (this is wrong for production)

---

## Design Decisions (already agreed upon — do not change)

### 1. Loop order is always Optimize → Export → Detect → Migrate

This is the correct scientific order. The optimization result at a given topology is
the scientific output. Migration is a topological event enabling the next iteration.

### 2. The iteration file is always the post-optimization, pre-migration state

```
ITERATION N:
  1. Optimize  (on current topology)
  2. Export    ← _iteration{N}_refined_contours.h5 (post-opt, pre-migration)
  3. Detect switches
       → none: CONVERGED, exit (file from step 2 is the final result)
       → some: Migrate
[next iteration starts on migrated topology]
```

This means:
- The last iteration's file always contains the clean optimized state (no pending migrations)
- The converged final file IS an iteration file — no separate "final" export needed

### 3. Auto-detect file type at load time (no --resume flag needed)

Inspect HDF5 keys on startup:

```
if 'x_opt' in file:
    # Base relaxation output
    → load via ContourAnalyzer (existing path)
    → enter loop at step 1 (Optimize)
    → starting_iteration = 0
else:
    # Iteration checkpoint (lambda_parameters / vp_edges / indicator_functions)
    → load via load_partition_from_refined_file
    → enter loop at step 3 (Detect → Migrate) for first pass, then normal loop
    → parse starting_iteration from filename
```

When loading from an iteration checkpoint, the first action is migration (step 3 of the
previous iteration that was saved pre-migration). After that, the loop runs normally.

### 4. Auto-detect starting iteration number from filename

Use the same regex already in `test_migration_and_continue.py`:

```python
import re
match = re.search(r'_iteration(\d+)_refined_contours\.h5$', args.solution)
starting_iteration = int(match.group(1)) if match else 0
```

Output files continue from `starting_iteration + 1`.

### 5. `--save-iterations` flag controls intermediate file export

- **Default (flag absent):** only the final converged file is saved. No intermediate files.
- **`--save-iterations`:** saves `_iteration{N}_refined_contours.h5` after each optimization step.

**Important note for the implementer:** When `--save-iterations` is not set, intermediate
checkpoints are not written, which means the run cannot be resumed mid-way. For long
production runs the user should consider enabling this flag as a safety net.

Optional enhancement (not required but clean): `--save-every N` saves only every N-th
iteration instead of all of them.

---

## HDF5 File Format Reference

### Base solution file (output of `examples/find_surface_partition.py`)

| Key / Attr | Type | Description |
|-----------|------|-------------|
| `x_opt` | dataset | Flat optimizer density vector, shape `(N * n_partitions,)` |
| `x0` | dataset | Initial condition |
| `vertices` | dataset | Mesh vertices `(N, 2 or 3)` |
| `faces` | dataset | Mesh faces `(T, 3)` int32 |
| `n_partitions` attr | int | Number of partition cells |
| `seed`, `lambda_penalty`, etc. | attrs | Run metadata |

**Detection:** presence of `x_opt` key identifies this as a base solution file.

### Iteration checkpoint file (output of the unified script)

| Key / Attr | Type | Description |
|-----------|------|-------------|
| `lambda_parameters` | dataset | Optimized VP λ values (active VPs only) |
| `vp_edges` | dataset | Edge indices per VP `(n_vps, 2)` int64 |
| `indicator_functions` | dataset | Updated vertex-cell labels `(N, n_cells)` after migration |
| `n_variable_points` attr | int | Number of active VPs |
| `n_cells` attr | int | Number of partition cells |
| `final_perimeter` attr | float | Perimeter at export time |
| `optimization_success` attr | bool | Whether optimizer converged |
| `optimization_iterations` attr | int | Number of optimizer iterations |
| `iteration_number` attr | int | Iteration counter |
| `timestamp` attr | str | ISO timestamp |
| `optimization_info/` group | group | `initial_perimeter`, `final_perimeter`, `perimeter_reduction`, `percent_reduction`, `constraint_violations` |
| (legacy) migration history | group | Saved by `save_type2_migration_history()` if `--use-legacy` |

**Detection:** presence of `lambda_parameters` key (and absence of `x_opt`) identifies
this as an iteration checkpoint.

**Critical:** the `indicator_functions` in iteration files reflect the topology **after
migration** (from the previous iteration). This is what allows correct resume: when you
load an iteration file, the indicators describe the migrated topology that needs to be
optimized next.

**Wait — contradiction with design decision #2?**

This is a subtle but important point. The iteration file is saved pre-migration (step 2 of
the current iteration). But to correctly resume from it, the file needs to represent the
*pre-migration* state of the current topology. When this file is loaded:
- `indicator_functions` = the topology that was just optimized (pre-migration)
- `lambda_parameters` = the optimized VP positions on that topology
- First action on resume = run migration on this topology, then optimize the result

So the `indicator_functions` in the file describe the **pre-migration** topology of the
iteration that was saved. This is consistent. The data_loader.py `load_partition_from_refined_file`
handles this correctly — it reads `indicator_functions` from the file and uses them to
reconstruct the partition.

---

## Features to Carry Over: Implementation Details

### Feature 1: Load from iteration checkpoint

Replace the load block in `main()` with conditional logic:

```python
import h5py, re

def detect_file_type(path):
    with h5py.File(path, 'r') as f:
        return 'base' if 'x_opt' in f else 'checkpoint'

file_type = detect_file_type(args.solution)

if file_type == 'base':
    # existing ContourAnalyzer path (already in refine_perimeter_iterative.py)
    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    indicators = analyzer.compute_indicator_functions()
    _, boundary_topology = analyzer.extract_contours_with_topology()
    partition = PartitionContour(mesh, indicators, boundary_topology=boundary_topology)
    starting_iteration = 0
    resume_mode = False
else:
    # carry over from test_migration_and_continue.py
    from examples.data_loader import load_partition_from_refined_file
    mesh, partition = load_partition_from_refined_file(args.solution, verbose=True)
    match = re.search(r'_iteration(\d+)_refined_contours\.h5$', args.solution)
    starting_iteration = int(match.group(1)) if match else 1
    resume_mode = True
```

### Feature 2: Resume entry point

The main loop needs a one-time flag for the resume case. On the first pass of a resumed
run, skip optimization (it was already done before the file was saved) and go straight to
detection and migration:

```python
skip_optimize_this_iteration = resume_mode  # True only on first pass of a resume

while not converged and topology_iteration < args.max_iterations:
    iteration_number = starting_iteration + topology_iteration + 1

    if not skip_optimize_this_iteration:
        # Phase 1: Optimize
        ...
        # Phase 2: Export (conditional on --save-iterations or last iteration)
        ...
    else:
        # Resuming: load opt_info from the checkpoint file for logging continuity
        skip_optimize_this_iteration = False

    # Phase 3: Detect + migrate (always runs)
    ...
```

### Feature 3: Starting iteration number and output naming

```python
iteration_number = starting_iteration + topology_iteration + 1
output_file = f"{base_output}_btol{args.boundary_tol}_iteration{iteration_number}_refined_contours.h5"
```

### Feature 4: `--save-iterations` flag

Add to argument parser:
```python
parser.add_argument('--save-iterations', action='store_true',
    help='Save an HDF5 checkpoint after each optimization step. '
         'Required for mid-run resume. Default: off (only final result saved).')
```

In the loop, gate the export:
```python
should_export = args.save_iterations or no_switches_detected or topology_iteration + 1 == args.max_iterations
if should_export:
    export_intermediate_state(...)
```

Always export on convergence and on the last iteration regardless of the flag.

### Feature 5: `_check_indicator_vp_consistency` after migration

Copy this function verbatim from `test_migration_and_continue.py` (lines 519–567).
Call it after the migration phase (Phase 4), before the next optimization:

```python
_check_indicator_vp_consistency(partition, mesh, logger)
```

### Feature 6: Perimeter roundtrip check in `export_intermediate_state`

The current `export_intermediate_state` in `refine_perimeter_iterative.py` does not
do a roundtrip check. After writing the HDF5 file, add the roundtrip block from
`test_migration_and_continue.py`'s `export_results` function (lines 637–673):

```python
# After h5py.File write closes:
try:
    from examples.data_loader import load_partition_from_refined_file
    mesh_r, partition_r = load_partition_from_refined_file(output_path, verbose=False)
    # ... recompute perimeter and compare to opt_info['final_perimeter']
except Exception as e:
    logger.warning(f"Roundtrip check skipped: {e}")
```

### Feature 7: `rebuild_triangle_segments_from_current_vps()` after Type 2 (legacy path)

In `apply_type2_migrations` (legacy path), after each successful Type 2 migration, call:
```python
partition.rebuild_triangle_segments_from_current_vps()
```
This is present in `test_migration_and_continue.py` but missing from
`refine_perimeter_iterative.py`.

### Feature 8: `boundary_segments` drift check after Type 2 (legacy path)

Copy the consistency check block from `test_migration_and_continue.py`'s
`apply_type2_migrations` function (lines 376–408) into the equivalent function in
`refine_perimeter_iterative.py`.

### Feature 9: Pre-execution VP revalidation before Type 1 (legacy path)

In the Type 1 migration loop, before calling `switcher.apply_type1_switch_v2`, add the
revalidation check from `test_migration_and_continue.py` (lines 221–239):

```python
for vp_idx in aux_vps:
    vp = partition.variable_points[vp_idx]
    if target_vertex not in vp.edge:
        # skip this migration — earlier migration invalidated this plan entry
        skip_migration = True
        break
```

### Feature 10: Load migration history from file on resume (legacy path)

When `resume_mode=True` and `--use-legacy`, load the saved history:

```python
if resume_mode and args.use_legacy and starting_iteration > 1:
    try:
        with h5py.File(args.solution, 'r') as f:
            migration_history = load_type2_migration_history(f)
        logger.info(f"Loaded migration history: {len(migration_history.records)} records")
    except Exception as e:
        logger.warning(f"Could not load migration history: {e}. Starting fresh.")
        migration_history = Type2MigrationHistory()
```

---

## Final summary of changes to `refine_perimeter_iterative.py`

| # | Change | Source |
|---|--------|--------|
| 1 | Dual loading: base solution OR iteration checkpoint | `test_migration_and_continue.py` |
| 2 | Resume entry point (skip optimize on first pass) | new logic |
| 3 | Auto-detect starting iteration from filename via regex | `test_migration_and_continue.py` |
| 4 | `--save-iterations` flag; always export on convergence/last iter | new argument |
| 5 | `_check_indicator_vp_consistency` after migration | `test_migration_and_continue.py` |
| 6 | Perimeter roundtrip check in export function | `test_migration_and_continue.py` |
| 7 | `rebuild_triangle_segments_from_current_vps()` after Type 2 (legacy) | `test_migration_and_continue.py` |
| 8 | `boundary_segments` drift check after Type 2 (legacy) | `test_migration_and_continue.py` |
| 9 | Pre-execution VP revalidation before Type 1 (legacy) | `test_migration_and_continue.py` |
| 10 | Load migration history from file on resume (legacy) | `test_migration_and_continue.py` |

---

## After implementation

1. Verify the unified script works on both input types (base solution and iteration file).
2. Delete `testing/test_migration_and_continue.py`.
3. Update the final summary block in `refine_perimeter_iterative.py` — remove the
   suggestion to use `test_migration_and_continue.py` (line 1010), replace with
   the resume command using the unified script itself.
4. Update `testing/README_testing.md` to reflect that only one script exists.

---

## Files to read before implementing

Read these files in full before writing a single line of code:

1. `testing/refine_perimeter_iterative.py` — the script being extended (1021 lines)
2. `testing/test_migration_and_continue.py` — the source of features to carry over (1087 lines)
3. `examples/data_loader.py` — `load_partition_from_refined_file` function (266 lines)
4. `src/find_contours.py` — `ContourAnalyzer` (296 lines)

Do not read or modify any other files. All changes are confined to
`testing/refine_perimeter_iterative.py`.
