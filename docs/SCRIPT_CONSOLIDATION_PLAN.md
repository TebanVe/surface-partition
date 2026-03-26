# Script Consolidation Plan: Unified Iterative Refinement Script

## Goal

Consolidate two testing scripts into one unified script:

- **Keep and extend**: `testing/refine_perimeter_iterative.py`
- **Absorb and delete**: `testing/test_migration_and_continue.py`

The result is a single script that handles both fresh runs (from base relaxation output)
and resumed runs (from a saved iteration checkpoint), with a clean and consistent loop.

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
- Does not store `pending_migration` flag in exported files
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

### 1. Loop order is always Optimize → Detect → Export → Migrate

This is the correct scientific order. The optimization result at a given topology is
the scientific output. Detection runs immediately after optimization (it is fast —
just proximity checks). The result is exported with a `pending_migration` flag that
records whether migration is needed. Migration is applied last, enabling the next
iteration.

**Note:** Detection is moved before export (compared to the old script). This is
necessary so the export file can record its own status (`pending_migration`), which
makes resume unambiguous.

### 2. The iteration file always records its own pending state

```
LOOP BODY:
  1. Optimize    (on current topology)
  2. Detect switches
  3. Export      ← _iteration{N}_refined_contours.h5
                    pending_migration = False  (no switches found → converged)
                    pending_migration = True   (switches found → migration will follow)
  4. If pending_migration=False → CONVERGED, exit
  5. Apply migrations (Type 2 first, then Type 1)
  [loop back to step 1 on migrated topology]
```

The `pending_migration` attribute stored in the HDF5 file makes every checkpoint
self-describing. The resume logic reads this flag to determine the correct entry point.

### 3. Three file states — each with one unambiguous action

| File type | `pending_migration` attr | First action on load |
|-----------|--------------------------|----------------------|
| Base solution (`x_opt` present) | n/a | **Optimize** (step 1) |
| Iteration checkpoint | `True` | **Migrate** (step 5), then Optimize |
| Iteration checkpoint | `False` | **Done** — converged result, nothing to do |

When resuming from an iteration checkpoint with `pending_migration=True`, **migration
always comes before optimization**. This is because the file represents a topology that
has been optimized but not yet migrated — the next action is always the topology change.

The only case where optimization comes first without prior migration is when starting
from a base solution file.

### 4. Auto-detect file type at load time (no `--resume` flag needed)

Inspect HDF5 keys on startup:

```python
def detect_file_type(path):
    with h5py.File(path, 'r') as f:
        if 'x_opt' in f:
            return 'base'
        pending = f.attrs.get('pending_migration', True)  # conservative default
        return 'checkpoint_pending' if pending else 'checkpoint_converged'
```

The load block then selects the correct entry point:

```python
file_type = detect_file_type(args.solution)

if file_type == 'base':
    # Fresh run — load via ContourAnalyzer
    load_base_solution(args.solution)
    starting_iteration = 0
    enter_at = 'optimize'

elif file_type == 'checkpoint_pending':
    # Resume — migration was pending when this file was saved
    load_iteration_checkpoint(args.solution)
    starting_iteration = parse_iteration_from_filename(args.solution)
    enter_at = 'migrate'   # migrate first, then optimize

elif file_type == 'checkpoint_converged':
    # Already converged — nothing to do
    logger.info("Input file is a converged result. No further work needed.")
    return 0
```

### 5. Auto-detect starting iteration number from filename

Use the same regex already in `test_migration_and_continue.py`:

```python
import re
match = re.search(r'_iteration(\d+)_refined_contours\.h5$', args.solution)
starting_iteration = int(match.group(1)) if match else 0
```

Output files continue from `starting_iteration + 1`.

### 6. `--save-iterations` flag controls intermediate file export

- **Default (flag absent):** only the final converged file is saved (the one with
  `pending_migration=False`). No intermediate files are written.
- **`--save-iterations`:** saves `_iteration{N}_refined_contours.h5` after every
  detect+export step regardless of whether migration is pending.

**Important:** When `--save-iterations` is not set, intermediate checkpoints are not
written, which means the run cannot be resumed if it is interrupted mid-way. For long
production runs, enable this flag as a safety net.

The export is always written unconditionally when:
- `pending_migration=False` (convergence): this IS the final result.
- The loop reaches `max_iterations`: force-export the last state.

```python
is_last_iteration = (topology_iteration + 1 == args.max_iterations)
should_export = args.save_iterations or (not pending_migration) or is_last_iteration
if should_export:
    export_intermediate_state(..., pending_migration=pending_migration)
```

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

**Detection:** presence of `x_opt` key → base solution file → enter at Optimize.

### Iteration checkpoint file (output of the unified script)

| Key / Attr | Type | Description |
|-----------|------|-------------|
| `lambda_parameters` | dataset | Optimized VP λ values (active VPs only) |
| `vp_edges` | dataset | Edge indices per VP `(n_vps, 2)` int64 |
| `indicator_functions` | dataset | Vertex-cell labels `(N, n_cells)` for current topology |
| `pending_migration` attr | bool | **True** = migration was detected, not yet applied; **False** = converged |
| `n_variable_points` attr | int | Number of active VPs |
| `n_cells` attr | int | Number of partition cells |
| `final_perimeter` attr | float | Perimeter at export time |
| `optimization_success` attr | bool | Whether optimizer converged |
| `optimization_iterations` attr | int | Number of optimizer iterations |
| `iteration_number` attr | int | Iteration counter |
| `timestamp` attr | str | ISO timestamp |
| `optimization_info/` group | group | `initial_perimeter`, `final_perimeter`, `perimeter_reduction`, `percent_reduction`, `constraint_violations` |
| (legacy) migration history | group | Saved by `save_type2_migration_history()` if `--use-legacy` |

**Detection:** presence of `lambda_parameters` key (and absence of `x_opt`) → iteration
checkpoint. Read `pending_migration` to determine entry point.

**What `indicator_functions` contains:** the topology of the iteration that was just
optimized — i.e., the state **before** the pending migration. When the checkpoint is
loaded for resume, this topology is reconstructed and migration is applied to it first.

---

## Features to Carry Over: Implementation Details

### Feature 1: Load from iteration checkpoint

Replace the load block in `main()` with the three-way conditional:

```python
import h5py, re

def detect_file_type(path):
    with h5py.File(path, 'r') as f:
        if 'x_opt' in f:
            return 'base'
        pending = bool(f.attrs.get('pending_migration', True))
        return 'checkpoint_pending' if pending else 'checkpoint_converged'

file_type = detect_file_type(args.solution)

if file_type == 'base':
    # Existing ContourAnalyzer path (already in refine_perimeter_iterative.py)
    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    indicators = analyzer.compute_indicator_functions()
    _, boundary_topology = analyzer.extract_contours_with_topology()
    partition = PartitionContour(mesh, indicators, boundary_topology=boundary_topology)
    starting_iteration = 0
    enter_at = 'optimize'

elif file_type == 'checkpoint_pending':
    # Carry over from test_migration_and_continue.py
    from examples.data_loader import load_partition_from_refined_file
    mesh, partition = load_partition_from_refined_file(args.solution, verbose=True)
    match = re.search(r'_iteration(\d+)_refined_contours\.h5$', args.solution)
    starting_iteration = int(match.group(1)) if match else 1
    enter_at = 'migrate'

elif file_type == 'checkpoint_converged':
    logger.info("Input file is a converged result (pending_migration=False). Nothing to do.")
    return 0
```

### Feature 2: Unified loop with `enter_at` control

The main loop handles both entry points cleanly with a one-time flag:

```python
first_pass = True

while not converged and topology_iteration < args.max_iterations:
    iteration_number = starting_iteration + topology_iteration + 1

    # ---- ENTRY POINT: MIGRATE (resume from checkpoint) ----
    if first_pass and enter_at == 'migrate':
        apply_migrations(...)   # migrate the loaded topology
        first_pass = False
        topology_iteration += 1
        continue                # loop back to optimize

    first_pass = False

    # ---- PHASE 1: Optimize ----
    optimize(...)

    # ---- PHASE 2: Detect ----
    pending_migration = detect_switches(...)

    # ---- PHASE 3: Export ----
    is_last = (topology_iteration + 1 == args.max_iterations)
    if args.save_iterations or not pending_migration or is_last:
        export(..., pending_migration=pending_migration)

    # ---- PHASE 4: Check convergence ----
    if not pending_migration:
        converged = True
        break

    # ---- PHASE 5: Migrate ----
    apply_migrations(...)
    topology_iteration += 1
```

### Feature 3: `pending_migration` attribute in export

In `export_intermediate_state`, add the flag to the HDF5 file:

```python
with h5py.File(output_path, 'w') as f:
    # ... existing datasets ...
    f.attrs['pending_migration'] = bool(pending_migration)
```

This attribute is the key that makes every checkpoint self-describing.

### Feature 4: Starting iteration number and output naming

```python
iteration_number = starting_iteration + topology_iteration + 1
output_file = f"{base_output}_btol{args.boundary_tol}_iteration{iteration_number}_refined_contours.h5"
```

### Feature 5: `--save-iterations` argument

```python
parser.add_argument('--save-iterations', action='store_true',
    help='Save an HDF5 checkpoint after each optimization step. '
         'Enables mid-run resume. Default: off (only final converged result saved).')
```

### Feature 6: `_check_indicator_vp_consistency` after migration

Copy this function verbatim from `test_migration_and_continue.py` (lines 519–567).
Call it after Phase 5 (migration), before the next iteration's optimization:

```python
_check_indicator_vp_consistency(partition, mesh, logger)
```

### Feature 7: Perimeter roundtrip check in `export_intermediate_state`

After the HDF5 file is written and closed, add the roundtrip block from
`test_migration_and_continue.py`'s `export_results` function (lines 637–673):

```python
try:
    from examples.data_loader import load_partition_from_refined_file
    mesh_r, partition_r = load_partition_from_refined_file(output_path, verbose=False)
    # recompute perimeter from reloaded state and compare to opt_info['final_perimeter']
except Exception as e:
    logger.warning(f"Roundtrip check skipped: {e}")
```

### Feature 8: `rebuild_triangle_segments_from_current_vps()` after Type 2 (legacy path)

In `apply_type2_migrations` (legacy path), after each successful Type 2 migration:

```python
partition.rebuild_triangle_segments_from_current_vps()
```

Present in `test_migration_and_continue.py`, missing from `refine_perimeter_iterative.py`.

### Feature 9: `boundary_segments` drift check after Type 2 (legacy path)

Copy the consistency check block from `test_migration_and_continue.py`'s
`apply_type2_migrations` function (lines 376–408) into the equivalent function in
`refine_perimeter_iterative.py`.

### Feature 10: Pre-execution VP revalidation before Type 1 (legacy path)

In the Type 1 migration loop, before calling `switcher.apply_type1_switch_v2`, add the
revalidation check from `test_migration_and_continue.py` (lines 221–239):

```python
for vp_idx in aux_vps:
    vp = partition.variable_points[vp_idx]
    if target_vertex not in vp.edge:
        skip_migration = True  # earlier migration invalidated this plan entry
        break
```

### Feature 11: Load migration history from file on resume (legacy path)

When `enter_at == 'migrate'` and `--use-legacy`:

```python
if enter_at == 'migrate' and args.use_legacy:
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
| 2 | `checkpoint_converged` early-exit (pending_migration=False) | new logic |
| 3 | Unified loop with `enter_at` control for resume | new logic |
| 4 | `pending_migration` attr written to every exported file | new — core design |
| 5 | Detection moved before export (so file knows its own state) | design change |
| 6 | Auto-detect starting iteration from filename via regex | `test_migration_and_continue.py` |
| 7 | `--save-iterations` flag; always export on convergence/last iter | new argument |
| 8 | `_check_indicator_vp_consistency` after migration | `test_migration_and_continue.py` |
| 9 | Perimeter roundtrip check in export function | `test_migration_and_continue.py` |
| 10 | `rebuild_triangle_segments_from_current_vps()` after Type 2 (legacy) | `test_migration_and_continue.py` |
| 11 | `boundary_segments` drift check after Type 2 (legacy) | `test_migration_and_continue.py` |
| 12 | Pre-execution VP revalidation before Type 1 (legacy) | `test_migration_and_continue.py` |
| 13 | Load migration history from file on resume (legacy) | `test_migration_and_continue.py` |

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
