# Phase 1 (Relaxation) Timing Profile

**Status:** Planned. No code changes yet.
**Branch:** `feat/phase1-timing-profile`. The plan was committed on this
branch as its entry point; implementation work should happen here, not
on `main`. If checking out a different branch to read this document,
switch back before editing code.
**Audience:** Fresh agent / developer adding timing instrumentation to the
Γ-convergence relaxation pipeline.
**Goal of the work:** Reproduce, for Phase 1, the same kind of breakdown
that Phase 2 already produces via `--profile` and `timing_profile.yaml`,
so we can locate the dominant cost (mesh assembly vs. projection vs.
sparse matvecs vs. I/O) on real fine-mesh runs *before* we change any
optimization code.
**Scope:** Instrumentation + plumbing + reporting only. Do **not** also
vectorise the FEM assembly, fuse the per-partition sparse-matvec loop,
warm-start the projection, or change the HDF5 flush cadence in the same
change set. Those are the candidate optimizations this profiling exists
to rank — leave them for a follow-up.

---

## 0. Context: what already exists for Phase 2

Read these first to match conventions:

- `src/profiling.py` — `ProfilingState` dataclass: timed callbacks via
  `record(cb, elapsed)`, optional `record_steiner(cb, n)`,
  `finalize()` → derived `%` breakdown, `to_yaml_dict(...)` → the
  on-disk schema, `to_index_dict()` → flat scalars for
  `experiment_index.yaml`.
- `src/optimization/perimeter_optimizer.py:55-184` — `IPOPTProblemAdapter`
  shows the wrap pattern:
  ```python
  if self._profile is not None:
      t0 = time.perf_counter()
  result = ...
  if self._profile is not None:
      self._profile.record('objective', time.perf_counter() - t0)
  ```
  **Zero overhead when `profile is None`** — this is the contract.
- `src/pipeline/pipeline_orchestrator.py:204` — `RefinementConfig.profile`
  boolean field.
- `src/pipeline/pipeline_orchestrator.py:271-280` — lazy construction of
  the `ProfilingState` (mesh-derived counters captured at that point).
- `src/pipeline/pipeline_orchestrator.py:762-789` — finalize +
  `_write_timing_profile()`: writes `timing_profile.yaml` into the
  campaign directory.
- `scripts/refine_perimeter.py:86-87, 165-168` — CLI flag wiring.
- `sweep/parameter_sweep.py:470-511` —
  `_extract_timing_metrics()` + `_flatten_timing()`: how
  `experiment_index.yaml` picks up `timing_*` scalars during
  `--mode collect`.
- `sweep/timing_analyzer.py:52-79, 87-100` — how those scalars are
  consumed to produce scaling figures.

Mirror this architecture; do not invent a parallel one.

---

## 1. What we need to measure (and why)

Per-level cost in PGD breaks down like this. Names in the right column
become the canonical callback names used in `record(...)`.

| Region | File:lines (current code) | Callback name | Why measure it |
|---|---|---|---|
| FEM matrix assembly | `src/mesh/tri_mesh.py:54-123` | `matrix_assembly` | Python `for t in range(T)` plus `lil_matrix[i,j] +=`. Expected to dominate setup at ≥10⁵ triangles. |
| Triangle areas | `src/mesh/tri_mesh.py:36-51` | `triangle_areas` | Same Python loop; called by assembly and by `get_mesh_statistics`. |
| Provider mesh build | `relaxation.py:_setup_level` line `mesh = provider.build()` | `mesh_build` | For implicit surfaces (marching cubes) this can rival assembly. |
| Initial condition (level 0) | `relaxation.py:_create_initial_condition` (random branch) | `init_condition` | One-shot but includes a full projection. |
| Inter-level interpolation | `relaxation.py:_create_initial_condition` (interp branch) → `mesh/interpolation.py` | `init_interpolate` | Only on level > 0. |
| Energy evaluation | `pgd_optimizer.compute_energy` | `energy` | Called **inside backtracking**, every trial step. Per-partition `for i in range(n)` sparse matvecs. |
| Gradient evaluation | `pgd_optimizer.compute_gradient` | `gradient` | Called twice per accepted iteration (line search + `g_post`). |
| Constraint vector | `pgd_optimizer.constraint_fun` | `constraints` | Once per iteration. |
| Projection inside backtrack | `pgd_optimizer.optimize` line `A_trial = orthogonal_projection_iterative(...)` → `optimization/projection.py:64-144` | `projection` | Likely dominant per-iter cost. Up to 100 inner iters per call, multiple calls per major iter. |
| Whole backtracking loop | `pgd_optimizer.optimize:288-307` | `backtrack` | Wall around the whole `while True:` so we can separate "search overhead" from `energy`+`projection` components. |
| HDF5 iteration save | `pgd_optimizer._save_iteration_h5` + the components recompute at line 320 | `h5_save` | Includes the redundant `compute_energy(return_components=True)`. |
| HDF5 flush | `pgd_optimizer.optimize:323-324` (`summary_fh.flush()` + `h5f.flush()`) | `h5_flush` | Per-iter fsync; can dominate I/O on networked storage. |
| Trigger plateau checks | `pgd_optimizer.optimize:349-371` | `trigger_check` | Slices history every iter; usually small, but worth confirming. |

**Counters** (in addition to wall time):

| Counter | Where to increment | Why |
|---|---|---|
| `major_iterations` | once per major PGD iter | denominator for means |
| `backtracks_per_iter_total` | += backtracks observed in this iter's `while True:` | want `mean_backtracks` |
| `projection_inner_iters_total` | += inner-iter count reported by `orthogonal_projection_iterative` | want `mean_projection_inner_iters` |
| `projection_invocations` | += 1 per `orthogonal_projection_iterative` call | denominator for above |

These are **per-level** stats: PGD runs once per refinement level with a
different mesh. Aggregate over levels in the final YAML structure (see §3).

---

## 2. Plumbing — where the `profile` argument has to flow

Mirror Phase 2's `profile=` keyword-argument plumbing. Zero overhead when
omitted everywhere.

```
scripts/find_surface_partition.py        --profile flag
    └─> RelaxationConfig.profile (bool)
            └─> run_relaxation(provider, config, ...)
                    └─> creates RelaxationProfilingState (once, before the level loop)
                    └─> _setup_level(provider, config, level, logger, profile=prof)
                            └─> mesh.compute_matrices(_prof=prof)
                            └─> mesh.get_mesh_statistics(_prof=prof)      # optional
                    └─> _create_initial_condition(... profile=prof)
                            └─> orthogonal_projection_iterative(..., _prof=prof)
                            └─> nearest_neighbor_interpolate(..., _prof=prof)   # optional
                    └─> _optimize_level(optimizer, config, x0, level, ..., profile=prof)
                            └─> optimizer.optimize(..., profile=prof)
                                    └─> compute_energy/compute_gradient/constraint_fun wrapped inline
                                    └─> orthogonal_projection_iterative(..., _prof=prof)
                                    └─> _save_iteration_h5 wrapped inline
                    └─> after loop: prof.finalize_level(level), prof.finalize()
                    └─> _write_timing_profile(solution_dir, ...)
```

Use the leading-underscore `_prof=` kwarg name for "pass-through" call
sites that don't otherwise know about profiling — this matches the
existing pattern in `vectorized_steiner.compute_steiner_*(_prof=...)`.
For top-level public APIs (`run_relaxation`, `optimizer.optimize`) use
`profile=` to match Phase 2.

---

## 3. The `RelaxationProfilingState` class — design

Add to `src/profiling.py`. Keep Phase 2's `ProfilingState` untouched.

```python
@dataclass
class RelaxationProfilingState:
    """Per-level + aggregate timing accumulator for Phase 1 PGD.

    Same zero-overhead pattern as ProfilingState: every call site is
    guarded by `if prof is not None:`. Per-level data is held in a
    list of dicts so we can write a single timing_profile.yaml at the
    end of the run.
    """

    n_partitions: int

    # Mutable per-level state — flushed into _levels by finalize_level().
    _current_level: Optional[int] = None
    _current_n_vertices: int = 0
    _current_n_triangles: int = 0
    _current_nnz_K: int = 0
    _current_nnz_M: int = 0
    _current_epsilon: float = 0.0

    _wall_s: Dict[str, float] = field(default_factory=dict, repr=False)
    _count: Dict[str, int]   = field(default_factory=dict, repr=False)

    # Counter primitives (not tied to a callback)
    _counters: Dict[str, int] = field(default_factory=dict, repr=False)

    # Wall-clock for the whole level (from level start to PGD end)
    _level_start_t: float = 0.0
    _level_wall_s: float  = 0.0

    _levels: List[Dict] = field(default_factory=list, repr=False)

    # Set by run_relaxation after the loop ends
    total_wall_s: float = 0.0

    _finalized: bool = False
    _summary: Dict = field(default_factory=dict, repr=False)
```

Methods:

```python
def begin_level(self, level: int, n_vertices: int, n_triangles: int,
                nnz_K: int, nnz_M: int, epsilon: float) -> None:
    """Reset per-level accumulators and start the level wall clock."""

def record(self, cb: str, elapsed: float) -> None:
    """Wall-clock accumulator. Same semantics as ProfilingState.record."""

def add_counter(self, name: str, n: int = 1) -> None:
    """Bump a named integer counter."""

def finalize_level(self) -> None:
    """Snapshot _wall_s / _count / _counters into _levels[len], reset."""

def finalize(self) -> None:
    """Compute aggregate summary across levels."""

def to_yaml_dict(self, *, surface: str, n_partitions: int,
                 refinement_levels: int) -> dict:
    """Build the timing_profile.yaml payload."""

def to_index_dict(self) -> dict:
    """Flat scalars for experiment_index.yaml."""
```

### YAML schema (per-run, written to `<run_dir>/solution/timing_profile.yaml`)

```yaml
schema_version: "1.0"
phase: relaxation
timestamp: 2026-06-01 14:32:08
run_metadata:
  surface: torus
  n_partitions: 10
  refinement_levels: 2

summary:
  total_wall_s: 412.55
  matrix_assembly_pct_wall: 8.31
  energy_pct_wall: 22.14
  gradient_pct_wall: 18.02
  projection_pct_wall: 39.71
  backtrack_pct_wall: 5.10          # backtrack overhead minus projection+energy
  h5_save_pct_wall: 3.50
  h5_flush_pct_wall: 1.04
  trigger_check_pct_wall: 0.21
  overhead_pct_wall: 1.97
  mean_backtracks_per_iter: 1.42
  mean_projection_inner_iters: 23.7
  major_iterations_total: 1840

levels:
  - level: 0
    n_vertices: 64512
    n_triangles: 128768
    nnz_K: 901376
    nnz_M: 901376
    epsilon: 7.21e-03
    level_wall_s: 188.21
    major_iterations: 920
    mean_backtracks_per_iter: 1.28
    mean_projection_inner_iters: 22.4
    callbacks:
      matrix_assembly: {invocation_count: 1, total_wall_s: 12.30, mean_wall_s: 12.30}
      triangle_areas:  {invocation_count: 2, total_wall_s: 0.41,  mean_wall_s: 0.205}
      mesh_build:      {invocation_count: 1, total_wall_s: 1.10,  mean_wall_s: 1.10}
      init_condition:  {invocation_count: 1, total_wall_s: 4.91,  mean_wall_s: 4.91}
      energy:          {invocation_count: 2080, total_wall_s: 41.5, mean_wall_s: 0.0199}
      gradient:        {invocation_count: 1840, total_wall_s: 33.8, mean_wall_s: 0.0184}
      constraints:     {invocation_count: 920,  total_wall_s: 1.10,  mean_wall_s: 0.0012}
      projection:      {invocation_count: 2080, total_wall_s: 74.6, mean_wall_s: 0.0359}
      backtrack:       {invocation_count: 920,  total_wall_s: 9.6,  mean_wall_s: 0.0104}
      h5_save:         {invocation_count: 920,  total_wall_s: 6.6,  mean_wall_s: 0.0072}
      h5_flush:        {invocation_count: 920,  total_wall_s: 1.9,  mean_wall_s: 0.0021}
      trigger_check:   {invocation_count: 920,  total_wall_s: 0.4,  mean_wall_s: 0.0004}
  - level: 1
    n_vertices: ...
    ...
```

### Notes on the schema

- `backtrack` wall is the time spent **inside the `while True:` loop**
  including its `projection` and `energy` calls. `summary.backtrack_pct_wall`
  is reported as the *net* line-search overhead — i.e.
  `(backtrack_total - energy_total_inside_backtrack - projection_total)`
  divided by `total_wall_s`. If extracting the "inside-backtrack" share is
  awkward, drop the net subtraction and document the overlap clearly in
  the YAML (e.g. add `notes: "backtrack includes energy & projection wall"`).
- `triangle_areas` may be called twice per level (once from
  `compute_matrices` line 67-68 and once from `get_mesh_statistics`
  line 143-150) — fine, the counter handles that.
- `init_condition` is one of `init_condition` (level 0, random+project) or
  `init_interpolate` (level>0, NN interp + project). The projection time
  inside it is **separately attributed** to `projection` via the
  `_prof=prof` hook on `orthogonal_projection_iterative`. The wall for
  `init_condition` therefore overlaps with `projection`; document this
  explicitly the same way the backtrack overlap is documented.
- The "overhead_pct_wall" line in `summary` is whatever fraction of
  `total_wall_s` isn't accounted for by any tracked callback (same
  meaning as in Phase 2's `ProfilingState`).

---

## 4. Concrete instrumentation, file by file

This section is intended to be followed literally.

### 4.1 `src/profiling.py`

Add the `RelaxationProfilingState` dataclass described in §3. Do **not**
modify `ProfilingState`.

### 4.2 `src/mesh/tri_mesh.py`

```python
def compute_matrices(self, _prof=None) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if _prof is not None:
        t0 = time.perf_counter()
    # ... existing body unchanged ...
    if _prof is not None:
        _prof.record('matrix_assembly', time.perf_counter() - t0)
    return self.mass_matrix, self.stiffness_matrix
```

Same wrapper on `_compute_triangle_areas` with callback name
`triangle_areas`.

Touch points: `get_mesh_statistics` (lines 142-152) does not need a
wrapper itself — it calls `_compute_triangle_areas` (already wrapped)
and `self.M.sum()`. Leave it untouched but accept `_prof=None` is **not**
required there.

### 4.3 `src/optimization/projection.py`

```python
def orthogonal_projection_iterative(A, c, d, v, max_iter=1000, tol=1e-10,
                                    logger=None, _prof=None) -> np.ndarray:
    if _prof is not None:
        t0 = time.perf_counter()
        _prof.add_counter('projection_invocations', 1)

    # ... existing body unchanged through the for-loop ...

    # After the for/else block, before final validation:
    if _prof is not None:
        # iter is the loop variable; if we broke early it holds the
        # iteration on which we broke; if we exhausted, it holds max_iter-1.
        _prof.add_counter('projection_inner_iters_total', iter + 1)
        _prof.record('projection', time.perf_counter() - t0)

    return A
```

The variable `iter` already exists in the loop (line 64). Capture it
*outside* the for-loop scope (Python keeps the binding after `for/else`).

`orthogonal_projection_direct` and `create_initial_condition_with_projection`
should also accept `_prof=None` and forward it.

### 4.4 `src/mesh/interpolation.py`

Wrap `nearest_neighbor_interpolate` with callback name `init_interpolate`
(accept `_prof=None`).

### 4.5 `src/optimization/pgd_optimizer.py`

Add `profile: Optional[RelaxationProfilingState] = None` to the
`optimize(...)` signature (last kwarg). Inside the function:

- Right before the main `for k in range(maxiter):` loop, no-op if
  `profile is None`.
- Wrap `compute_gradient(x)` call sites (lines 286 and 310):
  ```python
  if profile is not None: t = time.perf_counter()
  g = self.compute_gradient(x)
  if profile is not None: profile.record('gradient', time.perf_counter() - t)
  ```
- Wrap the **whole backtracking `while True:`** with the `backtrack`
  callback timing (start `t_bt` before line 288, record at first
  `break` and at the `step < 1e-12` break).
  - Track `n_backtracks` as a local counter incremented on every
    failed trial; after the loop call
    `profile.add_counter('backtracks_per_iter_total', n_backtracks)` and
    `profile.add_counter('major_iterations', 1)`.
- Inside the backtracking loop, wrap each `compute_energy(x_trial)`
  call with `energy`, and the `orthogonal_projection_iterative(...)`
  call by passing `_prof=profile` (and **don't** double-time it).
- Wrap `constraint_fun(x)` at line 311 with `constraints`.
- Wrap `_save_iteration_h5(...)` (including the
  `compute_energy(x, return_components=True)` recompute at line 320)
  with `h5_save`.
- Wrap the two `flush()` calls together with `h5_flush`.
- Wrap the refinement-trigger block lines 350-371 with `trigger_check`.

`compute_energy` and `compute_gradient` themselves do **not** need
internal hooks — wrapping every call site is sufficient.

### 4.6 `src/pipeline/relaxation.py`

`RelaxationConfig`: add field

```python
profile: bool = False
```

at the bottom of the dataclass (line 75 area). It is read by
`from_yaml_dict` via the existing field-iteration logic without any
extra change.

`run_relaxation`:

```python
prof = None
if config.profile:
    from ..profiling import RelaxationProfilingState
    prof = RelaxationProfilingState(n_partitions=config.n_partitions)

t_run_start = time.time()
# ... existing setup ...
for level in range(start_level, config.refinement_levels):
    level_ctx = _setup_level(provider, config, level, logger, profile=prof)
    if prof is not None:
        mesh = level_ctx['mesh']
        prof.begin_level(
            level=level,
            n_vertices=int(mesh.vertices.shape[0]),
            n_triangles=int(mesh.faces.shape[0]),
            nnz_K=int(mesh.K.nnz),
            nnz_M=int(mesh.M.nnz),
            epsilon=float(level_ctx['epsilon']),
        )
    x0 = _create_initial_condition(mesh, config, level,
                                   prev_vertices, prev_x_opt, profile=prof)
    level_result = _optimize_level(level_ctx['optimizer'], config, x0, level,
                                   provider, traces_dir, logger, profile=prof)
    ...
    if prof is not None:
        prof.finalize_level()

# After the loop
if prof is not None:
    prof.total_wall_s = time.time() - t_run_start
    prof.finalize()
    _write_timing_profile(prof, solution_dir, provider, config, logger)
```

Add a private helper:

```python
def _write_timing_profile(prof, solution_dir, provider, config, logger):
    tp_path = os.path.join(solution_dir, 'timing_profile.yaml')
    data = prof.to_yaml_dict(
        surface=provider.surface_name(),
        n_partitions=config.n_partitions,
        refinement_levels=config.refinement_levels,
    )
    with open(tp_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Timing profile written to: {tp_path}")
```

Thread `profile=` through `_setup_level`, `_create_initial_condition`,
`_optimize_level`. They forward it to the sub-call(s) as described.

### 4.7 `scripts/find_surface_partition.py`

Add the CLI flag (place near `--resume-from`):

```python
parser.add_argument('--profile', action='store_true', default=False,
                    help='Enable Phase 1 timing profiling. Writes '
                         'solution/timing_profile.yaml with per-level '
                         'wall-clock breakdown by callback. '
                         'Zero overhead when omitted.')
```

After loading `config`:

```python
if args.profile:
    config.profile = True
```

(Mirror `refine_perimeter.py:86-87`.)

### 4.8 `sweep/parameter_sweep.py`

Extend `_extract_run_metrics` (around line 514) to also pick up the
Phase 1 timing file:

```python
def _extract_relaxation_timing(run_dir: str) -> Optional[dict]:
    """Return the parsed solution/timing_profile.yaml, or None."""
    p = os.path.join(run_dir, "solution", "timing_profile.yaml")
    if not os.path.isfile(p):
        return None
    try:
        return _load_yaml(p)
    except Exception:
        return None


def _flatten_relaxation_timing(tdata: dict) -> dict:
    flat: dict = {}
    s = tdata.get("summary", {})
    flat["relax_timing_total_wall_s"] = s.get("total_wall_s")
    for cb in ("matrix_assembly", "energy", "gradient", "projection",
               "backtrack", "h5_save", "h5_flush", "trigger_check"):
        key = f"{cb}_pct_wall"
        flat[f"relax_timing_{key}"] = s.get(key)
    flat["relax_timing_mean_backtracks"] = s.get("mean_backtracks_per_iter")
    flat["relax_timing_mean_projection_inner_iters"] = s.get(
        "mean_projection_inner_iters")
    flat["relax_timing_levels"] = tdata.get("levels", [])
    return flat
```

Then in `_extract_run_metrics`, after the existing Phase 2 `timing`
block:

```python
relax_timing = _extract_relaxation_timing(run_dir)
if relax_timing:
    entry.update(_flatten_relaxation_timing(relax_timing))
```

### 4.9 `sweep/timing_analyzer.py`

For an initial version, **do not** rewrite the existing Phase 2 plotting
code. Add one new function (and one new `argparse` flag `--phase`
defaulting to `refinement`):

```python
def plot_relaxation_breakdown_vs_nvertices(runs, out_dir):
    """Stacked-area plot: per-callback wall fraction vs n_vertices,
    using relax_timing_* fields and n_vertices read from the first
    level of relax_timing_levels."""
```

Leave a TODO comment pointing at this plan if a full scaling-figure
suite for Phase 1 is wanted later.

### 4.10 `CLAUDE.md`

After this lands, add a parallel paragraph to the existing Phase 2
profiling note:

> Phase 1 timing profile: `--profile` on
> `scripts/find_surface_partition.py` writes
> `<run_dir>/solution/timing_profile.yaml` with per-level wall-clock
> breakdown by callback. Zero overhead when omitted.

Update the CLI snippet block (the "Phase 1" section) to include the new
flag.

---

## 5. What NOT to do in this change

- Do **not** vectorise `compute_matrices`, `_compute_triangle_areas`, or
  `compute_energy`/`compute_gradient`. Those optimizations come *after*
  measurement.
- Do **not** change the projection algorithm, its `max_iter` default, or
  its `tol`.
- Do **not** change the per-iteration HDF5 flush cadence.
- Do **not** merge the redundant `compute_energy(x, return_components=True)`
  recompute at line 320 with the line-search evaluation. We *want* the
  profile to show that redundancy clearly.
- Do **not** rename `ProfilingState`. Keep Phase 2 imports intact.
- Do **not** wire profiling into `compute_initial_perimeter` (the
  post-Phase-1 contour-extraction step). It's a one-shot at the end of
  `run_relaxation`; if we ever care, instrument later.
- Do **not** add unit tests for the profile values themselves
  (`test_profile_*.py`). The project has no pytest harness today; the
  verification below is sufficient.

---

## 6. Verification

The change is verified end-to-end when **all** of the following hold on
a real run.

Run on the smallest existing torus config to keep wall time short:

```bash
python scripts/find_surface_partition.py \
    --config parameters/torus_10part.yaml --profile
```

1. **File written.** `results/run_*/solution/timing_profile.yaml` exists
   and parses as YAML.
2. **Per-level coverage.** The `levels:` list has exactly
   `refinement_levels` entries, each with non-zero
   `matrix_assembly` and `projection` callbacks.
3. **Accounting closes.** For each level,
   `sum(callbacks[*].total_wall_s)` should be within ~15 % of
   `level_wall_s` (the gap is "overhead" — pure Python between hooks).
   If it's wildly off, a hook is missing.
4. **Counters non-zero.** `mean_backtracks_per_iter ≥ 1.0` and
   `mean_projection_inner_iters ≥ 1.0` in `summary`.
5. **Zero-overhead check.** Run **without** `--profile` on the same
   config. The end-of-level "elapsed" wall-clock printed by
   `_optimize_level` must be within 2 % of the `--profile` run's
   `level_wall_s`. (If the overhead is bigger than that, a `time.perf_counter()`
   call is firing on the inner hot path even when `profile is None`.)
6. **Sweep collect picks it up.** Run
   `python sweep/parameter_sweep.py --sweep <some-spec> --mode collect`
   on a results tree that contains at least one `--profile` Phase 1
   run; the corresponding entry in `experiment_index.yaml` must have
   `relax_timing_total_wall_s` (and the other `relax_timing_*` scalars)
   populated.

---

## 7. What we expect the profile to show (so we know how to read it)

Stating this up front so the next agent knows when the data is plausible
vs. when something is wrong:

- For a coarse torus (~10⁴ vertices, ~2×10⁴ triangles), expect
  `matrix_assembly` to be a few percent and per-iter costs
  (`projection`, `energy`, `gradient`) to dominate.
- As the mesh grows toward 10⁵–10⁶ vertices, `matrix_assembly` will
  grow as Θ(T) Python work and may rival or exceed the entire PGD loop
  — that's the "matrix assembly is a Python loop with lil_matrix
  inserts" symptom and is one of the candidate hot spots.
- Within the PGD loop, expect `projection` to be the single biggest
  callback (it's an inner iterative loop called multiple times per
  major iter). If `projection` is **not** dominant on a fine mesh,
  the actual bottleneck has been mis-identified and the optimisation
  shortlist in the parent discussion needs revisiting before changes
  go in.
- `mean_backtracks_per_iter` close to 1.0 means the Armijo step is
  almost always accepted on the first try (typical); much above ~3
  suggests `step0` is too large or `backtrack_rho` too small, in which
  case the bottleneck is "we're recomputing energy and re-projecting
  needlessly" rather than the energy/projection routines themselves.
- `mean_projection_inner_iters` close to `pgd_projection_max_iter`
  (default 100) means the projection is hitting its iteration cap
  every call, in which case the right intervention is a warm start or
  a looser inner tol — *not* speeding up the per-inner-iter cost.

These rules of thumb belong in the follow-up "where to optimise"
decision, not in this plan's implementation. Listing them here so the
next person reading `timing_profile.yaml` can interpret it without
reverse-engineering the optimizer.

---

## 8. Follow-ups (out of scope for this plan)

After timing data exists, the candidate optimizations to triage —
ranked by expected payoff but not committed:

1. Vectorise `compute_matrices` and `_compute_triangle_areas` via
   batched `coo_matrix` arrays.
2. Replace the `for i in range(n_partitions)` sparse-matvec loop in
   `compute_energy` / `compute_gradient` with a single `K @ phi`
   (sparse-times-dense) producing N×n at once.
3. Reuse line-search components instead of recomputing
   `compute_energy(x, return_components=True)` at save time.
4. Warm-start `orthogonal_projection_iterative` from the previous
   accepted iterate's projection state, and/or replace the inner
   `A_prev = A.copy()` stagnation check with a residual-delta check.
5. Drop per-iter HDF5 flushes; flush every K iters or on close only.

Each gets a separate plan or PR with its own measurement-driven
justification.
