# Phase 1 Seeded (Voronoi) Initialization

**Status:** Not Started
**Branch:** to be created by the implementing agent (suggested:
`feat/seeded-initialization`), off `main`.
**Audience:** Fresh agent implementing the feature.
**Goal:** Add a **selectable** initial-condition strategy to Phase 1 — keep the
current uniform-random initialization as the default, and add a seeded
(Voronoi) initialization that gives every cell a contiguous winning region from
iteration 0. This is the root-cause fix for the dormant-cell failure documented
in `docs/reference/phase1_dormant_cell_argmax_issue.md`.
**Scope:** Initialization only. Do **not** change the energy, the constraint
projection algorithm, the PGD loop, or the multi-level/interpolation logic.

---

## Background

Phase 1 relaxes a Γ-convergence energy over per-vertex soft memberships
`u ∈ ℝ^(V×N)`, constrained to sum-to-one per vertex and equal integrated area
per cell. The discrete partition is read off by winner-take-all (argmax).

The current level-0 initialization
(`create_initial_condition_with_projection` in
`src/optimization/projection.py`) starts every cell as a near-uniform random
field (~1/N everywhere). The optimizer must then **break symmetry from
scratch**, and at higher `N` some cells lose this noisy "land grab" and end up
**dormant** — they keep their full integrated area as a diffuse low film
(peak density ≪ 0.5) but never win a vertex, so they vanish from the discrete
partition. A run requesting `N` cells silently yields `N − k` regions, and the
surviving cells are no longer equal-area (the lost area is absorbed unevenly).

Empirical evidence (see the "Resolution Sweep" section of
`docs/reference/phase1_dormant_cell_argmax_issue.md`): increasing the
coarsest-mesh resolution from 154 to 320 vertices/cell at fixed seed/λ did
**not** reliably remove dormancy (dead-cell count `2 → 1 → 1 → 2`), and the
dead-cell *index* changed every run — confirming the failure is a numerical
symmetry-break artifact driven by the random initialization, not a geometric
limit. Seeded initialization attacks that root cause directly: if every cell
is handed a contiguous region before optimization starts, there is no land-grab
lottery to lose, and the same energy barrier that traps dormant cells instead
*protects* each cell's seeded region.

A dormant-cell **detector already exists** (`detect_dormant_cells` in
`src/partition/find_contours.py`, surfaced by `run_relaxation`); this plan
provides the fix the detector points at. The detector is the acceptance oracle
for this work: a successful seeded run reports zero dead/weak cells.

---

## Goal & requirements

1. **Selectable from the parameter YAML.** A `relaxation.init_method` option
   with at least two values: `random` (current behavior, **default**) and
   `seeded` (the new Voronoi strategy). No behavior change for existing configs
   that omit the field.
2. **Reproducible.** Seeded init must be deterministic given `relaxation.seed`,
   so reruns reproduce and different seeds give different (but valid) layouts.
3. **Efficient** (scientific-computing standards): no dense `V × N` distance
   matrices; use `scipy.spatial.cKDTree` (scipy is already a core dependency)
   for nearest-seed lookup and an incremental farthest-point sweep for seed
   selection. Initialization cost must be negligible next to the PGD loop.
4. **Consistent with the codebase**: reuse the existing constraint projection
   (`orthogonal_projection_iterative`); follow existing module structure,
   logging (`get_logger`), and the `RelaxationConfig.from_yaml_dict` field
   convention; Black, 4-space, relative imports under `src/`.
5. **Level-0 only.** Finer levels already warm-start by interpolation
   (`nearest_neighbor_interpolate`) and must remain unchanged — they inherit
   the healthy 30-region topology automatically.

---

## Design — the configuration option

Add one field to `RelaxationConfig` (`src/pipeline/relaxation.py`):

```python
init_method: str = 'random'   # 'random' (default, current) | 'seeded'
```

`RelaxationConfig.from_yaml_dict` already iterates dataclass fields and coerces
by declared type, so a string field is picked up with no parser change. In the
YAML it appears under `relaxation:`:

```yaml
relaxation:
  # Initial condition for level 0:
  #   random  - uniform random densities then projected (default; legacy)
  #   seeded  - Voronoi seed regions (one contiguous winner per cell from
  #             iteration 0; avoids dormant cells at higher N)
  init_method: seeded
```

**Decision (recommended): keep the seeded strategy parameter-free** — fix the
seed-point selection to farthest-point sampling (justified below). Do **not**
add a second knob (e.g. `seed_selection`) unless a concrete need appears; the
two-value `init_method` is the whole user-facing surface the feature needs.
This keeps the option set minimal and matches the "random vs proposed" ask.

No CLI flag is required (the request is specifically a parameter-file option).
A `--init-method` passthrough in `scripts/find_surface_partition.py` is
optional and out of scope unless trivial.

---

## Design — the seeded initialization algorithm

Produce a level-0 density `A ∈ ℝ^(V×N)` where each cell dominates one
contiguous patch, then project onto the constraints. Steps:

1. **Choose N seed vertices by farthest-point sampling (FPS / greedy
   k-center).** Start from one vertex chosen with a `seed`-seeded RNG, then
   repeatedly add the vertex with maximum distance to the current seed set,
   maintaining a running per-vertex min-distance array. Euclidean distance in
   the embedded R³ coordinates (consistent with the existing Euclidean
   `nearest_neighbor_interpolate`). FPS gives well-spread seeds → roughly
   equal Voronoi areas → the area-equalizing projection barely perturbs the
   regions, so each cell stays a clear winner. Cost `O(N·V)`, vectorized inner
   step, no `N×V` matrix.

   ```python
   rng = np.random.default_rng(seed)
   first = int(rng.integers(V))
   seeds = [first]
   d2 = np.sum((verts - verts[first])**2, axis=1)      # (V,)
   for _ in range(n_partitions - 1):
       nxt = int(np.argmax(d2))
       seeds.append(nxt)
       d2 = np.minimum(d2, np.sum((verts - verts[nxt])**2, axis=1))
   ```

2. **Assign every vertex to its nearest seed** (the Voronoi label) with a
   KD-tree — `O(V log N)`:

   ```python
   from scipy.spatial import cKDTree
   labels = cKDTree(verts[seeds]).query(verts)[1]       # (V,) in [0, N)
   ```

3. **Build the one-hot density and project onto the constraints**, reusing the
   existing projection (no new projection code):

   ```python
   A = np.zeros((V, n_partitions))
   A[np.arange(V), labels] = 1.0
   c = np.ones(n_partitions)
   d = (np.sum(v) / n_partitions) * np.ones(n_partitions)   # equal-area targets
   A = orthogonal_projection_iterative(A, c, d, v, max_iter=..., tol=...)
   return A.flatten()
   ```

   The projection equalizes the (slightly unequal) Voronoi areas and enforces
   sum-to-one while preserving each cell's dominant region. Confirm it
   converges on a near-binary input during verification.

**Why Euclidean and not geodesic:** the embedded-R³ nearest-seed is the cheap,
standard choice and is already how inter-level interpolation works in this
codebase. Geodesic Voronoi is out of scope.

---

## Phase 1 — Configuration plumbing & dispatch
**Status:** Not Started

- `src/pipeline/relaxation.py`: add `init_method: str = 'random'` to
  `RelaxationConfig` (bottom of the dataclass). No `from_yaml_dict` change
  needed (verify it coerces the string field).
- `src/pipeline/relaxation.py`: in `_create_initial_condition`, the **level-0
  branch** (`prev_vertices is None`) currently always calls the random builder.
  Dispatch on `config.init_method`: `'seeded'` → new seeded builder;
  anything else → existing random path (keep `'random'` as the catch-all so an
  unrecognized value degrades safely, with a `logger.warning`). The
  level > 0 interpolation branch is unchanged.
- `parameters/torus_30part.yaml` (and the other `parameters/*.yaml`): add the
  documented `init_method:` option. Default the committed configs to whichever
  the maintainer prefers, but the dataclass default stays `random` for
  backward compatibility.

## Phase 2 — Seeded initialization implementation
**Status:** Not Started

- Add `create_seeded_initial_condition(mesh, n_partitions, v, seed, logger=None,
  max_iter=..., tol=...)` implementing the algorithm above. **Recommended
  location:** a new module `src/optimization/initialization.py` (groups
  initialization strategies; import the existing random builder there too if a
  single dispatcher is preferred). Keeping the existing
  `create_initial_condition_with_projection` where it is (in `projection.py`)
  is fine — only `_create_initial_condition` needs to know both.
- Reuse `orthogonal_projection_iterative` from `src/optimization/projection.py`
  for step 3; reuse the same `max_iter`/`tol` the random path uses
  (`config.pgd_projection_max_iter` is for PGD; the random init currently uses
  its own defaults — match that).
- Determinism via `np.random.default_rng(seed)` (preferred over the global
  `np.random.seed`); document that seeded layout depends on `seed`.
- Logging: one `logger.info` summarizing method, N seeds, and projection
  convergence; no per-vertex spam.

## Phase 3 — Verification & docs sync
**Status:** Not Started

Verification (no pytest harness — empirical, mirroring the repo's existing
diagnostics):

1. **Fixes the known-dormant case.** Run the configuration that reliably
   produced dormancy — `parameters/torus_30part.yaml`-style, torus `N=30`,
   `λ=2.1`, `seed=84172851`, coarse base mesh (e.g. `nt=60, np=46`) — with
   `init_method: seeded`. The `dormant_cells` block in `solution/metadata.yaml`
   must report **0 dead and 0 weak** (29808-vertex final mesh), i.e. 30/30
   effective regions, and no dormant warning in `logs/relaxation.log`.
2. **Equal-area restored.** Compute discrete (argmax-territory) areas of the 30
   cells (lumped-mass vertex areas, as in the reference doc's diagnostic); they
   should be ≈ equal (target = total/30), not the unequal spread seen on
   dormant runs.
3. **Regression baseline.** The same config with `init_method: random` (or
   omitted) must reproduce the prior dormant behavior — confirms the toggle and
   that the default path is untouched.
4. **Determinism.** Same `seed` + `seeded` → identical solution across two runs.
5. **Efficiency.** Log/inspect initialization wall time; it must be negligible
   vs the PGD loop (KD-tree + FPS on ~10⁴–10⁵ vertices is milliseconds).

Docs sync (standing rule):
- `docs/reference/phase1_dormant_cell_argmax_issue.md`: mark mitigation
  **(3) Seeded initial condition** as implemented, pointing to the new code,
  and record the verification outcome.
- `CLAUDE.md`: document the `init_method` option and the new module under the
  src layout / "Modifying the PGD" or initialization notes.
- This plan: advance the Status fields; a fully-implemented plan should be
  deleted or folded into the reference doc per the docs-sync rule.

---

## Efficiency notes (scientific-computing standards)

- **No `V × N` distance matrix.** FPS keeps a single `(V,)` running min-distance
  array; nearest-seed uses a KD-tree query — both well below the cost of one PGD
  iteration.
- **Vectorize** the one-hot build (`A[np.arange(V), labels] = 1.0`) and all
  distance math (no Python per-vertex loops).
- **Reuse**, do not reimplement, the constraint projection.
- Memory footprint of `A` is `V × N` — identical to the random path; no extra
  large allocations.

---

## Out of scope

- Energy, projection algorithm, PGD loop, or interpolation changes.
- Mitigation (4) "mid-run rescue" from the reference doc.
- Geodesic Voronoi; additional seed-selection strategies beyond farthest-point.
- A CLI flag (parameter-file option only).
- Changing the default (`init_method` stays `random` in the dataclass so
  existing runs/results are reproducible).

## Open decisions for the implementer

1. **Module placement** — new `src/optimization/initialization.py` (recommended)
   vs adding the seeded function beside the random one in `projection.py`.
2. **One-hot epsilon** — exact 0 for non-winners (then project) vs a small ε.
   Exact 0 is expected to be fine since the PGD loop re-clips to
   `[1e-8, 1-1e-8]` and re-projects at entry; verify projection convergence.
3. **Committed-config default** — whether to flip any `parameters/*.yaml` to
   `init_method: seeded` by default, or leave them `random` and let the user
   opt in. (Dataclass default must remain `random` regardless.)

## Related documents

- Reference: `docs/reference/phase1_dormant_cell_argmax_issue.md` (the problem,
  the resolution-sweep evidence, and the mitigation list this implements).
- Code: `src/pipeline/relaxation.py` (`RelaxationConfig`,
  `_create_initial_condition`), `src/optimization/projection.py`
  (`create_initial_condition_with_projection`, `orthogonal_projection_iterative`),
  `src/partition/find_contours.py` (`detect_dormant_cells` — the acceptance
  oracle), `src/mesh/interpolation.py` (Euclidean nearest-neighbor precedent).
