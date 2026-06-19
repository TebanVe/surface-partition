# Phase 1: Dormant Cells Lost at Argmax Classification

## Issue Summary

During Phase 1 Γ-convergence relaxation on the torus with `n_partitions=30`,
two cells satisfy the equal-area integral constraint exactly throughout the
entire run but never become the argmax winner at any vertex. After
winner-take-all classification in `ContourAnalyzer.compute_indicator_functions`,
those cells produce zero boundary triangles and effectively vanish from the
discrete partition. A run requesting `N` cells silently produces an
`(N − k)`-cell partition for `k ≥ 1`.

This is not a bug in the optimizer or in the constraint projection — both work
correctly. It is a structural gap between the **continuous-density**
representation that PGD optimizes and the **discrete argmax classification**
that downstream code uses to extract a partition. The resulting density
configuration is a valid local minimizer of the constrained Γ-convergence
energy that happens to be unusable as a discrete partition.

## Context

Phase 1 minimises the Γ-convergence energy

```
E(u) = ε · uᵀ K u + (1/ε) · (u²(1−u)²)ᵀ M (u²(1−u)²) + λ_penalty · P(u)
```

over a density field `u ∈ ℝ^(n_vertices × N)`, subject to

- **Sum-to-one:** `Σ_k u_k(x) = 1` at every vertex (partition).
- **Equal areas:** `∫ u_k dA = total_area / N` for every cell `k`.

The constraints are enforced exactly at every PGD step by
`src/optimization/projection.py`. The optimizer sees mass, not argmax.

After Phase 1, `ContourAnalyzer.compute_indicator_functions` performs
**winner-take-all** classification: at each vertex `i`, the cell with the
largest `u_k(x_i)` wins. Boundary triangles are then extracted between cells
with different labels.

If a cell is never the argmax at any vertex, it contributes no boundary
triangles, no variable points, and is absent from the post-processed
partition — even though it still carries its full integrated mass in the
density field.

## Observed Behavior

Run:
```
results/torus_npart30/
  run_20260529_182712_surftorus_npart30_v1nt80-204_incr62_v2np66-182_incr58_lam2.0_seed52698790/
```

Configuration: `N=30`, `λ_penalty=2.0`, seed `52698790`, 3 refinement levels
(80×66 → 142×124 → 204×182, final 37 128 vertices). `λ=2.0` was the best
result across a `λ ∈ {1.0, 2.0, …, 10.0}` sweep, so the issue is not
attributable to a poor `λ` choice.

Phase 1 area report (`logs/relaxation.log`) shows every cell holding the
target area `0.78876324` (level 1) and `0.78945113` (level 3) at every logged
iteration.

Phase 1 contour extraction at the end of level 3:
```
Region 13: extracted 0 contour segments from 0 boundary triangles
Region 26: extracted 0 contour segments from 0 boundary triangles
```

The resulting partition has 28 surviving cells and 56 triple points
(consistent with 28 cells, not 30).

## Diagnostic Investigation

The diagnostic below reads the Phase 1 solution file directly. No code
in `src/` has been modified.

```python
import h5py, numpy as np

SOL = ("results/torus_npart30/run_20260529_182712_.../solution/"
       "surface_part30_..._20260529_182712.h5")

with h5py.File(SOL, "r") as f:
    x = f["x_opt"][:].reshape(-1, 30)   # (n_vertices, N)
    V = f["vertices"][:]
    F = f["faces"][:]

# Lumped P1 mass: v_i = (1/3) * sum of triangle areas incident to vertex i.
tri = V[F]
tri_area = 0.5 * np.linalg.norm(
    np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
v = np.zeros(V.shape[0])
for j in range(3):
    np.add.at(v, F[:, j], tri_area / 3.0)

mass_k       = x.T @ v
labels       = x.argmax(axis=1)
wins         = np.bincount(labels, minlength=30)
max_per_cell = x.max(axis=0)
```

### Findings

| Quantity                | Alive cells (typical) | Dead cells (13 and 26) |
| ----------------------- | --------------------- | ---------------------- |
| argmax wins             | ≈ 1 100               | **0**                  |
| `∫ u_k dA`              | 0.789451 (target)     | **0.789451 (target)**  |
| `max(u_k)`              | ≈ 0.94                | **0.083**              |
| `median(u_k)`           | ≈ 10⁻⁹                | **0.033**              |
| vertices with `u_k>0.5` | ≈ 1 100               | **0**                  |

Mass conservation across the partition is exact:
`Σ_k mass_k = 23.683534 = total_area` (matches to six decimal places).

Interpretation:

- **Mass preserved exactly.** The projection step did its job at every
  iteration; the integral constraint is never violated.
- **Density is uniformly diffuse, not localised.** The dead cells sit at
  `u ≈ 0.033` on about 24 000 of the 37 128 vertices. That uniform-low field
  integrates to `0.033 × 24 000 × (mean v_i) ≈ 0.79`, matching the target.
- **They are not in the race.** Their maximum is more than an order of
  magnitude below the alive-cell maximum (~0.94). No region exists where
  cells 13 or 26 are even close to winning.
- **Alive cells are essentially binary.** Median `≈ 10⁻⁹`, max `≈ 0.94`. The
  double-well term `u²(1−u)²` is satisfied: alive cells sit near 0 or near 1.

## Why This Is a Valid Local Minimum

The PGD iterate has no descent direction that would resurrect a dead cell:

1. The double-well term penalises any density value away from `{0, 1}`. A
   dead cell sits near the `u=0` well almost everywhere; raising it locally
   moves into the penalty band and *increases* energy.
2. Sum-to-one means raising `u_13` somewhere requires lowering some other
   `u_j` there. If that `u_j` is currently the argmax winner near 1, pulling
   it down moves it into the same penalty band.
3. The equal-area constraint is already satisfied by the diffuse-uniform
   state. The projection has no work to do.

The optimizer cannot see the discrete argmax structure — nothing in either
the energy or the constraints references it. The 28-cell argmax outcome is
therefore a feasible local minimiser of the continuous problem that the
optimizer correctly converges to.

This also explains the `λ_penalty` sweep result. At any `λ`:

- low `λ` → mushy densities, more cells stay diffuse;
- high `λ` → aggressive binarisation toward `{0, 1}`, dormant cells get
  pushed to 0 faster and never recover.

There is no `λ` that *forces* a dormant cell to acquire an argmax-winning
region, because no term in the energy rewards doing so.

## Detection From an Existing Solution

The condition can be detected directly from `x_opt`. Two equivalent signals:

```python
labels = x.argmax(axis=1)
wins   = np.bincount(labels, minlength=N)
dead   = np.where(wins == 0)[0]               # signal 1

max_per_cell = x.max(axis=0)
dormant      = np.where(max_per_cell < 0.5)[0]  # signal 2
```

Both flagged the same two cells (13, 26) here. A reasonable threshold on
`max(u_k)` is anywhere in `[0.3, 0.5]`: alive cells in this run sit at
`max ≥ 0.92`; the dead cells sit at `max ≈ 0.08`. The gap is large.

No such check currently exists in `ContourAnalyzer` or in the Phase 1
pipeline — a Phase 1 run completing with dormant cells is reported as
`success` and a partition file is written.

## Contributing Factors

The two-cell collapse on this specific run is shaped by:

- **Coarse level-1 mesh.** `nt=80, np=66` gives 5 280 vertices over
  `N=30` cells (≈ 176 vertices per cell). For comparison, the best
  `npart10` runs had ≈ 400 vertices per cell at level 1. Less per-cell
  capacity at the symmetry-breaking stage means more cells can fail to
  acquire a winning region before competition closes the door.

- **Uniform-mass initial condition.** `create_initial_condition` produces
  every cell with the same equal-mass profile. The early symmetry-break is
  noise-driven, and at `N=30` the probability that at least one cell loses
  the noise lottery is non-trivial.

- **Nearest-neighbour inter-level interpolation.** A cell that is
  diffuse-low at the end of level 1 is diffuse-low at the start of level 2.
  The level-2 projection redistributes mass but never relocates a cell's
  centre of presence. The dormancy is locked in by the time level 2 begins.

None of these factors *causes* the failure on its own — the underlying
issue is the argmax/density gap. They determine the probability that a
given seed and configuration ends up in a dormant-cell minimum.

## Empirical: Resolution Sweep (does more mesh fix it?)

To test mitigation (2) directly, a sweep was run at fixed `λ_penalty = 2.1`
and fixed seed `84172851`, varying only the base (level-1) mesh resolution
over four values (3 refinement levels each). Per-cell vertex budget at
level 1 ranged from 154 to 320 (still below the ≥ 400 rule-of-thumb, but
roughly doubling across the sweep). Dead cells were taken from the
`dormant_cells` metadata / `relaxation.log` warning; survivor area spread is
`(max − min) / mean` over the discrete (argmax-territory) areas of the
non-empty cells.

| base `nt × np` | level-1 vertices/cell | dead cells   | effective | survivor area spread |
| -------------- | --------------------- | ------------ | --------- | -------------------- |
| 70 × 66        | 154                   | `[5, 26]`    | 28/30     | 4.6 %                |
| 80 × 76        | 202                   | `[1]`        | 29/30     | 5.3 %                |
| 90 × 86        | 258                   | `[18]`       | 29/30     | 2.3 %                |
| 100 × 96       | 320                   | `[15, 27]`   | 28/30     | 3.4 %                |

Findings:

- **Dormancy persisted at every resolution.** The dead-cell count went
  `2 → 1 → 1 → 2` — not monotonically decreasing, and the *highest*-resolution
  run still lost two cells. Doubling the per-cell vertex budget (154 → 320)
  did **not** reliably reduce dormancy.
- **Which cell dies is random.** The dead indices change every run
  (`[5,26] → [1] → [18] → [15,27]`) even though the seed is fixed: each mesh
  has a different vertex count, so the uniform random init differs, the
  symmetry-break plays out differently, and a different cell loses. The
  failure is **not** tied to a fixed geometric location on the torus.
- **Survivor areas are unequal but mildly so here (2–5 %)** — milder than the
  15 % seen on the earlier `lam2.1_seed84172851` run that also had a
  degenerate speck cell, because with only 1–2 clean kills the lost area
  budget is reabsorbed more evenly. Still not an equal-area `N`-cell
  partition.

Caveats: the sweep stayed below the ≥ 400 vertices/cell threshold, and four
runs is a small sample. But the non-monotonic persistence across a 2×
resolution increase is strong evidence that, in this range and for this
seed/`λ`, the **random initialization dominates the outcome** — i.e.
resolution is a weak, unreliable lever, and the symmetry-break origin
(mitigation 3) is the one to attack.

## Possible Mitigations

These are options, not recommendations. Choice between them is open.

1. **Detection-only safeguard.** ✅ **Implemented.**
   `detect_dormant_cells()` in `src/partition/find_contours.py` classifies
   each cell as *dead* (zero argmax wins) or *weak* (peak density below the
   `WEAK_CELL_DENSITY_THRESHOLD = 0.5` argmax/contour level).
   `run_relaxation` calls it on the final solution and, via
   `_warn_if_dormant_cells`, logs a prominent warning (to console and
   `logs/relaxation.log`) pointing the user at the initial mesh resolution;
   `scripts/find_surface_partition.py` also prints a screen banner. The full
   result is persisted as the `dormant_cells` block in
   `solution/metadata.yaml`. This makes the issue visible and prevents silent
   N→N−k partitions but does not fix the optimizer outcome — it remains
   orthogonal to (2)–(4) below, which address the root cause.

2. **Higher-resolution level 1.** Increase `nt` and `np` so each cell has
   enough vertices for early competition. Rule of thumb: ≥ 400 vertices per
   cell on level 1 (`nt × np / N ≥ 400`). **Empirically weak (see the
   Resolution Sweep section).** Doubling the per-cell budget from 154 to 320
   at fixed seed/`λ` did not reliably reduce dormancy (dead count
   `2 → 1 → 1 → 2`). The sweep did not cross the 400/cell threshold, so much
   higher resolution *might* still help, but it is expensive and the trend
   gives no promise of converging to zero. Treat as, at best, a partial lever.

3. **Seeded initial condition.** ✅ **Implemented.**
   `create_seeded_initial_condition()` in `src/optimization/initialization.py`
   replaces the uniform level-0 initialisation with `N` Voronoi seed regions:
   `N` well-spread seed vertices are chosen by farthest-point sampling
   (incremental running min-distance array, Euclidean R³, deterministic via
   `np.random.default_rng(seed)`), every vertex is labelled by its nearest
   seed (`scipy.spatial.cKDTree`), and the one-hot density is projected with
   the existing `orthogonal_projection_iterative` to enforce sum-to-one and
   equal areas. This gives every cell a contiguous winning region from
   iteration 0, **independent of mesh resolution and seed**. It is selected
   per run by `relaxation.init_method: seeded` (default `random`); the level-0
   dispatch lives in `_create_initial_condition` (`src/pipeline/relaxation.py`),
   and finer levels still warm-start by interpolation unchanged.

   **Verification** (torus, `N=30`, `λ=2.1`, `seed=84172851`, coarse base mesh
   `nt=60, np=46`, 3 refinement levels, capped `max_iter`):
   - `init_method: random` → **dead [15, 19]**, weak [15, 19, 22],
     28/30 effective — reproduces the dormant failure.
   - `init_method: seeded` → **0 dead, 0 weak, 30/30 effective** on the same
     29 808-vertex final mesh, every cell's peak density = 1.000. The discrete
     (argmax-territory) lumped-mass cell areas are near-equal — min 0.7742,
     max 0.8020, mean 0.7894 (target total/30 = 0.7894), spread ≈ 3.5% —
     versus the random run's degenerate spread of ≈ 124% (two cells at 0
     territory). The continuous equal-area constraint itself holds to machine
     precision throughout (post-projection column-area spread ≈ 1e-14).
   - **Deterministic:** same `seed` + `seeded` reproduces an identical solution
     across reruns.
   - **Cost negligible:** the seeded build (FPS + KD-tree + projection) on the
     level-0 mesh (2 760 vertices) runs in ≈ 23 ms — far below the PGD loop.

   This addresses the symmetry-break problem at its source and is the lever the
   Resolution Sweep evidence (random init dominates the outcome) pointed to.

4. **Mid-run rescue.** Detect dormant cells between refinement levels and
   reseed them — split the densest cell, transfer mass, re-project. This
   keeps the cheap initialisation but adds recovery. Most complex of the
   three; requires careful interaction with the level-transition logic.

(1) has been implemented (it converts a silent failure into a loud one) and is
orthogonal to (2)–(4). **(3) is now also implemented and is the root-cause fix:
seeded initialisation produces valid N-region partitions at higher N** (verified
above on the previously-dormant `N=30` config). (2) remains a partial lever and
(4) is unimplemented.

## Open Questions

- **Is the failure geometric or numerical?** **Largely answered: numerical
  (symmetry-break), not geometric.** In the Resolution Sweep (fixed seed/`λ`,
  four meshes) the dead-cell *index* changed every run
  (`[5,26] → [1] → [18] → [15,27]`), so the failure is not tied to a fixed
  geometric small-cell limit on the torus; it is driven by the random
  initialization. This is the evidence base for prioritising mitigation (3)
  over (2). (A multi-seed scan would further quantify the per-config
  probability, but the location-independence is already established.)

- **Does the same pattern appear at intermediate `N`?** A scan over
  `N ∈ {12, 16, 20, 24, 30}` on a fixed level-1 mesh would identify the
  per-cell vertex budget below which dormancy starts appearing.

- **Does Phase 2 ever resurrect dormant cells?** It cannot — Phase 2 takes
  the discrete partition as input and only adjusts contour positions. Any
  cell missing from Phase 1's argmax output is gone for the rest of the
  pipeline.

## References

- `src/optimization/pgd_optimizer.py` — PGD with constrained Γ-convergence
  energy.
- `src/optimization/projection.py` — exact projection onto the sum-to-one
  and equal-area constraints.
- `src/partition/find_contours.py:compute_indicator_functions` — winner-
  take-all classification.
- Bogosel & Oudet, *Partitions of Minimal Length on Manifolds* — original
  method.
