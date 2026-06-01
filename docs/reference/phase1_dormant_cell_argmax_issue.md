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

## Possible Mitigations

These are options, not recommendations. Choice between them is open.

1. **Detection-only safeguard.** Add a `max(u_k) < threshold` check inside
   `ContourAnalyzer.compute_indicator_functions` (or at the end of
   `run_relaxation`). Surface the result as a warning, a hard error
   (`--strict`), or a metadata field (`dormant_cells: [13, 26]`). This
   makes the issue visible and prevents silent N→N−k partitions but does
   not fix the optimizer outcome.

2. **Higher-resolution level 1.** Increase `nt` and `np` so each cell has
   enough vertices for early competition. As a rule of thumb, target
   ≥ 400 vertices per cell on level 1 (`nt × np / N ≥ 400`). For `N=30`,
   that suggests `nt × np ≥ 12 000`, e.g. `nt=128, np=96`. Cheap on the
   coarse level; addresses the root cause for this specific configuration.

3. **Seeded initial condition.** Replace the uniform initialisation with
   one that places `N` distinct seed regions (e.g. `N` Voronoi cells around
   randomly chosen vertex centres, projected to satisfy the constraints).
   This gives every cell a winning region from iteration 0. More invasive
   than (2) but addresses the symmetry-break problem directly.

4. **Mid-run rescue.** Detect dormant cells between refinement levels and
   reseed them — split the densest cell, transfer mass, re-project. This
   keeps the cheap initialisation but adds recovery. Most complex of the
   three; requires careful interaction with the level-transition logic.

(1) is orthogonal to (2)–(4) and probably worth doing regardless of which
of the others is chosen, since it converts a silent failure into a loud one.

## Open Questions

- **Is the failure geometric or numerical?** The other two `npart30` seeds
  (`seed52698790` analysed here; `seed13131313`, `seed55369783`) have not
  been inspected. If all three lose exactly two cells in similar locations,
  the torus at `N=30` has a true small-cell limit at this resolution. If
  the count varies with seed, the failure is purely a numerical
  symmetry-break problem solvable by (2) or (3).

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
