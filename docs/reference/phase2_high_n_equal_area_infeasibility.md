# Phase 2: High-N Equal-Area Infeasibility (Diffuse "Runt" Cells)

At high partition counts (observed on the torus at `N=100`), Phase 2 perimeter
refinement **increases** the perimeter every iteration instead of decreasing it,
then crashes with IPOPT reporting *"Algorithm converged to a point of local
infeasibility."* The cause is not in Phase 2. It is that Phase 1 hands Phase 2 a
partition whose **discrete (winner-take-all) cell areas are grossly unequal**,
even though Phase 1's **continuous** equal-area constraint is satisfied to machine
precision. This makes the Phase 2 equal-area constraint infeasible at the starting
point; IPOPT (feasibility-first) spends the whole run fighting the area imbalance,
and dragging boundaries to grow a badly-undersized cell *costs* perimeter. This
document records the failure, the diagnostic evidence, why a finer mesh made it
worse, and what does and does not help.

This is the **not-dead-but-diffuse** cousin of
`docs/reference/phase1_dormant_cell_argmax_issue.md`. There the offending cells are
*dead* (zero argmax wins). Here every cell is alive (peak density = 1.0), but one
or more cells are **runts**: most of their density mass sits in a diffuse skirt
outside their argmax territory, so their winner-take-all area is a fraction of
their (correct) continuous area. Both failures are the same underlying
**continuous-density vs. discrete-argmax gap**; this one bites Phase 2's
equal-area constraint rather than the cell count.

## Context

Phase 1 minimises the Γ-convergence energy (see `pgd_optimizer.py`)

```
E(u) = ε · uᵀ K u + (1/ε) · (u²(1−u)²)ᵀ M (u²(1−u)²) + λ_penalty · P(u)
```

subject to **sum-to-one** (`Σ_k u_k(x) = 1` at every vertex) and **equal areas**
(`∫ u_k dA = total_area / N` for every cell `k`). Both constraints are enforced
exactly at every PGD step by `src/optimization/projection.py`. The interface width
is set by the mesh: `ε = √(mean_triangle_area) ≈ h` (`src/pipeline/relaxation.py`,
`_setup_level`). This is the paper's recipe (Bogosel & Oudet): a Modica-Mortola-type
diffuse-interface relaxation with `ε ∝ h`, the partition recovered by **winner-take-all**
thresholding of the densities, under an equal-area constraint. The Γ-convergence
guarantee is asymptotic (`ε → 0`); at finite `ε` the recovered areas differ from
the constrained continuous areas, and that gap grows as cells shrink toward `ε`.

Phase 2 (`PerimeterOptimizer` + IPOPT) then minimises total perimeter over the
variable-point positions subject to a **discrete equal-area constraint**: each
cell's FEM area must equal `total_area / N`. Its starting point is the
winner-take-all partition from Phase 1. **The "max constraint violation" IPOPT
prints at iteration 0 is exactly the worst cell's absolute area deviation from
target in the Phase 1 solution.**

## Observed Behaviour

Two `N=100` torus runs (`λ_penalty=2.1`, seed `84172851`), compared against the
clean `N=50` run at the same `λ`/seed:

| | **N=50 (works)** | **N=100 coarse** | **N=100 finer** |
| --- | --- | --- | --- |
| run | `run_20260625_113015_...npart50...` | `run_20260629_141012_...npart100...` | `run_20260701_143238_...npart100...` |
| finest mesh (V) | 348×328 (114 144) | 348×328 (114 144) | 494×464 (229 216) |
| final ε | 0.010186 | 0.010186 | **0.007188** |
| worst-cell area dev (of target 0.2369/0.4737) | **0.76 %** (0.0036) | 22.5 % (0.0532) | **67.3 %** (0.1595) |
| area spread (std/target) | 0.30 % | 2.38 % | 6.84 % |
| Phase 2 start constraint viol | 3.60e-3 | 5.32e-2 | 1.60e-1 |
| Phase 2 perimeter | 152.4 → **130.2 (−14.5 %)** | 216.7 → 228.5 (**rising**) | 215.9 → 226.1 (**rising**) |
| constraint viol at end | ~1e-11 (**feasible**) | stuck ~2.6–3.8 % | stuck ~14.7–15.3 % |
| Type-1 vertex collapses / iter | hundreds (362, 320, …) | 0–4 | 0–3 |
| outcome | converges cleanly | infeasibility crash (it 6) | infeasibility crash (it 9) |

Two controlled comparisons make the mechanism unambiguous:

- **N=50 vs N=100-coarse ran on the identical finest mesh** (348×328,
  ε = 0.010186). The *only* difference is the number of phases. Doubling N on the
  same mesh took the worst-cell area deviation from 0.76 % to 22.5 % and flipped
  Phase 2 from a clean −14.5 % perimeter drop to a rising-perimeter crash. **The
  driver is N, not the mesh.**
- **The finer mesh has a *smaller* ε** (0.00719 vs 0.01019 — a *sharper*
  interface) yet a *worse* runt cell (67.3 % vs 22.5 %). So the finer-is-worse
  result is **not** an interface-width effect; see "Why the finer mesh made it
  worse" below.

## Diagnostic Investigation

The discrete cell areas are the winner-take-all lumped-mass areas of the Phase 1
solution. This reproduces the Phase 2 start-of-run constraint violation exactly:

```python
import h5py, numpy as np, sys; sys.path.insert(0, '.')
from src.mesh.tri_mesh import TriMesh

with h5py.File(SOLUTION_H5, 'r') as f:
    V = f['vertices'][:]; F = f['faces'][:].astype(np.int64)
    x = f['x_opt'][:]; N = int(f.attrs['n_partitions'])

u = x.reshape(V.shape[0], N)          # (n_vertices, N)
v = np.asarray(TriMesh(V, F).v).ravel()   # lumped P1 mass per vertex
win = np.argmax(u, axis=1)            # winner-take-all label per vertex
areas = np.zeros(N); np.add.at(areas, win, v)
target = v.sum() / N
print('worst abs dev =', np.abs(areas - target).max())   # == IPOPT iter-0 constr_viol
```

Results: worst abs dev = 0.003599 (N=50), 0.053189 (N=100 coarse), 0.159512
(N=100 finer) — matching the Phase 2 logs' `Max constraint violation: 3.60e-03 /
5.32e-02 / 1.60e-01` to the digit.

### The runt cell (finer run, cell 92)

| Quantity | Value |
| --- | --- |
| continuous mass `∫ u₉₂ dA` | 0.23686 = **target exactly** |
| winner-take-all area | 0.07735 = **32.7 % of its own mass** |
| vertices won (argmax) | 689 |
| peak density `max(u₉₂)` | **1.000** (not dead, not weak) |
| mean argmax-density inside its patch | 0.827 |

Cell 92 holds a full target area's worth of continuous mass, exactly as the
equal-area projection requires — but two-thirds of that mass is spread thinly as
*second-place* density across neighbouring cells' territory. Its compact,
argmax-winning core is only a third of target. It is a perfectly valid continuous
minimiser that is a broken discrete partition cell. The imbalance is not confined
to this one cell either: the finer run has 8 cells >2 % off target (std/target
6.84 %), versus 3 cells and 2.38 % for the coarse run and essentially none for
N=50.

## Why Phase 2 Increases the Perimeter

IPOPT is a feasibility-first interior-point method. From a grossly infeasible
start it prioritises reducing the constraint violation (equalising areas) over the
objective (perimeter). Growing a cell that is at one-third of target area means
pushing its boundaries a long way outward against its neighbours, which
**increases** total boundary length. So the perimeter climbs monotonically while
the violation crawls down microscopically (finer run: 0.160 → 0.147 over nine
iterations — barely moving). Because the imbalance is so large and one cell is so
pinched, feasibility is never reached and IPOPT terminates at a "point of local
infeasibility." The migration loop keeps re-triggering, perimeter keeps climbing,
and the run crashes. Contrast N=50, which starts nearly feasible (0.76 %), reaches
feasibility (~1e-11) immediately, and is then free to actually minimise perimeter
(−14.5 %), firing hundreds of Type-1 collapses per iteration.

## Why the Finer Mesh Made It Worse

`parameters/torus_100part.yaml` (HISTORY note) attributes the coarse-mesh failure
to "too few vertices per cell to move boundaries finely enough to equalise 100
areas" and prescribes a √2 mesh scale-up as the fix. **The data falsifies this.**
The finer mesh started ~3× *more* infeasible (0.160 vs 0.053) and crashed the same
way. The bottleneck is not Phase 2's boundary resolution — it is Phase 1's
partition geometry.

The finer-is-worse result is a **non-convexity / local-minimum-selection** effect,
not a smooth mesh trend:

- ε is *smaller* on the finer mesh (0.00719 < 0.01019), so the interface is
  *sharper* relative to cell size. On interface-width grounds alone the finer mesh
  should give a *smaller* discrete-area gap, not a larger one.
- Phase 1 is non-convex. The seeded initial condition lands on different vertices
  on different meshes, and the PGD descent converges to a different local minimum.
  The finer run simply converged to a worse configuration in which one seed region
  got pinched into a runt.

So at fixed N in this regime, the outcome is **dominated by which local minimum
Phase 1 finds** (seed- and mesh-dependent), not by mesh resolution. This is why
throwing more mesh at the problem is unreliable — the same lesson the resolution
sweep in `phase1_dormant_cell_argmax_issue.md` reached for the dormant-cell
variant.

## The Role of ε and λ_penalty

Two knobs govern how crisp/compact each cell is, and therefore how small the
discrete-area gap is:

- **ε (interface width, = √mean_triangle_area).** The diffuse skirt has width
  ~ε. As N grows, the characteristic cell size `√(total_area / N)` shrinks toward
  ε, so the skirt eats a larger fraction of each cell. This is *the* reason the gap
  grows with N (N=50 vs N=100 on the identical mesh above). ε is tied to the mesh
  and is **not** independently tunable in the current pipeline without changing the
  `ε ∝ h` convention.

- **λ_penalty (crispness reward).** The penalty term is
  `λ · (1 − Var_w[u_k] / (μ_k(1−μ_k)))`, minimised when each cell's weighted
  variance reaches `μ(1−μ)` — the variance of a *sharp* indicator with the same
  mean. It directly rewards binarisation and penalises diffuse skirts, so it is the
  lever that most directly counteracts runt cells. **It was carried over unchanged
  from N=30/N=50 (`λ=2.1`) and never re-tuned for N=100** (the config admits this).
  Because cells are smaller at higher N while ε is fixed by the mesh, a fixed λ
  exerts relatively weaker sharpening pressure per cell — so λ plausibly needs to
  scale *up* with N. Caveat: λ is an add-on that biases the energy away from the
  true perimeter functional, so an over-large λ trades Γ-convergence fidelity for
  crispness. It is a knob to sweep, not a guaranteed win. (Note also the opposite
  failure documented for the dormant-cell issue: too-high λ can push a weak cell to
  zero *faster*. λ helps runts and hurts near-dormant cells — the safe operating
  band may narrow as N grows.)

## Relation to the Paper

The code faithfully implements Bogosel & Oudet's recipe: `ε ∝ h`,
winner-take-all recovery of the partition from the relaxed densities, and an
equal-area constraint. The Γ-convergence result is asymptotic in ε, and the
paper's demonstrations are at modest phase counts. Pushing to N=50–100+ on a torus
is beyond the validated regime, and the small-cell-relative-to-interface-width
regime — where winner-take-all areas depart materially from the constrained
continuous areas — is the expected trouble spot. Nothing here indicates a coding
error; it is a resolution limit of the diffuse-interface relaxation exposed at high
N. (Paper specifics on maximal N were not independently verified for this note.)

## Detection From an Existing Solution

Check *before* committing to a Phase 2 run (or a multi-day finer Phase 1). Reuse
the diagnostic above and flag on the winner-take-all area spread:

```python
worst_rel = np.abs(areas - target).max() / target
# N=50 clean run: 0.008.  Broken N=100 runs: 0.22 (coarse), 0.67 (finer).
if worst_rel > 0.05:
    print(f'WARNING: worst discrete cell area {worst_rel:.0%} off target — '
          f'Phase 2 equal-area constraint will start infeasible.')
```

This gate is now implemented as `detect_area_imbalance()`
(`src/partition/find_contours.py`, threshold `AREA_IMBALANCE_REL_THRESHOLD = 0.05`).
`run_relaxation` calls it on the final solution, logs a prominent warning + prints
a CLI banner when any cell exceeds threshold, and persists the result as the
`area_imbalance` block in `solution/metadata.yaml` (parallel to `dormant_cells`).
It is deliberately separate from `detect_dormant_cells()`, which catches
*dead*/*weak* cells but **not** runts: cell 92 has peak density 1.0 and wins 689
vertices, so it passes the dormant check while being a third of target area.
Verified against the three runs above — it fires on both N=100 solutions
(worst_abs_dev 0.160 / 0.053, matching their Phase 2 iter-0 constraint violations)
and stays silent on N=50 (0.0036, 0.8% < 5%).

## Possible Mitigations

Options, not recommendations; ordered cheapest-first.

1. **Try other seeds at N=100.** The failure is one (or a few) pathological cells
   and is seed/mesh-dependent (non-convex landscape). A different seed may give
   a clean partition, exactly as N=50 did. Cheapest possible test; tells you whether
   this is fundamental or bad luck. Prefer the *coarse* (N=50-per-cell) mesh — the
   finer mesh is proven not to help and is ~2× the cost.
2. **Discrete-area quality gate at end of Phase 1.** ✅ **Implemented** as
   `detect_area_imbalance()` (see "Detection" above): a bad partition is now flagged
   (log + CLI banner + `area_imbalance` metadata block) before Phase 2 rather than
   after a doomed multi-day run. Converts a silent failure into a loud one; does not
   fix the optimizer outcome. (Directly analogous to mitigation (1) in the
   dormant-cell doc.)
3. **Re-tune / scale λ_penalty with N.** Sweep λ at N=100 (coarse mesh, a couple
   of seeds) to push cells crisper and shrink the diffuse skirt. May reduce the gap;
   watch for over-binarisation and for the dormant-cell tradeoff.
4. **Phase-2 area homotopy / feasibility restoration.** Restore feasibility
   (equalise areas) *before* turning on the perimeter objective, or ramp the area
   target from the (unequal) starting areas to equal over a homotopy, so the
   optimizer is never asked to fix a large imbalance and minimise perimeter at once.
   Does not address the root cause but may make Phase 2 survivable on an imperfect
   Phase 1 partition. Note: when a cell is at one-third target, *no* amount of
   boundary motion equalises it without a large perimeter cost — the geometry is
   wrong coming in — so this is a band-aid, not a cure.
5. **Equalise discrete areas in Phase 1 (root-cause fix).** Sharpen diffuse cells
   or rebalance winner-take-all areas at the end of Phase 1 (e.g. an argmax-aware
   post-projection, or an energy/penalty term that references discrete territory).
   Most involved; the correct long-term direction for N≫50.

**Making the Phase 2 mesh finer is the wrong lever** and is empirically
counter-productive here.

## Open Questions

- **Where does the gap become fatal?** A waypoint run at **N=75** (coarse mesh,
  same λ/seed) would map the onset between the clean N=50 (0.76 %) and the broken
  N=100 (22–67 %). If N=75 lands at a few percent, Phase 2 may limp through, and it
  pinpoints the per-cell budget / N at which the method breaks. It is a diagnostic
  waypoint, not a fix for N=100. Cheaper than either N=100 run.
- **Does λ scaling recover N=100?** Untested. A λ × seed sweep at N=100 on the
  coarse mesh would answer both this and mitigation (1) at once.
- **Is a runt cell ever recoverable in Phase 2?** No. Phase 2 takes the discrete
  partition as input and only moves contours; it cannot un-pinch a cell that Phase 1
  delivered at a third of target area.

## Related Documents

- `docs/reference/phase1_dormant_cell_argmax_issue.md` — the dead/weak-cell variant
  of the same continuous-vs-discrete gap (this doc is its runt-cell sibling).
- `docs/plans/PHASE1_N1000_SCALING_PLAN.md` — high-N scaling effort; its premise
  that finer meshes carry the method to high N should be reconciled with this note.
- `parameters/torus_100part.yaml` — HISTORY note whose √2-mesh-scale-up hypothesis
  this note falsifies.
- `src/optimization/pgd_optimizer.py` — Phase 1 energy, `λ_penalty` variance
  penalty. `src/pipeline/relaxation.py` — `ε = √mean_triangle_area`.
- `src/optimization/perimeter_optimizer.py` — Phase 2 IPOPT equal-area constraint.
- `src/partition/find_contours.py` — winner-take-all classification;
  `detect_dormant_cells()` (which does *not* catch runts).
- Bogosel & Oudet, *Partitions of Minimal Length on Manifolds*, Experimental
  Mathematics (2023); arXiv:1606.02873 — the original method.
</content>
</invoke>
