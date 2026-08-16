# Replacing Phase 1 at High N — Shared Harness, then B, then C

**Status:** Not Started
**Revision:** v2 (2026-08-15) — v1 was rewritten after adversarial review; see
[Correction history](#correction-history).

## Background

Phase 1 (Γ-convergence relaxation by PGD) produces a *continuous* density field
whose winner-take-all readout is not a valid partition at high N: cells are area
imbalanced, and sometimes disconnected. The standing taxonomy is
[`../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`](../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md),
with the source proposal verbatim at
[`../reference/phase1_highn_proposal_ABCDE_original.md`](../reference/phase1_highn_proposal_ABCDE_original.md).
**A** (balanced readout by semi-discrete OT dual shifts) and **E** (connectivity
repair) shipped as `src/partition/balanced_readout.py`. **A2** was rejected in
its pure and adaptive forms. The gate on **B** and **C** was met 2026-08-12.

Prior attempts on the N=300 ladder, stated accurately:

| Attempt | Outcome |
|---|---|
| Territory-aware relaxation | Rejected — fixed N=200 *validity*, manufactured 14/200 disconnected cells; never an N=300 cost fix |
| A2, pure soft area constraint | Rejected — 32× faster, but 3–4/100 fragmented and a collapsed line search |
| A2, adaptive within-level switch | Rejected — same failure |
| A2, exact-at-level-0 hybrid | **Viable, with a dependency** — 6.83×, all gates pass, Phase 2 within +0.246%; requires the balanced readout. Measured at N=100; never run at N=300 |

### The motivating argument is cost, and only cost

PGD costs **26–80 h per N=300 solution** (measured `total_wall_s`: 48,132 s for
the N=100 deliverable; 92,751 s and 130,588 s for the two λ=12 N=300 runs). B and
C are estimated at minutes. That gap is what does not survive scaling to N=400,
N=500, N=1000.

**This plan does not claim PGD produces invalid partitions inherently.** An
adversarial review on 2026-08-15 (commit `c16fcc3`) withdrew three claims, and
none of them may be relied on here:

- "N=300 is energy-limited, not descent-limited" — **unsupported**; level 2
  floored its line search mid-descent with ‖g‖ = 31.11.
- The "~90 verts/cell runs-or-dies boundary" — **not established**.
- "Neither more iterations nor more levels reaches the gate" — extrapolates two
  levels.

Whether a better ladder could reach the gate is **unknown**. The case for B and C
rests on cost alone, which is independent of that question.

## Verified state of the evidence base

**Corpus enumerated (2026-08-15):** both working directories of this repository —
`/Users/teban/Development/projects/surface-partition/results/` (7 `*npart300*`
runs) **and** the sibling worktree
`/Users/teban/Development/projects/surface-partition-territory/results/`
(2 `*npart300*` runs). `results/` is untracked, so **each worktree has its own**;
enumerating only one is how v1 of this plan went wrong.

### N=300 Phase 2 baselines — these exist

| Run | Location | Mesh | Phase 2 best perimeter | Iters |
|---|---|---|---|---|
| `run_20260806_123326` λ=11.5 seed 61803399 | territory worktree | **V = 47,488** | **323.319247087254** | 19 |
| `run_20260808_191030` λ=11.5 seed 61803399 | territory worktree | **V = 114,144** | **322.96218780465847** | 19 |

Both via `readout/dualshift_gate0.05_repair/refinement/…`, plain energy
(`wta_*` flags all False), with `partition/` exports. `run_20260806_123326` is
also the control cited by this repo's own committed report 05.

**Cross-mesh scatter between them: 0.11%** — a usable nuisance-variance datum.

### Other anchors

| Fact | Status |
|---|---|
| N=100 deliverable `run_20260709_081548`: Phase 2 best **185.2546144718457** (iteration 020, minimum over 20), worst cell 0.7786%, 0 dead / 0 weak | ✅ re-verified |
| ⚠ That run's `metadata.yaml` has **no `disconnected_cells` block** — it predates the gate. "All gates clean" rests on CLAUDE.md, not run metadata | ❗ recompute in Phase 0d |
| λ=12 `run_20260714_224821` (seed 61803399), V=47,488: 9/300 imbalanced, worst 34.95% | ✅ |
| λ=12 `run_20260716_152451` (seed 27182818), V=47,488: 10/300 imbalanced, worst 40.62% | ✅ |
| ⚠ **Both λ=12 runs also fail the connectivity gate: 2/300 fragmented each** — recomputed with `testing/check_fragmentation.py`; absent from their metadata, which predates the gate. Seed 27182818's worst stray is **41.77% of target**, beyond anything the repair has been shown to fix | ❗ new |
| Neither λ=11.5 run *in this checkout* completed — `run_20260813_000958` crashed in the entry projection ~1 min in; `run_20260813_003231` stops at `checkpoint_level03.h5` | ✅ (irrelevant now — the worktree runs supersede them) |
| Control Phase 2 campaign settings: `method: ipopt`, `boundary_tol: 0.001`, `max_iterations: 20` | ✅ |

## Phase 0 — shared solver, harness, and baseline

**Status:** Not Started

Nothing downstream may modify any of this. Freezing it is what makes the arms
comparable, and it is where an error is most expensive.

### 0a — Generalize the balanced-assignment solver

`solve_dual_offsets(log_densities, lumped_mass, target, config)`
(`src/partition/balanced_readout.py:143`, sole caller at `:446`) uses the score
matrix only through `np.argmax(scores + psi, axis=1)`, so it is **correct** for
any additive score matrix. A calls it on `log u`; B will call it on diffused
indicators `y`; C on `−d²`.

**But it is not generic in convergence, and this is a real trap.** The
subgradient schedule (`dual_eta0 = 0.5`, `dual_decay = 0.02`, `dual_iters = 400`,
`balanced_readout.py:57-59`) is calibrated to log-density scale, where the N=300
readout needed ψ ∈ [−0.75, +3.46]. For C's `−d²` on this torus (R=1, r=0.6), cell
radius ≈ 0.27 at N=100 gives per-edge score granularity ≈ 0.008 and needed
offsets O(0.02) — while the first step is ≈ 0.15–0.5, a 10–25× overshoot decaying
only to 0.056 by iteration 400. Worse, the function **returns its best iterate**,
so non-convergence is silent. Even on its home scale the dual stage reached only
worst-dev 2.12% at V=47,488/N=300 before repair.

**Required:** score normalization (or scale-aware η) so any arm's scores land in
a comparable range.

**Acceptance — two gates, both required:**
1. The balanced readout is **byte-identical** before and after on a fixed input
   (protects the shipped A path).
2. **Convergence on foreign scales:** synthetic `−d²` and `y` score matrices
   reach worst-dev ≤ 1% within the iteration budget. Gate 1 alone cannot detect
   the scale trap — it only exercises A.

### 0b — Build the evaluation harness

One harness, used unchanged by every arm and by the control. Given labels on a
mesh it reports: all three validity gates; perimeter after Phase 2 under
**pinned** settings (`ipopt`, `boundary_tol 0.001`, `max_iterations 20`); and
wall time.

**Pinned decisions** (v1 left all of these open):

- **Arm mesh:** the control's finest, **V = 114,144** at N=100; **V = 47,488** at
  N=300.
- **Wall-time protocol:** PGD uses `timing_profile.yaml`'s `total_wall_s` (never
  `run_time_seconds`, which under-reports 3.6×). Arms produce no such file, so
  arm wall time is process wall-clock from a single-purpose driver, reported per
  level. **This asymmetry is disclosed, not hidden.**
- **Perimeter *before* Phase 2 is recorded but never used as a criterion.**
  Midpoint VP extraction inflates it ~13.3% (control: 214.34 → 185.85 after the
  first optimize), which swamps any difference between arms.
- **`max_iterations: 20` is the control's budget and may disadvantage an arm**
  whose boundaries need more migrations. If an arm is still improving at
  iteration 20, that is **reported as a censored result**, not scored as a loss.

### 0c — N=300 baseline: adopt, do not rebuild

v1 called this "build the missing baseline… a real compute cost on the critical
path." That was wrong — the baselines exist (above).

**✅ DONE (2026-08-15). Both runs are now in this checkout's `results/`**, copied
from the territory worktree (12.6 GB; sizes, file counts and both Phase 2
perimeters verified identical). Re-verified after the copy:

| Run | Mesh | Gates | Worst cell | Phase 2 |
|---|---|---|---|---|
| `run_20260808_191030` | V = 114,144 | **all three pass** | 0.64% | 322.96218780465847 |
| `run_20260806_123326` | V = 47,488 | **all three pass** | 1.63% | 323.319247087254 |

Zero compute spent. If a λ=12 baseline is later wanted for seed diversity,
readout + Phase 2 on the existing λ=12 solutions costs **≈ 17 s + ~40 min** at
V=47,488 — about an hour, not days.

⚠ **Two caveats on this data.** (i) `results/` is gitignored, so these 12.6 GB are
**not in version control** and now exist only as two copies on one disk — a
machine that hard-reset three times in the week of 2026-08-07..15. The
irreplaceable part is small: the two `partition/*.h5` exports total **20 MB**.
(ii) The older export
`torus_partition_run_20260806_123326_...h5` records
`source_run_id = "dualshift_gate0.05_repair"` — a campaign name, the exact bug
fixed by `05f0eb1` (cherry-picked as `ac7e79c`). It predates the fix; the
`run_20260808_191030` export, made after it, is correct. Re-export the older one
before citing its provenance.

### 0d — Shake down the harness on the control, before any arm exists

Push the **control's own labels** through the entire harness and confirm it
reproduces **185.2546144718457**. Also recompute the control's connectivity gate,
which its metadata never recorded.

If the harness cannot reproduce a known answer, no arm result from it means
anything. This replaces "use C as the shakedown."

## Phase 1 — B: auction-dynamics MBO

**Status:** Not Started

Threshold dynamics (Esedoğlu–Otto) with volume constraints by balanced assignment
(Jacobs–Kim–Léger, JCP 2018). **B descends the project's actual objective** — the
thresholding energy Γ-converges to perimeter — which is why the taxonomy ranks it
2, above C at 4.

Iterate: (1) diffuse each indicator for time τ by solving `(M + τK) y_k = M χ_k`,
one prefactorized sparse solve per level (`scipy.sparse.linalg.factorized`;
`sksparse` is not installed and is not needed); (2) reassign by balanced
thresholding `ω(i) = argmax_k [y_ik + ψ_k]` via 0a. Seeded init, the mesh ladder,
M/K/v, HDF5 formats and Phase 2 carry over unchanged.

**Pinned:** τ = (c·h)² with **c = 2** by default, h the mean edge length at the
level. Pinning occurs when the diffusion length falls to the mesh scale, so
**√τ/h must be reported per level and must exceed 1**; c is a calibration item
with that acceptance check, not a free parameter to tune against the result.

## Phase 2 — C: capacity-constrained geodesic Lloyd

**Status:** Not Started

N sites (from the existing farthest-point seeds in
`src/optimization/initialization.py`); geodesic distance by multi-source Dijkstra
on the mesh edge graph (`scipy.sparse.csgraph`); assignment by the 0a solver with
cost `d²`; Lloyd step to the `v`-weighted centroid, snapped to the nearest
vertex; labels straight to Phase 2.

C runs **after** B, as the control that makes B's number interpretable — if C
matches B, B's diffusion machinery bought nothing.

**Two limitations stated up front.** (i) C minimizes a *quantization* energy, not
perimeter. The honeycomb argument for it landing near-optimal needs *both* halves
— Hales (hexagons minimize perimeter) and Fejes Tóth/Gersho (hexagons minimize
quantization error) — and both are **flat-2D** results. This torus has Gaussian
curvature of both signs and hexagonal commensurability is the untested D(ii), so
neither theorem transfers. How far off C lands is exactly what this measures.
(ii) **Graph-Dijkstra is not the geodesic metric** — on a structured torus mesh,
edge-path distances are anisotropically inflated by several percent depending on
direction, which distorts C's cells directly and could alone consume the +1%
budget. Report this as a named confound; the heat-method alternative needs
gradient/divergence operators the repo does not have.

## Pre-registration

Written before any code exists.

### Primary — N=100, against `run_20260709_081548`

Control Phase 2 perimeter **185.2546144718457**, V = 114,144.

**Threshold calibration** (v1 asserted +1% with no basis): same-partition Phase 2
scatter is **0.024%** (structure-trigger A/B); cross-mesh N=300 scatter is
**0.11%**. So +1% is ≈ 10× the largest measured nuisance variation, and +5% is
the proposal's own refutation bound — **the "3–5%" range is resolved to 5%** here.

| Perimeter outcome | Verdict |
|---|---|
| ≤ 187.11 (+1%) | **Success** |
| 187.11 – 194.52 (+1% to +5%) | **Partial** |
| > 194.52 (+5%) | **Refuted on perimeter** |

⚠ The Partial lane auto-triggers on wall time if "large gain" is undefined, since
every arm is ~100× faster by construction. **"Large" is therefore not a
criterion**: Partial results are reported with both numbers and escalated to a
decision, never auto-promoted.

### Validity grading — corrected

v1's rule ("the readout is withheld because both arms are connected by
construction") rested on a **false premise**. The source proposal says the
opposite: for B, *"Connectivity is not per-step guaranteed"* (line 35); for C,
*"connected in practice; residual violations go to the repair stage (E)"*
(line 47). Refuting an arm for a stray island would discard it for a failure mode
the proposal predicted and **E was built to absorb**.

Also note **two of the three gates are vacuous for any balanced-assignment arm**:
dormant cannot fire on one-hot labels where every cell owns ≥ 480 vertices, and
area passes whenever the internal solver converges (vertex granularity ≈ 0.09% at
V=114k/N=100, against a 5% gate). **"All three gates pass" has content only on
connectivity.**

| Connectivity outcome | Verdict |
|---|---|
| 0 fragmented raw | **Clean** |
| Fragmented, but E repairs to 0 with strain comparable to the control's | **Viable, composed with E** — report island count, stray mass, repair moves/blocked |
| E cannot repair, or repair strain is extreme | **Refuted on validity** |

Area and dormant gates are still run and reported, and their vacuity is stated
alongside.

### Secondary — N=300 scaling

Against the adopted baselines: **322.96218780465847** at V=114,144 and
**323.319247087254** at V=47,488. Same thresholds. Report **both λ=12 seeds**
(61803399, 27182818) where seed sensitivity is at issue — fragmentation location
is seed-determined, so one seed is not a result.

### Rules that protect the comparison

1. Validity is graded on the lanes above, not pass/fail; the readout is applied
   to arms **only** in the "composed with E" lane, and that composition is
   disclosed in the result.
2. Gates are evaluated on the arm's **raw** labels first, always.
3. Wall time per the 0b protocol, with the PGD/arm asymmetry disclosed.
4. C's number is recorded before it is compared to B's.
5. No claim of the form "every / always / never" without naming the corpus **and
   how it was enumerated** — including which working directories were searched.
6. Censored results (arm still improving at `max_iterations: 20`) are reported as
   censored.

## Risks

| Risk | Mitigation |
|---|---|
| 0a's scale trap silently degrades an arm | 0a acceptance gate 2 on foreign score scales |
| Harness itself is wrong | 0d reproduces a known answer before any arm exists |
| Graph-Dijkstra anisotropy consumes C's error budget | Named confound; quantify against an analytic torus geodesic on a test case |
| B's mesh pinning | Report √τ/h per level; must exceed 1 |
| `max_iterations: 20` truncates an arm | Censored-result rule |
| Stopping after C looks acceptable, never building B | B runs **first**; C is the control |
| Both arms lose to PGD | Still decisive — would refute framing point 5 |

## Correction history

**v1 → v2 (2026-08-15), after adversarial review.** v1's baseline audit claimed
no N=300 Phase 2 baseline, no readout campaign, and no `run_20260806_123326`
existed. **All three were wrong**: they exist in the sibling worktree, whose
`results/` is a separate untracked directory. v1 enumerated one working directory
and stated a universal — the exact corpus-enumeration error corrected in
`c16fcc3` the day before, and a violation of v1's own rule 6. Consequences: Phase
0c collapsed from "expensive, critical path" to "adopt, zero compute"; rule 1's
connectivity premise was refuted against the source proposal; the λ=12 runs'
fragmentation (2/300 each) was missing; the arm ordering was reverted to B-first
after the "C is cheaper" claim failed to survive scrutiny (C is plausibly
*slower* at runtime — N Dijkstras × ~30 Lloyd iterations — and implementation
sizes are comparable); and the "four attempts failed" background mischaracterized
two of its four items.

## Related documents

- Taxonomy: [`../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`](../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md)
- Source proposal: [`../reference/phase1_highn_proposal_ABCDE_original.md`](../reference/phase1_highn_proposal_ABCDE_original.md)
- The gap: [`../reference/winner_take_all_partition_gap.md`](../reference/winner_take_all_partition_gap.md) §4b, §4c, §9b
- `docs/experiments/05-soft-area-constraint/`, `docs/experiments/06-subfloor-ladder/`
- Code: `src/partition/balanced_readout.py`, `src/optimization/initialization.py`, `src/mesh/tri_mesh.py`, `src/partition/find_contours.py`
