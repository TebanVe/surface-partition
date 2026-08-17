# Replacing Phase 1 at High N — Shared Harness, then B, then C

**Status:** Phase 0 ready to start
**Revision:** v3 (2026-08-15). v1 → v2 → v3 after two adversarial review rounds;
see [Correction history](#correction-history).

## Background

Phase 1 (Γ-convergence relaxation by PGD) produces a *continuous* density field
whose winner-take-all readout is not a valid partition at high N. The standing
taxonomy is
[`../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`](../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md);
the source proposal is verbatim at
[`../reference/phase1_highn_proposal_ABCDE_original.md`](../reference/phase1_highn_proposal_ABCDE_original.md).
**A** (balanced readout) and **E** (connectivity repair) shipped as
`src/partition/balanced_readout.py`. **A2** was rejected. The gate on **B** and
**C** was met 2026-08-12.

Prior attempts on the N=300 ladder, stated accurately:

| Attempt | Outcome |
|---|---|
| Territory-aware relaxation | Rejected — fixed N=200 *validity*, manufactured 14/200 disconnected cells; never an N=300 cost fix |
| A2, pure soft area constraint | Rejected — 32× faster, but 3–4/100 fragmented, collapsed line search |
| A2, adaptive within-level switch | Rejected — same failure |
| A2, exact-at-level-0 hybrid | **Viable, with a dependency** — 6.83×, all gates pass, Phase 2 +0.246%; requires the readout. N=100 only |

### The motivating argument is cost, and only cost

Measured PGD wall time for a completed N=300 solution ranges **22.0 h**
(`run_20260806_123326`, 79,068.6 s) to **79.3 h** (`run_20260808_191030`, whose
206,343.8 s covers only levels 3–4 and chains to 285,412 s across its resume).
The N=100 deliverable cost 48,132 s. B and C are *estimated* at minutes — an
estimate from the proposal, not a measurement, and the plan's only motivation.

**This plan does not claim PGD produces invalid partitions inherently.** Three
claims withdrawn by the 2026-08-15 review (`c16fcc3`) may not be relied on:
"N=300 is energy-limited"; the "~90 verts/cell" boundary; "neither more
iterations nor more levels reaches the gate". Whether a better ladder could reach
the gate is **unknown**. The case rests on cost alone.

## Verified state of the evidence base

**Corpus enumerated (2026-08-15):** both working directories —
`surface-partition/results/` and `surface-partition-territory/results/`.
`results/` is gitignored, so **each worktree has its own**; enumerating one is how
v1 went wrong.

### The two valid N=300 partitions — now in this checkout

Copied from the territory worktree 2026-08-15 (12.6 GB, verified identical), then
re-verified here:

| Run | Mesh | Gates | Worst cell | Phase 2 best | Best at |
|---|---|---|---|---|---|
| `run_20260806_123326` | **V = 47,488** | all pass | 1.63% | **323.319247087254** | iter 18 of 19 — **oscillating, plateaued** |
| `run_20260808_191030` | V = 114,144 | all pass | 0.64% | 322.96218780465847 | iter 19 of 19 — **still falling, censored** |

⚠ **These are one experiment, not two.** `run_20260808_191030`'s
`warm_start_path` points at `run_20260806_123326`'s solution: it *resumed* from
it. Consequences:

- The 0.11% gap between them is a **within-trajectory mesh-refinement effect**,
  not scatter between replicates. It is not a nuisance-variance datum.
- The V=114,144 number is **censored** — its last four iterates fall monotonically
  (322.9907 → 322.9700 → 322.9737 → **322.9622**), so the true PGD value is
  ≤ 322.96. Anchoring on it would bias in the arms' favour.
- `run_20260806_123326` oscillates over its last four (323.3204, 323.3434,
  323.3192, 323.3406) — genuinely plateaued, and therefore the **safe anchor**.

### Other anchors

| Fact | Status |
|---|---|
| N=100 control `run_20260709_081548`: Phase 2 **185.2546144718457** (iter 020, min of 20), worst 0.7786%, 0 dead/weak, **0 fragmented** | ✅ all re-verified |
| λ=12 `run_20260714_224821` (s61803399), V=47,488: 9 imbalanced / 34.95%; **2/300 fragmented** (290 → 12.14%, 274 → 8.32%) | ✅ recomputed |
| λ=12 `run_20260716_152451` (s27182818), V=47,488: 10 / 40.62%; **2/300 fragmented** (106 → **41.77%**, 214 → 9.64%) | ✅ recomputed |
| Vertex-granularity floor (`max vᵢ / target`): **1.011%** at V=47,488/N=300; **0.42%** at V=114,144/N=300; ~0.14% at V=114,144/N=100 | ✅ from readout metadata |
| Dual stage stalls **above** that floor on its home scale: `dual_worst_rel_dev` **2.02%** and **2.12%** at N=300 | ✅ |

### The 41.77% stray repairs cleanly — measured, not assumed

v2 called it "beyond anything the repair has been shown to fix." **That was
wrong.** Measured 2026-08-15 on `run_20260716_152451`:

```
source  : 10 imbalanced (worst 40.62%), 2 fragmented,  boundary 189.6959
repaired:  0 imbalanced (worst  1.80%), 0 fragmented,  boundary 194.9435
+2.77% boundary · 2279 vertices relabeled · 2469 moves, 5 blocked, 54 sweeps
VALID PARTITION: YES
```

Nothing like A2's +22% catastrophe. **Note 54 sweeps > the old cap of 50** — this
converged only because `max_repair_sweeps` was raised to 200 earlier today.

So a repaired λ=12 control is *usable*. But +2.77% boundary vs λ=11.5's +1.88%
enters Phase 2, and how much of that Phase 2 erases is unmeasured — a fraction of
a percent of residual inflation is the same order as the +1% success bar. Hence
the anchor rule below.

## Phase 0 — shared solver, harness, baseline

**Status:** Ready to start

### 0a — Generalize the balanced-assignment solver

`solve_dual_offsets` (`src/partition/balanced_readout.py:143`, sole caller `:446`)
uses scores only through `np.argmax(scores + psi, axis=1)`, so it is **correct**
for any additive score matrix. A uses `log u`; B will use diffused `y`; C uses
`−d²`.

**It is not generic in convergence.** The schedule (`dual_eta0 = 0.5`,
`dual_decay = 0.02`, `dual_iters = 400`, `:57-59`) is calibrated to log-density
scale (ψ ∈ [−0.75, +3.46]). For C's `−d²` at N=100, needed offsets are O(0.02)
while the first step is ≈ 0.15–0.5 — a 10–25× overshoot decaying only to 0.056 by
iteration 400. The function **returns its best iterate**, so non-convergence is
silent.

**Required:** score normalization (or scale-aware η).

**Acceptance — both gates required:**
1. Readout **byte-identical** before and after on a fixed input (protects A).
2. **Foreign-scale convergence** on synthetic `−d²` and `y` matrices, pinned at
   **V = 47,488 / N = 300** and **V = 114,144 / N = 100**, reaching
   worst-dev ≤ **max(1%, 2× vertex granularity)**. A flat 1% bar is *infeasible*
   at V=47,488/N=300, where the granularity floor alone is 1.011%.

Gate 1 alone exercises only the A path and structurally cannot detect the trap.

#### 0a status (2026-08-16): gate 1 PASSES, gate 2 FAILS — and the failure matters

Implemented as `normalize_scores` (default **off**, so A is untouched) plus
`assignment_margin_scale`. Harness: `testing/test_balanced_assignment_solver.py`.

**Gate 1 — PASS.** The shipped readout reproduces its recorded reference exactly
on `run_20260716_152451`: 0 imbalanced, worst 1.80%, 0 fragmented, 2,469 moves,
54 sweeps.

**Gate 2 — FAIL**, at both pinned meshes, for both arms:

| Scores | Mesh | bar | norm OFF | norm ON |
|---|---|---|---|---|
| C: `−d²` | V=47,488 / N=300 | 2.02% | 57.31% | **3.65%** |
| C: `−d²` | V=114,144 / N=100 | 1.00% | 0.23% | 0.34% (passes) |
| B: diffused | V=47,488 / N=300 | 2.02% | 45.35% | 45.35% |
| B: diffused | V=114,144 / N=100 | 1.00% | 28.59% | 35.40% |

Normalization is **directionally right and insufficient**. It buys C an order of
magnitude at N=300 (57.31% → 3.65%) and helps B once τ is large enough
(c=32: 188% → 12.83%), but neither arm reaches its bar. Two knobs were measured
and eliminated:

- **Margin percentile.** Swept p1/p5/p10/p25/p50. Higher is better for C
  (p50 best); B is bad at every percentile. Not a tuning problem.
- **Iteration budget.** Each dual iteration is an O(V·N) argmax — 14.2 M
  operations at V=47,488/N=300 — so the 10× budget that might close the gap
  costs ~20 min per solve. Not a practical fix.

**The conclusion is structural, and it revises a premise this plan inherited.**
The taxonomy's "A, B and C share one solver" is true of the *interface* — balanced
assignment given per-vertex scores — but not of the *algorithm*. `solve_dual_offsets`
is subgradient dual ascent, tuned for A's job: a small correction from an
already-nearly-balanced argmax. B and C ask it to rebalance **30–45% of the area
from an unbalanced start**, which it does not do at these sizes. Note the source
proposal names B *"auction dynamics"* — Bertsekas auction, not subgradient ascent —
so the mismatch is with our shipped approximation, not with B.

**Therefore 0a is not complete.** It needs a solver that converges from far away:
an auction algorithm or an entropic/Sinkhorn solve for the assignment, kept behind
the same interface so A's path stays byte-identical. That is the next task, and it
is larger than the rename this step was scoped as. **B must not be built until
gate 2 passes** — a non-converging assignment step would surface as an area-gate
failure attributed to B rather than to the harness, exactly the confusion the
area-gate lane was created to prevent.

### 0b — Build the evaluation harness

One harness, used unchanged by every arm and by the control.

**Pinned decisions:**

- **Phase 2 settings are pinned by copying the control campaign's entire
  `refinement.yaml`**, not by naming a few fields. The control uses `ipopt`,
  `boundary_tol 0.001`, `max_iterations 20`, **`exact_hessian: true`,
  `lbfgs_memory: 30`, `best_iterate: true`, `allow_partial_convergence: true`,
  `max_opt_iter: 200`, `tolerance: 1e-7`, `distance_preservation: preserve`.
  Naming only three fields would silently swap the exact Hessian for L-BFGS — a
  *different optimizer*. Note the two N=300 baselines used
  **`max_iterations: 19`**, not 20; match the campaign being compared against.
- **Arms run the same mesh ladder as PGD** (it carries over unchanged). The
  *comparison* mesh is the arm's final mesh, and it must be **mesh-matched to the
  baseline it is scored against**.
- **Wall time:** PGD from `timing_profile.yaml` `total_wall_s` (never
  `run_time_seconds`). **For resumed runs, chain the segments** —
  `run_20260808_191030` is 285,412 s, not the 206,344 s its profile shows. Arms
  produce no profile, so arm wall time is process wall-clock from a driver
  script, reported per level. **This asymmetry is disclosed, not hidden.**
- **Perimeter before Phase 2 is recorded, never scored.** Midpoint extraction
  starts the control at 214.34 against its 185.85 after one optimize step (+15.3%
  above the optimized value), which swamps any difference between arms.
- **Censored results** — an arm (or baseline) whose perimeter is still falling at
  the final iterate is reported as **censored**, with its trajectory. A censored
  arm is not scored against a threshold; a censored *baseline* is flagged as a
  bound (`≤ x`), not a value.

#### 0b status (2026-08-16): library and acceptance PASS

`src/partition/arm_harness.py` + `testing/test_arm_harness.py` (5/5 pass). The
harness writes an arm's labels in the **Phase 1 solution schema**, so
`refine_perimeter.py`, `check_fragmentation.py`, the viewers and
`export_partition.py` all consume arm output with no new flags; runs Phase 2 as a
**subprocess** against a campaign's whole `refinement.yaml`, so an arm travels the
exact code path the control travelled rather than a reimplementation; and reports
gates, best-iterate perimeter, trajectory, censoring and wall time.

**The apples-to-apples assumption is now measured, not assumed.** On the real
N=300 λ=12 solution, hardening the continuous field to one-hot leaves both
scored gates *identical* — area 9 cells / 34.95% either way, connectivity
`[290, 274]` either way. So representing an arm's output as hard labels costs it
nothing relative to PGD's continuous field. The dormant gate goes to peak density
1.0 exactly, confirming its vacuity for arms in code rather than in prose. The
vertex-granularity floor also reproduces the shipped readout's recorded value to
9 decimals (0.010108 at V=47,488/N=300).

Remaining for 0b: a thin CLI wrapper. **0d (the end-to-end shakedown against
185.2546144718457) is the real remaining gate** and has not been run.

### 0c — N=300 baseline: ✅ DONE

Both runs are in this checkout, verified. **Zero compute spent.** If a λ=12
baseline is wanted for seed diversity, readout + Phase 2 costs ≈ 19 s + ~40 min.

⚠ The copied campaigns' `refinement.yaml` `base_solution` is an **absolute path
into the territory worktree**; the iteration files use relative
`base_solution_path` and are fine. Canonicalize before relying on it.

⚠ `results/` is gitignored — 12.6 GB not in version control, two copies on one
disk. The irreplaceable part is small: the two `partition/*.h5` exports total
**20 MB**. Also, the older export records
`source_run_id = "dualshift_gate0.05_repair"` (a campaign name) — the bug fixed by
`05f0eb1`/`ac7e79c`; it predates the fix. Re-export before citing its provenance.

### 0d — Shake down the harness on the control, before any arm exists

Push the **control's own labels** through the whole harness and reproduce
**185.2546144718457** using the control campaign's own `refinement.yaml`.

**Acceptance tolerance:** bitwise on this machine and environment; otherwise
**≤ 0.01%**. Phase 2 involves IPOPT and migrations and its bit-determinism has
never been demonstrated in this repo, so a bare 16-digit demand would fail for
reasons that mean nothing. If the harness cannot reproduce a known answer, no arm
result from it means anything.

## Phase A — B: auction-dynamics MBO

*(named to avoid colliding with the project's own "Phase 1 / Phase 2")*
**Status:** Not Started

Threshold dynamics (Esedoğlu–Otto) with volume constraints by balanced assignment
(Jacobs–Kim–Léger, JCP 2018). **B descends the project's actual objective** — the
thresholding energy Γ-converges to perimeter — which is why the taxonomy ranks it
2, above C at 4.

Iterate: (1) diffuse each indicator for time τ via `(M + τK) y_k = M χ_k`, one
prefactorized solve per level (`scipy.sparse.linalg.factorized`; `sksparse` is
absent and unnecessary); (2) reassign by balanced thresholding
`ω(i) = argmax_k [y_ik + ψ_k]` via 0a.

**τ — a two-sided window, not a floor.** Start at τ = (c·h)² with **c = 2**, and
report per level:

- **√τ / h_max**, not `h_mean`. This mesh's edge lengths are anisotropic by
  **1.81×** (mean 0.0171 / max 0.0311 at V=114k; 0.0266 / 0.0483 at V=47,488), so
  c=2 on the mean gives **√τ/h_local ≈ 1.10 at the outer equator** — at the pin,
  in exactly the coarse-triangle band, while a mean-based report shows a
  comfortable 2.0.
- **√τ / R_cell**, which must stay ≪ 1 — already 0.34 at N=300 with c=2, and
  growing ~√N at fixed mesh, so c=2 approaches the over-merging ceiling by
  N≈1000.
- **The operational pinning detector:** labels frozen while the energy is *not*
  converged. Per-step motion is ≈ τκ, which must be resolvable (≳ h); at c=2 that
  is 0.25·h at N=100/V=114k and 0.67·h at N=300/V=47,488 — inside the freeze
  regime for late-stage near-round cells. Non-freezing needs c ≳ √(R_cell/h)
  ≈ 4.0 and 2.4 respectively.

If a fixed τ pins, **anneal τ** — this is what the MBO literature does.

## Phase B — C: capacity-constrained geodesic Lloyd

**Status:** Not Started

N sites (farthest-point seeds from `src/optimization/initialization.py`); geodesic
distance by multi-source Dijkstra (`scipy.sparse.csgraph`); assignment by the 0a
solver with cost `d²`; Lloyd step to the `v`-weighted centroid snapped to the
nearest vertex.

C runs **after** B, as the control that makes B's number interpretable.

**Two limitations up front.** (i) C minimizes a *quantization* energy, not
perimeter. The honeycomb argument needs *both* halves — Hales (perimeter) and
Fejes Tóth/Gersho (quantization) — and both are **flat-2D**. This torus has
curvature of both signs and hexagonal commensurability is the untested D(ii), so
neither transfers. (ii) **Graph-Dijkstra is not the geodesic metric** — edge-path
distances are anisotropically inflated, unquantified on this mesh, and share a
budget with the +1% bar. Quantify against an analytic torus geodesic before
scoring C.

## Pre-registration

### Primary — N=100, against `run_20260709_081548`

Anchor **185.2546144718457**, V = 114,144.

**Calibration.** Cross-partition Phase 2 difference 0.024% (structure-trigger arm
— *cross*-partition, not same-partition, and its run is on neither disk, so this
rests on CLAUDE.md alone; the direction is conservative). Within-trajectory
mesh effect 0.11%. So **+1% is ≈ 10× the largest measured nuisance quantity**,
and +5% is the proposal's own bound — **"3–5%" is resolved to 5%**.

| Perimeter | Verdict |
|---|---|
| ≤ 187.11 (+1%) | **Success** |
| 187.11 – 194.52 | **Partial** — reported with both numbers, escalated to a named decision, never auto-promoted |
| > 194.52 (+5%) | **Refuted on perimeter** |
| still falling at the final iterate | **Censored** — not scored |

### Secondary — N=300 scaling

**Anchor: 323.319247087254 at V = 47,488** (`run_20260806_123326`) — mesh-matched
and plateaued. The V=114,144 figure is **censored** and carries as a bound
(≤ 322.96), never as the denominator.

**λ=12 is seed-diversity context, not a success denominator.** Report both seeds
with repair strain disclosed (s27182818: 0/1.80%/0 at +2.77% boundary, 2,469
moves, 54 sweeps).

### Validity grading

The premise that arms are "connected by construction" is **false** — the proposal
says B's connectivity is *"not per-step guaranteed"* (line 35) and C's *"residual
violations go to the repair stage (E)"* (line 47).

| Gate | Lane |
|---|---|
| **Dormant** | **Vacuous** for any balanced-assignment arm — one-hot labels, every cell owns ≥ 99 vertices even in the worst pinned case. Reported, not scored. |
| **Area** | **NOT vacuous — it is the 0a-failure detector.** The dual stage demonstrably stalls above the granularity floor (2.02–2.12% at N=300 vs a 1.011% floor). Area > **2× granularity** ⇒ **harness fault: fix 0a before scoring the arm**, never "arm refuted". |
| **Connectivity** | 0 fragmented ⇒ **Clean**. Repairable by E with strain comparable to the controls' ⇒ **Viable, composed with E**, disclosed. E cannot repair, or strain far exceeds the measured +1.88%/+2.77% band ⇒ **Refuted on validity**. |

### Rules that protect the comparison

1. Validity graded on the lanes above; the readout touches an arm **only** in the
   composed-with-E lane, and that composition is disclosed.
2. Gates evaluated on the arm's **raw** labels first, always.
3. Wall time per 0b, with the PGD/arm asymmetry and any resume chaining disclosed.
4. **Pre-commitment**, stated as what it is *for* rather than as an order:
   thresholds are committed before **either** arm runs, and **each arm's raw
   result is committed before the other's is examined.** This protects under
   either running order — which matters, because B is currently blocked on 0a
   while C already passes 0a's gate at N=100, so C may well run first. (v1 tied
   the rule to C-then-B; v2 flipped the order and left the wording, making it a
   tautology; this version is order-independent.)
5. No "every / always / never" without naming the corpus **and how it was
   enumerated**, including which working directories were searched.
6. Censored results reported as censored, for arms *and* baselines.

## Risks

| Risk | Mitigation |
|---|---|
| 0a scale trap degrades an arm silently | Gate 2 on foreign scales; area gate as the detector |
| Harness itself is wrong | 0d reproduces a known answer first, with a stated tolerance |
| τ pins in the anisotropic band | Report √τ/h_max and √τ/R_cell; labels-frozen detector; anneal τ |
| Dijkstra anisotropy consumes C's budget | Quantify against an analytic geodesic before scoring |
| Repaired control inflates the baseline | Anchor on λ=11.5 mesh-matched; λ=12 is context with strain disclosed |
| Stopping after C looks acceptable | B runs first |
| Both arms lose to PGD | Still decisive — refutes framing point 5 |

## Correction history

**v1 → v2.** v1's baseline audit claimed no N=300 Phase 2 baseline, no readout
campaign, and no `run_20260806_123326`. All three were wrong — they exist in the
sibling worktree, whose `results/` is a separate untracked directory. v1
enumerated one working directory and stated a universal: the same
corpus-enumeration error corrected in `c16fcc3` the day before, and a violation of
v1's own rule. Also fixed: rule 1's false "connected by construction" premise; the
0a scale trap; harness asymmetries; uncalibrated thresholds; a background that
mischaracterized the A2 hybrid as a failure; and the ordering (reverted to B-first
— "C is cheaper" did not survive).

**v2 → v3.** Round 2 verified 18 of v2's claims to the digit and refuted three.
(i) **Area-gate vacuity was wrong** — v2 asserted "≥ 480 vertices per cell" and a
"0.09% granularity" that match no pinned configuration; the real floors are 1.011%
at V=47,488/N=300 and 0.42% at V=114,144/N=300, and the dual stage stalls *above*
them, so the area gate can genuinely fire and is now the 0a-failure detector.
(ii) **The harness pin contradicted 0d** — pinning three fields would have run
L-BFGS against a control built with an exact Hessian, so 0d could not have
reproduced 185.2546; the whole `refinement.yaml` is pinned instead, and the N=300
baselines use `max_iterations: 19`. (iii) **The adopted baselines are one resumed,
partly censored trajectory**, so the 0.11% is a mesh effect and the V=114k figure
is a bound. Also restored the pre-commitment rule v2 had reduced to a tautology,
corrected the PGD cost range to 22.0–79.3 h, replaced the τ criterion with a
two-sided window on `h_max`, and **measured** the 41.77% stray that v2 called
unrepairable — it repairs to 0/1.80%/0 at +2.77%.

## Related documents

- [`../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`](../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md)
- [`../reference/phase1_highn_proposal_ABCDE_original.md`](../reference/phase1_highn_proposal_ABCDE_original.md)
- [`../reference/winner_take_all_partition_gap.md`](../reference/winner_take_all_partition_gap.md) §4b, §4c, §9b
- `docs/experiments/05-soft-area-constraint/`, `docs/experiments/06-subfloor-ladder/`
- Code: `src/partition/balanced_readout.py`, `src/optimization/initialization.py`, `src/mesh/tri_mesh.py`, `src/partition/find_contours.py`
