# Publication Readiness — the high-N partition pipeline as an independent project

**Status:** Not Started. This document is an **assessment and a proposed
programme**; none of its studies has been run.
**Date:** 2026-08-20.
**Audience:** future agents/developers, and the author deciding whether this is a
paper.

## Background

This repository began as the upstream component of a larger project: compute
minimal-perimeter partitions on closed surfaces, hand them downstream. Two things
changed that.

1. The Γ-convergence relaxation (Phase 1) became impractical at high N — **22–79 h**
   for a completed N=300 solution — and its winner-take-all readout was not a valid
   partition there, requiring a repair stage.
2. Approach **B** (volume-constrained threshold dynamics / auction-dynamics MBO)
   was built and measured (`docs/experiments/08-mbo-auction-dynamics/`). It
   replaces Phase 1 at **211–319×** less wall time on that stage, produces
   connected equal-area partitions needing no repair, and reaches a **shorter**
   Phase 2 perimeter than the incumbent at both anchored N.

That combination — a working method at N nobody has demonstrated, plus a
quantified account of *why* the incumbent readout diverges — is plausibly an
independent publishable contribution. This document records **what the evidence
actually supports**, what it does not, and the cheapest programme that would close
the gap between the two.

**It deliberately does not overlap** with `PHASE1_N1000_SCALING_PLAN.md` (sparse
representation / GPU for *PGD*) or `PHASE1_N1000_VALIDITY_PLAN.md` (the rejected
territory-aware fixes). Both target making **PGD** scale. This document assumes B
is the forward path and asks what is needed to publish it.

## The framing the evidence supports

**Not** *"the Γ-convergence relaxation fails at high N."* We never established
that, and three related claims have already been withdrawn (`c16fcc3`): "N=300 is
energy-limited", the "~90 verts/cell" boundary, and "neither more iterations nor
more levels reaches the gate".

The defensible claim is narrower, more precise, and accuses nobody:

> The Γ-convergence theory guarantees that the **relaxed energy** converges to
> perimeter as ε → 0. It does **not** guarantee that the **argmax readout** of a
> finite-ε, finite-mesh minimizer is an equal-area partition. At finite ε these
> two objects diverge, and the divergence grows with N.

⚠ **Sharpened 2026-08-21 after reading the source paper directly.** The readout is
**not** an unexamined choice of ours — it is the paper's own equation (5–1), and the
paper *does* discuss it, naming a triple-point void and zigzag contour length as its
difficulties. Its reported "Area tol." of 2–5×10⁻⁷ is the residual of the
*continuous* constraint, not of the extracted territory. **The reason our artifacts
do not appear there is range:** the method is demonstrated at **n ∈ [2, 11] on the
torus with exactly our R=1, r=0.6**, and n ≤ 32 on the sphere. We work at N = 100–400
— **10–36× beyond it**. So the honest claim is a *scaling* result, not an oversight,
and any manuscript must say "beyond the demonstrated range", never "unnoticed".

The divergence is exact, not statistical:

```
T_k − Ā  =  gain_k − lost_k
```

the net mass exchange across the interface band — a quantity neither the energy
nor the constraints reference. The theorem is untouched; the winner-take-all
readout is a practical choice with an unquantified error term, and quantifying it
is the contribution.

**This framing also removes the paper's most obvious referee attack.** Under it,
you are reporting a property of the *readout*, so you do not need PGD to have
converged — which matters, because we cannot show that it did (see below).

## What is ESTABLISHED

Each item is measured and reproducible in this repository.

| Claim | Evidence |
|---|---|
| B replaces Phase 1 at 211–319× on that stage | 214–248 s vs 48,132 / 79,069 s (`timing_profile` `total_wall_s`) |
| B reaches shorter Phase 2 perimeter than the best available PGD pipeline at both anchored N, at equal Phase 2 budget | 184.4118 / 184.1615 vs 185.2546; 319.9428 vs 323.3192 |
| B's raw labels are **connected** at N=100, 300, 400 | 0 fragmented; independently re-verified via `testing/check_fragmentation.py`, 0 cells with even sub-threshold speckle |
| B needs no readout and no repair stage | raw PGD at N=300 gives 10 imbalanced (worst 36.15%) + 2 fragmented and requires A+E; B requires nothing |
| The win is not the initialisation | init-only (approach C's step 0) reaches 189.6470 vs B's 184.4118 — MBO contributes +2.761% |
| The win is not seed noise | N=100 seed spread 0.136%, 3.3× smaller than the margin |
| B produces a valid N=400 partition in ~15 min | 0 fragmented, worst area 0.7773% (bar 1.1214%), perimeter 368.660323 |
| The N=400 partition obeys √N scaling | on a fixed mesh, N=100 × 2 = 368.8235 vs measured 368.6603 — **0.044%** after Phase 2, 0.111% before |
| The WTA gap has an exact accounting identity | `docs/math/07-phase1-wta-balance/`, Prop. 1 |
| Where the N=300 damage happens | levels 0–1 do no work (L0 flips *zero* labels); L2 does all of it, reaching 36.15%; splits are born mid-L2 while imbalance falls 234 → 10 |

## What is CONSISTENT with the data but NOT established

- **The extreme-value story.** That per-cell band exchange behaves like a random
  variable so the worst of N drifts like `σ√(2 ln N)`. It is a **model**. The
  distribution of `gain_k`/`lost_k` has never been measured, and the three data
  points we have (N=100/200/300) differ in λ *and* mesh, so they cannot test it.
- **That splits come from projection non-locality.** The locality criterion is
  3-for-3 at predicting failures (territory-aware term, A2, A2 adaptive), but no
  split under the *plain* energy has ever been traced to a projection step.
- **That B's advantage is combinatorial rather than geometric.** B enters Phase 2
  from a 0.243% *longer* boundary and exits 0.455% shorter — a real crossover, but
  nothing isolates topology from geometry.

## What is UNKNOWN

- **Whether PGD converged.** N=300 level 2 floored its line search at iteration
  6,628 with the energy still falling (−0.667 over the last 1,000) and ‖g‖ = 31.11.
  **No stationarity or KKT measure exists anywhere in the traces.** So the failure
  may be the optimizer, not the formulation.
- **Whether a better ladder would reach the gate.** Explicitly withdrawn as a
  claim; never tested.
- **PGD's own seed-to-seed variance.** Every N=100 PGD run across *both* worktrees
  is seed 84172851; every N=300 λ=11.5 run is seed 61803399. Each anchor is a
  single trajectory.
- **Over-merging.** A three-part instrument (fragmentation, isoperimetric ratio,
  core loss) detected nothing across a 9× span of √τ/R_cell. Not excluded — merely
  undetected.
- **Anything beyond the torus at r/R = 0.6.** The other surface providers are
  unmaintained and unverified since the energy fix.

## What CANNOT be claimed, and why

1. **"The relaxation fails at high N."** Not established; see above.
2. **"B beats PGD."** The anchors are single trajectories on one machine, and the
   two N use *different baselines* (raw PGD at N=100; PGD + readout at N=300), so
   the margins must never be pooled or read as an N-trend.
3. **"B is 211–319× faster."** True only of the replaced stage. End-to-end it is
   ≈23× (N=100), ≈31× (N=300), ≈9× (N=400). Against the structure-trigger PGD
   variant the incumbent-best figure is ≈153×.
4. **Anything comparative at N=400.** That run is exploratory and has no anchor.
5. **Novelty.** Our literature check was two PDFs (title pages, abstracts, keyword
   counts) plus the standing bibliography. That is **not** a literature review.

---

## Phase 1 — Three cheap studies on data already on disk
**Status:** Not Started. Highest value per unit compute; do these before writing
any mechanism claim.

### S1 — Did PGD converge?
Recompute a projected-gradient / KKT residual on the saved trace iterates
(`traces/*_internal_data.hdf5` hold `x`) at each level's termination for the N=100
and N=300 controls. Answers the referee's first question and settles whether
"PGD did not reach a valid partition" is a statement about the optimizer or the
energy. **Cost: hours of CPU at most, no new runs.**

### S2 — Is B's partition a *lower-energy* configuration of the same functional?
**The potential centerpiece.** Take B's final labels, project them onto the Phase 1
constraint set exactly as the seeded initial condition is built, and run PGD from
there **at the finest level only**, warm-started, for a bounded iteration budget.
Compare the converged Γ-energy against the control's final-level energy.

- If B's basin is **lower**: the variational principle was right all along and PGD
  is simply a poor descent scheme for it at high N. That is a stronger *and more
  generous* paper than "we replaced their method".
- If it is **higher**: B is trading energy for validity, which is also publishable
  but a different story.

⚠ Do not attempt to evaluate the energy on a hard one-hot indicator directly — the
Dirichlet term is not comparable between a hard indicator and an ε-scale profile.
The comparison must be basin-to-basin under the same optimizer. **Cost: hours, not
days, because only the finest level runs and it is warm-started.**

### S3 — Measure the band exchange
Compute `gain_k` and `lost_k` per cell on the existing N=100/200/300 solutions and
plot their distribution. Turns the `√(2 ln N)` argument from an assertion into a
measurement, and shows directly whether the worst cell is a tail event or a
systematic one. **Cost: minutes.**

## Phase 2 — A real literature search
**Status:** Not Started. Cheap insurance; do it before any novelty claim.

Threshold dynamics on graphs and surfaces is an active area (van Gennip et al. on
graphs; MBO on point clouds and in data clustering), and auction dynamics itself
came from a volume-constrained setting, so **"MBO on a surface" is probably not
novel on its own.** What is plausibly unreported is the *combination*:
volume-constrained MBO for equal-area minimal-perimeter partitions on a **closed,
curved, unstructured triangulated** surface at N in the hundreds, benchmarked
head-to-head against the Γ-relaxation.

Specific questions to answer: has volume-constrained MBO been run on unstructured
closed surfaces? has the winner-take-all readout gap been reported for
Γ-convergence partition methods? do surface-remeshing or graphics venues cover
this under different vocabulary? **Phrase any result as "we are not aware of",
never "first", and name the corpus and how it was enumerated.**

## Phase 3 — Scaling to N = 500 and N = 1000
**Status:** Not Started. This is the practical headline and the original goal.

A curve of wall time and validity versus N, out to where PGD cannot follow, is a
stronger argument than any perimeter margin. Use the exploratory `--config` mode
(no anchor exists at these N). Extrapolating measured B ladders (N=300: 248 s;
N=400: 920 s at 5 levels), the Phase 1 replacement should stay in the tens of
minutes; verify rather than assume.

Watch for: the τ over-merge cap binding on more levels as cells shrink relative to
the coarse mesh (it bound on 0 levels at N=100, 1 at N=300, 2 at N=400); the
vertex-granularity floor rising (0.56% at N=400/V=114,144) until the finest mesh
must grow; and peak memory, which is the dense `V × N` score matrix plus two
same-size transients inside the assignment solver (~1.5 GB at N=400).

## Phase 4 — Phase 2 is becoming the bottleneck
**Status:** Not Started. Likely the most consequential engineering finding.

Measured Phase 2 wall: ~33 min at N=100 (≈14,900 variable points), ~50 min at
N=300, **122 min at N=400** (29,244 variable points). It scales worse than
linearly in variable points, and B has already made Phase 1 only **11%** of
end-to-end cost at N=400.

At N=1000 this plausibly dominates completely. Before investing further in Phase 1,
profile Phase 2 at N=400 (`--profile`) and decide whether the next target is the
IPOPT/exact-Hessian path rather than the relaxation. **This reverses the project's
standing assumption that Phase 1 is the expensive stage.**

## Phase 5 — Breadth of the evidence base
**Status:** Not Started. Required for a paper, not for a decision.

- **Seeds.** At minimum 3 seeds per configuration for B; and — separately —
  measure **PGD's** own seed variance at the anchor configurations, without which
  "B beats PGD by more than PGD's own seed lottery" cannot be written.
- **A second surface.** The torus at r/R = 0.6 is the only validated geometry. The
  ellipsoid / double-torus / Banchoff-Chmutov providers exist but are unmaintained
  and unverified since the energy fix (`6ff71a0`). Reviving one is a real cost and
  should be scoped before being promised.
- **A second machine**, to separate hardware from method in the timing claims.

## Phase 6 — Split forensics
**Status:** Not Started. Lower priority; interesting rather than load-bearing.

At the iteration a split is born (N=300 mid-level-2, iterations ~2,500 and ~4,500),
decompose the update into its gradient and projection contributions and test
whether the new component's vertices had `u_k` raised by the *projection*. Would
convert the locality criterion from a 3-for-3 heuristic into a measured mechanism.

---

## Risks

| Risk | Note |
|---|---|
| Novelty claim collapses under a real search | Phase 2 is cheap; do it early |
| S2 shows B's basin is *higher* energy | Not fatal — it changes the story from "better descent" to "energy traded for validity", which is still publishable, but the framing must follow the result, not precede it |
| The paper is computational, with no new theorem | Acceptable, but the bar becomes reproducibility and thoroughness — where this repo is unusually strong (pre-registration, negative controls, six documented measurement artefacts, figures regenerable from committed data) |
| Over-claiming, again | This project's headline claims have been corrected repeatedly — report 06's was refuted, report 07's corrected three times, and report 08 required correcting a false per-level claim. Assume the same rate applies to anything written here |
| Phase 2 becomes the wall at N=1000 | Phase 4; may redirect the whole programme |

## Related documents

- Measured result: `docs/experiments/08-mbo-auction-dynamics/`
- Programme and full pre-registration: `docs/plans/PHASE1_BC_REPLACEMENT_PLAN.md`
- Standing explanation of the readout gap: `docs/reference/winner_take_all_partition_gap.md`
- Taxonomy of approaches A–E: `docs/reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`
- The exact WTA identity: `docs/math/07-phase1-wta-balance/`
- Instrument qualification: `docs/experiments/07-phase0-shared-harness/`
- **Distinct, PGD-focused plans this one does not supersede:**
  `docs/plans/PHASE1_N1000_SCALING_PLAN.md`,
  `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md`
- Code: `src/partition/mbo_auction.py`, `scripts/run_mbo_arm.py`,
  `scripts/score_mbo_arm.py`, `testing/test_mbo_auction.py`
