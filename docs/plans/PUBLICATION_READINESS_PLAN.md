# Publication Readiness — the high-N partition pipeline as an independent project

**Status:** assessment and proposed programme, **partly executed.** Per-phase
status lines below are authoritative; this summary is a pointer, not a record.

| Phase | State |
|---|---|
| 1 — S1/S2/S3, three cheap studies | **Not Started** — still the highest value per unit compute |
| 2 — literature search | **Partly answered** (2026-09-17/23): 35 papers held, doc 10 §2's prior-art table, the collaborator reading path. Not a systematic search |
| 3 — scaling | **DONE through N=1000**, on two meshes, both exported and finalised |
| 4 — Phase 2 is the bottleneck | **Measured; its original premise refuted** at four values of N |
| 5 — breadth | **Partly done** — geometry breadth exists (three aspect ratios, three sizes, two implicit surfaces); seeds and a second machine do not |
| 6 — split forensics | **Not Started** |

**Date:** 2026-08-20; revised 2026-09-30.
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
torus with exactly our R=1, r=0.6**, and n ≤ 32 on the sphere. We work at
N = 100–1000 — **9–91× beyond it**. So the honest claim is a *scaling* result, not an oversight,
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

**And it disposes of the "where is your N=400 baseline?" objection.** Requiring a
PGD anchor at every N asks the incumbent to do exactly what the new method exists
because it cannot do. The absence of an N=400 anchor is not a hole in the
evidence — it is the motivation, and it is measured: on the same mesh, one
hundred cells *fewer*, PGD needed 224× the wall time and still failed two of the
three validity gates. At N=400 the honest claim is existence and validity, which
is stronger than a margin because nothing on the other side survives to compare
against.

## The scope of the paper — settled 2026-09-17/23

Added after the provenance work (`docs/math/10-mbo-auction-dynamics/` §2, read
from the papers now held in `docs/papers/`) and written up in plain language in
`docs/explanations/two_methods_explained.md` §4–§5. **Neither method is ours,
and the manuscript must say so in its first paragraph.**

- **Method A** is Bogosel & Oudet 2017, implemented faithfully.
- **Method B** is Merriman–Bence–Osher (1992/1994) with the Esedoğlu–Otto
  variational reading (2015) and Jacobs–Merkurjev–Esedoğlu's volume constraints
  (2018) — and JME **already tiled the flat 2-torus into 64 equal cells with
  it**. The implicit-Euler blur is the graph-MBO step. Merriman–Ruuth ran the
  multiphase scheme on a *curved* torus in 2007, unconstrained.

⚠ An earlier novelty sentence in this repository ("uniform Cartesian grid via
FFT, flat domain, small N, outer problem") was **wrong on three of four
clauses** and is corrected in the docstring, report 08 §2 and doc 10 §2.2. The
defensible claim is **combination, surface and scale** — never any single
ingredient.

### What is reportable as new, ranked

1. **A diagnosis** — why the Γ-relaxation stops producing valid partitions at
   large N: the readout gap, its exact identity, and the measured artefacts. Not
   found reported for Γ-convergence partition methods, and it concerns anyone
   using them beyond a few dozen cells.
2. **A working pipeline at a scale nobody has shown** — equal-area
   minimal-perimeter partitions of a *curved, triangulated* surface with the
   diffusion done by surface finite elements, at N = 400–1000, coupled to exact
   contour refinement and three validity gates. The held literature has the flat
   torus at 64 cells, the curved torus at 5 unconstrained regions, and grain
   growth at 10⁵ unconstrained grains. Nobody has the combination.
3. **A head-to-head comparison on identical meshes at equal downstream budget** —
   cost, validity and final perimeter — with an attribution control (+2.8% over
   its own initialisation) and negative controls. The papers that proposed B
   never compared it with the Γ-relaxation.
4. **An implementation choice with its price quantified** — the exact auction
   replaced by an inexact transportation-dual solve with real vertex masses, and
   the consequences worked out: no dissipation theorem, a surviving inequality
   (gate G4), a granularity bar, non-integrality. **The contribution is the
   judgment, not the solver** — why this solver here, what it required, what it
   cost, how we know it sufficed. A paragraph and a proposition, not a
   contribution to optimal transport, and **never "better than the auction"**,
   which was never measured.
5. **Practical knowledge that transfers** — the τ window and what freezing and
   over-merging look like; that cost is governed by verts-per-cell, not by
   V × N; how to build the mesh ladder; and that once Phase 1 is fixed, Phase 2
   becomes the bottleneck.
6. **Two small mathematical observations** about doing this with finite elements
   rather than an FFT: the backward-Euler blur moves interfaces at **half** the
   speed the Gaussian would (interface constant ½ against 1/√π), and the
   discrete energy is not guaranteed to decrease. Modest, new, and checked —
   doc 10 §5 and §7, with a committed script regenerating every number.

### How to position it

A **computational** paper, not a methods paper: *we needed partitions with a
thousand equal cells; the standard method fails, for a reason we can name and
measure; an existing scheme from another community does it in minutes; here is
the evidence, the comparison and the recipe.*

Two rules, both learned the hard way:

- **Never write "first."** Write "we are not aware of", and name the corpus that
  was searched — doc 10 §2 lists it.
- **Keep the comparison and the existence results in separate sections.** At
  N = 100/300 there is a baseline and a margin. Above N = 300 there is no method-A
  baseline *because method A stopped producing valid partitions*, and that
  absence is the finding, not a gap in the experiment.

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
| B produces a valid N=400 partition in ~15 min | 0 fragmented, worst area 0.7773% (bar 1.1214%), perimeter 368.660323; re-verified 2026-08-22 with `check_fragmentation.py` (all three gates) and at the Phase 2 best iterate (max equal-area violation **8.18e-07**, 400 cells, 29,287 VPs) |
| At **matched mesh** V=114,144, PGD is both slower and invalid with *fewer* cells | PGD N=300: 206,344 s, 3 imbalanced (worst 24.81%), 2 fragmented → invalid raw. B N=400: 920 s, all gates pass. **224×** |
| The N=400 deliverable is exported and finalised | `arm_mbo_20260820_133709.../partition/torus_partition_n400_s84172851.h5`, `finalised=True`, iterate 19 |
| √N scaling holds across **three** N on a fixed mesh | N=100→400 −0.044%, **N=400→500 −0.015%**, N=100→500 −0.059%, all after Phase 2. The N=500 value was predicted in `parameters/torus_500part_mbo.yaml` *before* the run |
| B produces a valid N=500 partition in 22 min | all three gates on raw labels; 0 fragmented; worst cell 1.0993% (bar 1.4017%); Phase 2 **412.1138** at iterate 10/20, max equal-area violation 1.10e-06; exported and finalised |
| B produces a valid N=750 partition in 60 min | all three gates; **0 fragmented at every level** (a first at high N); worst cell 1.2569% (bar 1.5165%); Phase 2 **504.9214** at iterate 14/20, max violation 4.68e-07; exported and finalised |
| Phase 1 cost tracks **verts-per-cell**, not N | at matched V=24,948: 3.24 / 5.09 / 24.61 s per step for N=400/500/750 (62 / 50 / 33 v/cell). At the *finest* levels, where v/cell stays 200+, cost is within **+8.9%** of plain V·N, and per-step cost is *flat* across N=750's levels 0–2 while V triples |
| Phase 2 scales superlinearly in variable points | 0.250 / 0.270 / 0.389 / 0.479 s per VP at N=400/500/750/1000; exponents 1.69, 2.00, **1.74**. ⚠ An earlier row said "steepening" — the fourth point **refutes** that; 2.00 was an excursion. Read as *superlinear, ≈1.7–2.0, no evident trend*. First **fixed-mesh** pair (N=750→1000 at V=114,144) gives **2.36**, suggesting mesh effects confounded the mixed-mesh figures |
| A sub-threshold τ ladder rung is survivable under B | freeze ratio 0.964 (N=750) and 0.897 (N=1000) on the standard 100×96 base: both pass all three gates. The level captures 18%/26% of its available improvement vs 55–89% for healthy levels, and manufactures 2/3 fragmented cells that the ladder heals completely (`2→0…`, `3→2→1→0…`). Contrast PGD, where a starved level 0 leaves a *permanent* runt (report 06) |
| Worst cell saturates at ≈1.61× the granularity floor on a fixed mesh | V=114,144: 1.386 / 1.568 / 1.610 / 1.609 at N=400/500/750/1000 |
| B produces a valid N=1000 partition | **on two meshes.** V=209,568: 7,776 s Phase 1, worst cell 1.3473%, 0 fragmented, Phase 2 **583.2558** at iterate 15/20. Mesh-matched V=114,144: 3,803 s, worst 2.2560% (bar 2.8035%), 0 fragmented, Phase 2 **583.1417** at 16/20. Both exported and finalised |
| B is valid across **three torus aspect ratios and three surface sizes** | the nine-run mc-study set: R/r ∈ {1.400, 1.667, 2.500} at constant area, and Rr ∈ {0.600, 1.200, 2.400} at fixed shape. All nine valid and exported (`docs/reference/deliverables.yaml`, group `mc-study`) |
| B is valid on **two non-torus closed surfaces** | double torus (genus 2, V=12,448) and Banchoff-Chmutov order 4 (genus 5, V=26,928), N=10, all three gates on raw labels, 0 fragmented, 7 s and 17 s of Phase 1; both exported under schema 2.0. ⚠ N=10 only — this is breadth of *surface*, not of N |
| All nine N=50–1000 deliverables share one exported mesh | 114,144 verts / 228,288 faces, all `finalised=True`, equal-area to 0.0014–0.0272% |
| Phase 2's cost is **not** the solver | N=500: IPOPT 459 s of an 8,829 s campaign = **5.2%**; 201 solver iters per topology iteration, i.e. capped every time |
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
- **~~Anything beyond the torus at r/R = 0.6.~~** ✅ **Closed for B** (2026-09-02/07):
  three aspect ratios and three sizes on the torus, plus the double torus and
  Banchoff-Chmutov. What remains unknown is **PGD** off the r/R = 0.6 torus — its
  results on the other providers are *invalid*, not merely unverified (all 21
  archived runs fail the gates), so no cross-surface comparison exists. And B's
  non-torus evidence is **N=10 only**.

## What CANNOT be claimed, and why

1. **"The relaxation fails at high N."** Not established; see above.
2. **"B beats PGD."** The anchors are single trajectories on one machine, and the
   two N use *different baselines* (raw PGD at N=100; PGD + readout at N=300), so
   the margins must never be pooled or read as an N-trend.
3. **"B is 211–319× faster."** True only of the replaced stage. End-to-end it is
   ≈23× (N=100), ≈31× (N=300), ≈9× (N=400). Against the structure-trigger PGD
   variant the incumbent-best figure is ≈153×.
4. **A perimeter comparison at N=400.** No PGD anchor exists there, so no
   margin may be quoted. ⚠ **This is narrower than the earlier wording**
   ("anything comparative at N=400"), which was wrong: the *matched-mesh*
   statement is legitimate and is not a perimeter comparison. At the identical
   finest mesh V=114,144, PGD at **N=300** took 206,344 s and produced 3
   imbalanced (worst 24.81%) + 2 fragmented cells — an invalid partition needing
   the readout — while B at **N=400** took 920 s and passed all three gates
   (worst 0.78%, 0 fragmented). Stating that is a claim about *validity and
   cost*, both measured on both sides. What may not be said is anything of the
   form "B's N=400 perimeter beats PGD's".
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
**Status:** **Partly answered** — 35 papers held and read at the cited sections,
which settled the prior-art table but is **not** a systematic search. The
graphics / remeshing question is still open, and it is the one that gates any
novelty sentence.

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

**Partly answered 2026-09-17/23.** The corpus is now **35 held PDFs**, each row
of `docs/papers/README.md` recording what was verified *from the copy*; the
prior-art table is `docs/math/10-mbo-auction-dynamics/` §2; and
`docs/explanations/reading_path_for_a_collaborator.md` orders the papers as a
discovery path, which doubles as the enumeration a manuscript must name. From
that reading: Jacobs–Merkurjev–Esedoğlu 2018 §4.2
already computes equal-area minimal-perimeter tessellations of the **flat**
2-torus at N=64 and area-preserving flow at N=160 (exact auction, uniform grid);
Merriman–Ruuth 2007 run multiphase MBO on a **curved** torus (5 regions,
unconstrained, closest-point method); Wang–Osting 2019 do diffusion-generated
*Dirichlet* partitions of the sphere (k ≤ 20); grain-growth MBO runs at 10⁵
grains. So "MBO on a surface" and "large N" are each established separately, and
the combination above is what remains unreported *in that corpus* — which is
**thirty-odd papers read at the cited sections, not a systematic search**, and
the graphics/remeshing question is **still open**. That open question is the one
thing standing between the current state and a defensible novelty sentence, and
it cannot be closed by reading more of the same corpus.

## Phase 3 — Scaling to N = 1000
**Status:** ✅ **DONE.** N=400, 500, 750 and **1000** all run in exploratory
`--config` mode, all valid on raw labels, all exported and finalised
(2026-08-20 → 08-27). N=750 and N=1000 were each run **twice** — once on a
re-based ladder for partition quality, once on the standard ladder so the whole
N=50→1000 series shares one exported mesh. This was the practical headline and
the original goal, and it is met.

| N | mesh V | Phase 1 | worst cell (bar) | fragmented | Phase 2 |
|---|---:|---:|---|---:|---:|
| 400 | 114,144 | 920 s | 0.7773% (1.1214%) | 0 | 368.6603 |
| 500 | 114,144 | 1,322 s | 1.0993% (1.4017%) | 0 | 412.1138 |
| 750 | 158,260 | 3,614 s | 1.2569% (1.5165%) | 0 | 504.9214 |
| 750 | 114,144 | 2,502 s | 1.6928% (2.1026%) | 0 | 504.1626 |
| 1000 | 209,568 | 7,776 s | 1.3473% | 0 | 583.2558 |
| 1000 | 114,144 | 3,803 s | 2.2560% (2.8035%) | 0 | 583.1417 |

**What N=1000 settled.** Phase 1 stayed in the **hours, not days** — 2.16 h on
the finer mesh, 1.06 h mesh-matched — against a PGD incumbent that needed 57.3 h
for an *invalid* N=300 on a coarser mesh. And the **sub-threshold ladder rung
was survivable**: the mesh-matched run's level 0 sits at freeze ratio 0.897 with
9.6 v/cell, below 1.0, and still passed every gate, healing its 3 fragmented
cells across the ladder (`3→2→1→0→0`).

⚠ **The two N=1000 runs are not interchangeable.** The finer mesh gives the
better partition (worst cell 1.35% vs 2.26%); the common mesh gives the only
cross-N comparability. A coarser mesh also *under-measures* perimeter — 583.1417
against 583.2558, −0.020% — because a boundary of fewer longer segments is a
shorter approximation.

**Two things N=750 settled, both since confirmed at N=1000.**

1. **The ladder must be re-based as N grows, and the τ diagnostics say when.** At
   N=750 the standard `100x96` base gives 12.8 v/cell and a freeze ratio of
   **0.964 — below 1.0**, i.e. the coarse-triangle band cannot move. Dropping that
   rung (base `162x154`) fixed it *and* produced the first high-N run with **0
   fragmented cells at every level**, where N=300 and N=500 each had one at level
   0. Check the ratio before choosing a ladder; never fix a freeze by raising
   `tau_c`, which walks into the over-merge wall opposite.
2. **Phase 1 is no longer the constraint; Phase 2 is, and steeply.** See Phase 4.

A curve of wall time and validity versus N, out to where PGD cannot follow, is a
stronger argument than any perimeter margin — **and that curve now exists**, six
points from N=400 to N=1000 across two meshes, with the incumbent unable to
produce a valid partition anywhere on it.

The three things this phase said to watch all materialised as predicted: the τ
over-merge cap binds on more levels as cells shrink (0 at N=100, 1 at N=300, 2 at
N=400); the vertex-granularity floor rises until the finest mesh must grow
(0.56% at N=400/V=114,144 → 1.40% at N=1000 on the same mesh, which is why the
worst cell reaches 2.26% there); and peak memory — the dense `V × N` score matrix
plus two same-size transients in the assignment solver — reached **~5.0 GB at
N=1000** against ~1.5 GB at N=400.

**The remaining scaling question is no longer Phase 1.** Going past N=1000 is a
Phase 2 problem (below), not a Phase 1 one.

## Phase 4 — Phase 2 is the bottleneck, and the solver is not why
**Status:** **Partially measured — and this phase's original premise is REFUTED,
now at two values of N.** The N=500 and N=750 Phase 2 campaigns were profiled
(2026-08-22/23).

Measured Phase 2 wall: ~33 min at N=100 (≈14,900 variable points), ~50 min at
N=300, **122 min at N=400** (29,244 variable points). It scales worse than
linearly in variable points, and B has already made Phase 1 only **11%** of
end-to-end cost at N=400.

**At N=1000 it does dominate, as predicted.** This phase's estimate of "roughly
7–9 h of Phase 2 at N=1000" was written before the run; measured from the
campaign log it came in at **8.80 h** on the finer mesh (V=209,568) and **5.36 h**
mesh-matched (V=114,144) — the finer-mesh figure landing inside the predicted
band, near its top. Against Phase 1's 2.16 h and 1.06 h, **Phase 2 is 80% and
83% of end-to-end**. **This reverses the project's standing assumption that Phase
1 is the expensive stage**, and the reversal is now measured at every N from 400
up, not inferred.

**What the profile actually showed, and it is not what this phase assumed.** This
section used to say: profile Phase 2 and "decide whether the next target is the
IPOPT/exact-Hessian path". Measured at N=500 (`timing_profile.yaml`, campaign
`ipopt_btol0.001_lbfgs30_hess_bestiter_partial`):

| | N=500 | N=750 |
|---|---|---|
| whole Phase 2 campaign (iterate timestamps) | **8,829 s** (2.45 h) | **18,323 s** (5.09 h) |
| inside `optimizer.optimize()` | 459 s | 659 s |
| **solver share** | **5.2%** | **3.6%** |

IPOPT ran 201 solver iterations per topology iteration at *both* N, so it hits
`max_opt_iter: 200` every time, and the internal callback split is essentially
identical (Hessian 40.4% vs 40.7% of solver time). **The solver is a shrinking
share of a growing problem — optimising it buys at most 5%, and less as N rises.**
The exact-Hessian path is the wrong target.

**Phase 2 scales superlinearly in variable points, and that is now the binding
constraint on N:**

| | variable points | wall | s/VP | exponent |
|---|---|---|---|---|
| N=400 | 29,287 | 7,326 s | 0.250 | — |
| N=500 | 32,704 | 8,829 s | 0.270 | 1.69 |
| N=750 | 47,088 | 18,323 s | 0.389 | 2.00 |
| **N=1000** | **62,472** | **31,670 s** | **0.479** | **1.74** |

⚠ **The fourth point refutes "steepening"**, which the three-point version of
this table asserted. The exponents go 1.69 → 2.00 → **1.74**, so 2.00 was an
excursion, not a trend. The defensible statement is **superlinear, ≈1.7–2.0, with
no evident trend** — never "quadratic" and never "steepening". Note also that
these four points span *different meshes*; the first **fixed-mesh** pair
(N=750→1000 at V=114,144) gives **2.36**, which suggests mesh effects were
confounding the mixed-mesh figures and that a same-mesh series is the one to
trust. This is the second time a premise in this phase has been overturned by
the next data point.

⚠ **What is established is the solver's share, not the composition of the other
95%.** The candidates are contour rebuild, Steiner setup, migration detection, and
checkpoint export with its roundtrip verification — none of them measured. **The
revised Phase 4 task is to decompose that remainder**, which needs instrumentation
that does not currently exist (the profiler only wraps `optimize()`).

⚠ **`timing_profile.yaml`'s `summary.total_wall_s` is NOT campaign wall** — it is
the accumulated `optimize()` time, so it reads 459 s against the campaign's
8,829 s, a **19× under-report**. Same class as the `run_time_seconds` trap in
Phase 1 metadata (3.6×), which is already documented. Take campaign wall from
iterate timestamps. Any prior reasoning that used this field as campaign wall
should be re-checked.

## Phase 5 — Breadth of the evidence base
**Status:** **Partly done.** Geometry breadth exists; seed breadth does not, and
it is now the binding gap for a paper.

- **Seeds — the real gap, and unchanged.** At minimum 3 seeds per configuration
  for B; and — separately — measure **PGD's** own seed variance at the anchor
  configurations, without which "B beats PGD by more than PGD's own seed lottery"
  cannot be written. Every N=100 PGD run across both worktrees is seed 84172851
  and every N=300 λ=11.5 run is seed 61803399, so each anchor is still a single
  trajectory. **This is the cheapest remaining item that changes what may be
  claimed.**
- **~~A second surface.~~** ✅ **Done for B** (2026-09-02/07), further than this
  bullet asked:
  - **Torus geometry**, nine runs: aspect ratio R/r ∈ {1.400, 1.667, 2.500} at
    constant area, and size Rr ∈ {0.600, 1.200, 2.400} at fixed shape. All valid,
    all exported (`deliverables.yaml`, group `mc-study`).
  - **Two non-torus closed surfaces**: double torus (genus 2) and
    Banchoff-Chmutov order 4 (genus 5), N=10, all gates passing on raw labels,
    0 fragmented, 7 s and 17 s of Phase 1.

  ⚠ Two limits on that. The non-torus runs are **N=10 only**, so they show the
  method transfers to a surface, not that it transfers *at scale*. And there is
  **no PGD comparison available on them at all** — all 21 archived non-torus PGD
  runs fail the gates — so these are existence results, never margins.
  The practical lesson, worth a paragraph in any manuscript: on a marching-cubes
  mesh the ladder must be chosen by **stiffness conditioning** rather than vertex
  count, and λ cannot be calibrated from ‖g‖∞ there at all — which is itself an
  argument for B, since B has no λ.
- **A second machine**, to separate hardware from method in the timing claims.
  **Not done.** All timings in this document are one Mac mini.

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
| ~~Phase 2 becomes the wall at N=1000~~ | **Realised, and quantified:** 8.80 h against Phase 1's 2.16 h on the finer mesh — 80% of end-to-end. It did not redirect the programme, because N=1000 completed anyway; it redirects anything *past* N=1000 |
| Geometry breadth reads as broader than it is | Three aspect ratios and two non-torus surfaces exist, but the non-torus runs are **N=10** and have no PGD counterpart. Do not let "validated on genus-2 and genus-5 surfaces" stand next to "N = 1000" without the qualifier |

## Related documents

- Measured result: `docs/experiments/08-mbo-auction-dynamics/`
- **Derivation and provenance of B: `docs/math/10-mbo-auction-dynamics/`** — §2 is
  the prior-art table, Remark 6.1 the solver-substitution judgment
- **Paper scope in plain language: `docs/explanations/two_methods_explained.md`**
  §4 (what is ours) and §5 (how to position it)
- Reading path / corpus enumeration:
  `docs/explanations/reading_path_for_a_collaborator.md`
- The exported partitions and who consumes them:
  `docs/reference/deliverables.yaml`, `docs/reference/DOWNSTREAM_CONSUMERS.md`
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
