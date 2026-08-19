# Replacing Phase 1 at High N — Shared Harness, then B, then C

**Status:** **Phase 0 COMPLETE** — 0a ✅ (gates 1, 2 12/12, 3) · 0b ✅ · 0c ✅ · 0d ✅ bitwise. **B and C are both unblocked.** No new assignment solver needed.
**Revision:** v4 (2026-08-17), after three adversarial review rounds; see
[Correction history](#correction-history).

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

**Status:** 0b / 0c / 0d complete; 0a complete except the production configuration (see the corrected gate 2 result below)

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

#### ⚠ That conclusion was REFUTED by review (2026-08-16). Gate 2 was the wrong test.

The reading above — "the solver is structurally wrong; build an auction or
Sinkhorn" — is **withdrawn**. Three independent errors, all verified here:

**1. The fixtures modelled a state the arms occupy for one iteration.** Gate 2
posed a *cold start*: rebalance a raw geometric guess in one shot with ψ=0. B and
C are iterative, so from outer-iteration 1 onward each assignment starts from an
already-balanced partition. Measured on a minimal iterated C (Lloyd, 25 outer
iterations, existing solver, cold each time):

| | it 0 | it 1 | it 2 | its 3–24 |
|---|---|---|---|---|
| V=47,488/N=300 | 3.654% | 2.166% | **1.351%** | **1.35–1.63% — 23 consecutive iterations under the bar** |
| V=114,144/N=100 | 0.339% | 0.157% | 0.146% | 0.13–0.17% |

Iterated B likewise self-corrects *inside its own loop* with no solver change:
45.35 → 8.44 (it 7) → 2.02 (it 18) → 1.72% (it 19).

**2. Two knobs were declared dead without being swept.** Both pass, re-verified:

| Knob | Result at V=47,488/N=300 (bar 2.022%) |
|---|---|
| `dual_iters` 400 → 4000 | **1.996% — PASS**, in 135 s |
| `dual_eta0` 0.5 → **2.0** (400 iters) | **1.672% — PASS** |

The "10× budget costs ~20 min" figure was **operations counted, not time
measured**: the real cost is 33.9 ms/iteration, so 4,000 iterations is **2.3
minutes**. And the percentile sweep had been rising monotonically to its p50
endpoint — a boundary optimum read as "not a tuning problem" instead of as an
arrow pointing past the range swept.

**3. The bar is indefensible: approach A fails it on its own home scores.** From
readout metadata in this checkout, the shipped dual stage stalls at
`dual_worst_rel_dev` **2.1215%** (`run_20260806_123326`) against the 2.022% bar
this gate sets at that very mesh, and **2.0218%** (`run_20260808_191030`) where
the formula demands 1%. Gate 2 required foreign scales to beat the home scale —
while excluding the repair stage that produces A's actual 1.80% / 0-imbalanced
result. The formula also tracks the wrong variable: the stall is ~1.7–2.1% at
both V=47,488 and V=114,144 while granularity halves between them.

**4. The one-shot B fixture is unsolvable by any ψ-family method, including the
ones proposed to fix it.** At c=2 it sits in the freeze regime by this plan's own
τ criterion, and one diffusion step from a Euclidean-nearest init gives a
near-binary field (p50 margin 0.499 on a range ≤ 1), so too little mass sits in
flippable margins for `argmax(y + ψ)` to represent balance. Verified by
exhaustion: dual-400 = dual-4000 = 45.353%; η₀ swept 0.25–16 at two decays, all
≥ 38.4%; annealed **Sinkhorn 40.8%**. It never tested the solver.

**5. No new solver is needed.** Sinkhorn, implemented and measured, **loses to the
incumbent on 5 of 6 score matrices** (C@300 3.318 vs 1.996; C@100 0.563 vs 0.149).
Auction could in principle beat the ψ-family floor on the one-shot fixture, but
that is compiled-code project work built solely to pass a fixture modelling a
state B recovers from by itself.

**Corrected 0a remedy**, cheapest first:

1. **Fix the harness, not the solver** — score matrices from outer iterations ≥ 1,
   cold start reported separately as informational; bar calibrated per fixture
   against a strong-reference run and floored at A's measured home-scale stall;
   add the production pin **V=114,144/N=300** (never tested); more than one seed.
2. **Default a higher effective step when `normalize_scores` is on** (η₀ ≈ 2, or
   normalize by ~p10–p25 of the margin). Cold C then passes at 1.672%.
3. Early-stop at the bar, and `np.bincount` for `np.add.at` in the hot loop
   (1.28× free).
4. Expose `psi0` for in-loop callers **with a best-of-{ψ₀, 0} guard** — warm start
   is worth 60–100× per assignment once a trajectory settles, but is *harmful*
   early on B (measured 87–110% blowups from a stale ψ).
5. If B is built, initialize it from one balanced C-scores assignment: its first
   MBO assignment then starts at 2.9–3.6% instead of 45%.

**"B must not be built until gate 2 passes" is inverted:** gate 2 as posed must be
rewritten before it is allowed to block anything.

#### ✅ Gate 2, rewritten, now PASSES (2026-08-17)

Rewritten per the remedy above: iterated Lloyd (C) and MBO (B) fixtures scored on
*settled* outer iterations with the cold start reported as informational; B's
diffusion moved to c=4 out of the freeze regime; bar calibrated per fixture at
`max(2 × granularity, 1.25 × strong-reference)`; two seeds. Solver default
`normalized_eta0 = 2.0` when normalization is on.

| Fixture | seed | cold | **settled** | strong-ref | bar | |
|---|---|---|---|---|---|---|
| C, V=47,488/N=300 | 0 | 1.672% | **1.717%** | 1.358% | 2.022% | PASS |
| C, V=47,488/N=300 | 1 | 1.786% | **1.582%** | 1.413% | 2.022% | PASS |
| B, V=47,488/N=300 | 0 | 1.968% | **1.563%** | 1.398% | 2.022% | PASS |
| B, V=47,488/N=300 | 1 | 1.553% | **1.587%** | 1.354% | 2.022% | PASS |
| C, V=114,144/N=100 | 0 | 0.177% | **0.194%** | 0.132% | 0.280% | PASS |
| C, V=114,144/N=100 | 1 | 0.172% | **0.146%** | 0.175% | 0.280% | PASS |
| B, V=114,144/N=100 | 0 | 0.157% | **0.179%** | 0.147% | 0.280% | PASS |
| B, V=114,144/N=100 | 1 | 0.193% | **0.163%** | 0.169% | 0.280% | PASS |

**The incumbent solver was never the problem.** B, which the old fixture scored
at 45.35% and pronounced structurally unsolvable, sits at **1.55–1.59%** once
tested in the state it actually occupies — a fixture change, not a solver change.
The cold start is now unremarkable too (1.55–1.97% at N=300), because
`normalized_eta0 = 2.0` fixed the one real defect the original sweep had walked
up to and mistaken for a wall.

Every settled figure sits within ~15–25% of its own strong reference, so the
default 400-iteration configuration is close to what the score matrix admits.

**Final run — all 12 fixtures PASS**, three meshes × two arms × two seeds, with
the A-home-stall floor applied:

| Fixture | granularity | bar | C settled | B settled |
|---|---|---|---|---|
| V=47,488 / N=300 | 1.011% | 2.122% | 1.717% / 1.582% | 1.563% / 1.587% |
| V=114,144 / N=100 | 0.140% | 0.280% | 0.194% / 0.146% | 0.179% / 0.163% |
| **V=114,144 / N=300** (production) | 0.421% | 2.022% | 0.888% / 0.702% | 0.679% / 0.629% |

**Honest accounting of why the production pin flipped.** Two changes were made
after it failed, and only one of them mattered:

* **The A-home-stall floor was decisive.** Under the old bar (0.841%), C seed 0
  at **0.888%** would still fail. The floor is what changed the verdict.
* **Extra sampling did not rescue it.** Going from 3 outer iterations / 1 seed to
  4 / 2 moved seed 0 from 0.919% to 0.888% and revealed seed 1 at 0.702% — real
  seed spread, but seed 0 remains above the old granularity bar.

So the fairness of this result rests entirely on whether the floor is the right
bar, not on more samples.

#### ⚠ Third review (2026-08-17): the floor was wrong, and the gate was theatre

**Two withdrawals.**

**1. The "2–3× better than the incumbent" headline is withdrawn.** The floor used
A's stall *as recorded in the shipped runs* — 2.0218% / 2.1215% — which is A
running with `normalize_scores` **off**. The same solver, same log densities,
normalized, same 400-iteration budget, reaches **0.9072%** and **1.8384%**. The
recorded figures are a configuration artifact, so the floor let an arm clear a
bar the incumbent only fails because of a setting.

| Config | old bar | **corrected bar** |
|---|---|---|
| V=47,488/N=300 | 2.122% | 2.022% (2 × granularity binds) |
| V=114,144/N=100 | 0.280% | 0.280% |
| **V=114,144/N=300** | 2.022% | **0.907%** |

Honest restatement, **corrected again on 2026-08-18**: the arms and the incumbent
are at **parity, ±10%**. Measured at genuinely equal footing — A's own dual on its
own log densities, normalized, at the *same 2000-iteration budget the arms are
scored with* — A reaches **0.5991%** at the production pin, against C
**0.624 / 0.573%** and B **0.537 / 0.573%**. So C seed 0 is *worse* than the
incumbent and B is ~10% better. The earlier "clear of parity" reading came from
comparing arms at 2000 iterations against A at 400: a budget difference, not a
method difference. (A repaired-C figure of 0.349% appears earlier in this
document; it was measured by a review subagent and never self-reproduced — an
independent reproduction gives 0.298%. Treat it as indicative.) The substantive conclusion — the assignment step will not be the
bottleneck — survives every consistent choice of bar. The headline did not.

*Incidental, and free:* **approach A would halve its own dual stall by enabling
`normalize_scores`** — a real improvement to shipped production code, found while
testing something else.

**2. Gate 2's PASS/FAIL machinery was theatre.** Its bar included
`1.25 × strong-reference`, and the strong reference was computed **with the
solver under test** — so a broken solver inflated its own bar in proportion to
how broken it was. Six deliberately crippled solvers all passed. The weakest:
**one that returns ψ ≡ 0 and does no iterations at all**, scoring 45.4% (C) and
80.2% (B) against self-scaled bars of 57.2% and 124.3%.

Fixed: the strong reference is now reported as context and **never enters the
bar**; fixture-internal assignments use a **pinned reference config** so the
trajectory cannot degenerate along with a broken solver; and settled quality is
scored by **median, not min** — min ratchets monotonically with sampling (the
production pin's C seed 0 moved 0.919 → 0.888 → 0.799% purely by sampling more,
with a median of 0.919% and no convergence trend). One consequence to record: the
plan's earlier claim *"extra sampling did not rescue it"* is **false at 8
iterations**, where the min reaches 0.799%.

Also reconciled: the harness's own `suspect_solver_failure` fired at 2 ×
granularity (0.841% at the pin) while this gate declared 0a complete at 2.022% —
so in a real arm evaluation the fault flag would have fired about half the time
while the gate said the solver was fine. Both now read one number,
`assignment_quality_bar` in `arm_harness.py`.

#### Corrected gate 2 result: 11/12 pass, C fails at the production pin

| Fixture | bar | C (s0/s1) | B (s0/s1) |
|---|---|---|---|
| V=47,488/N=300 | 2.022% | 1.946% / 1.960% | 1.621% / 1.655% |
| V=114,144/N=100 | 0.280% | 0.213% / 0.194% | 0.181% / 0.175% |
| **V=114,144/N=300** | **0.907%** | **0.919% ❌** / 0.773% | 0.686% / 0.667% |

**C seed 0 at the production configuration misses by 0.012 points (1.3%).** That
0.919% is not noise — it reproduces the independent 8-outer-iteration median
exactly. C seed 1 passes at 0.773%; **B passes on both seeds at every fixture.**

**Under this plan's own area lane, this is a solver-configuration fault, not
evidence against C** — "worst deviation above the quality bar ⇒ fix 0a before
scoring the arm." The diagnosis is unambiguous: the *strong reference* on those
same scores (same solver, 1500 iterations instead of 400) reaches **0.685%**,
comfortably inside the bar. So the algorithm is fine and the default **budget**
is the binding constraint at production scale.

The pre-registered remedy already names the fix — earlier remedy item 3,
"early-stop at the bar, and raise the default budget" — which was never
implemented. Early stopping makes this *cheaper*, not merely longer: easy
fixtures halt as soon as they clear the bar, and only the hard production
configuration pays for extra iterations.

#### 🔒 PRE-REGISTERED PREDICTION (written 2026-08-17, BEFORE implementing)

Committed *before* the early-stop + budget change is written, so the result
cannot be re-explained after the fact. Rationale: of my three threshold/config
adjustments so far, the two made under outside review were correct and the one I
made myself after seeing a failure — the incumbent floor — was **wrong**. That is
a one-for-one bad record on self-directed post-failure adjustment, so this one is
pinned in advance instead of justified afterwards.

**The change:** `solve_dual_offsets` stops as soon as it reaches
`assignment_quality_bar`, and its budget for normalized scores is raised. The
**acceptance bar does not move** — it stays 0.907% at the production pin.

**Predictions, all falsifiable:**

| # | Prediction |
|---|---|
| P1 | C at V=114,144/N=300 **seed 0** settles at **≤ 0.78%** (from 0.919%) — passing |
| P2 | C at V=114,144/N=300 **seed 1** stays **≤ 0.78%** (from 0.773%) |
| P3 | B at V=114,144/N=300 lands **≤ 0.70%** on both seeds (from 0.686 / 0.667%) — essentially unchanged |
| P4 | C at V=47,488/N=300 **improves to ≤ 1.75%** (from 1.946 / 1.960%, against a 2.022% bar it was nearly touching) |
| P5 | Gate 3 still **rejects** the null solver (early-stop must not make a broken solver look converged) |
| P6 | Gate 1 still **byte-identical** — approach A must be untouched, since it does not use the normalized path |
| P7 | Total gate-2 wall time does **not** increase, despite the higher ceiling: the easy fixtures halt early |

**Falsifier.** If P1 misses — C stays above 0.907% at the production pin — then
the budget hypothesis is **wrong**, C has a genuine convergence problem at
production scale, and it must be reported as such rather than tuned at again. In
that case C is not scoreable on perimeter and the plan says so.

**P7 is the honest test of motive.** If this were really about making a failure
disappear, wall time would rise. Early-stop is supposed to make the common case
*cheaper*; that is why the change is B's infrastructure and not C's excuse.

#### ✅ FINAL — 18/18 under the corrected protocol (2026-08-18)

Re-run after review 4 with **all** of its must-fixes in place: the bar re-anchored
to equal footing (production 0.907% → **0.841%**), a **third seed**, grading from
the **returned labels** rather than the solver's self-report, and a genuine
4000-iteration strong reference set through the field the normalized path
actually reads. 2 h 39 m.

| Fixture | bar | C (s0/s1/s2) | B (s0/s1/s2) |
|---|---|---|---|
| V=47,488/N=300 | 2.022% | 1.472 / 1.392 / 1.495% | 1.405 / 1.390 / 1.406% |
| V=114,144/N=100 | 0.280% | 0.146 / 0.142 / 0.144% | 0.161 / 0.151 / 0.138% |
| **V=114,144/N=300** | **0.841%** | **0.624 / 0.573 / 0.609%** | **0.537 / 0.573 / 0.607%** |

The third seed lands inside the existing spread on every fixture, so the n=2
verdicts were not luck. The strong reference is now genuinely longer than the
tested config and mostly slightly better, so the earlier circularity is gone.

**Against the incumbent at equal footing** (A's own dual, normalized, same 2000
budget → **0.5991%** at the production pin): C spans 0.573–0.624%, B spans
0.537–0.607%. **Parity, ±10%** — B a little better, C a little worse. This is the
honest comparison and it is *not* the "2–3×" or "clear of parity" that two earlier
versions of this document claimed.

**What that means for the plan:** the assignment step is not a differentiator
between the arms and the incumbent, and it is not a bottleneck for either. It was
never supposed to be — 0a's job was to make the shared primitive usable on foreign
score scales so that **perimeter** can be measured. That is done.

#### ⚠ On "survived adversarial review"

Four adversarial reviews inform this document, and each found real defects that
changed it. They were **subagents dispatched from the same session as the work
they reviewed** — independent of the author's reasoning, not of the author's
framing, tooling, or choice of what to point them at. Every number they produced
that is load-bearing here has since been self-reproduced, except where explicitly
marked indicative. "Survived adversarial review" is doing evidentiary work in this
plan and a reader should know exactly how much weight it can carry.

#### ✅ RESULT vs the pre-registered prediction (2026-08-18) — 2 substantive hits, 4 guards held, 1 substantive miss

Early stop + raised budget, gate measuring at full budget. **12/12 PASS.**

| Fixture | bar | C (s0/s1) | B (s0/s1) |
|---|---|---|---|
| V=47,488/N=300 | 2.022% | 1.472% / 1.392% | 1.405% / 1.390% |
| V=114,144/N=100 | 0.280% | 0.146% / 0.142% | 0.161% / 0.151% |
| **V=114,144/N=300** | **0.907%** | **0.624% / 0.573%** | **0.537% / 0.573%** |

| # | Predicted | Actual | |
|---|---|---|---|
| P1 | C prod s0 ≤ 0.78% | **0.624%** | ✅ |
| P2 | C prod s1 ≤ 0.78% | **0.573%** | ✅ *(guard: status quo was already 0.773%)* |
| P3 | B prod ≤ 0.70% both | **0.537% / 0.573%** | ✅ *(guard: status quo was already 0.686/0.667%)* |
| P4 | C @47,488 ≤ 1.75% | **1.472% / 1.392%** | ✅ |
| P5 | gate 3 rejects null solver | 19.030% / 6.680%, rejected | ✅ |
| P6 | gate 1 byte-identical | exact | ✅ |
| P7 | wall time does not increase | **93 min vs ~41** | ❌ |

**The budget hypothesis is confirmed.** C at the production pin went 0.919% →
**0.624%**, clearing its bar on merit.

⚠ The original wording added *"and `settled ≈ strong-ref` on almost every row,
i.e. the solver is converging rather than being cut off."* **That was circular** —
`strong_cfg` set `dual_iters`, but since `6bb78cb` the normalized path reads
`normalized_dual_iters`, so the "strong reference" was the *identical config
under test*. The underlying fact survives on genuine long runs (production C at
4000 iterations → 0.619% vs 0.624% at 2000; B at 8000 → 0.532% vs 0.537%), but
the committed evidence for it was not evidence. `STRONG_ITERS` is now 4000 and set
through the field the normalized path actually reads.

**P7 failed, and the failure is instructive.** It was the one prediction whose
*mechanism* was wrong, and the first (early-stop-in-the-gate) run is why: with
early stopping on, the gate passed 12/12 while every number hugged its bar from
below — 0.885% at the production pin where the solver actually reaches 0.620% on
the same scores. The verdict was never wrong; the *numbers* were artefacts of the
stopping rule. Measuring honestly costs 93 minutes instead of 41. Early stopping
remains in the solver for **production** use, where B calls it every MBO step,
but it is barred from the measurement path.

**Had the prediction not been committed in advance, a 12/12 PASS would have been
banked and the masked numbers written into this document as fact.** That is what
the pre-registration was for, and it is the only reason the defect surfaced.

**Superseded.** This paragraph said 0a was incomplete at the production configuration and that C must not be scored there. Both were resolved by the budget fix above (C 0.919% -> 0.624%). Kept as a marker because it sat under a "Phase 0 COMPLETE" header for a day, contradicting it.

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
**≤ 0.01%**. If the harness cannot reproduce a known answer, no arm result from it
means anything.

#### ✅ 0d PASSES — BITWISE (2026-08-17)

`testing/test_phase0d_shakedown.py`. The N=100 control's own labels, hardened to
one-hot and pushed through the whole harness under the control's own campaign
`refinement.yaml`:

```
reference    : 185.2546144718457
harness      : 185.2546144718457   (iter 20 of 20)
relative diff: 0.000000%     bitwise: True     wall: 2023 s
```

Three things this settles, beyond "the instrument works":

1. **The apples-to-apples claim is exact, not approximate.** The arm path (hard
   labels → one-hot → Phase 2) and the control path (continuous field → Phase 2)
   land on the *same bits*. Hardening costs an arm nothing measurable.
2. **Phase 2 is bit-deterministic on this machine** — previously an open
   load-bearing assumption, listed as unverified in the round-2 review.
3. **⚠ The N=100 anchor is itself CENSORED.** The control's best iterate is its
   *last* (iteration 20 of 20), so 185.2546144718457 is a **bound, not a
   converged value** — the control was still improving when its cap stopped it.
   Under this plan's own censoring rule that makes the primary anchor a `≤`.
   The thresholds are unaffected in direction (a lower true value makes the bar
   *stricter*, never looser), but the anchor should be quoted as a bound.

## Phase A — B: auction-dynamics MBO

*(named to avoid colliding with the project's own "Phase 1 / Phase 2")*
**Status:** **Prototype built** on `feat/phase1-mbo-auction-dynamics` —
`src/partition/mbo_auction.py`, `scripts/run_mbo_arm.py`,
`testing/test_mbo_auction.py`. Gates and negative controls run first; **no scored
run until the pre-registration below is committed and NC3-CAL has pinned the
pinning detector.** See [Phase A pre-registration](#phase-a--pre-registration-b)
for the numbers committed in advance.

Threshold dynamics (Esedoğlu–Otto) with volume constraints by balanced assignment
(Jacobs–Kim–Léger, JCP 2018). **B descends the project's actual objective** — the
thresholding energy Γ-converges to perimeter — which is why the taxonomy ranks it
2, above C at 4.

Iterate: (1) diffuse each indicator for time τ via `(M + τK) y_k = M χ_k`, one
prefactorized solve per level (`scipy.sparse.linalg.factorized`; `sksparse` is
absent and unnecessary); (2) reassign by balanced thresholding
`ω(i) = argmax_k [y_ik + ψ_k]` via 0a.

**B's initialization is a HARD REQUIREMENT, not a convenience.** B must start
from **one balanced C-scores assignment**. Measured 2026-08-17: a
proposal-faithful cold start (seeded-Voronoi labels, unbalanced) leaves the
assignment at **68.16%** on the first diffused scores, against **1.97%** from a
pre-balanced start. Gate 2's PASS for B is *conditional* on this init — building
B without it invalidates the evidence behind it.

**τ — a two-sided window, not a floor.** Start at τ = (c·h)² with **c = 4** — not
c = 2, which this section's own non-freeze criterion below (c ≳ √(R_cell/h) ≈ 4.0
at N=100, 2.4 at N=300) excludes, and which gate 2's fixture work showed to be
the regime where balance is unrepresentable as `argmax(y + ψ)` by *any* method.
The earlier "c = 2" here contradicted the criterion two bullets under it. Report
per level:

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

### Phase A — pre-registration (B)

Committed **2026-08-19, before any scored run**, after four adversarial review
rounds on the design. Thresholds are not renegotiated after seeing results.

#### What the implementation pins, and why

- **τ = min((4·h_mean)², (ρ·R_cell)²), ρ = 1.0.** c=4, not 2. The cap is the
  over-merge side of a two-sided window; ρ=1.0 is the loosest defensible value
  (at √τ = R_cell the diffused indicator has spread over the whole cell) and any
  tighter ρ is a preference, so the over-merge **instrument** adjudicates instead.
  Measured across both ladders, the cap binds on **exactly one level**, N=300
  level 0 (c_eff 2.68). Both √τ/h_max and √τ/R_cell are reported per level.
- **One ruler.** Rule and window both use h_mean. The same criterion on h_max
  fails at N=100 L2/L3/L4 where h_mean fails only at L4, so **"L4 pins and nowhere
  else" is not derivable under either ruler** — an earlier draft mixed the two.
  Both are reported; the calibrated probe is the arbiter.
- **The initialization is a hard requirement** (one balanced C-scores assignment),
  since gate 2's PASS for B is conditional on it.
- **Early stopping is production-only and barred from the measurement path.** Every
  level ends with a full-budget re-assignment; in-loop figures carry an
  `early_stop_capped` tag and are never quoted as assignment quality. This
  reproduces report 07's artefact 5 by construction and designs around it.
- ⚠ **The ladder is read from the anchor run's `experiment.yaml`, never from
  `parameters/`.** `parameters/torus_300part_seeded_lam11p5_original_energy.yaml`
  says `refinement_levels: 5` (V=114,144), raised post-hoc for a resume, while the
  anchor `run_20260806_123326` used 3 (V=47,488) — verified on disk. The driver
  asserts `arm_final_V == anchor_V` and refuses to score otherwise.

#### Pinned numeric constants

Transcribed here for the record. **These were pinned as `MBOConfig` defaults in
commit `75040c8`, before any scored run**, so they were fixed in a committed
artifact even though this prose entry is later; they are not being chosen now.

`tau_c = 4.0` · `rho = 1.0` · `churn_tol = 1e-4` (fraction of V flipping per step)
· `patience = 3` consecutive steps · `max_iters = 200` per level ·
`anneal_factor = 1.5`, `max_anneals = 3` · `probe_factors = (2.0, 4.0)` ·
`nc3_K = 79.2482`, `nc3_f_min = 5.634e-03` (from NC3-CAL, commit `e1e6288`).

⚠ **No prediction is attached to `max_iters`.** My planning notes contained "no
level hits max_iters", but that was never committed, so it is **not** scored as a
prediction. `hit_max_iters` is reported factually per level; a level that reaches
the cap is recorded as a censored level, not as a passed or failed forecast.

#### ⚠ The monotonicity theorem does not transfer — a finding, not a bug

Esedoğlu–Otto monotonicity is a minorize–maximize argument needing (i) the *same*
symmetric PSD form in the energy and the assignment objective, and (ii) an *exact*
assignment. **Both fail here, for independent reasons.** The theorem's object is
`M A_τ = M(M+τK)⁻¹M`, symmetric PSD; our step maximizes `⟨χ',Dy⟩` with `D = diag(v)`
**lumped**, and `D A_τ` is not symmetric. And Jacobs–Kim–Léger get a theorem because
Bertsekas' auction is *exact*, whereas `solve_dual_offsets` is subgradient ascent
returning its best iterate. **This is the price of Phase 0's substitution**, made
correctly (Sinkhorn and auction both lost to the incumbent on assignment quality) —
paid here rather than there.

What survives holds by construction, needs neither concavity nor exactness, and is
computable every step:

> **⟨χ', D y⟩ − ⟨χ, D y⟩ ≥ Σ_k ψ_k (T_k − T'_k)**

The RHS is exactly the slack inexactness costs; it vanishes at perfect balance. So
`E_τ` is a **descriptive trend, never a gate**, and G4 gates the inequality instead.

**Measured 2026-08-19 by me at V=9,600**, four configurations (N∈{100,300} ×
early-stop/full-budget, 25/40 steps): slack `|Σψ(T−T′)|/⟨χ,Dy⟩` runs
**1.66e-4 to 1.45e-3**, five to seven orders of magnitude above the 1e-9 tolerance
an earlier draft proposed — so 1e-9 was never defensible. Inequality violation was
**exactly 0.00e+00** in all four. A review reported 4 and 5 `E_τ` increases in those
step counts; I did not reproduce them at V=9,600 (0 increases in all four,
including one run still churning at step 40).

**Resolved 2026-08-19 by G4, in the review's favour.** Re-run at **V=47,488 /
N=300** — 40 active steps — the loop produces **1 `E_τ` increase, +2.542e-05
relative**, against a measured slack of 2.075e-4 at that configuration. So the
increases are real, they are bounded by the slack exactly as the mechanism
predicts, and my failure to see them was a **fixture artefact**: at V=9,600 the
partition freezes before one appears. The earlier "unresolved" label is withdrawn.
This is the third time in this programme that a null result turned out to be a
fixture that could not produce the effect it was looking for.

#### 🔒 Predictions

| # | Prediction |
|---|---|
| P1 | N=100 Phase 2 perimeter: point **188.0 (+1.5%)**, 80% interval [185.5, 193.0]. Modal verdict **PARTIAL** |
| P2 | **Seed spread at N=100 < 0.5%**. If > 1%, the primary falsifier cannot be scored at ±1% and I report that instead of a verdict |
| P3 | N=100 area worst dev ≤ **0.28%** (bar); point 0.15% |
| P4 | N=100 fragmented ≤ **3**; point 0 |
| P5 | N=100 ladder ≤ **25 min/seed** vs 48,132 s ⇒ ≥ 30× |
| P6 | N=300 vs 323.319247: point **331.4 (+2.5%)**, 80% interval [323.3, 341.0] — upper end deliberately crosses the +5% line (339.5); ~10–15% on refutation |
| P7 | N=300 area ≤ **2.02%** at V=47,488, fragmented ≤ 6, ladder ≤ 15 min ⇒ ≥ 88×. *At N=300 level 0 granularity is 5.000% against a 10.000% bar (V=9,600, verified) — a large deviation there is granularity, not a solver fault* |
| P8 | Under any calibration separating frozen from converged, **≤ 2 of 5 N=100 levels flagged pinned**. The level list is **P8′**, committed after NC3-CAL, still before scoring |
| P9 | Over-merge shows as **fragmentation, not cell death** (balanced assignment cannot let a cell die); worst fragmentation over the N=300 ladder at **level 0** |
| P10 | (a) descent inequality violation ≤ **1e-12** relative at every **active** step; (b) lumped `E_τ` falls overall, excursions ≤ **max(2e-3, 1.5 × the run's own max slack)** on < 25% of active steps (see the amendment below); (c) anything above that is not dual slack and B is not scoreable until explained |
| P11 | NC1 and NC4 **fail** as designed; NC3-CAL separates frozen from converged with ≥ 2× margin (**or reports NC3 unusable**); NC2 then fires; **NC5 separates c=8 from c=4** on ≥ 1 of three instruments |
| P12 | Phase 2 completes all 20 (N=100) / 19 (N=300) topology iterations without aborting |
| P13 | **REPORTED WITH THE SAME PROMINENCE AS P1, never as a sub-clause** — it is the most likely way this experiment surprises us. **SF-a attribution:** `(P₂(init) − P₂(B))/P₂(init) ≥ 0.003`. If < 0.3%, the headline is *"the balanced geodesic init does the work"*, B's descent claim is unearned, and the number is a **preview of C** — which must then be said plainly rather than reported as B |

#### Amendment to P10(b), made 2026-08-19 BEFORE any scored run

The fixed **2e-3** ceiling was set as 1.5 × a **1.45e-3** slack measured over four
configurations. **The first new configuration run under G4 reached 1.868e-3**
(N=300, V=9,600, early-stop), putting the fixed bar within **1.07×** of what the
mechanism itself can legitimately produce — the same defect that sank the 1e-9
before it, one revision later and from the same cause: pinning a constant to the
largest number seen so far.

P10(b) now gates on **max(2e-3, 1.5 × the run's own measured max slack)**, with the
pre-registered 2e-3 retained as a floor and the binding term disclosed in the
output. This is not a loosening: P10(c) asks whether an excursion exceeds *what dual
slack can produce*, and a fixed constant was only ever a proxy for that quantity —
tying the bar to the measured slack tests the stated hypothesis directly. The
per-run slack is reported alongside every verdict, so a bar that moved can always
be seen to have moved.

#### G4's non-vacuity guard fired on its first run, as designed

At V=9,600 the N=300 configuration is 32 verts/cell and **froze after 12 of 40
steps**, so 28 steps of flat tail would have certified the gate. The guard failed it
(`got 12`, needs ≥ 20) while every substantive check passed. G4's N=300 fixture
moved to **V=47,488** (158 verts/cell), where the dynamics stay live past 40 steps.
Recorded because a gate whose guard has never fired is indistinguishable from one
that cannot fire.

#### Lane: Phase 2 aborting (decided in advance)

A2's arm made IPOPT abort with `EXIT: Restoration Failed!` at iteration 3 of 20.
An abort is **REFUTATION ON GEOMETRY** — reported with the iterate reached and the
best perimeter before it, and **explicitly not scored** against the +1%/+5% bars.
**One retry** is permitted, disclosed as "composed with E".

⚠ **E-for-arms must not receive a one-hot field.** `apply_balanced_readout` scores
by `log u`, so on one-hot input a flip needs ψ gaps > `|log 1e-300| = 690.8` and the
dual cannot move anything, while `rebalance_boundaries`' ranking is −690.78 for
*every* candidate — all ties. Repair strain would then be compared against control
bands (+1.88%/+2.77%) measured with a meaningful ranking. The retry therefore passes
the arm's **diffused y** as row-wise `softmax(y)` (so `log u_ik = y_ik + const_i`,
the constant cancelling in both argmax and dual), asserts the field is not one-hot,
and runs `normalize_scores=True` — which `scripts/balanced_readout.py` does not
expose, so the driver calls `apply_balanced_readout` directly.

Asymmetry disclosed either way: the **N=300 baseline came through the readout**, so
composing with E is like-for-like there; the **N=100 control did not**, so a
composed-with-E N=100 number is labelled as not like-for-like.

#### The over-merge instrument

Assignment quality **cannot** see over-merging — the balanced assignment hits its
area target by construction. (An earlier draft cited gate 2's healthy B fixture at
√τ/R_cell = 0.671 as evidence that ρ=0.5 was too conservative; that was an
assignment-quality measurement standing in for a geometry claim, the same class of
error as the theatre bar and the early-stop mask, and it is withdrawn.) Three
instruments, recorded per level and swept in NC5: **fragmented-cell count**;
**per-cell isoperimetric ratio** `Q_k = P_k²/(4πA_k)`; and **core loss** — cells
whose own diffused-score peak vertex lies outside their territory. If NC5 shows
c=8 ≈ c=4 on all three, the finding is *"no instrument here distinguishes
over-merging"*, not *"the risk is absent"*.

### Phase A — gate and negative-control results (2026-08-19)

Run on this machine by me; every number below is self-produced. **No scored ladder
run had been started when these were recorded.**

#### Gates: G1–G8 all PASS (257 s)

Two are worth recording beyond the verdict.

**G2 independently reproduces the plan's τ table** — 2.196 / 0.863 at V=9,600
N=100, 2.202 / 0.671 at V=47,488 N=300, 2.204 / 0.250 at V=114,144 N=100 — from a
recomputation that shares no code with `tau_diagnostics`.

**G4's non-vacuity guard fired on its first run and failed the gate.** The N=300
fixture at V=9,600 (32 verts/cell) froze after **12 of 40** steps, so 28 steps of
flat tail would otherwise have certified it. Moving the fixture to V=47,488 (158
verts/cell) gives 40/40 active and it passes. *A gate whose guard has never fired
is indistinguishable from one that cannot fire.*

| G4 config | active | worst inequality violation | max slack | E_τ increases |
|---|---|---|---|---|
| N=100, V=9,600, full budget | 40/40 | −4.314e-07 | 1.665e-04 | 0/40 |
| N=300, V=47,488, early stop | 40/40 | −2.914e-08 | 2.075e-04 | **1/40, +2.542e-05** |

**G7 earns its place:** a 2× τ error is rejected at 1.63e-01 relative, while mass
conservation is *exactly* τ-blind (`1ᵀK = 0`, verified at 2.5e-15) — G1, G2 and NC4
would all pass a mis-scaled τ.

#### Negative controls

| | Result | |
|---|---|---|
| **NC1** cold init | 45.55% raw Voronoi → 50.67% after one diffusion → **50.6742% after balancing** vs **1.6156%** warm (bar 2.0216%) | **PASS**, 31.4× |
| **NC4** wrong operator | 4.060e+02 vs correct 3.577e-15 | **PASS** |
| **NC3-CAL** | frozen 260.72 vs converged 24.09 dE/tail ⇒ `nc3_K = 79.2482`, `nc3_f_min = 5.634e-03`, **10.82× margin** | **PASS** |
| **NC5** over-merge, 158 v/cell | no instrument separates c=8 from c=4 | **FAIL — reported as pre-registered** |
| **NC5b** over-merge, 32 v/cell (post-hoc) | same, to √τ/R_cell = **2.99** | **null** |
| **NC2** c=2 freeze, V=114,144 | flagged **PINNED** (24 steps, 24 active, 88 s); moved_frac 4.60e-03 / 9.76e-03, dE/tail +42.99 / +89.82 | **PASS** |

**NC1 is stronger than it was designed to be.** The concern was that it is
near-tautological — that it only shows the first step is worse. What it actually
shows is that the balanced assignment **cannot repair a cold start at all**: the
solver returns 50.6742% against a raw argmax of 50.67%, i.e. its ψ=0 iterate, so
the dual moves *nothing*. The hard init requirement is not a convenience.

**NC3-CAL confirms the review's objection quantitatively.** A bare "labels moved
AND E_τ dropped" sign test fires on **both** states — the converged one moves 31–39
vertices with ΔE_τ +8.9e-03. Only the threshold separates them, and the 4× rung
carries it (frozen 154 → 261 between 2× and 4×), so the geometric ladder earned its
place. ⚠ The `moved_frac` half of the AND separates by only **1.9×** against
dE/tail's 10.8×; the discrimination is carried by dE/tail alone, and a pinned level
with small motion could be missed.

#### ⚠ NC5 + NC5b: the over-merge side of the τ window is not demonstrable here

| level | v/cell | √τ/R_cell at c = 2 / 4 / 8 | fragmented | Q̄ | Q_worst | boundary |
|---|---|---|---|---|---|---|
| V=47,488, N=300 | 158 | 0.335 / 0.671 / 1.341 | 0 / 0 / 0 | 1.6149 → 1.5649 → **1.5543** | 2.036 → 1.928 → 1.874 | 189.71 → 186.79 → 186.18 |
| V=9,600, N=300 | 32 | 0.747 / 1.495 / **2.990** | 0 / 0 / 0 | 1.6269 → 1.5949 → **1.5770** | 2.381 → 2.194 → 2.174 | 190.26 → 188.37 → 187.33 |

Across a **9× span of √τ/R_cell** (0.335 → 2.990) and two levels differing 5× in
resolution, **no over-merge signature appears on any of the three instruments**, and
core loss is 0 everywhere. The pre-registered signature was Q̄ / fragmentation /
core-loss rising with c; every one is flat or **falling**.

Two readings, and the difference matters:

- The instruments are **not blind** — Q̄ moves 3.8%, Q_worst 8%, boundary 1.9%
  monotonically with τ. A blind instrument would not move.
- But they move in the direction of **improvement**, so what is established is that
  *no damage is detectable*, not that over-merging cannot occur. **The pre-registered
  verdict stands as written: no instrument in this plan distinguishes over-merging.**

**Consequence for ρ, stated without acting on it.** The `ρ = 1.0` cap is
**unsupported by any measurement here** — it binds only at N=300 level 0, and at
exactly that level c=8 (√τ/R_cell = 2.99) is better than c=4 on every instrument.
**The cap is NOT removed**: it is pre-registered, and changing a constant after
seeing results is the tuning this whole apparatus exists to prevent. It is also
*conservative* — it lowers c to 2.68 at that level, and c=2 is measurably worse
than c=4, so keeping it biases against B, which is the safe direction. N=100 is
unaffected (no level triggers the cap there). Recorded as a follow-up, not a change.

Likewise the boundary proxy hints c=8 would give a shorter Phase 2 starting
boundary (186.18 vs 186.79, 0.33%). **c stays at 4.** Noted as a follow-up.

**A mechanism consistent with these data, not established by them:** the *volume
constraint* is what makes MBO robust to large τ here. All N indicators are diffused
equally, so the relative ordering of `y` still localizes each cell near its core,
and ψ then holds every area at target no matter how far the field has spread. That
is the property auction dynamics was introduced for; if it is what is happening,
the over-merge side of the window is much less binding on a *volume-constrained*
scheme than on plain thresholding. Testing that is a separate experiment.

#### ⚠ Ex-ante concern about NC3-CAL's `f_min`, recorded BEFORE NC2 returned

`nc3_f_min` is an absolute *fraction of vertices*, calibrated at V=9,600. The
vertices a τ-probe can move are boundary vertices, which scale like `1/h ~ √V`, so
their **fraction** scales like `1/√V`. From V=9,600 to V=114,144 that is a factor
`√(114144/9600) = 3.45` smaller: the frozen state's 7.81e-03 would land near
**2.3e-03**, *below* the pinned `f_min = 5.634e-03`. Since the detector ANDs
dE/tail with moved-fraction, a genuinely frozen fine level could be reported as
NOT pinned — a false negative created by the threshold, not by the dynamics.

Written down before the result so the diagnosis cannot be retrofitted. If NC2 does
report not-pinned, the correct reading is **the detector's `f_min` is not
mesh-transferable**, not "the finest level is healthy".

**Outcome: NC2 PASSES, and the prediction MISSED.** c=2 at V=114,144 is flagged
pinned (24 steps, 24 active, 88 s), with moved_frac **4.60e-03 at 2×** and
**9.76e-03 at 4×** — the 4× rung clears `f_min` comfortably, where I predicted
~2.3e-03 and suppression.

The mechanism I reasoned from was real but I applied it to the wrong quantity.
NC2's state is far *more* frozen than the calibration's: `c/c_lo = 2/4.003 = 0.50`
against the frozen anchor's 0.929. A deeper freeze leaves more reachable
improvement, so the probe moves more vertices — and that effect outweighed the
`1/√V` shrinkage I had predicted. **The 4× rung is what carried it** (4.60e-03 →
9.76e-03), the second time in this work that the geometric ladder was load-bearing.

⚠ **This does not retire the concern for P8′.** NC2 runs at c=2; the *scored*
N=100 ladder runs at c=4, where every level sits at `c/c_lo` between 1.00 and 1.86
— near the boundary, not deeply frozen — so probe motion will be far smaller and
`f_min` can still suppress. The concern stands where it was raised.

#### 🔒 P8′ — committed after NC3-CAL, before any scored run

The calibration gives two anchors on the level's own freeze ratio `c / c_lo`
(h_mean ruler): **frozen** at 0.929 → dE/tail 260.7, **converged** at 1.858 →
24.1. Interpolating `log(dE/tail)` linearly between them puts the pinned threshold
`K = 79.25` at **c/c_lo ≈ 1.394**. The N=100 ladder at c=4 sits at:

| level | V | c_lo | **c/c_lo** | vs threshold 1.394 |
|---|---|---|---|---|
| L0 | 9,600 | 2.153 | 1.858 | above — not flagged |
| L1 | 24,948 | 2.735 | 1.462 | above — not flagged |
| L2 | 47,488 | 3.214 | **1.244** | below — flagged |
| L3 | 77,220 | 3.630 | **1.102** | below — flagged |
| L4 | 114,144 | 4.003 | **0.999** | below — flagged |

**P8′, stated as two predictions because the detector has two clauses:**

- **On dE/tail alone: L2, L3 and L4 flag — 3 of 5.** That would **falsify P8**,
  which committed to "at most 2 of 5". I am recording the falsification of my own
  prediction in advance rather than quietly widening it.
- **On the pre-registered AND (dE/tail *and* moved-fraction): 0 or 1 flag**,
  because `f_min` is an absolute vertex fraction that shrinks like `1/√V` (the
  ex-ante concern above), so it suppresses exactly the fine levels dE/tail flags.

Both verdicts are reported per level.

##### 🔒 P8 scoring rule — committed BEFORE the first scored run

Split explicitly, because as previously written "untestable" could have absorbed an
ordinary miss, and that is the one thing this protocol cannot allow:

- **The two clauses AGREE and more than 2 of the 5 N=100 levels are flagged ⇒ P8
  FAILED.** Recorded plainly as a miss. Not reinterpreted, not widened, not
  reported as untestable.
- **ONLY if the two clauses disagree about WHICH levels are pinned** is P8
  *untestable as written* — and then both clause-wise level lists are reported and
  the finding is that the detector is not mesh-transferable.

A missed prediction is a fine outcome and often the most useful one: report 07's
P7 missed, was recorded as a miss, and was the single most informative result of
that phase. P8′ already predicts P8 will fail; it is allowed to fail cleanly.

#### ⚠⚠ HEADLINE NULL: a three-part over-merge instrument was built and it detected NOTHING

Stated here at full prominence rather than inside the results table, because its
consequence is a standing limit on what this plan's evidence can support.

Fragmentation, per-cell isoperimetric ratio, and core loss were all implemented and
swept over a **9× span of √τ/R_cell (0.335 → 2.990)** at two levels differing 5× in
resolution. **Not one of them registered any over-merge penalty.** Core loss and
fragmentation were 0 in all six configurations; compactness moved monotonically the
*other* way.

**Therefore the ρ question is OPEN, and no future claim about `c` or `ρ` may lean on
this plan's evidence in either direction** — not "over-merging is absent" (never
shown), and not "the cap is needed" (never shown either). The cap is retained
because it is pre-registered and because it biases *against* B, which is the right
direction to be wrong; that is a discipline argument, not an empirical one.

#### 🔖 NAMED FOLLOW-UP: test removing the τ cap — the current evidence says it may cost perimeter for nothing

Recorded as a named experiment so it survives, and stated at full strength because
it is stronger than a hint. **Two measurements point the same way:** NC5 found no
over-merge penalty at c=8 on *any* of the three instruments, and c=8 starts Phase 2
from a **0.33% shorter label boundary** (186.18 vs 186.79) — **a third of the entire
+1% success bar**. The best evidence currently available is that the ρ=1.0 cap may
be costing perimeter for no measured benefit.

**This must not touch the present run.** `c = 4` and `ρ = 1.0` are pre-registered;
changing either after seeing NC5 is precisely the tuning this apparatus exists to
prevent. The follow-up is a separate arm, scored through the same harness.

### Phase A — SCORED RESULT 1 of 4: N=100, seed 84172851 (2026-08-19)

`results/arm_mbo_20260819_173411_npart100_V114144_seed84172851`. Scored by
`scripts/score_mbo_arm.py`, which applies the committed thresholds mechanically.

| level | V | steps | active | worst area | Q̄ | wall |
|---|---|---|---|---|---|---|
| L0 | 9,600 | 25 | 22 | 1.8068% | 1.558 | 5.3 s |
| L1 | 24,948 | 43 | 42 | 0.4950% | 1.547 | 23.0 s |
| L2 | 47,488 | 18 | 18 | 0.3304% | 1.557 | 36.5 s |
| L3 | 77,220 | 15 | 12 | 0.2095% | 1.560 | 61.1 s |
| L4 | 114,144 | 15 | 15 | **0.1504%** | 1.558 | 96.9 s |

**Gates on RAW labels: all clean.** 0 dead / 0 weak, 0 imbalanced (worst 0.1504%
vs a 0.2803% bar), **0 fragmented**. Re-verified through the independent path
`testing/check_fragmentation.py`, which additionally reports **0 cells with more
than one raw component including sub-threshold speckle** — the "compose with E"
lane is unused, on a method whose connectivity the literature does *not* guarantee
per step.

**Phase 2: 184.4117703525215 at iterate 18 of 20, NOT censored, full 20 iterates,
no abort.** Against the anchor 185.2546144718457 that is **−0.455%**, at **228 s**
of Phase 1 versus 48,132 s — **210.9×**.

⚠ **State this as "at equal Phase 2 budget", not "B beats PGD".** The anchor is
**censored** — the control's best iterate is its *last* (20 of 20), so it was still
improving when the cap stopped it and its converged value is a **bound**,
≤ 185.2546. B's best is iterate **18** of 20, so B had plateaued while the control
had not. The defensible claim is therefore: *at the same pinned campaign and the
same 20 topology iterations, B finishes 0.455% shorter, having converged where the
control had not.* Whether PGD would pass B given more Phase 2 iterations is
**unmeasured**, and the asymmetry cuts both ways — B's own convergence is evidence
for it, the control's censoring is evidence against.

**Named follow-up:** re-run the control's Phase 2 past 20 iterations to see whether
it drops below 184.4118. That is a *new* measurement against a pre-registered
anchor, so it does not belong in this run, but it is the one experiment that would
settle the comparison.

| # | Predicted | Actual | |
|---|---|---|---|
| P1 | 188.0 (+1.5%), modal **PARTIAL** | **184.4118 (−0.455%), SUCCESS** | ❌ **MISS, in B's favour** |
| P3 | ≤ 0.28%, point 0.15% | **0.1504%** | ✅ |
| P4 | ≤ 3, point 0 | **0** | ✅ |
| P5 | ≤ 25 min, ≥ 30×; point 15 min / 53× | **3.8 min / 210.9×** | ✅ bound; ❌ point estimate 4× pessimistic |
| P8 | ≤ 2 of 5 flagged | **0 flagged, clauses agree** | ✅ **HELD** |
| P10 | viol ≤ 1e-12; excursions ≤ ceiling on < 25% | viol **−3.53e-09**; **0** increases / 109 active steps | ✅ |
| P12 | 20 iterates, no abort | **20, no abort** | ✅ |

#### Two of my own predictions missed, and the second one matters more

**P1 missed and I was wrong about the mechanism.** My rationale was that "B only
ever sees perimeter through a coarse-τ surrogate, PGD saw the Γ-energy for 13.4 h".
That reasoning is refuted: B is *better*. Note also that B **enters** Phase 2 from a
*longer* boundary and still finishes ahead — measured directly rather than inferred:

| labels | `label_boundary_length` | vs control |
|---|---|---|
| control PGD (argmax of `x_opt`) | **107.3410** | — |
| B (MBO) | 107.6014 | **+0.243%** |
| init, i.e. C step 0 | 116.1989 | +8.252% |

(An earlier draft *inferred* the control's figure as 214.34/2 from its recorded
Phase 2 `initial_perimeter`. Computing it directly gives a ratio of **1.9968**, so
the inference was right to 0.16% — but it is now measured, not assumed.)

So B starts **0.243% behind** the control and finishes **0.455% ahead**: a genuine
crossover. The advantage is therefore not in B's initial geometry but in the
*combinatorial structure* Phase 2 is handed. **That is a hypothesis consistent with
these data, not a measurement** — nothing here isolates topology from geometry.

MBO also shortens its own init's boundary by **7.399%** (116.1989 → 107.6014),
which is direct evidence that the MBO loop does substantial work rather than
inheriting C's answer. The boundary proxy is *not* the scored quantity, so this
anticipates P13 without settling it.

**P8′ missed, and it invalidates the model I built it on.** P8′ predicted the dE
clause would flag L2, L3 and L4 — falsifying P8 — from a log-linear interpolation
of `dE/tail` in the freeze ratio `c/c_lo` between the two NC3-CAL anchors.
**Measured, `dE/tail` is essentially flat across the whole ladder**: 15.76, 21.62,
15.11, 14.22, **16.49** for L0…L4, against a threshold of 79.25 — and L4, sitting
exactly at `c/c_lo = 0.999`, scores 16.49 where my model demanded ~260.

So **the freeze ratio does not predict pinning here**, the two-point interpolation
was unjustified, and P8 held on the merits (both clauses empty ⇒ they agree)
rather than by the `f_min` suppression I expected. The `f_min` mesh-transfer
concern is therefore *untested* by this run, not resolved: nothing reached the dE
clause for `f_min` to suppress.

⚠ **P10's amendment earned itself.** Max slack was **1.508e-03**, so the adaptive
ceiling was 2.261e-03 and the pre-registered 2e-3 floor would have sat at only
1.33× the mechanism's own output — the same defect, one revision later, had it not
been amended before scoring.

⚠ **This result is not attributable to B until SF-a returns.** B's init is
approach C's step 0; if the init alone reaches ~184.4, this is a C result reported
under B's name. P13 decides it and is running.

### Phase A — how the N=300 secondary must be read

Verified before the run: the anchor `run_20260806_123326`'s own `experiment.yaml`
gives **3 levels → V = 47,488**, matching its solution exactly, while the live
`parameters/torus_300part_seeded_lam11p5_original_energy.yaml` gives **5 levels →
V = 114,144**. So the MF3 trap is not hypothetical here — it is the actual
difference between a mesh-matched comparison and one biased in B's favour, and the
driver's `arm_final_V == anchor_V` assertion is what prevents it. Campaign:
`readout/dualshift_gate0.05_repair/...`, `max_iterations: 19`, seed 61803399.

⚠ **The N=300 baseline is PGD *plus* the balanced readout (A + E), not raw PGD**,
because raw PGD fails its own gates at N=300 (10 imbalanced, worst 36.15%, 2
fragmented). B's **raw** labels are therefore scored against *the best available
PGD pipeline*. That asymmetry **disfavours B** and is the correct comparison to
make — but it must be stated whichever way the number falls, and it means a B
result at N=300 is not comparable in kind to the N=100 one, where the control
needed no readout.

#### Deferred, as a named follow-up

`docs/experiments/08-mbo-auction-dynamics/` is written **once B's verdict is known
and reviewed** — report 06's headline was refuted and report 07's corrected three
times, twice in the arms' favour, so LaTeX at the moment of first result is the
most expensive artifact to revise. **This is a deferral, not a cancellation.**

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
