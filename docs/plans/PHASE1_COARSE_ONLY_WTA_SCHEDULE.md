# Future Plan — Coarse-Only WTA Schedule (staged energy across mesh levels)

**Run the expensive territory-aware machinery on the coarse levels only *until the discrete
structure is correct* — a data-driven, gate-conditioned switch, not a fixed level — then refine
with the cheap original energy. Keeps the winner-take-all fix while making it scale to N≈1000.**

*(Naming note: "coarse-only" is shorthand. The expensive work is confined to however many coarse
levels it takes to reach a correct structure — one at low N, more as N grows. The switch is
decided by measurement, never by a hardcoded level number.)*

**Status:** PROPOSED / future work — now **unblocked**: the precondition is met. The
territory-aware term has been shown to *work* (`docs/experiments/04-territory-aware-highn-validation/`,
2026-07-20): on the bad seed and λ it drives the runt from −34% to **0 imbalanced cells**
(worst ±2.1%). This is the intended next experiment. **The measurement refines the design
(see §4/§5): the switch to the cheap energy must come after the first level that clears the
gate — level 1 at N=200, not level 0** — because the coarsest level is resolution-floor-limited
and cannot cross the 5% gate on its own. Best made a *gate-conditioned* switch (keep the
expensive machinery until `detect_area_imbalance` reports 0, then switch).

**Provenance / origin:** emerged from analysis during the first end-to-end N=200 territory
test (`run_20260717_102306`, config `parameters/torus_200part_coarse_seeded_lam9_territory_test.yaml`,
N=200, λ=9, seed 84172851, 3 levels), 2026-07-18, whose level-0 relaxation ran the full
30,000-iteration cap (~9.5 h) without triggering — the first time any run reached the cap on
the coarsest mesh. Numbers below are from that run's log and the diagnosis docs.

**Pairs with:**
- `docs/plans/PHASE1_TERRITORY_AWARE_IMPLEMENTATION_PLAN.md` — the executable spec for the
  term/trim/optimizer fix this plan schedules across levels (call it "Phase 1: all-levels";
  this is "Phase 2: coarse-only").
- `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` — the diagnosis (accounting identity, the √N
  band-fraction scaling law, the frozen-optimizer finding) this plan relies on.
- `docs/plans/PHASE1_N1000_SCALING_PLAN.md` — the *performance* wall (GPU/sparse/exact
  projection). This plan is a **complementary, algorithm-level** speedup that reduces total
  work regardless of backend; the two compose.
- `docs/reference/winner_take_all_partition_gap.md` §9 — the empirical high-N picture.

---

## 1. Why this matters (the end goal and the wall)

The partitions feed **molecular / numerical simulation on the torus**, which needs partitions
of **order N ≈ 1000+ cells** to be scientifically useful. N=300 is the current wall; the
territory-aware energy term (a winner-take-all "balance" penalty + a discrete-area trim + a
projected-gradient/trigger fix) was built to close the high-N validity gap. Early evidence
(the N=200 test) is that it *unfreezes the optimizer and does genuine, sustained descent* —
but at a **large computational cost**, and that cost is the thing that would make it
impractical at N=1000 if left as-is. This plan is the algorithmic move to make the fix
affordable at scale.

---

## 2. Why it is slow — the technical / optimization background

Three compounding causes, all measured or derived on the N=200 test run:

1. **First-order PGD crawls on the harder landscape.** The winner-take-all balance term (and
   the discrete-area trim) add coupling to the objective; projected gradient descent is a
   first-order method with a slow (linear) convergence rate. Level 0 (9,600 vertices) ran the
   full **30,000-iteration cap without converging** — energy still descending (sawtoothing
   between ~409 and ~433 over the last few thousand iterations, net decline only ~3–4 units
   per 1,000 iterations). It did **not** trigger early because the P2 fix removed the
   *frozen-optimizer* premature trigger that used to fire at ~48 iterations (see
   `PHASE1_N1000_VALIDITY_PLAN.md` §2.3): the descent is now genuine, just slow.
2. **Higher per-iteration cost.** Each iteration now pays for the extra O(V·N) WTA gradient
   term *and* the P2 reduced-gradient dual solve (Gauss–Seidel sweeps). On the N=200 test:
   ~1.2 s/iter at level 0 (9,600 verts) rising to ~3.4 s/iter at level 1 (24,948 verts).
3. **The cost compounds on finer meshes.** Because the expensive machinery runs on *every*
   level, the fine levels — which hold the overwhelming majority of the vertices and thus of
   the compute — pay the full expensive-term + dual-solve cost per iteration, at 2.6×–5× the
   vertex count of level 0. This is the dominant term in the total wall time.

Also relevant: the **trim's retargeting cadence** (default every 200 iters) makes the energy
a moving target (the sawtooth), which both adds re-descent overhead and prevents a clean late
plateau so the level runs to the cap. That is partly tunable (see the implementation plan's
follow-ups) but is orthogonal to the structural point below.

---

## 3. Why it is relevant from a computational-time perspective (the arithmetic)

The N=200 test, extrapolated from its measured per-level rates:

| Level | Vertices | ~s/iter (measured) | If it needs ~30k iters | Share of total compute |
|---|---|---|---|---|
| 0 (coarsest) | 9,600 | ~1.2 | **~9.5 h (measured, capped)** | small |
| 1 | 24,948 | ~3.4 | ~28 h | large |
| 2 (finest, 3-lvl) | 47,488 | ~6–7 (est.) | ~2+ days | largest |

A full 3-level N=200 run projects to **~2–4 days**. At N=1000 under the mesh-budget policy
(hold vertices-per-cell fixed ⇒ V ∝ N ⇒ finest meshes ~0.6–1M vertices;
`PHASE1_N1000_VALIDITY_PLAN.md` §5 / `PHASE1_N1000_SCALING_PLAN.md`), this is **weeks per run**
before any GPU/sparse work — untenable.

**The key observation:** almost all of that cost is the expensive machinery running on the
*fine* levels. If the expensive work could be confined to the **cheapest** mesh (level 0),
and the fine levels run the **fast original energy**, the total collapses toward "one
expensive coarse solve + cheap fine refinements" — which is affordable at N=1000. That is the
entire point of this plan.

---

## 4. The enabling insight — balance is a coarse-level property

The winner-take-all imbalance is **decided at the coarse level**, so the expensive corrective
force is only needed there:

- **The runt is manufactured at the coarse level.** It forms during condensation on level 0,
  where the band fraction `f_b = 3.72/√(2V/N)` is *largest* (level 0 at N=200: ~48 verts/cell,
  f_b ≈ 20%). See `winner_take_all_partition_gap.md` §5 (the runt is made during relaxation,
  not the init) and §9.
- **Finer levels have a smaller intrinsic gap** (more verts/cell → smaller f_b) *and*
  warm-start by interpolation from the coarse structure — so they have far less imbalance to
  begin with and start from an already-formed partition.
- **The original energy at finer levels does not manufacture runts — it reduces imbalance via
  resolution.** Direct evidence: N=150 ran the *original* energy at all levels and the
  worst-cell deviation *shrank* with refinement, −10.6% (2 lvl) → −5.84% (3 lvl) → −1.24%
  (5 lvl). Refining a good structure with the cheap energy improves balance; it does not
  break it.

**Chain of reasoning:** fix the runt on level 0 with the expensive term → the fine levels
never inherit a runt → the cheap original energy just sharpens a clean, balanced structure →
resolution preserves/improves the balance. This is a coarse-to-fine / nested-iteration pattern
(cf. multigrid nested iteration): do the globally-corrective, expensive work on the coarse
grid where the structure is set, and use the cheap smoother on the fine grids.

**Measured refinement (2026-07-20, `run_20260717_102306`; `docs/experiments/04-…`).** The
"fix it on level 0" framing above is *too aggressive* — the coarsest level cannot fully fix
the runt. Recomputed per-level WTA imbalance: **level 0 floors at ~10% worst deviation** (twice
the 5% gate) and flatlines there (tail rate 0.23%/1000it, 4 cells still over gate at the 30k
cap) — it is **resolution-floor-limited** (`f_b ∝ 1/√(V/N)`; 48 verts/cell → ~10% floor).
**Level 1 (124 verts/cell) is the first level that crosses the gate**, and it does so *steeply*
(2.2%/1000it, 0 imbalanced by ~iter 9000, floor ~2%). Level 2 then started at 0 imbalanced and
held — corroborating the §4 claim that the cheap energy preserves a balanced structure. So the
corrected chain is: **run the expensive machinery on the coarse levels until a level clears the
gate (level 1 here), then the cheap energy carries the balanced fine levels.** Refinement is
*necessary* (the gate lies between the level-0 and level-1 floors), not merely faster. Bonus:
~124 verts/cell sufficed to cross the gate — the mesh-budget floor (§5 of the validity plan)
may be ~250–300, not ~600, a 2–4× compute rebate at N=1000 worth confirming.

---

## 5. The proposal — an adaptive, gate-conditioned energy schedule

**The switch is decided by the measured structure, not by a level number.** The expensive
machinery stays on as long as the winner-take-all structure is *not yet correct*, and hands off
to the cheap energy once it is — wherever in the level ladder that happens. "Level 1 at N=200"
(§4) is the *instance* this rule produced there, not the rule. As `N` grows the band fraction
grows (`f_b ∝ √N`), so a correct structure is only reachable on a finer mesh — and the rule
**automatically** keeps the expensive machinery on for more levels, with no config change. The
cost is spent exactly where it is needed, tied to the resolution-floor law.

**The rule.** Carry a runtime mode `wta_active`, starting **on**. After each level completes
(before refining to the next), evaluate `detect_area_imbalance` on the winner-take-all state and
read its worst-cell deviation `w = max|dev|`; then choose the *next* level's mode with a
hysteresis band:

- **Stay ON (expensive)** while `w ≥ switch_margin`. *"Still far from a correct structure ⇒
  keep optimizing with the not-cheap method."* This is the default; the machinery simply runs
  level after level until the structure is ready. It naturally handles a **floor-limited coarse
  level** (§4): level 0 plateaus at its ~10% floor (never `< switch_margin`), so the rule keeps
  the machinery on for level 1 — no special case, no level number.
- **Switch OFF (expensive → cheap)** once `w < switch_margin`, i.e. comfortably *below* the 5%
  gate with headroom (default `switch_margin ≈ 3%`, ≈0.6× the gate — not merely *at* the gate,
  so a cheap level has room before it could drift back over). *"We are ready ⇒ hand off."*
- **Re-ARM (cheap → expensive)** if a cheap level ends with `w > rearm_threshold`
  (default = the gate, 5%) — a drift-back safety net. The hysteresis band `[switch_margin,
  rearm_threshold] = [3%, 5%]` prevents mode oscillation.

**When active (expensive):** WTA balance term (γ) + discrete trim + P2 reduced gradient, and
**refine on the balance/stationarity plateau** (not raw energy — the trim's retarget sawtooth
removes the energy plateau, so the energy-based trigger cannot fire; §4). **When inactive
(cheap):** original energy `E₀` (Dirichlet + double well + λ), trim **off** by default so the
normal energy-plateau refinement trigger works again (the familiar fast low-N behaviour), with
the **re-arm condition as the drift safety net** in place of an always-on guardrail.

Thresholds get calibrated-once defaults (like γ) and are tunable; the mechanism, not the exact
numbers, is the design. Variants to ablate (§9): keeping a *light* trim (long period) or P2 on
the cheap levels as a stronger-but-costlier guardrail than re-arm alone.

---

## 6. The cost split — why keep the trim, drop the term

Separating the machinery by cost is what makes the guardrail nearly free:

| Piece | Per-iteration cost | Role | Fine-level decision |
|---|---|---|---|
| WTA gradient term | O(V·N) **every iteration** | the front force that moves fences | **drop** (the expensive part) |
| P2 reduced-gradient (dual solve) | Gauss–Seidel sweeps **every iteration** | unfreezes the descent | optional (test) |
| Discrete trim | O(N) **every ~200 iters**, reuses the argmax already computed by `detect_area_imbalance` | retargets the projection toward discrete equality | **drop by default** — its retarget sawtooth blocks the energy-plateau refinement trigger (§4); the re-arm condition (§5) is the drift safety net instead. *Keep-light* (long period) is an ablation. |

So a fine level running plain `E₀` is genuinely cheap **and** triggers normally; drift is caught
by the per-level gate check that re-arms the full machinery (§5), not by an always-on trim. The
expensive per-iteration gradient term and dual solve are dropped once the structure is correct.

---

## 7. The risk and its mitigation

**Risk:** the original energy has **no territory-restoring force** — it minimizes perimeter
under *continuous* equal-area and is blind to the winner-take-all side of the band. So a cell
that is only *marginally* balanced at level 0 could **drift back** toward a deficit as the fine
level moves boundaries to shave perimeter.

**Why it is bounded / mitigations:**
- The fine level's band fraction is smaller, so there is less room to drift.
- The switch fires only with **headroom** (`w < switch_margin ≈ 3%`, below the 5% gate), so a
  cell must drift a full margin before it even reaches the gate.
- The **re-arm condition** (§5) is the primary safety net: if any cheap level ends with a cell
  back over the gate, the full machinery is re-enabled for the next level automatically.
- Empirically (N=150), the original energy at finer levels reduced rather than increased
  imbalance.
- If drift is observed repeatedly (re-arm oscillating), fall back to the *light-γ* or
  *long-period-trim* fine-level variant (§5, §9) as a stronger standing guardrail.

**Caveat — "balanced at coarse" ≠ "balanced at fine" exactly:** the argmax boundaries shift as
the mesh refines, so coarse-level balance transfers only approximately; the trend is favorable
(finer = smaller gap) and the trim mops up the residual. This is *why* the trim is kept rather
than going fully bare.

---

## 8. Implementation specification

**Config (extend `RelaxationConfig` in `src/pipeline/relaxation.py`).** A single selector
plus two thresholds — the existing flags define *what machinery runs when active*; this decides
*when it is active*:
- `wta_schedule: str = 'off'` — `'off'` (default, current behaviour, byte-identical) /
  `'all_levels'` (the machinery on every level, i.e. the run report 04 measured) /
  `'adaptive'` (this plan: the gate-conditioned switch of §5).
- `wta_switch_margin: float = 0.03` — switch expensive→cheap once `max|dev| < margin`.
- `wta_rearm_threshold: float = 0.05` — re-arm cheap→expensive if a cheap level ends with
  `max|dev| > threshold` (default = the `AREA_IMBALANCE_REL_THRESHOLD` gate).

(A lower-level `*_levels` list form — `wta_balance_levels: [...]`, etc. — can back a *manual*
fixed schedule for ablations, but `adaptive` is the primary mode and should not need it.)

**Code changes.**
- **Runtime mode in the level loop.** `run_relaxation` (`src/pipeline/relaxation.py`) already
  iterates levels and builds the optimizer per level. Carry a `wta_active` bool (init `True`
  under `adaptive`). In `_setup_level`, when `adaptive`, pass the *effective* flags
  (`wta_balance_enabled`, `wta_trim_enabled`, `pgd_reduced_gradient` = `wta_active`; γ from
  config). The flags already exist and are read per level — only their **runtime gating** is
  new (small, localized).
- **The gate check + switch, at each level transition.** After a level's PGD returns, compute
  `detect_area_imbalance` on the winner-take-all state (this is the *same* argmax/bincount the
  trim already runs — a few ms, negligible), take `w = worst |dev|`, and update `wta_active`
  per the §5 hysteresis rule *before* building the next level. Log the decision (`w`, old→new
  mode) so the schedule is visible in `relaxation.log`.
- **Balance-plateau refinement trigger (prerequisite for the expensive levels).** While
  `wta_active`, the trim sawtooth blocks the energy-plateau trigger (§4), so an expensive level
  runs to the iteration cap. Add a trigger mode that fires on a plateau of the *reduced-gradient
  stationarity* `‖g_t‖` (already computed by P2) or of `max|dev|` between trim retargets, in
  `ProjectedGradientOptimizer`'s trigger logic. Cheap levels keep the normal energy-plateau
  trigger. Without this the adaptive schedule still works but wastes ~cap-minus-plateau
  iterations per expensive level (§4: ~3 h at N=200 level 0).
- **State hygiene.** Reset the trim target `d = Ā·1` at each level transition (implementation
  plan §3); ensure toggling the WTA term/P2 between levels leaves no stale optimizer state.
- Everything defaults to `wta_schedule: 'off'` ⇒ byte-for-byte unchanged from `main`.

**No new math** — reuses the term/trim/optimizer of `docs/math/07-phase1-wta-balance/`; it only
changes *when* they run and *how the level decides to stop*.

---

## 9. Validation / experiments

Prerequisite **met**: the all-levels N=200 territory run is valid on the bad seed
(`docs/experiments/04-territory-aware-highn-validation/`: −34% → 0 imbalanced by level 1),
establishing that the coarse territory-aware relaxation balances the partition. The open
question this plan tests is the one the all-levels run did *not*: **does the cheap `E₀` on the
fine levels preserve that balance?** (Report 04's fine level still ran the expensive energy.)

1. **Head-to-head on the same N=200 case** (`torus_200part_coarse_seeded_lam9`, seed 84172851):
   `wta_schedule: adaptive` vs `all_levels`. Compare **both**:
   - **Validity:** final `area_imbalance.n_imbalanced` (must stay 0; worst cell < 5%). Confirm
     the adaptive switch fires where expected (log `w` and the switch level — predicted after
     level 1 here) and that no cheap level re-arms.
   - **Wall time:** total + per-level breakdown (expect the fine levels much cheaper, and each
     expensive level to trigger *before* the 30k cap once the balance-plateau trigger is in).
   Success = balance preserved through the cheap fine levels **and** a large wall-time drop.
2. **Ablations:**
   - (a) `switch_margin` sensitivity (e.g. 2% / 3% / 4%): does a tighter margin prevent drift;
     does a looser one save iterations without re-arming?
   - (b) fine levels bare `E₀` (default) vs `E₀` + *light/long-period trim* guardrail: is the
     re-arm safety net enough, or is a standing guardrail needed?
   - (c) P2 kept vs dropped on the cheap levels.
   - (d) does the balance-plateau trigger cut the expensive-level iteration count materially
     (vs running to the cap)?
3. **N=300**, then the N=250/500/1000 ladder under the mesh-budget policy — the real test of
   whether the adaptive schedule (expensive machinery confined to the coarse levels *it needs*)
   makes N=1000 affordable. Watch the switch level rise with N (the design's central claim).

Record the outcome in `docs/experiments/` (a sibling to the all-levels validation report),
with the per-level timing tables as the headline evidence.

---

## 10. Definition of done

- [x] Prerequisite met: all-levels territory run is valid on the bad seed
      (`docs/experiments/04-territory-aware-highn-validation/`).
- [ ] Adaptive gate-conditioned schedule implemented (`wta_schedule: adaptive`, switch/re-arm
      thresholds), flag-gated, default `off` (all-levels unchanged); switch decision logged.
- [ ] Balance-plateau refinement trigger implemented for the expensive levels (else they run
      to the iteration cap).
- [x] Checkpoint/resume hardened (per-level solution checkpoint) so a multi-day cluster run
      survives interruption — the report-04 run died mid-level-2 and lost all but traces.
      `checkpoint_per_level` (default on) writes `solution/checkpoint_level{L}.h5` after each
      completed level (atomic move, newest kept, cleaned up on success); `--resume-from` a
      checkpoint restarts at the next level and, under `adaptive`, restores `wta_active`
      instead of re-arming. Two latent resume bugs fixed alongside: (i) `completed_levels`
      was written as `len(levels_meta)` — the levels run *this invocation* — so resuming a
      resumed run restarted at the wrong rung; it is now the absolute ladder position.
      (ii) YAML 1.1 parses an unquoted `wta_schedule: off` as the boolean `False`, which read
      as `'False' != 'off'` and took the scheduled path, force-overriding the individual
      `wta_*` flags to off; `RelaxationConfig.__post_init__` now normalizes and validates the
      value (an unrecognized schedule raises instead of silently relaxing with plain E₀).
      Verified: kill mid-level-2 → resume → correct rung (32×22), schedule state restored,
      final `completed_levels=3`; off-path solution attrs unchanged; both branch harnesses
      (`validate_pgd_optimizations --equivalence`, `test_wta_balance_gradient_analytical`) PASS.
- [ ] Head-to-head N=200 (`adaptive` vs `all_levels`): validity preserved (n_imbalanced 0),
      the switch fires after level 1 with no re-arm, AND wall time substantially reduced;
      per-level timing recorded.
- [ ] Ablations (switch_margin, bare-E₀ vs light-trim, P2 on/off, balance-plateau trigger) run
      and tabulated.
- [ ] N=300 confirmed; scaling ladder started; the switch level observed to rise with N.
- [ ] Experiment writeup in `docs/experiments/`; CLAUDE.md + affected plan docs updated.

---

## 11. References

- Internal: `docs/plans/PHASE1_TERRITORY_AWARE_IMPLEMENTATION_PLAN.md` (the term/trim/P2 this
  schedules), `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` (diagnosis: accounting identity,
  band-fraction √N law, frozen optimizer), `docs/plans/PHASE1_N1000_SCALING_PLAN.md`
  (performance wall — composes with this), `docs/reference/winner_take_all_partition_gap.md` §9,
  `docs/math/07-phase1-wta-balance/` (the term's derivation).
- Coarse-to-fine / nested iteration principle: W. L. Briggs, V. E. Henson, S. F. McCormick,
  *A Multigrid Tutorial*, 2nd ed., SIAM, 2000 (nested iteration / full multigrid — expensive
  coarse-grid work, cheap fine-grid smoothing). U. Trottenberg, C. Oosterlee, A. Schüller,
  *Multigrid*, Academic Press, 2001.
- Method context: B. Bogosel, É. Oudet, *Partitions of Minimal Length on Manifolds*,
  Experimental Mathematics 31(3), 2023; arXiv:1606.02873.
