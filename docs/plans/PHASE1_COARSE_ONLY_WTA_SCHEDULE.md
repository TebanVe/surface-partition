# Future Plan — Coarse-Only WTA Schedule (staged energy across mesh levels)

**Run the expensive territory-aware machinery only on the coarsest level, then refine with
the cheap original energy — to keep the winner-take-all fix while making it scale to N≈1000.**

**Status:** PROPOSED / future work. **Conditional** on the territory-aware term
(`docs/plans/PHASE1_TERRITORY_AWARE_IMPLEMENTATION_PLAN.md`, implemented on
`feat/phase1-territory-aware-relaxation`) first being shown to *work* — i.e. that a
coarse-level territory-aware relaxation actually produces a balanced partition. If it does
not, this plan is moot; if it does, this is the intended next experiment.

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

---

## 5. The proposal — staged (per-level) energy schedule

Run a **per-level schedule** of the objective and optimizer:

- **Level 0 (coarsest):** full territory-aware machinery — WTA balance term (γ) **+** discrete
  trim **+** P2 reduced-gradient/trigger fix. Goal: produce a *balanced* partition (all
  territories within the gate, no runt).
- **Levels ≥ 1 (finer):** switch to the **original energy E₀** (Dirichlet + double well + λ),
  **keep the cheap discrete trim as a guardrail**, and **drop the expensive WTA gradient
  term**. Goal: sharpen the balanced structure cheaply; the trim catches any residual drift.

Optional variants to evaluate (see §8):
- Whether to keep P2 at finer levels. Dropping P2 reverts to the original line search — which,
  on a *warm-started, already-balanced* structure, may legitimately trigger early and fast
  (like the healthy low-N behavior), or may under-converge; to be measured.
- A *light* WTA term (small γ) at finer levels instead of fully off, as a stronger guardrail
  than the trim alone (at some extra cost).

---

## 6. The cost split — why keep the trim, drop the term

Separating the machinery by cost is what makes the guardrail nearly free:

| Piece | Per-iteration cost | Role | Fine-level decision |
|---|---|---|---|
| WTA gradient term | O(V·N) **every iteration** | the front force that moves fences | **drop** (the expensive part) |
| P2 reduced-gradient (dual solve) | Gauss–Seidel sweeps **every iteration** | unfreezes the descent | optional (test) |
| Discrete trim | O(N) **every ~200 iters**, reuses the argmax already computed by `detect_area_imbalance` | retargets the projection toward discrete equality | **keep** (near-free guardrail) |

So a fine level running `E₀ + trim` gets a discrete-equality guard at negligible cost, without
paying the expensive per-iteration gradient term or dual solve.

---

## 7. The risk and its mitigation

**Risk:** the original energy has **no territory-restoring force** — it minimizes perimeter
under *continuous* equal-area and is blind to the winner-take-all side of the band. So a cell
that is only *marginally* balanced at level 0 could **drift back** toward a deficit as the fine
level moves boundaries to shave perimeter.

**Why it is bounded / mitigations:**
- The fine level's band fraction is smaller, so there is less room to drift.
- The **kept trim** re-targets the projection each cadence to pull discrete areas back toward
  equal — a cheap corrective that directly opposes the drift.
- Empirically (N=150), the original energy at finer levels reduced rather than increased
  imbalance.
- If drift is still observed near the gate, fall back to the *light-γ* fine-level variant.

**Caveat — "balanced at coarse" ≠ "balanced at fine" exactly:** the argmax boundaries shift as
the mesh refines, so coarse-level balance transfers only approximately; the trend is favorable
(finer = smaller gap) and the trim mops up the residual. This is *why* the trim is kept rather
than going fully bare.

---

## 8. Implementation specification

**Config (extend `RelaxationConfig` in `src/pipeline/relaxation.py`).** Add a per-level
schedule. Two possible shapes (pick during implementation):
- Simplest: a boolean `wta_coarse_only: bool = False` that, when true, applies the WTA term
  (+ P2, per a sub-flag) only on level 0 and `E₀ + trim` thereafter.
- More general: explicit per-level lists, e.g. `wta_balance_levels: [0]`,
  `wta_trim_levels: [0,1,2]`, `pgd_reduced_gradient_levels: [0]`, so any schedule is
  expressible. Prefer this if the simple flag proves too rigid.

**Code changes.**
- The multi-level loop in `run_relaxation` (`src/pipeline/relaxation.py`) already iterates
  levels and constructs the optimizer per level. Read the schedule and pass the right
  per-level flags (`wta_balance_enabled`, `wta_balance_gamma`, `wta_trim_enabled`,
  `pgd_reduced_gradient`) into `ProjectedGradientOptimizer` for each level. This is a small
  change — the flags already exist (implemented on the feature branch); only their *per-level
  gating* is new.
- Ensure the trim's projection-target state `d` is reset to `Ā·1` at each level transition
  (already specified in the implementation plan §3) and that dropping the WTA term between
  levels is clean (no stale state).
- Everything defaults to the current all-levels behavior; `wta_coarse_only` off ⇒ byte-for-byte
  unchanged.

**No new math** — this plan reuses the term/trim/optimizer already derived in
`docs/math/07-phase1-wta-balance/`; it only changes *where* they run.

---

## 9. Validation / experiments

Prerequisite: the all-levels N=200 test must first come back **valid** (n_imbalanced → 0),
establishing that the coarse territory-aware relaxation balances the partition.

1. **Head-to-head on the same N=200 case** (`torus_200part_coarse_seeded_lam9`, seed 84172851):
   coarse-only schedule vs the all-levels run. Compare **both**:
   - **Validity:** final `area_imbalance.n_imbalanced` (must stay 0; worst cell < 5%).
   - **Wall time:** total, and the per-level breakdown (expect the fine levels to be much
     cheaper).
   Success = balance preserved through the cheap fine levels **and** a large wall-time drop.
2. **Ablations:**
   - (a) coarse WTA → fine `E₀` only (no trim): does the balance drift? (measures whether the
     trim guardrail is necessary).
   - (b) coarse WTA → fine `E₀ + trim` (the recommended default).
   - (c) P2 kept vs dropped at finer levels.
   - (d) light-γ fine-level variant, if (a)/(b) show drift.
3. **N=300**, then the N=250/500/1000 ladder under the mesh-budget policy — the real test of
   whether confining the expensive work to level 0 makes N=1000 affordable.

Record the outcome in `docs/experiments/` (a sibling to the all-levels validation report),
with the per-level timing tables as the headline evidence.

---

## 10. Definition of done

- [ ] Prerequisite met: all-levels territory run is valid (coarse level balances the partition).
- [ ] Per-level schedule implemented, flag-gated, default-off (all-levels unchanged).
- [ ] Head-to-head N=200: validity preserved (n_imbalanced 0) AND wall time substantially
      reduced vs all-levels; per-level timing recorded.
- [ ] Ablations (trim on/off, P2 on/off, light-γ) run and tabulated.
- [ ] N=300 confirmed; scaling ladder started.
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
