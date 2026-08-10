# Phase 1 High-N Validity Plan — Territory-Aware Relaxation (WTA balance term + optimizer fix)

**Status:** IN PROGRESS — P1 (WTA balance term + discrete trim) and P2
(reduced-gradient/trigger fix) are implemented behind default-off flags on
`feat/phase1-territory-aware-relaxation` per the executable spec
`docs/plans/PHASE1_TERRITORY_AWARE_IMPLEMENTATION_PLAN.md` (see its progress
log). The §5.3 verifications are done: the frozen-optimizer/identity findings
below were re-confirmed on `run_20260714_224821` (Stage 0), and the P1 gradient
passes the FD check. The confirming experiment (§4) is prepared but not yet run.
The diagnosis text below is retained as the standing rationale.

**Provenance:** produced by a Fable analysis agent on 2026-07-16, from the N=300
winner-take-all scaling wall. Quantitative claims marked *measured* come from the
agent's in-memory numerical probes of `results/run_20260714_224821_.../solution/*.h5`
(N=300, λ=12, seed 61803399, V=47,488 vertices, ε=0.01579); *derived* = arithmetic.
**These numbers should be re-verified before implementation** (see §5.3) — the diagnosis
is independently checkable and the P1 gradient must pass a finite-difference check.

**Pairs with:** `docs/plans/PHASE1_N1000_SCALING_PLAN.md` — that plan is the *performance*
wall (speed/memory at N≥1000). This is the distinct *validity* wall (producing a valid
winner-take-all partition at all). Both must hold at N=1000.
**Related:** `docs/reference/winner_take_all_partition_gap.md` §9 (the empirical
λ-window / seed-lottery / two-sub-type findings this plan explains and supersedes as the
proposed fix).

---

## 1. Problem and goal

The Phase 1 relaxation enforces equal area on the **continuous** density field, but the
deliverable is the **winner-take-all (argmax)** partition. At high region count N the
discrete argmax areas drift from equal even though the continuous areas are equal. Valid
partitions exist at N=100/150/200; N=300 is stuck (best run: 9 of 300 cells over the 5%
area gate, worst −34.9%). The **end goal is molecular simulation, which needs N ≈ 1000+**,
so the fix must *scale*, not clear N=300 as a one-off. Per-N seed hunting does not scale.

---

## 2. Diagnosis

### 2.1 The accounting identity (root cause)

With continuous mass pinned at Ā = A/N (measured residual 1.3e-9), for every cell k:

> **T_k − Ā = gain_k − lost_k**

where T_k = winner-take-all territory, lost_k = cell k's mass on vertices it does *not*
win, gain_k = foreign sub-argmax mass on vertices it *does* win. Verified exactly on the
λ=12 solution (corr 1.0000, max error 1e-15).

**The key point:** nothing in the energy or the constraints references which *side* of the
argmax boundary a unit of interface-band mass sits on. `compute_energy`
(`src/optimization/pgd_optimizer.py`) prices |∇u| and the well u(1−u) symmetrically about
u=0.5; the projection (`src/optimization/projection.py`) pins only ∫u_k. So `gain − lost`
is an **uncontrolled output**, and the whole imbalance is its variance.
*Measured:* corr(relative area deviation, lost-paint fraction) = **0.936** across all 300
cells; even healthy cells lose ~12.4% of their paint (σ 3.2%), matching the interface-band
model (predicted band fraction 20.9% at N=300, half of it lost ⇒ 10.5% vs 12.4% measured).
The 9 flagged cells are the deep tail of this same distribution (lost fractions 8.6–45.5%).

### 2.2 The scaling law (why it worsens with N)

The controlling dimensionless parameter is the **band fraction**
`f_b = 3.72/√(2V/N)` (∝ √N at fixed mesh V; equivalently set by vertices-per-cell). Mean
lost paint ≈ f_b/2; its dispersion widens with f_b; cells fail the 5% gate when their loss
deviates ≳5% from the mean. Ladder from the actual runs:

| run | V finest | verts/cell | f_b | outcome |
|---|---|---|---|---|
| N=100 5L (`run_20260709_081548`) | 114,144 | 1,141 | 7.8% | worst 0.8%, valid |
| N=150 3L→5L | 47,488→114,144 | 317→761 | 14.8%→9.5% | 5.8%→**1.2%**, valid |
| N=200 3L (good seed) | 47,488 | 237 | 17.1% | 2.7%, valid (seed-lucky) |
| N=300 3L | 47,488 | 158 | 20.9% | **34.9%, 9 over — stuck** |
| *N=1000 on today's 5L mesh (derived)* | 114,144 | 114 | 24.6% | *structurally infeasible* |

Three compounding N-effects: **(i) structural** — f_b ∝ √N at fixed V, the dominant threat
toward N=1000; **(ii) algorithmic** — coarse levels drop below the freeze threshold so the
cheap multi-level equalization phase disappears and runts entrench at the finest level;
**(iii) λ** — the (artificial, §2.3) ceiling arrives sooner. The seed lottery is just the
random draw of which cell lands in the tail.

### 2.3 Finding — the optimizer is frozen; the λ ceiling is an artifact

From the run traces (`traces/pgd_part300_*_summary.out`) and direct probing of the final
iterate:
- Every level ends in a **dead line search** (steps pinned at step0·ρ⁴⁰ ≈ 9e-13 = 40
  consecutive Armijo rejections, then the `step < 1e-12` bail at `pgd_optimizer.py:336`);
  |ΔE| = 0 over the last 30 iterations. The plateau trigger (`pgd_optimizer.py:408-429`)
  fires **on a frozen optimizer**, always. λ only sets *when* (level 2 froze at iter 7,678
  for λ=12 vs 407 for λ=13).
- The frozen iterate is **not stationary**: projecting g onto the active-constraint tangent
  space leaves KKT residual ‖g_t‖ = 19.9 vs ‖g‖ = 32.5 — **61% of the gradient is feasible
  descent**, concentrated in the halo bands. 40 tangential steps recovered −15.9 energy
  units from the "converged" state.
- Cause: the step P(clip(x − s·g)) is contaminated — g is mostly bound-infeasible (97.6% of
  entries pinned) — and the projection is **non-idempotent** (projecting the already-feasible
  x moves it 3.6e-4 and *costs* +9.2e-3 energy — a composition noise floor from
  clip+rank-1+renorm+rescale, `projection.py:106-123`) that no small step beats. **Fixing
  only the Armijo test would not help; the direction/projection must change.**

⇒ The λ ceiling (and the "needed λ grows with N" ladder) is a property of a **broken
stopping mechanism**, not the landscape. Fixable — see P2.

### 2.4 Finding — fixing the optimizer alone does NOT close the gap

The 40-step tangential descent recovered 16 energy units **but the winner map did not move**
(worst cell bit-identical at 34.95%, n_imbalanced 9). A post-hoc mass-retarget probe
(feeding runts up to +35% continuous mass) cleared near-threshold cells (9→6) but not the
entrenched worst (boosted paint just thickens a losing halo). ⇒ The residual imbalance is a
**feature of the energy under the equal-continuous-mass constraint**, locally stable; it
must be controlled **during relaxation** by a term that sees the territory. Hence P1.

---

## 3. Proposals (ranked)

### P1 (top) — territory-aware soft-WTA balance penalty + exact discrete-target trim

**The penalty term** (added to E in `compute_energy`). Per vertex i, sharpened
partition-consistent weights and per-cell soft territory:

    w_{k,i} = u_{k,i}² / S_i,   S_i = Σ_j u_{j,i}²      (Σ_k w_{k,i} = 1)
    T_k = Σ_i v_i · w_{k,i}                              (Σ_k T_k = A identically; v = lumped mass)
    P_bal(u) = (γ/2) · Σ_k r_k²,    r_k = (T_k − Ā)/Ā,   Ā = A/N

**The gradient** (exact, O(V·N), two row-reductions — same cost class as the existing λ
penalty; add to `compute_gradient`):

    ∂T_k/∂u_{j,i}   = v_i · (2 u_{j,i}/S_i) · (δ_{kj} − w_{k,i})
    ∂P_bal/∂u_{j,i} = (γ/Ā) · v_i · (2 u_{j,i}/S_i) · [ r_j − Σ_k r_k · w_{k,i} ]

**Why it works (the mathematical argument):**
- It is the only class of term that closes the §2.1 identity — it couples `gain − lost`
  (via the softmax surrogate of the argmax) back into the optimality conditions.
- **Front force, not a mass rescale.** Deep inside a crisp cell (δ − w) → 0 (w one-hot),
  so the force acts *only on the interface/halo band*, pushing a deficit cell's front out
  and its surplus neighbours' fronts in, magnitude ∝ signed deficit r. This is exactly the
  mechanism the §2.4 retarget probe showed is missing for entrenched runts.
- **Self-deactivating — no λ-style ceiling.** The λ crispness penalty has a ceiling because
  its gradient is constant pressure that never vanishes; P_bal's gradient ∝ r → 0 on
  balanced partitions. Its Gauss–Newton Hessian block γJᵀJ ⪰ 0 adds curvature only along
  the N balance directions (indefinite part ∝ r, vanishes at balance) — *improves*
  conditioning near the solution.
- **Safe against the documented failure modes.** At the symmetric state u ≡ 1/N: w = 1/N,
  T_k = Ā, r = 0, gradient = 0 — cannot deepen the random-init trap or perturb seeded
  symmetry-breaking. For a starving/dormant cell T_k → 0, r_k → −1: a restorative lift ∝
  u_k — the "reward for winning territory" gap-doc §3 noted no term provides. Indifferent
  to crispness once balanced ⇒ no over-crisping.
- **Γ-consistency:** soft-vs-hard territory gap is O(ε·P_k) (band-confined), so P_bal →
  (γ/2)Σ(true territory deviation)²/Ā² as ε→0; it vanishes on the true minimizer's
  constraint T_k = Ā, so the perimeter Γ-limit is unchanged.

**γ scaling with N (no per-N tuning):** everything is in *relative* residuals; per-entry
gradient magnitude ∼ γ·v_i·(u/S)/Ā ∼ γ/(verts-per-cell) — N-invariant under the P3 mesh
policy. Calibrate γ **once** (e.g. at N=100: choose γ so a 5% deficit gives a front force
~10% of the well gradient at a band vertex).

**Exact discrete-target trim (~30 lines, complements P1):** every J≈200 iterations update
the projection's area targets d (the projection already accepts arbitrary d) by a damped
controller

    d ← d + β·(Ā·1 − T_wta),    β ≈ 0.5,    clamp each d_k to Ā ± 20%

(Σd preserved since Σ T_wta = A). Its fixed point is **exact discrete equality T_wta = Ā**,
independent of any residual soft/hard surrogate gap. Applied while densities are still soft
(from level 0) it *steers condensation* rather than fighting entrenchment. Cost O(N)
amortized; T_wta = one argmax pass (already computed by `detect_area_imbalance`). Note this
deliberately unpins the *continuous* masses by O(ε·P_k) — Phase 2 is indifferent (its iter-0
feasibility is exactly the *discrete* equality this delivers).

**Risks & mitigation:** γ too large distorts fronts early (perimeter cost) — start at the
calibrated value, verify perimeter parity at N=100 vs `run_20260709_081548`. The trim's
damped Picard loop has no global-convergence proof on the nonconvex inner problem — clamp +
damping + J-spacing keep it a bounded perturbation; validation is cheap (§4).

**Compute at N=1000:** one extra O(V·N) elementwise pass in energy and gradient (a few % of
an iteration; the projection dominates at ~86%). Compatible with the scaling plan's sparse
ELL representation (w, S, T are reductions over stored entries) and the dual-Newton
projection (d_k are its targets).

### P2 (prerequisite) — fix the step direction, acceptance test, and triggers

Location: `pgd_optimizer.py:306-341` (line search) and `:408-429` (triggers).
1. **Step along the reduced gradient** g_t = g − α⊗1 − v⊗βᵀ restricted to the free set
   (entries off the box faces), where α∈ℝ^V (sum-to-one duals) and β∈ℝ^N (area duals) are
   found by a few Gauss–Seidel sweeps on the two dual blocks (each sweep = two O(V·N)
   reductions; ~5–10 sweeps with warm-started duals). Then x⁺ = P(clip(x − s·g_t)); the
   projection now has O(s²) work instead of fighting the bound-infeasible bulk of g.
2. **Acceptance:** prox-form sufficient decrease E(x⁺) ≤ E − (c/s)‖x⁺ − x‖² (or Armijo
   against ‖g_t‖² — the probe showed measured dE matched −s‖g_t‖² to 3 digits).
3. **Triggers:** replace raw-gnorm criteria with projected stationarity ‖g_t‖ (or
   ‖x⁺−x‖/s); make `refine_delta_energy` relative to the level's cumulative decrease; and
   treat **"no accepted step in the patience window" = STALLED, not converged** (log it;
   do not fire the refinement trigger on it).
4. Longer term this merges with the scaling plan's **dual semismooth-Newton projection**: an
   exact projection restores P(x−sg) − x → −s·g_t and removes the 3.6e-4 / +9.2e-3
   non-idempotency noise floor. **This finding upgrades that plan item from performance to
   correctness.**

Effects: coarse levels actually relax (restores the cheap equalization phase), level
endpoints become true constrained-stationary points, the λ ceiling disappears (λ can return
to a moderate, N-independent ~5), and K(N) measurements for the N=1000 plan become
meaningful. **Measured caveat: P2 alone does not fix validity** (§2.4) — necessary, not
sufficient. Risk: low (textbook gradient-projection for this constraint set); validate with
`testing/validate_pgd_optimizations.py` stage-A/B at N=50.

### P3 — mesh-budget validity floor (the answer to "ε vs cell size")

Do **not** decouple ε below h: ε ≈ √(mean triangle area) is the FEM-resolvability coupling
(ε < h ⇒ nodal quadrature of u(1−u) under-integrates, interfaces mesh-pin — the inner/outer
torus pathology of §2.1 everywhere). The correct reading of "ε should scale with cell size"
is **hold vertices-per-cell fixed, i.e. V ∝ N** (then ε/cell-diameter = √(N/2V) is fixed).
Policy: finest-level verts/cell ≥ ~600 (f_b ≤ ~10%) for the N=100–150 gate margin; with P1
in the loop this floor likely relaxes to ~250–300 (measure, don't assume). At N=1000 ⇒ V ≈
0.6–1.1M — exactly the scaling plan's working point, so validity and performance walls meet
at the same mesh and its arithmetic stands. Optional 2nd-order: spatially varying
ε(x) = √(local triangle area) (per-triangle K/M weights; Modica–Mortola constant c_W = 1/3
is ε-independent, so the perimeter limit is unchanged) to remove the 4:1 torus anisotropy
that concentrates deficit cells on the inner equator.

### P4 — initialization (hygiene, not a fix)

Keep `init_method: seeded` (mandatory on the corrected well). The lottery is a *symptom*:
the runt is manufactured during condensation, not fixed by the init (gap-doc §5). Per-N seed
hunting is a non-scaling stopgap; P1 is designed to make the outcome seed-insensitive.
Optional cheap improvement: a few v-weighted Lloyd/CVT sweeps after farthest-point sampling
(`initialization.py:23-50`) to equalize seed capacities deterministically — buys iterations,
decides nothing.

### P5 — parameter-file (bridge only)

There is **no pure-YAML fix at N=300**: λ is pinned at ~12 by the (artificial) ceiling and
the only remaining lever is the seed lottery. The one config move that attacks the
structural term is resolution: a 5-level resume of the λ=12 run (V=114,144, verts/cell 380,
f_b 13.2%) should shrink the fine-region artifact cells (N=150 precedent) but is *predicted
insufficient* for the entrenched runts (worst ≈ −9% > 5% gate). Post-P1/P2 the scaling
config is: λ ≈ 5–6 (constant), γ calibrated once at N=100, verts/cell ≥ ~600 finest, 5
levels, seeded init — no per-N tuning.

---

## 4. Top recommendation and cheapest confirming experiment

**Recommendation: implement P1 (WTA balance term + discrete trim) with P2 as its enabling
prerequisite.** P1 is the only avenue that closes the §2.1 identity; it is self-deactivating
(no ceiling), relative-residual-based (no per-N tuning), O(V·N) (no compute wall),
sparse-representation-compatible, and it converts the seed lottery from a *validity* question
into a *perimeter-quality* one. P2 is required because a frozen optimizer (§2.3) cannot relax
retargeted fronts.

**Cheapest confirming experiment** (a step toward N=1000, not a N=300 one-off): re-run the
**known-failing N=200 config** — λ=9, seed 84172851, 3 levels, V=47,488
(`run_20260712_224424`: two entrenched runts at −34%/−38%, immune to λ 7→9) — with **P1+P2
enabled and nothing else changed**. Cost ~6–15 h CPU against an existing failed control.
- **Confirms** if `detect_area_imbalance` goes 2 → 0 (worst < 5%) *without* touching seed or
  λ — the anti-lottery property N=1000 requires.
- **Refutes** if runts persist above the gate ⇒ fallback ladder is P2+P3 (correct optimizer
  + verts/cell ≥ 600) and hard-constraint variants (soft-territory equality via augmented
  Lagrangian) return to the table.
- On success: N=300 A/B vs `run_20260714_224821` (predict 9 → ≤1 over gate), then fold the
  P1 gate metrics into the scaling plan's §6 Phase-5 ladder (N=250/500/1000).

---

## 5. Implementation notes

### 5.1 Key files and locations
- `src/optimization/pgd_optimizer.py` — P_bal in `compute_energy`; its gradient in
  `compute_gradient`; the reduced-gradient step + prox acceptance in `optimize` (lines
  ~306-341); trigger logic (lines ~408-429).
- `src/optimization/projection.py` — the area targets `d` are already parametric (the trim
  updates them); dual-Newton projection is the longer-term exact-projection upgrade.
- `src/pipeline/relaxation.py` — controller (trim) state carried across levels; re-initialize
  the per-level `d` to Ā·1 after interpolation; ε = √mean_triangle_area assembly (P3).
- `src/partition/find_contours.py` — T_wta measurement reuses `detect_area_imbalance`
  internals.

### 5.2 Suggested phasing
1. **Verify the diagnosis** (§5.3) — cheap, no production run.
2. **P2** (optimizer/trigger fix) behind a config flag; validate with the N=50 stage-A/B
   gates. Confirm it does not change validity yet (expected).
3. **P1** term + gradient (FD-checked) + trim, behind a config flag (e.g. `wta_balance: {γ}`).
4. Run the **§4 confirming experiment** (N=200 bad seed).
5. If confirmed: N=300 A/B, then the N=250/500/1000 ladder with P3's mesh budget.

### 5.3 Validation / verify-before-trusting
- **FD gradient check** for ∂P_bal/∂u against a Richardson finite difference (mirror
  `testing/test_steiner_gradient_analytical.py`). Non-negotiable before trusting P1.
- **Re-verify the frozen-optimizer finding** (§2.3) directly on `run_20260714_224821` — it
  is a property of the *existing* solution (compute the reduced-gradient KKT residual and the
  projection non-idempotency), no new run needed. This is the cheapest, highest-leverage
  check and gates whether P2 is worth building.
- **N=50 regression** via `testing/validate_pgd_optimizations.py` to ensure P2 preserves the
  documented serial-optimization equivalence.
- Perimeter-parity A/B at N=100 to ensure P1's γ does not distort fronts.

### 5.4 Open questions to resolve during implementation
- Exact γ calibration constant (start from the N=100 rule above; may need one tuning pass).
- Trim cadence J and damping β (start J≈200, β≈0.5; the clamp bounds the risk).
- Whether the reduced-gradient dual solve (Gauss–Seidel on α, β) is cheap enough per
  iteration at N=1000, or needs the dual-Newton projection from the outset.
- Whether P1 lets the P3 verts/cell floor drop to ~250–300 (a compute saving to measure).

---

## 6. Status and decision

**Not implemented.** This document exists so the analysis survives the session. The decision
to implement — and on which branch — is deferred. If pursued, follow §5.2, starting with the
zero-cost §5.3 verification of the frozen-optimizer finding and an FD check of the P1
gradient before any production run.
