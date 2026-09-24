---
paths:
  - "src/partition/mbo_auction.py"
  - "src/partition/arm_harness.py"
  - "scripts/run_mbo_arm.py"
  - "parameters/*mbo*.yaml"
  - "testing/test_mbo_auction.py"
  - "testing/test_balanced_assignment_solver.py"
  - "testing/test_arm_harness.py"
---

# Approach B — auction-dynamics MBO

A **replacement for Phase 1**, not an adjustment to it. The state is a hard
labelling throughout, so there is never a continuous field whose winner-take-all
readout can diverge. Per step: diffuse `(M + tau K) y_k = M chi_k` (one
prefactorized LU per level), then reassign by balanced thresholding
`argmax_k [y_ik + psi_k]` via `solve_dual_offsets`. Hard-required init: one
balanced C-scores assignment.

Status: **built and measured**, valid to N=1000 on the torus and N=10 on both
implicit surfaces. Full results, tables and attribution controls:
`docs/experiments/08-mbo-auction-dynamics/`. Derivation and provenance:
`docs/math/10-mbo-auction-dynamics/`.

## The `mbo:` config block

Approach B has far fewer free parameters than PGD — there is no
`lambda_penalty` to calibrate per N, because the time step is *derived*:
`tau = min((tau_c*h_mean)^2, (rho*R_cell)^2)`, computed per level from the mesh
and N, never set by anyone. Only the three constants that change **the answer**
are config-reachable:

```yaml
mbo:
  tau_c: 4.0        # freeze <-> over-merge window
  rho: 1.0          # over-merge ceiling; loosest defensible value
  max_iters: 200    # per-level step cap; 3x headroom over the worst seen (66)
```

Precedence is **CLI > `mbo:` block > `MBOConfig` defaults**, and the driver
prints which source each value came from. A config without the block is
bit-identical to the pre-2026-08-22 behaviour.

The rest of `MBOConfig` — `churn_tol`, `patience`, `anneal_factor`,
`max_anneals`, `early_stop_in_loop`, `finalize_full_budget`, the NC3 probe — is
**deliberately not config-reachable**: it defines the measurement *protocol*
rather than the result. `early_stop_in_loop` especially, because early stopping
once flattered a result into looking good (report 07, artefact 5). None of the
three has ever needed changing: across 18 levels in 4 runs every level reported
`converged: true`, `hit_max_iters: false`, `n_anneals: 0`, worst level 66 steps.
The when-to-change note lives in `parameters/torus_500part_mbo.yaml`.

## tau is a TWO-SIDED window, and a fixed `c` does not hold it

Below `c ~ sqrt(R_cell/h)` the scheme **freezes** (per-step motion ~ tau*kappa
falls under one edge length); above `sqrt(tau) ~ R_cell` it **over-merges** (the
diffused indicator has spread over the whole cell).

The trap: a fixed `c` holds `sqrt(tau)/h_max` constant at 2.20 while
`sqrt(tau)/R_cell` varies **6x** across the ladders in use (0.250 at N=100
V=114,144 up to 1.495 at N=300 V=9,600), so the two sides pull in opposite
directions as the mesh coarsens or N rises.

**Report `sqrt(tau)/h_max`, never `h_mean` alone** — the torus mesh is **1.81x
anisotropic**, so a mean-based figure shows a comfortable 2.20 while the
coarse-triangle band sits below its own local non-freeze threshold of 2.97. The
two rulers disagree about which levels are at risk, so neither settles it by
algebra; the calibrated pinning probe (NC3) is the arbiter.

⚠ Prefer **dropping a ladder level** over raising `tau_c` when a level freezes —
freezing is a property of the mesh/cell ratio, and raising `tau_c` walks toward
the over-merge side of the same two-sided window.

## A freeze ratio < 1.0 is NOT a warning at high verts-per-cell

The ratio falls as cells grow relative to the mesh, so a well-resolved
configuration necessarily runs low; reading `< 1.0` as a defect is a mistake —
it fires on all three levels of the *validated* torus N=10 smoke config. The
measured band that passes every gate reaches **0.524** (mc_study C1, n=25) and
0.614 (B1, n=50). Judge a ratio against that measured band, not against 1.0. The
genuinely dangerous end is the opposite one — 9.6-12.8 v/cell, where a cell is a
handful of vertices.

## A sub-threshold rung is survivable — it costs work, not correctness

When the over-merge cap is active the freeze ratio reduces to
`sqrt(R_cell/h_max)`, which *falls* as cells shrink, so the standard `100x96`
base drops below 1.0 at high N: **0.964 at N=750, 0.897 at N=1000**. Both ran
and passed all three gates.

A sub-threshold level captures **18% (N=750) / 26% (N=1000)** of the improvement
available to it, against 55-89% for a healthy level — a *partial* freeze. It
also manufactures fragmented cells (2 and 3), which the ladder then heals
completely: `2->0->0->0->0` and `3->2->1->0->0`. **This is the distinction from
PGD**, where a starved level 0 leaves a *permanent* runt
(`docs/experiments/06-subfloor-ladder/`); under B the same insult is transient.

⚠ The ratio does not predict how much a sub-threshold level accomplishes —
N=750 has the better ratio and more v/cell yet captured *less*. Prefer re-basing
the ladder when you can; accept a sub-threshold rung when procedure-matching
matters more than partition quality.

## Cost is NOT ~V*N — verts-per-cell is the likely driver

Measured at the matched mesh V=24,948: **N=400 3.24 s/step (62.4 v/cell), N=500
5.09 s/step (49.9), N=750 24.61 s/step (33.3)** — x1.57 for x1.25 cells
(exponent 2.0), then x4.83 for x1.50 cells (exponent **3.9**). The
balanced-assignment dual appears to need more subgradient iterations when each
cell owns fewer vertices, which confines the blow-up to *coarse* levels.

Tested at N=750 and held on two lines: the finest level (211 v/cell) came in at
35.96 s/step against a pure-V*N prediction of 33.03 (**+8.9%**); and per-step
cost is *flat* across levels 0-2 (23.7 / 23.0 / 22.5 s) while V triples —
tripling the mesh was free because v/cell tripled with it. Four N, not a law.

Consequences: (1) never budget a new N by scaling V*N — run a `--levels 1` smoke
test first; (2) a ladder whose coarse levels are starved is expensive as well as
risky.

## On a fixed mesh the worst cell saturates at ~1.61x the granularity floor

Measured at V=114,144: N=400 1.386, N=500 1.568, N=750 **1.610**, N=1000
**1.609**. If it holds, quality is predictable from the mesh alone (worst ~ 1.61
x granularity, which scales as N/V), so a mesh can be sized for a target area
error without running anything. Four points, no mechanism — a pattern, not a law.

## Instrument discipline

**Assignment quality cannot detect over-merging.** The balanced assignment hits
its area target *by construction*, so a smeared, non-local cell scores just as
well as a compact one. Anything reasoning about `tau` being too large must use a
geometry instrument: fragmented-cell count, per-cell isoperimetric ratio
`Q_k = P_k^2/(4*pi*A_k)`, or core loss (cells whose own diffused-score peak
vertex lies outside their territory). Citing an assignment-quality result
against an over-merge claim is the same class of error as the theatre bar and
the early-stop mask (`docs/experiments/07-phase0-shared-harness/`).

**Do not use `E_tau` monotonicity as a gate.** The Esedoglu-Otto theorem does not
transfer: the step maximizes `<chi',Dy>` with **lumped** `D`, while the theorem's
object `M A_tau` is the symmetric one, *and* Jacobs-Merkurjev-Esedoglu need an
**exact** auction where `solve_dual_offsets` is an inexact subgradient dual. Real
`E_tau` increases are observed (4 of 96 active steps at N=300). What holds by
construction, and what `testing/test_mbo_auction.py` G4 gates, is
`<chi',Dy> - <chi,Dy> >= sum_k psi_k (T_k - T'_k)` — measured violation <= 6.1e-08.

**Only connectivity has content as a gate here.** Dormant is *vacuous* (one-hot
labels give peak density 1.0 by construction; `arm_harness.py` flags
`vacuous_for_arms: True`); area is *near-expected*, since equal area is what the
assignment step optimizes, so passing means the solver converged — Phase 0
established it as the solver-failure detector, not a validity test. Nothing in B
enforces connectivity and the source proposal states it is "not per-step
guaranteed". Saying "all three gates pass" is true but misleading.

## Gates

```bash
python testing/test_mbo_auction.py                      # G1-G8
python testing/test_mbo_auction.py --negative-controls   # NC1/NC4/NC5 + NC3-CAL
```

Every gate has a failing counterpart: G1 mutates its own input, G7 rejects a 2x
tau error that mass conservation is blind to, G4 refuses to certify on a frozen
tail, NC5 must separate c=8 from c=4 or the finding is recorded as "no
instrument here sees over-merge".
