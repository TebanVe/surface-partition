# Replacing Phase 1 at High N — Shared Harness, then C, then B

**Status:** Not Started

## Background

Phase 1 (Γ-convergence relaxation by PGD) produces a *continuous* density field
whose winner-take-all readout is not a valid partition at high N: cells are area
imbalanced, and occasionally disconnected. Four attempts to make the N=300 ladder
cheaper have failed (territory-aware relaxation; A2 soft area constraint; A2's
exact-at-level-0 hybrid; A2's adaptive within-level switch), and one attempt to
show the ladder itself was the defect (`docs/experiments/06-subfloor-ladder/`)
refuted its own hypothesis.

The standing taxonomy of high-N approaches is
[`../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`](../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md)
(source proposal archived verbatim beside it). **A** (balanced readout by
semi-discrete OT dual shifts) and **E** (connectivity repair) shipped, as
`src/partition/balanced_readout.py`. **A2** was rejected. The gate on **B** and
**C** was met on 2026-08-12.

**The motivating argument is about scaling, not about PGD's ceiling.** The
balanced readout repairs a broken partition at extraction in ~17 s, which is
excellent as a repair tool and is not a path forward: it requires first spending
days of compute producing an invalid partition. That cost grows with N, so a
method whose output needs repair does not reach N=400, N=500, or N=1000. B and C
both **abolish the category** — every iterate is already a hard partition with
exactly equal discrete areas, so there is never a continuous field whose readout
can diverge, and nothing to repair.

### What this plan deliberately does not assume

An adversarial review on 2026-08-15 withdrew three claims that would otherwise
motivate this work, and this plan must not lean on them:

- **"N=300 is energy-limited, not descent-limited"** is *unsupported*. Level 2 of
  the λ=11.5 run floored its line search at iteration 6,628 with the energy still
  falling (−0.667 over the last 1,000) and ‖g‖ = 31.11. Whether a better ladder
  could reach the gate is **unknown**.
- **The "~90 verts/cell runs-or-dies boundary"** is not established; v/cell is
  isolated by exactly one controlled pair.
- **"Neither more iterations nor more levels reaches the gate"** extrapolates two
  levels.

So this plan is *not* justified by "PGD is exhausted." It is justified by the
scaling argument above, which is independent of PGD's ceiling.

## Verified state of the evidence base (checked 2026-08-15)

This section exists because the falsifiers below depend on baselines that were
assumed to exist. Four of them do not.

| Fact | Status |
|---|---|
| **No N=300 run has ever had Phase 2 run on it** — no `refinement/` directory exists under any `results/*npart300*` | ❌ blocking |
| **No N=300 readout campaign exists on disk** | ❌ |
| `run_20260806_123326`, cited in CLAUDE.md as the N=300 readout measurement | ❌ not on disk |
| **Neither λ=11.5 run completed** — `run_20260813_000958` has no solution file; `run_20260813_003231` stops at `checkpoint_level03.h5` | ❌ |
| Usable N=300 PGD solution: **λ=12, `run_20260714_224821`, seed 61803399** — 9/300 imbalanced, worst 34.95%, 0 dead, 0 weak, **V = 47,488** (224×212) | ✅ |
| Second λ=12 solution: `run_20260716_152451`, seed 27182818 — 10/300 imbalanced, worst 40.62% | ✅ |
| **N=100 deliverable `run_20260709_081548`** — Phase 2 best perimeter **185.2546144718457** (campaign `ipopt_btol0.001_lbfgs30_hess_bestiter_partial`, iteration 020), all gates clean, worst cell 0.78% | ✅ verified |

Two consequences:

1. **The primary falsifier must anchor at N=100**, where a complete, validated,
   independently re-verified Phase 2 baseline exists. This also matches the source
   proposal's framing point 5: *"run the C-vs-relaxation comparison at N=100
   before investing further in relaxation-side machinery."*
2. **N=300 is the scaling test, and it needs a baseline built first.** The λ=12
   solution at V=47,488 is exactly the mesh B's falsifier already names, so the
   ladder cost is already paid — but readout + Phase 2 on it has never been run.
   That is Phase 0's deliverable, not B's.

## Phase 0 — shared balanced-assignment solver + evaluation harness

**Status:** Not Started

Everything in Phases 1 and 2 depends on this, and **nothing downstream may modify
it.** Freezing the shared pieces first is what makes C's and B's numbers
comparable. It is also where an error is most expensive: today's corrections all
traced to a measurement harness quietly shaping its own conclusion.

### 0a — Extract the balanced-assignment solver

`solve_dual_offsets(log_densities, lumped_mass, target, config)` in
`src/partition/balanced_readout.py:143` is already generic in everything but its
parameter name: its only use of the score matrix is
`np.argmax(log_densities + psi, axis=1)`. A, B and C all need exactly this —
*balanced assignment given arbitrary per-vertex scores*:

- **A** calls it on `log u` (densities) — shipped.
- **B** calls it on `y`, the diffused indicators.
- **C** calls it on `−d²`, negative squared geodesic distance.

**Do this as an in-place generalization, not a new module.** The readout is
production code that produced the valid N=300 partition; a function it already
exercises is better tested than a new one. Rename the parameter to `scores`,
document that any additive score matrix is valid, and leave the call site in
`apply_balanced_readout` unchanged.

**Acceptance:** the balanced readout is byte-identical before and after on a
fixed input. This is a refactor with a hard equivalence gate, not a behavior
change.

### 0b — Build the evaluation harness

One harness, used unchanged by every arm. Given a set of labels on a mesh it
must report:

- all three validity gates (`detect_dormant_cells`, `detect_area_imbalance`,
  `detect_disconnected_cells`);
- perimeter **before** Phase 2 (from `ContourAnalyzer`);
- perimeter **after** Phase 2, same campaign settings as the control;
- wall time, taken from `timing_profile.yaml`'s `total_wall_s` — **never**
  `metadata.yaml`'s `run_time_seconds`, which is the last level's time only and
  under-reports by 3.6× on the N=100 deliverable.

Any arm's output is one-hot densities in the Phase 1 solution schema, so
`ContourAnalyzer`, `refine_perimeter.py` and `check_fragmentation.py` consume it
unchanged with no new flags.

### 0c — Build the missing N=300 baseline

Run the balanced readout, then Phase 2, on **λ=12 `run_20260714_224821`**
(V=47,488). This produces the PGD-side number that B's pre-registered falsifier
compares against, and it does not exist today. Record its perimeter, gates and
wall time through the harness from 0b.

**This is a real compute cost and it is on the critical path.** It must be paid
whether or not B and C ever run, because without it there is no N=300 comparison
at all.

## Phase 1 — C: capacity-constrained geodesic Lloyd

**Status:** Not Started

Runs **first**, for three reasons: it is the cheaper candidate in its own right;
it is simpler, so it shakes down the harness with less machinery around it; and
having the baseline recorded *before* the candidate's number exists is what stops
"good" being defined after the fact.

Drop the density field entirely. N sites; geodesic distance from each site
(multi-source Dijkstra on the mesh edge graph via `scipy.sparse.csgraph`, or the
heat method); assignment by the 0a solver with cost `d²` — making it a geodesic
power diagram with prescribed capacities; Lloyd step moves each site to its
cell's `v`-weighted centroid, snapped back to the nearest mesh vertex; ~30
iterations; labels straight to Phase 2. Start from the existing farthest-point
seeds in `src/optimization/initialization.py`.

Exact discrete balance holds by construction and is tail-immune, so it scales to
N=1000 trivially.

**Known limitation, stated up front:** C minimizes a *quantization* energy
(`Σ_k ∫ d²`), not perimeter. It produces equal-area round cells, which on a flat
domain tend to hexagons — and by the honeycomb theorem hexagons are
perimeter-optimal, so C should land *near* the right answer. The torus is curved,
so geodesic Voronoi ≠ minimal perimeter, and C is expected to be systematically
off by an amount nobody has measured. Measuring it is the point.

## Phase 2 — B: auction-dynamics MBO

**Status:** Not Started

Threshold dynamics (Esedoğlu–Otto), with volume constraints by balanced
assignment (Jacobs–Kim–Léger, JCP 2018 — published for this problem class).
Unlike C, B descends *the project's actual objective*: the thresholding energy
Γ-converges to perimeter.

Iterate: (1) diffuse each indicator for time τ by solving `(M + τK) y_k = M χ_k`,
one prefactorized sparse Cholesky per level, reusing `TriMesh`'s existing FEM
matrices; (2) reassign every vertex by balanced thresholding,
`ω(i) = argmax_k [y_ik + ψ_k]`, via the 0a solver. Stray islands shrink to
extinction under mean-curvature flow — the opposite dynamic to the rejected trim.

Seeded init, the mesh ladder, M/K/v, the HDF5 formats and Phase 2 all carry over
unchanged.

**Named risk:** mesh pinning when `√τ ≲ h`. Set `τ ~ h²` and report the ratio
`√τ / h` for every level, so pinning is visible rather than inferred from a bad
result.

## Pre-registration

Written before any code exists. Report 06 pre-registered and honored its refute
branch, and that was the one thing the adversarial review praised without
qualification.

### Primary (N=100, baseline exists)

Control: `run_20260709_081548`, Phase 2 perimeter **185.2546144718457**, all
gates clean.

| Outcome | Verdict |
|---|---|
| Arm passes all three gates **and** Phase 2 perimeter within **+1%** (≤ 187.11) | **Success** — viable replacement |
| Within **+1% to +5%** (≤ 194.52) | **Partial** — viable only if wall-time gain is large; report, do not ship |
| Worse than **+5%**, or fails any gate | **Refuted** for that arm |

### Secondary (N=300, scaling — baseline built in Phase 0c)

Control: λ=12 `run_20260714_224821` at V=47,488, after readout + Phase 2. Same
thresholds. B's original falsifier stands: a ~150-line prototype at that mesh, 50
iterations, refuted if perimeter is >3–5% worse after Phase 2.

### Rules that protect the comparison

These exist to close the routes by which an arm could look good without working.

1. **The balanced readout is NOT applied to B's or C's output.** Both claim to
   produce balanced, connected partitions *by construction*. Repairing them would
   conceal exactly the failure this tests. The readout is applied only to the PGD
   control, which needs it.
2. **Gates are evaluated on the arm's raw labels**, before any Phase 2.
3. **Both N=300 λ=12 seeds** (61803399 and 27182818) are reported, not the better
   one. Fragmentation location is seed-determined, so a single seed is not a
   result.
4. **Wall time comes from `timing_profile.yaml` `total_wall_s`.**
5. **C's number is recorded and committed before B is run.**
6. No claim of the form "every / always / never" without naming the corpus it was
   computed over and how that corpus was enumerated.

## Risks

| Risk | Mitigation |
|---|---|
| Phase 0c's N=300 baseline is expensive and on the critical path | Price it before starting; N=100 primary falsifier does not depend on it |
| B's mesh pinning silently degrades results | Report `√τ / h` per level as a first-class output |
| C is good enough, making B unnecessary | This is a **success**, not a waste — C is simpler and scales better |
| Both arms lose to PGD on perimeter | Still decisive: it would mean the relaxation is doing something necessary, and would refute framing point 5 |
| Harness error invalidates both arms | Adversarial review of *this plan* before implementation; C first as the cheap shakedown |

## Related documents

- Taxonomy: [`../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`](../reference/PHASE1_HIGHN_APPROACHES_ABCDE.md)
- Source proposal, verbatim: [`../reference/phase1_highn_proposal_ABCDE_original.md`](../reference/phase1_highn_proposal_ABCDE_original.md)
- The gap this closes: [`../reference/winner_take_all_partition_gap.md`](../reference/winner_take_all_partition_gap.md) §4b, §4c, §9b
- Why A2 was rejected: `docs/experiments/05-soft-area-constraint/`
- Why the ladder is not the defect: `docs/experiments/06-subfloor-ladder/`
- Code: `src/partition/balanced_readout.py`, `src/optimization/initialization.py`, `src/mesh/tri_mesh.py`
