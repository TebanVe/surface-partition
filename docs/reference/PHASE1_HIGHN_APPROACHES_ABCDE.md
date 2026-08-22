# Phase 1 High-N Approaches A / A2 / B / C / D / E

The standing taxonomy of candidate fixes for the continuous–discrete gap at high
N, their status, and what each one still needs. The full original proposal is
preserved verbatim at
[`phase1_highn_proposal_ABCDE_original.md`](phase1_highn_proposal_ABCDE_original.md);
this document is the interpretive layer — what has since been built, measured, or
refuted, and what that changes.

## Provenance

One agent produced the whole taxonomy. Dispatched 2026-08-07 11:50 from the
session in the `surface-partition-territory` worktree (`c2b1c512-…`),
`subagent_type: general-purpose`, model fable, task-id `ab6c3399b28003fed`; 6m14s
over 95,696 tokens, returning *"Proposal: Closing the Continuous–Discrete Gap at
High N."* The prompt was explicitly adversarial — *"fresh, concrete, critical
proposals — not agreement… Challenge our framing where it deserves it"* — with a
rejected-list the agent had to clear before reusing any idea.

Its raw output under `/private/tmp/…/tasks/` was lost in the 2026-08-14 power
reset. The text survives in the session JSONL and in the agent's own transcript at
`~/.claude/projects/-Users-teban-Development-projects-surface-partition-territory/c2b1c512-d2f3-45ca-9f2e-5784618de99c/subagents/agent-ab6c3399b28003fed.jsonl`.
It has now been archived into the repo because it was very nearly lost twice.

**Correction to an earlier version of this file.** A prior draft asserted the set
was "A2 / B / C" and that B and C "were never designed." Both claims were wrong.
The set is **A / B / C / D / E**, and **A2 is a sub-branch of A**, not a peer of
B and C — it is the *"Extension (A2), once proven"* paragraph inside approach A.
B and C were specified in full, including falsifiers. That matters for one reason
above all: **rejecting A2 says nothing whatever about B or C.**

## Status

| | Approach | Rank | Status |
|---|---|---|---|
| **A** | Balanced readout via per-cell dual shifts (semi-discrete OT at extraction) | 1 | **Shipped** — `src/partition/balanced_readout.py` |
| **A2** | *Extension of A*: soft continuous area constraint, delete the iterative projection | — | **Rejected, measured** — `docs/experiments/05-soft-area-constraint/`, three arms (soft, hybrid, adaptive switch) |
| **B** | Volume-constrained multiphase threshold dynamics (auction-dynamics MBO) | 2 | ✅ **BUILT AND MEASURED (2026-08-19)** — `src/partition/mbo_auction.py`. Succeeds on both anchors: N=100 −0.455% / −0.590% at 211–225×, N=300 −1.044% at 318.8×. **0 FRAGMENTED at both N on RAW labels, no readout needed** — connectivity being the only gate with content for B (dormant is vacuous on one-hot labels; area is what B optimizes). Raw PGD gives 2 fragmented at N=300. See `docs/plans/PHASE1_BC_REPLACEMENT_PLAN.md` Phase A |
| **C** | Capacity-constrained geodesic power diagram + Lloyd | 4 | **Never started.** But its *step 0* was measured as B's attribution control: balanced geodesic init + Phase 2 = **189.65, i.e. +2.38% above the PGD control** (a PARTIAL verdict) for 67 s. Real C iterates Lloyd so that is a **lower bound** on C — but C's starting point sits behind both B and PGD |
| **D** | Geometry/mesh policy — aspect ratio, spatially varying ε, coarse-level floor | 5 | **(iii) measured** — `docs/experiments/06-subfloor-ladder/`. (i) r/R sweep and (ii) hexagonal commensurability untested |
| **E** | Discrete repair — connectivity-preserving relabeling | 3 | **Shipped** — the connectivity repair inside the balanced readout |

## The gate on B and C has been met

The 2026-08-13 briefing that opened the A2 work ends:

> "Do not start work on approaches B (auction-dynamics MBO) or C
> (capacity-constrained geodesic Lloyd). A2's result determines whether either is
> needed."

A2 came back rejected on two independent defects. **The gate was met on
2026-08-12 and never recorded anywhere outside that conversation** — which is why
B and C then went missing for two days. That is the gap this document closes.

## What the 2026-08-14 work changes

Everything done on 2026-08-14 lives inside **approach D(iii)** — the coarse-level
resolution floor — which the original taxonomy ranked **last of five** and
described as *"pure variance reduction — it cannot beat the √(2 ln N) tail drift,
so it is a supporting lever, never the fix."* The measurements independently
corroborate that low ranking:

- **D(iii) does not fix N=300.** The λ=11.5 control's coarse levels (32 and
  83 verts/cell) die in under 60 iterations and change nothing (230 → 234
  imbalanced — in fact L0 flips *zero* argmax labels); level 2 does all the work,
  reaching 36.15% worst; levels 3–4 give decelerating returns to 24.81%. Note
  level 2 is **not** shown to have converged — its line search floored at 6,628
  with the energy still falling and ‖g‖ = 31.11 — so "N=300 is energy-limited" is
  consistent with these data, not established by them. See §4b of
  [`winner_take_all_partition_gap.md`](winner_take_all_partition_gap.md).
- **The floor is real; its location is not established.** D(iii) proposed a floor
  of ~150–300 verts/cell. What is measured is a *lower* and different boundary —
  where a level stops running at all, doing *no* work rather than noisy work — and
  report 06 measures the cost at N=100: a permanent 15.92% runt against a
  control's 0.78%. **But the "83–92 verts/cell" figure previously stated here is
  refuted** (2026-08-15 adversarial review): `run_20260629_141012` is 96 v/cell and
  dies at onset 32, its λ=2.1 being out of window. Verts-per-cell is causally
  implicated by exactly one controlled pair (report 06's own arm); everything else
  pooled into that range varies in λ and straddles the 2026-07-06 energy fix.
  Treat it as a rough prior, never a design constant.

**The apparent contradiction in D(iii) is RESOLVED (2026-08-15, adversarial
review) — both measurements are correct, and they are different runs.** D(iii)
asserts *"your §4b data show splits are born at level 0 (8 → 6 → 3 → 3 across
levels)."* The 2026-08-14 reconstruction of the N=300 λ=11.5 control finds 0
fragmented after levels 0 and 1, with the two splits appearing **mid-level-2** (at
iterations 2,500 and 4,500) while the imbalanced count falls 234 → 10. Both
recompute exactly:

- **8 → 6 → 3 → 3** is **N=200 `run_20260722_121925`**, which had the
  **territory-aware machinery active** (its levels 0 and 1 ran the 30,000-iteration
  cap and never floored). Its level-0 splits are therefore evidence about **the
  discrete-area trim**, not about resolution.
- The N=300 λ=11.5 figures are the **plain energy**, a different run entirely.

So D(iii)'s *citation* was accurate for the §4b of its time; its **attribution of
those splits to resolution** was not, and report 06's refutation of that
attribution stands. There was never a measurement conflict to settle — the
resolution was sitting in §4b's own first paragraph ("Where it was first seen:
N=200, `run_20260722_121925`"). Cheap cross-checks between our own documents are
the lesson here, not more measurement.

**And it sharpens §4's most pointed claim.** The proposal's framing point 5 —
*"at high N you may not need the Γ-relaxation at all… run the C-vs-relaxation
comparison at N=100 before investing further in relaxation-side machinery; if the
gap after Phase 2 is <1%, the N=1000 plan should be B or C + Phase 2, with the
relaxation retired above N≈200"* — was written before four arms failed to make the
N=300 ladder cheaper and before D(iii) was measured. Every result since has
pointed the same way.

## ⚠ A citation error that originated in the archived proposal

The proposal attributes auction dynamics to **"Jacobs–Kim–Léger"**. That is
**wrong**. The paper — *Auction Dynamics: A Volume Constrained MBO Scheme*, J.
Comput. Phys. **354** (2018) 288–310, doi 10.1016/j.jcp.2017.10.036 — is by
**Matt Jacobs, Ekaterina Merkurjev and Selim Esedoğlu**. Verified 2026-08-21
against dblp (`journals/jcphy/JacobsME18`) and ScienceDirect; volume, pages and
year were correct all along.

The archived proposal is **left verbatim**, because it is a record of what was
said. But the error propagated from it into seven downstream documents — the two
experiment reports, the B/C plan, this file, `CLAUDE.md`, and the arm's own module
docstring — and survived roughly two weeks and six adversarial reviews before
anyone checked an author list. Two things worth taking from that. Every review so
far attacked *measurements*; none attacked *citations*. And the correct attribution
is materially more informative: **Esedoğlu is an author of both the
threshold-dynamics paper and the auction-dynamics paper**, so approach B is one
group extending its own work, not two independent lineages meeting.

## What B and C actually are

Both **replace Phase 1 outright** rather than adjusting it. Full mechanisms,
costs, and falsifiers are in the archived original; the essentials:

**B — auction-dynamics MBO** (Esedoğlu–Otto threshold energies; Jacobs–Merkurjev–Esedoğlu,
JCP 2018, published for exactly this problem class). Iterate: (1) diffuse each
indicator for time τ by solving `(M + τK) y_k = M χ_k`, one prefactorized sparse
Cholesky per level; (2) reassign every vertex by *balanced* thresholding —
literally approach A applied to `y` instead of `u`:
`ω(i) = argmax_k [y_ik + ψ_k]`. It does not close the gap, it **abolishes the
category**: there is never a continuous field whose readout can diverge, and the
iterate after every step is a hard partition with equal discrete areas **to the
solver's convergence, not exactly** — measured 2026-08-16, per-step balance in an
iterated MBO loop settles at ~1.4–2.0% worst deviation at N=300, against a
one-vertex granularity floor of ~1.0%. Good enough in practice, and the "exactly"
here was wrong as written.
Stray islands shrink to extinction under mean-curvature flow — the opposite
dynamic to the trim. Seeded init, the mesh ladder, M/K/v, the HDF5 formats and
Phase 2 all carry over unchanged. Estimated ~100× faster than PGD. Known failure
mode: mesh pinning when √τ ≲ h, so set τ ~ h². **Falsifier:** a ~150-line
prototype at level 2 (V=47,488), N=300, 50 iterations; all three gates plus
perimeter against the λ=12 PGD solution; refuted if perimeter is >3–5% worse
after Phase 2.

**C — capacity-constrained geodesic Lloyd.** Drop the field entirely. N sites
(start from the farthest-point seeds); geodesic distances via the heat method or
Dijkstra; assignment = balanced transport with cost d² — the same dual-shift
solver as A, making it a geodesic power diagram with prescribed capacities
(Aurenhammer/Mérigot); Lloyd step moves each site to its cell's v-weighted
centroid; ~30 iterations, labels straight to Phase 2. Exact discrete balance by
construction, tail-immune, trivially scales to N=1000. Its real value is as the
"shame test" of framing point 5.

Note A, B and C **share one solver** — balanced assignment given per-vertex
scores — which already exists and is validated in `balanced_readout.py`. So B's
thresholding step is code the project has already shipped and measured.

## The filter any new mechanism must pass

From the proposal's own process note, and now three-for-three at predicting
failures (territory-aware term, A2, A2's adaptive switch):

> **Through what operator does the correction reach the field, and is that
> operator local?**

The projection is nonlocal — that alone predicted islands. A's boundary-monotone
growth and B's curvature flow both pass by construction. Apply it as the first
filter, before any code.

## Related documents

- [`phase1_highn_proposal_ABCDE_original.md`](phase1_highn_proposal_ABCDE_original.md) — the proposal, verbatim.
- [`winner_take_all_partition_gap.md`](winner_take_all_partition_gap.md) — the standing explanation; §4b holds the N=300 ladder reconstruction, §4c the locality criterion.
- `docs/experiments/05-soft-area-constraint/` — A2, measured and rejected.
- `docs/experiments/06-subfloor-ladder/` — D(iii), measured.
- `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` — the *separate* P1–P5 enumeration from a different agent (P1 territory-aware: implemented and rejected). Do not confuse P1–P5 with A–E.
- Code: `src/partition/balanced_readout.py` (A and E, shipped; also B's thresholding step), `src/optimization/initialization.py` (the seeding C generalises).
