# Relaxation versus threshold dynamics, explained from zero

*Written 2026-09-17 as an orientation note for the author, before studying the
two Phase 1 methods in detail: what each one holds in its hands while it
searches, why the first stopped working for us at many cells, why the second
does not have that failure, what in all of this is ours and what is taken from
the literature, and how it can honestly be positioned as new work. Every number
is from `docs/experiments/08-mbo-auction-dynamics/` (report 08) and
`CLAUDE.md`'s deliverable tables unless a source is named; the mathematics is
`docs/math/06-phase1-energy-discretization/` (method A),
`docs/math/09-balanced-readout/` (its failure) and
`docs/math/10-mbo-auction-dynamics/` (method B, with the prior-art table). The
companion notes `three_artefacts_of_the_readout.md` and
`balanced_readout_explained.md` go deeper into method A's failure and its
repair.*

## 0. The problem, in one sentence

Take a closed surface — for us, a torus — and cut it into $n$ pieces of equal
area so that the total length of the cutting lines is as short as possible.
Every vertex of the triangle mesh must end up belonging to exactly one piece.

Both methods below are ways of *searching* for that partition. Everything that
matters about the difference between them comes down to one question: **what
does the method hold in its hands while it searches?**

## 1. Method A — the Γ-relaxation (what we used first)

**What it holds.** For every cell $i$, a *fuzzy* membership map $u_i$: a number
between 0 and 1 at every vertex, "how much this vertex belongs to cell $i$". At
each vertex the $n$ numbers add up to 1. So a vertex can be 0.6 cell 3, 0.3
cell 7, 0.1 cell 12. The other notes call this *paint*.

**What it minimises.** An energy with two competing terms. One term punishes
*sharp changes* of $u_i$ across the surface — it wants the maps smooth. The other
punishes *fuzziness* — it wants every value to be exactly 0 or 1. The cheapest way
to satisfy both at once is a map that is 1 inside a region, 0 outside, with a
thin blurred band along the boundary; and the energy of that band is
proportional to the *length* of the boundary. That is the Modica–Mortola /
Γ-convergence idea: minimising this energy stands in for minimising perimeter,
and the theory says the stand-in becomes exact as the band width $\varepsilon$
goes to zero. Equal areas are enforced as a constraint on the *amount of paint*
each cell has, $\sum_j v_j u_{ij} = |S|/n$, where $v_j$ is the little patch of
area belonging to vertex $j$.

**How it searches.** Gradient descent. Nudge all the maps a little downhill in
energy, then *project* them back onto the constraints (paint sums to 1 at every
vertex; every cell has equal paint), repeat — thousands to 30,000 times — then
move to a finer mesh and repeat. At the end, read off the partition by giving
each vertex to the cell with the most paint there: the winner-take-all
("argmax") readout.

**Why it went wrong for us at many cells.** These are the things the paper needs
to report, and each is measured:

1. *The readout gap.* The constraint is on the paint, but what you deliver is
   the territory — and the two coincide only when every map is exactly 0/1. At
   large $n$ some maps stay diffuse, and a cell can have exactly the right
   amount of paint while owning a third of the territory it should: a "runt".
   Nothing in the energy sees territory, so nothing prevents this, and a finer
   mesh does not reliably fix it. (`three_artefacts_of_the_readout.md`;
   doc 09.)
2. *Split cells.* Nothing in the energy penalises a cell being in two pieces.
   At $n = 200$–$300$ some are, and refinement makes the islands *grow*.
3. *Cost.* Projecting back onto the constraints was 93 % of the running time.
   $n = 100$ took 13.4 hours; $n = 300$ on the fine mesh took 57 hours and still
   failed two of the three validity checks.
4. *A knob that has to be tuned per $n$.* The fuzziness penalty $\lambda$ has a
   working window — too low gives runts, too high gives mush — and the window
   moves with $n$ (≈5 at $n = 100$, ≈11.5 at $n = 300$). Coarse mesh levels can
   also "die" after a few iterations without doing any work.

We tried to repair this two ways. Changing the energy so it cares about
territory (the "territory-aware" relaxation) made things worse — 14 of 200 cells
split — and was removed. Fixing the *readout* instead (the balanced readout,
`balanced_readout_explained.md`) works, costs 17 seconds, and gave the only
valid $n = 300$ partitions we have. But it is a repair of the output, not of the
method, and it left the cost problem untouched.

## 2. Method B — threshold dynamics with balanced assignment (what works)

**What it holds.** A *hard* label at every vertex. Vertex $j$ belongs to cell
$k$, full stop. There is no paint, so there is no readout, so there is no readout
gap: **what you constrain is exactly what you deliver.** Failure 1 above is not
fixed by method B — it is *absent* by construction.

**One step, in two moves.**

1. *Blur.* For each cell, take its 0/1 indicator (1 on the cell's vertices, 0
   elsewhere) and let it diffuse for a short time $\tau$ — think of
   Gaussian-blurring a black-and-white image. On a mesh this is one linear
   solve per cell, $(M + \tau K)\,y = M\chi$, with the matrix factorised once per
   mesh level and reused for all $n$ cells.
2. *Reassign.* Give each vertex to the cell whose blurred value is largest
   there — but first add a per-cell **handicap** $\psi_k$, chosen so that every
   cell ends up with exactly its target area. The handicaps are prices: a cell
   that came out too small gets a bonus and wins more vertices; one too large
   gets a penalty. Finding the $n$ prices is a small transportation problem —
   whose problem and whose solver this is matters for the paper, and is the
   subject of §2b.

Repeat until the labels stop changing (30–66 steps in every run we have), carry
the labels to the finer mesh by nearest vertex, and repeat there.

**Why blur-then-threshold shortens boundaries.** Blurring averages a function
over a neighbourhood. At a point where a cell's boundary *bulges outward*, most
of the neighbourhood lies outside the cell, so after blurring the cell's value
there drops below its competitors' and the point is lost. At a *dent* the cell
gains. Bulges retreat, dents fill in: the boundary moves inward in proportion to
its curvature. Motion by curvature is exactly the motion that reduces length
fastest — it is the gradient flow of perimeter. That is the 1992
Merriman–Bence–Osher observation. Esedoğlu and Otto showed in 2015 that each
blur-and-threshold step is a descent step on an energy which, like method A's,
tends to perimeter as $\tau \to 0$; and the handicaps turn curvature flow into
*area-preserving* curvature flow (Jacobs, Merkurjev and Esedoğlu, 2018), which
is precisely our problem.

**Why it is fast and robust.** No line search, no projection loop, no
$\lambda$. The one parameter, $\tau$, is *derived* from the mesh and $n$ at each
level, not tuned: too small and nothing moves (the blur does not reach the next
vertex — the scheme "freezes"); too large and a cell's own signal is smeared over
its whole territory (it "over-merges"). That is a two-sided window we
characterised and stay inside. $n = 100$ takes 4 minutes instead of 13 hours;
$n = 1000$ takes about an hour and passes all three validity checks on the raw
labels, with no repair.

**Two honest caveats.** Connectivity is not enforced by method B either — a
cell *could* come out in two pieces. In the *final* labels it never has, at any
$n$ we ran (method A gave 2 to 14 split cells); at intermediate mesh levels one
to three have appeared — most on a level whose $\tau$ sat below the freeze
window — and the next level healed them every time. And the theory's "each step decreases the energy"
guarantee does not strictly transfer to our implementation, for two small
technical reasons (doc 10, §7); we gate on a weaker inequality that does hold.

## 2b. The prices: whose problem, whose solver

The sentence "finding the prices is a small transportation problem, solved by
the same routine the balanced readout already used" packs together one thing
that is the authors' and one thing that is ours. Kept apart:

**The problem is theirs.** Jacobs, Merkurjev and Esedoğlu (2018) showed that
once the equal-area constraint is added, "give each vertex to the largest
blurred value" becomes a *linear assignment problem*: assign points to cells to
maximise the total blurred score, subject to every cell receiving exactly its
quota. They then observed (their eq. (11)) that the solution is again a
thresholding, with each cell's score *shifted* by that cell's Lagrange
multiplier — one number per cell. Those multipliers are exactly our handicaps
$\psi_k$. So "there are $n$ prices and the balanced threshold is
$\arg\max_k [y_{ik} + \psi_k]$" is their structure. (For two cells it is older
still: Ruuth and Wetton, 2003, shifted the single threshold level away from
$\tfrac12$ until the volume came out right.)

**Their solver.** They solve that assignment problem *exactly* with Bertsekas'
auction algorithm (1988) — the operations-research procedure in which
unassigned "persons" bid for "objects" and prices rise until everyone is
matched; hence the name *auction dynamics*. Exactness is what earns them a
theorem: every step provably decreases the energy.

**Our solver.** We solve the *dual* of the same linear programme by subgradient
ascent: guess the prices, compute the areas they produce, raise the price of
every cell that came out too small and lower it for every cell that came out
too large, repeat with a decaying step, keep the best iterate. That is
`solve_dual_offsets` in `src/partition/balanced_readout.py`, written for the
balanced readout — the repair of method A — before method B existed. Method B
reuses it unchanged, feeding it blurred indicators instead of log-densities (the
one addition was a normalisation of the score scale so the step schedule
transfers). It is *inexact*: it stops at a quality bar set by the one-vertex
granularity and leaves a residual area error.

**Why our problem is not quite their problem.** On their uniform grids every
grid cell has the same volume, so the problem is a true *assignment* (equal-size
sets, integer quotas). Our mesh vertices carry unequal areas $v_j$ — the lumped
mass — so ours is a *transportation* problem with real-valued supplies. Doc 09
proves the consequence: the optimum is generically fractional, so exact equal
areas are *unattainable* with whole vertices, which is why we work to a
granularity bar rather than to zero.

**What the difference costs and buys.** It costs the theorem: the "each step
decreases the energy" guarantee does not transfer to an inexact solve (doc 10
§7.2, item 2; item 1 is a separate finite-element issue). We gate instead on a
weaker inequality that holds for *any* priced-argmax labelling, exact or not,
and we measured the slack it leaves. That is a genuine loss of guarantee and the
paper should say so. It buys simplicity: nothing new had to be written for
method B's inner loop, one primitive serves approaches A, B and C, and it
converged in every run (30–66 steps per level).

**Is this a new approach to the dual transportation problem?** No, and it
should not be described as one. Subgradient ascent on the transportation dual —
"grow the cells that are too small" — is the standard method for this class of
problem (Aurenhammer, Hoffmann and Aronov 1998; Mérigot 2011; Peyré and Cuturi
§5), and doc 09 proves that our update *is* that subgradient. Jacobs et al.
themselves note that in the two-phase case the multiplier search can be
replaced by a sort. What is *adapted*, and what we have not found reported, is
the substitution itself: an inexact, non-uniform-mass transportation solve
inside the MBO step in place of the exact auction, together with the analysis
of what the inexactness costs — the surviving inequality, the granularity bar,
the non-integrality. That is reportable as an **implementation choice with an
analysis attached**, in one paragraph and a proposition, not as a method
contribution and not in the title.

**What "adapted to our problem, and why" actually consists of.** The
contribution is not the solver but the *judgment* of using it here, and that
judgment has definite content, each point documented and measured:

1. *The problem had to be adapted anyway.* JME's auction assumes every point
   carries the same volume — a true assignment with integer quotas. Our mesh
   vertices carry unequal areas, so ours is a transportation problem with real
   supplies. Whatever solver we chose, that adaptation was forced by the mesh,
   and its consequence — no exact balance is attainable with whole vertices, so
   a granularity bar is the correct stopping rule — is ours (doc 09).
2. *The primitive already existed and was already validated.* The dual-ascent
   routine was built and qualified for the readout repair (report 07: the
   harness reproduces the reference perimeter bitwise; gate 2 discriminates).
   Reusing a validated component instead of writing a new one is a defensible
   engineering decision, and the paper can say so.
3. *Inexactness is acceptable because the scheme is iterated.* That is the
   Phase 0 finding: the solver only looked inadequate on a cold-start test that
   no iterated method ever faces; from step 1 onward every assignment starts
   from an already-balanced partition and the routine converges under the bar
   every time. The judgment is conditional on that structure, and it was
   checked rather than assumed.
4. *The cost is bounded, not hand-waved.* The lost theorem is replaced by an
   inequality that holds for any priced labelling (gate G4), with the slack
   measured across every run. A reader can see exactly what was given up.

Together that is a contribution of *judgment with evidence*: why a standard
solver was the right choice in this position, what it required, what it cost,
and how we know. One paragraph and a proposition is the right weight for it.

**The one limit.** We can say the choice *sufficed* and what it cost; we cannot
say it was *better* than the auction, because an auction was never implemented
or measured on this problem. "We chose this for these reasons and it worked" is
the honest sentence; "this beats the auction" is not available to us. And "new"
means new in the papers we hold — JME themselves already vary the assignment
step on graphs, and the two-phase literature replaces the search with a sort —
so it is a modest, local novelty.

One piece of history belongs here. In Phase 0 (report 07) we very nearly *did*
build an auction solver, on evidence that method B's assignments needed
something stronger than the subgradient routine. That evidence was a fixture
artefact — a cold-start test that no iterated method ever faces. Sinkhorn, the
entropic alternative, was implemented and measured and *lost* to the subgradient
routine on five of six score matrices. An auction was never implemented. The
honest sentence for the paper: *the balanced threshold is JME's; we solve its
transportation dual inexactly by subgradient ascent, with the routine built for
our readout repair, rather than exactly by auction, and we quantify what that
costs.*

## 3. Side by side

| | Γ-relaxation (A) | Threshold dynamics (B) |
|---|---|---|
| What it holds | $n$ fuzzy maps in $[0,1]$ | one hard label per vertex |
| What stands in for perimeter | energy of the blurred boundary band | heat that escapes a cell in time $\tau$ |
| What the area constraint acts on | the paint (fuzzy mass) | the delivered territory itself |
| One step | gradient step + iterative projection | blur ($n$ linear solves) + priced reassignment |
| Steps per mesh level | thousands, up to the 30,000 cap | 30–66 |
| Knobs | $\lambda$ per $n$, $\varepsilon$, step size, patience… | $\tau$ (derived); three constants never changed |
| Readout gap (runts) | yes — the central failure | none, by construction |
| Split cells | not enforced; 2–14 in the final labels | not enforced; 0 in the final labels at every $n$ (1–3 born mid-ladder, healed) |
| Phase 1 time, $n = 100$ / $300$ | 13.4 h / 57 h (and invalid at 300) | 228 s / 248 s |
| Largest valid $n$ | 300, only with the readout repair | 1000, raw |
| Final perimeter at equal Phase 2 budget | 185.25 / 323.32 | 184.41 (−0.46 %) / 319.94 (−1.04 %) |

The perimeter margin is real but small; the *validity* and *cost* margins are
the story.

## 4. What is ours, and what is not

Neither method is ours, and the write-up should say so in the first paragraph.

- Method A is Bogosel and Oudet's 2017 method, implemented faithfully
  (`docs/papers/`, `bogosel2017partitions`).
- Method B is Merriman–Bence–Osher's scheme (1992/1994) with the
  Esedoğlu–Otto variational reading (2015) and Jacobs–Merkurjev–Esedoğlu's
  area constraints (2018). That last paper already tiled the *flat* torus into
  64 equal cells with it. The implicit-Euler blur we use on the mesh is the one
  the graph-MBO literature uses. Merriman and Ruuth ran the multiphase scheme on
  a *curved* torus in 2007, unconstrained. An earlier description of our work
  overstated its novelty on three of four counts; doc 10 §2 has the corrected
  table, read from the papers themselves.

What *is* reportable as new work, ranked:

1. **A diagnosis.** Why the Γ-relaxation stops producing valid partitions at
   large $n$: the readout gap, with the exact identity behind it and the
   measured artefacts. We have not found this reported for Γ-convergence
   partition methods, and it concerns anyone who uses them beyond a few dozen
   cells.
2. **A working pipeline at a scale nobody has shown.** Equal-area
   minimal-perimeter partitions of a *curved, triangulated* surface, with the
   diffusion done by surface finite elements, at $n = 400$–$1000$, coupled to an
   exact contour refinement (Phase 2) and three validity gates. The literature
   we hold has the flat torus at 64 cells, the curved torus at 5 unconstrained
   regions, and grain growth at $10^5$ unconstrained grains. Nobody has the
   combination.
3. **A head-to-head comparison on identical meshes at equal downstream
   budget** — cost, validity and final perimeter — with an attribution control
   (the dynamics beat their own initialisation by 2.8 %) and negative controls.
   The papers that proposed B never compared it with the Γ-relaxation.
3b. **An implementation choice with its price quantified:** the exact auction
   replaced by an inexact transportation-dual solve with real vertex masses,
   and the consequences (no dissipation theorem, a surviving inequality, a
   granularity bar, non-integrality) worked out — §2b. The contribution is
   the *judgment* — why this solver here, what it required, what it cost, how
   we know it sufficed — not the solver. A paragraph and a proposition, not a
   contribution to optimal transport, and never "better than the auction",
   which was not measured.
4. **Practical knowledge that transfers:** the $\tau$ window and what freezing
   and over-merging look like; that cost is governed by vertices-per-cell, not
   by (vertices × cells); how to build the mesh ladder; and that once Phase 1 is
   fixed, Phase 2 becomes the bottleneck.
5. **Two small mathematical observations** about doing this with finite
   elements instead of an FFT: the backward-Euler blur moves interfaces at half
   the speed the Gaussian would (its interface constant is $\tfrac12$ instead of
   $1/\sqrt\pi$), and the discrete energy is not guaranteed to decrease. Modest,
   but new and checked (doc 10 §5, §7, with a committed script that regenerates
   every number).

## 5. How to position it

A *computational* paper, not a methods paper: "we needed partitions with a
thousand equal cells; the standard method fails, for a reason we can name and
measure; an existing scheme from another community does it in minutes; here is
the evidence, the comparison and the recipe." Two rules for writing it, both
learned the hard way this week:

- never say "first" — say "we are not aware of", and name the corpus that was
  searched (doc 10 §2 lists it);
- keep the $n = 100$/$300$ *comparison* and the $n = 400$–$1000$ *existence*
  results separate. Above 300 there is no method-A baseline to compare against,
  because method A stopped producing valid partitions — and that absence is
  itself the finding, not a gap in the experiment.

## 6. Where to study it, in order

1. `three_artefacts_of_the_readout.md`, then doc 09 §1–4 — the failure of
   method A, with the pictures in report 01.
2. Merriman, Bence, Osher 1992 (`docs/papers/cam92-18.pdf`; 12 pages, very
   readable) — the blur-and-threshold idea.
3. Esedoğlu and Otto 2015, §1–4 — why one step is a descent step, and what
   energy it descends.
4. Jacobs, Merkurjev, Esedoğlu 2018, §1–3 and §4.2 — the area handicaps as
   prices, and the flat-torus tilings.
5. Doc 10 §4–6 — the same three things written for our mesh and our code; §2 for
   who did what.
6. Report 08 — the evidence.

## The one-paragraph version

Method A searches with fuzzy membership maps and reads the partition off at the
end; the area rule is enforced on the maps, not on what is read off, and at
many cells the two drift apart — runts, split cells, days of computation, a
knob to tune per $n$. Method B searches with hard labels, so there is nothing to
read off: each step blurs every cell for a moment and re-labels every vertex to
the cell whose blurred value wins after a per-cell price is added, prices set so
that all areas come out equal (the priced threshold is Jacobs–Merkurjev–Esedoğlu's;
the standard transportation-dual solver we set the prices with, in place of
their exact auction, is a choice we pay for with a lost theorem and account
for). Blurring moves boundaries by their curvature,
which is the fastest way to shorten them. Neither method is ours. What is ours
is the diagnosis of why A fails, the pipeline that takes B to a thousand
equal-area cells on a curved triangulated surface with an exact refinement
behind it, the head-to-head comparison on the same meshes, and the practical
rules for running it.
