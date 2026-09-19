# The balanced readout, explained from zero

*Written 2026-09-17 as speaker preparation for `docs/presentations/01-pgd-status-brief/`
(slide 12). A plain-language, step-by-step account of how the balanced readout
turns a relaxation solution with runts and split cells into an equal-area,
connected partition — without touching the relaxation. Every number is from the
n = 300 run `run_20260806_123326` (47,488 vertices) and its
`readout/dualshift_gate0.05_repair/`. The code is
`src/partition/balanced_readout.py`; the mathematics is
`docs/math/09-balanced-readout/`; the companion note on the artefacts it fixes is
`three_artefacts_of_the_readout.md`.*

## 0. What "at extraction time" means, and why fix it there

The relaxation ends with a *paint* field: for every cell $i$ and every mesh vertex,
a number $u_i \in [0,1]$ saying how strongly the cell claims that vertex. That is
not yet a partition. The partition is made by the **readout**: at each vertex,
give the vertex to the cell with the most paint. Everything downstream (contour
extraction, the perimeter optimisation, the export) starts from those labels.

The three artefacts — empty, runt, split cells — are all defects of *territory*,
and territory is decided only at the readout. The energy never sees it. So there
are two places one could intervene:

- **inside the relaxation**, by adding something to the energy or the constraints
  that cares about territory. This was tried (the "territory-aware" energy) and it
  failed: enforcing territory through the non-local projection manufactured
  14 split cells out of 200, and the ladder became ~20 days long;
- **at the readout**, by changing the rule that turns paint into labels. This is
  the balanced readout. It costs ~17 s, leaves the relaxation untouched, and
  produced the valid n = 300 partition.

The whole idea: *keep the paint, change the rule.*

## 1. The rule change — a handicap per cell

The plain readout is "the vertex goes to the largest $u_i$". The balanced readout
gives every cell a number $\psi_i$ — one per cell, 300 numbers at n = 300 — and
reads

$$
\text{vertex } x \;\to\; \arg\max_i \big[\, \log u_i(x) + \psi_i \,\big].
$$

Think of $\psi_i$ as a **handicap** in a race, or a per-cell **multiplier on its
paint**: adding $\psi_i$ to $\log u_i$ is the same as multiplying $u_i$ by
$e^{\psi_i}$ before comparing. A cell that ended up with too little territory gets a
positive $\psi_i$ — its paint now counts for more — and wins more vertices. A cell
with too much gets a negative one.

The crucial property is **where** the extra wins happen. A vertex changes hands
only if the handicap is enough to overturn the comparison, and that is only
possible where the comparison was close to begin with — at the cell's own border,
in the transition band, where its paint is 0.4 against a neighbour's 0.45. Deep
inside a neighbour's core, where the cell has 0.02 against 0.95, no reasonable
handicap flips anything. So raising $\psi_i$ **grows cell $i$ outward from its own
core, one ring of border vertices at a time**. This is the locality that the
projection lacks: the projection adds paint *everywhere*, including far away where
it plants islands; the handicap changes labels *only at the border*.

A two-vertex example. At a border vertex the paint is $(0.45, 0.40, 0.15)$ for
cells 1, 2, 3: cell 1 wins. Give cell 2 a handicap $\psi_2 = 0.2$: the scores are
$\log 0.45 = -0.80$ against $\log 0.40 + 0.2 = -0.72$, so cell 2 now wins it. At a
vertex inside cell 1's core the paint is $(0.90, 0.05, 0.05)$: $\log 0.90 = -0.11$
against $\log 0.05 + 0.2 = -2.80$; cell 1 keeps it comfortably. The same $\psi_2$
moved the border and did nothing to the interior.

Why the logarithm rather than the raw paint? Three reasons. Adding to $\log u$ is
multiplying $u$, so the handicap is a *relative* boost and behaves the same for a
cell whose paint is at 0.4 and one at 0.04. A vertex where the cell has exactly
zero paint stays unwinnable ($\log 0 = -\infty$; the code floors it at $10^{-300}$),
so a cell can only expand into places where it already has *some* claim. And the
log-plus-offset form is exactly the structure of the optimal-transport dual that
the next section solves.

## 2. Finding the handicaps — a thermostat loop

Nobody guesses $\psi$. It is found by a loop that behaves like 300 thermostats:

1. Start with every $\psi_i = 0$ (that is the plain readout).
2. Read out the labels with the current $\psi$ and measure each cell's territory
   $T_i$ (the summed vertex areas it won).
3. For each cell, compare $T_i$ with the target $A/n$. If the cell is short, raise
   $\psi_i$ in proportion to the shortfall; if it is over, lower it:
   $\psi_i \leftarrow \psi_i - \eta_t\,(T_i - A/n)/(A/n)$.
4. Shrink the step a little ($\eta_t = 0.5/(1 + 0.02\,t)$) and go back to 2.

Run 400 rounds and keep the best labelling seen (the one with the smallest worst
deviation). Each round is one argmax over the $V \times n$ score matrix, so the
whole loop is seconds.

For the mathematically minded: this is subgradient ascent on the (concave) dual
of a transportation problem — "assign vertices to cells so that every cell gets
area $A/n$, changing the source scores as little as possible" — and the $\psi_i$
are its dual variables, one per capacity constraint. The thermostat update *is*
the dual subgradient. One does not need this to use it, but it is why the loop
converges and why there is nothing to tune per n.

Two limits, both honest and both small:

- **Vertices are indivisible.** A cell's territory can only change by whole
  vertex areas, so equality is reached only up to about one vertex. At n = 300 on
  47,488 vertices a cell is ~158 vertices, so one vertex is ~1 % of a cell: the
  shifts cannot do better than ≈1 %, and they reach 2.1 %.
- **It fixes areas, not shapes.** The handicap only re-labels vertices; it does
  not create or move geometry. And nothing in the loop knows about connectivity —
  which is why there is a second stage.

**What it did at n = 300.** Starting from 10 cells outside the ±5 % gate with the
worst at −36 %, the shifts end with **0 cells outside and a worst of 2.1 %**. The
handicaps needed ranged from $\psi = -0.75$ to $+3.46$; the largest is a
$e^{3.46} \approx 32\times$ multiplier — the worst runt needed its paint counted
32 times over before it could claim its fair share of border vertices. The label
boundary grew only 0.5 % (189.7 → 190.7), because the changes are at borders, not
islands.

## 3. The catch — shifts can *worsen* connectivity on the way

The shifts do not create islands the way the projection does, but they can
**multiply** ones that already exist, and they can plant specks. A runt with a big
handicap wins every near-tie it has anywhere on the surface; most near-ties are
at its border, but a few are in far-away bands where its paint happens to be close
to the locals'. In the n = 300 run the two split cells went from **2 pieces each to
4 pieces each** after the shifts. So the readout cannot stop here: **the repair
stage is necessary, not cosmetic.**

## 4. Repair, stage A — give every island to its best neighbour

For each cell, take the vertices it owns and split them into connected pieces on
the mesh graph (edges of the triangulation, so the torus wrap-around is respected
and a piece is never fooled by the flat parametrisation). Keep the largest piece as
the cell. Every other piece is a stray, and it is handed to the **neighbouring cell
it shares the longest border with**, measured as the summed length of the mesh
edges between the piece and that neighbour.

This is unconditionally safe: gluing a piece onto a cell it already touches cannot
disconnect that cell, and the donor loses only a piece that was disconnected from
it anyway. After stage A every cell is one piece by construction. (If a cell were
nothing but strays it would end up empty and the report says so; this has not
happened.)

**At n = 300:** 6 islands absorbed — 3 from cell 274, 3 from cell 290 — totalling
about half a cell's area; the largest was 75 vertices of cell 290 given to cell
197, the smallest 2 vertices.

## 5. Repair, stage B — restore equal areas one border vertex at a time

Stage A unbalances things again: the cells that swallowed islands are now too
big, the cells that lost them too small. Stage B fixes that with the smallest
possible operation: move **one border vertex** from a richer cell to a poorer
neighbour, repeatedly. Three rules govern each move of vertex $x$ (area $v_x$) from
donor $d$ to receiver $r$:

- **Only if it helps.** The move must reduce the total imbalance
  $\sum_i (T_i - A/n)^2$, which works out to exactly the condition
  $T_d - T_r > v_x$: the donor must be richer than the receiver by more than the
  vertex being moved. So the repair never overshoots.
- **Cheapest first.** Among all allowed moves, do first the one whose vertex
  already prefers the receiver most — the largest $\log u_r(x) - \log u_d(x)$.
  These are vertices the paint was nearly giving to the receiver anyway, so the
  new border stays where the relaxation wanted it and the boundary length grows
  as little as possible.
- **Never disconnect anyone.** The receiver stays connected automatically (the
  vertex touches it). The donor is checked *exactly*: if removing the vertex would
  cut the donor in two, the move is skipped and counted as blocked.

A sweep tries every allowed move once, in that order; sweeps repeat until no move
helps, with a cap of 200. The repair reports its own strain — moves made, moves
blocked, sweeps used, whether it hit the cap — so a solution that is beyond
repair says so instead of silently returning a degraded partition.

**At n = 300:** 1,485 single-vertex moves in 38 sweeps, 1 move blocked, final
worst deviation **1.63 %**, all 300 cells connected. The total boundary ended at
193.3, +1.9 % over the raw readout — the price of connecting two cells and
rebalancing 300 — and the perimeter optimisation that follows takes that back
and more.

## 6. Handing the result on

The stage writes a new solution file, in the same format the relaxation writes,
and never modifies the source. Because everything downstream reads densities only
through "largest value wins", the new labels are encoded by **swapping** two
numbers at each relabelled vertex: the old winner's paint and the new winner's
paint exchange places. Row sums are untouched (the partition of unity still
holds), the new label is now the largest value, and the swap is reversible given
the stored labels — the original field can be recovered exactly. The $\psi$
vector is stored alongside for the record. At n = 300, 1,840 of 47,488 vertices
(3.9 %) were relabelled in total.

From there the contour stage runs completely unchanged and reaches perimeter
323.32 on this mesh (322.96 on the 114,144-vertex one, whose raw readout had 3
runts and 2 splits and was repaired the same way to a worst cell of 0.64 %).

## 7. Why this works where fixing the energy did not

Both approaches enforce the same thing — equal territory — and the difference is
entirely *which operator* enforces it:

| | inside the relaxation (rejected) | at the readout (shipped) |
|---|---|---|
| lever | the projection, which adds paint to a cell **everywhere** | a handicap, which flips labels only at **near-ties = the cell's border** |
| side effect | plants islands far from the core: 14/200 split cells | grows the cell outward from its core; islands only *multiplied*, then removed by repair |
| cost | every gradient step of every level; ladder ~20 days at n = 200 | ~17 s once, after the relaxation |
| exactness | approximate, fights the energy | equal to one-vertex granularity, at any n |

The one-line version: the projection asks "where can this cell get more paint?"
and the answer is "anywhere"; the handicap asks "where is this cell already almost
winning?" and the answer is "at its own border". Balance enforced through a
non-local operator buys area with far-away territory; enforced through a local
one it buys it next door.

## 8. What it does not do

- It does not make an **empty** cell appear in a sensible place. A cell with no
  core has no border to grow from; its near-ties are scattered specks. Empty
  cells are prevented earlier, by the seeded start.
- It does not improve the **geometry** of a cell — a runt made round by 32× paint
  is still where the relaxation left it. Shapes are the contour stage's job.
- It cannot balance below **one vertex** of area, and it adds boundary length
  (+1.9 % here) that the contour stage must then remove.
- Its strain (moves, blocked moves, sweeps, cap) is the honesty signal: a readout
  that reports hitting the sweep cap or many blocked moves is telling you the
  relaxation left more than a readout can fix.

## The one-paragraph version for the talk

The artefacts are decided at the readout, so that is where we fix them. Every cell
gets a handicap $\psi_i$, and a vertex goes to the cell with the largest
$\log u_i + \psi_i$. Because a handicap can only overturn near-ties, raising a
runt's $\psi_i$ grows it outward from its own border — locally, unlike the
projection. A simple loop sets the 300 handicaps so every cell's territory hits
the target, to within one vertex. Then, since nothing in that loop knows about
connectivity, a repair pass gives each stray island to the neighbour it touches
most and moves single border vertices between cells until the areas are equal
again, never cutting a cell in two. Seventeen seconds, the relaxation untouched:
10 runts and 2 split cells become 0 and 0, and the contour optimisation runs
unchanged.
