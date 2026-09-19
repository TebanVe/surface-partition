# The three artefacts of the winner-take-all readout, in the simplest terms

*Written 2026-09-17 as speaker preparation for `docs/presentations/01-pgd-status-brief/`
(slides 9–12). Plain-language explanation of why the empty, runt and split cells
appear, what generates them, and why mesh refinement shrinks the runts but not the
splits. Numbers are from the two n = 300 runs `run_20260806_123326` (47,488
vertices) and `run_20260808_191030` (114,144 vertices); the island-size
measurement at the end was made for this note. The rigorous version is
`docs/reference/winner_take_all_partition_gap.md`.*

## The one idea underneath all three

Every cell is a layer of **paint** on the surface — the density $u_i$, between 0 and 1
at each point. The relaxation controls two things about the paint:

1. **Each cell gets the same amount of paint** (the area constraint,
   $\int_S u_i = |S|/n$). This is enforced *exactly*, after every step, by the
   projection.
2. **Paint is pushed to be either 0 or 1** (the double well), and **the transition
   zones should be short** (the gradient term).

But the partition is not the paint. The partition is **territory**: at each vertex,
whichever cell has the *most* paint there takes the vertex (the readout, $\phi_i$).
Nothing in the energy and nothing in the constraints ever looks at who has the most
paint anywhere. "Equal paint" is guaranteed; "equal territory" is not. All three
artefacts are ways for a cell to have the right amount of paint and the wrong
territory.

## 1. The empty cell — a cell that never gets a home

Start from random densities: every cell has roughly $1/n$ of paint at every vertex.
At $n = 300$ that is 0.003 everywhere — a thin uniform film, nowhere the strongest.
Now, the double well wants to push each film to 0 or 1, but *which* vertices should
become 1 for cell 17 rather than cell 18? From a symmetric start there is no reason
to prefer one over the other, and gradient descent has no way to invent one: the
symmetric state is a stationary point — the gradient is the same for every cell —
and with the corrected (steeper) well it is actually a *local minimum*, so descent
sits there. A cell that never breaks the symmetry never gets a core where it is the
strongest. It keeps its full share of paint, spread thin everywhere at ~0.08, and
wins zero vertices. Its mass is perfect; its territory is empty.

That is why this one is *resolved* by the seeded start: give every cell a contiguous
Voronoi region at iteration 0 and it has a home from the beginning. The artefact
never gets the chance to form.

## 2. The runt — paint hiding in the transition bands

With seeding, every cell has a core where its paint is 1. But between any two cells
there is a **transition band** of width $\sim\varepsilon \approx h$ (one mesh edge)
where paint goes smoothly from 1 to 0, and in that band several cells have
intermediate values — 0.6 here, 0.3 there. The readout hands each band vertex to
whoever is highest, so band vertices are won by a hair.

A cell's paint is therefore in two places: in its core (all of it counted as
territory) and in the bands around it (counted as territory only where it happens
to be the highest). A cell whose paint sits more in the bands than its neighbours' —
because it started small, or was squeezed at a coarse level — has the right *mass*
but loses the band vertices to neighbours, so its *territory* falls short. Since the
mass constraint is already satisfied by paint sitting in the band, nothing pushes
that paint back into the core. It is a perfectly good minimiser: the energy sees
short interfaces and a cell whose values are mostly 0 or 1. It is a runt only in the
readout.

Two things make this worse as $n$ grows. The bands make up a larger fraction of the
surface (roughly $\sqrt{n/N}$ of it), so there is more room to hide paint in. And the
cells are tiny in mesh terms at the coarse levels: at $n = 300$ the first level has
32 vertices per cell — a cell that is *mostly* band — and a cell that starts starved
there carries the deficit up the ladder. This is the artefact that the crispness
penalty $\lambda$ attacks (it penalises a cell whose values are not sharply 0/1),
which is why $\lambda$ had to grow with $n$, and why it stops working when its
ceiling is reached.

## 3. The split cell — the price of refilling a runt from far away

The mass constraint is enforced by a projection that is **non-local**: when cell $i$
is short of paint, the projection adds paint to $\varphi^i$ *everywhere on the
surface at once*, not near its core. Most of that added paint lands where cell $i$
is far behind and changes nothing. But in some distant band between two *other*
cells, where nobody is clearly winning, a small uniform boost can be enough to make
cell $i$ the highest at a few vertices. Cell $i$ now wins an island there.

Once the island exists, it is stable. The double well happily sharpens it to 1 — a
small crisp patch is exactly the kind of thing the well approves of. The gradient
term would like to shrink it (its boundary costs perimeter), but shrinking it
removes paint from cell $i$, the projection refills the cell everywhere, and the
island comes back. The energy has no term that says "a cell should be one piece";
connectivity is simply not something the energy can see. So a balanced, crisp,
two-piece cell is a fixed point of "gradient step, then refill".

And note *when* it appears in our runs: fragmentation is born mid-level, exactly
while the imbalance is being reduced. That is not a coincidence. The runt and the
split are the same event seen twice — a starved cell being refilled by a non-local
operator, which buys its missing area wherever it can, including far from home.

## Why refinement shrinks the runts but not the splits

Observation from the deck, going from 47,488 to 114,144 vertices with the same
$\lambda$ and seed: imbalanced cells 10 → 3 (worst −36 % → −25 %), split cells
2 → 2.

**Runts shrink** because a runt is a *diffuse* defect: its missing territory is paint
sitting in the bands. Refining the mesh makes $\varepsilon$ smaller, so the bands get
thinner and the well's weight $1/\varepsilon$ gets larger — there is less room to
hide paint, and hiding it costs more. A runt's deficit can never exceed the paint in
the bands, so as the bands shrink, the deficit is squeezed.

**Splits don't** because a split is a *crisp* defect. Both pieces are already at
density 1 with sharp edges; when the densities are interpolated onto the finer mesh,
the island comes along unchanged, and on the finer mesh it looks even more
legitimate — a well-resolved little cell. The only force against it, the island's
own perimeter, is cancelled by the refill. To connect the two pieces the cell would
have to grow a bridge through a neighbour's territory, and every step of that bridge
raises the energy before it lowers it. Refinement does not supply the push to get
over that barrier; it just resolves the island better.

The measurement below confirms this in a rather pointed way. The two split cells are
**the same two cells** on both meshes (274 and 290). And their islands did not shrink
— they **grew**. Cell 290 is the most telling: on the coarse mesh it was both a runt
(69 % of its area) and split; on the fine mesh its area is exactly right (100.1 %)
and it got there by **growing its island**, not its core. Refinement fixed the runt
by feeding paint wherever the cell already wins — and the far-away island is one of
those places. That is the whole mechanism in one cell: balance bought with
disconnection.

| mesh vertices | cell | pieces, as a fraction of one cell's area | territory / target |
|--:|--:|---|--:|
| 47,488 | 274 | 0.874 + **0.106** | 0.979 |
| 47,488 | 290 | 0.644 + **0.041** | 0.685 |
| 114,144 | 274 | 0.743 + **0.240** | 0.983 |
| 114,144 | 290 | 0.828 + **0.173** | 1.001 |

(Computed with `detect_disconnected_cells` / `detect_area_imbalance` from
`src/partition/find_contours.py` on the two runs' `solution/*.h5`; the imbalanced
sets are {16, 72, 124, 195, 197, 222, 275, 281, 290, 299} on the coarse mesh and
{195, 275, 299} on the fine one.)

## The one-sentence version for the talk

The relaxation guarantees each cell the same amount of paint but never asks where
the paint wins. A cell with no home stays empty; a cell whose paint sits in the
transition bands becomes a runt; and refilling a runt from everywhere at once plants
islands, which the energy cannot see and refinement cannot remove — which is why the
fix has to act on the readout, where territory is actually decided, rather than on
the energy.
