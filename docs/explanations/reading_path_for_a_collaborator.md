# A reading path for a collaborator: from the problem to the implementation

*Written 2026-09-21 as a first draft to iterate on. Purpose: a list of papers,
ordered as a **discovery path** rather than a bibliography, to hand to a
collaborator who does not know that we have already built and measured a
replacement for Phase 1. Each paper answers the question the previous one
leaves open, so a reader who follows the path arrives at the design of
`src/partition/mbo_auction.py` without being told it. Every paper named is held
in `docs/papers/` (indexed, with what was verified from each copy, in
`docs/papers/README.md`) unless marked otherwise; the PDFs are gitignored
third-party material and live only on this machine. The companion note
`two_methods_explained.md` is the plain-language account of the two methods;
`docs/math/10-mbo-auction-dynamics/` §2 is the prior-art table these papers
were read for.*

## How to frame it

"We have a working relaxation up to about 300 cells; beyond that it fails the
validity checks and costs days. We need a thousand. Read stages 0, 1 and 2 and
tell me what you would build."

The six papers of stages 0–2 are the *must* set — roughly 120 pages, reading
only the sections named. Stage 3 turns the idea into something that runs on our
meshes. Stage 4 is the one real design decision the path leaves open. If the
collaborator comes back proposing constrained threshold dynamics with a
mesh-based heat step, that is independent confirmation the path is natural —
which is itself worth recording when the work is written up.

## Stage 0 — the problem and the incumbent (1 paper)

**Bogosel & Oudet 2017, *Partitions of Minimal Length on Manifolds***
(`Partitions of Minimal Length on Manifolds.pdf`). What we implemented. Read it
for two things: the relaxed energy with the equal-area constraint on the
densities, and **eq. (5-1)** — the winner-take-all readout. The demonstrations
stop at $n \le 11$ on the torus and $n \le 32$ on the sphere. The question to
plant: *what happens to that readout when $n$ is in the hundreds?* Our answer is
in `three_artefacts_of_the_readout.md` and `docs/math/09-balanced-readout/`; the
collaborator can be left to suspect it.

## Stage 1 — the pivot: evolve a hard partition directly (3 papers, in order)

1. **Merriman, Bence & Osher 1992, *Diffusion Generated Motion by Mean
   Curvature*** (`cam92-18.pdf`, 12 pages). The whole idea in its simplest
   form: blur a set's indicator, threshold at $\tfrac12$, repeat; the boundary
   moves by curvature, which is the fastest way to shorten it. This is the
   "what if we never held a fuzzy field at all?" moment.
2. **Merriman, Bence & Osher 1994, *Motion of Multiple Junctions: A Level Set
   Approach*** (`1-s2.0-S0021999184711053-main.pdf`; a scan, no text layer).
   The multiphase version: blur every cell, give each point to the largest —
   the argmax step, with junctions handled for free.
3. **Esedoğlu & Otto 2015, *Threshold Dynamics for Networks with Arbitrary
   Surface Tensions*** (`10.1002@cpa.21527.pdf`; §1–4 and Lemma 5.2 suffice).
   The bridge back to what the collaborator already knows: each
   blur-and-threshold step is a descent step on an energy that Γ-converges to
   perimeter. Both methods minimise a perimeter surrogate; this one does it on
   hard sets. This is where "the readout gap cannot exist here" becomes
   obvious.

Background, not required for the path: **Evans 1993**
(`Evans-ConvergenceAlgorithmMean-1993.pdf`) is the rigorous convergence proof
for the two-phase scheme.

## Stage 2 — putting the area constraint in (1 paper, the key one)

**Jacobs, Merkurjev & Esedoğlu 2018, *Auction Dynamics: A Volume Constrained
MBO Scheme*** (`1-s2.0-S0021999117308033-main.pdf`; §1–3 and **§4.2**). The
constrained threshold step is a linear assignment problem; its Lagrange
multipliers shift each cell's threshold; they solve it exactly with Bertsekas'
auction. Point at §4.2 explicitly: **equal-area tessellation of the flat torus
at 64 cells**. A collaborator who has read stage 0 will ask "can this be done on
*our* curved, meshed torus at a thousand cells?" — which is the project.

The two-phase antecedent is **Ruuth & Wetton 2003, *A Simple Scheme for
Volume-Preserving Motion by Mean Curvature*** (`ruuth_wetton_2003.pdf`; §2–3,
12 pages): threshold the diffused indicator not at $\tfrac12$ but at the level
that preserves the phase volume. Short and worth reading before JME — it is the
one-cell version of the shifted threshold.

## After stage 2: Esedoğlu–Otto versus JME, in one table

The two papers are easy to blur together, and the collaborator will ask what
the second adds. Short version: **Esedoğlu–Otto 2015 is the theory of the
unconstrained scheme; JME 2018 is that scheme plus the area constraint.** JME
say so themselves — "we obtain our scheme by appealing to a variational
framework for the MBO algorithm developed by Esedoğlu and Otto."

| | Esedoğlu & Otto 2015 | Jacobs, Merkurjev & Esedoğlu 2018 |
|---|---|---|
| **Question** | Why does blur-then-threshold work, and how do you extend it to $N$ phases whose interfaces have *different* surface tensions $\sigma_{ij}$ (grain boundaries in a polycrystal)? | How do you impose a prescribed volume on every phase without losing the scheme's simplicity? |
| **Central idea** | The "heat content" energy $E_\tau$ — heat that escapes each phase in time $\tau$. One MBO step is exactly the minimiser of the *linearisation* of $E_\tau$ at the current partition, so MBO is a minimising-movements scheme for an energy that Γ-converges to weighted perimeter. | Minimise that same linearisation *subject to* each phase keeping its volume. The constrained problem is a **linear assignment problem**; its Lagrange multipliers shift each phase's threshold (their eq. (11)); solve it with Bertsekas' auction. |
| **What it proves** | Energy non-increase at every step whenever the kernel is nonnegative with nonnegative Fourier transform (Lemma 5.2, §5.2); which surface-tension matrices admit a monotone scheme; Γ-convergence of $E_\tau$ (Appendix A). | Dissipation carries over to the constrained step because the current partition is itself feasible — but only with an *exact* assignment. Consistency with volume-preserving weighted curvature flow. |
| **Constraints** | None. | Equality (exact volumes) and inequality (upper/lower bounds) on each phase. |
| **Surface tensions** | Arbitrary $\sigma_{ij}$ and mobilities — the paper's main generality. | Constant multiples of one anisotropy, mostly all equal. |
| **Numerics** | Small phase counts: front-tracking and exact-solution comparisons, topological change, wetting, nucleation. | §4: equal-area tilings of the flat 2-torus (64 → hexagons; 17), area-preserving Voronoi (160), 3-D (8 → Weaire–Phelan with temperature; 32). §5: semi-supervised learning on graphs with class-size constraints. |
| **Domain** | Continuum, periodic box; Gaussian kernel by convolution. | Same, plus graphs with the similarity matrix used directly as the kernel. |

**What each gives us.** From Esedoğlu–Otto: the energy $E_\tau$ that doc 10
tracks, the reason a threshold step is a descent step, the kernel conditions
our finite-element resolvent must satisfy, and the Γ-limit that makes $E_\tau$ a
perimeter surrogate — the bridge to the Γ-convergence view the incumbent already
used. From JME: the balanced threshold $\arg\max_k [y_{ik} + \psi_k]$ itself,
the fact that the $\psi$ are Lagrange multipliers of an assignment problem
(which is what lets us swap their auction for a transportation-dual solve), the
exactness caveat that explains why the dissipation theorem does not transfer to
our inexact solve, and the precedent that this already tiles a flat torus into
64 equal cells.

One way to hold it: **E–O explain the step; JME constrain it.** Without E–O the
constrained step would be an ad-hoc modification with no reason to expect it to
reduce perimeter; without JME, E–O's scheme gives curvature flow that lets cells
shrink and vanish — no equal areas, which is our whole problem.

## Stage 3 — from a grid to a mesh (3 papers)

JME blur with an FFT on a uniform grid. Our surface is an unstructured
triangulation, so the collaborator needs to see how the diffusion step is done
without one:

1. **Garcia-Cardona, Merkurjev, Bertozzi, Flenner & Percus 2014, *Multiclass
   Data Segmentation Using Diffuse Interface Methods on Graphs***
   (`Multiclass_Data_Segmentation_Using_Diffuse_Interface_Methods_on_Graphs.pdf`;
   §3, eqs. (27)–(29)). Multiclass MBO on a weighted graph with an
   **implicit-Euler** heat step, $(I + \delta t\, L)\,U = U^n$. Replace the
   graph Laplacian with the surface finite-element pair and you have our
   $(M + \tau K)\,y = M\chi$.
2. **Dziuk & Elliott 2013, *Finite Element Methods for Surface PDEs***
   (`finite-element-methods-for-surface-pdes.pdf`; §4 and §6 only). The P1 mass
   and stiffness matrices on a triangulated surface and the semi-discrete heat
   equation $M\dot y + Ky = 0$. Supplies the operator.
3. **van Gennip, Guillen, Osting & Bertozzi 2014, *Mean Curvature, Threshold
   Dynamics, and Phase Field Theory on Finite Graphs***
   (`s00032-014-0216-8.pdf`; §4.2, Theorems 4.2–4.4). Why the time step is a
   **two-sided window** on a discrete domain: too small and nothing moves
   (pinning), too large and everything collapses. The theory behind the one
   parameter we have to choose.

**Merriman & Ruuth 2007, *Diffusion Generated Motion of Curves on Surfaces***
(`1-s2.0-S0021999107001301-main.pdf`; §5, §7, §8.3) is the prior attempt at
surfaces — closest-point extension to a 3-D grid, five regions on a torus and a
world map on the sphere. Reading it shows the idea transfers to curved surfaces
and makes the case for doing the diffusion *on* the mesh instead.

## Stage 4 — the assignment machinery (what the prices are)

- **Bertsekas 1988, *The Auction Algorithm*** (`BF02186476.pdf`; §1–2). What
  JME actually solve with, and the "prices" intuition.
- **Peyré & Cuturi 2019, *Computational Optimal Transport***
  (`1803.00567v4.pdf`; **§5 only**, semi-discrete). The same problem as a
  transportation dual, and the subgradient solver we ended up using instead.
  Read together with Bertsekas, this is where the "auction or dual ascent?"
  decision presents itself; the collaborator can reach their own view before
  seeing ours (`docs/math/10-mbo-auction-dynamics/`, Remark 6.1, and
  `two_methods_explained.md` §2b).
- **Aurenhammer, Hoffmann & Aronov 1998, *Minkowski-Type Theorems and
  Least-Squares Clustering*** — power diagrams / capacity-constrained
  assignment, which is what the initialisation is. **Not held** as a PDF; cited
  in `docs/math/shared/references.bib`.
- **Bogosel & Oudet 2021, *Longest Minimal Length Partitions***
  (`2102.02891v2.pdf`; §3.3). The same group using capacity-constrained Voronoi
  diagrams with more than 100 cells as an initialisation — a useful "they
  thought this too" signal.

## Stage 5 — scale and generality (once the design is in hand)

- **Elsey, Esedoğlu & Smereka 2009 / 2011**
  (`1-s2.0-S0021999109004082-main.pdf`, `rspa.2010.0194.pdf`) —
  diffusion-generated motion at $10^5$ grains: the machinery is not
  intrinsically small-$n$.
- **Esedoğlu & Jacobs 2017, *Convolution Kernels and Stability of Threshold
  Dynamics Methods*** (`esedoglu_jacobs.pdf`, author's preprint; Theorem 5.1).
  The theory holds for kernels other than the Gaussian, which is what licenses
  the finite-element resolvent we use.
- **Laux & Otto 2016** (`s00526-016-1053-0.pdf`), **Laux & Swartz 2017**
  (`1601.02467v2.pdf`) — convergence theory for the multiphase and
  volume-constrained schemes; background only.

## Adjacent, for orientation (not on the path)

**Wang & Osting 2019** (`1-s2.0-S0377042718306824-main.pdf`;
diffusion-generated *Dirichlet* partitions of tori and the sphere, $k \le 20$),
**Hu, Liu & Wang 2024** (`2405.16040v1.pdf`; auction dynamics as the inner
solver of the outer problem, $n \le 9$), **Wang, Li, Wei & Wang 2017**
(`1-s2.0-S0021999117305910-main.pdf`; iterative thresholding for image
segmentation). Good for seeing the same machinery used on neighbouring
problems; none is needed to derive ours.

## The path in one table

| Stage | Question it answers | Papers | Pages to read |
|---|---|---|---|
| 0 | What is the method we have, and where does it read the partition off? | Bogosel–Oudet 2017 | ~15 |
| 1 | Can a hard partition be evolved directly, and does that still minimise perimeter? | MBO 1992, MBO 1994, Esedoğlu–Otto 2015 §1–4 | ~50 |
| 2 | How is the equal-area constraint imposed on hard labels? | Ruuth–Wetton 2003 §2–3; Jacobs–Merkurjev–Esedoğlu 2018 §1–3, §4.2 | ~35 |
| 3 | How is the blur done on a triangulated surface, and how is the time step chosen? | Garcia-Cardona 2014 §3; Dziuk–Elliott 2013 §4, §6; van Gennip 2014 §4.2; (Merriman–Ruuth 2007) | ~40 |
| 4 | Auction or transport dual? What is the initialisation? | Bertsekas 1988; Peyré–Cuturi §5; (Aurenhammer 1998); Bogosel–Oudet 2021 §3.3 | ~40 |
| 5 | Does it scale, and does the theory cover our kernel? | Elsey 2009/2011; Esedoğlu–Jacobs 2017; Laux–Otto; Laux–Swartz | as needed |

## Open items for the next iteration of this note

- Whether to add Modica & Mortola / Modica 1987 to stage 0 for a collaborator
  who has not seen Γ-convergence at all (not held as PDFs; cited in the `.bib`).
- Whether Aurenhammer et al. 1998 should be obtained so the path has no gaps
  (Ruuth & Wetton 2003 was obtained on 2026-09-22).
- What to send *after* the collaborator has proposed something: presumably
  `two_methods_explained.md`, then report 08 and doc 10.
