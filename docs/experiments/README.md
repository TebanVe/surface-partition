# Experiments — surface-partition

Reproducible **measured studies**: the empirical results of the project. Where
`docs/math/` derives what *should* be true (the spec) and `docs/plans/` records
intended work, this folder reports what the code *measured* when we ran it to
**validate or refute a hypothesis** — convergence behaviour, error distributions,
scaling exponents, failure-mode forensics.

This is a distinct genre from a math derivation or a reference explainer, so it
has its own rules. The system mirrors the sibling `link-list-torus` project's
`docs/experiments/`.

## What belongs here

A study with a **question → method → measurement → conclusion** shape, whose
numbers come from running the code on specific inputs. The motivating case:

- the winner-take-all discrete cell-area gap at high N — why Phase 2 perimeter
  refinement fails at N=100, and where the "runt" cell is manufactured
  (`01-winner-take-all-partition-gap/`).

What does **not** belong here: derivations (→ `docs/math/`), permanent
explanations / known-issue writeups (→ `docs/reference/`), design plans for
not-yet-implemented work (→ `docs/plans/`). A measured study often *pairs with* a
reference doc: the report shows the measurement; the reference doc is the standing
explanation. Cross-link them.

## Layout and conventions

Mirrors the `docs/math/` LaTeX system so notation stays consistent — reports
**reuse the math shared macros and bibliography** (`\input{../../math/shared/macros}`,
`\bibliography{../../math/shared/references}` when citing):

```
docs/experiments/
├── README.md                  ← this file
├── Makefile                   ← master build (`make all`, `make NN-slug`)
├── NN-topic-slug/
│   ├── main.tex               ← the report
│   ├── Makefile               ← per-report build
│   ├── make_figures.py        ← the study script that produces the figures
│   ├── fig_*.pdf              ← vector figures (embedded by main.tex)
│   └── main.pdf               ← compiled output
```

Numbering and slugs follow the `docs/math/AUTHORING_GUIDE.md` rule
(`NN-topic-slug`, next available number). Build with `make` (needs `latexmk`;
LaTeX lives at `/Library/TeX/texbin`).

### Provenance is mandatory

A measured report is only science if it is reproducible. **Every report opens
with a provenance block** stating:

- **date** of the run(s);
- the **source run(s)** under `results/` the numbers came from (by run directory
  / run-id), and the surface + parameters (N, λ, seed, mesh schedule);
- the **script** that produced the figures (e.g. `make_figures.py`, committed
  beside the report);
- **versions** (Python, numpy, scipy, matplotlib) and any seed.

A report whose numbers cannot be regenerated from that block is incomplete. When
practical, include a numerical **anchor** — a value that must be reproduced (e.g.
the Phase 2 iteration-0 constraint violation) so a reader can confirm the
reconstruction is correct.

### Cite the derivation / reference

A measured study validates, refutes, or characterises something stated elsewhere.
State the cross-reference: `01-winner-take-all-partition-gap/` characterises the
failure explained in `docs/reference/winner_take_all_partition_gap.md` and gates
the scaling in `docs/plans/PHASE1_N1000_SCALING_PLAN.md`.

### Status label

Each report carries an explicit status — **measured** (numbers in hand),
**partial** (some regimes run), or **planned/skeleton** (structure only, awaiting
the run).
