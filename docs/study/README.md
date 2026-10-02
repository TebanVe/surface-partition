# `docs/study/` — a study companion to threshold dynamics

A learning document, written while reading the papers on
[`../explanations/reading_path_for_a_collaborator.md`](../explanations/reading_path_for_a_collaborator.md).
It is for a reader with physics and applied-maths training who wants the
**intuition** behind each method, enough to read `src/partition/mbo_auction.py`,
rather than the proofs. Each chapter builds one idea from familiar physics, works
the key derivation, checks it with a small experiment, and maps it onto the code.
The rigorous account of the implemented method is
[`../math/10-mbo-auction-dynamics/`](../math/10-mbo-auction-dynamics/); this
companion is the road to it.

It differs from its neighbours: `explanations/` is plain-language Markdown with no
derivations; `math/` derives what the code computes, for reviewers; the separate
`partition-reading-notes` repo summarises the papers only and must not mention
our implementation.

## Layout

```
study/
├── main.tex              the book: front matter, roadmap, \include per chapter
├── chapters/chNN_*.tex   one file per chapter
├── study.bib             references not in ../math/shared/references.bib
├── scripts/chNN_*.py     regenerates chapter NN's figures and numbers
├── data/chNN.yaml        committed output — every computed number quoted
├── figures/chNN_*.pdf    committed figures
└── main.pdf              tracked, so it can be read without building
```

## Chapters

| Ch. | Topic | Status |
|---|---|---|
| 1 | Diffusion-generated motion: the MBO scheme, heat content, descent | written |
| 2 | Constraints and prices: Lagrange multipliers, Ruuth–Wetton | planned |
| 3 | Auction dynamics (JME 2018) and Bertsekas' auction | planned |
| 4 | Diffusion on a triangulated surface; other kernels | planned |
| 5 | The time step on a discrete domain (van Gennip et al.) | planned |

## Build

```bash
python docs/study/scripts/ch01_mbo.py                 # figures + data/ch01.yaml (~20 s)
PATH="/Library/TeX/texbin:$PATH" make -C docs/study   # main.pdf
```

The repository's reproducibility rule applies: a number in a chapter must come
from that chapter's committed `data/chNN.yaml`. Shared notation is
`../math/shared/macros.tex`; bibliography is plain bibtex over
`../math/shared/references.bib` and `study.bib`.
