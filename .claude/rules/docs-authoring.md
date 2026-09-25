---
paths:
  - "docs/**"
---

# Writing documents under `docs/`

**What exists and where it lives is `docs/README.md`.** This file is how to write
one. To create a document, use the **`/new-doc` skill** — it classifies the
document and supplies the correct template.

Classification, in one line each: not-yet-implemented work -> `plans/`;
permanent explanation -> `reference/`; derivation of something the code computes
-> `math/` (LaTeX); a measured study -> `experiments/` (LaTeX); a deck ->
`presentations/` (Marp); plain-language speaker prep -> `explanations/`
(Markdown); a reply or hand-off to a repo that consumes our exports ->
`downstream/` (Markdown). Full rules and the templates:
`docs/math/AUTHORING_GUIDE.md` and `docs/experiments/README.md`.

⚠ **A reply in `downstream/` is held to the reporting standard**, not to the
standard of a chat message: it will be quoted back at us, so every number in it
needs a committed script and committed output, exactly as a math doc does.

## Computed numbers must be regenerable by a committed script

**Mandatory since 2026-09-17** (`docs/math/AUTHORING_GUIDE.md` §8b), extending
the experiments folder's rule to `docs/math/`. A document that quotes any number
produced by running code must carry:

1. a committed `check_numerics.py` (math) or `make_figures.py` (experiments);
2. its committed output — `numerics.yaml` or `data.yaml`;
3. a provenance block: date, source run(s) under `results/`, producing script,
   library versions, a numerical anchor;
4. determinism — fixed seeds, recorded parametrisation.

**Reviewer-supplied numbers are not admissible. Inline session Python is not a
source.** When reviewing a document, the question to ask of every number is
"which committed file regenerates this?" — if the answer is a session or a
person, it is not auditable.

This was paid for: `docs/math/10-mbo-auction-dynamics/` was first written with
its §5.3/§5.6/§7.2 numbers computed inline during the authoring session and two
figures copied from a reviewer's scratch script; the §5.6 jittered-centre
statistics depended on an unrecorded seed, and two independent runs disagreed by
±0.03. `10-mbo-auction-dynamics/` is now the model to copy.

## LaTeX

`latexmk` + LaTeX live at **`/Library/TeX/texbin`**, which is **not on the Bash
tool's PATH** (a non-login shell). `which pdflatex` reporting "not found" is
misleading. Build with:

```bash
PATH="/Library/TeX/texbin:$PATH" make -C docs/math/NN-slug
PATH="/Library/TeX/texbin:$PATH" make -C docs/experiments all
```

Build artefacts are gitignored; `*.tex`, `*.bib`, `Makefile`, `*.md` and the
`main.pdf` outputs are **tracked** — so rebuild and stage `main.pdf` whenever
`main.tex` or the bibliography changes. Math and experiments share
`docs/math/shared/{macros.tex,references.bib}`; experiments reuse them via
`\input{../../math/shared/macros}`.

Bibliography: `surface-partition` uses plain **bibtex** (the `plain` style prints
`note` fields, so long paths in a note overflow the line — point at
`docs/papers/README.md` instead). The separate `partition-reading-notes` repo
uses **biblatex + biber**; do not copy one repo's preamble into the other.

Every held PDF gets a row in `docs/papers/README.md` recording what was verified
**from the copy**, not from memory — that is where citation errors concentrate.

## Presentations

Marp Markdown + the vendored **neobeam** theme. Three things found while setting
it up, all still true:

- Marp's `![bg right]` split layout **breaks neobeam's header/footer** — use the
  columns grid.
- Frame titles are the `<!-- header: -->` directive, not a heading.
- marp-cli **hangs under `make`** unless stdin is redirected from `/dev/null`
  (`deck.mk` does this).

Figures must be PNG or SVG — **Marp cannot embed the reports' `fig_*.pdf`**. The
vendored theme in `shared/` is pinned by commit in `NOTICE.md` and never edited.

**A presentation presents; it does not establish.** Every number on a slide must
trace to a report under `docs/experiments/`, a reference doc, the deliverables
record, or a run under `results/`, and decks reuse the reports' own status labels
and caveats rather than upgrading hedged findings into headlines.

## Keeping documents in sync with the code

When a code change is motivated by, or invalidates, a document under `plans/` or
`reference/`, update that document **in the same change**. For a plan: advance
its phase status and fold in findings from implementation. A fully-implemented
plan should be deleted, or have its lasting explanation moved to `reference/`.
For a reference doc: correct whatever the change made inaccurate.

The one case this does *not* reliably cover — an incidental change touching a
symbol some unopened doc references — is analysed and deliberately accepted in
`docs/reference/DOCUMENTATION_SYNC_RULE.md`.
