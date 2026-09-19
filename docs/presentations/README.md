# Presentations — surface-partition

Slide decks and talk material that **present** the results of this research and
codebase to collaborators, or that structure a discussion with people outside the
project. Where `docs/experiments/` establishes a result and `docs/reference/`
explains it, this folder packages what is already established for an audience.

## What belongs here

A presentation built from results the repository already records: a progress
update for collaborators, a talk, a seminar deck, a discussion pack for a design
decision. Two rules follow from that:

- **A presentation presents; it does not establish.** Every number, figure and
  claim on a slide must trace to something on disk — an experiment report under
  `docs/experiments/`, a reference doc, a deliverable in the CLAUDE.md tables, or
  a run directory under `results/`. If a slide needs a number that exists nowhere
  yet, the measurement goes in `docs/experiments/` first and the slide cites it.
- **Say what is established and what is not, in the same words the reports use.**
  The reports are careful about this (measured / partial / planned; "consistent
  with, not established by"; the four things that matter more than the
  perimeter). A deck that upgrades a hedged finding into a headline misrepresents
  the work. Reuse the reports' status labels and caveats rather than paraphrasing
  them.

What does **not** belong here: derivations (→ `docs/math/`), measured studies
(→ `docs/experiments/`), permanent explanations (→ `docs/reference/`), design
plans (→ `docs/plans/`), user guides (→ `docs/guides/`).

## Format: Marp + neobeam

Decks are written in **Marp** Markdown (one `main.md` per deck) and styled with
the **neobeam** theme — a modern take on LaTeX beamer — through a project
extension, `shared/surface-partition.css`. Markdown was chosen because the
deliverable and result tables in CLAUDE.md and the reports are already GFM tables
and paste in verbatim, the source diffs like any other file under `docs/`, and
KaTeX math works. The rendered `main.pdf` is tracked so collaborators need no
toolchain.

Prerequisites: **node** and a Chromium-family browser (Chrome) for the PDF
export. Marp CLI itself is run through `npx` at a pinned version
(`MARP_VERSION` in `shared/deck.mk`), so nothing is installed globally. Building
needs network access on first use (npx download) and on every build (the theme
imports Roboto / Roboto Mono / Noto Sans Math from Google Fonts); the exported PDF
embeds the fonts and is self-contained.

## Layout and conventions

One directory per deck, numbered like the sibling trees (`NN-topic-slug`, next
available number), registered in the master `Makefile`'s `DOCS`:

```
docs/presentations/
├── README.md                  ← this file
├── Makefile                   ← master build (`make all`, `make NN-slug`)
├── .gitignore                 ← re-includes main.pdf and fig_* against the root rules
├── shared/
│   ├── neobeam.css            ← vendored upstream theme, never edited (see NOTICE.md)
│   ├── NOTICE.md              ← licence + pinned upstream commit
│   ├── surface-partition.css  ← the theme every deck selects: neobeam + divider,
│   │                            columns, compact tables, figure/caption rules
│   ├── deck.mk                ← shared per-deck make rules (pdf / png / html / preview)
│   └── template.md            ← copy to start a deck; exercises every construct
└── NN-topic-slug/
    ├── README.md              ← audience, occasion, date, status, provenance list
    ├── main.md                ← the deck (starts from shared/template.md)
    ├── Makefile               ← one line: `include ../shared/deck.mk`
    ├── make_figures.py        ← builds every fig_* from results/ (see below)
    ├── fig_*.png / fig_*.svg  ← figures the deck embeds
    └── main.pdf               ← the rendered deck (tracked)
```

Build with `make` inside the deck directory (or `make NN-slug` here):

| target | output | use |
|---|---|---|
| `make` | `main.pdf` | the tracked, shareable deck |
| `make png` | `main.NNN.png` | one PNG per slide — **review every slide this way before committing** |
| `make html` | `main.html` | browser playback with speaker view |
| `make preview` | — | live-reloading preview server while editing |
| `make figures` | `fig_*` | runs `make_figures.py` |

### Each deck's `README.md` is its provenance block

Mirroring the mandatory provenance block of an experiment report, every deck's
`README.md` states:

- **audience and occasion** (who it was made for, when it was or will be given);
- **provenance**: the list of reports, reference docs, deliverables and run
  directories the slides draw on, so a reader can go from any slide back to the
  measurement;
- **status**: draft / given on `<date>` / superseded by `NN-…`.

### Figures come from a script, not from screenshots

Marp embeds PNG and SVG, **not PDF**, so the `fig_*.pdf` figures of the experiment
reports cannot be reused directly. Each deck has a committed `make_figures.py`
that produces every `fig_*` it embeds:

- partition renders via `src/visualization/partition_screenshots.py`
  (`render_partition_screenshots()`, offscreen PyVista) from the exported
  deliverable `.h5` files, with a fixed camera per view and one colour scheme for
  the whole deck, at ≥ 2× the slide's pixel size so they stay crisp in the PDF;
- plots as SVG from the same data the reports use (`arm_report.yaml`, a report's
  `data.yaml`, `timing_profile.yaml`), never from numbers typed by hand.

### Writing the slides — neobeam idioms and the traps found while setting this up

Start from `shared/template.md`; it exercises every construct below and builds.

- **Frame titles are a directive, not a heading.** `<!-- header: 'Title' -->` at
  the top of a slide sets the beamer-style title bar; `#`–`###` inside the body
  are in-slide headings. The title slide is `<!-- _class: title -->` with the
  `# Title / ## Author / > ### Affiliation / ## Occasion` structure of the
  template; a section divider is `<!-- _class: divider -->` with one `# heading`.
- **Two columns** = `<div class="columns">` with two child `<div>`s (add
  `columns-wide-left` / `columns-wide-right` for 3:2). This is HTML, which
  `deck.mk` enables with `--html`; leave a blank line after each `<div>` so the
  Markdown inside is parsed. ⚠ **Do not use Marp's `![bg right:…]` split layout**:
  Marp narrows the section itself, so neobeam's absolute header and footer bars
  get cut at the split (verified) — the columns grid is the only layout that
  composes with the theme.
- **Figures**: `![width:100%](fig_x.png)` inside a column, or `![height:…]` on a
  full-width slide; a `######` heading on the line right after the image is its
  caption (neobeam's idiom). `surface-partition.css` removes neobeam's 50 %
  max-width and rounded corners on images — both wrong for plots.
- **Definition / callout blocks**: a blockquote whose first line is a `####`
  heading. The first three on a deck get distinct colours.
- **Tables**: paste GFM tables as they are; right-align numeric columns with
  `--:`. `surface-partition.css` lifts neobeam's 60 % width cap on tables. Past
  ~8 rows add `<!-- _class: compact -->` to the slide.
- **Provenance line on a slide**: `<p class="source">Source: …</p>` as the last
  element; it renders small and sits at the bottom of the body.
- **Footer**: three `**bold**` fields in the front-matter `footer:` become the
  three footer segments (author · deck title · occasion). Use
  `<!-- _footer: '' -->` on the title slide.
- **Overflow is visible, not silent.** Marp's default theme makes tables
  `overflow:auto` flex items, so on a too-full slide neobeam *shrinks* tables
  and definition blocks (rows vanish) instead of overflowing.
  `surface-partition.css` sets `flex-shrink: 0` on every slide child, so an
  overfull slide runs past the footer in `make png` and gets caught.
- **Display math** is excluded from Marp's auto-scaling in the theme header
  (`@auto-scaling fittingHeader,code`): a `$$` block placed first on a slide was
  being fitted to a sliver.
- neobeam's README warns of a preview-vs-build colour mismatch in the VS Code
  extension (`--build-multiplier`); the CLI build is the reference, so review
  with `make png`, not with the editor preview.

`deck.mk` redirects stdin from `/dev/null`: marp-cli reads stdin to EOF whenever
stdin is not a TTY, so without that it hangs under `make` with
"Currently waiting data from stdin stream".
