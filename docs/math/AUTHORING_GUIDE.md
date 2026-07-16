# Math Documentation — Authoring Guide

This guide is for any agent or developer asked to create a new mathematical
derivation document in this folder.  Read it in full before writing any LaTeX.

---

## 1. Folder layout

```
docs/math/
├── AUTHORING_GUIDE.md          ← this file
├── Makefile                    ← master build: `make all` or `make NN-slug`
├── shared/
│   ├── macros.tex              ← ALL shared notation — edit here, not in documents
│   └── references.bib          ← shared bibliography — add new entries here
└── NN-topic-slug/              ← one subdirectory per document
    ├── main.tex                ← LaTeX source (the only file you edit)
    ├── Makefile                ← per-document build (copy from any sibling)
    └── main.pdf                ← compiled output (produced by `make`)
```

Each document is entirely self-contained in its own numbered subdirectory.
The PDF is always `NN-topic-slug/main.pdf`.

---

## 2. Naming convention

Subdirectory names follow `NN-topic-slug`:

| Field | Rule |
|---|---|
| `NN` | Two-digit zero-padded integer. Use the next available number. |
| `topic-slug` | Lowercase, hyphen-separated, ≤ 4 words describing the content. |

Examples: `01-phase2-derivatives`, `02-steiner-analytical`, `03-phase1-energy`.

The number controls the ordering in file browsers and in the master `Makefile`
`DOCS` list.  Never reuse or renumber an existing slot.

---

## 3. Creating a new document — checklist

1. **Create the subdirectory**: `mkdir docs/math/NN-your-slug/`

2. **Copy the per-document Makefile** from any existing sibling (all are identical):
   ```bash
   cp docs/math/01-phase2-derivatives/Makefile docs/math/NN-your-slug/
   ```

3. **Create `main.tex`** using the template in Section 4 below.

4. **Add to the master Makefile**: open `docs/math/Makefile` and append your
   slug to the `DOCS` variable:
   ```makefile
   DOCS = 01-phase2-derivatives 02-steiner-analytical   # add yours here
   ```

5. **Compile and verify**: from inside your new directory:
   ```bash
   cd docs/math/NN-your-slug
   make
   ```
   The build must complete with zero errors.  One or two LaTeX float-placement
   warnings (`h changed to ht`) are acceptable; anything else must be fixed.

6. **Add bibliography entries** to `docs/math/shared/references.bib` if your
   document cites new sources.

---

## 4. `main.tex` template

```latex
\documentclass[11pt,a4paper]{article}

\input{../shared/macros}      % loads ALL packages and notation — do not add
                               % packages directly in main.tex

\hypersetup{
  colorlinks = true,
  linkcolor  = blue!60!black,
  citecolor  = green!50!black,
  urlcolor   = blue!60!black,
  pdftitle   = {Your Document Title},
  pdfauthor  = {surface-partition project}
}

\title{\textbf{Your Document Title}\\[0.5em]
  \large Brief subtitle if needed}
\author{}
\date{Month Year}

\begin{document}
\maketitle

\begin{abstract}
One paragraph: what quantities this document derives, which phase/module
of the codebase they belong to, and which are analytical vs.\ FD.
\end{abstract}

\tableofcontents
\newpage

\section{Overview}
% State the optimization problem or mathematical context.
% Cross-reference related documents if applicable.

\section{Notation and Setup}
% Always include this section.  Start from the lambda convention.
% If a symbol is already in shared/macros.tex, USE IT — do not redefine.
% If you need a new symbol, ADD IT to shared/macros.tex, not here.

\section{...}
% One section per major quantity or topic.
% Each section that has a corresponding implementation must end with a
% \begin{tcolorbox} block naming the Python function (see Section 5).

\section*{Summary table}
% Always end with a summary table mapping quantities to
% (analytical / FD) and the implementing function.

\bibliographystyle{plain}
\bibliography{../shared/references}

\end{document}
```

---

## 5. Style conventions

### 5.1 Code callout boxes

Every derivation that has a direct Python implementation must be followed
by a shaded callout box:

```latex
\begin{tcolorbox}[colback=blue!4!white, colframe=blue!40!black,
                  title={Code: short description}, fontupper=\small]
\codefile{src/partition/module\_name.py}: \code{function\_name(pa)}\\[2pt]
Optionally: one or two lines of the key code pattern.
\end{tcolorbox}
```

Use `\codefile{}` for file paths and `\code{}` for function/variable names
(both defined in `shared/macros.tex`).

### 5.2 Equation numbering

Label every displayed equation that is referenced elsewhere:
```latex
\begin{equation}
  P = \sum_s w_s \norm{\vp{a} - \vp{b}}
  \label{eq:perimeter}
\end{equation}
```
Use `\eqref{eq:perimeter}` (not `(\ref{...})`) in the text.

Label format: `eq:short-descriptive-name`.  Avoid generic labels like
`eq:1` or `eq:main`.

### 5.3 Theorems and remarks

Use the environments from `shared/macros.tex`:

| Environment | When to use |
|---|---|
| `\begin{remark}` | A non-obvious consequence, a sign convention, a degenerate case. |
| `\begin{prop}` | A derived formula stated as a standalone result. |
| `\begin{defn}` | A named object introduced formally (e.g. "edge direction"). |

### 5.4 Summary table

Every document must end with a `\section*{Summary table}` that lists every
quantity covered, whether it is analytical or FD, and the implementing
function.  Model after the one in `01-phase2-derivatives/main.tex`.

---

## 6. Available macros (`shared/macros.tex`)

### Mesh and vertices

| Command | Renders as | Meaning |
|---|---|---|
| `\V` | $V$ | vertex position matrix |
| `\F` | $F$ | face array |

### Variable points

| Command | Renders as | Meaning |
|---|---|---|
| `\lam` | $\lambda$ | scalar VP parameter |
| `\Lam` | $\boldsymbol{\lambda}$ | full parameter vector |
| `\vp{i}` | $\mathbf{p}_i$ | 3D position of VP $i$ |
| `\ed{i}` | $\mathbf{d}_i$ | edge direction $V[v_{1,i}] - V[v_{2,i}]$ |

### Steiner / triple-point

| Command | Renders as | Meaning |
|---|---|---|
| `\Sv` | $\mathbf{S}$ | Steiner (Fermat-Torricelli) point |
| `\Ki` | $K_i$ | projection matrix $(I - n_i n_i^\top)/r_i$ |
| `\Mmat` | $M$ | sum of $K_i$ matrices at a triple point |

### Unit vectors and normals

| Command | Renders as | Meaning |
|---|---|---|
| `\nhat` | $\hat{\mathbf{n}}$ | unit normal to a triangle |
| `\Dhat` | $\hat{\boldsymbol{\Delta}}$ | unit segment direction |

### Operations

| Command | Renders as | Meaning |
|---|---|---|
| `\norm{x}` | $\|x\|$ | Euclidean norm |
| `\abs{x}` | $\|x\|$ | absolute value |
| `\ip{u}{v}` | $\langle u, v \rangle$ | inner product |
| `\Proj{n}` | $(I - n n^\top)$ | projection perpendicular to unit vector $n$ |

### Calculus

| Command | Renders as | Meaning |
|---|---|---|
| `\pd{f}{x}` | $\partial f / \partial x$ | partial derivative |
| `\pdd{f}{x}{y}` | $\partial^2 f / \partial x \partial y$ | mixed second partial |
| `\grad` | $\nabla$ | gradient |
| `\Hess{f}` | $\nabla^2 f$ | Hessian |

### Optimization objects

| Command | Renders as | Meaning |
|---|---|---|
| `\Preg` | $P_{\mathrm{reg}}$ | regular perimeter |
| `\Pst` | $P_{\mathrm{st}}$ | Steiner perimeter |
| `\Areg` | $A^{\mathrm{reg}}$ | regular cell area |
| `\Ast` | $A^{\mathrm{st}}$ | Steiner area contribution |
| `\Atgt` | $\bar{A}$ | target area per cell |
| `\Lagr` | $\mathcal{L}$ | Lagrangian |
| `\objfac` | $\sigma$ | IPOPT objective scaling factor |

### Code references

| Command | Renders as | Meaning |
|---|---|---|
| `\code{name}` | `name` (typewriter) | function or variable name |
| `\codefile{path}` | `path` (small typewriter) | file path |

**Adding a new macro**: if you need a symbol not listed here, add it to
`shared/macros.tex` with a comment explaining what it represents.  Never
define `\newcommand` inside `main.tex`.

---

## 7. Bibliography (`shared/references.bib`)

Current entries:

| Key | Reference |
|---|---|
| `bogosel2023partitions` | Bogosel & Oudet — the foundational paper for Phase 1 and Phase 2 |
| `wachter2006ipopt` | Wächter & Biegler — IPOPT algorithm |
| `nocedal2006numerical` | Nocedal & Wright — Numerical Optimization textbook |
| `kuhn1974steiner` | Kuhn — Steiner's problem / Fermat-Torricelli construction |

Add new entries to `shared/references.bib`, not inline in `main.tex`.

---

## 8. Scope policy

**Only derive what is currently implemented in the codebase.**  If a
formula is planned but not yet coded, do not include its derivation.
Instead, add a `\begin{remark}` noting that the quantity is currently
computed by finite differences and cross-referencing the plan document
in `docs/`.

This keeps each document as a faithful snapshot of the live code, not
a forward-looking design document.

---

## 9. Existing documents

| Directory | Topic | Status |
|---|---|---|
| `01-phase2-derivatives/` | Perimeter value/gradient/Hessian; area Jacobian/Hessian; Steiner FD schemes | Complete |
| `02-phase2-timing-profile/` | Empirical IPOPT callback timing; Steiner FD bottleneck; scaling outlook | Living document |
| `03-analytical-steiner-derivatives/` | Analytical Steiner first/second derivatives | Complete |
| `04-phase1-timing-profile/` | Empirical Phase 1 PGD timing profile; projection bottleneck; line-search thrashing | Complete |
| `05-phase1-nregion-scaling/` | Empirical wall-time scaling with number of regions; projections to N=50/100/1000 | Complete |
| `06-phase1-energy-discretization/` | Phase 1 Γ-convergence energy: Dirichlet term, corrected double well (q=u(1-u)), Modica–Mortola limit, crispness penalty, and all gradients | Complete |

When you create a new document, add a row to this table.
