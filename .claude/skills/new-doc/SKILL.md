---
name: new-doc
description: Create a new documentation file under docs/ for the surface-partition project. Use when the user wants to write up a design plan, a reference/explanatory document, a mathematical derivation, or a measured experiment/study. Guides classification (plans vs reference vs math vs experiments vs presentations) and provides the correct template.
---

# Creating a New Document

Use this skill when the user wants to capture something as a document under `docs/`.
It covers **creation only**. Keeping documents in sync with the codebase afterward
is a standing rule in `CLAUDE.md` ("Keeping Documentation in Sync") and applies
automatically — it is not part of this skill.

## Step 1 — Classify the document

Decide which folder it belongs in:

| Content | Folder | Format |
|---------|--------|--------|
| Work not yet implemented (an actionable design plan) | `docs/plans/` | Markdown |
| Permanent explanation: methodology, a known issue, a primer | `docs/reference/` | Markdown |
| Mathematical derivation of a quantity computed in the code | `docs/math/` | LaTeX |
| A measured study whose numbers come from running the code (question→method→measurement→conclusion) | `docs/experiments/` | LaTeX |
| A slide deck / talk that presents already-established results to collaborators | `docs/presentations/` | chosen per deck |

If the request is ambiguous between plan and reference, ask the user. A quick test:
- **Plan** = "here is what we are going to build." It has phases and a status; it
  becomes stale once the work is done.
- **Reference** = "here is how this tricky thing works." It explains existing
  behavior; it stays relevant indefinitely.
- **Experiment** = "here is what the code measured when we ran it" — a reproducible
  study with a provenance block and a status label. It often *pairs with* a
  reference doc (the experiment is the measurement; the reference is the standing
  explanation), so cross-link the two.

**Hybrid documents** — if the content is part plan and part lasting explanation,
split it: put the actionable phases in `docs/plans/` and the explanatory material
in `docs/reference/`, with each cross-referencing the other.

**Math documents** are out of scope for this skill. If the document is a
derivation, do not use the templates below — follow `docs/math/AUTHORING_GUIDE.md`
instead, which specifies the directory naming, the `main.tex` template, shared
macros, and the scope policy.

**Experiment reports** are likewise out of scope for the Markdown templates below —
they are LaTeX with their own rules. Follow `docs/experiments/README.md`: the
`NN-topic-slug/` LaTeX layout (reusing the math shared macros), a **mandatory
provenance block** (date, source run(s) under `results/`, producing script,
library versions, a numerical anchor), a status label (measured / partial /
planned), and vector `fig_*.pdf` figures produced by a committed `make_figures.py`.
Copy `docs/experiments/01-winner-take-all-partition-gap/` as the worked example.

**Presentations** are also out of scope for the Markdown templates below. A
presentation *presents* results that are already established elsewhere — it never
introduces a new measurement. Follow `docs/presentations/README.md`: one
`NN-topic-slug/` directory per deck, whose own `README.md` is the provenance block
(audience, occasion, the reports/runs/deliverables the slides draw on, status);
the source format is chosen per deck and the rendered `main.pdf` is tracked. If a
slide needs a number that exists nowhere on disk, write the experiment report
first and have the slide cite it.

## Step 2 — Choose a filename

- `docs/plans/` and `docs/reference/`: `UPPER_SNAKE_CASE.md`, descriptive
  (e.g. `EXACT_HESSIAN_AND_ANALYTICAL_STEINER_PLAN.md`,
  `TOPOLOGY_SWITCH_METHODOLOGY.md`). Plan filenames conventionally end in `_PLAN`.
- Check the target folder first so the new name fits the existing set and does not
  collide.

## Step 3 — Write the document from the template

### Plan document (`docs/plans/`)

```markdown
# Title

**Status:** Not Started | In Progress | Partially Implemented | Implemented

## Background
<why this work is needed; what problem it solves; what currently exists>

## Phase 1 — <name>
**Status:** Not Started | In Progress | Done
<what to build, concretely enough to act on>

## Phase 2 — <name>
**Status:** Not Started | In Progress | Done
<...>

## Related documents
- Code: `src/...`
- Reference: `docs/reference/...`
- Math: `docs/math/...`
```

### Reference document (`docs/reference/`)

```markdown
# Title

<one paragraph: what this document covers and why it is non-obvious enough
to need writing down>

## <Section>
<...>

## Related documents
- `docs/plans/...`
- `docs/math/...`
```

## Step 4 — Make it self-contained

- Every cross-reference uses a repo-relative path (`docs/reference/FOO.md`,
  `src/optimization/perimeter_optimizer.py`) so links survive file moves.
- Do not reference a document that does not exist or has been deleted. If a needed
  reference is missing, either inline the content or flag it to the user.
- State assumptions and definitions the reader needs rather than relying on
  context from a conversation that future readers will not have.
