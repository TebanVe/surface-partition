# `docs/` — what lives where

Nine folders, each a distinct genre. Getting the genre right is the point of
the split: a derivation, a measurement and a standing explanation age
differently and are checked differently.

| Folder | Genre | Format | Index |
|---|---|---|---|
| [`math/`](math/) | Derivations of quantities the code computes | LaTeX | [`AUTHORING_GUIDE.md`](math/AUTHORING_GUIDE.md) |
| [`experiments/`](experiments/) | Measured studies: question → method → measurement → conclusion | LaTeX | [`README.md`](experiments/README.md) |
| [`reference/`](reference/) | Permanent explanations: methodology, known issues, primers | Markdown | [`README.md`](reference/README.md) |
| [`plans/`](plans/) | Design plans for work not yet implemented | Markdown | [`README.md`](plans/README.md) |
| [`explanations/`](explanations/) | Plain-language accounts written to be *said* | Markdown | [`README.md`](explanations/README.md) |
| [`presentations/`](presentations/) | Slide decks presenting established results | Marp | [`README.md`](presentations/README.md) |
| [`guides/`](guides/) | User guides and onboarding documents | LaTeX | — |
| [`papers/`](papers/) | Third-party PDFs (gitignored) + a verified index | — | [`README.md`](papers/README.md) |
| [`downstream/`](downstream/) | Replies and hand-off material for the repos that consume our exports | Markdown | [`README.md`](downstream/README.md) |

`math/`, `experiments/`, `guides/` and `presentations/` use `NN-topic-slug/`
directories, each holding its source and its tracked compiled output. How to
write any of them: `.claude/rules/docs-authoring.md`, or run the `/new-doc`
skill.

## Where to look for a given question

| Question | Document |
|---|---|
| Why does the winner-take-all readout fail at high N? | [`reference/winner_take_all_partition_gap.md`](reference/winner_take_all_partition_gap.md) — §4b splits, §4c the locality criterion, §9b the readout |
| What are the candidate high-N approaches and their status? | [`reference/PHASE1_HIGHN_APPROACHES_ABCDE.md`](reference/PHASE1_HIGHN_APPROACHES_ABCDE.md) |
| How does approach B work, and what did it measure? | [`math/10-mbo-auction-dynamics/`](math/10-mbo-auction-dynamics/) and [`experiments/08-mbo-auction-dynamics/`](experiments/08-mbo-auction-dynamics/) |
| Which exported partition should I hand downstream? | [`reference/deliverables.yaml`](reference/deliverables.yaml) (`python scripts/check_deliverables.py`) |
| What does the exported file contain? | [`reference/PARTITION_EXPORT_SCHEMA_GENERAL.md`](reference/PARTITION_EXPORT_SCHEMA_GENERAL.md) |
| Where is the time going? | [`math/04-phase1-timing-profile/`](math/04-phase1-timing-profile/) (Phase 1), [`math/02-phase2-timing-profile/`](math/02-phase2-timing-profile/) (Phase 2), [`plans/PUBLICATION_READINESS_PLAN.md`](plans/PUBLICATION_READINESS_PLAN.md) Phase 4 |
| How do I explain this to someone from zero? | [`explanations/`](explanations/) |
| What is the forward plan? | [`plans/PHASE1_BC_REPLACEMENT_PLAN.md`](plans/PHASE1_BC_REPLACEMENT_PLAN.md) (active), [`plans/PUBLICATION_READINESS_PLAN.md`](plans/PUBLICATION_READINESS_PLAN.md) |

## Two standing rules

**Provenance.** Every measured report opens with a provenance block: date,
source run(s) under `results/`, the producing script, library versions, and a
numerical anchor. Every report carries a status label — **measured** /
**partial** / **planned**.

**Reproducibility.** Any document quoting a number produced by running code must
carry the committed script that regenerates it and that script's committed
output (`check_numerics.py` → `numerics.yaml` for math, `make_figures.py` →
`data.yaml` for experiments). Reviewer-supplied numbers and inline session
Python are not admissible sources. `math/10-mbo-auction-dynamics/` is the model.
