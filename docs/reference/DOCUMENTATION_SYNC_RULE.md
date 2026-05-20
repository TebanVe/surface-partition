# Documentation Sync Rule — Scope and Known Limitation

The `CLAUDE.md` section "Keeping Documentation in Sync" instructs that code
changes keep `docs/` and `CLAUDE.md` accurate. This document records *how* that
rule actually triggers, the one case it does **not** reliably cover, and the
deliberate decision (2026-05-20) to accept that gap for now. It exists so the
analysis does not have to be re-derived if doc drift later becomes a problem.

## The rule is a soft instruction, not a hook

The sync rule lives in `CLAUDE.md` as prose. It is **not** an event hook: nothing
fires deterministically when a file changes. The rule "triggers" only when the
agent *notices*, while reasoning, that its precondition holds — and the agent can
only notice something visible in its context. So the rule's reliability is
entirely a function of: **is the affected document loaded in context at the
moment the change is made?**

## Three cases

| Case | Affected doc visible? | Reliable? |
|------|----------------------|-----------|
| 1. A change touches something `CLAUDE.md` describes (script, class, directory, flag, convention, dependency). | Yes — `CLAUDE.md` is always in context. | **Yes.** |
| 2. A change is plan-implementation work. | Yes — the plan was read to do the work. | **Yes.** |
| 3. An incidental change (unrelated feature) modifies a function/class that some `docs/` file happens to reference, and that doc was never opened. | No — the agent has no awareness the doc exists. | **No — this drift is missed.** |

"Reliable" here means attention-dependent but dependable, because the comparison
material is present — not mechanically guaranteed.

## Decision (2026-05-20): accept the Case 3 gap

Option A — accept the gap — was chosen. Closing Case 3 would require actively
grepping `docs/` for every symbol touched, i.e. a per-change scan whose token
cost was judged not worth paying yet. The rule still reliably delivers its main
value: `CLAUDE.md` never drifts, and plans are updated as they are implemented.
Incidental drift is also low-risk in practice — this repo's reference docs
describe concepts, methodology, and architecture, which rarely hinge on a single
function signature.

**Revisit trigger:** if reference-doc drift is actually observed in practice,
implement one of the deferred options below.

## Deferred options for closing Case 3

- **Option B — rename/remove grep.** When a change renames or deletes a public
  class, function, or file, grep `docs/` once for the old name. Cheap,
  deterministic, catches the worst drift (broken references); does not fire on
  ordinary edits.
- **Option C — batch reconciliation.** Check `docs/` for staleness once at a
  natural boundary (PR creation, end of a feature) instead of per-change. Bounded
  cost, concentrated at a review moment.
- **Option D — PostToolUse hook.** Deterministic, but a hook can only do a
  mechanical symbol grep (≈ Option B), not the semantic judgment of whether a
  doc was invalidated. More machinery, little added intelligence.

Recommended combination if revisited: **B + C** — B for cheap broken-reference
detection, C for everything else at a bounded moment.

## Related documents
- Rule text: `CLAUDE.md`, section "Keeping Documentation in Sync".
- Document creation workflow: `.claude/skills/new-doc/SKILL.md`.
