# `docs/explanations/` — plain-language accounts

Written to be *said*, not cited: the simplest-terms account of a mechanism the
reference docs treat rigorously, typically prepared for a talk or a
collaborator. Markdown, one file per topic, each opening with a note on what it
was written for and which run(s) any number comes from.

| Document | What it explains |
|---|---|
| [`two_methods_explained.md`](two_methods_explained.md) | Γ-relaxation (A) versus threshold dynamics (B) from zero: what each holds while it searches, why blur-then-threshold shortens boundaries, why A's readout gap is absent in B by construction — and **what is ours and what is not**, with how to position the work for publication. §2b separates whose problem and whose solver in the priced threshold |
| [`balanced_readout_explained.md`](balanced_readout_explained.md) | The readout fix from zero: the per-cell handicap `argmax[log u + ψ]`, why it is local, the thermostat loop that sets ψ, island absorption, and the n=300 numbers at every stage |
| [`three_artefacts_of_the_readout.md`](three_artefacts_of_the_readout.md) | Paint versus territory: the empty / runt / split cells, why refinement shrinks runts but not splits |
| [`reading_path_for_a_collaborator.md`](reading_path_for_a_collaborator.md) | The held papers ordered as a **discovery path** for someone not told the answer, stages 0–5, plus a one-table comparison of Esedoğlu–Otto 2015 against JME 2018 |

These pair with `docs/reference/winner_take_all_partition_gap.md`, which is the
rigorous version of the same material.
