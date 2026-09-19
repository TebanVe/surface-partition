---
marp: true
theme: surface-partition
paginate: true
math: katex
footer: '**Author**
         **Deck title**
         **Occasion, YYYY-MM-DD**'
---
<!-- _class: title -->
<!-- _footer: '' -->
# Deck title

## Author

> ### Affiliation
> Institution

## Occasion, YYYY-MM-DD

---
<!-- header: 'A slide with a frame title' -->
The frame title comes from the `header` directive, not from a heading.
Headings inside the body are in-slide headings:

## In-slide heading
- Bullet with **bold**, *italic*, `code` and inline math $\varepsilon \, u^\top K u$
- Second bullet

> #### Definition block
> Blockquote whose first line is a `####` heading. The first three on a deck get distinct colours.

---
<!-- _class: divider -->
<!-- _header: '' -->
# Section divider

## Optional subtitle

---
<!-- header: 'Two columns: figure beside text' -->
<div class="columns columns-wide-left">
<div>

![width:100%](fig_example.png)
###### Caption: a `######` heading right after the image.

</div>
<div>

**Right column**
- Point one
- Point two
- Point three

$$
E_\varepsilon(u) = \varepsilon\, u^\top K u + \tfrac{1}{\varepsilon}\, q^\top M q,
\qquad q = u(1-u)
$$

</div>
</div>

<p class="source">Source: results/&lt;run&gt;/…, docs/experiments/NN-slug/</p>

---
<!-- header: 'A dense results table' -->
<!-- _class: compact -->
| N | mesh (V) | Phase 1 wall | worst cell | fragmented | Phase 2 perimeter |
|--:|--:|--:|--:|--:|--:|
| 50 | 114,144 | — | — | 0 | 130.1020 |
| 100 | 114,144 | 48,132 s | 0.78% | 0 | 185.2546 |
| 150 | 114,144 | — | — | 0 | 228.1566 |
| 200 | 114,144 | — | — | 0 | 262.1096 |
| 300 | 114,144 | 206,344 s | 0.64% | 0 (after readout) | 322.9622 |
| 300 | 47,488 | 79,069 s | 1.63% | 0 (after readout) | 323.3192 |

<p class="source">Source: numbers here are layout placeholders — a real deck cites CLAUDE.md's deliverable table and the run directories.</p>
