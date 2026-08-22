# Reference papers and books

Local copies of sources cited in `docs/math/` and `docs/experiments/`, kept so
that bibliography entries can be checked against the actual title pages rather
than against memory.

**The PDFs themselves are NOT tracked** — they are third-party material and are
caught by the repo-root `*.pdf` ignore rule (92 MB). Only this index is
version-controlled. Several `references.bib` notes cite paths in this directory;
if a file is missing, it was never committed and must be re-fetched.

| File | Cited as | Verified from the copy |
|---|---|---|
| `Partitions of Minimal Length on Manifolds.pdf` | `bogosel2017partitions` — **the paper this repository implements** | The article's own "To cite this article" line: *Bogosel & Oudet (2017), Experimental Mathematics, **26:4, 496-508**, DOI 10.1080/10586458.2016.1223570*, published online 04 Oct 2016. **Read for content**, not just citation: eq. (5-1) defines the winner-take-all readout; the paper names a triple-point void and zigzag contour length as its issues; "Area tol." 2–5e-7 is the *continuous* constraint residual; demonstrated at n ∈ [2,11] on the torus (R=1, r=0.6), n ≤ 32 on the sphere |
| `1803.00567v4.pdf` | `peyre2019computational` — Peyré & Cuturi, *Computational Optimal Transport* | Authors, title, and the full FnT citation block (vol. 11, no. 5–6, pp. 355–607, 2019) are carried in the arXiv copy itself |
| `[GSM 58] ... Villani - Topics in Optimal Transportation ...pdf` | `villani2003topics` — Villani, *Topics in Optimal Transportation* | Author, title, series (Graduate Studies in Mathematics 58), AMS, 2003 |
| `NET005_...pdf` | `ahuja1993network` — Ahuja, Magnanti & Orlin, *Network Flows* | Authors, title, subtitle, Prentice Hall. **Year 1993 was not read off the title page** |
| `2405.16040v1.pdf` | `hu2024iterative` — Hu, Liu & Wang, *Iterative Thresholding Methods for Longest Minimal Length Partitions* | Authors (Shilong Hu, Hao Liu, Dong Wang), title, v1 25 May 2024. Zero occurrences of manifold/Riemannian/torus in the full text |
| `2102.02891v2.pdf` | `bogosel2021longest` — Bogosel & Oudet, *Longest Minimal Length Partitions* | Authors, title, v2 7 June 2021 |

⚠ The remaining entries in `docs/math/shared/references.bib` carry an upstream
note that volume and page numbers should be checked against published sources
before external distribution. That note still stands for every entry with no row
above.
