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
| `cam92-18.pdf` | `merriman1992diffusion` — Merriman, Bence & Osher, *Diffusion Generated Motion by Mean Curvature*, UCLA CAM Report 92-18 | Title page (April 1992). Fetched 2026-09-17 from `ww3.math.ucla.edu/camreport/`. The original two-phase MBO scheme |
| `1-s2.0-S0021999184711053-main.pdf` | `merriman1994motion` — MBO, *Motion of Multiple Junctions: A Level Set Approach* | JCP 112, 334–363 (1994), read off the scanned first page (**no text layer**; use page images). The multiphase argmax step |
| `Evans-ConvergenceAlgorithmMean-1993.pdf` | `evans1993convergence` | JSTOR header: Indiana Univ. Math. J. 42(2), 533–557, 1993 |
| `10.1002@cpa.21527.pdf` | `esedoglu2015threshold` — Esedoğlu & Otto | CPAM vol. LXVIII, 808–864 (2015). **Read for content**: §4 heat content, Lemma 5.2 (dissipation needs only G ≥ 0, Ĝ ≥ 0), Appendix A (Γ-convergence, smooth radial kernels) |
| `1-s2.0-S0021999117308033-main.pdf` | `jacobs2018auction` — Jacobs, Merkurjev & Esedoğlu, *Auction Dynamics* | JCP 354, 288–310 (2018). **Read for content**: §3 eq. (10)–(11) is the shifted-argmax; **§4.2 equal-area tessellation of the flat 2-torus at N=64 and 17, area-preserving 160-cell Voronoi, 3D N=8/32** — the closest prior work; §5 graphs with unit-volume nodes |
| `esedoglu_jacobs.pdf` | `esedoglu2017kernels` — Esedoğlu & Jacobs, *Convolution Kernels and Stability of Threshold Dynamics Methods* | Author's preprint (2 Aug 2016) from the first author's page; Theorem 5.1 (Γ-convergence for nonnegative symmetric L¹ kernels with finite first moment). Journal details (SINUM 55(5), 2017) are from the publisher listing, **not** the copy |
| `1601.02467v2.pdf` | `laux2017bulk` — Laux & Swartz, *Convergence of Thresholding Schemes Incorporating Bulk Effects* | arXiv v2 (1 Dec 2016). First scheme proved is Ruuth–Wetton's volume-preserving thresholding. Journal version (IFB 2017) **not verified** |
| `s00526-016-1053-0.pdf` | `laux2016convergence` — Laux & Otto | Calc. Var. (2016) 55:129 header |
| `1-s2.0-S0021999198960284-main.pdf` | `ruuth1998multiphase` — Ruuth, *A Diffusion-Generated Approach to Multiphase Motion* | JCP 145, 166–192 (1998) header |
| `BF02186476.pdf` | `bertsekas1988auction` | Ann. Oper. Res. 14, 105–123 (1988) header |
| `s00032-014-0216-8.pdf` | `vangennip2014mean` — van Gennip, Guillen, Osting & Bertozzi | Milan J. Math. 82, 3–65 (2014) header. **Read for content**: Theorems 4.2–4.4 (graph MBO pinned below τ ≈ 0.4/ρ(Δ), trivial above a λ₂ bound) |
| `Multiclass_Data_Segmentation_Using_Diffuse_Interface_Methods_on_Graphs.pdf` | `garcia2014multiclass` — Garcia-Cardona, Merkurjev, Bertozzi, Flenner & Percus | IEEE TPAMI 36(8), 1600–**1613** (2014); last page read off the copy (some indexes say 1614). **Read for content**: eqs. (19)–(23), implicit-Euler graph diffusion in a truncated eigenbasis |
| `1-s2.0-S0021999107001301-main.pdf` | `merriman2007surfaces` — Merriman & Ruuth, *Diffusion Generated Motion of Curves on Surfaces* | JCP 225, 2267–2282 (2007). **Read for content**: §7 junctions, §8.3 five regions on an embedded torus (closest-point method) |
| `1-s2.0-S0377042718306824-main.pdf` | `wang2019dirichlet` — Wang & Osting, *A Diffusion Generated Method for Computing Dirichlet Partitions* | JCAM 351, 302–316 (2019). Flat tori + sphere (FFT/SHT), k ≤ 20, Dirichlet energy not perimeter |
| `1-s2.0-S0021999109004082-main.pdf` | `elsey2009grain` — Elsey, Esedoğlu & Smereka | JCP 228, 8015–8033 (2009). 166,927 grains in 2D |
| `rspa.2010.0194.pdf` | `elsey2011large` — Elsey, Esedoğlu & Smereka | Proc. R. Soc. A 467, 381–401 (2011). 133,110 grains in 3D |
| `1-s2.0-S0021999117305910-main.pdf` | `wang2017thresholding` — Wang, Li, Wei & Wang, *An Efficient Iterative Thresholding Method for Image Segmentation* | JCP 350, 657–667 (2017) |
| `BFb0082859.pdf` | `dziuk1988beltrami` — Dziuk, *Finite Elements for the Beltrami Operator on Arbitrary Surfaces* | The whole LNM 1357 volume (Hildebrandt & Leis, eds.); Dziuk is pp. 142–155 per the volume's table of contents |
| `finite-element-methods-for-surface-pdes.pdf` | `dziuk2013surfacefem` — Dziuk & Elliott | Acta Numerica 22, 289–396 (2013) header |
| `s002080050159.pdf` | `alberti1998nonlocal` — Alberti & Bellettini, *A Nonlocal Anisotropic Model for Phase Transitions. Part I* | Math. Ann. 310, 527–560 (1998). **Part I only** (optimal profile); the Γ-convergence companion (EJAM 9, 1998) is not held |
| `manifolds_perimeter.pdf` | `bogosel2017partitions` (arXiv typesetting) | Same paper as the Experimental Mathematics copy above; a pre-existing copy (file dated Dec 2024) whose equation numbering differs from the journal's |

**Not held, cited second-hand (marked in their `.bib` notes):** `ruuth2003volume`
(Ruuth & Wetton 2003, J. Sci. Comput. 19, 373–384 — paywalled; content from
JME §2 and Laux–Swartz). The proceedings version of MBO 1992 and the journal
versions of Laux–Swartz and Esedoğlu–Jacobs are likewise unverified, as their
rows say.

⚠ The remaining entries in `docs/math/shared/references.bib` carry an upstream
note that volume and page numbers should be checked against published sources
before external distribution. That note still stands for every entry with no row
above.
