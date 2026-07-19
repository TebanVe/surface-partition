# Per-report build (see docs/experiments/README.md).
# Reuses the shared math macros and bibliography.
PDF  = main.pdf
TEX  = main.tex
DEPS = ../../math/shared/macros.tex ../../math/shared/references.bib
FIGS = fig_projection_gap.pdf fig_kkt_certificate.pdf

$(PDF): $(TEX) $(DEPS) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# The figures are produced by the study script (run from the repo root):
#   python docs/experiments/03-dual-projection-verification/make_figures.py
$(FIGS):
	@echo "Missing $@ — regenerate with: python docs/experiments/03-dual-projection-verification/make_figures.py (from repo root)"
	@false

clean:
	latexmk -C

.PHONY: clean
