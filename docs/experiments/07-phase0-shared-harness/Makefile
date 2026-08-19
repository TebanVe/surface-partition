# Per-report build (see docs/experiments/README.md).
# Reuses the shared math macros and bibliography.
PDF  = main.pdf
TEX  = main.tex
DEPS = ../../math/shared/macros.tex ../../math/shared/references.bib
FIGS = fig_final_vs_bar.pdf fig_discrimination.pdf fig_masking.pdf

$(PDF): $(TEX) $(DEPS) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# The figures are produced by the study script (run from the repo root):
#   python docs/experiments/07-phase0-shared-harness/make_figures.py
$(FIGS):
	@echo "Missing $@ — regenerate with: python docs/experiments/07-phase0-shared-harness/make_figures.py (from repo root)"
	@false

clean:
	latexmk -C

.PHONY: clean
