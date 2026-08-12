# Per-report build (see docs/experiments/README.md).
# Reuses the shared math macros and bibliography.
PDF  = main.pdf
TEX  = main.tex
DEPS = ../../math/shared/macros.tex ../../math/shared/references.bib
FIGS = fig_two_defects.pdf fig_churn.pdf

$(PDF): $(TEX) $(DEPS) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# The figures are produced by the study script (run from the repo root):
#   python docs/experiments/05-soft-area-constraint/make_figures.py
$(FIGS):
	@echo "Missing $@ — regenerate with: python docs/experiments/05-soft-area-constraint/make_figures.py (from repo root)"
	@false

clean:
	latexmk -C

.PHONY: clean
