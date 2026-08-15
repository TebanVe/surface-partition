# Per-report build (see docs/experiments/README.md).
# Reuses the shared math macros and bibliography.
PDF  = main.pdf
TEX  = main.tex
DEPS = ../../math/shared/macros.tex ../../math/shared/references.bib
FIGS = fig_floor_onset.pdf fig_runt_persistence.pdf

$(PDF): $(TEX) $(DEPS) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# The figures are produced by the study script (run from the repo root):
#   python docs/experiments/06-subfloor-ladder/make_figures.py
$(FIGS):
	@echo "Missing $@ — regenerate with: python docs/experiments/06-subfloor-ladder/make_figures.py (from repo root)"
	@false

clean:
	latexmk -C

.PHONY: clean
