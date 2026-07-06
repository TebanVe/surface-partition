# Per-report build (see docs/experiments/README.md).
# Reuses the shared math macros and bibliography.
PDF  = main.pdf
TEX  = main.tex
DEPS = ../../math/shared/macros.tex ../../math/shared/references.bib
FIGS = fig_area_trajectory.pdf fig_final_distribution.pdf fig_mass_vs_territory.pdf

$(PDF): $(TEX) $(DEPS) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# The figures are produced by the study script (run from the repo root):
#   python docs/experiments/01-winner-take-all-partition-gap/make_figures.py
$(FIGS):
	@echo "Missing $@ — regenerate with: python docs/experiments/01-winner-take-all-partition-gap/make_figures.py (from repo root)"
	@false

clean:
	latexmk -C

.PHONY: clean
