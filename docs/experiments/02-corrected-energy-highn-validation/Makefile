# Per-report build (see docs/experiments/README.md).
# Reuses the shared math macros and bibliography.
PDF  = main.pdf
TEX  = main.tex
DEPS = ../../math/shared/macros.tex ../../math/shared/references.bib
FIGS = fig_runt_resolution.pdf fig_phase2_convergence.pdf fig_init_trap.pdf

$(PDF): $(TEX) $(DEPS) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# The figures are produced by the study script (run from the repo root):
#   python docs/experiments/02-corrected-energy-highn-validation/make_figures.py
$(FIGS):
	@echo "Missing $@ — regenerate with: python docs/experiments/02-corrected-energy-highn-validation/make_figures.py (from repo root)"
	@false

clean:
	latexmk -C

.PHONY: clean
