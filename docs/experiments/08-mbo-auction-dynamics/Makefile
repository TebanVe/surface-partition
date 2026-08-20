# Per-report build (see docs/experiments/README.md).
# Reuses the shared math macros and bibliography.
PDF  = main.pdf
TEX  = main.tex
DEPS = ../../math/shared/macros.tex ../../math/shared/references.bib
FIGS = fig_perimeter.pdf fig_ladder.pdf fig_tau_window.pdf \
       fig_overmerge_null.pdf fig_predictions.pdf

$(PDF): $(TEX) $(DEPS) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# Figures are produced from the COMMITTED data.yaml (run from the repo root):
#   python docs/experiments/08-mbo-auction-dynamics/make_figures.py
# data.yaml is itself re-derivable from the scored runs, when results/ is on disk:
#   python docs/experiments/08-mbo-auction-dynamics/extract_data.py
$(FIGS):
	@echo "Missing $@ — regenerate with: python docs/experiments/08-mbo-auction-dynamics/make_figures.py (from repo root)"
	@false

clean:
	latexmk -C

.PHONY: clean
