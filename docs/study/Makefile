PDF  = main.pdf
TEX  = main.tex
DEPS = $(wildcard chapters/*.tex) study.bib ../math/shared/macros.tex \
       ../math/shared/references.bib $(wildcard figures/*.pdf)

$(PDF): $(TEX) $(DEPS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

# Every computed number a chapter quotes comes from the COMMITTED data/chNN.yaml,
# regenerated with its figures by scripts/chNN_*.py (run from the repo root).
figures:
	cd ../.. && python docs/study/scripts/ch01_mbo.py

clean:
	latexmk -C

.PHONY: clean figures
