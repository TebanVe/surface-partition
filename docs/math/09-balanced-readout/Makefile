PDF = main.pdf
TEX = main.tex
DEPS = ../shared/macros.tex ../shared/references.bib

$(PDF): $(TEX) $(DEPS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	latexmk -C

.PHONY: clean
