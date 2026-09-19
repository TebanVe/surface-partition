# Shared build rules for one Marp deck. A deck's own Makefile is just:
#
#     include ../shared/deck.mk
#
# Targets:
#   make            build main.pdf (the tracked, shareable output)
#   make png        render every slide to main.NNN.png for visual review
#   make html       self-contained main.html (speaker view, browser playback)
#   make preview    live-reloading browser preview of main.md
#   make figures    run make_figures.py if the deck has one
#   make clean      remove build products (never the figures)
#
# Marp CLI is run through npx, so the only prerequisites are node and a
# Chromium-family browser for the PDF/PNG export (Chrome is used here). The
# version is pinned for reproducible builds; bump it deliberately and rebuild.
# --html is enabled because the two-column layout in surface-partition.css is
# a <div>; the sources are our own, so the Marp HTML caveat does not apply.
# Recipes redirect stdin from /dev/null: marp-cli reads stdin to EOF whenever
# stdin is not a TTY, so under make (or an editor task) it otherwise hangs
# with "Currently waiting data from stdin stream".

MARP_VERSION ?= 4.5.1
MARP         ?= npx -y @marp-team/marp-cli@$(MARP_VERSION)

SRC     = main.md
PDF     = main.pdf
SHARED  = ../shared
THEMES  = $(SHARED)/neobeam.css $(SHARED)/surface-partition.css
FIGURES = $(wildcard fig_*.png fig_*.svg fig_*.jpg)

MARP_FLAGS = --theme-set $(SHARED)/neobeam.css $(SHARED)/surface-partition.css \
             --html --allow-local-files

$(PDF): $(SRC) $(THEMES) $(FIGURES)
	$(MARP) $(SRC) $(MARP_FLAGS) --pdf -o $(PDF) < /dev/null

png: $(SRC) $(THEMES) $(FIGURES)
	$(MARP) $(SRC) $(MARP_FLAGS) --images png < /dev/null

html: $(SRC) $(THEMES) $(FIGURES)
	$(MARP) $(SRC) $(MARP_FLAGS) -o main.html < /dev/null

preview:
	$(MARP) $(SRC) $(MARP_FLAGS) -s .

figures:
	@if [ -f make_figures.py ]; then python make_figures.py; else echo "no make_figures.py in $$(pwd)"; fi

clean:
	rm -f $(PDF) main.html main.[0-9][0-9][0-9].png

.PHONY: png html preview figures clean
