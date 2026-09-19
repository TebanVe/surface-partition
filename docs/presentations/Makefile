# Master build for all surface-partition presentation decks.
#
# Each deck lives in its own NN-topic-slug/ subdirectory and builds to
# NN-topic-slug/main.pdf via the shared rules in shared/deck.mk. Add every new
# deck's directory name to DOCS.
#
#   make all        build every deck
#   make NN-slug    build a single deck
#   make clean      remove build products for every deck
#
# Read README.md before adding a deck.

DOCS = 01-pgd-status-brief

all: $(DOCS)

# Explicit per-doc rule (a match-anything `%:` rule is not applied to .PHONY
# targets by GNU Make, so `make all` / `make NN-slug` would silently do nothing).
$(DOCS):
	$(MAKE) -C $@ main.pdf

clean:
	for d in $(DOCS); do $(MAKE) -C $$d clean; done

.PHONY: all clean $(DOCS)
