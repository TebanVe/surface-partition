# Master build for all surface-partition experiment reports.
#
# Each report lives in its own NN-topic-slug/ subdirectory and builds to
# NN-topic-slug/main.pdf. Add every new report's directory name to DOCS.
#
#   make all        build every report
#   make NN-slug    build a single report
#   make clean      remove build artifacts for every report
#
# Read README.md before adding a report.

DOCS = 01-winner-take-all-partition-gap 02-corrected-energy-highn-validation

all: $(DOCS)

# Explicit per-doc rule (a match-anything `%:` rule is not applied to .PHONY
# targets by GNU Make, so `make all` / `make NN-slug` would silently do nothing).
$(DOCS):
	$(MAKE) -C $@ main.pdf

clean:
	for d in $(DOCS); do $(MAKE) -C $$d clean; done

.PHONY: all clean $(DOCS)
