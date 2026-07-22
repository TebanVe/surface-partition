DOCS = 01-phase2-derivatives 02-phase2-timing-profile 03-analytical-steiner-derivatives 04-phase1-timing-profile 05-phase1-nregion-scaling 06-phase1-energy-discretization 08-dual-newton-projection

all: $(DOCS)

%:
	$(MAKE) -C $@ main.pdf

clean:
	for d in $(DOCS); do $(MAKE) -C $$d clean; done

.PHONY: all clean $(DOCS)
