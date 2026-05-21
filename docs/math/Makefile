DOCS = 01-phase2-derivatives 02-phase2-timing-profile 03-analytical-steiner-derivatives

all: $(DOCS)

%:
	$(MAKE) -C $@ main.pdf

clean:
	for d in $(DOCS); do $(MAKE) -C $$d clean; done

.PHONY: all clean $(DOCS)
