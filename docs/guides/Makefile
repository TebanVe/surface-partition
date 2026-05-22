DOCS = 01-pelle-user-guide

all: $(DOCS)

%:
	$(MAKE) -C $@ main.pdf

clean:
	for d in $(DOCS); do $(MAKE) -C $$d clean; done

.PHONY: all clean $(DOCS)
