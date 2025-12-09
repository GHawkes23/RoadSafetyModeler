ARCHIVE   := cs455-FinalProject-Ben-Gavin.tar

SOURCES := $(shell find . -type f \( -name '*.java' -o -iname 'readme.txt' -o -iname 'build.gradle' \))

.PHONY: archive clean

archive:
	@echo "Creating $(ARCHIVE)…"
	tar -cf $(ARCHIVE) $(SOURCES)
	@echo "Wrote $(ARCHIVE)"

clean:
	rm -f $(ARCHIVE)
