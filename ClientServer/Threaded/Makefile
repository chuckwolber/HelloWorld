PROJECTS = Basic TLS
.PHONY: clean $(PROJECTS)

all: $(PROJECTS)

Basic:
	$(MAKE) -C $@ all
TLS:
	$(MAKE) -C $@ all

clean:
	$(MAKE) -C Basic clean
	$(MAKE) -C TLS clean

