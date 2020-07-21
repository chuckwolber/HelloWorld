PROJECTS = ThreadedClientServer
.PHONY: clean $(PROJECTS)

all: $(PROJECTS)

ThreadedClientServer:
	$(MAKE) -C $@ all

clean:
	$(MAKE) -C ThreadedClientServer clean

