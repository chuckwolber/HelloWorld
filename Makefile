PROJECTS = ClientServer
.PHONY: clean $(PROJECTS)

all: $(PROJECTS)

ClientServer:
	$(MAKE) -C $@ all

clean:
	$(MAKE) -C ClientServer clean

