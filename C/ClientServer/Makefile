VERSION := 0.0.1
PROJECTS := Threaded

CLEAN := $(patsubst %,clean-%,$(PROJECTS))
.PHONY: clean all $(PROJECTS)

all: $(PROJECTS)

$(PROJECTS):
	$(MAKE) -C $@ all

clean: $(CLEAN)
clean-%:
	make -C $(patsubst clean-%,%,$@) clean

