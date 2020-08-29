VERSION := 0.0.1
PROJECTS := C C++

.PHONY: clean all $(PROJECTS) $(CLEAN)

all: $(PROJECTS)

$(PROJECTS):
	$(MAKE) -C $@ all

clean:
	make -C C clean
	make -C C++ clean

