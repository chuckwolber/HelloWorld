VERSION=0.0.1

CC = g++
CXXFLAGS = -g -Wall -Werror -Wextra -Wconversion -Wuninitialized -Wformat-security -pedantic -std=c++11
BUILD_DIR = build

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	UUID := -luuid
endif

.PHONY: all clean uuid

all: uuid

builddir: $(BUILD_DIR)
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

uuid: $(BUILD_DIR)/uuid
$(BUILD_DIR)/uuid: $(BUILD_DIR) $(BUILD_DIR)/UUID.o $(BUILD_DIR)/main.o | builddir
	$(CC) $(CXXFLAGS) $(BUILD_DIR)/main.o $(BUILD_DIR)/UUID.o -o $@

$(BUILD_DIR)/main.o: main.cpp | builddir
	$(CC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/UUID.o: UUID.cpp UUID.h | builddir
	$(CC) $(CXXFLAGS) $(UUID) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

