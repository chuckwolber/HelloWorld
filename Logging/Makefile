VERSION=0.0.1

CC = gcc
CXXFLAGS = -g -Wall -Werror -Wextra -Wconversion -Wuninitialized -Wformat-security -pedantic
BUILD_DIR = build

.PHONY: all clean logger

all: builddir logger

builddir: $(BUILD_DIR)
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

logger: $(BUILD_DIR)/logger
$(BUILD_DIR)/logger: $(BUILD_DIR) $(BUILD_DIR)/log.o $(BUILD_DIR)/main.o
	$(CC) $(CXXFLAGS) $(BUILD_DIR)/log.o $(BUILD_DIR)/main.o -o $@

$(BUILD_DIR)/main.o: main.c
	$(CC) $(CXXFLAGS) -c $< -o $@	

$(BUILD_DIR)/log.o: log.c
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
