VERSION=0.0.1

CC = gcc
NCC = nvcc

CXXFLAGS = -g -Wall -Werror -Wextra -Wconversion -Wuninitialized -Wformat-security -pedantic -pthread
BUILD_DIR = build

EXECS = hello hello_cu vector_add vector_add_cu vector_add_parallel_cu vector_add_parallel_blocks_cu

.PHONY: all clean $(EXECS)

all: $(EXECS)

builddir: $(BUILD_DIR)
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

hello: $(BUILD_DIR)/hello
$(BUILD_DIR)/hello: hello.c | $(BUILD_DIR)
	$(CC) hello.c -o $@

hello_cu: $(BUILD_DIR)/hello_cu
$(BUILD_DIR)/hello_cu: hello.cu | $(BUILD_DIR)
	$(NCC) hello.cu -o $@

vector_add: $(BUILD_DIR)/vector_add
$(BUILD_DIR)/vector_add: vector_add.c | $(BUILD_DIR)
	$(CC) vector_add.c -o $@

vector_add_cu: $(BUILD_DIR)/vector_add_cu
$(BUILD_DIR)/vector_add_cu: vector_add.cu | $(BUILD_DIR)
	$(NCC) vector_add.cu -o $@

vector_add_parallel_cu: $(BUILD_DIR)/vector_add_parallel_cu
$(BUILD_DIR)/vector_add_parallel_cu: vector_add_parallel.cu | $(BUILD_DIR)
	$(NCC) vector_add_parallel.cu -o $@

vector_add_parallel_blocks_cu: $(BUILD_DIR)/vector_add_parallel_blocks_cu
$(BUILD_DIR)/vector_add_parallel_blocks_cu: vector_add_parallel_blocks.cu | $(BUILD_DIR)
	$(NCC) vector_add_parallel_blocks.cu -o $@

profile: all
	sudo nvprof --unified-memory-profiling off $(BUILD_DIR)/hello_cu
	sudo nvprof --unified-memory-profiling off $(BUILD_DIR)/vector_add_cu
	sudo nvprof --unified-memory-profiling off $(BUILD_DIR)/vector_add_parallel_cu
	sudo nvprof --unified-memory-profiling off $(BUILD_DIR)/vector_add_parallel_blocks_cu

clean:
	rm -rf $(BUILD_DIR)
