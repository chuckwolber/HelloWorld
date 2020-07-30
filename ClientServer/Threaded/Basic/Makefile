VERSION=0.0.1

CC = gcc
CXXFLAGS = -g -Wall -Werror -Wextra -Wconversion -Wuninitialized -Wformat-security -pedantic -pthread
BUILD_DIR = build

SOURCE_CLIENT = client.c
OBJS_CLIENT = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCE_CLIENT))
EXEC_CLIENT = $(patsubst %.c,%,$(SOURCE_CLIENT))

SOURCE_SERVER = server.c
OBJS_SERVER = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCE_SERVER))
EXEC_SERVER = $(patsubst %.c,%,$(SOURCE_SERVER))

SOURCE_COMMON = common.c
OBJS_COMMON = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCE_COMMON))

.PHONY: all clean $(EXEC_CLIENT) $(EXEC_SERVER)

all: builddir $(EXEC_CLIENT) $(EXEC_SERVER)

builddir: $(BUILD_DIR)
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(EXEC_CLIENT): $(BUILD_DIR)/$(EXEC_CLIENT)
$(BUILD_DIR)/$(EXEC_CLIENT): $(BUILD_DIR) $(OBJS_CLIENT) $(OBJS_COMMON)
	$(CC) $(CXXFLAGS) $(OBJS_CLIENT) $(OBJS_COMMON) -o $@

$(EXEC_SERVER): $(BUILD_DIR)/$(EXEC_SERVER)
$(BUILD_DIR)/$(EXEC_SERVER): $(BUILD_DIR) $(OBJS_SERVER) $(OBJS_COMMON)
	$(CC) $(CXXFLAGS) $(OBJS_SERVER) $(OBJS_COMMON) -o $@

$(OBJS_CLIENT): $(SOURCE_CLIENT)
	$(CC) $(CXXFLAGS) -c $< -o $@

$(OBJS_SERVER): $(SOURCE_SERVER)
	$(CC) $(CXXFLAGS) -c $< -o $@

$(OBJS_COMMON): $(SOURCE_COMMON)
	$(CC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

