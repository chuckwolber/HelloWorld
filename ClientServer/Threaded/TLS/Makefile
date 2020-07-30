VERSION=0.0.1

CC = gcc
CXXFLAGS = -g -Wall -Werror -Wextra -Wconversion -Wuninitialized -Wformat-security -pedantic -pthread
BUILD_DIR = build

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	SSL = -lssl -lcrypto
endif
ifeq ($(UNAME_S),Darwin)
	SSLINC = -I/usr/local/opt/openssl/include
	SSL = -L/usr/local/opt/openssl/lib -lssl -lcrypto
endif

SOURCE_CLIENT = client.c
OBJS_CLIENT = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCE_CLIENT))
EXEC_CLIENT = $(patsubst %.c,%,$(SOURCE_CLIENT))

SOURCE_SERVER = server.c
OBJS_SERVER = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCE_SERVER))
EXEC_SERVER = $(patsubst %.c,%,$(SOURCE_SERVER))

SOURCE_COMMON = common.c
OBJS_COMMON = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCE_COMMON))

.PHONY: all clean cleancerts $(EXEC_CLIENT) $(EXEC_SERVER)

all: $(EXEC_CLIENT) $(EXEC_SERVER)

builddir: $(BUILD_DIR)
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(EXEC_CLIENT): $(BUILD_DIR)/$(EXEC_CLIENT)
$(BUILD_DIR)/$(EXEC_CLIENT): $(BUILD_DIR) $(OBJS_CLIENT) $(OBJS_COMMON)
	$(CC) $(CXXFLAGS) $(OBJS_CLIENT) $(SSLINC) $(SSL) $(OBJS_COMMON) -o $@

$(EXEC_SERVER): $(BUILD_DIR)/$(EXEC_SERVER) cert.pem
$(BUILD_DIR)/$(EXEC_SERVER): $(BUILD_DIR) $(OBJS_SERVER) $(OBJS_COMMON)
	$(CC) $(CXXFLAGS) $(OBJS_SERVER) $(SSLINC) $(SSL) $(OBJS_COMMON) -o $@

cert.pem:
	openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
	openssl rsa -in key.pem -out key.nocrypt.pem

$(OBJS_CLIENT): $(SOURCE_CLIENT)
	$(CC) $(CXXFLAGS) $(SSLINC) -c $< -o $@

$(OBJS_SERVER): $(SOURCE_SERVER)
	$(CC) $(CXXFLAGS) $(SSLINC) -c $< -o $@

$(OBJS_COMMON): $(SOURCE_COMMON)
	$(CC) $(CXXFLAGS) $(SSLINC) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

cleancerts:
	rm -f *.pem

