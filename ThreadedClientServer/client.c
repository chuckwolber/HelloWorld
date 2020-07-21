/**
 * MIT License
 *
 * Copyright (c) 2020 Chuck Wolber
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "common.h"

#include <netdb.h>

char* SERVER_ADDR_STRING;
char* CLIENT_STRING;
unsigned int PORT;
int NUM_ARGS = 3;

int set_args(int argc, char** argv);
void usage(const char* exec, const char* message);
int server_connect();

int main(int argc, char **argv) {
    if (!set_args(argc, argv))
        return(1);

    fprintf(stdout, "Client Started...\n");
    fprintf(stdout, "Server Address: %s:%d\n\n", SERVER_ADDR_STRING, PORT);

    int sock_fd = server_connect();
    
    send_s(sock_fd, CLIENT_STRING);
    fprintf(stdout, "Sent String: %s\n", CLIENT_STRING);

    CLIENT_STRING = recv_s(sock_fd);
    fprintf(stdout, "Received Response: %s\n", CLIENT_STRING);
    
    fprintf(stdout, "\nClient Exiting...\n");
    close(sock_fd);

    return 0;
}

int server_connect() {
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0)
        error("ERROR: Failed to open socket");

    struct hostent *server = gethostbyname(SERVER_ADDR_STRING);
    if (server == NULL)
        error("ERROR: Failed to get hostent");
    
    struct sockaddr_in sock_addr;
    sock_addr.sin_family = AF_INET;
    memcpy((char*)&sock_addr.sin_addr.s_addr, (char*)server->h_addr, server->h_length);
    sock_addr.sin_port = htons(PORT);

    if (connect(sock_fd, (struct sockaddr *)&sock_addr, sizeof(sock_addr)) < 0) {
        close(sock_fd);
        error("Failed to connect to server");
    }

    return sock_fd;
}

int set_args(int argc, char **argv) {
    if (argc < (NUM_ARGS + 1)) {
        usage(argv[0], "Missing argument!");
        return FALSE;
    } else if (argc > (NUM_ARGS + 1)) {
        usage(argv[0], "Too many arguments!");
        return FALSE;
    }

    struct in_addr server_addr;
    SERVER_ADDR_STRING = argv[1];
    if (inet_pton(AF_INET, SERVER_ADDR_STRING, &server_addr) != TRUE) {
        usage(argv[0], "Invalid server IP address!");
        return FALSE;
    }

    PORT = (unsigned int)atoi(argv[2]);
    if (PORT < PORT_MIN || PORT > PORT_MAX) {
        usage(argv[0], "Invalid port requested.");
        return FALSE;
    }

    CLIENT_STRING = argv[3];

    return TRUE;
}

void usage(const char* exec, const char* message) {
    if (message != NULL)
        fprintf(stderr, "ERROR: %s\n", message);
    
    fprintf(stderr, "\nusage: %s ipv4_address port client_string\n", exec);
    fprintf(stderr, " * Address must be a valid IPv4 dotted quad string.\n");
    fprintf(stderr, " * Port must be >= %d and <= %d\n", PORT_MIN, PORT_MAX);
    fprintf(stderr, " * Client string is a quoted string.\n");
    fprintf(stderr, "\n");
}
