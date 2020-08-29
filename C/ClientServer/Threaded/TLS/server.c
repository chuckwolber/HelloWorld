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

#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/types.h>

uint16_t PORT;
int NUM_ARGS = 1;

pthread_mutex_t MUTEX_STDOUT;
pthread_mutex_t MUTEX_STDERR;
SSL_CTX *ctx = NULL;

int check_args(int argc, char** argv);
void init_mutexes();
int server_start();
void service_clients(int sock_fd); /* Never expected to return. */
void* client_thread(void* client_fd);
void usage(const char* exec, const char* message);

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN); /* Avoid SIGPIPE on short writes. */
    if (!check_args(argc, argv))
        return(1);

    ctx = ssl_init(SERVER);
    init_mutexes();
    int sock_fd = server_start();
    service_clients(sock_fd); /* Not expected to exit. */

    pthread_mutex_destroy(&MUTEX_STDOUT);
    pthread_mutex_destroy(&MUTEX_STDERR);
    SSL_CTX_free(ctx);
    close(sock_fd);

    return(0);
}

void init_mutexes() {
    errno = pthread_mutex_init(&MUTEX_STDOUT, NULL);
    if (errno < 0)
        error("ERROR: Unable to initialize STDOUT mutex");
    errno = pthread_mutex_init(&MUTEX_STDERR, NULL);
    if (errno < 0)
        error("ERROR: Unable to initialize STDERR mutex");
}

int server_start() {
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0)
        error("ERROR: Failed to open socket");

    struct sockaddr_in sock_addr;
    memset((char*)&sock_addr, 0, sizeof(sock_addr));
    sock_addr.sin_family = AF_INET;
    sock_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    sock_addr.sin_port = htons(PORT);

    if (bind(sock_fd, (struct sockaddr *)&sock_addr, sizeof(sock_addr)) < 0) {
        close(sock_fd);
        error("Failed to bind to socket");
    }

    if (listen(sock_fd, 5) < 0) {
        close(sock_fd);
        error("Failed to start listening to socket");
    }

    fprintf(stdout, "Server Started...\n");
    fprintf(stdout, "Listening on port: %d\n", PORT);

    return sock_fd;
}

/* This function is not expected to exit. */
void service_clients(int sock_fd) {
    struct sockaddr_in cli_addr;
    socklen_t cli_addr_len = sizeof(cli_addr);
    pthread_t cli_thread;

    while(TRUE) {
        char c_addr[16];
        int *client_fd = malloc(sizeof(int));
        if (client_fd == NULL) {
            perror("ERROR: malloc() failure allocating client_fd");
            sleep(1); /* Prevent a runaway. */
            continue;
        }
        *client_fd = accept(sock_fd, (struct sockaddr *)&cli_addr, &cli_addr_len);

        pthread_mutex_lock(&MUTEX_STDOUT);
        fprintf(stdout, "\n");
        fprintf(stdout, "[%d] Client Connected!\n", *client_fd);
        fprintf(stdout, "[%d] Client Address: %s:%d\n", *client_fd,
            inet_ntop(AF_INET, &cli_addr.sin_addr, c_addr, 16),
            ntohs(cli_addr.sin_port));
        pthread_mutex_unlock(&MUTEX_STDOUT);

        if (*client_fd < 0) {
            free(client_fd);
            perror("ERROR: Failed to accept client connection");
            sleep(1);
            continue;
        }

        errno = pthread_create(&cli_thread, NULL, &client_thread, client_fd);
        if (errno < 0) {
            free(client_fd);
            perror("ERROR: Failed to create client thread");
            sleep(1);
            continue;
        }

        errno = pthread_detach(cli_thread);
        if (errno < 0) {
            free(client_fd);
            perror("ERROR: Failed to detach client thread");
            sleep(1);
            continue;
        }
    }
}

void* client_thread(void* client_fd) {
    int ret_val;
    int fd = *((int *)client_fd);

    SSL* ssl = SSL_new(ctx);
    if (ssl == NULL) {
        ssl_error();
        pthread_mutex_lock(&MUTEX_STDERR);
        fprintf(stderr, "[%d] ERROR: Failed to create SSL pointer.\n", fd);
        pthread_mutex_unlock(&MUTEX_STDERR);
        goto close_fd;
    }

    if (SSL_set_fd(ssl, fd) <= 0) {
        ssl_error();
        pthread_mutex_lock(&MUTEX_STDERR);
        fprintf(stderr, "[%d] ERROR: Failed to assign fd to SSL pointer.\n", fd);
        pthread_mutex_unlock(&MUTEX_STDERR);
        goto close_ssl;
    }
    
    ret_val = SSL_accept(ssl);
    if (ret_val <= 0) {
        ssl_io_error(ssl, ret_val);
        ssl_error();
        pthread_mutex_lock(&MUTEX_STDERR);
        fprintf(stderr, "[%d] ERROR: Failed to get TLS handshake.\n", fd);
        pthread_mutex_unlock(&MUTEX_STDERR);
        goto close_ssl;
    }

    char* buf = ssl_recv_s(ssl);
    if (buf == NULL) {
        pthread_mutex_lock(&MUTEX_STDERR);
        fprintf(stderr, "[%d] ERROR: Client closed socket.\n", fd);
        pthread_mutex_unlock(&MUTEX_STDERR);
        goto close_ssl;
    }

    size_t len = strlen(buf);
    char* rev = (char*)calloc(len+1, sizeof(char));
    if (rev == NULL) {
        fprintf(stderr, "[%d] ERROR: Failed to allocate memory for rev buf.\n", fd);
        goto close;
    }

    size_t j = 0;
    for (size_t i=len; i>0; i--)
        rev[j++] = buf[i-1];
    
    if (ssl_send_s(ssl, rev) < 0) {
        pthread_mutex_lock(&MUTEX_STDERR);
        fprintf(stderr, "[%d] Client closed socket...\n", fd);
        pthread_mutex_unlock(&MUTEX_STDERR);
        goto close;
    }

    pthread_mutex_lock(&MUTEX_STDOUT);
    fprintf(stdout, "[%d] Received String: %s\n", fd, buf);
    fprintf(stdout, "[%d] Sent String: %s\n", fd, rev);
    fprintf(stdout, "[%d] Closing client connection...\n", fd);
    pthread_mutex_unlock(&MUTEX_STDOUT);

close:
    free(rev);
    free(buf);
close_ssl:
    if (SSL_shutdown(ssl) == 0)
        SSL_shutdown(ssl);
    SSL_free(ssl);
close_fd:
    close(fd);
    free(client_fd);

    return NULL;
}

int check_args(int argc, char **argv) {
    if (argc < (NUM_ARGS + 1)) {
        usage(argv[0], "Missing port argument!");
        return FALSE;
    } else if (argc > (NUM_ARGS + 1)) {
        usage(argv[0], "Invalid number of arguments!");
        return FALSE;
    }

    PORT = (uint16_t)atoi(argv[1]);
    if (PORT < PORT_MIN || PORT > PORT_MAX) {
        usage(argv[0], "Invalid port requested.");
        return FALSE;
    }

    return TRUE;
}

void usage(const char* exec, const char* message) {
    if (message != NULL)
        fprintf(stderr, "ERROR: %s\n", message);
    
    fprintf(stderr, "\nusage: %s port\n", exec);
    fprintf(stderr, " * Port must be >= %d and <= %d\n", PORT_MIN, PORT_MAX);
    fprintf(stderr, " * Set SSLKEYLOGFILE for wireshark decryption.\n");
    fprintf(stderr, "\n");
}
