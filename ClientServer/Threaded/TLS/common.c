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

static const char* env_keylog = "SSLKEYLOGFILE";

void key_callback(const SSL *ssl, const char *line) {
    char* keylog = getenv(env_keylog);
    (void)ssl;

    if (keylog != NULL) {
        fprintf(stdout, "Keylog File: %s=%s\n", env_keylog, keylog);
        FILE *fp = fopen(keylog, "w");
        if (fp != NULL) {
            fprintf(fp, "%s", line);
            fclose(fp);
        }
    }
}

SSL_CTX* ssl_init(enum CS_TYPE type) {
    SSL_CTX *ctx;
    const SSL_METHOD *method;

    if (type == CLIENT)
        method = TLS_client_method();
    else
        method = TLS_server_method();
    
    ctx = SSL_CTX_new(method);
    if (!ctx) 
        error("ERROR: Unable to create SSL context");
    
    if (SSL_CTX_set_max_proto_version(ctx, TLS1_2_VERSION) <= 0)
        error("ERROR: Unable to set minimum TLS version");

    if (type == SERVER) {
        if (SSL_CTX_use_certificate_file(ctx, "cert.pem", SSL_FILETYPE_PEM) <= 0)
            error("ERROR: Unable to open certificate");

        if (SSL_CTX_use_PrivateKey_file(ctx, "key.nocrypt.pem", SSL_FILETYPE_PEM) <= 0 )
            error("ERROR: Unable to open key");

        if (getenv(env_keylog))
            SSL_CTX_set_keylog_callback(ctx, &key_callback);
    }

    return ctx;
}

char* ssl_recv_s(SSL* ssl) {
    size_t size = sizeof(char);
    char* buf = (char*)malloc(size);
    if (buf == NULL)
        error("ERROR: Failed to allocate memory");

    while (TRUE) {
        char* nbuf = realloc(buf, size+1);
        if (nbuf == NULL)
            error("ERROR: Failed to allocate memory");
        else
            buf = nbuf;
        if (ssl_recv_n(ssl, buf+size-1, 1) < 0)
            return NULL;
        if (buf[size-1] == '\0')
            break;
        size++;
    }

    return buf;
}

int ssl_send_s(SSL* ssl, char* buf) {
    return ssl_send_n(ssl, buf, (int)(strlen(buf)+1));
}

int ssl_recv_n(SSL* ssl, char* buf, int len) {
    int cur_bytes_received = 0;
    int tot_bytes_received = 0;

    while (len > 0) {
        cur_bytes_received = SSL_read(ssl, buf, len);
        tot_bytes_received += cur_bytes_received;
        if (cur_bytes_received < 0) {
            ssl_io_error(ssl, (int)cur_bytes_received);
            ssl_error();
            return -1;
        } else if (cur_bytes_received == 0) {
            return -1; /* Client closed connection. */
        }
        len -= cur_bytes_received;
        buf += cur_bytes_received;
    }

    return tot_bytes_received;
}

int ssl_send_n(SSL* ssl, char* buf, int len) {
    int cur_bytes_sent = 0;
    int tot_bytes_sent = 0;

    while (len > 0) {
        cur_bytes_sent = SSL_write(ssl, buf, len);
        tot_bytes_sent += cur_bytes_sent;
        if (cur_bytes_sent < 0) {
            ssl_io_error(ssl, cur_bytes_sent);
            ssl_error();
            return -1;
        } else if (cur_bytes_sent == 0) {
            return -1; /* Client closed connection. */
        }
        len -= cur_bytes_sent;
        buf += cur_bytes_sent;
    }

    return tot_bytes_sent;
}

void error(const char* msg) {
    perror(msg);
    fprintf(stderr, "Exiting...\n");
    exit(1);
}

void ssl_error() {
    unsigned long err;
    do {
        err = ERR_get_error();
        fprintf(stdout, "ERROR: %s\n", ERR_error_string(err, NULL));
    } while (err != 0);
}

int ssl_io_error(const SSL* ssl, int ret) {
    int error = SSL_get_error(ssl, ret);
    switch(error) {
        case SSL_ERROR_NONE:
            fprintf(stdout, "SSL_ERROR_NONE\n");
            break;
        case SSL_ERROR_ZERO_RETURN:
            fprintf(stdout, "SSL_ERROR_ZERO_RETURN\n");
            break;
        case SSL_ERROR_WANT_READ:
            fprintf(stdout, "SSL_ERROR_WANT_READ\n");
            break;
        case SSL_ERROR_WANT_WRITE:
            fprintf(stdout, "SSL_ERROR_WANT_WRITE\n");
            break;
        case SSL_ERROR_WANT_CONNECT:
            fprintf(stdout, "SSL_ERROR_WANT_CONNECT\n");
            break;
        case SSL_ERROR_WANT_ACCEPT:
            fprintf(stdout, "SSL_ERROR_WANT_ACCEPT\n");
            break;
        case SSL_ERROR_WANT_X509_LOOKUP:
            fprintf(stdout, "SSL_ERROR_WANT_X509_LOOKUP\n");
            break;
        case SSL_ERROR_WANT_ASYNC:
            fprintf(stdout, "SSL_ERROR_WANT_ASYNC\n");
            break;
        case SSL_ERROR_WANT_ASYNC_JOB:
            fprintf(stdout, "SSL_ERROR_WANT_ASYNC_JOB\n");
            break;
        case SSL_ERROR_WANT_CLIENT_HELLO_CB:
            fprintf(stdout, "SSL_ERROR_WANT_CLIENT_HELLO_CB\n");
            break;
        case SSL_ERROR_SYSCALL:
            fprintf(stdout, "SSL_ERROR_SYSCALL\n");
            break;
        case SSL_ERROR_SSL:
            fprintf(stdout, "SSL_ERROR_SSL\n");
            break;
        default:
            fprintf(stdout, "GCC_MADE_ME_DO_IT\n");
            break;
    }

    return error;
}
