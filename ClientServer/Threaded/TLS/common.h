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

#include <arpa/inet.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef COMMON_H
#define COMMON_H

static const int TRUE = 1;
static const int FALSE = 0;
static const unsigned int PORT_MIN = 1024;
static const unsigned int PORT_MAX = 65535;

enum CS_TYPE {CLIENT, SERVER};

SSL_CTX* ssl_init(enum CS_TYPE type);

/* Send and receive a NULL terminated string. */
char* ssl_recv_s(SSL* ssl);
int ssl_send_s(SSL* ssl, char* buf);

/* Send and receive. */
int ssl_recv_n(SSL* ssl, char* buf, int len);
int ssl_send_n(SSL* ssl, char* buf, int len);

void error(const char* msg);
void ssl_error();
int ssl_io_error(const SSL* ssl, int ret);

#endif // COMMON_H
