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

char* recv_s(int sockfd) {
    size_t size = sizeof(char);
    char* buf = (char*)malloc(size);
    if (buf == NULL)
        error("ERROR: Failed to allocate memory");

    while (TRUE) {
        char* nbuf = realloc(buf, 1);
        if (nbuf == NULL)
            error("ERROR: Failed to allocate memory");
        else
            buf = nbuf;
        if (recv_n(sockfd, buf+size-1, 1) < 0)
            return NULL;
        if (buf[size-1] == '\0')
            break;
        size++;
    }

    return buf;
}

ssize_t send_s(int sockfd, char* buf) {
    return send_n(sockfd, buf, (ssize_t)(strlen(buf)+1));
}

ssize_t recv_n(int sockfd, char* buf, ssize_t len) {
    ssize_t cur_bytes_received = 0;
    ssize_t tot_bytes_received = 0;

    while (len > 0) {
        cur_bytes_received = recv(sockfd, buf, (size_t)len, 0);
        tot_bytes_received += cur_bytes_received;
        if (cur_bytes_received < 0) {
            perror("ERROR: recv() failed");
            return -1;
        } else if (cur_bytes_received == 0) {
            fprintf(stderr, "[%d] ERROR: Client closed socket...\n", sockfd);
            return -1;
        }
        len -= cur_bytes_received;
        buf += cur_bytes_received;
    }

    return tot_bytes_received;
}

ssize_t send_n(int sockfd, char* buf, ssize_t len) {
    ssize_t cur_bytes_sent = 0;
    ssize_t tot_bytes_sent = 0;

    while (len > 0) {
        cur_bytes_sent = send(sockfd, buf, (size_t)len, 0);
        tot_bytes_sent += cur_bytes_sent;
        if (cur_bytes_sent < 0) {
            perror("ERROR: send() failed");
            return -1;
        } else if (cur_bytes_sent == 0) {
            fprintf(stderr, "[%d] ERROR: Client closed socket...\n", sockfd);
            return -1;
        }
        len -= cur_bytes_sent;
        buf += cur_bytes_sent;
    }

    return tot_bytes_sent;
}

void error(const char* msg) {
    perror(msg);
    exit(1);
}
