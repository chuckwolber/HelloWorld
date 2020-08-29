/**
 * SPDX-License-Identifier: MIT
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

#include <iostream>
#include <map>

#include "UUID.h"

void constructors();
void strings();
void map();

int main() {
    constructors();
    strings();
    map();
    return 0;
}

void constructors() {
    std::cout << "Constructurs... " << std::endl;
    std::string s1("5FA0CB83-03E6-4754-9D60-AE2736C83E5B");
    uuid_t uuid_1;
    uuid_parse(s1.c_str(), uuid_1);
    UUID u1(s1);
    UUID u2(uuid_1);
    UUID u3(u1);

    std::cout << u1.toLowerCaseString() << std::endl;
    std::cout << u1.toUpperCaseString() << std::endl;
    std::cout << u2.toLowerCaseString() << std::endl;
    std::cout << u2.toUpperCaseString() << std::endl;
    std::cout << u3.toLowerCaseString() << std::endl;
    std::cout << u3.toUpperCaseString() << std::endl;
}

void strings() {
    std::cout << "Strings... " << std::endl;

    UUID u1;
    UUID u2;
    std::cout << u1.toLowerCaseString() << std::endl;
    std::cout << u1.toUpperCaseString() << std::endl;
    std::cout << u2.toLowerCaseString() << std::endl;
    std::cout << u2.toUpperCaseString() << std::endl;
}

void map() {
    std::cout << "Map... " << std::endl;

    UUID u1;
    UUID u2;
    std::cout << u1.toLowerCaseString() << std::endl;
    std::cout << u1.toUpperCaseString() << std::endl;
    std::cout << u2.toLowerCaseString() << std::endl;
    std::cout << u2.toUpperCaseString() << std::endl;

    std::string vu1 = "String1";
    std::string vu2 = "String2";

    std::map<UUID,std::string> m1;
    m1[u1] = vu1;
    m1[u2] = vu2;

    std::cout << "Looping over map..." << std::endl;
    for (const std::pair<UUID, std::string> &entry : m1) {
        UUID ux = entry.first;
        std::cout << ux.toLowerCaseString() << std::endl;
    }
}
