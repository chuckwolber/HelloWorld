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

#ifndef UUID_H
#define UUID_H

#include <string>
#include <uuid/uuid.h>

static const size_t UUID_STRLEN = 36;

class UUID {
    public:
        UUID();
        UUID(uuid_t uuid);
        UUID(std::string uuid);
        UUID(const UUID& obj);

        UUID& operator= (const UUID& rhs);
        bool  operator()(const UUID& left, const UUID& right) const;
        bool  operator==(const UUID& rhs) const;
        bool  operator> (const UUID& rhs) const;
        bool  operator< (const UUID& rhs) const;
        bool  operator>=(const UUID& rhs) const;
        bool  operator<=(const UUID& rhs) const;

        std::string toLowerCaseString();
        std::string toUpperCaseString();
        bool isNull();

    private:
        void init();

        uuid_t uuid;
        std::string ucUUID;
        std::string lcUUID;
};

#endif // UUID_H
