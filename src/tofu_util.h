/*
 * Copyright (c) 2018-2020 Zhixu Zhao
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _TOFU_UTIL_H_
#define _TOFU_UTIL_H_

#include <stdlib.h>

#define TOFU_MAXLINE 4096

#define TOFU_EXPORT __attribute__((visibility("default")))

#define tofu_free free

#ifdef __cplusplus
#define TOFU_CPPSTART extern "C" {
#define TOFU_CPPEND }
TOFU_CPPSTART
#endif

void *tofu_alloc(size_t size);
void tofu_memcpy(void *dst, void *src, size_t size);
void *tofu_clone(const void *src, size_t size);
void tofu_copy(const void *src, void *dst, size_t size);
void *tofu_repeat(void *data, size_t size, int times);
int tofu_compute_length(int ndim, const int *dims);
int tofu_read_floats(const char *filename, int num, float *buf);
void tofu_warn_msg(const char *fmt, ...);
void tofu_warn_cont(int error, const char *fmt, ...);
void tofu_warn_ret(const char *fmt, ...);
void tofu_err_quit(const char *fmt, ...);
void tofu_err_bt(const char *fmt, ...);
void tofu_err_exit(int error, const char *fmt, ...);
void tofu_err_sys(const char *fmt, ...);
void tofu_err_dump(const char *fmt, ...);


#ifdef __cplusplus
TOFU_CPPEND
#endif

#endif /* _TOFU_UTIL_H_ */
