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

#ifndef _TL_UTIL_H_
#define _TL_UTIL_H_

#include <stdlib.h>

#define TL_MAXLINE 4096

#define TL_EXPORT __attribute__((visibility("default")))

#define tl_free free

#ifdef __cplusplus
#define TL_CPPSTART extern "C" {
#define TL_CPPEND }
TL_CPPSTART
#endif

void *tl_alloc(size_t size);
void tl_memcpy(void *dst, void *src, size_t size);
void *tl_clone(const void *src, size_t size);
void tl_copy(const void *src, void *dst, size_t size);
void *tl_repeat(void *data, size_t size, int times);
int tl_compute_length(int ndim, const int *dims);
int tl_read_floats(const char *filename, int num, float *buf);
void tl_warn_msg(const char *fmt, ...);
void tl_warn_cont(int error, const char *fmt, ...);
void tl_warn_ret(const char *fmt, ...);
void tl_err_quit(const char *fmt, ...);
void tl_err_bt(const char *fmt, ...);
void tl_err_exit(int error, const char *fmt, ...);
void tl_err_sys(const char *fmt, ...);
void tl_err_dump(const char *fmt, ...);

/* CUDA support removed */

#ifdef __cplusplus
TL_CPPEND
#endif

#endif /* _TL_UTIL_H_ */
