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

#include <errno.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#include "tofu_util.h"

TOFU_EXPORT void *tofu_alloc(size_t size)
{
    void *p;

    assert(size > 0);
    p = malloc(size);
    if (p == NULL)
        tofu_err_dump("malloc(%lu) failed", size);

    return p;
}

TOFU_EXPORT void tofu_memcpy(void *dst, void *src, size_t size)
{
    memmove(dst, src, size);
}

TOFU_EXPORT void *tofu_clone(const void *src, size_t size)
{
    void *p;

    assert(src);
    p = tofu_alloc(size);
    memmove(p, src, size);
    return p;
}

TOFU_EXPORT void tofu_copy(const void *src, void *dst, size_t size)
{
    assert(src && dst);
    memmove(dst, src, size);
}

TOFU_EXPORT void *tofu_repeat(void *data, size_t size, int times)
{
    void *p, *dst;
    int i;

    assert(data && times > 0);
    dst = p = tofu_alloc(size * times);
    for (i = 0; i < times; i++, p = (char *)p + size)
        memmove(p, data, size);
    return dst;
}

TOFU_EXPORT int tofu_compute_length(int ndim, const int *dims)
{
    int i, len;

    assert(ndim > 0);
    assert(dims);
    for (i = 0, len = 1; i < ndim; i++) {
        assert(dims[i] > 0);
        len *= dims[i];
    }
    return len;
}

TOFU_EXPORT int tofu_read_floats(const char *filename, int num, float *buf)
{
    FILE *fp;
    int count = 0;

    if (!(fp = fopen(filename, "r")))
        tofu_err_dump("fopen(%s) failed", filename);

    for (int i = 0; i < num; i++) {
        count += fscanf(fp, "%*[^0-9eE.+-]%f", buf++);
    }
    fclose(fp);
    return count;
}

/* The following functions are taken from APUE, the 3rd version. */
static void err_doit(int errnoflag, int error, const char *fmt, va_list ap)
{
    char buf[TOFU_MAXLINE];

    vsnprintf(buf, TOFU_MAXLINE - 1, fmt, ap);
    if (errnoflag)
        snprintf(buf + strlen(buf), TOFU_MAXLINE - strlen(buf) - 1, ": %s", strerror(error));
    strcat(buf, "\n");
    fflush(stdout);
    fputs(buf, stderr);
    fflush(NULL);
}

/*
 * Nonfatal error unrelated to a system call.
 * Print a message and return.
 */
void tofu_warn_msg(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    err_doit(0, 0, fmt, ap);
    va_end(ap);
}

/*
 * Nonfatal error unrelated to a system call.
 * Error code passed as explict parameter.
 * Print a message and return.
 */
void tofu_warn_cont(int error, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    err_doit(1, error, fmt, ap);
    va_end(ap);
}

/*
 * Nonfatal error related to a system call.
 * Print a message and return.
 */
void tofu_warn_ret(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    err_doit(1, errno, fmt, ap);
    va_end(ap);
}

/*
 * Fatal error unrelated to a system call.
 * Print a message and terminate.
 */
void tofu_err_quit(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    err_doit(0, 0, fmt, ap);
    va_end(ap);
    exit(1);
}

/*
 * Fatal error unrelated to a system call.
 * Print a message, dump core, and terminate.
 */
void tofu_err_bt(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    err_doit(0, 0, fmt, ap);
    va_end(ap);
    abort();
    exit(1);
}

/*
 * Fatal error unrelated to a system call.
 * Error code passed as explict parameter.
 * Print a message and terminate.
 */
void tofu_err_exit(int error, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    err_doit(1, error, fmt, ap);
    va_end(ap);
    exit(1);
}

/*
 * Fatal error related to a system call.
 * Print a message and terminate.
 */
void tofu_err_sys(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    err_doit(1, errno, fmt, ap);
    va_end(ap);
    exit(1);
}

/*
 * Fatal error related to a system call.
 * Print a message, dump core, and terminate.
 */
void tofu_err_dump(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    err_doit(1, errno, fmt, ap);
    va_end(ap);
    abort();
    /* dump core and terminate */
    exit(1);
    /* shouldn’t get here */
}
