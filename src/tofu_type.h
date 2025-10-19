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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _TOFU_TYPE_H_
#define _TOFU_TYPE_H_

#include "tofu_util.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* clang-format off */
enum tofu_bool_t { TOFU_FALSE = 0, TOFU_TRUE = 1 };
typedef enum tofu_bool_t tofu_bool_t;

/* keep the size and the enum order in sync with tofu_type.c */
enum tofu_dtype {
    TOFU_DTYPE_INVALID = -1,
    TOFU_DOUBLE = 0,
    TOFU_FLOAT,
    TOFU_INT64,
    TOFU_INT32,
    TOFU_INT16,
    TOFU_INT8,
    TOFU_UINT64,
    TOFU_UINT32,
    TOFU_UINT16,
    TOFU_UINT8,
    TOFU_BOOL,
    TOFU_DTYPE_SIZE
};
typedef enum tofu_dtype tofu_dtype;
#define TOFU_DTYPE_MAX_SIZE sizeof(double)

/* keep the size and the enum order in sync with tofu_type.c */
enum tofu_elew_op {
    TOFU_ELEW_OP_INVALID = -1,
    TOFU_MUL = 0,
    TOFU_DIV,
    TOFU_SUM,
    TOFU_SUB,
    TOFU_MAX,
    TOFU_MIN,
    TOFU_POW,
    TOFU_ELEW_OP_SIZE
};
typedef enum tofu_elew_op tofu_elew_op;

/* keep the size and the enum order in sync with tofu_type.c */
enum tofu_resize_type {
    TOFU_RESIZE_TYPE_INVALID = -1,
    TOFU_NEAREST = 0,
    TOFU_LINEAR,
    TOFU_RESIZE_TYPE_SIZE
};
typedef enum tofu_resize_type tofu_resize_type;

enum tofu_sort_dir {
    TOFU_SORT_DIR_INVALID = -1,
    TOFU_SORT_DIR_ASCENDING = 0,
    TOFU_SORT_DIR_DESCENDING,
    TOFU_SORT_DIR_SIZE
};
typedef enum tofu_sort_dir tofu_sort_dir;
/* clang-format on */

#define tofu_check_resize_type(rtype) assert(rtype >= 0 && rtype < TOFU_RESIZE_TYPE_SIZE)

typedef int (*tofu_fprintf_func)(FILE *fp, const char *fmt, void *p);
typedef int (*tofu_cmp_func)(void *p1, void *p2);
typedef void (*tofu_elew_func)(void *p1, void *p2, void *r, tofu_elew_op elew_op);

#define tofu_check_dtype(dtype) assert(dtype >= 0 && dtype < TOFU_DTYPE_SIZE)

#define tofu_check_elew_op(op) assert(op >= 0 && op < TOFU_ELEW_OP_SIZE)

#define tofu_check_sort_dir(dir) assert(dir >= 0 && dir < TOFU_SORT_DIR_SIZE)

#ifdef __cplusplus
TOFU_CPPSTART
#endif

/* pointer subtraction and pointer addition */
static inline ptrdiff_t tofu_psub(void *p1, void *p2, size_t dsize)
{
    return (((uint8_t *)(p1) - (uint8_t *)(p2)) / ((ptrdiff_t)dsize));
}

static inline void *tofu_padd(void *p, ptrdiff_t offset, size_t dsize)
{
    return ((uint8_t *)(p) + (offset) * (dsize));
}

/* array element assignment */
static inline void tofu_passign(void *pd, ptrdiff_t offd, void *ps, ptrdiff_t offs, size_t dsize)
{
    memmove(tofu_padd((pd), (offd), (dsize)), tofu_padd((ps), (offs), (dsize)), (dsize));
}

static inline void tofu_pmove(void *pd, ptrdiff_t offd, void *ps, ptrdiff_t offs, size_t dsize,
                            size_t n)
{
    memmove(tofu_padd((pd), (offd), (dsize)), tofu_padd((ps), (offs), (dsize)), (dsize) * (n));
}

size_t tofu_size_of(tofu_dtype dtype);
const char *tofu_dtype_fmt(tofu_dtype dtype);
const char *tofu_dtype_name(tofu_dtype dtype);
tofu_dtype tofu_dtype_from_str(const char *str);
void tofu_dtype_max(tofu_dtype dtype, void *ret);
void tofu_dtype_min(tofu_dtype dtype, void *ret);
double tofu_dtype_max_double(tofu_dtype dtype);
double tofu_dtype_min_double(tofu_dtype dtype);
void tofu_lrelu(void *pd, const void *ps, float negslope, tofu_dtype dtype);
void tofu_convert(void *pd, tofu_dtype dtype_d, const void *ps, tofu_dtype dtype_s);

int tofu_fprintf(FILE *fp, const char *fmt, void *p, tofu_dtype dtype);
tofu_fprintf_func tofu_fprintf_getfunc(tofu_dtype dtype);
int tofu_cmp(void *p1, void *p2, tofu_dtype dtype);
tofu_cmp_func tofu_cmp_getfunc(tofu_dtype dtype);
tofu_elew_op tofu_elew_op_from_str(char *str);
const char *tofu_elew_op_name(tofu_elew_op op);
void tofu_elew(void *p1, void *p2, void *res, tofu_elew_op elew_op, tofu_dtype dtype);
tofu_elew_func tofu_elew_getfunc(tofu_dtype dtype);

const char *tofu_resize_type_name(tofu_resize_type rtype);
tofu_resize_type tofu_resize_type_from_str(const char *str);

const char *tofu_sort_dir_name(tofu_sort_dir dir);
tofu_sort_dir tofu_sort_dir_from_str(const char *str);

static inline ptrdiff_t tofu_pointer_sub(void *p1, void *p2, tofu_dtype dtype)
{
    return tofu_psub((p1), (p2), tofu_size_of(dtype));
}

static inline void *tofu_pointer_add(void *p, ptrdiff_t offset, tofu_dtype dtype)
{
    return tofu_padd((p), (offset), tofu_size_of(dtype));
}

static inline void tofu_pointer_assign(void *pd, ptrdiff_t offd, void *ps, ptrdiff_t offs,
                                     tofu_dtype dtype)
{
    tofu_passign((pd), (offd), (ps), (offs), tofu_size_of(dtype));
}

#ifdef __cplusplus
TOFU_CPPEND
#endif


#endif /* _TOFU_TYPE_H_ */
