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

#include <math.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <assert.h>
#include "tofu_util.h"
#include "tofu_type.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static const size_t dtype_size[TOFU_DTYPE_SIZE] = {
    sizeof(double),   sizeof(float),   sizeof(int64_t),  sizeof(int32_t),
    sizeof(int16_t),  sizeof(int8_t),  sizeof(uint64_t), sizeof(uint32_t),
    sizeof(uint16_t), sizeof(uint8_t), sizeof(tofu_bool_t)
};

static const char *dtype_fmt[TOFU_DTYPE_SIZE] = { "%.3f", "%.3f", "%ld", "%d", "%d", "%d",
                                                "%lu",  "%u",   "%u",  "%u", "%d" };

static const char *dtype_name[TOFU_DTYPE_SIZE] = { "TOFU_DOUBLE", "TOFU_FLOAT", "TOFU_INT64",  "TOFU_INT32",
                                                 "TOFU_INT16",  "TOFU_INT8",  "TOFU_UINT64", "TOFU_UINT32",
                                                 "TOFU_UINT16", "TOFU_UINT8", "TOFU_BOOL" };

TOFU_EXPORT void tofu_dtype_max(tofu_dtype dtype, void *ret)
{
    assert(ret);
    switch (dtype) {
    case TOFU_DOUBLE:
        *(double *)ret = DBL_MAX;
        break;
    case TOFU_FLOAT:
        *(float *)ret = FLT_MAX;
        break;
    case TOFU_INT64:
        *(int64_t *)ret = INT64_MAX;
        break;
    case TOFU_INT32:
        *(int32_t *)ret = INT32_MAX;
        break;
    case TOFU_INT16:
        *(int16_t *)ret = INT16_MAX;
        break;
    case TOFU_INT8:
        *(int8_t *)ret = INT8_MAX;
        break;
    case TOFU_UINT64:
        *(uint64_t *)ret = UINT64_MAX;
        break;
    case TOFU_UINT32:
        *(uint32_t *)ret = UINT32_MAX;
        break;
    case TOFU_UINT16:
        *(uint16_t *)ret = UINT16_MAX;
        break;
    case TOFU_UINT8:
        *(uint8_t *)ret = UINT8_MAX;
        break;
    case TOFU_BOOL:
        *(tofu_bool_t *)ret = 1;
        break;
    default:
        assert(0 && "unsupported tofu_dtype");
        break;
    }
}

TOFU_EXPORT void tofu_dtype_min(tofu_dtype dtype, void *ret)
{
    assert(ret);
    switch (dtype) {
    case TOFU_DOUBLE:
        *(double *)ret = -DBL_MAX;
        break;
    case TOFU_FLOAT:
        *(float *)ret = -FLT_MAX;
        break;
    case TOFU_INT64:
        *(int64_t *)ret = INT64_MIN;
        break;
    case TOFU_INT32:
        *(int32_t *)ret = INT32_MIN;
        break;
    case TOFU_INT16:
        *(int16_t *)ret = INT16_MIN;
        break;
    case TOFU_INT8:
        *(int8_t *)ret = INT8_MIN;
        break;
    case TOFU_UINT64:
        *(uint64_t *)ret = 0;
        break;
    case TOFU_UINT32:
        *(uint32_t *)ret = 0;
        break;
    case TOFU_UINT16:
        *(uint16_t *)ret = 0;
        break;
    case TOFU_UINT8:
        *(uint8_t *)ret = 0;
        break;
    case TOFU_BOOL:
        *(tofu_bool_t *)ret = 0;
        break;
    default:
        assert(0 && "unsupported tofu_dtype");
        break;
    }
}

TOFU_EXPORT double tofu_dtype_max_double(tofu_dtype dtype)
{
    void *max;
    double max_d;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    tofu_convert(&max_d, TOFU_DOUBLE, max, dtype);
    tofu_free(max);

    return max_d;
}

TOFU_EXPORT double tofu_dtype_min_double(tofu_dtype dtype)
{
    void *min;
    double min_d;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    tofu_convert(&min_d, TOFU_DOUBLE, min, dtype);
    tofu_free(min);

    return min_d;
}

TOFU_EXPORT size_t tofu_size_of(tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    return dtype_size[dtype];
}

TOFU_EXPORT const char *tofu_dtype_fmt(tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    return dtype_fmt[dtype];
}

TOFU_EXPORT const char *tofu_dtype_name(tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    return dtype_name[dtype];
}

TOFU_EXPORT tofu_dtype tofu_dtype_from_str(const char *str)
{
    if (!strcmp(str, "TOFU_DOUBLE"))
        return TOFU_DOUBLE;
    if (!strcmp(str, "TOFU_FLOAT"))
        return TOFU_FLOAT;
    if (!strcmp(str, "TOFU_INT64"))
        return TOFU_INT64;
    if (!strcmp(str, "TOFU_INT32"))
        return TOFU_INT32;
    if (!strcmp(str, "TOFU_INT16"))
        return TOFU_INT16;
    if (!strcmp(str, "TOFU_INT8"))
        return TOFU_INT8;
    if (!strcmp(str, "TOFU_UINT64"))
        return TOFU_UINT64;
    if (!strcmp(str, "TOFU_UINT32"))
        return TOFU_UINT32;
    if (!strcmp(str, "TOFU_UINT16"))
        return TOFU_UINT16;
    if (!strcmp(str, "TOFU_UINT8"))
        return TOFU_UINT8;
    if (!strcmp(str, "TOFU_BOOL"))
        return TOFU_BOOL;
    return -1;
}

/* tofu_fprintf_func */
static int fprintf_double(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_DOUBLE], *(double *)p);
    else
        return fprintf(fp, fmt, *(double *)p);
}

static int fprintf_float(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_FLOAT], *(float *)p);
    else
        return fprintf(fp, fmt, *(float *)p);
}

static int fprintf_int64(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_INT64], *(int64_t *)p);
    else
        return fprintf(fp, fmt, *(int64_t *)p);
}

static int fprintf_int32(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_INT32], *(int32_t *)p);
    else
        return fprintf(fp, fmt, *(int32_t *)p);
}

static int fprintf_int16(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_INT16], *(int16_t *)p);
    else
        return fprintf(fp, fmt, *(int16_t *)p);
}

static int fprintf_int8(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_INT8], *(int8_t *)p);
    else
        return fprintf(fp, fmt, *(int8_t *)p);
}

static int fprintf_uint64(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_UINT64], *(uint64_t *)p);
    else
        return fprintf(fp, fmt, *(uint64_t *)p);
}

static int fprintf_uint32(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_UINT32], *(uint32_t *)p);
    else
        return fprintf(fp, fmt, *(uint32_t *)p);
}

static int fprintf_uint16(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_UINT16], *(uint16_t *)p);
    else
        return fprintf(fp, fmt, *(uint16_t *)p);
}

static int fprintf_uint8(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_UINT8], *(uint8_t *)p);
    else
        return fprintf(fp, fmt, *(uint8_t *)p);
}

static int fprintf_bool(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TOFU_BOOL], *(tofu_bool_t *)p);
    else
        return fprintf(fp, fmt, *(tofu_bool_t *)p);
}

static tofu_fprintf_func fprintf_func[TOFU_DTYPE_SIZE] = {
    fprintf_double, fprintf_float,  fprintf_int64,  fprintf_int32, fprintf_int16, fprintf_int8,
    fprintf_uint64, fprintf_uint32, fprintf_uint16, fprintf_uint8, fprintf_bool
};

TOFU_EXPORT int tofu_fprintf(FILE *fp, const char *fmt, void *p, tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    return fprintf_func[dtype](fp, fmt, p);
}

TOFU_EXPORT tofu_fprintf_func tofu_fprintf_getfunc(tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    return fprintf_func[dtype];
}

/* tofu_cmp_func */
static int cmp_double(void *p1, void *p2)
{
    double d1 = *(double *)p1;
    double d2 = *(double *)p2;
    if (d1 < d2) return -1;
    if (d1 > d2) return 1;
    return 0;
}

static int cmp_float(void *p1, void *p2)
{
    float f1 = *(float *)p1;
    float f2 = *(float *)p2;
    if (f1 < f2) return -1;
    if (f1 > f2) return 1;
    return 0;
}

static int cmp_int64(void *p1, void *p2)
{
    int64_t i1 = *(int64_t *)p1;
    int64_t i2 = *(int64_t *)p2;
    if (i1 < i2) return -1;
    if (i1 > i2) return 1;
    return 0;
}

static int cmp_int32(void *p1, void *p2)
{
    int32_t i1 = *(int32_t *)p1;
    int32_t i2 = *(int32_t *)p2;
    if (i1 < i2) return -1;
    if (i1 > i2) return 1;
    return 0;
}

static int cmp_int16(void *p1, void *p2)
{
    int16_t i1 = *(int16_t *)p1;
    int16_t i2 = *(int16_t *)p2;
    if (i1 < i2) return -1;
    if (i1 > i2) return 1;
    return 0;
}

static int cmp_int8(void *p1, void *p2)
{
    int8_t i1 = *(int8_t *)p1;
    int8_t i2 = *(int8_t *)p2;
    if (i1 < i2) return -1;
    if (i1 > i2) return 1;
    return 0;
}

static int cmp_uint64(void *p1, void *p2)
{
    uint64_t u1 = *(uint64_t *)p1;
    uint64_t u2 = *(uint64_t *)p2;
    if (u1 < u2) return -1;
    if (u1 > u2) return 1;
    return 0;
}

static int cmp_uint32(void *p1, void *p2)
{
    uint32_t u1 = *(uint32_t *)p1;
    uint32_t u2 = *(uint32_t *)p2;
    if (u1 < u2) return -1;
    if (u1 > u2) return 1;
    return 0;
}

static int cmp_uint16(void *p1, void *p2)
{
    uint16_t u1 = *(uint16_t *)p1;
    uint16_t u2 = *(uint16_t *)p2;
    if (u1 < u2) return -1;
    if (u1 > u2) return 1;
    return 0;
}

static int cmp_uint8(void *p1, void *p2)
{
    uint8_t u1 = *(uint8_t *)p1;
    uint8_t u2 = *(uint8_t *)p2;
    if (u1 < u2) return -1;
    if (u1 > u2) return 1;
    return 0;
}

static int cmp_bool(void *p1, void *p2)
{
    return *(tofu_bool_t *)p1 - *(tofu_bool_t *)p2;
}

static tofu_cmp_func cmp_func[TOFU_DTYPE_SIZE] = { cmp_double, cmp_float, cmp_int64,  cmp_int32,
                                               cmp_int16,  cmp_int8,  cmp_uint64, cmp_uint32,
                                               cmp_uint16, cmp_uint8, cmp_bool };

TOFU_EXPORT int tofu_cmp(void *p1, void *p2, tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    return cmp_func[dtype](p1, p2);
}

TOFU_EXPORT tofu_cmp_func tofu_cmp_getfunc(tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    return cmp_func[dtype];
}

/* tofu_elew_func */
typedef void (*elew_op_func)(void *p1, void *p2, void *res);

static const char *elew_op_name[TOFU_ELEW_OP_SIZE] = { "TOFU_MUL", "TOFU_DIV", "TOFU_SUM", "TOFU_SUB",
                                                     "TOFU_MAX", "TOFU_MIN", "TOFU_POW" };

TOFU_EXPORT tofu_elew_op tofu_elew_op_from_str(char *str)
{
    if (!strcmp(str, "TOFU_MUL"))
        return TOFU_MUL;
    if (!strcmp(str, "TOFU_DIV"))
        return TOFU_DIV;
    if (!strcmp(str, "TOFU_SUM"))
        return TOFU_SUM;
    if (!strcmp(str, "TOFU_SUB"))
        return TOFU_SUB;
    if (!strcmp(str, "TOFU_MAX"))
        return TOFU_MAX;
    if (!strcmp(str, "TOFU_MIN"))
        return TOFU_MIN;
    if (!strcmp(str, "TOFU_POW"))
        return TOFU_POW;
    return -1;
}

TOFU_EXPORT const char *tofu_elew_op_name(tofu_elew_op op)
{
    tofu_check_elew_op(op);
    return elew_op_name[op];
}

static void mul_double(void *p1, void *p2, void *res)
{
    *(double *)res = *(double *)p1 * *(double *)p2;
}

static void div_double(void *p1, void *p2, void *res)
{
    assert(*(double *)p2);
    *(double *)res = *(double *)p1 / *(double *)p2;
}

static void sum_double(void *p1, void *p2, void *res)
{
    *(double *)res = *(double *)p1 + *(double *)p2;
}

static void sub_double(void *p1, void *p2, void *res)
{
    *(double *)res = *(double *)p1 - *(double *)p2;
}

static void max_double(void *p1, void *p2, void *res)
{
    *(double *)res = max(*(double *)p1, *(double *)p2);
}

static void min_double(void *p1, void *p2, void *res)
{
    *(double *)res = min(*(double *)p1, *(double *)p2);
}

static void pow_double(void *p1, void *p2, void *res)
{
    *(double *)res = pow(*(double *)p1, *(double *)p2);
}

static elew_op_func elew_op_double[TOFU_ELEW_OP_SIZE] = { mul_double, div_double, sum_double,
                                                        sub_double, max_double, min_double,
                                                        pow_double };

static void elew_double(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_double[elew_op](p1, p2, res);
}

static void mul_float(void *p1, void *p2, void *res)
{
    *(float *)res = *(float *)p1 * *(float *)p2;
}

static void div_float(void *p1, void *p2, void *res)
{
    assert(*(float *)p2);
    *(float *)res = *(float *)p1 / *(float *)p2;
}

static void sum_float(void *p1, void *p2, void *res)
{
    *(float *)res = *(float *)p1 + *(float *)p2;
}

static void sub_float(void *p1, void *p2, void *res)
{
    *(float *)res = *(float *)p1 - *(float *)p2;
}

static void max_float(void *p1, void *p2, void *res)
{
    *(float *)res = max(*(float *)p1, *(float *)p2);
}

static void min_float(void *p1, void *p2, void *res)
{
    *(float *)res = min(*(float *)p1, *(float *)p2);
}

static void pow_float(void *p1, void *p2, void *res)
{
    *(float *)res = powf(*(float *)p1, *(float *)p2);
}

static elew_op_func elew_op_float[TOFU_ELEW_OP_SIZE] = { mul_float, div_float, sum_float, sub_float,
                                                       max_float, min_float, pow_float };

static void elew_float(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_float[elew_op](p1, p2, res);
}

static void mul_int64(void *p1, void *p2, void *res)
{
    *(int64_t *)res = *(int64_t *)p1 * *(int64_t *)p2;
}

static void div_int64(void *p1, void *p2, void *res)
{
    assert(*(int64_t *)p2);
    *(int64_t *)res = *(int64_t *)p1 / *(int64_t *)p2;
}

static void sum_int64(void *p1, void *p2, void *res)
{
    *(int64_t *)res = *(int64_t *)p1 + *(int64_t *)p2;
}

static void sub_int64(void *p1, void *p2, void *res)
{
    *(int64_t *)res = *(int64_t *)p1 - *(int64_t *)p2;
}

static void max_int64(void *p1, void *p2, void *res)
{
    *(int64_t *)res = max(*(int64_t *)p1, *(int64_t *)p2);
}

static void min_int64(void *p1, void *p2, void *res)
{
    *(int64_t *)res = min(*(int64_t *)p1, *(int64_t *)p2);
}

static void pow_int64(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(int64_t *)p1;
    d2 = (double)*(int64_t *)p2;
    dr = pow(d1, d2);
    if (dr >= INT64_MAX)
        *(int64_t *)res = INT64_MAX;
    else if (dr <= INT64_MIN)
        *(int64_t *)res = INT64_MIN;
    else
        *(int64_t *)res = (int64_t)dr;
}

static elew_op_func elew_op_int64[TOFU_ELEW_OP_SIZE] = { mul_int64, div_int64, sum_int64, sub_int64,
                                                       max_int64, min_int64, pow_int64 };

static void elew_int64(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_int64[elew_op](p1, p2, res);
}

static void mul_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = *(int32_t *)p1 * *(int32_t *)p2;
}

static void div_int32(void *p1, void *p2, void *res)
{
    assert(*(int32_t *)p2);
    *(int32_t *)res = *(int32_t *)p1 / *(int32_t *)p2;
}

static void sum_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = *(int32_t *)p1 + *(int32_t *)p2;
}

static void sub_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = *(int32_t *)p1 - *(int32_t *)p2;
}

static void max_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = max(*(int32_t *)p1, *(int32_t *)p2);
}

static void min_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = min(*(int32_t *)p1, *(int32_t *)p2);
}

static void pow_int32(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(int32_t *)p1;
    d2 = (double)*(int32_t *)p2;
    dr = pow(d1, d2);
    if (dr >= INT32_MAX)
        *(int32_t *)res = INT32_MAX;
    else if (dr <= INT32_MIN)
        *(int32_t *)res = INT32_MIN;
    else
        *(int32_t *)res = (int32_t)dr;
}

static elew_op_func elew_op_int32[TOFU_ELEW_OP_SIZE] = { mul_int32, div_int32, sum_int32, sub_int32,
                                                       max_int32, min_int32, pow_int32 };

static void elew_int32(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_int32[elew_op](p1, p2, res);
}

static void mul_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = *(int16_t *)p1 * *(int16_t *)p2;
}

static void div_int16(void *p1, void *p2, void *res)
{
    assert(*(int16_t *)p2);
    *(int16_t *)res = *(int16_t *)p1 / *(int16_t *)p2;
}

static void sum_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = *(int16_t *)p1 + *(int16_t *)p2;
}

static void sub_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = *(int16_t *)p1 - *(int16_t *)p2;
}

static void max_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = max(*(int16_t *)p1, *(int16_t *)p2);
}

static void min_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = min(*(int16_t *)p1, *(int16_t *)p2);
}

static void pow_int16(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(int16_t *)p1;
    d2 = (double)*(int16_t *)p2;
    dr = pow(d1, d2);
    if (dr >= INT16_MAX)
        *(int16_t *)res = INT16_MAX;
    else if (dr <= INT16_MIN)
        *(int16_t *)res = INT16_MIN;
    else
        *(int16_t *)res = (int16_t)dr;
}

static elew_op_func elew_op_int16[TOFU_ELEW_OP_SIZE] = { mul_int16, div_int16, sum_int16, sub_int16,
                                                       max_int16, min_int16, pow_int16 };

static void elew_int16(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_int16[elew_op](p1, p2, res);
}

static void mul_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = *(int8_t *)p1 * *(int8_t *)p2;
}

static void div_int8(void *p1, void *p2, void *res)
{
    assert(*(int8_t *)p2);
    *(int8_t *)res = *(int8_t *)p1 / *(int8_t *)p2;
}

static void sum_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = *(int8_t *)p1 + *(int8_t *)p2;
}

static void sub_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = *(int8_t *)p1 - *(int8_t *)p2;
}

static void max_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = max(*(int8_t *)p1, *(int8_t *)p2);
}

static void min_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = min(*(int8_t *)p1, *(int8_t *)p2);
}

static void pow_int8(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(int8_t *)p1;
    d2 = (double)*(int8_t *)p2;
    dr = pow(d1, d2);
    if (dr >= INT8_MAX)
        *(int8_t *)res = INT8_MAX;
    else if (dr <= INT8_MIN)
        *(int8_t *)res = INT8_MIN;
    else
        *(int8_t *)res = (int8_t)dr;
}

static elew_op_func elew_op_int8[TOFU_ELEW_OP_SIZE] = { mul_int8, div_int8, sum_int8, sub_int8,
                                                      max_int8, min_int8, pow_int8 };

static void elew_int8(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_int8[elew_op](p1, p2, res);
}

static void mul_uint64(void *p1, void *p2, void *res)
{
    *(uint64_t *)res = *(uint64_t *)p1 * *(uint64_t *)p2;
}

static void div_uint64(void *p1, void *p2, void *res)
{
    assert(*(uint64_t *)p2);
    *(uint64_t *)res = *(uint64_t *)p1 / *(uint64_t *)p2;
}

static void sum_uint64(void *p1, void *p2, void *res)
{
    *(uint64_t *)res = *(uint64_t *)p1 + *(uint64_t *)p2;
}

static void sub_uint64(void *p1, void *p2, void *res)
{
    *(uint64_t *)res = *(uint64_t *)p1 - *(uint64_t *)p2;
}

static void max_uint64(void *p1, void *p2, void *res)
{
    *(uint64_t *)res = max(*(uint64_t *)p1, *(uint64_t *)p2);
}

static void min_uint64(void *p1, void *p2, void *res)
{
    *(uint64_t *)res = min(*(uint64_t *)p1, *(uint64_t *)p2);
}

static void pow_uint64(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(uint64_t *)p1;
    d2 = (double)*(uint64_t *)p2;
    dr = pow(d1, d2);
    if (dr >= UINT64_MAX)
        *(uint64_t *)res = UINT64_MAX;
    else
        *(uint64_t *)res = (uint64_t)dr;
}

static elew_op_func elew_op_uint64[TOFU_ELEW_OP_SIZE] = { mul_uint64, div_uint64, sum_uint64,
                                                        sub_uint64, max_uint64, min_uint64,
                                                        pow_uint64 };

static void elew_uint64(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_uint64[elew_op](p1, p2, res);
}

static void mul_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = *(uint32_t *)p1 * *(uint32_t *)p2;
}

static void div_uint32(void *p1, void *p2, void *res)
{
    assert(*(uint32_t *)p2);
    *(uint32_t *)res = *(uint32_t *)p1 / *(uint32_t *)p2;
}

static void sum_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = *(uint32_t *)p1 + *(uint32_t *)p2;
}

static void sub_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = *(uint32_t *)p1 - *(uint32_t *)p2;
}

static void max_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = max(*(uint32_t *)p1, *(uint32_t *)p2);
}

static void min_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = min(*(uint32_t *)p1, *(uint32_t *)p2);
}

static void pow_uint32(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(uint32_t *)p1;
    d2 = (double)*(uint32_t *)p2;
    dr = pow(d1, d2);
    if (dr >= UINT32_MAX)
        *(uint32_t *)res = UINT32_MAX;
    else
        *(uint32_t *)res = (uint32_t)dr;
}

static elew_op_func elew_op_uint32[TOFU_ELEW_OP_SIZE] = { mul_uint32, div_uint32, sum_uint32,
                                                        sub_uint32, max_uint32, min_uint32,
                                                        pow_uint32 };

static void elew_uint32(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_uint32[elew_op](p1, p2, res);
}

static void mul_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = *(uint16_t *)p1 * *(uint16_t *)p2;
}

static void div_uint16(void *p1, void *p2, void *res)
{
    assert(*(uint16_t *)p2);
    *(uint16_t *)res = *(uint16_t *)p1 / *(uint16_t *)p2;
}

static void sum_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = *(uint16_t *)p1 + *(uint16_t *)p2;
}

static void sub_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = *(uint16_t *)p1 - *(uint16_t *)p2;
}

static void max_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = max(*(uint16_t *)p1, *(uint16_t *)p2);
}

static void min_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = min(*(uint16_t *)p1, *(uint16_t *)p2);
}

static void pow_uint16(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(uint16_t *)p1;
    d2 = (double)*(uint16_t *)p2;
    dr = pow(d1, d2);
    if (dr >= UINT16_MAX)
        *(uint16_t *)res = UINT16_MAX;
    else
        *(uint16_t *)res = (uint16_t)dr;
}

static elew_op_func elew_op_uint16[TOFU_ELEW_OP_SIZE] = { mul_uint16, div_uint16, sum_uint16,
                                                        sub_uint16, max_uint16, min_uint16,
                                                        pow_uint16 };

static void elew_uint16(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_uint16[elew_op](p1, p2, res);
}

static void mul_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = *(uint8_t *)p1 * *(uint8_t *)p2;
}

static void div_uint8(void *p1, void *p2, void *res)
{
    assert(*(uint8_t *)p2);
    *(uint8_t *)res = *(uint8_t *)p1 / *(uint8_t *)p2;
}

static void sum_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = *(uint8_t *)p1 + *(uint8_t *)p2;
}

static void sub_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = *(uint8_t *)p1 - *(uint8_t *)p2;
}

static void max_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = max(*(uint8_t *)p1, *(uint8_t *)p2);
}

static void min_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = min(*(uint8_t *)p1, *(uint8_t *)p2);
}

static void pow_uint8(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(uint8_t *)p1;
    d2 = (double)*(uint8_t *)p2;
    dr = pow(d1, d2);
    if (dr >= UINT8_MAX)
        *(uint8_t *)res = UINT8_MAX;
    else
        *(uint8_t *)res = (uint8_t)dr;
}

static elew_op_func elew_op_uint8[TOFU_ELEW_OP_SIZE] = { mul_uint8, div_uint8, sum_uint8, sub_uint8,
                                                       max_uint8, min_uint8, pow_uint8 };

static void elew_uint8(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_uint8[elew_op](p1, p2, res);
}

static void mul_bool(void *p1, void *p2, void *res)
{
    int r = *(tofu_bool_t *)p1 * *(tofu_bool_t *)p2;
    if (r)
        *(tofu_bool_t *)res = TOFU_TRUE;
    else
        *(tofu_bool_t *)res = TOFU_FALSE;
}

static void div_bool(void *p1, void *p2, void *res)
{
    assert(*(tofu_bool_t *)p2);
    int r = *(tofu_bool_t *)p1 / *(tofu_bool_t *)p2;
    if (r)
        *(tofu_bool_t *)res = TOFU_TRUE;
    else
        *(tofu_bool_t *)res = TOFU_FALSE;
}

static void sum_bool(void *p1, void *p2, void *res)
{
    int r = *(tofu_bool_t *)p1 + *(tofu_bool_t *)p2;
    if (r)
        *(tofu_bool_t *)res = TOFU_TRUE;
    else
        *(tofu_bool_t *)res = TOFU_FALSE;
}

static void sub_bool(void *p1, void *p2, void *res)
{
    int r = *(tofu_bool_t *)p1 - *(tofu_bool_t *)p2;
    if (r)
        *(tofu_bool_t *)res = TOFU_TRUE;
    else
        *(tofu_bool_t *)res = TOFU_FALSE;
}

static void max_bool(void *p1, void *p2, void *res)
{
    *(tofu_bool_t *)res = max(*(tofu_bool_t *)p1, *(tofu_bool_t *)p2);
}

static void min_bool(void *p1, void *p2, void *res)
{
    *(tofu_bool_t *)res = min(*(tofu_bool_t *)p1, *(tofu_bool_t *)p2);
}

static void pow_bool(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(tofu_bool_t *)p1;
    d2 = (double)*(tofu_bool_t *)p2;
    dr = pow(d1, d2);
    if (dr > 0 || dr < 0)
        *(tofu_bool_t *)res = TOFU_TRUE;
    else
        *(tofu_bool_t *)res = TOFU_FALSE;
}

static elew_op_func elew_op_bool[TOFU_ELEW_OP_SIZE] = { mul_bool, div_bool, sum_bool, sub_bool,
                                                      max_bool, min_bool, pow_bool };

static void elew_bool(void *p1, void *p2, void *res, tofu_elew_op elew_op)
{
    tofu_check_elew_op(elew_op);
    elew_op_bool[elew_op](p1, p2, res);
}

static tofu_elew_func elew_func[TOFU_DTYPE_SIZE] = { elew_double, elew_float, elew_int64,  elew_int32,
                                                 elew_int16,  elew_int8,  elew_uint64, elew_uint32,
                                                 elew_uint16, elew_uint8, elew_bool };

TOFU_EXPORT void tofu_elew(void *p1, void *p2, void *res, tofu_elew_op elew_op, tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    elew_func[dtype](p1, p2, res, elew_op);
}

TOFU_EXPORT tofu_elew_func tofu_elew_getfunc(tofu_dtype dtype)
{
    tofu_check_dtype(dtype);
    return elew_func[dtype];
}

/* tofu_lrelu */
#define LRELU(pd, ps, ns, type)                                                                    \
    do {                                                                                           \
        type _s = *(type *)(ps);                                                                   \
        *(type *)(pd) = _s >= 0 ? _s : _s * (type)(ns);                                            \
    } while (0)

TOFU_EXPORT void tofu_lrelu(void *pd, const void *ps, float negslope, tofu_dtype dtype)
{
    tofu_check_dtype(dtype);

    switch (dtype) {
    case TOFU_DOUBLE:
        LRELU(pd, ps, negslope, double);
        break;
    case TOFU_FLOAT:
        LRELU(pd, ps, negslope, float);
        break;
    case TOFU_INT32:
        LRELU(pd, ps, negslope, int32_t);
        break;
    case TOFU_INT16:
        LRELU(pd, ps, negslope, int16_t);
        break;
    case TOFU_INT8:
        LRELU(pd, ps, negslope, int8_t);
        break;
    case TOFU_UINT32:
        LRELU(pd, ps, negslope, uint32_t);
        break;
    case TOFU_UINT16:
        LRELU(pd, ps, negslope, uint16_t);
        break;
    case TOFU_UINT8:
        LRELU(pd, ps, negslope, uint8_t);
        break;
    case TOFU_BOOL:
        LRELU(pd, ps, negslope, tofu_bool_t);
        break;
    default:
        assert(0 && "unsupported tofu_dtype");
        break;
    }
}
#undef LRELU

/* tofu_convert */
TOFU_EXPORT void tofu_convert(void *pd, tofu_dtype dtype_d, const void *ps, tofu_dtype dtype_s)
{
    tofu_check_dtype(dtype_d);
    tofu_check_dtype(dtype_s);

    double val_d;
    float val_f;
    int64_t val_i64;
    uint64_t val_u64;
    int32_t val_i32;
    uint32_t val_u32;
    int16_t val_i16;
    uint16_t val_u16;
    int8_t val_i8;
    uint8_t val_u8;

    switch (dtype_d) {
    case TOFU_DOUBLE:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            *(double *)pd = *(double *)ps;
            break;
        case TOFU_FLOAT:
            *(double *)pd = (double)*(float *)ps;
            break;
        case TOFU_INT64:
            *(double *)pd = (double)*(int64_t *)ps;
            break;
        case TOFU_INT32:
            *(double *)pd = (double)*(int32_t *)ps;
            break;
        case TOFU_INT16:
            *(double *)pd = (double)*(int16_t *)ps;
            break;
        case TOFU_INT8:
            *(double *)pd = (double)*(int8_t *)ps;
            break;
        case TOFU_UINT64:
            *(double *)pd = (double)*(uint64_t *)ps;
            break;
        case TOFU_UINT32:
            *(double *)pd = (double)*(uint32_t *)ps;
            break;
        case TOFU_UINT16:
            *(double *)pd = (double)*(uint16_t *)ps;
            break;
        case TOFU_UINT8:
            *(double *)pd = (double)*(uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(double *)pd = (double)*(tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_FLOAT:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= FLT_MAX)
                *(float *)pd = FLT_MAX;
            else if (val_d <= -FLT_MAX)
                *(float *)pd = -FLT_MAX;
            else
                *(float *)pd = (float)val_d;
            break;
        case TOFU_FLOAT:
            *(float *)pd = *(float *)ps;
            break;
        case TOFU_INT64:
            *(float *)pd = (float)*(int64_t *)ps;
            break;
        case TOFU_INT32:
            *(float *)pd = (float)*(int32_t *)ps;
            break;
        case TOFU_INT16:
            *(float *)pd = (float)*(int16_t *)ps;
            break;
        case TOFU_INT8:
            *(float *)pd = (float)*(int8_t *)ps;
            break;
        case TOFU_UINT64:
            *(float *)pd = (float)*(uint64_t *)ps;
            break;
        case TOFU_UINT32:
            *(float *)pd = (float)*(uint32_t *)ps;
            break;
        case TOFU_UINT16:
            *(float *)pd = (float)*(uint16_t *)ps;
            break;
        case TOFU_UINT8:
            *(float *)pd = (float)*(uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(float *)pd = (float)*(tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_INT64:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT64_MAX)
                *(int64_t *)pd = INT64_MAX;
            else if (val_d <= INT64_MIN)
                *(int64_t *)pd = INT64_MIN;
            else
                *(int64_t *)pd = (int64_t)val_d;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT64_MAX)
                *(int64_t *)pd = INT64_MAX;
            else if (val_f <= INT64_MIN)
                *(int64_t *)pd = INT64_MIN;
            else
                *(int64_t *)pd = (int64_t)val_f;
            break;
        case TOFU_INT64:
            *(int64_t *)pd = *(int64_t *)ps;
            break;
        case TOFU_INT32:
            *(int64_t *)pd = (int64_t) * (int32_t *)ps;
            break;
        case TOFU_INT16:
            *(int64_t *)pd = (int64_t) * (int16_t *)ps;
            break;
        case TOFU_INT8:
            *(int64_t *)pd = (int64_t) * (int8_t *)ps;
            break;
        case TOFU_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= INT64_MAX)
                *(int64_t *)pd = INT64_MAX;
            else
                *(int64_t *)pd = (int64_t)val_u64;
            break;
        case TOFU_UINT32:
            *(int64_t *)pd = (int64_t) * (uint32_t *)ps;
            break;
        case TOFU_UINT16:
            *(int64_t *)pd = (int64_t) * (uint16_t *)ps;
            break;
        case TOFU_UINT8:
            *(int64_t *)pd = (int64_t) * (uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(int64_t *)pd = (int64_t) * (tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_INT32:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else if (val_d <= INT32_MIN)
                *(int32_t *)pd = INT32_MIN;
            else
                *(int32_t *)pd = (int32_t)val_d;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else if (val_f <= INT32_MIN)
                *(int32_t *)pd = INT32_MIN;
            else
                *(int32_t *)pd = (int32_t)val_f;
            break;
        case TOFU_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else
                *(int32_t *)pd = (int32_t)val_i64;
            break;
        case TOFU_INT32:
            *(int32_t *)pd = *(int32_t *)ps;
            break;
        case TOFU_INT16:
            *(int32_t *)pd = (int32_t) * (int16_t *)ps;
            break;
        case TOFU_INT8:
            *(int32_t *)pd = (int32_t) * (int8_t *)ps;
            break;
        case TOFU_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else
                *(int32_t *)pd = (int32_t)val_u64;
            break;
        case TOFU_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else
                *(int32_t *)pd = (int32_t)val_u32;
            break;
        case TOFU_UINT16:
            /* printf("*ps = %d\n", *(uint16_t *)ps); */
            *(int32_t *)pd = (int32_t) * (uint16_t *)ps;
            /* printf("*pd = %d\n", *(int32_t *)pd); */
            break;
        case TOFU_UINT8:
            *(int32_t *)pd = (int32_t) * (uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(int32_t *)pd = (int32_t) * (tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_INT16:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_d <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_d;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_f <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_f;
            break;
        case TOFU_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_i64 <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_i64;
            break;
        case TOFU_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_i32 <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_i32;
            break;
        case TOFU_INT16:
            *(int16_t *)pd = *(int16_t *)ps;
            break;
        case TOFU_INT8:
            *(int16_t *)pd = (int16_t) * (int8_t *)ps;
            break;
        case TOFU_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else
                *(int16_t *)pd = (int16_t)val_u64;
            break;
        case TOFU_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else
                *(int16_t *)pd = (int16_t)val_u32;
            break;
        case TOFU_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else
                *(int16_t *)pd = (int16_t)val_u16;
            break;
        case TOFU_UINT8:
            *(int16_t *)pd = (int16_t) * (uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(int16_t *)pd = (int16_t) * (tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_INT8:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_d <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_d;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_f <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_f;
            break;
        case TOFU_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_i64 <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_i64;
            break;
        case TOFU_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_i32 <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_i32;
            break;
        case TOFU_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_i16 <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_i16;
            break;
        case TOFU_INT8:
            *(int8_t *)pd = *(int8_t *)ps;
            break;
        case TOFU_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u32;
            break;
        case TOFU_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u64;
            break;
        case TOFU_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u16;
            break;
        case TOFU_UINT8:
            val_u8 = *(uint8_t *)ps;
            if (val_u8 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u8;
            break;
        case TOFU_BOOL:
            *(int8_t *)pd = (int8_t) * (tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_UINT64:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT64_MAX)
                *(uint64_t *)pd = UINT64_MAX;
            else if (val_d < 0)
                *(uint64_t *)pd = 0;
            else
                *(uint64_t *)pd = (uint64_t)val_d;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT64_MAX)
                *(uint64_t *)pd = UINT64_MAX;
            else if (val_f < 0)
                *(uint64_t *)pd = 0;
            else
                *(uint64_t *)pd = (uint64_t)val_f;
            break;
        case TOFU_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= 0)
                *(uint64_t *)pd = (uint64_t)val_i64;
            else
                *(uint64_t *)pd = 0;
            break;
        case TOFU_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= 0)
                *(uint64_t *)pd = (uint64_t)val_i32;
            else
                *(uint64_t *)pd = 0;
            break;
        case TOFU_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= 0)
                *(uint64_t *)pd = (uint64_t)val_i16;
            else
                *(uint64_t *)pd = 0;
            break;
        case TOFU_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint64_t *)pd = (uint64_t)val_i8;
            else
                *(uint64_t *)pd = 0;
            break;
        case TOFU_UINT64:
            *(uint64_t *)pd = *(uint64_t *)ps;
            break;
        case TOFU_UINT32:
            *(uint64_t *)pd = *(uint32_t *)ps;
            break;
        case TOFU_UINT16:
            *(uint64_t *)pd = (uint64_t) * (uint16_t *)ps;
            break;
        case TOFU_UINT8:
            *(uint64_t *)pd = (uint64_t) * (uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(uint64_t *)pd = (uint64_t) * (tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_UINT32:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else if (val_d < 0)
                *(uint32_t *)pd = 0;
            else
                *(uint32_t *)pd = (uint32_t)val_d;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else if (val_f < 0)
                *(uint32_t *)pd = 0;
            else
                *(uint32_t *)pd = (uint32_t)val_f;
            break;
        case TOFU_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else if (val_i64 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i64;
            else
                *(uint32_t *)pd = 0;
            break;
        case TOFU_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i32;
            else
                *(uint32_t *)pd = 0;
            break;
        case TOFU_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i16;
            else
                *(uint32_t *)pd = 0;
            break;
        case TOFU_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i8;
            else
                *(uint32_t *)pd = 0;
            break;
        case TOFU_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else
                *(uint32_t *)pd = (uint32_t)val_u64;
            break;
        case TOFU_UINT32:
            *(uint32_t *)pd = *(uint32_t *)ps;
            break;
        case TOFU_UINT16:
            *(uint32_t *)pd = (uint32_t) * (uint16_t *)ps;
            break;
        case TOFU_UINT8:
            *(uint32_t *)pd = (uint32_t) * (uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(uint32_t *)pd = (uint32_t) * (tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_UINT16:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_d < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_d;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_f < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_f;
            break;
        case TOFU_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_i64 < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_i64;
            break;
        case TOFU_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_i32 < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_i32;
            break;
        case TOFU_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= 0)
                *(uint16_t *)pd = (uint16_t)val_i16;
            else
                *(uint16_t *)pd = 0;
            break;
        case TOFU_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint16_t *)pd = (uint16_t)val_i8;
            else
                *(uint16_t *)pd = 0;
            break;
        case TOFU_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else
                *(uint16_t *)pd = (uint16_t)val_u64;
            break;
        case TOFU_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else
                *(uint16_t *)pd = (uint16_t)val_u32;
            break;
        case TOFU_UINT16:
            *(uint16_t *)pd = *(uint16_t *)ps;
            break;
        case TOFU_UINT8:
            *(uint16_t *)pd = (uint16_t) * (uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(uint16_t *)pd = (uint16_t) * (tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_UINT8:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_d < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_d;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_f < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_f;
            break;
        case TOFU_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_i64 < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_i64;
            break;
        case TOFU_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_i32 < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_i32;
            break;
        case TOFU_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_i16 < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_i16;
            break;
        case TOFU_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint8_t *)pd = (uint8_t)val_i8;
            else
                *(uint8_t *)pd = 0;
            break;
        case TOFU_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else
                *(uint8_t *)pd = (uint8_t)val_u64;
            break;
        case TOFU_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else
                *(uint8_t *)pd = (uint8_t)val_u32;
            break;
        case TOFU_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else
                *(uint8_t *)pd = (uint8_t)val_u16;
            break;
        case TOFU_UINT8:
            *(uint8_t *)pd = *(uint8_t *)ps;
            break;
        case TOFU_BOOL:
            *(uint8_t *)pd = (uint8_t) * (tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    case TOFU_BOOL:
        switch (dtype_s) {
        case TOFU_DOUBLE:
            val_d = *(double *)ps;
            if (val_d > 0 || val_d < 0)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_FLOAT:
            val_f = *(float *)ps;
            if (val_f > 0 || val_f < 0)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_UINT8:
            val_u8 = *(uint8_t *)ps;
            if (val_u8)
                *(tofu_bool_t *)pd = TOFU_TRUE;
            else
                *(tofu_bool_t *)pd = TOFU_FALSE;
            break;
        case TOFU_BOOL:
            *(tofu_bool_t *)pd = *(tofu_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tofu_dtype");
            break;
        }
        break;
    default:
        assert(0 && "unsupported tofu_dtype");
        break;
    }
}

static const char *resize_type_name[TOFU_RESIZE_TYPE_SIZE] = { "TOFU_NEAREST", "TOFU_LINEAR" };

TOFU_EXPORT const char *tofu_resize_type_name(tofu_resize_type rtype)
{
    tofu_check_resize_type(rtype);
    return resize_type_name[rtype];
}

TOFU_EXPORT tofu_resize_type tofu_resize_type_from_str(const char *str)
{
    if (!strcmp(str, "TOFU_NEAREST"))
        return TOFU_NEAREST;
    if (!strcmp(str, "TOFU_LINEAR"))
        return TOFU_LINEAR;
    return -1;
}

static const char *sort_dir_name[TOFU_SORT_DIR_SIZE] = { "TOFU_SORT_DIR_ASCENDING",
                                                       "TOFU_SORT_DIR_DESCENDING" };

TOFU_EXPORT const char *tofu_sort_dir_name(tofu_sort_dir dir)
{
    tofu_check_sort_dir(dir);
    return sort_dir_name[dir];
}

TOFU_EXPORT tofu_sort_dir tofu_sort_dir_from_str(const char *str)
{
    if (!strcmp(str, "TOFU_SORT_DIR_ASCENDING"))
        return TOFU_SORT_DIR_ASCENDING;
    if (!strcmp(str, "TOFU_SORT_DIR_DESCENDING"))
        return TOFU_SORT_DIR_DESCENDING;
    return -1;
}
