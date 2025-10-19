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

#include <float.h>
#include "test_tofu.h"
#include "lightnettest/ln_test.h"
#include "tofu_type.h"
#include "tofu_util.h"
#include "tofu_check.h"

static void checked_setup(void)
{
}

static void checked_teardown(void)
{
}

LN_TEST_START(test_tofu_size_of)
{
    ck_assert_int_eq(tofu_size_of(TOFU_DOUBLE), sizeof(double));
    ck_assert_int_eq(tofu_size_of(TOFU_FLOAT), sizeof(float));
    ck_assert_int_eq(tofu_size_of(TOFU_INT32), sizeof(int32_t));
    ck_assert_int_eq(tofu_size_of(TOFU_INT16), sizeof(int16_t));
    ck_assert_int_eq(tofu_size_of(TOFU_INT8), sizeof(int8_t));
    ck_assert_int_eq(tofu_size_of(TOFU_UINT32), sizeof(uint32_t));
    ck_assert_int_eq(tofu_size_of(TOFU_UINT16), sizeof(uint16_t));
    ck_assert_int_eq(tofu_size_of(TOFU_UINT8), sizeof(uint8_t));
    ck_assert_int_eq(tofu_size_of(TOFU_BOOL), sizeof(tofu_bool_t));
}
LN_TEST_END

LN_TEST_START(test_tofu_psub)
{
    double p_double[2];
    float p_float[2];
    int32_t p_int32[2];
    int16_t p_int16[2];
    int8_t p_int8[2];
    uint32_t p_uint32[2];
    uint16_t p_uint16[2];
    uint8_t p_uint8[2];
    tofu_bool_t p_bool[2];

    ck_assert_int_eq(tofu_psub(&p_double[1], &p_double[0], tofu_size_of(TOFU_DOUBLE)), 1);
    ck_assert_int_eq(tofu_psub(&p_double[0], &p_double[1], tofu_size_of(TOFU_DOUBLE)), -1);
    ck_assert_int_eq(tofu_psub(&p_double[0], &p_double[0], tofu_size_of(TOFU_DOUBLE)), 0);

    ck_assert_int_eq(tofu_psub(&p_float[1], &p_float[0], tofu_size_of(TOFU_FLOAT)), 1);
    ck_assert_int_eq(tofu_psub(&p_float[0], &p_float[1], tofu_size_of(TOFU_FLOAT)), -1);
    ck_assert_int_eq(tofu_psub(&p_float[0], &p_float[0], tofu_size_of(TOFU_FLOAT)), 0);

    ck_assert_int_eq(tofu_psub(&p_int32[1], &p_int32[0], tofu_size_of(TOFU_INT32)), 1);
    ck_assert_int_eq(tofu_psub(&p_int32[0], &p_int32[1], tofu_size_of(TOFU_INT32)), -1);
    ck_assert_int_eq(tofu_psub(&p_int32[0], &p_int32[0], tofu_size_of(TOFU_INT32)), 0);

    ck_assert_int_eq(tofu_psub(&p_int16[1], &p_int16[0], tofu_size_of(TOFU_INT16)), 1);
    ck_assert_int_eq(tofu_psub(&p_int16[0], &p_int16[1], tofu_size_of(TOFU_INT16)), -1);
    ck_assert_int_eq(tofu_psub(&p_int16[0], &p_int16[0], tofu_size_of(TOFU_INT16)), 0);

    ck_assert_int_eq(tofu_psub(&p_int8[1], &p_int8[0], tofu_size_of(TOFU_INT8)), 1);
    ck_assert_int_eq(tofu_psub(&p_int8[0], &p_int8[1], tofu_size_of(TOFU_INT8)), -1);
    ck_assert_int_eq(tofu_psub(&p_int8[0], &p_int8[0], tofu_size_of(TOFU_INT8)), 0);

    ck_assert_int_eq(tofu_psub(&p_uint32[1], &p_uint32[0], tofu_size_of(TOFU_UINT32)), 1);
    ck_assert_int_eq(tofu_psub(&p_uint32[0], &p_uint32[1], tofu_size_of(TOFU_UINT32)), -1);
    ck_assert_int_eq(tofu_psub(&p_uint32[0], &p_uint32[0], tofu_size_of(TOFU_UINT32)), 0);

    ck_assert_int_eq(tofu_psub(&p_uint16[1], &p_uint16[0], tofu_size_of(TOFU_UINT16)), 1);
    ck_assert_int_eq(tofu_psub(&p_uint16[0], &p_uint16[1], tofu_size_of(TOFU_UINT16)), -1);
    ck_assert_int_eq(tofu_psub(&p_uint16[0], &p_uint16[0], tofu_size_of(TOFU_UINT16)), 0);

    ck_assert_int_eq(tofu_psub(&p_uint8[1], &p_uint8[0], tofu_size_of(TOFU_UINT8)), 1);
    ck_assert_int_eq(tofu_psub(&p_uint8[0], &p_uint8[1], tofu_size_of(TOFU_UINT8)), -1);
    ck_assert_int_eq(tofu_psub(&p_uint8[0], &p_uint8[0], tofu_size_of(TOFU_UINT8)), 0);

    ck_assert_int_eq(tofu_psub(&p_bool[1], &p_bool[0], tofu_size_of(TOFU_BOOL)), 1);
    ck_assert_int_eq(tofu_psub(&p_bool[0], &p_bool[1], tofu_size_of(TOFU_BOOL)), -1);
    ck_assert_int_eq(tofu_psub(&p_bool[0], &p_bool[0], tofu_size_of(TOFU_BOOL)), 0);
}
LN_TEST_END

LN_TEST_START(test_tofu_padd)
{
    double p_double[3];
    float p_float[3];
    int32_t p_int32[3];
    int16_t p_int16[3];
    int8_t p_int8[3];
    uint32_t p_uint32[3];
    uint16_t p_uint16[3];
    uint8_t p_uint8[3];
    tofu_bool_t p_bool[3];

    ck_assert_ptr_eq(tofu_padd(&p_double[1], 1, tofu_size_of(TOFU_DOUBLE)), &p_double[2]);
    ck_assert_ptr_eq(tofu_padd(&p_double[1], -1, tofu_size_of(TOFU_DOUBLE)), &p_double[0]);
    ck_assert_ptr_eq(tofu_padd(&p_double[1], 0, tofu_size_of(TOFU_DOUBLE)), &p_double[1]);

    ck_assert_ptr_eq(tofu_padd(&p_float[1], 1, tofu_size_of(TOFU_FLOAT)), &p_float[2]);
    ck_assert_ptr_eq(tofu_padd(&p_float[1], -1, tofu_size_of(TOFU_FLOAT)), &p_float[0]);
    ck_assert_ptr_eq(tofu_padd(&p_float[1], 0, tofu_size_of(TOFU_FLOAT)), &p_float[1]);

    ck_assert_ptr_eq(tofu_padd(&p_int32[1], 1, tofu_size_of(TOFU_INT32)), &p_int32[2]);
    ck_assert_ptr_eq(tofu_padd(&p_int32[1], -1, tofu_size_of(TOFU_INT32)), &p_int32[0]);
    ck_assert_ptr_eq(tofu_padd(&p_int32[1], 0, tofu_size_of(TOFU_INT32)), &p_int32[1]);

    ck_assert_ptr_eq(tofu_padd(&p_int16[1], 1, tofu_size_of(TOFU_INT16)), &p_int16[2]);
    ck_assert_ptr_eq(tofu_padd(&p_int16[1], -1, tofu_size_of(TOFU_INT16)), &p_int16[0]);
    ck_assert_ptr_eq(tofu_padd(&p_int16[1], 0, tofu_size_of(TOFU_INT16)), &p_int16[1]);

    ck_assert_ptr_eq(tofu_padd(&p_int8[1], 1, tofu_size_of(TOFU_INT8)), &p_int8[2]);
    ck_assert_ptr_eq(tofu_padd(&p_int8[1], -1, tofu_size_of(TOFU_INT8)), &p_int8[0]);
    ck_assert_ptr_eq(tofu_padd(&p_int8[1], 0, tofu_size_of(TOFU_INT8)), &p_int8[1]);

    ck_assert_ptr_eq(tofu_padd(&p_uint32[1], 1, tofu_size_of(TOFU_UINT32)), &p_uint32[2]);
    ck_assert_ptr_eq(tofu_padd(&p_uint32[1], -1, tofu_size_of(TOFU_UINT32)), &p_uint32[0]);
    ck_assert_ptr_eq(tofu_padd(&p_uint32[1], 0, tofu_size_of(TOFU_UINT32)), &p_uint32[1]);

    ck_assert_ptr_eq(tofu_padd(&p_uint16[1], 1, tofu_size_of(TOFU_UINT16)), &p_uint16[2]);
    ck_assert_ptr_eq(tofu_padd(&p_uint16[1], -1, tofu_size_of(TOFU_UINT16)), &p_uint16[0]);
    ck_assert_ptr_eq(tofu_padd(&p_uint16[1], 0, tofu_size_of(TOFU_UINT16)), &p_uint16[1]);

    ck_assert_ptr_eq(tofu_padd(&p_uint8[1], 1, tofu_size_of(TOFU_UINT8)), &p_uint8[2]);
    ck_assert_ptr_eq(tofu_padd(&p_uint8[1], -1, tofu_size_of(TOFU_UINT8)), &p_uint8[0]);
    ck_assert_ptr_eq(tofu_padd(&p_uint8[1], 0, tofu_size_of(TOFU_UINT8)), &p_uint8[1]);

    ck_assert_ptr_eq(tofu_padd(&p_bool[1], 1, tofu_size_of(TOFU_BOOL)), &p_bool[2]);
    ck_assert_ptr_eq(tofu_padd(&p_bool[1], -1, tofu_size_of(TOFU_BOOL)), &p_bool[0]);
    ck_assert_ptr_eq(tofu_padd(&p_bool[1], 0, tofu_size_of(TOFU_BOOL)), &p_bool[1]);
}
LN_TEST_END

LN_TEST_START(test_tofu_passign)
{
    double p_double[2] = {0, 1};
    float p_float[2] = {0, 1};
    int32_t p_int32[2] = {0, 1};
    int16_t p_int16[2] = {0, 1};
    int8_t p_int8[2] = {0, 1};
    uint32_t p_uint32[2] = {0, 1};
    uint16_t p_uint16[2] = {0, 1};
    uint8_t p_uint8[2] = {0, 1};
    tofu_bool_t p_bool[2] = {0, 1};

    tofu_passign(p_double, 0, p_double, 1, tofu_size_of(TOFU_DOUBLE));
    ck_assert(p_double[0] == p_double[1]);

    tofu_passign(p_float, 0, p_float, 1, tofu_size_of(TOFU_FLOAT));
    ck_assert(p_float[0] == p_float[1]);

    tofu_passign(p_int32, 0, p_int32, 1, tofu_size_of(TOFU_INT32));
    ck_assert(p_int32[0] == p_int32[1]);

    tofu_passign(p_int16, 0, p_int16, 1, tofu_size_of(TOFU_INT16));
    ck_assert(p_int16[0] == p_int16[1]);

    tofu_passign(p_int8, 0, p_int8, 1, tofu_size_of(TOFU_INT8));
    ck_assert(p_int8[0] == p_int8[1]);

    tofu_passign(p_uint32, 0, p_uint32, 1, tofu_size_of(TOFU_UINT32));
    ck_assert(p_uint32[0] == p_uint32[1]);

    tofu_passign(p_uint16, 0, p_uint16, 1, tofu_size_of(TOFU_UINT16));
    ck_assert(p_uint16[0] == p_uint16[1]);

    tofu_passign(p_uint8, 0, p_uint8, 1, tofu_size_of(TOFU_UINT8));
    ck_assert(p_uint8[0] == p_uint8[1]);

    tofu_passign(p_bool, 0, p_bool, 1, tofu_size_of(TOFU_BOOL));
    ck_assert(p_bool[0] == p_bool[1]);
}
LN_TEST_END

LN_TEST_START(test_tofu_dtype_fmt)
{
    const char *fmt;

    fmt = tofu_dtype_fmt(TOFU_DOUBLE);
    ck_assert_str_eq(fmt, "%.3f");

    fmt = tofu_dtype_fmt(TOFU_FLOAT);
    ck_assert_str_eq(fmt, "%.3f");

    fmt = tofu_dtype_fmt(TOFU_INT32);
    ck_assert_str_eq(fmt, "%d");

    fmt = tofu_dtype_fmt(TOFU_INT16);
    ck_assert_str_eq(fmt, "%d");

    fmt = tofu_dtype_fmt(TOFU_INT8);
    ck_assert_str_eq(fmt, "%d");

    fmt = tofu_dtype_fmt(TOFU_UINT32);
    ck_assert_str_eq(fmt, "%u");

    fmt = tofu_dtype_fmt(TOFU_UINT16);
    ck_assert_str_eq(fmt, "%u");

    fmt = tofu_dtype_fmt(TOFU_UINT8);
    ck_assert_str_eq(fmt, "%u");

    fmt = tofu_dtype_fmt(TOFU_BOOL);
    ck_assert_str_eq(fmt, "%d");
}
LN_TEST_END

LN_TEST_START(test_tofu_dtype_max)
{
    void *max;
    tofu_dtype dtype;

    dtype = TOFU_DOUBLE;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_double_eq_tol(*(double *)max, DBL_MAX, 1e-6);
    tofu_free(max);

    dtype = TOFU_FLOAT;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_double_eq_tol(*(float *)max, FLT_MAX, 1e-6);
    tofu_free(max);

    dtype = TOFU_INT32;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_int_eq(*(int32_t *)max, INT32_MAX);
    tofu_free(max);

    dtype = TOFU_INT16;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_int_eq(*(int16_t *)max, INT16_MAX);
    tofu_free(max);

    dtype = TOFU_INT8;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_int_eq(*(int8_t *)max, INT8_MAX);
    tofu_free(max);

    dtype = TOFU_UINT32;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_uint_eq(*(uint32_t *)max, UINT32_MAX);
    tofu_free(max);

    dtype = TOFU_UINT16;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_uint_eq(*(uint16_t *)max, UINT16_MAX);
    tofu_free(max);

    dtype = TOFU_UINT8;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_uint_eq(*(uint8_t *)max, UINT8_MAX);
    tofu_free(max);

    dtype = TOFU_BOOL;
    max = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_max(dtype, max);
    ck_assert_int_eq(*(tofu_bool_t *)max, TOFU_TRUE);
    tofu_free(max);
}
LN_TEST_END

LN_TEST_START(test_tofu_dtype_min)
{
    void *min;
    tofu_dtype dtype;

    dtype = TOFU_DOUBLE;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_double_eq_tol(*(double *)min, -DBL_MAX, 1e-6);
    tofu_free(min);

    dtype = TOFU_FLOAT;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_double_eq_tol(*(float *)min, -FLT_MAX, 1e-6);
    tofu_free(min);

    dtype = TOFU_INT32;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_int_eq(*(int32_t *)min, INT32_MIN);
    tofu_free(min);

    dtype = TOFU_INT16;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_int_eq(*(int16_t *)min, INT16_MIN);
    tofu_free(min);

    dtype = TOFU_INT8;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_int_eq(*(int8_t *)min, INT8_MIN);
    tofu_free(min);

    dtype = TOFU_UINT32;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_uint_eq(*(uint32_t *)min, 0);
    tofu_free(min);

    dtype = TOFU_UINT16;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_uint_eq(*(uint16_t *)min, 0);
    tofu_free(min);

    dtype = TOFU_UINT8;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_uint_eq(*(uint8_t *)min, 0);
    tofu_free(min);

    dtype = TOFU_BOOL;
    min = tofu_alloc(tofu_size_of(dtype));
    tofu_dtype_min(dtype, min);
    ck_assert_int_eq(*(tofu_bool_t *)min, TOFU_FALSE);
    tofu_free(min);
}
LN_TEST_END

LN_TEST_START(test_tofu_dtype_max_double)
{
    tofu_dtype dtype;
    double max;

    dtype = TOFU_DOUBLE;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)DBL_MAX, 1e-6);

    dtype = TOFU_FLOAT;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)FLT_MAX, 1e-6);

    dtype = TOFU_INT32;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)INT32_MAX, 1e-6);

    dtype = TOFU_INT16;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)INT16_MAX, 1e-6);

    dtype = TOFU_INT8;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)INT8_MAX, 1e-6);

    dtype = TOFU_UINT32;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)UINT32_MAX, 1e-6);

    dtype = TOFU_UINT16;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)UINT16_MAX, 1e-6);

    dtype = TOFU_UINT8;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)UINT8_MAX, 1e-6);

    dtype = TOFU_BOOL;
    max = tofu_dtype_max_double(dtype);
    ck_assert_double_eq_tol(max, (double)TOFU_TRUE, 1e-6);
}
LN_TEST_END

LN_TEST_START(test_tofu_dtype_min_double)
{
    tofu_dtype dtype;
    double min;

    dtype = TOFU_DOUBLE;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)-DBL_MAX, 1e-6);

    dtype = TOFU_FLOAT;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)-FLT_MAX, 1e-6);

    dtype = TOFU_INT32;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)INT32_MIN, 1e-6);

    dtype = TOFU_INT16;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)INT16_MIN, 1e-6);

    dtype = TOFU_INT8;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)INT8_MIN, 1e-6);

    dtype = TOFU_UINT32;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)0, 1e-6);

    dtype = TOFU_UINT16;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)0, 1e-6);

    dtype = TOFU_UINT8;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)0, 1e-6);

    dtype = TOFU_BOOL;
    min = tofu_dtype_min_double(dtype);
    ck_assert_double_eq_tol(min, (double)TOFU_FALSE, 1e-6);
}
LN_TEST_END

LN_TEST_START(test_tofu_pointer_sub)
{
    double p_double[2];
    float p_float[2];
    int32_t p_int32[2];
    int16_t p_int16[2];
    int8_t p_int8[2];
    uint32_t p_uint32[2];
    uint16_t p_uint16[2];
    uint8_t p_uint8[2];
    tofu_bool_t p_bool[2];

    ck_assert_int_eq(tofu_pointer_sub(&p_double[1], &p_double[0], (TOFU_DOUBLE)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_double[0], &p_double[1], (TOFU_DOUBLE)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_double[0], &p_double[0], (TOFU_DOUBLE)), 0);

    ck_assert_int_eq(tofu_pointer_sub(&p_float[1], &p_float[0], (TOFU_FLOAT)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_float[0], &p_float[1], (TOFU_FLOAT)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_float[0], &p_float[0], (TOFU_FLOAT)), 0);

    ck_assert_int_eq(tofu_pointer_sub(&p_int32[1], &p_int32[0], (TOFU_INT32)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_int32[0], &p_int32[1], (TOFU_INT32)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_int32[0], &p_int32[0], (TOFU_INT32)), 0);

    ck_assert_int_eq(tofu_pointer_sub(&p_int16[1], &p_int16[0], (TOFU_INT16)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_int16[0], &p_int16[1], (TOFU_INT16)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_int16[0], &p_int16[0], (TOFU_INT16)), 0);

    ck_assert_int_eq(tofu_pointer_sub(&p_int8[1], &p_int8[0], (TOFU_INT8)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_int8[0], &p_int8[1], (TOFU_INT8)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_int8[0], &p_int8[0], (TOFU_INT8)), 0);

    ck_assert_int_eq(tofu_pointer_sub(&p_uint32[1], &p_uint32[0], (TOFU_UINT32)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_uint32[0], &p_uint32[1], (TOFU_UINT32)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_uint32[0], &p_uint32[0], (TOFU_UINT32)), 0);

    ck_assert_int_eq(tofu_pointer_sub(&p_uint16[1], &p_uint16[0], (TOFU_UINT16)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_uint16[0], &p_uint16[1], (TOFU_UINT16)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_uint16[0], &p_uint16[0], (TOFU_UINT16)), 0);

    ck_assert_int_eq(tofu_pointer_sub(&p_uint8[1], &p_uint8[0], (TOFU_UINT8)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_uint8[0], &p_uint8[1], (TOFU_UINT8)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_uint8[0], &p_uint8[0], (TOFU_UINT8)), 0);

    ck_assert_int_eq(tofu_pointer_sub(&p_bool[1], &p_bool[0], (TOFU_BOOL)), 1);
    ck_assert_int_eq(tofu_pointer_sub(&p_bool[0], &p_bool[1], (TOFU_BOOL)), -1);
    ck_assert_int_eq(tofu_pointer_sub(&p_bool[0], &p_bool[0], (TOFU_BOOL)), 0);
}
LN_TEST_END

LN_TEST_START(test_tofu_pointer_add)
{
    double p_double[3];
    float p_float[3];
    int32_t p_int32[3];
    int16_t p_int16[3];
    int8_t p_int8[3];
    uint32_t p_uint32[3];
    uint16_t p_uint16[3];
    uint8_t p_uint8[3];
    tofu_bool_t p_bool[3];

    ck_assert_ptr_eq(tofu_pointer_add(&p_double[1], 1, (TOFU_DOUBLE)), &p_double[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_double[1], -1, (TOFU_DOUBLE)), &p_double[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_double[1], 0, (TOFU_DOUBLE)), &p_double[1]);

    ck_assert_ptr_eq(tofu_pointer_add(&p_float[1], 1, (TOFU_FLOAT)), &p_float[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_float[1], -1, (TOFU_FLOAT)), &p_float[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_float[1], 0, (TOFU_FLOAT)), &p_float[1]);

    ck_assert_ptr_eq(tofu_pointer_add(&p_int32[1], 1, (TOFU_INT32)), &p_int32[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_int32[1], -1, (TOFU_INT32)), &p_int32[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_int32[1], 0, (TOFU_INT32)), &p_int32[1]);

    ck_assert_ptr_eq(tofu_pointer_add(&p_int16[1], 1, (TOFU_INT16)), &p_int16[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_int16[1], -1, (TOFU_INT16)), &p_int16[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_int16[1], 0, (TOFU_INT16)), &p_int16[1]);

    ck_assert_ptr_eq(tofu_pointer_add(&p_int8[1], 1, (TOFU_INT8)), &p_int8[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_int8[1], -1, (TOFU_INT8)), &p_int8[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_int8[1], 0, (TOFU_INT8)), &p_int8[1]);

    ck_assert_ptr_eq(tofu_pointer_add(&p_uint32[1], 1, (TOFU_UINT32)), &p_uint32[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_uint32[1], -1, (TOFU_UINT32)), &p_uint32[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_uint32[1], 0, (TOFU_UINT32)), &p_uint32[1]);

    ck_assert_ptr_eq(tofu_pointer_add(&p_uint16[1], 1, (TOFU_UINT16)), &p_uint16[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_uint16[1], -1, (TOFU_UINT16)), &p_uint16[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_uint16[1], 0, (TOFU_UINT16)), &p_uint16[1]);

    ck_assert_ptr_eq(tofu_pointer_add(&p_uint8[1], 1, (TOFU_UINT8)), &p_uint8[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_uint8[1], -1, (TOFU_UINT8)), &p_uint8[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_uint8[1], 0, (TOFU_UINT8)), &p_uint8[1]);

    ck_assert_ptr_eq(tofu_pointer_add(&p_bool[1], 1, (TOFU_BOOL)), &p_bool[2]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_bool[1], -1, (TOFU_BOOL)), &p_bool[0]);
    ck_assert_ptr_eq(tofu_pointer_add(&p_bool[1], 0, (TOFU_BOOL)), &p_bool[1]);
}
LN_TEST_END

LN_TEST_START(test_tofu_pointer_assign)
{
    double p_double[2] = {0, 1};
    float p_float[2] = {0, 1};
    int32_t p_int32[2] = {0, 1};
    int16_t p_int16[2] = {0, 1};
    int8_t p_int8[2] = {0, 1};
    uint32_t p_uint32[2] = {0, 1};
    uint16_t p_uint16[2] = {0, 1};
    uint8_t p_uint8[2] = {0, 1};
    tofu_bool_t p_bool[2] = {0, 1};

    tofu_pointer_assign(p_double, 0, p_double, 1, (TOFU_DOUBLE));
    ck_assert(p_double[0] == p_double[1]);

    tofu_pointer_assign(p_float, 0, p_float, 1, (TOFU_FLOAT));
    ck_assert(p_float[0] == p_float[1]);

    tofu_pointer_assign(p_int32, 0, p_int32, 1, (TOFU_INT32));
    ck_assert(p_int32[0] == p_int32[1]);

    tofu_pointer_assign(p_int16, 0, p_int16, 1, (TOFU_INT16));
    ck_assert(p_int16[0] == p_int16[1]);

    tofu_pointer_assign(p_int8, 0, p_int8, 1, (TOFU_INT8));
    ck_assert(p_int8[0] == p_int8[1]);

    tofu_pointer_assign(p_uint32, 0, p_uint32, 1, (TOFU_UINT32));
    ck_assert(p_uint32[0] == p_uint32[1]);

    tofu_pointer_assign(p_uint16, 0, p_uint16, 1, (TOFU_UINT16));
    ck_assert(p_uint16[0] == p_uint16[1]);

    tofu_pointer_assign(p_uint8, 0, p_uint8, 1, (TOFU_UINT8));
    ck_assert(p_uint8[0] == p_uint8[1]);

    tofu_pointer_assign(p_bool, 0, p_bool, 1, (TOFU_BOOL));
    ck_assert(p_bool[0] == p_bool[1]);
}
LN_TEST_END

LN_TEST_START(test_tofu_fprintf)
{
    FILE *fp;
    double val_double = 0.12345;
    float val_float = 0.12345;
    int32_t val_int32 = -1;
    int16_t val_int16 = -1;
    int8_t val_int8 = -1;
    uint32_t val_uint32 = 1;
    uint16_t val_uint16 = 1;
    uint8_t val_uint8 = 1;
    tofu_bool_t val_bool = TOFU_TRUE;
    char s[10];

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_double, TOFU_DOUBLE), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "0.123");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_float, TOFU_FLOAT), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "0.123");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, "%.1f", &val_float, TOFU_FLOAT), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "0.1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, "%.1f", &val_double, TOFU_DOUBLE), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "0.1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_int32, TOFU_INT32), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "-1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_int16, TOFU_INT16), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "-1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_int8, TOFU_INT8), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "-1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_uint32, TOFU_UINT32), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_uint16, TOFU_UINT16), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_uint8, TOFU_UINT8), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_int_ge(tofu_fprintf(fp, NULL, &val_bool, TOFU_BOOL), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "1");
    fclose(fp);
}
LN_TEST_END

LN_TEST_START(test_tofu_fprintf_getfunc)
{
    FILE *fp;
    double val_double = 0.12345;
    float val_float = 0.12345;
    int32_t val_int32 = -1;
    int16_t val_int16 = -1;
    int8_t val_int8 = -1;
    uint32_t val_uint32 = 1;
    uint16_t val_uint16 = 1;
    uint8_t val_uint8 = 1;
    tofu_bool_t val_bool = TOFU_TRUE;
    tofu_fprintf_func gfprintf_func;
    char s[10];

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_DOUBLE);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_double), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "0.123");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_FLOAT);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_float), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "0.123");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_FLOAT);
    ck_assert_int_ge(gfprintf_func(fp, "%.1f", &val_float), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "0.1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_DOUBLE);
    ck_assert_int_ge(gfprintf_func(fp, "%.1f", &val_double), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "0.1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_INT32);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int32), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "-1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_INT16);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int16), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "-1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_INT8);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int8), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "-1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_UINT32);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint32), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_UINT16);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint16), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_UINT8);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint8), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "1");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    gfprintf_func = tofu_fprintf_getfunc(TOFU_BOOL);
    ck_assert_int_ge(gfprintf_func(fp, NULL, &val_bool), 0);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
    ck_assert_str_eq(s, "1");
    fclose(fp);
}
LN_TEST_END

LN_TEST_START(test_tofu_cmp)
{
    double val1_double = 1, val2_double = 2;
    float val1_float = 1, val2_float = 2;
    int32_t val1_int32 = 1, val2_int32 = 2;
    int16_t val1_int16 = 1, val2_int16 = 2;
    int8_t val1_int8 = 1, val2_int8 = 2;
    uint32_t val1_uint32 = 1, val2_uint32 = 2;
    uint16_t val1_uint16 = 1, val2_uint16 = 2;
    uint8_t val1_uint8 = 1, val2_uint8 = 2;
    tofu_bool_t val1_bool = TOFU_FALSE, val2_bool = TOFU_TRUE;

    ck_assert(tofu_cmp(&val1_double, &val2_double, TOFU_DOUBLE) < 0);
    ck_assert(tofu_cmp(&val2_double, &val1_double, TOFU_DOUBLE) > 0);
    ck_assert(tofu_cmp(&val1_double, &val1_double, TOFU_DOUBLE) == 0);

    ck_assert(tofu_cmp(&val1_float, &val2_float, TOFU_FLOAT) < 0);
    ck_assert(tofu_cmp(&val2_float, &val1_float, TOFU_FLOAT) > 0);
    ck_assert(tofu_cmp(&val1_float, &val1_float, TOFU_FLOAT) == 0);

    ck_assert(tofu_cmp(&val1_int32, &val2_int32, TOFU_INT32) < 0);
    ck_assert(tofu_cmp(&val2_int32, &val1_int32, TOFU_INT32) > 0);
    ck_assert(tofu_cmp(&val1_int32, &val1_int32, TOFU_INT32) == 0);

    ck_assert(tofu_cmp(&val1_int16, &val2_int16, TOFU_INT16) < 0);
    ck_assert(tofu_cmp(&val2_int16, &val1_int16, TOFU_INT16) > 0);
    ck_assert(tofu_cmp(&val1_int16, &val1_int16, TOFU_INT16) == 0);

    ck_assert(tofu_cmp(&val1_int8, &val2_int8, TOFU_INT8) < 0);
    ck_assert(tofu_cmp(&val2_int8, &val1_int8, TOFU_INT8) > 0);
    ck_assert(tofu_cmp(&val1_int8, &val1_int8, TOFU_INT8) == 0);

    ck_assert(tofu_cmp(&val1_uint32, &val2_uint32, TOFU_UINT32) < 0);
    ck_assert(tofu_cmp(&val2_uint32, &val1_uint32, TOFU_UINT32) > 0);
    ck_assert(tofu_cmp(&val1_uint32, &val1_uint32, TOFU_UINT32) == 0);

    ck_assert(tofu_cmp(&val1_uint16, &val2_uint16, TOFU_UINT16) < 0);
    ck_assert(tofu_cmp(&val2_uint16, &val1_uint16, TOFU_UINT16) > 0);
    ck_assert(tofu_cmp(&val1_uint16, &val1_uint16, TOFU_UINT16) == 0);

    ck_assert(tofu_cmp(&val1_uint8, &val2_uint8, TOFU_UINT8) < 0);
    ck_assert(tofu_cmp(&val2_uint8, &val1_uint8, TOFU_UINT8) > 0);
    ck_assert(tofu_cmp(&val1_uint8, &val1_uint8, TOFU_UINT8) == 0);

    ck_assert(tofu_cmp(&val1_bool, &val2_bool, TOFU_BOOL) < 0);
    ck_assert(tofu_cmp(&val2_bool, &val1_bool, TOFU_BOOL) > 0);
    ck_assert(tofu_cmp(&val1_bool, &val1_bool, TOFU_BOOL) == 0);
}
LN_TEST_END

LN_TEST_START(test_tofu_cmp_getfunc)
{
    double val1_double = 1, val2_double = 2;
    float val1_float = 1, val2_float = 2;
    int32_t val1_int32 = 1, val2_int32 = 2;
    int16_t val1_int16 = 1, val2_int16 = 2;
    int8_t val1_int8 = 1, val2_int8 = 2;
    uint32_t val1_uint32 = 1, val2_uint32 = 2;
    uint16_t val1_uint16 = 1, val2_uint16 = 2;
    uint8_t val1_uint8 = 1, val2_uint8 = 2;
    tofu_bool_t val1_bool = TOFU_FALSE, val2_bool = TOFU_TRUE;
    tofu_cmp_func gcmp_func;

    gcmp_func = tofu_cmp_getfunc(TOFU_DOUBLE);
    ck_assert(gcmp_func(&val1_double, &val2_double) < 0);
    ck_assert(gcmp_func(&val2_double, &val1_double) > 0);
    ck_assert(gcmp_func(&val1_double, &val1_double) == 0);

    gcmp_func = tofu_cmp_getfunc(TOFU_FLOAT);
    ck_assert(gcmp_func(&val1_float, &val2_float) < 0);
    ck_assert(gcmp_func(&val2_float, &val1_float) > 0);
    ck_assert(gcmp_func(&val1_float, &val1_float) == 0);

    gcmp_func = tofu_cmp_getfunc(TOFU_INT32);
    ck_assert(gcmp_func(&val1_int32, &val2_int32) < 0);
    ck_assert(gcmp_func(&val2_int32, &val1_int32) > 0);
    ck_assert(gcmp_func(&val1_int32, &val1_int32) == 0);

    gcmp_func = tofu_cmp_getfunc(TOFU_INT16);
    ck_assert(gcmp_func(&val1_int16, &val2_int16) < 0);
    ck_assert(gcmp_func(&val2_int16, &val1_int16) > 0);
    ck_assert(gcmp_func(&val1_int16, &val1_int16) == 0);

    gcmp_func = tofu_cmp_getfunc(TOFU_INT8);
    ck_assert(gcmp_func(&val1_int8, &val2_int8) < 0);
    ck_assert(gcmp_func(&val2_int8, &val1_int8) > 0);
    ck_assert(gcmp_func(&val1_int8, &val1_int8) == 0);

    gcmp_func = tofu_cmp_getfunc(TOFU_UINT32);
    ck_assert(gcmp_func(&val1_uint32, &val2_uint32) < 0);
    ck_assert(gcmp_func(&val2_uint32, &val1_uint32) > 0);
    ck_assert(gcmp_func(&val1_uint32, &val1_uint32) == 0);

    gcmp_func = tofu_cmp_getfunc(TOFU_UINT16);
    ck_assert(gcmp_func(&val1_uint16, &val2_uint16) < 0);
    ck_assert(gcmp_func(&val2_uint16, &val1_uint16) > 0);
    ck_assert(gcmp_func(&val1_uint16, &val1_uint16) == 0);

    gcmp_func = tofu_cmp_getfunc(TOFU_UINT8);
    ck_assert(gcmp_func(&val1_uint8, &val2_uint8) < 0);
    ck_assert(gcmp_func(&val2_uint8, &val1_uint8) > 0);
    ck_assert(gcmp_func(&val1_uint8, &val1_uint8) == 0);

    gcmp_func = tofu_cmp_getfunc(TOFU_BOOL);
    ck_assert(gcmp_func(&val1_bool, &val2_bool) < 0);
    ck_assert(gcmp_func(&val2_bool, &val1_bool) > 0);
    ck_assert(gcmp_func(&val1_bool, &val1_bool) == 0);
}
LN_TEST_END

LN_TEST_START(test_tofu_elew)
{
    double val1_double = 1, val2_double = 2, val3_double;
    float val1_float = 1, val2_float = 2, val3_float;
    int32_t val1_int32 = 1, val2_int32 = 2, val3_int32;
    int16_t val1_int16 = 1, val2_int16 = 2, val3_int16;
    int8_t val1_int8 = 1, val2_int8 = 2, val3_int8;
    uint32_t val1_uint32 = 1, val2_uint32 = 2, val3_uint32;
    uint16_t val1_uint16 = 1, val2_uint16 = 2, val3_uint16;
    uint8_t val1_uint8 = 1, val2_uint8 = 2, val3_uint8;
    tofu_bool_t val1_bool = TOFU_FALSE, val2_bool = TOFU_TRUE, val3_bool;

    /* TOFU_MUL */
    tofu_elew(&val1_double, &val2_double, &val3_double, TOFU_MUL, TOFU_DOUBLE);
    ck_assert(val3_double == 2);

    tofu_elew(&val1_float, &val2_float, &val3_float, TOFU_MUL, TOFU_FLOAT);
    ck_assert(val3_float == 2);

    tofu_elew(&val1_int32, &val2_int32, &val3_int32, TOFU_MUL, TOFU_INT32);
    ck_assert(val3_int32 == 2);

    tofu_elew(&val1_int16, &val2_int16, &val3_int16, TOFU_MUL, TOFU_INT16);
    ck_assert(val3_int16 == 2);

    tofu_elew(&val1_int8, &val2_int8, &val3_int8, TOFU_MUL, TOFU_INT8);
    ck_assert(val3_int8 == 2);

    tofu_elew(&val1_uint32, &val2_uint32, &val3_uint32, TOFU_MUL, TOFU_UINT32);
    ck_assert(val3_uint32 == 2);

    tofu_elew(&val1_uint16, &val2_uint16, &val3_uint16, TOFU_MUL, TOFU_UINT16);
    ck_assert(val3_uint16 == 2);

    tofu_elew(&val1_uint8, &val2_uint8, &val3_uint8, TOFU_MUL, TOFU_UINT8);
    ck_assert(val3_uint8 == 2);

    tofu_elew(&val1_bool, &val2_bool, &val3_bool, TOFU_MUL, TOFU_BOOL);
    ck_assert(val3_bool == TOFU_FALSE);

    /* TOFU_DIV */
    tofu_elew(&val2_double, &val1_double, &val3_double, TOFU_DIV, TOFU_DOUBLE);
    ck_assert(val3_double == 2);

    tofu_elew(&val2_float, &val1_float, &val3_float, TOFU_DIV, TOFU_FLOAT);
    ck_assert(val3_float == 2);

    tofu_elew(&val2_int32, &val1_int32, &val3_int32, TOFU_DIV, TOFU_INT32);
    ck_assert(val3_int32 == 2);

    tofu_elew(&val2_int16, &val1_int16, &val3_int16, TOFU_DIV, TOFU_INT16);
    ck_assert(val3_int16 == 2);

    tofu_elew(&val2_int8, &val1_int8, &val3_int8, TOFU_DIV, TOFU_INT8);
    ck_assert(val3_int8 == 2);

    tofu_elew(&val2_uint32, &val1_uint32, &val3_uint32, TOFU_DIV, TOFU_UINT32);
    ck_assert(val3_uint32 == 2);

    tofu_elew(&val2_uint16, &val1_uint16, &val3_uint16, TOFU_DIV, TOFU_UINT16);
    ck_assert(val3_uint16 == 2);

    tofu_elew(&val2_uint8, &val1_uint8, &val3_uint8, TOFU_DIV, TOFU_UINT8);
    ck_assert(val3_uint8 == 2);

    /* tofu_elew(&val2_bool, &val1_bool, &val3_bool, TOFU_DIV, TOFU_BOOL); */
    /* ck_assert(val3_bool == 2); */

    /* TOFU_SUM */
    tofu_elew(&val2_double, &val1_double, &val3_double, TOFU_SUM, TOFU_DOUBLE);
    ck_assert(val3_double == 3);

    tofu_elew(&val2_float, &val1_float, &val3_float, TOFU_SUM, TOFU_FLOAT);
    ck_assert(val3_float == 3);

    tofu_elew(&val2_int32, &val1_int32, &val3_int32, TOFU_SUM, TOFU_INT32);
    ck_assert(val3_int32 == 3);

    tofu_elew(&val2_int16, &val1_int16, &val3_int16, TOFU_SUM, TOFU_INT16);
    ck_assert(val3_int16 == 3);

    tofu_elew(&val2_int8, &val1_int8, &val3_int8, TOFU_SUM, TOFU_INT8);
    ck_assert(val3_int8 == 3);

    tofu_elew(&val2_uint32, &val1_uint32, &val3_uint32, TOFU_SUM, TOFU_UINT32);
    ck_assert(val3_uint32 == 3);

    tofu_elew(&val2_uint16, &val1_uint16, &val3_uint16, TOFU_SUM, TOFU_UINT16);
    ck_assert(val3_uint16 == 3);

    tofu_elew(&val2_uint8, &val1_uint8, &val3_uint8, TOFU_SUM, TOFU_UINT8);
    ck_assert(val3_uint8 == 3);

    tofu_elew(&val2_bool, &val1_bool, &val3_bool, TOFU_SUM, TOFU_BOOL);
    ck_assert(val3_bool == 1);

    /* TOFU_SUB */
    tofu_elew(&val2_double, &val1_double, &val3_double, TOFU_SUB, TOFU_DOUBLE);
    ck_assert(val3_double == 1);

    tofu_elew(&val2_float, &val1_float, &val3_float, TOFU_SUB, TOFU_FLOAT);
    ck_assert(val3_float == 1);

    tofu_elew(&val2_int32, &val1_int32, &val3_int32, TOFU_SUB, TOFU_INT32);
    ck_assert(val3_int32 == 1);

    tofu_elew(&val2_int16, &val1_int16, &val3_int16, TOFU_SUB, TOFU_INT16);
    ck_assert(val3_int16 == 1);

    tofu_elew(&val2_int8, &val1_int8, &val3_int8, TOFU_SUB, TOFU_INT8);
    ck_assert(val3_int8 == 1);

    tofu_elew(&val2_uint32, &val1_uint32, &val3_uint32, TOFU_SUB, TOFU_UINT32);
    ck_assert(val3_uint32 == 1);

    tofu_elew(&val2_uint16, &val1_uint16, &val3_uint16, TOFU_SUB, TOFU_UINT16);
    ck_assert(val3_uint16 == 1);

    tofu_elew(&val2_uint8, &val1_uint8, &val3_uint8, TOFU_SUB, TOFU_UINT8);
    ck_assert(val3_uint8 == 1);

    tofu_elew(&val2_bool, &val1_bool, &val3_bool, TOFU_SUB, TOFU_BOOL);
    ck_assert(val3_bool == 1);

    /* TOFU_MAX */
    tofu_elew(&val2_double, &val1_double, &val3_double, TOFU_MAX, TOFU_DOUBLE);
    ck_assert(val3_double == 2);

    tofu_elew(&val2_float, &val1_float, &val3_float, TOFU_MAX, TOFU_FLOAT);
    ck_assert(val3_float == 2);

    tofu_elew(&val2_int32, &val1_int32, &val3_int32, TOFU_MAX, TOFU_INT32);
    ck_assert(val3_int32 == 2);

    tofu_elew(&val2_int16, &val1_int16, &val3_int16, TOFU_MAX, TOFU_INT16);
    ck_assert(val3_int16 == 2);

    tofu_elew(&val2_int8, &val1_int8, &val3_int8, TOFU_MAX, TOFU_INT8);
    ck_assert(val3_int8 == 2);

    tofu_elew(&val2_uint32, &val1_uint32, &val3_uint32, TOFU_MAX, TOFU_UINT32);
    ck_assert(val3_uint32 == 2);

    tofu_elew(&val2_uint16, &val1_uint16, &val3_uint16, TOFU_MAX, TOFU_UINT16);
    ck_assert(val3_uint16 == 2);

    tofu_elew(&val2_uint8, &val1_uint8, &val3_uint8, TOFU_MAX, TOFU_UINT8);
    ck_assert(val3_uint8 == 2);

    tofu_elew(&val2_bool, &val1_bool, &val3_bool, TOFU_MAX, TOFU_BOOL);
    ck_assert(val3_bool == 1);

    /* TOFU_MIN */
    tofu_elew(&val2_double, &val1_double, &val3_double, TOFU_MIN, TOFU_DOUBLE);
    ck_assert(val3_double == 1);

    tofu_elew(&val2_float, &val1_float, &val3_float, TOFU_MIN, TOFU_FLOAT);
    ck_assert(val3_float == 1);

    tofu_elew(&val2_int32, &val1_int32, &val3_int32, TOFU_MIN, TOFU_INT32);
    ck_assert(val3_int32 == 1);

    tofu_elew(&val2_int16, &val1_int16, &val3_int16, TOFU_MIN, TOFU_INT16);
    ck_assert(val3_int16 == 1);

    tofu_elew(&val2_int8, &val1_int8, &val3_int8, TOFU_MIN, TOFU_INT8);
    ck_assert(val3_int8 == 1);

    tofu_elew(&val2_uint32, &val1_uint32, &val3_uint32, TOFU_MIN, TOFU_UINT32);
    ck_assert(val3_uint32 == 1);

    tofu_elew(&val2_uint16, &val1_uint16, &val3_uint16, TOFU_MIN, TOFU_UINT16);
    ck_assert(val3_uint16 == 1);

    tofu_elew(&val2_uint8, &val1_uint8, &val3_uint8, TOFU_MIN, TOFU_UINT8);
    ck_assert(val3_uint8 == 1);

    tofu_elew(&val2_bool, &val1_bool, &val3_bool, TOFU_MIN, TOFU_BOOL);
    ck_assert(val3_bool == 0);

    /* TOFU_POW */
    tofu_elew(&val2_double, &val1_double, &val3_double, TOFU_POW, TOFU_DOUBLE);
    ck_assert(val3_double == 2);

    tofu_elew(&val2_float, &val1_float, &val3_float, TOFU_POW, TOFU_FLOAT);
    ck_assert(val3_float == 2);

    tofu_elew(&val2_int32, &val1_int32, &val3_int32, TOFU_POW, TOFU_INT32);
    ck_assert(val3_int32 == 2);

    tofu_elew(&val2_int16, &val1_int16, &val3_int16, TOFU_POW, TOFU_INT16);
    ck_assert(val3_int16 == 2);

    tofu_elew(&val2_int8, &val1_int8, &val3_int8, TOFU_POW, TOFU_INT8);
    ck_assert(val3_int8 == 2);

    tofu_elew(&val2_uint32, &val1_uint32, &val3_uint32, TOFU_POW, TOFU_UINT32);
    ck_assert(val3_uint32 == 2);

    tofu_elew(&val2_uint16, &val1_uint16, &val3_uint16, TOFU_POW, TOFU_UINT16);
    ck_assert(val3_uint16 == 2);

    tofu_elew(&val2_uint8, &val1_uint8, &val3_uint8, TOFU_POW, TOFU_UINT8);
    ck_assert(val3_uint8 == 2);

    tofu_elew(&val2_bool, &val1_bool, &val3_bool, TOFU_POW, TOFU_BOOL);
    ck_assert(val3_bool == 1);
}
LN_TEST_END

LN_TEST_START(test_tofu_elew_getfunc)
{
    double val1_double = 1, val2_double = 2, val3_double;
    float val1_float = 1, val2_float = 2, val3_float;
    int32_t val1_int32 = 1, val2_int32 = 2, val3_int32;
    int16_t val1_int16 = 1, val2_int16 = 2, val3_int16;
    int8_t val1_int8 = 1, val2_int8 = 2, val3_int8;
    uint32_t val1_uint32 = 1, val2_uint32 = 2, val3_uint32;
    uint16_t val1_uint16 = 1, val2_uint16 = 2, val3_uint16;
    uint8_t val1_uint8 = 1, val2_uint8 = 2, val3_uint8;
    tofu_bool_t val1_bool = TOFU_FALSE, val2_bool = TOFU_TRUE, val3_bool;
    tofu_elew_func elew_func;

    elew_func = tofu_elew_getfunc(TOFU_DOUBLE);
    elew_func(&val2_double, &val1_double, &val3_double, TOFU_MUL);
    ck_assert(val3_double == 2);

    elew_func = tofu_elew_getfunc(TOFU_FLOAT);
    elew_func(&val2_float, &val1_float, &val3_float, TOFU_MUL);
    ck_assert(val3_float == 2);

    elew_func = tofu_elew_getfunc(TOFU_INT32);
    elew_func(&val2_int32, &val1_int32, &val3_int32, TOFU_DIV);
    ck_assert(val3_int32 == 2);

    elew_func = tofu_elew_getfunc(TOFU_INT16);
    elew_func(&val2_int16, &val1_int16, &val3_int16, TOFU_SUM);
    ck_assert(val3_int16 == 3);

    elew_func = tofu_elew_getfunc(TOFU_INT8);
    elew_func(&val2_int8, &val1_int8, &val3_int8, TOFU_MAX);
    ck_assert(val3_int8 == 2);

    elew_func = tofu_elew_getfunc(TOFU_UINT32);
    elew_func(&val2_uint32, &val1_uint32, &val3_uint32, TOFU_MIN);
    ck_assert(val3_uint32 == 1);

    elew_func = tofu_elew_getfunc(TOFU_UINT16);
    elew_func(&val2_uint16, &val1_uint16, &val3_uint16, TOFU_POW);
    ck_assert(val3_uint16 == 2);

    elew_func = tofu_elew_getfunc(TOFU_UINT8);
    elew_func(&val2_uint8, &val1_uint8, &val3_uint8, TOFU_MUL);
    ck_assert(val3_uint8 == 2);

    elew_func = tofu_elew_getfunc(TOFU_BOOL);
    elew_func(&val2_bool, &val1_bool, &val3_bool, TOFU_MUL);
    ck_assert(val3_bool == TOFU_FALSE);
}
LN_TEST_END

LN_TEST_START(test_tofu_convert)
{
    double val_d;
    float val_f;
    int32_t val_i32;
    uint32_t val_u32;
    int16_t val_i16;
    uint16_t val_u16;
    int8_t val_i8;
    uint8_t val_u8;
    tofu_bool_t val_b;

    const double val_d_max = DBL_MAX;
    const double val_d_min = -DBL_MAX;
    const double val_d_normal = 1.0;
    const float val_f_max = FLT_MAX;
    const float val_f_min = -FLT_MAX;
    const float val_f_normal = 1.0;
    const int32_t val_i32_max = INT32_MAX;
    const int32_t val_i32_min = INT32_MIN;
    const int32_t val_i32_normal = 1;
    const uint32_t val_u32_max = UINT32_MAX;
    const uint32_t val_u32_min = 0;
    const uint32_t val_u32_normal = 1;
    const int16_t val_i16_max = INT16_MAX;
    const int16_t val_i16_min = INT16_MIN;
    const int16_t val_i16_normal = 1;
    const uint16_t val_u16_max = UINT16_MAX;
    const uint16_t val_u16_min = 0;
    const uint16_t val_u16_normal = 1;
    const int8_t val_i8_max = INT8_MAX;
    const int8_t val_i8_min = INT8_MIN;
    const int8_t val_i8_normal = 1;
    const uint8_t val_u8_max = UINT8_MAX;
    const uint8_t val_u8_min = 0;
    const uint8_t val_u8_normal = 1;
    const tofu_bool_t val_b_true = TOFU_TRUE;
    const tofu_bool_t val_b_false = TOFU_FALSE;

    /* TOFU_DOUBLE */
    tofu_convert(&val_d, TOFU_DOUBLE, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_d == val_d_max);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_d == val_d_min);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_d == val_d_normal);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_f_max, TOFU_FLOAT);
    ck_assert(val_d == (double)val_f_max);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_f_min, TOFU_FLOAT);
    ck_assert(val_d == (double)val_f_min);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_d == (double)val_f_normal);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i32_max, TOFU_INT32);
    ck_assert(val_d == (double)val_i32_max);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i32_min, TOFU_INT32);
    ck_assert(val_d == (double)val_i32_min);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i32_normal, TOFU_INT32);
    ck_assert(val_d == (double)val_i32_normal);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i16_max, TOFU_INT16);
    ck_assert(val_d == (double)val_i16_max);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i16_min, TOFU_INT16);
    ck_assert(val_d == (double)val_i16_min);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i16_normal, TOFU_INT16);
    ck_assert(val_d == (double)val_i16_normal);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i8_max, TOFU_INT8);
    ck_assert(val_d == (double)val_i8_max);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i8_min, TOFU_INT8);
    ck_assert(val_d == (double)val_i8_min);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_i8_normal, TOFU_INT8);
    ck_assert(val_d == (double)val_i8_normal);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_u32_max, TOFU_UINT32);
    ck_assert(val_d == (double)val_u32_max);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_d == (double)val_u32_normal);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_u16_max, TOFU_UINT16);
    ck_assert(val_d == (double)val_u16_max);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_d == (double)val_u16_normal);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_u8_max, TOFU_UINT8);
    ck_assert(val_d == (double)val_u8_max);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_d == (double)val_u8_normal);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_b_true, TOFU_BOOL);
    ck_assert(val_d == (double)val_b_true);
    tofu_convert(&val_d, TOFU_DOUBLE, &val_b_false, TOFU_BOOL);
    ck_assert(val_d == (double)val_b_false);

    /* TOFU_FLOAT */
    tofu_convert(&val_f, TOFU_FLOAT, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_f == val_f_max);
    tofu_convert(&val_f, TOFU_FLOAT, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_f == val_f_min);
    tofu_convert(&val_f, TOFU_FLOAT, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_f == (float)val_d_normal);
    tofu_convert(&val_f, TOFU_FLOAT, &val_f_max, TOFU_FLOAT);
    ck_assert(val_f == val_f_max);
    tofu_convert(&val_f, TOFU_FLOAT, &val_f_min, TOFU_FLOAT);
    ck_assert(val_f == val_f_min);
    tofu_convert(&val_f, TOFU_FLOAT, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_f == val_f_normal);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i32_max, TOFU_INT32);
    ck_assert(val_f == (float)val_i32_max);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i32_min, TOFU_INT32);
    ck_assert(val_f == (float)val_i32_min);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i32_normal, TOFU_INT32);
    ck_assert(val_f == (float)val_i32_normal);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i16_max, TOFU_INT16);
    ck_assert(val_f == (float)val_i16_max);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i16_min, TOFU_INT16);
    ck_assert(val_f == (float)val_i16_min);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i16_normal, TOFU_INT16);
    ck_assert(val_f == (float)val_i16_normal);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i8_max, TOFU_INT8);
    ck_assert(val_f == (float)val_i8_max);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i8_min, TOFU_INT8);
    ck_assert(val_f == (float)val_i8_min);
    tofu_convert(&val_f, TOFU_FLOAT, &val_i8_normal, TOFU_INT8);
    ck_assert(val_f == (float)val_i8_normal);
    tofu_convert(&val_f, TOFU_FLOAT, &val_u32_max, TOFU_UINT32);
    ck_assert(val_f == (float)val_u32_max);
    tofu_convert(&val_f, TOFU_FLOAT, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_f == (float)val_u32_normal);
    tofu_convert(&val_f, TOFU_FLOAT, &val_u16_max, TOFU_UINT16);
    ck_assert(val_f == (float)val_u16_max);
    tofu_convert(&val_f, TOFU_FLOAT, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_f == (float)val_u16_normal);
    tofu_convert(&val_f, TOFU_FLOAT, &val_u8_max, TOFU_UINT8);
    ck_assert(val_f == (float)val_u8_max);
    tofu_convert(&val_f, TOFU_FLOAT, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_f == (float)val_u8_normal);
    tofu_convert(&val_f, TOFU_FLOAT, &val_b_true, TOFU_BOOL);
    ck_assert(val_f == (float)val_b_true);
    tofu_convert(&val_f, TOFU_FLOAT, &val_b_false, TOFU_BOOL);
    ck_assert(val_f == (float)val_b_false);

    /* TOFU_INT32 */
    tofu_convert(&val_i32, TOFU_INT32, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_i32 == val_i32_max);
    tofu_convert(&val_i32, TOFU_INT32, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_i32 == val_i32_min);
    tofu_convert(&val_i32, TOFU_INT32, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_i32 == (int32_t)val_d_normal);
    tofu_convert(&val_i32, TOFU_INT32, &val_f_max, TOFU_FLOAT);
    ck_assert(val_i32 == val_i32_max);
    tofu_convert(&val_i32, TOFU_INT32, &val_f_min, TOFU_FLOAT);
    ck_assert(val_i32 == val_i32_min);
    tofu_convert(&val_i32, TOFU_INT32, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_i32 == (int32_t)val_f_normal);
    tofu_convert(&val_i32, TOFU_INT32, &val_i32_max, TOFU_INT32);
    ck_assert(val_i32 == val_i32_max);
    tofu_convert(&val_i32, TOFU_INT32, &val_i32_min, TOFU_INT32);
    ck_assert(val_i32 == val_i32_min);
    tofu_convert(&val_i32, TOFU_INT32, &val_i32_normal, TOFU_INT32);
    ck_assert(val_i32 == val_i32_normal);
    tofu_convert(&val_i32, TOFU_INT32, &val_i16_max, TOFU_INT16);
    ck_assert(val_i32 == (int32_t)val_i16_max);
    tofu_convert(&val_i32, TOFU_INT32, &val_i16_min, TOFU_INT16);
    ck_assert(val_i32 == (int32_t)val_i16_min);
    tofu_convert(&val_i32, TOFU_INT32, &val_i16_normal, TOFU_INT16);
    ck_assert(val_i32 == (int32_t)val_i16_normal);
    tofu_convert(&val_i32, TOFU_INT32, &val_i8_max, TOFU_INT8);
    ck_assert(val_i32 == (int32_t)val_i8_max);
    tofu_convert(&val_i32, TOFU_INT32, &val_i8_min, TOFU_INT8);
    ck_assert(val_i32 == (int32_t)val_i8_min);
    tofu_convert(&val_i32, TOFU_INT32, &val_i8_normal, TOFU_INT8);
    ck_assert(val_i32 == (int32_t)val_i8_normal);
    tofu_convert(&val_i32, TOFU_INT32, &val_u32_max, TOFU_UINT32);
    ck_assert(val_i32 == val_i32_max);
    tofu_convert(&val_i32, TOFU_INT32, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_i32 == (int32_t)val_u32_normal);
    tofu_convert(&val_i32, TOFU_INT32, &val_u16_max, TOFU_UINT16);
    ck_assert(val_i32 == (int32_t)val_u16_max);
    tofu_convert(&val_i32, TOFU_INT32, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_i32 == (int32_t)val_u16_normal);
    tofu_convert(&val_i32, TOFU_INT32, &val_u8_max, TOFU_UINT8);
    ck_assert(val_i32 == (int32_t)val_u8_max);
    tofu_convert(&val_i32, TOFU_INT32, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_i32 == (int32_t)val_u8_normal);
    tofu_convert(&val_i32, TOFU_INT32, &val_b_true, TOFU_BOOL);
    ck_assert(val_i32 == (int32_t)val_b_true);
    tofu_convert(&val_i32, TOFU_INT32, &val_b_false, TOFU_BOOL);
    ck_assert(val_i32 == (int32_t)val_b_false);

    /* TOFU_INT16 */
    tofu_convert(&val_i16, TOFU_INT16, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_i16 == val_i16_max);
    tofu_convert(&val_i16, TOFU_INT16, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_i16 == val_i16_min);
    tofu_convert(&val_i16, TOFU_INT16, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_i16 == (int16_t)val_d_normal);
    tofu_convert(&val_i16, TOFU_INT16, &val_f_max, TOFU_FLOAT);
    ck_assert(val_i16 == val_i16_max);
    tofu_convert(&val_i16, TOFU_INT16, &val_f_min, TOFU_FLOAT);
    ck_assert(val_i16 == val_i16_min);
    tofu_convert(&val_i16, TOFU_INT16, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_i16 == (int16_t)val_f_normal);
    tofu_convert(&val_i16, TOFU_INT16, &val_i32_max, TOFU_INT32);
    ck_assert(val_i16 == val_i16_max);
    tofu_convert(&val_i16, TOFU_INT16, &val_i32_min, TOFU_INT32);
    ck_assert(val_i16 == val_i16_min);
    tofu_convert(&val_i16, TOFU_INT16, &val_i32_normal, TOFU_INT32);
    ck_assert(val_i16 == (int16_t)val_i32_normal);
    tofu_convert(&val_i16, TOFU_INT16, &val_i16_max, TOFU_INT16);
    ck_assert(val_i16 == val_i16_max);
    tofu_convert(&val_i16, TOFU_INT16, &val_i16_min, TOFU_INT16);
    ck_assert(val_i16 == val_i16_min);
    tofu_convert(&val_i16, TOFU_INT16, &val_i16_normal, TOFU_INT16);
    ck_assert(val_i16 == val_i16_normal);
    tofu_convert(&val_i16, TOFU_INT16, &val_i8_max, TOFU_INT8);
    ck_assert(val_i16 == (int16_t)val_i8_max);
    tofu_convert(&val_i16, TOFU_INT16, &val_i8_min, TOFU_INT8);
    ck_assert(val_i16 == (int16_t)val_i8_min);
    tofu_convert(&val_i16, TOFU_INT16, &val_i8_normal, TOFU_INT8);
    ck_assert(val_i16 == (int16_t)val_i8_normal);
    tofu_convert(&val_i16, TOFU_INT16, &val_u32_max, TOFU_UINT32);
    ck_assert(val_i16 == val_i16_max);
    tofu_convert(&val_i16, TOFU_INT16, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_i16 == (int16_t)val_u32_normal);
    tofu_convert(&val_i16, TOFU_INT16, &val_u16_max, TOFU_UINT16);
    ck_assert(val_i16 == val_i16_max);
    tofu_convert(&val_i16, TOFU_INT16, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_i16 == (int16_t)val_u16_normal);
    tofu_convert(&val_i16, TOFU_INT16, &val_u8_max, TOFU_UINT8);
    ck_assert(val_i16 == (int16_t)val_u8_max);
    tofu_convert(&val_i16, TOFU_INT16, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_i16 == (int16_t)val_u8_normal);
    tofu_convert(&val_i16, TOFU_INT16, &val_b_true, TOFU_BOOL);
    ck_assert(val_i16 == (int16_t)val_b_true);
    tofu_convert(&val_i16, TOFU_INT16, &val_b_false, TOFU_BOOL);
    ck_assert(val_i16 == (int16_t)val_b_false);

    /* TOFU_INT8 */
    tofu_convert(&val_i8, TOFU_INT8, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_i8 == val_i8_max);
    tofu_convert(&val_i8, TOFU_INT8, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_i8 == val_i8_min);
    tofu_convert(&val_i8, TOFU_INT8, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_i8 == (int8_t)val_d_normal);
    tofu_convert(&val_i8, TOFU_INT8, &val_f_max, TOFU_FLOAT);
    ck_assert(val_i8 == val_i8_max);
    tofu_convert(&val_i8, TOFU_INT8, &val_f_min, TOFU_FLOAT);
    ck_assert(val_i8 == val_i8_min);
    tofu_convert(&val_i8, TOFU_INT8, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_i8 == (int8_t)val_f_normal);
    tofu_convert(&val_i8, TOFU_INT8, &val_i32_max, TOFU_INT32);
    ck_assert(val_i8 == val_i8_max);
    tofu_convert(&val_i8, TOFU_INT8, &val_i32_min, TOFU_INT32);
    ck_assert(val_i8 == val_i8_min);
    tofu_convert(&val_i8, TOFU_INT8, &val_i32_normal, TOFU_INT32);
    ck_assert(val_i8 == (int8_t)val_i32_normal);
    tofu_convert(&val_i8, TOFU_INT8, &val_i16_max, TOFU_INT16);
    ck_assert(val_i8 == val_i8_max);
    tofu_convert(&val_i8, TOFU_INT8, &val_i16_min, TOFU_INT16);
    ck_assert(val_i8 == val_i8_min);
    tofu_convert(&val_i8, TOFU_INT8, &val_i16_normal, TOFU_INT16);
    ck_assert(val_i8 == val_i8_normal);
    tofu_convert(&val_i8, TOFU_INT8, &val_i8_max, TOFU_INT8);
    ck_assert(val_i8 == val_i8_max);
    tofu_convert(&val_i8, TOFU_INT8, &val_i8_min, TOFU_INT8);
    ck_assert(val_i8 == val_i8_min);
    tofu_convert(&val_i8, TOFU_INT8, &val_i8_normal, TOFU_INT8);
    ck_assert(val_i8 == val_i8_normal);
    tofu_convert(&val_i8, TOFU_INT8, &val_u32_max, TOFU_UINT32);
    ck_assert(val_i8 == val_i8_max);
    tofu_convert(&val_i8, TOFU_INT8, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_i8 == (int8_t)val_u32_normal);
    tofu_convert(&val_i8, TOFU_INT8, &val_u16_max, TOFU_UINT16);
    ck_assert(val_i8 == val_i8_max);
    tofu_convert(&val_i8, TOFU_INT8, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_i8 == (int8_t)val_u16_normal);
    tofu_convert(&val_i8, TOFU_INT8, &val_u8_max, TOFU_UINT8);
    ck_assert(val_i8 == val_i8_max);
    tofu_convert(&val_i8, TOFU_INT8, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_i8 == (int8_t)val_u8_normal);
    tofu_convert(&val_i8, TOFU_INT8, &val_b_true, TOFU_BOOL);
    ck_assert(val_i8 == (int8_t)val_b_true);
    tofu_convert(&val_i8, TOFU_INT8, &val_b_false, TOFU_BOOL);
    ck_assert(val_i8 == (int8_t)val_b_false);

    /* TOFU_UINT32 */
    tofu_convert(&val_u32, TOFU_UINT32, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_u32 == val_u32_max);
    tofu_convert(&val_u32, TOFU_UINT32, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_u32 == val_u32_min);
    tofu_convert(&val_u32, TOFU_UINT32, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_u32 == (uint32_t)val_d_normal);
    tofu_convert(&val_u32, TOFU_UINT32, &val_f_max, TOFU_FLOAT);
    ck_assert(val_u32 == val_u32_max);
    tofu_convert(&val_u32, TOFU_UINT32, &val_f_min, TOFU_FLOAT);
    ck_assert(val_u32 == val_u32_min);
    tofu_convert(&val_u32, TOFU_UINT32, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_u32 == (uint32_t)val_f_normal);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i32_max, TOFU_INT32);
    ck_assert(val_u32 == (uint32_t)val_i32_max);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i32_min, TOFU_INT32);
    ck_assert(val_u32 == val_u32_min);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i32_normal, TOFU_INT32);
    ck_assert(val_u32 == (uint32_t)val_i32_normal);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i16_max, TOFU_INT16);
    ck_assert(val_u32 == (uint32_t)val_i16_max);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i16_min, TOFU_INT16);
    ck_assert(val_u32 == val_u32_min);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i16_normal, TOFU_INT16);
    ck_assert(val_u32 == (uint32_t)val_i16_normal);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i8_max, TOFU_INT8);
    ck_assert(val_u32 == (uint32_t)val_i8_max);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i8_min, TOFU_INT8);
    ck_assert(val_u32 == val_u32_min);
    tofu_convert(&val_u32, TOFU_UINT32, &val_i8_normal, TOFU_INT8);
    ck_assert(val_u32 == (uint32_t)val_i8_normal);
    tofu_convert(&val_u32, TOFU_UINT32, &val_u32_max, TOFU_UINT32);
    ck_assert(val_u32 == val_u32_max);
    tofu_convert(&val_u32, TOFU_UINT32, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_u32 == val_u32_normal);
    tofu_convert(&val_u32, TOFU_UINT32, &val_u16_max, TOFU_UINT16);
    ck_assert(val_u32 == (uint32_t)val_u16_max);
    tofu_convert(&val_u32, TOFU_UINT32, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_u32 == (uint32_t)val_u16_normal);
    tofu_convert(&val_u32, TOFU_UINT32, &val_u8_max, TOFU_UINT8);
    ck_assert(val_u32 == (uint32_t)val_u8_max);
    tofu_convert(&val_u32, TOFU_UINT32, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_u32 == (uint32_t)val_u8_normal);
    tofu_convert(&val_u32, TOFU_UINT32, &val_b_true, TOFU_BOOL);
    ck_assert(val_u32 == (uint32_t)val_b_true);
    tofu_convert(&val_u32, TOFU_UINT32, &val_b_false, TOFU_BOOL);
    ck_assert(val_u32 == (uint32_t)val_b_false);

    /* TOFU_UINT16 */
    tofu_convert(&val_u16, TOFU_UINT16, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_u16 == val_u16_max);
    tofu_convert(&val_u16, TOFU_UINT16, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_u16 == val_u16_min);
    tofu_convert(&val_u16, TOFU_UINT16, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_u16 == (uint16_t)val_d_normal);
    tofu_convert(&val_u16, TOFU_UINT16, &val_f_max, TOFU_FLOAT);
    ck_assert(val_u16 == val_u16_max);
    tofu_convert(&val_u16, TOFU_UINT16, &val_f_min, TOFU_FLOAT);
    ck_assert(val_u16 == val_u16_min);
    tofu_convert(&val_u16, TOFU_UINT16, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_u16 == (uint16_t)val_f_normal);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i32_max, TOFU_INT32);
    ck_assert(val_u16 == val_u16_max);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i32_min, TOFU_INT32);
    ck_assert(val_u16 == val_u16_min);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i32_normal, TOFU_INT32);
    ck_assert(val_u16 == (uint16_t)val_i32_normal);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i16_max, TOFU_INT16);
    ck_assert(val_u16 == (uint16_t)val_i16_max);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i16_min, TOFU_INT16);
    ck_assert(val_u16 == val_u16_min);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i16_normal, TOFU_INT16);
    ck_assert(val_u16 == (uint16_t)val_i16_normal);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i8_max, TOFU_INT8);
    ck_assert(val_u16 == (uint16_t)val_i8_max);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i8_min, TOFU_INT8);
    ck_assert(val_u16 == val_u16_min);
    tofu_convert(&val_u16, TOFU_UINT16, &val_i8_normal, TOFU_INT8);
    ck_assert(val_u16 == (uint16_t)val_i8_normal);
    tofu_convert(&val_u16, TOFU_UINT16, &val_u32_max, TOFU_UINT32);
    ck_assert(val_u16 == val_u16_max);
    tofu_convert(&val_u16, TOFU_UINT16, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_u16 == (uint16_t)val_u32_normal);
    tofu_convert(&val_u16, TOFU_UINT16, &val_u16_max, TOFU_UINT16);
    ck_assert(val_u16 == val_u16_max);
    tofu_convert(&val_u16, TOFU_UINT16, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_u16 == val_u16_normal);
    tofu_convert(&val_u16, TOFU_UINT16, &val_u8_max, TOFU_UINT8);
    ck_assert(val_u16 == (uint16_t)val_u8_max);
    tofu_convert(&val_u16, TOFU_UINT16, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_u16 == (uint16_t)val_u8_normal);
    tofu_convert(&val_u16, TOFU_UINT16, &val_b_true, TOFU_BOOL);
    ck_assert(val_u16 == (uint16_t)val_b_true);
    tofu_convert(&val_u16, TOFU_UINT16, &val_b_false, TOFU_BOOL);
    ck_assert(val_u16 == (uint16_t)val_b_false);

    /* TOFU_UINT8 */
    tofu_convert(&val_u8, TOFU_UINT8, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_u8 == val_u8_max);
    tofu_convert(&val_u8, TOFU_UINT8, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_u8 == val_u8_min);
    tofu_convert(&val_u8, TOFU_UINT8, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_u8 == (uint8_t)val_d_normal);
    tofu_convert(&val_u8, TOFU_UINT8, &val_f_max, TOFU_FLOAT);
    ck_assert(val_u8 == val_u8_max);
    tofu_convert(&val_u8, TOFU_UINT8, &val_f_min, TOFU_FLOAT);
    ck_assert(val_u8 == val_u8_min);
    tofu_convert(&val_u8, TOFU_UINT8, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_u8 == (uint8_t)val_f_normal);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i32_max, TOFU_INT32);
    ck_assert(val_u8 == val_u8_max);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i32_min, TOFU_INT32);
    ck_assert(val_u8 == val_u8_min);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i32_normal, TOFU_INT32);
    ck_assert(val_u8 == (uint8_t)val_i32_normal);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i16_max, TOFU_INT16);
    ck_assert(val_u8 == val_u8_max);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i16_min, TOFU_INT16);
    ck_assert(val_u8 == val_u8_min);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i16_normal, TOFU_INT16);
    ck_assert(val_u8 == (uint8_t)val_i16_normal);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i8_max, TOFU_INT8);
    ck_assert(val_u8 == (uint8_t)val_i8_max);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i8_min, TOFU_INT8);
    ck_assert(val_u8 == val_u8_min);
    tofu_convert(&val_u8, TOFU_UINT8, &val_i8_normal, TOFU_INT8);
    ck_assert(val_u8 == (uint8_t)val_i8_normal);
    tofu_convert(&val_u8, TOFU_UINT8, &val_u32_max, TOFU_UINT32);
    ck_assert(val_u8 == val_u8_max);
    tofu_convert(&val_u8, TOFU_UINT8, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_u8 == (uint8_t)val_u32_normal);
    tofu_convert(&val_u8, TOFU_UINT8, &val_u16_max, TOFU_UINT16);
    ck_assert(val_u8 == val_u8_max);
    tofu_convert(&val_u8, TOFU_UINT8, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_u8 == (uint8_t)val_u16_normal);
    tofu_convert(&val_u8, TOFU_UINT8, &val_u8_max, TOFU_UINT8);
    ck_assert(val_u8 == val_u8_max);
    tofu_convert(&val_u8, TOFU_UINT8, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_u8 == val_u8_normal);
    tofu_convert(&val_u8, TOFU_UINT8, &val_b_true, TOFU_BOOL);
    ck_assert(val_u8 == (uint8_t)val_b_true);
    tofu_convert(&val_u8, TOFU_UINT8, &val_b_false, TOFU_BOOL);
    ck_assert(val_u8 == (uint8_t)val_b_false);

    /* TOFU_BOOL */
    tofu_convert(&val_b, TOFU_BOOL, &val_d_max, TOFU_DOUBLE);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_d_min, TOFU_DOUBLE);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_d_normal, TOFU_DOUBLE);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_f_max, TOFU_FLOAT);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_f_min, TOFU_FLOAT);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_f_normal, TOFU_FLOAT);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i32_max, TOFU_INT32);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i32_min, TOFU_INT32);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i32_normal, TOFU_INT32);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i16_max, TOFU_INT16);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i16_min, TOFU_INT16);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i16_normal, TOFU_INT16);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i8_max, TOFU_INT8);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i8_min, TOFU_INT8);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_i8_normal, TOFU_INT8);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_u32_max, TOFU_UINT32);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_u32_normal, TOFU_UINT32);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_u16_max, TOFU_UINT16);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_u16_normal, TOFU_UINT16);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_u8_max, TOFU_UINT8);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_u8_normal, TOFU_UINT8);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_b_true, TOFU_BOOL);
    ck_assert(val_b == val_b_true);
    tofu_convert(&val_b, TOFU_BOOL, &val_b_false, TOFU_BOOL);
    ck_assert(val_b == val_b_false);
}
LN_TEST_END

LN_TEST_START(test_tofu_sort_dir_name)
{
    ck_assert_str_eq(tofu_sort_dir_name(TOFU_SORT_DIR_ASCENDING),
                     "TOFU_SORT_DIR_ASCENDING");
    ck_assert_str_eq(tofu_sort_dir_name(TOFU_SORT_DIR_DESCENDING),
                     "TOFU_SORT_DIR_DESCENDING");
}
LN_TEST_END

LN_TEST_START(test_tofu_sort_dir_from_str)
{
    ck_assert_int_eq(tofu_sort_dir_from_str("TOFU_SORT_DIR_ASCENDING"),
                     TOFU_SORT_DIR_ASCENDING);
    ck_assert_int_eq(tofu_sort_dir_from_str("TOFU_SORT_DIR_DESCENDING"),
                     TOFU_SORT_DIR_DESCENDING);
    ck_assert_int_eq(tofu_sort_dir_from_str("sdf"), -1);
}
LN_TEST_END
/* end of tests */

LN_TEST_TCASE_START(type, checked_setup, checked_teardown)
{
    LN_TEST_ADD_TEST(test_tofu_size_of);
    LN_TEST_ADD_TEST(test_tofu_psub);
    LN_TEST_ADD_TEST(test_tofu_padd);
    LN_TEST_ADD_TEST(test_tofu_passign);
    LN_TEST_ADD_TEST(test_tofu_dtype_fmt);
    LN_TEST_ADD_TEST(test_tofu_dtype_max);
    LN_TEST_ADD_TEST(test_tofu_dtype_min);
    LN_TEST_ADD_TEST(test_tofu_dtype_max_double);
    LN_TEST_ADD_TEST(test_tofu_dtype_min_double);
    LN_TEST_ADD_TEST(test_tofu_pointer_sub);
    LN_TEST_ADD_TEST(test_tofu_pointer_add);
    LN_TEST_ADD_TEST(test_tofu_pointer_assign);
    LN_TEST_ADD_TEST(test_tofu_fprintf);
    LN_TEST_ADD_TEST(test_tofu_fprintf_getfunc);
    LN_TEST_ADD_TEST(test_tofu_cmp);
    LN_TEST_ADD_TEST(test_tofu_cmp_getfunc);
    LN_TEST_ADD_TEST(test_tofu_elew);
    LN_TEST_ADD_TEST(test_tofu_elew_getfunc);
    LN_TEST_ADD_TEST(test_tofu_convert);
    LN_TEST_ADD_TEST(test_tofu_sort_dir_name);
    LN_TEST_ADD_TEST(test_tofu_sort_dir_from_str);
}
LN_TEST_TCASE_END

LN_TEST_ADD_TCASE(type);
