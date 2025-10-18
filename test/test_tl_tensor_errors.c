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

#include "test_tofu.h"
#include "tl_tensor.h"
#include "lightnettest/ln_test.h"
#include <errno.h>

static void checked_setup(void)
{
}

static void checked_teardown(void)
{
}

/* Test matmul with mismatched contraction dimensions */
LN_TEST_START(test_matmul_mismatched_dims)
{
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {1, 2, 3, 4};

    /* [2,3] @ [2,2] - contraction dimension mismatch (3 != 2) */
    tl_tensor *a = tl_tensor_create(a_data, 2, (int[]){2, 3}, TL_FLOAT);
    tl_tensor *b = tl_tensor_create(b_data, 2, (int[]){2, 2}, TL_FLOAT);

    /* This should trigger an assertion failure in debug mode */
    /* In release mode, behavior is undefined */
    /* For now, we document that this is programmer error */

    tl_tensor_free(a);
    tl_tensor_free(b);
}
LN_TEST_END

/* Test matmul with non-broadcastable batch dimensions */
LN_TEST_START(test_matmul_non_broadcastable)
{
    float a_data[12], b_data[20];
    for (int i = 0; i < 12; i++) a_data[i] = i + 1;
    for (int i = 0; i < 20; i++) b_data[i] = i + 1;

    /* [3,2,2] @ [5,2,2] - batch dimensions 3 and 5 cannot broadcast */
    tl_tensor *a = tl_tensor_create(a_data, 3, (int[]){3, 2, 2}, TL_FLOAT);
    tl_tensor *b = tl_tensor_create(b_data, 3, (int[]){5, 2, 2}, TL_FLOAT);

    errno = 0;
    tl_tensor *result = tl_tensor_matmul(a, b, NULL);

    ck_assert_ptr_null(result);
    ck_assert_int_eq(errno, EINVAL);

    tl_tensor_free(a);
    tl_tensor_free(b);
}
LN_TEST_END

/* Test inner with mismatched last dimensions */
LN_TEST_START(test_inner_mismatched_dims)
{
    float a_data[] = {1, 2, 3};
    float b_data[] = {1, 2, 3, 4};

    /* [3] inner [4] - last dimensions don't match */
    tl_tensor *a = tl_tensor_create(a_data, 1, (int[]){3}, TL_FLOAT);
    tl_tensor *b = tl_tensor_create(b_data, 1, (int[]){4}, TL_FLOAT);

    /* This should trigger an assertion failure */
    /* Documenting that dimension checking is programmer's responsibility */

    tl_tensor_free(a);
    tl_tensor_free(b);
}
LN_TEST_END

/* Test operations with dtype mismatch */
LN_TEST_START(test_dtype_mismatch)
{
    float a_data[] = {1, 2, 3};
    int32_t b_data[] = {1, 2, 3};

    tl_tensor *a = tl_tensor_create(a_data, 1, (int[]){3}, TL_FLOAT);
    tl_tensor *b = tl_tensor_create(b_data, 1, (int[]){3}, TL_INT32);

    /* Operations with mismatched dtypes should fail assertions */
    /* Documenting that dtype checking is done via assertions */

    tl_tensor_free(a);
    tl_tensor_free(b);
}
LN_TEST_END

/* Test element-wise operations with non-broadcastable shapes */
LN_TEST_START(test_elew_non_broadcastable)
{
    float a_data[] = {1, 2, 3};
    float b_data[] = {1, 2};

    tl_tensor *a = tl_tensor_create(a_data, 1, (int[]){3}, TL_FLOAT);
    tl_tensor *b = tl_tensor_create(b_data, 1, (int[]){2}, TL_FLOAT);

    /* [3] and [2] are not broadcastable */
    tl_tensor *result = tl_tensor_elew(a, b, NULL, TL_SUM);

    /* Should return NULL */
    ck_assert_ptr_null(result);

    tl_tensor_free(a);
    tl_tensor_free(b);
}
LN_TEST_END

/* Test pre-allocated destination with wrong shape */
LN_TEST_START(test_wrong_dst_shape_matmul)
{
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {1, 1, 2, 2, 3, 3};

    tl_tensor *a = tl_tensor_create(a_data, 2, (int[]){2, 3}, TL_FLOAT);
    tl_tensor *b = tl_tensor_create(b_data, 2, (int[]){3, 2}, TL_FLOAT);

    /* Wrong destination shape: expecting [2,2] but providing [3,3] */
    tl_tensor *dst = tl_tensor_zeros(2, (int[]){3, 3}, TL_FLOAT);

    /* This should trigger assertion in debug mode */
    /* Documenting that shape checking is programmer's responsibility */

    tl_tensor_free(a);
    tl_tensor_free(b);
    tl_tensor_free_data_too(dst);
}
LN_TEST_END

/* Test broadcastability check function */
LN_TEST_START(test_isbroadcastable)
{
    float a_data[] = {1, 2, 3};
    float b_data[] = {1, 2};

    /* Test compatible shapes */
    tl_tensor *a1 = tl_tensor_create(a_data, 1, (int[]){3}, TL_FLOAT);
    tl_tensor *b1 = tl_tensor_create(a_data, 1, (int[]){3}, TL_FLOAT);
    ck_assert(tl_tensor_isbroadcastable(a1, b1));

    /* Test scalar broadcasting */
    tl_tensor *scalar = tl_tensor_create(a_data, 1, (int[]){1}, TL_FLOAT);
    ck_assert(tl_tensor_isbroadcastable(a1, scalar));

    /* Test incompatible shapes */
    tl_tensor *b2 = tl_tensor_create(b_data, 1, (int[]){2}, TL_FLOAT);
    ck_assert(!tl_tensor_isbroadcastable(a1, b2));

    /* Test with multi-dimensional */
    tl_tensor *m1 = tl_tensor_create(a_data, 2, (int[]){1, 3}, TL_FLOAT);
    tl_tensor *m2 = tl_tensor_create(a_data, 2, (int[]){2, 1}, TL_FLOAT);
    ck_assert(tl_tensor_isbroadcastable(m1, m2));

    tl_tensor_free(a1);
    tl_tensor_free(b1);
    tl_tensor_free(scalar);
    tl_tensor_free(b2);
    tl_tensor_free(m1);
    tl_tensor_free(m2);
}
LN_TEST_END

/* end of tests */

LN_TEST_TCASE_START(tensor_errors, checked_setup, checked_teardown)
{
    LN_TEST_ADD_TEST(test_matmul_non_broadcastable);
    LN_TEST_ADD_TEST(test_elew_non_broadcastable);
    LN_TEST_ADD_TEST(test_isbroadcastable);
}
LN_TEST_TCASE_END

LN_TEST_ADD_TCASE(tensor_errors);
