/*
 * Copyright (c) 2018-2020 Zhixu Zhao
 *
 * Comprehensive unit tests for broadcasting mechanism internals.
 *
 * These tests validate the individual components of the broadcasting system:
 * - Shape calculation
 * - Stride calculation
 * - Index mapping
 * - Mathematical properties
 *
 * Ground truth established via test_broadcast_ground_truth.py
 */

#include "test_tofu.h"
#include "lightnettest/ln_test.h"
#include "../src/tl_tensor_broadcast_internal.h"
#include "tl_tensor.h"
#include "tl_util.h"

#define ARR(type, varg...) (type[]){varg}

static void checked_setup(void)
{
}

static void checked_teardown(void)
{
}

/*
 * TEST 1: Broadcasting Shape Calculation
 *
 * Validates that output shapes are correctly computed from input shapes
 * following NumPy's broadcasting rules.
 */
LN_TEST_START(test_broadcast_shape_calculation)
{
    tl_tensor *t1, *t2;
    int out_ndim;
    int out_dims[TL_MAXDIM];

    // Test case 1: Same shape -> same shape
    t1 = tl_tensor_zeros(2, ARR(int, 3, 3), TL_FLOAT);
    t2 = tl_tensor_zeros(2, ARR(int, 3, 3), TL_FLOAT);

    tl_compute_broadcast_dims(t1, t2, &out_ndim, out_dims);

    ck_assert_int_eq(out_ndim, 2);
    ck_assert_int_eq(out_dims[0], 3);
    ck_assert_int_eq(out_dims[1], 3);

    tl_tensor_free_data_too(t1);
    tl_tensor_free_data_too(t2);

    // Test case 2: Scalar (1,) broadcast to (2, 3) -> (2, 3)
    t1 = tl_tensor_zeros(1, ARR(int, 1), TL_FLOAT);
    t2 = tl_tensor_zeros(2, ARR(int, 2, 3), TL_FLOAT);

    tl_compute_broadcast_dims(t1, t2, &out_ndim, out_dims);

    ck_assert_int_eq(out_ndim, 2);
    ck_assert_int_eq(out_dims[0], 2);
    ck_assert_int_eq(out_dims[1], 3);

    tl_tensor_free_data_too(t1);
    tl_tensor_free_data_too(t2);

    // Test case 3: Row vector (3,) broadcast to (2, 3) -> (2, 3)
    t1 = tl_tensor_zeros(1, ARR(int, 3), TL_FLOAT);
    t2 = tl_tensor_zeros(2, ARR(int, 2, 3), TL_FLOAT);

    tl_compute_broadcast_dims(t1, t2, &out_ndim, out_dims);

    ck_assert_int_eq(out_ndim, 2);
    ck_assert_int_eq(out_dims[0], 2);
    ck_assert_int_eq(out_dims[1], 3);

    tl_tensor_free_data_too(t1);
    tl_tensor_free_data_too(t2);

    // Test case 4: Column (2, 1) broadcast to (2, 3) -> (2, 3)
    t1 = tl_tensor_zeros(2, ARR(int, 2, 1), TL_FLOAT);
    t2 = tl_tensor_zeros(2, ARR(int, 2, 3), TL_FLOAT);

    tl_compute_broadcast_dims(t1, t2, &out_ndim, out_dims);

    ck_assert_int_eq(out_ndim, 2);
    ck_assert_int_eq(out_dims[0], 2);
    ck_assert_int_eq(out_dims[1], 3);

    tl_tensor_free_data_too(t1);
    tl_tensor_free_data_too(t2);

    // Test case 5: 3D broadcast (2, 1, 4) with (1, 3, 4) -> (2, 3, 4)
    t1 = tl_tensor_zeros(3, ARR(int, 2, 1, 4), TL_FLOAT);
    t2 = tl_tensor_zeros(3, ARR(int, 1, 3, 4), TL_FLOAT);

    tl_compute_broadcast_dims(t1, t2, &out_ndim, out_dims);

    ck_assert_int_eq(out_ndim, 3);
    ck_assert_int_eq(out_dims[0], 2);
    ck_assert_int_eq(out_dims[1], 3);
    ck_assert_int_eq(out_dims[2], 4);

    tl_tensor_free_data_too(t1);
    tl_tensor_free_data_too(t2);
}
LN_TEST_END

/*
 * TEST 2: Broadcasting Stride Calculation
 *
 * Validates that broadcasting strides are correctly computed.
 * Strides determine how output coordinates map to input indices.
 */
LN_TEST_START(test_broadcast_stride_calculation)
{
    tl_tensor *t;
    int out_dims[TL_MAXDIM];
    int strides[TL_MAXDIM];

    // Test case 1: Same shape - no broadcasting, regular strides
    t = tl_tensor_zeros(2, ARR(int, 3, 3), TL_FLOAT);
    out_dims[0] = 3;
    out_dims[1] = 3;

    tl_compute_broadcast_strides(t, 2, out_dims, strides);

    ck_assert_int_eq(strides[0], 3);  // Row stride
    ck_assert_int_eq(strides[1], 1);  // Column stride

    tl_tensor_free_data_too(t);

    // Test case 2: Scalar (1,) to (3, 3) - all strides are 0
    t = tl_tensor_zeros(1, ARR(int, 1), TL_FLOAT);
    out_dims[0] = 3;
    out_dims[1] = 3;

    tl_compute_broadcast_strides(t, 2, out_dims, strides);

    ck_assert_int_eq(strides[0], 0);  // Broadcast in first dim
    ck_assert_int_eq(strides[1], 0);  // Broadcast in second dim

    tl_tensor_free_data_too(t);

    // Test case 3: Row vector (3,) to (2, 3) - first stride 0, second stride 1
    t = tl_tensor_zeros(1, ARR(int, 3), TL_FLOAT);
    out_dims[0] = 2;
    out_dims[1] = 3;

    tl_compute_broadcast_strides(t, 2, out_dims, strides);

    ck_assert_int_eq(strides[0], 0);  // Broadcast across rows
    ck_assert_int_eq(strides[1], 1);  // Normal column stride

    tl_tensor_free_data_too(t);

    // Test case 4: Column (3, 1) to (3, 3) - first stride 1, second stride 0
    t = tl_tensor_zeros(2, ARR(int, 3, 1), TL_FLOAT);
    out_dims[0] = 3;
    out_dims[1] = 3;

    tl_compute_broadcast_strides(t, 2, out_dims, strides);

    ck_assert_int_eq(strides[0], 1);  // Normal row stride
    ck_assert_int_eq(strides[1], 0);  // Broadcast across columns

    tl_tensor_free_data_too(t);

    // Test case 5: Row (1, 3) to (2, 3) - first stride 0, second stride 1
    t = tl_tensor_zeros(2, ARR(int, 1, 3), TL_FLOAT);
    out_dims[0] = 2;
    out_dims[1] = 3;

    tl_compute_broadcast_strides(t, 2, out_dims, strides);

    ck_assert_int_eq(strides[0], 0);  // Broadcast in first dim
    ck_assert_int_eq(strides[1], 1);  // Normal stride in second dim

    tl_tensor_free_data_too(t);

    // Test case 6: 3D with middle broadcast (2, 1, 4) to (2, 3, 4)
    t = tl_tensor_zeros(3, ARR(int, 2, 1, 4), TL_FLOAT);
    out_dims[0] = 2;
    out_dims[1] = 3;
    out_dims[2] = 4;

    tl_compute_broadcast_strides(t, 3, out_dims, strides);

    ck_assert_int_eq(strides[0], 4);  // Normal stride in first dim (2 * 1 * 4 / 2)
    ck_assert_int_eq(strides[1], 0);  // Broadcast in middle dim
    ck_assert_int_eq(strides[2], 1);  // Normal stride in last dim

    tl_tensor_free_data_too(t);
}
LN_TEST_END

/*
 * TEST 3: Broadcasting Index Mapping
 *
 * Validates that output coordinates correctly map to input indices
 * using the computed strides.
 */
LN_TEST_START(test_broadcast_index_mapping)
{
    tl_tensor *src;
    int out_ndim;
    int out_dims[TL_MAXDIM];
    int strides[TL_MAXDIM];

    // Test case: Column (2, 1) broadcast to (2, 3)
    // Expected: Each row value repeats across 3 columns
    int32_t src_data[] = {10, 20};
    src = tl_tensor_create(src_data, 2, ARR(int, 2, 1), TL_INT32);

    out_dims[0] = 2;
    out_dims[1] = 3;
    out_ndim = 2;

    tl_compute_broadcast_strides(src, out_ndim, out_dims, strides);

    // Strides should be [1, 0] - move 1 per row, 0 per column (broadcast)
    ck_assert_int_eq(strides[0], 1);
    ck_assert_int_eq(strides[1], 0);

    // Verify index mapping for specific output positions
    // output[0,0] -> src_idx = 0*1 + 0*0 = 0 -> value 10
    // output[0,1] -> src_idx = 0*1 + 1*0 = 0 -> value 10
    // output[0,2] -> src_idx = 0*1 + 2*0 = 0 -> value 10
    // output[1,0] -> src_idx = 1*1 + 0*0 = 1 -> value 20
    // output[1,1] -> src_idx = 1*1 + 1*0 = 1 -> value 20
    // output[1,2] -> src_idx = 1*1 + 2*0 = 1 -> value 20

    int src_idx;

    src_idx = 0 * strides[0] + 0 * strides[1];
    ck_assert_int_eq(src_idx, 0);
    ck_assert_int_eq(((int32_t *)src->data)[src_idx], 10);

    src_idx = 0 * strides[0] + 1 * strides[1];
    ck_assert_int_eq(src_idx, 0);
    ck_assert_int_eq(((int32_t *)src->data)[src_idx], 10);

    src_idx = 0 * strides[0] + 2 * strides[1];
    ck_assert_int_eq(src_idx, 0);
    ck_assert_int_eq(((int32_t *)src->data)[src_idx], 10);

    src_idx = 1 * strides[0] + 0 * strides[1];
    ck_assert_int_eq(src_idx, 1);
    ck_assert_int_eq(((int32_t *)src->data)[src_idx], 20);

    src_idx = 1 * strides[0] + 1 * strides[1];
    ck_assert_int_eq(src_idx, 1);
    ck_assert_int_eq(((int32_t *)src->data)[src_idx], 20);

    src_idx = 1 * strides[0] + 2 * strides[1];
    ck_assert_int_eq(src_idx, 1);
    ck_assert_int_eq(((int32_t *)src->data)[src_idx], 20);

    tl_tensor_free(src);
}
LN_TEST_END

/*
 * TEST 4: Broadcasting with Ground Truth Data
 *
 * Uses the test data generated by test_broadcast_ground_truth.py
 * to validate end-to-end broadcasting behavior against NumPy.
 */
LN_TEST_START(test_broadcast_with_ground_truth)
{
    tl_tensor *src1, *src2, *dst;
    int out_ndim;
    int out_dims[TL_MAXDIM];
    int strides1[TL_MAXDIM];
    int strides2[TL_MAXDIM];

    // Ground truth test case 4 from JSON:
    // shape1: (2, 1), shape2: (2, 3) -> output: (2, 3)
    // data1: [1, 2], data2: [100, 101, 102, 103, 104, 105]
    // expected: [101, 102, 103, 105, 106, 107]
    // strides1: [1, 0], strides2: [3, 1]

    int32_t data1[] = {1, 2};
    int32_t data2[] = {100, 101, 102, 103, 104, 105};
    int32_t expected[] = {101, 102, 103, 105, 106, 107};

    src1 = tl_tensor_create(data1, 2, ARR(int, 2, 1), TL_INT32);
    src2 = tl_tensor_create(data2, 2, ARR(int, 2, 3), TL_INT32);

    // Compute output shape
    tl_compute_broadcast_dims(src1, src2, &out_ndim, out_dims);

    ck_assert_int_eq(out_ndim, 2);
    ck_assert_int_eq(out_dims[0], 2);
    ck_assert_int_eq(out_dims[1], 3);

    // Compute strides
    tl_compute_broadcast_strides(src1, out_ndim, out_dims, strides1);
    tl_compute_broadcast_strides(src2, out_ndim, out_dims, strides2);

    ck_assert_int_eq(strides1[0], 1);
    ck_assert_int_eq(strides1[1], 0);
    ck_assert_int_eq(strides2[0], 3);
    ck_assert_int_eq(strides2[1], 1);

    // Perform the actual broadcast operation
    dst = tl_tensor_elew_broadcast(src1, src2, NULL, TL_SUM);

    ck_assert_int_eq(dst->ndim, 2);
    ck_assert_int_eq(dst->dims[0], 2);
    ck_assert_int_eq(dst->dims[1], 3);
    // Manually verify array elements since ck_assert_array_int_eq might not be available
    for (int i = 0; i < dst->len; i++) {
        ck_assert_int_eq(((int32_t *)dst->data)[i], expected[i]);
    }

    tl_tensor_free(src1);
    tl_tensor_free(src2);
    tl_tensor_free_data_too(dst);
}
LN_TEST_END

/*
 * TEST 5: Broadcasting Properties
 *
 * Tests mathematical properties that broadcasting should satisfy.
 */
LN_TEST_START(test_broadcast_properties)
{
    tl_tensor *t1, *t2;
    int out_ndim1, out_ndim2;
    int out_dims1[TL_MAXDIM];
    int out_dims2[TL_MAXDIM];

    // Property 1: Commutativity of shape calculation
    // broadcast_shape(A, B) == broadcast_shape(B, A)
    t1 = tl_tensor_zeros(2, ARR(int, 3, 1), TL_FLOAT);
    t2 = tl_tensor_zeros(2, ARR(int, 1, 3), TL_FLOAT);

    tl_compute_broadcast_dims(t1, t2, &out_ndim1, out_dims1);
    tl_compute_broadcast_dims(t2, t1, &out_ndim2, out_dims2);

    ck_assert_int_eq(out_ndim1, out_ndim2);
    ck_assert_int_eq(out_dims1[0], out_dims2[0]);
    ck_assert_int_eq(out_dims1[1], out_dims2[1]);

    tl_tensor_free_data_too(t1);
    tl_tensor_free_data_too(t2);

    // Property 2: Identity - broadcast_shape(A, A) == A.shape
    t1 = tl_tensor_zeros(2, ARR(int, 3, 3), TL_FLOAT);
    t2 = tl_tensor_zeros(2, ARR(int, 3, 3), TL_FLOAT);

    tl_compute_broadcast_dims(t1, t2, &out_ndim1, out_dims1);

    ck_assert_int_eq(out_ndim1, 2);
    ck_assert_int_eq(out_dims1[0], 3);
    ck_assert_int_eq(out_dims1[1], 3);

    tl_tensor_free_data_too(t1);
    tl_tensor_free_data_too(t2);

    // Property 3: Data preservation - broadcasting doesn't change values
    int32_t src_data[] = {1, 2, 3};
    t1 = tl_tensor_create(src_data, 1, ARR(int, 3), TL_INT32);

    tl_tensor *broadcasted = tl_tensor_broadcast_to(t1, NULL, 2, ARR(int, 2, 3));

    // Check that each row contains the original data
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
            int idx = row * 3 + col;
            ck_assert_int_eq(((int32_t *)broadcasted->data)[idx], src_data[col]);
        }
    }

    tl_tensor_free(t1);
    tl_tensor_free_data_too(broadcasted);
}
LN_TEST_END

/* end of tests */

LN_TEST_TCASE_START(broadcast_mechanism, checked_setup, checked_teardown)
{
    LN_TEST_ADD_TEST(test_broadcast_shape_calculation);
    LN_TEST_ADD_TEST(test_broadcast_stride_calculation);
    LN_TEST_ADD_TEST(test_broadcast_index_mapping);
    LN_TEST_ADD_TEST(test_broadcast_with_ground_truth);
    LN_TEST_ADD_TEST(test_broadcast_properties);
}
LN_TEST_TCASE_END

LN_TEST_ADD_TCASE(broadcast_mechanism);
