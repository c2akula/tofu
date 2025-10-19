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

#include "tofu_tensor_internal.h"

/**
 * Implements NumPy-compatible inner product
 * 
 * For 1-D arrays: Standard inner product of vectors (sum of element-wise products)
 * For N-D arrays: Sum product over the last axes
 * 
 * The last dimension of both arrays must match in length.
 * Other dimensions are subject to broadcasting rules.
 */
TOFU_EXPORT tofu_tensor *tofu_tensor_inner(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst)
{
    assert(src1->data && src2->data);
    assert(src1->dtype == src2->dtype);
    
    /* Validate the last dimensions match */
    int last_dim1 = src1->ndim > 0 ? src1->dims[src1->ndim - 1] : 1;
    int last_dim2 = src2->ndim > 0 ? src2->dims[src2->ndim - 1] : 1;
    
    if (last_dim1 != last_dim2) {
        tofu_warn_ret("Last dimensions must match for inner product: %d != %d", last_dim1, last_dim2);
        return NULL;
    }
    
    /* For 1D arrays, compute the standard inner product */
    if (src1->ndim <= 1 && src2->ndim <= 1) {
        // Simple vector inner product - element-wise multiply and sum
        if (dst) {
            assert(dst->data);
            assert(dst->ndim == 1);
            assert(dst->dims[0] == 1);
            assert(src1->dtype == dst->dtype);
        } else {
            dst = tofu_tensor_zeros(1, (int[]){ 1 }, src1->dtype);
        }
        
        size_t dsize = tofu_size_of(src1->dtype);
        void *s1_data = src1->data;
        void *s2_data = src2->data;
        void *d_data = dst->data;
        tofu_elew_func elew = tofu_elew_getfunc(src1->dtype);
        char elew_prod[TOFU_DTYPE_MAX_SIZE];
        
        memset(dst->data, 0, tofu_tensor_size(dst));
        for (int i = 0; i < last_dim1; i++) {
            elew(tofu_padd(s1_data, i, dsize), tofu_padd(s2_data, i, dsize), elew_prod, TOFU_MUL);
            elew(elew_prod, d_data, d_data, TOFU_SUM);
        }
        
        return dst;
    }
    
    /* Compute the output shape: concatenate shapes without last dimensions
     * For NumPy compatibility: output.shape = src1.shape[:-1] + src2.shape[:-1]
     * Examples:
     *   [2,3] x [2,3] -> [2] + [2] -> [2,2]
     *   [4,5,3] x [6,7,3] -> [4,5] + [6,7] -> [4,5,6,7]
     *   [3] x [3] -> [] + [] -> scalar (handled above in 1-D case)
     */

    int src1_trunc_ndim = src1->ndim - 1;
    int src2_trunc_ndim = src2->ndim - 1;
    int out_ndim = src1_trunc_ndim + src2_trunc_ndim;
    int out_dims[TOFU_MAXDIM];

    // Copy src1's dimensions (except last)
    for (int i = 0; i < src1_trunc_ndim; i++) {
        out_dims[i] = src1->dims[i];
    }

    // Append src2's dimensions (except last)
    for (int i = 0; i < src2_trunc_ndim; i++) {
        out_dims[src1_trunc_ndim + i] = src2->dims[i];
    }
    
    // Allocate/validate destination tensor
    if (dst) {
        assert(dst->data);
        assert(dst->ndim == out_ndim);
        for (int i = 0; i < out_ndim; i++) {
            assert(dst->dims[i] == out_dims[i]);
        }
        assert(src1->dtype == dst->dtype);
    } else {
        dst = tofu_tensor_zeros(out_ndim, out_dims, src1->dtype);
    }
    
    // Calculate strides for cartesian product (not broadcasting)
    // First src1_trunc_ndim dimensions of output correspond to src1
    // Last src2_trunc_ndim dimensions of output correspond to src2
    int *src1_strides = (int *)tofu_alloc(sizeof(int) * src1_trunc_ndim);
    int *src2_strides = (int *)tofu_alloc(sizeof(int) * src2_trunc_ndim);

    // Calculate strides for src1 (standard row-major order)
    if (src1_trunc_ndim > 0) {
        src1_strides[src1_trunc_ndim - 1] = 1;
        for (int i = src1_trunc_ndim - 2; i >= 0; i--) {
            src1_strides[i] = src1_strides[i + 1] * src1->dims[i + 1];
        }
    }

    // Calculate strides for src2 (standard row-major order)
    if (src2_trunc_ndim > 0) {
        src2_strides[src2_trunc_ndim - 1] = 1;
        for (int i = src2_trunc_ndim - 2; i >= 0; i--) {
            src2_strides[i] = src2_strides[i + 1] * src2->dims[i + 1];
        }
    }
    
    // Implement inner product with cartesian product semantics
    size_t dsize = tofu_size_of(src1->dtype);
    int *dst_coords = (int *)tofu_alloc(sizeof(int) * out_ndim);
    tofu_elew_func elew = tofu_elew_getfunc(src1->dtype);
    char elew_prod[TOFU_DTYPE_MAX_SIZE];
    char inner_sum[TOFU_DTYPE_MAX_SIZE];
    memset(dst->data, 0, tofu_tensor_size(dst));

    // For each position in the output
    for (int di = 0; di < dst->len; di++) {
        // Get coordinates in the output tensor
        tofu_tensor_coords(dst, di, dst_coords);

        // Split coordinates: first part for src1, second part for src2
        // Output coords [i0, i1, ..., j0, j1, ...] where:
        //   i's index into src1 (first src1_trunc_ndim dims)
        //   j's index into src2 (last src2_trunc_ndim dims)

        int base_idx1 = 0;
        for (int j = 0; j < src1_trunc_ndim; j++) {
            base_idx1 += dst_coords[j] * src1_strides[j];
        }

        int base_idx2 = 0;
        for (int j = 0; j < src2_trunc_ndim; j++) {
            base_idx2 += dst_coords[src1_trunc_ndim + j] * src2_strides[j];
        }

        // Initialize sum for the inner product
        memset(inner_sum, 0, dsize);

        // For each element along the last dimension, compute element-wise product and sum
        for (int k = 0; k < last_dim1; k++) {
            // Calculate the full indices into the input tensors
            int full_idx1 = base_idx1 * last_dim1 + k;
            int full_idx2 = base_idx2 * last_dim2 + k;

            // Compute the element-wise product
            elew(tofu_padd(src1->data, full_idx1, dsize),
                 tofu_padd(src2->data, full_idx2, dsize),
                 elew_prod, TOFU_MUL);

            // Add to the running sum
            elew(elew_prod, inner_sum, inner_sum, TOFU_SUM);
        }

        // Store the result in the output tensor
        tofu_passign(dst->data, di, inner_sum, 0, dsize);
    }
    
    // Free allocated resources
    tofu_free(src1_strides);
    tofu_free(src2_strides);
    tofu_free(dst_coords);
    
    return dst;
}
