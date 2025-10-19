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

/*
 * Broadcast subtraction along a specific axis.
 *
 * Subtracts a reduced tensor from a source tensor, broadcasting along the specified axis.
 * This is useful for operations like layer normalization where we need to subtract
 * mean values that have been reduced along an axis.
 *
 * Parameters:
 *   src: Source tensor (e.g., [2, 3, 4])
 *   reduced: Reduced tensor (e.g., [2, 4] if axis=1)
 *   dst: Destination tensor (can be NULL to allocate)
 *   axis: The axis along which the subtraction is broadcast
 *
 * Returns:
 *   dst tensor with result: dst[...] = src[...] - reduced[...]
 *
 * Example:
 *   src: [2, 3] = [[1, 2, 3],
 *                  [4, 5, 6]]
 *   reduced: [2] = [2, 5]  (mean along axis 1)
 *   axis: 1
 *   result: [2, 3] = [[-1, 0, 1],
 *                     [-1, 0, 1]]
 */
TOFU_EXPORT tofu_tensor *tofu_tensor_sub_broadcast(const tofu_tensor *src,
                                              const tofu_tensor *reduced,
                                              tofu_tensor *dst,
                                              int axis)
{
    assert(src && src->data);
    assert(reduced && reduced->data);
    assert(axis >= 0 && axis < src->ndim);
    assert(src->dtype == reduced->dtype);

    /* Verify reduced tensor has correct shape
     * meanreduce/sumreduce keep the dimension but set it to 1 */
    assert(reduced->ndim == src->ndim);
    for (int i = 0; i < src->ndim; i++) {
        if (i == axis) {
            assert(reduced->dims[i] == 1);
        } else {
            assert(reduced->dims[i] == src->dims[i]);
        }
    }

    /* Create or validate destination */
    if (!dst) {
        dst = tofu_tensor_zeros(src->ndim, src->dims, src->dtype);
    } else {
        assert(dst->data);
        assert(dst->ndim == src->ndim);
        assert(dst->dtype == src->dtype);
        for (int i = 0; i < src->ndim; i++) {
            assert(dst->dims[i] == src->dims[i]);
        }
    }

    /* Compute dimension strides */
    int axis_size = src->dims[axis];
    int outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= src->dims[i];
    }
    int inner_size = 1;
    for (int i = axis + 1; i < src->ndim; i++) {
        inner_size *= src->dims[i];
    }

    /* Perform broadcast subtraction */
    for (int outer = 0; outer < outer_size; outer++) {
        for (int inner = 0; inner < inner_size; inner++) {
            /* Index into reduced tensor (axis dimension is 1, so we skip j loop for it) */
            int reduced_idx = outer * inner_size + inner;

            double reduced_val;
            TOFU_TENSOR_DATA_TO(reduced, reduced_idx, reduced_val, TOFU_DOUBLE);

            /* Broadcast along axis dimension */
            for (int j = 0; j < axis_size; j++) {
                int src_idx = outer * axis_size * inner_size + j * inner_size + inner;

                double src_val;
                TOFU_TENSOR_DATA_TO(src, src_idx, src_val, TOFU_DOUBLE);

                double result = src_val - reduced_val;
                TOFU_TENSOR_DATA_FROM(dst, src_idx, result, TOFU_DOUBLE);
            }
        }
    }

    return dst;
}
