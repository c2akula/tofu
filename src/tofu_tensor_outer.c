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

#include "tofu_tensor.h"
#include "tofu_util.h"
#include <assert.h>

/*
 * Outer product with NumPy-compatible semantics.
 *
 * Behavior:
 * - Flattens both input tensors
 * - Computes: out[i,j] = a[i] * b[j]
 * - Always produces 2-D output: [m,n] where m=a.size, n=b.size
 *
 * Key difference from inner/matmul:
 * - No summation involved (simple element-wise multiplication)
 * - Inputs are flattened first (doesn't preserve structure)
 * - No broadcasting (straightforward cartesian product)
 */
tofu_tensor *tofu_tensor_outer(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst)
{
    assert(src1 && src2);
    assert(src1->dtype == src2->dtype);

    /* Outer product uses total size (flattened) */
    int m = src1->len;  /* Total elements in src1 */
    int n = src2->len;  /* Total elements in src2 */

    /* Output is always 2-D: [m, n] */
    int out_dims[2] = {m, n};

    /* Create or validate destination tensor */
    if (!dst) {
        dst = tofu_tensor_zeros(2, out_dims, src1->dtype);
    } else {
        assert(dst->ndim == 2);
        assert(dst->dims[0] == m);
        assert(dst->dims[1] == n);
        assert(dst->dtype == src1->dtype);
    }

    /* Compute outer product: out[i,j] = src1[i] * src2[j] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double val1, val2, result;
            TOFU_TENSOR_DATA_TO(src1, i, val1, TOFU_DOUBLE);
            TOFU_TENSOR_DATA_TO(src2, j, val2, TOFU_DOUBLE);
            result = val1 * val2;
            TOFU_TENSOR_DATA_FROM(dst, i * n + j, result, TOFU_DOUBLE);
        }
    }

    return dst;
}
