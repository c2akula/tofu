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

#include "tl_tensor.h"
#include "tl_util.h"
#include <assert.h>
#include <string.h>
#include <errno.h>

/*
 * Matrix multiplication with NumPy-compatible broadcasting semantics.
 *
 * Behavior:
 * - 1-D @ 1-D: dot product (scalar)
 * - 2-D @ 2-D: standard matrix multiplication
 * - N-D @ 1-D: matrix-vector for last 2 dims, preserve batch dims
 * - 1-D @ N-D: vector-matrix for last 2 dims, preserve batch dims
 * - N-D @ N-D: batch matrix multiplication with broadcasting on batch dims
 *
 * Key difference from inner():
 * - matmul() broadcasts batch dimensions (shared indices)
 * - inner() uses cartesian product (independent indices)
 */
tl_tensor *tl_tensor_matmul(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst)
{
    assert(src1 && src2);
    assert(src1->dtype == src2->dtype);

    int is_src1_1d = (src1->ndim == 1);
    int is_src2_1d = (src2->ndim == 1);

    /* Special case: 1-D @ 1-D -> scalar (dot product) */
    if (is_src1_1d && is_src2_1d) {
        assert(src1->dims[0] == src2->dims[0]);

        if (!dst) {
            dst = tl_tensor_zeros(1, (int[]){1}, src1->dtype);
        }

        /* Compute dot product */
        for (int k = 0; k < src1->dims[0]; k++) {
            double val1, val2, result;
            TL_TENSOR_DATA_TO(src1, k, val1, TL_DOUBLE);
            TL_TENSOR_DATA_TO(src2, k, val2, TL_DOUBLE);
            TL_TENSOR_DATA_TO(dst, 0, result, TL_DOUBLE);
            result += val1 * val2;
            TL_TENSOR_DATA_FROM(dst, 0, result, TL_DOUBLE);
        }
        return dst;
    }

    /* Handle 1-D cases by temporarily treating them as 2-D */
    const tl_tensor *a = src1;
    const tl_tensor *b = src2;
    int squeeze_left = 0;
    int squeeze_right = 0;

    tl_tensor *a_temp = NULL;
    tl_tensor *b_temp = NULL;

    if (is_src1_1d) {
        /* Prepend dimension: [n] -> [1, n] */
        squeeze_left = 1;
        int new_dims[TL_MAXDIM];
        new_dims[0] = 1;
        new_dims[1] = src1->dims[0];
        a_temp = tl_tensor_reshape((tl_tensor*)src1, 2, new_dims);
        a = a_temp;
    }

    if (is_src2_1d) {
        /* Append dimension: [n] -> [n, 1] */
        squeeze_right = 1;
        int new_dims[TL_MAXDIM];
        new_dims[0] = src2->dims[0];
        new_dims[1] = 1;
        b_temp = tl_tensor_reshape((tl_tensor*)src2, 2, new_dims);
        b = b_temp;
    }

    /* Now both a and b have at least 2 dimensions */
    /* Extract batch dimensions (all except last 2) */
    int a_batch_ndim = a->ndim - 2;
    int b_batch_ndim = b->ndim - 2;

    /* Matrix dimensions */
    int m = a->dims[a->ndim - 2];  /* rows of a */
    int n = a->dims[a->ndim - 1];  /* cols of a / rows of b */
    int p = b->dims[b->ndim - 1];  /* cols of b */

    /* Verify contraction dimension matches */
    assert(n == b->dims[b->ndim - 2]);

    /* Compute broadcast dimensions for batch dimensions */
    int max_batch_ndim = (a_batch_ndim > b_batch_ndim) ? a_batch_ndim : b_batch_ndim;
    int batch_dims[TL_MAXDIM];

    for (int i = 0; i < max_batch_ndim; i++) {
        int a_idx = a_batch_ndim - max_batch_ndim + i;
        int b_idx = b_batch_ndim - max_batch_ndim + i;

        int a_dim = (a_idx >= 0) ? a->dims[a_idx] : 1;
        int b_dim = (b_idx >= 0) ? b->dims[b_idx] : 1;

        if (a_dim == b_dim) {
            batch_dims[i] = a_dim;
        } else if (a_dim == 1) {
            batch_dims[i] = b_dim;
        } else if (b_dim == 1) {
            batch_dims[i] = a_dim;
        } else {
            /* Broadcasting error */
            tl_tensor_free(a_temp);
            tl_tensor_free(b_temp);
            errno = EINVAL;
            return NULL;
        }
    }

    /* Calculate output dimensions */
    int out_ndim = max_batch_ndim + 2 - squeeze_left - squeeze_right;
    int out_dims[TL_MAXDIM];
    int out_idx = 0;

    for (int i = 0; i < max_batch_ndim; i++) {
        out_dims[out_idx++] = batch_dims[i];
    }
    if (!squeeze_left) out_dims[out_idx++] = m;
    if (!squeeze_right) out_dims[out_idx++] = p;

    /* Create or validate destination tensor */
    if (!dst) {
        dst = tl_tensor_zeros(out_ndim, out_dims, src1->dtype);
    } else {
        assert(dst->ndim == out_ndim);
        for (int i = 0; i < out_ndim; i++) {
            assert(dst->dims[i] == out_dims[i]);
        }
    }

    /* Compute strides for broadcasting */
    int a_batch_strides[TL_MAXDIM];
    int b_batch_strides[TL_MAXDIM];

    /* Compute strides for a's batch dimensions */
    for (int i = 0; i < max_batch_ndim; i++) {
        int a_idx = a_batch_ndim - max_batch_ndim + i;
        if (a_idx >= 0 && a->dims[a_idx] > 1) {
            int stride = 1;
            for (int j = a_idx + 1; j < a->ndim; j++) {
                stride *= a->dims[j];
            }
            a_batch_strides[i] = stride;
        } else {
            a_batch_strides[i] = 0;  /* broadcast */
        }
    }

    /* Compute strides for b's batch dimensions */
    for (int i = 0; i < max_batch_ndim; i++) {
        int b_idx = b_batch_ndim - max_batch_ndim + i;
        if (b_idx >= 0 && b->dims[b_idx] > 1) {
            int stride = 1;
            for (int j = b_idx + 1; j < b->ndim; j++) {
                stride *= b->dims[j];
            }
            b_batch_strides[i] = stride;
        } else {
            b_batch_strides[i] = 0;  /* broadcast */
        }
    }

    /* Matrix strides (last 2 dimensions) */
    int a_row_stride = n;
    int b_row_stride = p;

    /* Perform batched matrix multiplication */
    int total_batches = 1;
    for (int i = 0; i < max_batch_ndim; i++) {
        total_batches *= batch_dims[i];
    }

    for (int batch_idx = 0; batch_idx < total_batches; batch_idx++) {
        /* Compute batch coordinates */
        int batch_coords[TL_MAXDIM];
        int temp_idx = batch_idx;
        for (int i = max_batch_ndim - 1; i >= 0; i--) {
            batch_coords[i] = temp_idx % batch_dims[i];
            temp_idx /= batch_dims[i];
        }

        /* Compute base offsets using broadcast strides */
        int a_base = 0;
        int b_base = 0;
        for (int i = 0; i < max_batch_ndim; i++) {
            a_base += batch_coords[i] * a_batch_strides[i];
            b_base += batch_coords[i] * b_batch_strides[i];
        }

        /* Perform matrix multiplication for this batch */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0.0;

                for (int k = 0; k < n; k++) {
                    int a_idx = a_base + i * a_row_stride + k;
                    int b_idx = b_base + k * b_row_stride + j;

                    double a_val, b_val;
                    TL_TENSOR_DATA_TO(a, a_idx, a_val, TL_DOUBLE);
                    TL_TENSOR_DATA_TO(b, b_idx, b_val, TL_DOUBLE);
                    sum += a_val * b_val;
                }

                /* Compute output index */
                int dst_idx;
                if (squeeze_left && squeeze_right) {
                    /* Result is 1-D or 0-D */
                    dst_idx = batch_idx;
                } else if (squeeze_left) {
                    /* Result drops left dimension */
                    dst_idx = batch_idx * p + j;
                } else if (squeeze_right) {
                    /* Result drops right dimension */
                    dst_idx = batch_idx * m + i;
                } else {
                    /* Full 2-D output per batch */
                    dst_idx = batch_idx * m * p + i * p + j;
                }

                TL_TENSOR_DATA_FROM(dst, dst_idx, sum, TL_DOUBLE);
            }
        }
    }

    /* Clean up temporary tensors */
    tl_tensor_free(a_temp);
    tl_tensor_free(b_temp);

    return dst;
}
