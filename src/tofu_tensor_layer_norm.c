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
#include "tofu_tensor_internal.h"

/*
 * Layer Normalization: normalize along specified axis
 * Formula: y = gamma * (x - mean) / sqrt(variance + eps) + beta
 *
 * Args:
 *   src: input tensor
 *   dst: output tensor (can be NULL)
 *   gamma: scale parameter (1D tensor matching axis size, can be NULL for no scaling)
 *   beta: shift parameter (1D tensor matching axis size, can be NULL for no shift)
 *   axis: axis along which to normalize
 *   eps: epsilon for numerical stability (typically 1e-5)
 */
TOFU_EXPORT tofu_tensor *tofu_tensor_layer_norm(const tofu_tensor *src, tofu_tensor *dst,
                                          const tofu_tensor *gamma, const tofu_tensor *beta,
                                          int axis, double eps)
{
    int thread_num, reduce_vol, batch_vol;
    int i, di, si;
    int dim_size;

    assert(src && src->data);
    assert(axis < src->ndim && axis >= 0);
    assert(eps > 0);

    if (gamma) {
        assert(gamma->ndim == 1 && gamma->dims[0] == src->dims[axis]);
    }
    if (beta) {
        assert(beta->ndim == 1 && beta->dims[0] == src->dims[axis]);
    }

    if (dst) {
#ifndef NDEBUG
        assert(dst->data);
        assert(src->dtype == dst->dtype);
        assert(tofu_tensor_issameshape(src, dst));
#endif
    } else {
        dst = tofu_tensor_zeros(src->ndim, src->dims, src->dtype);
    }

    /* Calculate iteration parameters */
    for (i = axis + 1, thread_num = 1; i < src->ndim; i++)
        thread_num *= src->dims[i];
    reduce_vol = thread_num;
    batch_vol = thread_num * src->dims[axis];
    for (i = 0; i < axis; i++)
        thread_num *= src->dims[i];

    dim_size = src->dims[axis];

    /* Perform layer norm for each "thread" */
    for (di = 0; di < thread_num; di++) {
        si = (batch_vol - reduce_vol) * (di / reduce_vol) + di;

        /* Compute mean */
        double mean = 0.0;
        for (i = 0; i < dim_size; i++) {
            double val;
            TOFU_TENSOR_DATA_TO(src, si + i * reduce_vol, val, TOFU_DOUBLE);
            mean += val;
        }
        mean /= dim_size;

        /* Compute variance */
        double variance = 0.0;
        for (i = 0; i < dim_size; i++) {
            double val;
            TOFU_TENSOR_DATA_TO(src, si + i * reduce_vol, val, TOFU_DOUBLE);
            double diff = val - mean;
            variance += diff * diff;
        }
        variance /= dim_size;

        /* Normalize, scale, and shift */
        double std = sqrt(variance + eps);
        for (i = 0; i < dim_size; i++) {
            double val;
            TOFU_TENSOR_DATA_TO(src, si + i * reduce_vol, val, TOFU_DOUBLE);

            /* Normalize */
            val = (val - mean) / std;

            /* Scale if gamma provided */
            if (gamma) {
                double g;
                TOFU_TENSOR_DATA_TO(gamma, i, g, TOFU_DOUBLE);
                val *= g;
            }

            /* Shift if beta provided */
            if (beta) {
                double b;
                TOFU_TENSOR_DATA_TO(beta, i, b, TOFU_DOUBLE);
                val += b;
            }

            TOFU_TENSOR_DATA_FROM(dst, si + i * reduce_vol, val, TOFU_DOUBLE);
        }
    }

    return dst;
}
