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
#include "tl_tensor_internal.h"

/* Softmax with numerical stability (subtract max before exp) */
TL_EXPORT tl_tensor *tl_tensor_softmax(const tl_tensor *src, tl_tensor *dst, int axis)
{
    int thread_num, reduce_vol, batch_vol;
    int i, di, si;
    int dim_size;

    assert(src && src->data);
    assert(axis < src->ndim && axis >= 0);

    if (dst) {
#ifndef NDEBUG
        assert(dst->data);
        assert(src->dtype == dst->dtype);
        assert(tl_tensor_issameshape(src, dst));
#endif
    } else {
        dst = tl_tensor_zeros(src->ndim, src->dims, src->dtype);
    }

    /* Calculate iteration parameters */
    for (i = axis + 1, thread_num = 1; i < src->ndim; i++)
        thread_num *= src->dims[i];
    reduce_vol = thread_num;
    batch_vol = thread_num * src->dims[axis];
    for (i = 0; i < axis; i++)
        thread_num *= src->dims[i];

    dim_size = src->dims[axis];

    /* Perform softmax for each "thread" */
    for (di = 0; di < thread_num; di++) {
        si = (batch_vol - reduce_vol) * (di / reduce_vol) + di;

        /* Find max along axis for numerical stability */
        double max_val = -INFINITY;
        for (i = 0; i < dim_size; i++) {
            double val;
            TL_TENSOR_DATA_TO(src, si + i * reduce_vol, val, TL_DOUBLE);
            if (val > max_val)
                max_val = val;
        }

        /* Compute exp(x - max) and sum */
        double sum = 0.0;
        for (i = 0; i < dim_size; i++) {
            double val;
            TL_TENSOR_DATA_TO(src, si + i * reduce_vol, val, TL_DOUBLE);
            val = exp(val - max_val);
            TL_TENSOR_DATA_FROM(dst, si + i * reduce_vol, val, TL_DOUBLE);
            sum += val;
        }

        /* Normalize by sum */
        for (i = 0; i < dim_size; i++) {
            double val;
            TL_TENSOR_DATA_TO(dst, si + i * reduce_vol, val, TL_DOUBLE);
            val /= sum;
            TL_TENSOR_DATA_FROM(dst, si + i * reduce_vol, val, TL_DOUBLE);
        }
    }

    return dst;
}
