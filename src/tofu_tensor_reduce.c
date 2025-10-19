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

/* Sum reduction along an axis */
TOFU_EXPORT tofu_tensor *tofu_tensor_sumreduce(const tofu_tensor *src, tofu_tensor *dst, int axis)
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
        for (i = 0; i < dst->ndim; i++)
            assert(i == axis ? dst->dims[i] == 1 : dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tofu_tensor_zeros_slice(src, axis, 1, src->dtype);
    }

    /* Calculate iteration parameters */
    for (i = axis + 1, thread_num = 1; i < dst->ndim; i++)
        thread_num *= dst->dims[i];
    reduce_vol = thread_num;
    batch_vol = thread_num * src->dims[axis];
    for (i = 0; i < axis; i++)
        thread_num *= dst->dims[i];

    dim_size = src->dims[axis];

    /* Perform sum reduction */
    for (di = 0; di < thread_num; di++) {
        si = (batch_vol - reduce_vol) * (di / reduce_vol) + di;

        /* Initialize sum to zero */
        double sum = 0.0;

        /* Sum along axis */
        for (i = 0; i < dim_size; i++) {
            double val;
            TOFU_TENSOR_DATA_TO(src, si + i * reduce_vol, val, TOFU_DOUBLE);
            sum += val;
        }

        /* Store result */
        TOFU_TENSOR_DATA_FROM(dst, di, sum, TOFU_DOUBLE);
    }

    return dst;
}

/* Mean reduction along an axis */
TOFU_EXPORT tofu_tensor *tofu_tensor_meanreduce(const tofu_tensor *src, tofu_tensor *dst, int axis)
{
    int dim_size;

    assert(src && src->data);
    assert(axis < src->ndim && axis >= 0);

    /* First compute sum */
    dst = tofu_tensor_sumreduce(src, dst, axis);

    /* Then divide by dim_size */
    dim_size = src->dims[axis];
    double divisor = (double)dim_size;

    for (int i = 0; i < dst->len; i++) {
        double val;
        TOFU_TENSOR_DATA_TO(dst, i, val, TOFU_DOUBLE);
        val /= divisor;
        TOFU_TENSOR_DATA_FROM(dst, i, val, TOFU_DOUBLE);
    }

    return dst;
}
