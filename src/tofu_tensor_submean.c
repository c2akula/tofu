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

/* src: H*W*C, dst: C*H*W */
TOFU_EXPORT tofu_tensor *tofu_tensor_submean(const tofu_tensor *src, tofu_tensor *dst, const double *mean)
{
    assert(src && src->data);
    assert(mean);
    assert(src->ndim == 3);
    assert(src->dims[2] == 3);
    int new_dims[] = { src->dims[2], src->dims[0], src->dims[1] };
    int c, i, H, W, C;
    double data;

    if (dst) {
        assert(dst->data);
        assert(dst->ndim == src->ndim);
        assert(dst->dims[0] == 3);
    } else {
        dst = tofu_tensor_zeros(src->ndim, new_dims, TOFU_FLOAT);
    }

    H = src->dims[0];
    W = src->dims[1];
    C = src->dims[2];
    for (c = 0; c < C; c++) {
        for (i = 0; i < H * W; i++) {
            TOFU_TENSOR_DATA_TO(src, i * C + c, data, TOFU_DOUBLE);
            data = data - mean[c];
            TOFU_TENSOR_DATA_FROM(dst, c * H * W + i, data, TOFU_DOUBLE);
        }
    }

    return dst;
}
