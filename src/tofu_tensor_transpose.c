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

TOFU_EXPORT tofu_tensor *tofu_tensor_transpose(const tofu_tensor *src, tofu_tensor *dst, const int *axes)
{
    int i;

    /* If axes is NULL, reverse all dimensions */
    int default_axes[TOFU_MAXDIM];
    if (!axes) {
        for (i = 0; i < src->ndim; i++)
            default_axes[i] = src->ndim - 1 - i;
        axes = default_axes;
    }

#ifndef NDEBUG
    int tmp[TOFU_MAXDIM] = { 0 };
    for (i = 0; i < src->ndim; i++)
        tmp[axes[i]] = 1;
    for (i = 0; i < src->ndim; i++)
        assert(tmp[i] && "axes don't match src tensor's shape");
    assert(src && src->data);
#endif
    if (dst) {
#ifndef NDEBUG
        assert(dst->data);
        assert(src->dtype == dst->dtype);
        assert(src->len == dst->len);
        assert(src->ndim == dst->ndim);
        for (i = 0; i < dst->ndim; i++)
            assert(src->dims[axes[i]] == dst->dims[i]);
#endif
    } else {
        int d_dims[TOFU_MAXDIM];
        for (i = 0; i < src->ndim; i++)
            d_dims[i] = src->dims[axes[i]];
        dst = tofu_tensor_zeros(src->ndim, d_dims, src->dtype);
    }

    int di, si;
    int s_ids[TOFU_MAXDIM], d_ids[TOFU_MAXDIM];
    size_t dsize = tofu_size_of(src->dtype);
    int ndim = dst->ndim;

    for (di = 0; di < dst->len; di++) {
        tofu_get_coords(di, d_ids, ndim, dst->dims);
        for (i = 0; i < ndim; i++)
            s_ids[axes[i]] = d_ids[i];
        si = tofu_get_index(s_ids, ndim, src->dims);

        tofu_passign(dst->data, di, src->data, si, dsize);
    }

    return dst;
}
