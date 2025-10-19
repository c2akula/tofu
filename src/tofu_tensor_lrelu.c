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

TOFU_EXPORT tofu_tensor *tofu_tensor_lrelu(const tofu_tensor *src, tofu_tensor *dst, float negslope)
{
    assert(src && src->data);
    if (dst) {
        assert(dst && dst->data);
        assert(tofu_tensor_issameshape(dst, src));
        assert(dst->dtype == src->dtype);
    } else {
        dst = tofu_tensor_zeros(src->ndim, src->dims, src->dtype);
    }

    tofu_dtype dtype = src->dtype;
    size_t dsize = tofu_size_of(dtype);
    for (int i = 0; i < src->len; i++)
        tofu_lrelu(tofu_padd(dst->data, i, dsize), tofu_padd(src->data, i, dsize), negslope, dtype);

    return dst;
}
