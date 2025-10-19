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

TOFU_EXPORT tofu_tensor *tofu_tensor_elew(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst,
                                    tofu_elew_op elew_op)
{
    int thread_num;
    int di;
    size_t dsize;
    tofu_dtype dtype;
    void *s1_data, *s2_data, *d_data;
    void *elew_res;
    tofu_elew_func elew;

    assert(src1->data && src2->data);
    assert(src1->dtype == src2->dtype);
    
    // Check if shapes are the same for direct operation
    if (tofu_tensor_issameshape(src1, src2)) {
        // Same shapes - use standard element-wise operation
        if (dst) {
            assert(dst->data);
            assert(tofu_tensor_issameshape(src1, dst));
            assert(src1->dtype == dst->dtype);
        } else {
            dst = tofu_tensor_zeros(src1->ndim, src1->dims, src1->dtype);
        }

        thread_num = dst->len;
        s1_data = src1->data;
        s2_data = src2->data;
        d_data = dst->data;
        dtype = src1->dtype;
        dsize = tofu_size_of(dtype);
        elew = tofu_elew_getfunc(dtype);
        elew_res = tofu_alloc(dsize);
        for (di = 0; di < thread_num; di++) {
            elew(tofu_padd(s1_data, di, dsize), tofu_padd(s2_data, di, dsize), elew_res, elew_op);
            tofu_passign(d_data, di, elew_res, 0, dsize);
        }
        tofu_free(elew_res);
    } else if (tofu_tensor_isbroadcastable(src1, src2)) {
        // Different shapes but broadcastable - use broadcasting element-wise operation
        return tofu_tensor_elew_broadcast(src1, src2, dst, elew_op);
    } else {
        // Not broadcastable - error
        tofu_warn_ret("Tensors are not broadcastable for element-wise operation");
        return NULL;
    }

    return dst;
}

TOFU_EXPORT tofu_tensor *tofu_tensor_elew_param(const tofu_tensor *src, double param, tofu_tensor *dst,
                                          tofu_elew_op elew_op)
{
    int thread_num;
    int di;
    size_t dsize;
    tofu_dtype dtype;
    void *s_data, *d_data;
    void *elew_res, *param_data;
    tofu_elew_func elew;
    tofu_tensor *param_tensor = NULL;

    assert(src && src->data);

    // Decide whether to use broadcasting or direct approach
    if (src->ndim > 1) {
        // For higher-dimensional tensors, use broadcasting with a scalar tensor
        param_data = tofu_alloc(tofu_size_of(src->dtype));
        tofu_convert(param_data, src->dtype, &param, TOFU_DOUBLE);
        param_tensor = tofu_tensor_create(param_data, 1, (int[]){1}, src->dtype);
        
        // Use the broadcasting element-wise operation
        dst = tofu_tensor_elew_broadcast(param_tensor, src, dst, elew_op);
        
        // Clean up
        tofu_tensor_free(param_tensor);
        tofu_free(param_data);
        return dst;
    }
    
    // For simple cases, use the direct approach for better performance
    if (dst) {
        assert(dst->data);
        assert(tofu_tensor_issameshape(src, dst));
        assert(src->dtype == dst->dtype);
    } else {
        dst = tofu_tensor_zeros(src->ndim, src->dims, src->dtype);
    }

    thread_num = dst->len;
    s_data = src->data;
    d_data = dst->data;
    dtype = src->dtype;
    dsize = tofu_size_of(dtype);
    elew = tofu_elew_getfunc(dtype);
    elew_res = tofu_alloc(dsize);
    param_data = tofu_alloc(dsize);
    tofu_convert(param_data, dtype, &param, TOFU_DOUBLE);
    for (di = 0; di < thread_num; di++) {
        elew(tofu_padd(s_data, di, dsize), param_data, elew_res, elew_op);
        tofu_passign(d_data, di, elew_res, 0, dsize);
    }
    tofu_free(elew_res);
    tofu_free(param_data);

    return dst;
}
