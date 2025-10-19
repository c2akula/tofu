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

#ifndef _TOFU_TENSOR_BROADCAST_INTERNAL_H_
#define _TOFU_TENSOR_BROADCAST_INTERNAL_H_

#include "tofu_tensor.h"

/**
 * Internal broadcasting functions exposed for testing purposes.
 *
 * NOTE: These functions are internal implementation details and should NOT
 * be used by external code. They are only exposed to enable comprehensive
 * unit testing of the broadcasting mechanism.
 */

#ifdef __cplusplus
TOFU_CPPSTART
#endif

/**
 * Calculate output dimensions after broadcasting two tensors
 *
 * @param t1 First tensor
 * @param t2 Second tensor
 * @param out_ndim Output: number of dimensions in broadcast result
 * @param out_dims Output: dimension sizes (must be pre-allocated with TOFU_MAXDIM size)
 */
void tofu_compute_broadcast_dims(const tofu_tensor *t1, const tofu_tensor *t2,
                               int *out_ndim, int *out_dims);

/**
 * Computes broadcasting strides for a tensor
 *
 * Broadcasting strides determine how to map output coordinates to input indices.
 * A stride of 0 indicates that dimension is being broadcast (repeated).
 *
 * @param t Input tensor
 * @param out_ndim Number of dimensions in output (may be > t->ndim)
 * @param out_dims Output dimension sizes
 * @param strides Output: computed strides (must be pre-allocated with out_ndim size)
 */
void tofu_compute_broadcast_strides(const tofu_tensor *t, int out_ndim,
                                   const int *out_dims, int *strides);

#ifdef __cplusplus
TOFU_CPPEND
#endif

#endif /* _TOFU_TENSOR_BROADCAST_INTERNAL_H_ */
