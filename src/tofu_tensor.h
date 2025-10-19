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

#ifndef _TOFU_TENSOR_H_
#define _TOFU_TENSOR_H_

#include "tofu_type.h"

#define TOFU_MAXDIM 8

#define TOFU_TENSOR_DATA(tensor, index) tofu_pointer_add((tensor)->data, (index), (tensor)->dtype)

#define TOFU_TENSOR_DATA_TO(tensor, index, var, var_dtype)                                           \
    tofu_convert(&(var), (var_dtype), TOFU_TENSOR_DATA((tensor), (index)), (tensor)->dtype)

#define TOFU_TENSOR_DATA_FROM(tensor, index, var, var_dtype)                                         \
    tofu_convert(TOFU_TENSOR_DATA((tensor), (index)), (tensor)->dtype, &(var), (var_dtype))

#define TOFU_TENSOR_DATA_ASSIGN(dst, di, src, si)                                                    \
    tofu_convert(TOFU_TENSOR_DATA((dst), (di)), (dst)->dtype, TOFU_TENSOR_DATA((src), (si)), (src)->dtype)

/* clang-format off */
struct tofu_tensor {
    tofu_dtype          dtype;
    int               len;
    int               ndim;
    int              *dims;
    void             *data;
    struct tofu_tensor *owner;         /* data owner, NULL if it's itself */
    void             *backend_data;  /* for other backend dependent data */
};
typedef struct tofu_tensor tofu_tensor;
/* clang-format on */

#ifdef __cplusplus
TOFU_CPPSTART
#endif

int tofu_tensor_index(const tofu_tensor *t, int *coords);
void tofu_tensor_coords(const tofu_tensor *t, int index, int *coords);
int tofu_tensor_issameshape(const tofu_tensor *t1, const tofu_tensor *t2);
int tofu_tensor_isbroadcastable(const tofu_tensor *t1, const tofu_tensor *t2);
tofu_tensor *tofu_tensor_create(void *data, int ndim, const int *dims, tofu_dtype dtype);
void tofu_tensor_free(tofu_tensor *t);
void tofu_tensor_free_data_too(tofu_tensor *t);
size_t tofu_tensor_size(tofu_tensor *t);
tofu_tensor *tofu_tensor_zeros(int ndim, const int *dims, tofu_dtype dtype);
tofu_tensor *tofu_tensor_clone(const tofu_tensor *src);
tofu_tensor *tofu_tensor_repeat(const tofu_tensor *src, int times);
tofu_tensor *tofu_tensor_arange(double start, double stop, double step, tofu_dtype dtype);
void tofu_tensor_rearange(tofu_tensor *src, double start, double stop, double step);
void tofu_tensor_fprint(FILE *stream, const tofu_tensor *t, const char *fmt);
void tofu_tensor_print(const tofu_tensor *t, const char *fmt);
int tofu_tensor_save(const char *file_name, const tofu_tensor *t, const char *fmt);
tofu_tensor *tofu_tensor_create_slice(void *data, const tofu_tensor *src, int axis, int len,
                                  tofu_dtype dtype);
tofu_tensor *tofu_tensor_zeros_slice(const tofu_tensor *src, int axis, int len, tofu_dtype dtype);
tofu_tensor *tofu_tensor_slice(const tofu_tensor *src, tofu_tensor *dst, int axis, int start, int len);
tofu_tensor *tofu_tensor_slice_nocopy(tofu_tensor *src, tofu_tensor *dst, int axis, int start, int len);
tofu_tensor *tofu_tensor_concat(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst, int axis);
tofu_tensor *tofu_tensor_reshape(tofu_tensor *src, int ndim, const int *dims);
void tofu_tensor_reshape_src(tofu_tensor *src, int ndim, const int *dims);
tofu_tensor *tofu_tensor_maxreduce(const tofu_tensor *src, tofu_tensor *dst, tofu_tensor *arg, int axis);
tofu_tensor *tofu_tensor_sumreduce(const tofu_tensor *src, tofu_tensor *dst, int axis);
tofu_tensor *tofu_tensor_meanreduce(const tofu_tensor *src, tofu_tensor *dst, int axis);
tofu_tensor *tofu_tensor_sub_broadcast(const tofu_tensor *src, const tofu_tensor *reduced, tofu_tensor *dst, int axis);
tofu_tensor *tofu_tensor_softmax(const tofu_tensor *src, tofu_tensor *dst, int axis);
tofu_tensor *tofu_tensor_layer_norm(const tofu_tensor *src, tofu_tensor *dst,
                                const tofu_tensor *gamma, const tofu_tensor *beta,
                                int axis, double eps);
tofu_tensor *tofu_tensor_elew(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst,
                          tofu_elew_op elew_op);
tofu_tensor *tofu_tensor_elew_param(const tofu_tensor *src, double param, tofu_tensor *dst,
                                tofu_elew_op elew_op);
tofu_tensor *tofu_tensor_inner(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst);
tofu_tensor *tofu_tensor_matmul(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst);
tofu_tensor *tofu_tensor_outer(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst);
tofu_tensor *tofu_tensor_transpose(const tofu_tensor *src, tofu_tensor *dst, const int *axes);
tofu_tensor *tofu_tensor_lrelu(const tofu_tensor *src, tofu_tensor *dst, float negslope);
tofu_tensor *tofu_tensor_convert(const tofu_tensor *src, tofu_tensor *dst, tofu_dtype dtype_d);
tofu_tensor *tofu_tensor_resize(const tofu_tensor *src, tofu_tensor *dst, const int *new_dims,
                            tofu_resize_type rtype);
tofu_tensor *tofu_tensor_submean(const tofu_tensor *src, tofu_tensor *dst, const double *mean);
tofu_tensor *tofu_tensor_broadcast_to(const tofu_tensor *src, tofu_tensor *dst, int ndim, const int *dims);
tofu_tensor *tofu_tensor_elew_broadcast(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst,
                               tofu_elew_op elew_op);


#ifdef __cplusplus
TOFU_CPPEND
#endif

#endif /* _TOFU_TENSOR_H_ */

/* Create a tensor with heap-allocated data (safe for gradients)
 * IMPORTANT: Do NOT use compound literals like (float[]){1.0f, 2.0f} for
 * gradient data, as they create stack memory that becomes invalid.
 * Use this function or manually malloc the data.
 */
TOFU_EXPORT tofu_tensor* tofu_tensor_create_with_values(const float* values, int ndim, const int* dims);

