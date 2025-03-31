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

#ifndef _TL_TENSOR_H_
#define _TL_TENSOR_H_

#include "tl_type.h"

#define TL_MAXDIM 8

#define TL_TENSOR_DATA(tensor, index) tl_pointer_add((tensor)->data, (index), (tensor)->dtype)

#define TL_TENSOR_DATA_TO(tensor, index, var, var_dtype)                                           \
    tl_convert(&(var), (var_dtype), TL_TENSOR_DATA((tensor), (index)), (tensor)->dtype)

#define TL_TENSOR_DATA_FROM(tensor, index, var, var_dtype)                                         \
    tl_convert(TL_TENSOR_DATA((tensor), (index)), (tensor)->dtype, &(var), (var_dtype))

#define TL_TENSOR_DATA_ASSIGN(dst, di, src, si)                                                    \
    tl_convert(TL_TENSOR_DATA((dst), (di)), (dst)->dtype, TL_TENSOR_DATA((src), (si)), (src)->dtype)

/* clang-format off */
struct tl_tensor {
    tl_dtype          dtype;
    int               len;
    int               ndim;
    int              *dims;
    void             *data;
    struct tl_tensor *owner;         /* data owner, NULL if it's itself */
    void             *backend_data;  /* for other backend dependent data */
};
typedef struct tl_tensor tl_tensor;
/* clang-format on */

#ifdef __cplusplus
TL_CPPSTART
#endif

int tl_tensor_index(const tl_tensor *t, int *coords);
void tl_tensor_coords(const tl_tensor *t, int index, int *coords);
int tl_tensor_issameshape(const tl_tensor *t1, const tl_tensor *t2);
int tl_tensor_isbroadcastable(const tl_tensor *t1, const tl_tensor *t2);
tl_tensor *tl_tensor_create(void *data, int ndim, const int *dims, tl_dtype dtype);
void tl_tensor_free(tl_tensor *t);
void tl_tensor_free_data_too(tl_tensor *t);
size_t tl_tensor_size(tl_tensor *t);
tl_tensor *tl_tensor_zeros(int ndim, const int *dims, tl_dtype dtype);
tl_tensor *tl_tensor_clone(const tl_tensor *src);
tl_tensor *tl_tensor_repeat(const tl_tensor *src, int times);
tl_tensor *tl_tensor_arange(double start, double stop, double step, tl_dtype dtype);
void tl_tensor_rearange(tl_tensor *src, double start, double stop, double step);
void tl_tensor_fprint(FILE *stream, const tl_tensor *t, const char *fmt);
void tl_tensor_print(const tl_tensor *t, const char *fmt);
int tl_tensor_save(const char *file_name, const tl_tensor *t, const char *fmt);
tl_tensor *tl_tensor_create_slice(void *data, const tl_tensor *src, int axis, int len,
                                  tl_dtype dtype);
tl_tensor *tl_tensor_zeros_slice(const tl_tensor *src, int axis, int len, tl_dtype dtype);
tl_tensor *tl_tensor_slice(const tl_tensor *src, tl_tensor *dst, int axis, int start, int len);
tl_tensor *tl_tensor_slice_nocopy(tl_tensor *src, tl_tensor *dst, int axis, int start, int len);
tl_tensor *tl_tensor_concat(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst, int axis);
tl_tensor *tl_tensor_reshape(tl_tensor *src, int ndim, const int *dims);
void tl_tensor_reshape_src(tl_tensor *src, int ndim, const int *dims);
tl_tensor *tl_tensor_maxreduce(const tl_tensor *src, tl_tensor *dst, tl_tensor *arg, int axis);
tl_tensor *tl_tensor_elew(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst,
                          tl_elew_op elew_op);
tl_tensor *tl_tensor_elew_param(const tl_tensor *src, double param, tl_tensor *dst,
                                tl_elew_op elew_op);
tl_tensor *tl_tensor_dot_product(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst);
tl_tensor *tl_tensor_transpose(const tl_tensor *src, tl_tensor *dst, const int *axes);
tl_tensor *tl_tensor_lrelu(const tl_tensor *src, tl_tensor *dst, float negslope);
tl_tensor *tl_tensor_convert(const tl_tensor *src, tl_tensor *dst, tl_dtype dtype_d);
tl_tensor *tl_tensor_resize(const tl_tensor *src, tl_tensor *dst, const int *new_dims,
                            tl_resize_type rtype);
tl_tensor *tl_tensor_submean(const tl_tensor *src, tl_tensor *dst, const double *mean);
tl_tensor *tl_tensor_broadcast_to(const tl_tensor *src, tl_tensor *dst, int ndim, const int *dims);
tl_tensor *tl_tensor_elew_broadcast(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst,
                               tl_elew_op elew_op);


#ifdef __cplusplus
TL_CPPEND
#endif

#endif /* _TL_TENSOR_H_ */
