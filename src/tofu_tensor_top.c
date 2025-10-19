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

/* static void top1(void *src, void *dst, int32_t *arg, int len, int stride, */
/*                  tofu_dtype dtype, int largest) */
/* { */
/*     void *tmp; */
/*     void *elem; */
/*     int32_t idx; */
/*     int i; */
/*     tofu_cmp_func cmp; */
/*     size_t dsize; */

/*     assert(src); */
/*     assert(dst); */
/*     dsize = tofu_size_of(dtype); */
/*     tmp = tofu_alloc(dsize); */
/*     cmp = tofu_cmp_getfunc(dtype); */

/*     if (largest) { */
/*         tofu_dtype_min(dtype, tmp); */
/*         for (i = 0; i < len; i += stride) { */
/*             elem = tofu_padd(src, i, dsize); */
/*             if (cmp(tmp, elem) < 0) { */
/*                 tofu_passign(tmp, 0, elem, 0, dsize); */
/*                 idx = i; */
/*             } */
/*         } */
/*     } else { */
/*         tofu_dtype_max(dtype, tmp); */
/*         for (i = 0; i < len; i += stride) { */
/*             elem = tofu_padd(src, i, dsize); */
/*             if (cmp(tmp, elem) > 0) { */
/*                 tofu_passign(tmp, 0, elem, 0, dsize); */
/*                 idx = i; */
/*             } */
/*         } */
/*     } */
/*     tofu_passign(dst, 0, tmp, 0, dsize); */
/*     tofu_free(tmp); */
/*     *arg = idx; */
/* } */

/* tofu_tensor *tofu_tensor_topk(const tofu_tensor *src, tofu_tensor *dst, tofu_tensor *arg, */
/*                           int axis, int k, int sorted, int largest) */
/* { */
/*     int i; */
/*     int strides[TOFU_MAXDIM]; */
/*     int cmp_seq; */

/*     assert(src && src->data); */
/*     assert(k > 0 && k <= src->dims[src->ndim]); */
/*     assert(axis < src->ndim && axis >= 0); */

/*     if (dst) { */
/* #ifndef NDEBUG */
/*         assert(dst->data); */
/*         assert(src->dtype == dst->dtype); */
/*         for (i = 0; i < dst->ndim; i++) */
/*             assert(i == axis ? dst->dims[i] == k : */
/*                    dst->dims[i] == src->dims[i]); */
/* #endif */
/*     } else { */
/*         dst = tofu_tensor_zeros_slice(src, axis, k, src->dtype); */
/*     } */
/*     if (arg) { */
/* #ifndef NDEBUG */
/*         assert(arg->data); */
/*         assert(arg->dtype == TOFU_INT32); */
/*         for (i = 0; i < arg->ndim; i++) */
/*             assert(i == axis ? arg->dims[i] == k : */
/*                    arg->dims[i] == src->dims[i]); */
/* #endif */
/*     } */

/*     tofu_get_strides(src, strides); */
/*     if (k == 1) { */
/*         for (i = ) */
/*     } */

/* } */
