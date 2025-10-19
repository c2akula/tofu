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

TOFU_EXPORT int tofu_tensor_index(const tofu_tensor *t, int *coords)
{
    assert(t);
    assert(coords);
#ifdef NDEBUG
    for (int i = 0; i < t->ndim; i++)
        assert(coords[i] >= 0 && coords[i] < t->dims[i]);
#endif
    return tofu_get_index(coords, t->ndim, t->dims);
}

TOFU_EXPORT void tofu_tensor_coords(const tofu_tensor *t, int index, int *coords)
{
    assert(t);
    assert(index >= 0 && index < t->len);
    assert(coords);
    tofu_get_coords(index, coords, t->ndim, t->dims);
}

TOFU_EXPORT int tofu_tensor_issameshape(const tofu_tensor *t1, const tofu_tensor *t2)
{
    int ndim;

    assert(t1 && t2);
    if (t1->ndim == t2->ndim) {
        ndim = t1->ndim;
        while (--ndim >= 0)
            if (t1->dims[ndim] != t2->dims[ndim])
                return 0;
        return 1;
    }
    return 0;
}

/* Implementation moved to tofu_tensor_broadcast.c */

TOFU_EXPORT tofu_tensor *tofu_tensor_create(void *data, int ndim, const int *dims, tofu_dtype dtype)
{
    tofu_tensor *t;

    assert(ndim > 0 && ndim <= TOFU_MAXDIM);
    for (int i = 0; i < ndim; i++)
        assert(dims[i] > 0);
    tofu_check_dtype(dtype);

    t = (tofu_tensor *)tofu_alloc(sizeof(tofu_tensor));
    t->len = tofu_compute_length(ndim, dims);
    t->ndim = ndim;
    t->dims = (int *)tofu_clone(dims, sizeof(int) * ndim);
    t->dtype = dtype;
    t->backend_data = NULL;
    t->data = data;
    t->owner = NULL;

    return t;
}

TOFU_EXPORT void tofu_tensor_free(tofu_tensor *t)
{
    if (!t)
        return;
    tofu_free(t->dims);
    tofu_free(t);
}

TOFU_EXPORT void tofu_tensor_free_data_too(tofu_tensor *t)
{
    if (!t)
        return;
    tofu_free(t->data);
    tofu_tensor_free(t);
}

TOFU_EXPORT tofu_tensor *tofu_tensor_zeros(int ndim, const int *dims, tofu_dtype dtype)
{
    tofu_tensor *t;
    size_t size;

    t = tofu_tensor_create(NULL, ndim, dims, dtype);
    t->owner = t;
    size = t->len * tofu_size_of(dtype);
    t->data = tofu_alloc(size);
    memset(t->data, 0, size);
    return t;
}

TOFU_EXPORT size_t tofu_tensor_size(tofu_tensor *t)
{
    return t->len * tofu_size_of(t->dtype);
}

TOFU_EXPORT tofu_tensor *tofu_tensor_clone(const tofu_tensor *src)
{
    void *data;
    tofu_tensor *dst;

    assert(src);
    data = tofu_clone(src->data, src->len * tofu_size_of(src->dtype));
    dst = tofu_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

TOFU_EXPORT tofu_tensor *tofu_tensor_repeat(const tofu_tensor *src, int times)
{
    void *data;
    int *dims;
    tofu_tensor *dst;

    assert(src);
    data = tofu_repeat(src->data, src->len * tofu_size_of(src->dtype), times);
    dims = (int *)tofu_alloc(sizeof(int) * (src->ndim + 1));
    memmove(dims + 1, src->dims, sizeof(int) * (src->ndim));
    dims[0] = times;
    dst = tofu_tensor_create(data, src->ndim + 1, dims, src->dtype);
    dst->owner = dst;
    tofu_free(dims);
    return dst;
}

TOFU_EXPORT tofu_tensor *tofu_tensor_arange(double start, double stop, double step, tofu_dtype dtype)
{
    int dims[1];
    tofu_tensor *dst;
    double len, elem;
    size_t dsize;

    dsize = tofu_size_of(dtype);
#ifdef TOFU_DEBUG
    double max_d, min_d;
    max_d = tofu_dtype_max_double(dtype);
    min_d = tofu_dtype_min_double(dtype);
    assert(start >= min_d && start <= max_d);
    assert(stop >= min_d && stop <= max_d);
    assert(step >= min_d && step <= max_d);
    assert(step != 0);
#endif

    /* Calculate length supporting all cases: positive/negative step, empty arrays */
    double diff = stop - start;
    if ((step > 0 && diff <= 0) || (step < 0 && diff >= 0)) {
        /* Empty array: incompatible step direction or start == stop */
        return NULL;
    } else {
        len = ceil(diff / step);
    }

    if (len > INT32_MAX)
        return NULL;

    dims[0] = (int)len;
    dst = tofu_tensor_zeros(1, dims, dtype);
    for (int i = 0; i < dims[0]; i++) {
        elem = start + step * i;
        tofu_convert(tofu_padd(dst->data, i, dsize), dtype, &elem, TOFU_DOUBLE);
    }

    return dst;
}

TOFU_EXPORT void tofu_tensor_rearange(tofu_tensor *src, double start, double stop, double step)
{
    double len, elem;
    size_t dsize;

#ifdef TOFU_DEBUG
    double max_d, min_d;
    max_d = tofu_dtype_max_double(src->dtype);
    min_d = tofu_dtype_min_double(src->dtype);
    assert(start >= min_d && start <= max_d);
    assert(stop >= min_d && stop <= max_d);
    assert(step >= min_d && step <= max_d);
    assert(step != 0);
#endif

    /* Calculate length supporting all cases: positive/negative step, empty arrays */
    double diff = stop - start;
    if ((step > 0 && diff <= 0) || (step < 0 && diff >= 0)) {
        len = 0;  /* Empty array: incompatible step direction or start == stop */
    } else {
        len = ceil(diff / step);
    }

    dsize = tofu_size_of(src->dtype);

    assert(len <= INT32_MAX);
    assert(src->ndim == 1);
    assert(src->len == (int)len);
    assert(src->data);

    for (int i = 0; i < src->len; i++) {
        elem = start + step * i;
        tofu_convert(tofu_padd(src->data, i, dsize), src->dtype, &elem, TOFU_DOUBLE);
    }
}

TOFU_EXPORT void tofu_tensor_fprint(FILE *stream, const tofu_tensor *t, const char *fmt)
{
    int ndim, len, *dims; /* pointer short cut */
    void *data;
    tofu_dtype dtype;
    size_t dsize;

    /* dimision size and how deep current chars go */
    int *dim_sizes, *dim_levels;
    /* buffer for brackets */
    char *left_buf, *right_buf;
    char *lp, *rp;
    size_t right_len;
    int i, j, k;

    assert(stream && t);
    ndim = t->ndim;
    len = t->len;
    dims = t->dims;
    data = t->data;
    dtype = t->dtype;
    dsize = tofu_size_of(dtype);

    dim_sizes = (int *)tofu_alloc(sizeof(int) * ndim);
    dim_levels = (int *)tofu_alloc(sizeof(int) * ndim);
    dim_sizes[ndim - 1] = dims[ndim - 1];
    dim_levels[ndim - 1] = 0;
    left_buf = (char *)tofu_alloc(sizeof(char) * (ndim + 1));
    right_buf = (char *)tofu_alloc(sizeof(char) * (ndim + 1));
    lp = left_buf;
    rp = right_buf;

    for (i = ndim - 2; i >= 0; i--) {
        dim_sizes[i] = dims[i] * dim_sizes[i + 1];
        dim_levels[i] = 0;
    }
    for (i = 0; i < len; i++) {
        for (j = 0; j < ndim; j++) {
            if (i % dim_sizes[j] == 0)
                dim_levels[j]++;
            if (dim_levels[j] == 1) {
                *lp++ = '[';
                dim_levels[j]++;
            }
            if (dim_levels[j] == 3) {
                *rp++ = ']';
                if (j != 0 && dim_levels[j] > dim_levels[j - 1]) {
                    *lp++ = '[';
                    dim_levels[j] = 2;
                } else
                    dim_levels[j] = 0;
            }
        }
        *lp = *rp = '\0';
        fprintf(stream, "%s", right_buf);
        if (*right_buf != '\0') {
            fprintf(stream, "\n");
            right_len = strlen(right_buf);
            for (k = ndim - right_len; k > 0; k--)
                fprintf(stream, " ");
        }
        fprintf(stream, "%s", left_buf);
        if (*left_buf == '\0')
            fprintf(stream, " ");
        tofu_fprintf(stream, fmt, tofu_padd(data, i, dsize), dtype);
        lp = left_buf, rp = right_buf;
    }
    for (j = 0; j < ndim; j++)
        fprintf(stream, "]");
    fprintf(stream, "\n");

    tofu_free(dim_sizes);
    tofu_free(dim_levels);
    tofu_free(left_buf);
    tofu_free(right_buf);
}

TOFU_EXPORT void tofu_tensor_print(const tofu_tensor *t, const char *fmt)
{
    tofu_tensor_fprint(stdout, t, fmt);
}

TOFU_EXPORT int tofu_tensor_save(const char *file_name, const tofu_tensor *t, const char *fmt)
{
    FILE *fp;

    fp = fopen(file_name, "w");
    if (!fp) {
        tofu_warn_ret("ERROR: cannot open %s", file_name);
        return -1;
    }
    tofu_tensor_fprint(fp, t, fmt);
    fclose(fp);
    return 0;
}

/* Create a tensor with heap-allocated data (safe for gradients) */
TOFU_EXPORT tofu_tensor* tofu_tensor_create_with_values(const float* values, int ndim, const int* dims)
{
    /* Calculate total elements */
    int len = 1;
    for (int i = 0; i < ndim; i++) {
        len *= dims[i];
    }
    
    /* Allocate data on heap */
    float* data = (float*)tofu_alloc(len * sizeof(float));
    memcpy(data, values, len * sizeof(float));
    
    /* Create tensor */
    tofu_tensor* t = tofu_tensor_create(data, ndim, dims, TOFU_FLOAT);
    t->owner = t;  /* This tensor owns its data */
    return t;
}
