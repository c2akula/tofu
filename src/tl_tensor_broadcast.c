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

#include "tl_tensor_internal.h"

/**
 * Checks if two tensors are broadcastable to each other
 * 
 * Broadcasting rules (compatible with NumPy):
 * 1. Arrays with fewer dimensions are prepended with dimensions of size 1
 * 2. Output dimensions are the max of the input dimensions
 * 3. An input can be broadcast to an output if for each dimension:
 *    - its size matches the output dimension, or
 *    - its size is 1 (which is then broadcast to match the output)
 */
TL_EXPORT int tl_tensor_isbroadcastable(const tl_tensor *t1, const tl_tensor *t2)
{
    int i, pos1, pos2;
    int max_ndim = (t1->ndim > t2->ndim) ? t1->ndim : t2->ndim;
    
    for (i = 0; i < max_ndim; i++) {
        pos1 = t1->ndim - 1 - i;
        pos2 = t2->ndim - 1 - i;
        
        // If we've exhausted dimensions in one tensor, it's broadcasted from 1
        int dim1 = (pos1 >= 0) ? t1->dims[pos1] : 1;
        int dim2 = (pos2 >= 0) ? t2->dims[pos2] : 1;
        
        // Dimensions must be equal or one of them should be 1 for broadcasting
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return 0;
        }
    }
    
    return 1;
}

/**
 * Calculate output dimensions after broadcasting two tensors
 */
static void tl_compute_broadcast_dims(const tl_tensor *t1, const tl_tensor *t2, int *out_ndim, int *out_dims)
{
    int i, pos1, pos2;
    int max_ndim = (t1->ndim > t2->ndim) ? t1->ndim : t2->ndim;
    
    *out_ndim = max_ndim;
    
    for (i = 0; i < max_ndim; i++) {
        pos1 = t1->ndim - 1 - i;
        pos2 = t2->ndim - 1 - i;
        
        // If we've exhausted dimensions in one tensor, it's broadcasted from 1
        int dim1 = (pos1 >= 0) ? t1->dims[pos1] : 1;
        int dim2 = (pos2 >= 0) ? t2->dims[pos2] : 1;
        
        // The output dimension is the maximum
        out_dims[max_ndim - 1 - i] = (dim1 > dim2) ? dim1 : dim2;
    }
}

/**
 * Computes broadcasting strides for a tensor
 */
static void tl_compute_broadcast_strides(const tl_tensor *t, int out_ndim, const int *out_dims, int *strides)
{
    int i, pos;
    
    // Initialize all strides to 0 (meaning a dimension is broadcasted)
    for (i = 0; i < out_ndim; i++) {
        strides[i] = 0;
    }
    
    // Calculate actual strides only for non-broadcast dimensions
    int stride = 1;
    for (i = out_ndim - 1; i >= 0; i--) {
        pos = t->ndim - (out_ndim - i);
        
        if (pos >= 0) {
            if (t->dims[pos] == out_dims[i]) {
                strides[i] = stride;
            } else if (t->dims[pos] == 1) {
                // This dimension is being broadcasted, stride remains 0
                strides[i] = 0;
            }
            stride *= t->dims[pos];
        }
    }
}

/**
 * Broadcasts a tensor to the specified shape
 */
TL_EXPORT tl_tensor *tl_tensor_broadcast_to(const tl_tensor *src, tl_tensor *dst, int ndim, const int *dims)
{
    int i;
    
    // Validate inputs
    tl_check_tensor(src);
    assert(ndim > 0 && ndim <= TL_MAXDIM);
    for (i = 0; i < ndim; i++) {
        assert(dims[i] > 0);
    }
    
    // Check if broadcasting is possible
    int src_pos;
    for (i = 0; i < ndim; i++) {
        src_pos = src->ndim - (ndim - i);
        if (src_pos >= 0) {
            if (src->dims[src_pos] != 1 && src->dims[src_pos] != dims[i]) {
                tl_warn_ret("Cannot broadcast from shape %d to %d at dim %d", 
                           src->dims[src_pos], dims[i], i);
                return NULL;
            }
        }
    }
    
    // Create output tensor if not provided
    if (dst == NULL) {
        dst = tl_tensor_zeros(ndim, dims, src->dtype);
    } else {
        assert(dst->ndim == ndim);
        for (i = 0; i < ndim; i++) {
            assert(dst->dims[i] == dims[i]);
        }
    }
    
    // Compute broadcasting strides
    int *src_strides = (int *)tl_alloc(sizeof(int) * ndim);
    tl_compute_broadcast_strides(src, ndim, dims, src_strides);
    
    // Perform the broadcasting
    size_t dsize = tl_size_of(src->dtype);
    int *dst_coords = (int *)tl_alloc(sizeof(int) * ndim);
    int *src_coords = (int *)tl_alloc(sizeof(int) * src->ndim);
    
    for (i = 0; i < dst->len; i++) {
        // Calculate destination coordinates
        tl_tensor_coords(dst, i, dst_coords);
        
        // Calculate source index based on broadcast strides
        int src_idx = 0;
        for (int j = 0; j < ndim; j++) {
            if (src_strides[j] > 0) {
                src_idx += dst_coords[j] * src_strides[j];
            }
        }
        
        // Copy the data
        tl_passign(dst->data, i, src->data, src_idx, dsize);
    }
    
    tl_free(src_strides);
    tl_free(dst_coords);
    tl_free(src_coords);
    
    return dst;
}

/**
 * Element-wise operation with broadcasting support
 */
TL_EXPORT tl_tensor *tl_tensor_elew_broadcast(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst,
                                         tl_elew_op elew_op)
{
    int i;
    
    // Validate inputs
    tl_check_tensor(src1);
    tl_check_tensor(src2);
    tl_check_elew_op(elew_op);
    assert(src1->dtype == src2->dtype);
    
    // Check if broadcasting is possible
    if (!tl_tensor_isbroadcastable(src1, src2)) {
        tl_warn_ret("Tensors are not broadcastable");
        return NULL;
    }
    
    // Calculate output dimensions
    int out_ndim;
    int *out_dims = (int *)tl_alloc(sizeof(int) * TL_MAXDIM);
    tl_compute_broadcast_dims(src1, src2, &out_ndim, out_dims);
    
    // Create output tensor if not provided
    if (dst == NULL) {
        dst = tl_tensor_zeros(out_ndim, out_dims, src1->dtype);
    } else {
        assert(dst->ndim == out_ndim);
        for (i = 0; i < out_ndim; i++) {
            assert(dst->dims[i] == out_dims[i]);
        }
        assert(dst->dtype == src1->dtype);
    }
    
    // Compute broadcasting strides
    int *src1_strides = (int *)tl_alloc(sizeof(int) * out_ndim);
    int *src2_strides = (int *)tl_alloc(sizeof(int) * out_ndim);
    tl_compute_broadcast_strides(src1, out_ndim, out_dims, src1_strides);
    tl_compute_broadcast_strides(src2, out_ndim, out_dims, src2_strides);
    
    // Perform the element-wise operation with broadcasting
    size_t dsize = tl_size_of(src1->dtype);
    int *dst_coords = (int *)tl_alloc(sizeof(int) * out_ndim);
    void *src1_val = tl_alloc(dsize);
    void *src2_val = tl_alloc(dsize);
    void *dst_val = tl_alloc(dsize);
    tl_elew_func elew = tl_elew_getfunc(src1->dtype);
    
    for (i = 0; i < dst->len; i++) {
        // Calculate destination coordinates
        tl_tensor_coords(dst, i, dst_coords);
        
        // Calculate source indices based on broadcast strides
        int src1_idx = 0;
        int src2_idx = 0;
        
        for (int j = 0; j < out_ndim; j++) {
            src1_idx += dst_coords[j] * src1_strides[j];
            src2_idx += dst_coords[j] * src2_strides[j];
        }
        
        // Perform the element-wise operation
        void *src1_ptr = tl_padd(src1->data, src1_idx, dsize);
        void *src2_ptr = tl_padd(src2->data, src2_idx, dsize);
        
        elew(src1_ptr, src2_ptr, dst_val, elew_op);
        tl_passign(dst->data, i, dst_val, 0, dsize);
    }
    
    tl_free(out_dims);
    tl_free(src1_strides);
    tl_free(src2_strides);
    tl_free(dst_coords);
    tl_free(src1_val);
    tl_free(src2_val);
    tl_free(dst_val);
    
    return dst;
}