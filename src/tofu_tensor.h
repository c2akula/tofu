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

/**
 * @brief Convert multi-dimensional coordinates to flat index
 * @param t Tensor (cannot be NULL)
 * @param coords Array of coordinates, length must be t->ndim
 * @return Flat index into tensor data array
 * @pre t and coords must not be NULL
 * @note Violating preconditions triggers assert() and crashes
 */
int tofu_tensor_index(const tofu_tensor *t, int *coords);

/**
 * @brief Convert flat index to multi-dimensional coordinates
 * @param t Tensor (cannot be NULL)
 * @param index Flat index into tensor data array
 * @param coords Output array for coordinates, length must be t->ndim
 * @pre t and coords must not be NULL
 * @pre index must be in range [0, tofu_tensor_size(t))
 * @note Violating preconditions triggers assert() and crashes
 */
void tofu_tensor_coords(const tofu_tensor *t, int index, int *coords);

/**
 * @brief Check if two tensors have the same shape
 * @param t1 First tensor (cannot be NULL)
 * @param t2 Second tensor (cannot be NULL)
 * @return 1 if same shape, 0 otherwise
 * @pre t1 and t2 must not be NULL
 * @note Violating preconditions triggers assert() and crashes
 */
int tofu_tensor_issameshape(const tofu_tensor *t1, const tofu_tensor *t2);

/**
 * @brief Check if two tensors can be broadcast together (NumPy semantics)
 * @param t1 First tensor (cannot be NULL)
 * @param t2 Second tensor (cannot be NULL)
 * @return 1 if broadcastable, 0 otherwise
 * @pre t1 and t2 must not be NULL
 * @note Follows NumPy broadcasting rules:
 *       - Arrays with fewer dimensions are prepended with size-1 dimensions
 *       - Size-1 dimensions are stretched to match the other array
 *       - Dimensions must match or one must be 1
 * @note Violating preconditions triggers assert() and crashes
 */
int tofu_tensor_isbroadcastable(const tofu_tensor *t1, const tofu_tensor *t2);

/**
 * @brief Create a tensor with existing data buffer
 * @param data Pointer to data buffer (cannot be NULL)
 * @param ndim Number of dimensions (must be > 0 and <= TOFU_MAXDIM)
 * @param dims Array of dimension sizes, length must be ndim
 * @param dtype Data type (TOFU_FLOAT, TOFU_INT32, etc.)
 * @return Pointer to newly allocated tensor (caller owns, must call tofu_tensor_free)
 * @pre data, dims must not be NULL; ndim > 0 and <= TOFU_MAXDIM
 * @note The tensor does NOT take ownership of the data buffer
 * @note Caller must manage data lifetime and free both tensor and data
 * @note If tensor is passed to tofu_graph_param(), the graph takes ownership
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_free
 * @see tofu_tensor_zeros for allocating data buffer automatically
 */
tofu_tensor *tofu_tensor_create(void *data, int ndim, const int *dims, tofu_dtype dtype);

/**
 * @brief Free tensor structure (does NOT free data buffer)
 * @param t Tensor to free (can be NULL, no-op if NULL)
 * @note Does NOT free the data buffer - caller must free data separately
 * @note Do NOT call if tensor was passed to tofu_graph_param() (graph owns it)
 * @see tofu_tensor_free_data_too to free both tensor and data
 */
void tofu_tensor_free(tofu_tensor *t);

/**
 * @brief Free both tensor structure and data buffer
 * @param t Tensor to free (can be NULL, no-op if NULL)
 * @note Frees both the tensor and its associated data buffer
 * @note Only use if tensor owns its data (created with tofu_tensor_zeros, etc.)
 * @note Do NOT call if tensor was created with tofu_tensor_create (caller owns data)
 * @note Do NOT call if tensor was passed to tofu_graph_param() (graph owns it)
 */
void tofu_tensor_free_data_too(tofu_tensor *t);

/**
 * @brief Get total number of elements in tensor
 * @param t Tensor (cannot be NULL)
 * @return Total element count (product of all dimensions)
 * @pre t must not be NULL
 * @note Violating preconditions triggers assert() and crashes
 */
size_t tofu_tensor_size(tofu_tensor *t);

/**
 * @brief Create a tensor initialized to zeros with allocated data buffer
 * @param ndim Number of dimensions (must be > 0 and <= TOFU_MAXDIM)
 * @param dims Array of dimension sizes, length must be ndim
 * @param dtype Data type (TOFU_FLOAT, TOFU_INT32, etc.)
 * @return Pointer to newly allocated tensor (caller owns, must call tofu_tensor_free_data_too)
 * @pre dims must not be NULL; ndim > 0 and <= TOFU_MAXDIM
 * @note This allocates both tensor structure and data buffer
 * @note Caller must call tofu_tensor_free_data_too to free both
 * @note If tensor is passed to tofu_graph_param(), the graph takes ownership
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_free_data_too
 * @see tofu_tensor_create if you want to manage data buffer yourself
 */
tofu_tensor *tofu_tensor_zeros(int ndim, const int *dims, tofu_dtype dtype);

/**
 * @brief Create a deep copy of a tensor
 * @param src Source tensor to clone (cannot be NULL)
 * @return Pointer to newly allocated tensor (caller owns, must call tofu_tensor_free_data_too)
 * @pre src must not be NULL
 * @note Creates both new tensor structure and new data buffer
 * @note Copies all data from source to new tensor
 * @note Caller must call tofu_tensor_free_data_too to free clone
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_free_data_too
 */
tofu_tensor *tofu_tensor_clone(const tofu_tensor *src);

/**
 * @brief Create a tensor by repeating data multiple times
 * @param src Source tensor to repeat (cannot be NULL)
 * @param times Number of repetitions (must be > 0)
 * @return Pointer to newly allocated tensor (caller owns, must call tofu_tensor_free_data_too)
 * @pre src must not be NULL; times > 0
 * @note Creates new tensor with size = src->len * times
 * @note Caller must call tofu_tensor_free_data_too to free result
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_free_data_too
 */
tofu_tensor *tofu_tensor_repeat(const tofu_tensor *src, int times);

/**
 * @brief Create a 1-D tensor with evenly spaced values (similar to NumPy arange)
 * @param start Starting value (inclusive)
 * @param stop Ending value (exclusive)
 * @param step Step size between values
 * @param dtype Data type for the resulting tensor
 * @return Pointer to newly allocated 1-D tensor (caller owns, must call tofu_tensor_free_data_too)
 * @pre step must not be zero; (stop-start)/step must be positive
 * @note Creates values [start, start+step, start+2*step, ..., stop)
 * @note Number of elements = ceil((stop - start) / step)
 * @note Caller must call tofu_tensor_free_data_too to free result
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_rearange for in-place filling
 */
tofu_tensor *tofu_tensor_arange(double start, double stop, double step, tofu_dtype dtype);

/**
 * @brief Fill existing tensor with evenly spaced values (in-place arange)
 * @param src Tensor to fill (cannot be NULL)
 * @param start Starting value (inclusive)
 * @param stop Ending value (exclusive)
 * @param step Step size between values
 * @pre src must not be NULL; step must not be zero
 * @note Fills tensor with [start, start+step, start+2*step, ...]
 * @note Number of values written is min(tensor size, ceil((stop-start)/step))
 * @note Modifies tensor data in-place
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_arange for allocating new tensor
 */
void tofu_tensor_rearange(tofu_tensor *src, double start, double stop, double step);

/**
 * @brief Print tensor to file stream with custom format
 * @param stream File stream to write to (cannot be NULL)
 * @param t Tensor to print (cannot be NULL)
 * @param fmt Format string for each element (e.g., "%.6f", "%d")
 * @pre stream and t must not be NULL
 * @note Prints multi-dimensional structure with brackets
 * @note Uses specified format for numeric values
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_print for printing to stdout
 */
void tofu_tensor_fprint(FILE *stream, const tofu_tensor *t, const char *fmt);

/**
 * @brief Print tensor to stdout with custom format
 * @param t Tensor to print (cannot be NULL)
 * @param fmt Format string for each element (e.g., "%.6f", "%d")
 * @pre t must not be NULL
 * @note Convenience wrapper for tofu_tensor_fprint(stdout, t, fmt)
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_fprint for printing to arbitrary stream
 */
void tofu_tensor_print(const tofu_tensor *t, const char *fmt);

/**
 * @brief Save tensor to file with custom format
 * @param file_name Path to output file (cannot be NULL)
 * @param t Tensor to save (cannot be NULL)
 * @param fmt Format string for each element (e.g., "%.6f", "%d")
 * @return 0 on success, non-zero on error
 * @pre file_name and t must not be NULL
 * @note Creates or overwrites file at specified path
 * @note Uses same format as tofu_tensor_print
 * @note Violating preconditions triggers assert() and crashes
 */
int tofu_tensor_save(const char *file_name, const tofu_tensor *t, const char *fmt);

/**
 * @brief Create tensor descriptor for slice with existing data buffer
 * @param data Pointer to slice data buffer (cannot be NULL)
 * @param src Source tensor defining shape (cannot be NULL)
 * @param axis Axis along which to slice
 * @param len Length of slice along specified axis
 * @param dtype Data type for slice
 * @return Pointer to newly allocated tensor descriptor (caller owns, must call tofu_tensor_free)
 * @pre data and src must not be NULL; axis < src->ndim; len > 0
 * @note Does NOT copy data, only creates tensor structure
 * @note Caller manages data buffer lifetime
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_slice for copying slice data
 */
tofu_tensor *tofu_tensor_create_slice(void *data, const tofu_tensor *src, int axis, int len,
                                  tofu_dtype dtype);

/**
 * @brief Create zero-initialized tensor matching slice shape
 * @param src Source tensor defining shape (cannot be NULL)
 * @param axis Axis along which slice would be taken
 * @param len Length of slice along specified axis
 * @param dtype Data type for new tensor
 * @return Pointer to newly allocated zero-filled tensor (caller owns, must call tofu_tensor_free_data_too)
 * @pre src must not be NULL; axis < src->ndim; len > 0
 * @note Allocates new data buffer filled with zeros
 * @note Caller must call tofu_tensor_free_data_too to free result
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_zeros_slice(const tofu_tensor *src, int axis, int len, tofu_dtype dtype);

/**
 * @brief Extract slice from tensor (copies data)
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param axis Axis along which to slice
 * @param start Starting index along axis
 * @param len Length of slice
 * @return Result tensor (caller owns if dst was NULL)
 * @pre src must not be NULL; axis < src->ndim
 * @pre start >= 0 and start + len <= src->dims[axis]
 * @note Copies slice data to destination
 * @note If dst is NULL, allocates new tensor with correct shape
 * @note If dst is non-NULL, it must have correct shape for slice
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_slice_nocopy for view without copying
 */
tofu_tensor *tofu_tensor_slice(const tofu_tensor *src, tofu_tensor *dst, int axis, int start, int len);

/**
 * @brief Create view of tensor slice (no data copy, view operation)
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param axis Axis along which to slice
 * @param start Starting index along axis
 * @param len Length of slice
 * @return Result tensor sharing data with source (caller owns if dst was NULL)
 * @pre src must not be NULL; axis < src->ndim
 * @pre start >= 0 and start + len <= src->dims[axis]
 * @note Does NOT copy data - result shares memory with source
 * @note Modifying result will modify source tensor
 * @note Source must outlive result tensor
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_slice for copying slice data
 */
tofu_tensor *tofu_tensor_slice_nocopy(tofu_tensor *src, tofu_tensor *dst, int axis, int start, int len);

/**
 * @brief Concatenate two tensors along specified axis
 * @param src1 First tensor (cannot be NULL)
 * @param src2 Second tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param axis Axis along which to concatenate
 * @return Result tensor (caller owns if dst was NULL)
 * @pre src1 and src2 must not be NULL; axis < src1->ndim
 * @pre All dimensions except axis must match between src1 and src2
 * @note Result dims[axis] = src1->dims[axis] + src2->dims[axis]
 * @note If dst is NULL, allocates new tensor with correct shape
 * @note If dst is non-NULL, it must have correct shape
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_concat(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst, int axis);

/**
 * @brief Reshape tensor to new dimensions (view operation, no data copy)
 * @param src Source tensor (cannot be NULL)
 * @param ndim Number of dimensions for reshaped tensor
 * @param dims Array of new dimension sizes
 * @return New tensor structure sharing data with source (caller owns, must call tofu_tensor_free)
 * @pre src and dims must not be NULL; ndim > 0
 * @pre Product of dims must equal tofu_tensor_size(src)
 * @note Does NOT copy data - result shares memory with source
 * @note Source must outlive result tensor
 * @note Only changes shape metadata, not data layout
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_reshape_src for in-place reshape
 */
tofu_tensor *tofu_tensor_reshape(tofu_tensor *src, int ndim, const int *dims);

/**
 * @brief Reshape tensor in-place (modifies source tensor metadata)
 * @param src Tensor to reshape (cannot be NULL)
 * @param ndim Number of dimensions for reshaped tensor
 * @param dims Array of new dimension sizes
 * @pre src and dims must not be NULL; ndim > 0
 * @pre Product of dims must equal tofu_tensor_size(src)
 * @note Modifies src tensor structure in-place
 * @note Does NOT copy or reallocate data
 * @note Only changes shape metadata
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_reshape for creating new view
 */
void tofu_tensor_reshape_src(tofu_tensor *src, int ndim, const int *dims);

/**
 * @brief Reduce tensor along axis using max operation
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param arg Argmax indices tensor (can be NULL if indices not needed)
 * @param axis Axis along which to reduce
 * @return Result tensor with dims[axis] removed (caller owns if dst was NULL)
 * @pre src must not be NULL; axis < src->ndim
 * @note Output shape: src->dims with dims[axis] removed
 * @note If arg is non-NULL, fills it with indices of maximum values
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_sumreduce, tofu_tensor_meanreduce
 */
tofu_tensor *tofu_tensor_maxreduce(const tofu_tensor *src, tofu_tensor *dst, tofu_tensor *arg, int axis);

/**
 * @brief Reduce tensor along axis using sum operation
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param axis Axis along which to reduce
 * @return Result tensor with dims[axis] removed (caller owns if dst was NULL)
 * @pre src must not be NULL; axis < src->ndim
 * @note Output shape: src->dims with dims[axis] removed
 * @note Computes sum of all elements along specified axis
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_meanreduce, tofu_tensor_maxreduce
 */
tofu_tensor *tofu_tensor_sumreduce(const tofu_tensor *src, tofu_tensor *dst, int axis);

/**
 * @brief Reduce tensor along axis using mean operation
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param axis Axis along which to reduce
 * @return Result tensor with dims[axis] removed (caller owns if dst was NULL)
 * @pre src must not be NULL; axis < src->ndim
 * @note Output shape: src->dims with dims[axis] removed
 * @note Computes arithmetic mean of all elements along specified axis
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_sumreduce, tofu_tensor_maxreduce
 */
tofu_tensor *tofu_tensor_meanreduce(const tofu_tensor *src, tofu_tensor *dst, int axis);

/**
 * @brief Subtract reduced tensor from source with broadcasting
 * @param src Source tensor (cannot be NULL)
 * @param reduced Reduced tensor to subtract (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param axis Axis along which reduction was performed
 * @return Result tensor with same shape as src (caller owns if dst was NULL)
 * @pre src and reduced must not be NULL; axis < src->ndim
 * @pre reduced->ndim = src->ndim - 1 (one dimension removed)
 * @note Broadcasts reduced tensor back along axis and subtracts
 * @note Useful for normalization operations (subtract mean, etc.)
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_sub_broadcast(const tofu_tensor *src, const tofu_tensor *reduced, tofu_tensor *dst, int axis);

/**
 * @brief Apply softmax activation along specified axis
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param axis Axis along which to apply softmax
 * @return Result tensor with same shape as src (caller owns if dst was NULL)
 * @pre src must not be NULL; axis < src->ndim
 * @note Computes exp(x_i) / sum(exp(x_j)) along axis
 * @note Uses numerically stable implementation (subtracts max before exp)
 * @note Output values sum to 1.0 along specified axis
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_softmax(const tofu_tensor *src, tofu_tensor *dst, int axis);

/**
 * @brief Apply layer normalization with learnable affine transform
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param gamma Scale parameter tensor (can be NULL for no scaling)
 * @param beta Shift parameter tensor (can be NULL for no shift)
 * @param axis Axis along which to normalize
 * @param eps Small constant for numerical stability (typically 1e-5)
 * @return Result tensor with same shape as src (caller owns if dst was NULL)
 * @pre src must not be NULL; axis < src->ndim; eps > 0
 * @note Normalizes: (x - mean) / sqrt(variance + eps)
 * @note Then applies: gamma * normalized + beta (if gamma/beta non-NULL)
 * @note If gamma/beta are NULL, only normalization is applied
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_layer_norm(const tofu_tensor *src, tofu_tensor *dst,
                                const tofu_tensor *gamma, const tofu_tensor *beta,
                                int axis, double eps);

/**
 * @brief Apply element-wise binary operation with broadcasting
 * @param src1 First tensor (cannot be NULL)
 * @param src2 Second tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param elew_op Operation to apply (ADD, SUB, MUL, DIV, etc.)
 * @return Result tensor (caller owns if dst was NULL)
 * @pre src1 and src2 must not be NULL
 * @pre src1 and src2 must be broadcastable (NumPy rules)
 * @note Supports NumPy-style broadcasting
 * @note Operations: ADD (+), SUB (-), MUL (*), DIV (/), POW (^), etc.
 * @note If dst is NULL, allocates new tensor with broadcast shape
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_elew_param for tensor-scalar operations
 */
tofu_tensor *tofu_tensor_elew(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst,
                          tofu_elew_op elew_op);

/**
 * @brief Apply element-wise operation between tensor and scalar
 * @param src Source tensor (cannot be NULL)
 * @param param Scalar parameter
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param elew_op Operation to apply (ADD, SUB, MUL, DIV, etc.)
 * @return Result tensor with same shape as src (caller owns if dst was NULL)
 * @pre src must not be NULL
 * @note Applies operation element-wise: op(tensor_element, param)
 * @note Operations: ADD (+), SUB (-), MUL (*), DIV (/), POW (^), etc.
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_elew for tensor-tensor operations
 */
tofu_tensor *tofu_tensor_elew_param(const tofu_tensor *src, double param, tofu_tensor *dst,
                                tofu_elew_op elew_op);
/**
 * @brief Compute inner product (sum-product over last axes)
 * @param src1 First tensor (cannot be NULL)
 * @param src2 Second tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @return Result tensor (caller owns if dst was NULL)
 * @pre src1 and src2 must not be NULL
 * @pre src1->dims[src1->ndim-1] must equal src2->dims[src2->ndim-1]
 * @note Behavior:
 *       - 1-D × 1-D: Dot product → scalar
 *       - 2-D × 2-D: result[i,j] = sum(a[i,:] * b[j,:])
 *       - N-D × N-D: Cartesian product of non-last dimensions
 * @note Output shape: (*a.shape[:-1], *b.shape[:-1])
 * @note If dst is NULL, caller must call tofu_tensor_free_data_too
 * @note If dst is non-NULL, it must have the correct shape
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_matmul for matrix multiplication (different semantics)
 */
tofu_tensor *tofu_tensor_inner(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst);

/**
 * @brief Compute matrix multiplication with broadcasting
 * @param src1 Left operand tensor (cannot be NULL)
 * @param src2 Right operand tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @return Result tensor (caller owns if dst was NULL)
 * @pre src1 and src2 must not be NULL
 * @pre src1->dims[src1->ndim-1] must equal src2->dims[src2->ndim-2]
 * @note Behavior:
 *       - 1-D @ 1-D: Dot product → scalar
 *       - 2-D @ 2-D: Standard matrix multiplication
 *       - N-D @ 1-D: Matrix-vector (drops last dim)
 *       - 1-D @ N-D: Vector-matrix (drops first dim)
 *       - N-D @ N-D: Batch matmul with broadcasting
 * @note Broadcasts batch dimensions, contracts last of a with second-to-last of b
 * @note Most commonly used operation for neural networks
 * @note If dst is NULL, caller must call tofu_tensor_free_data_too
 * @note If dst is non-NULL, it must have the correct shape
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_inner for inner product (different semantics)
 */
tofu_tensor *tofu_tensor_matmul(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst);

/**
 * @brief Compute outer product (cartesian product without summation)
 * @param src1 First tensor (cannot be NULL)
 * @param src2 Second tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @return Result tensor (caller owns if dst was NULL)
 * @pre src1 and src2 must not be NULL
 * @note Behavior:
 *       - Flattens both input tensors
 *       - Computes: result[i,j] = a[i] * b[j]
 *       - Always produces 2-D output
 * @note Output shape: [a.size, b.size] where size is total element count
 * @note If dst is NULL, caller must call tofu_tensor_free_data_too
 * @note If dst is non-NULL, it must have the correct shape
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_outer(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst);

/**
 * @brief Transpose tensor by permuting dimensions
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param axes Permutation array (can be NULL for reverse order)
 * @return Result tensor (caller owns if dst was NULL)
 * @pre src must not be NULL
 * @pre If axes is non-NULL, it must be a permutation of [0, 1, ..., ndim-1]
 * @note If axes is NULL, reverses dimension order (e.g., [2,3,4] → [4,3,2])
 * @note If axes is non-NULL, permutes according to axes (e.g., axes=[1,0] swaps dims)
 * @note For 2-D matrix, axes=NULL transposes (rows ↔ columns)
 * @note If dst is NULL, caller must call tofu_tensor_free_data_too
 * @note If dst is non-NULL, it must have the correct shape
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_transpose(const tofu_tensor *src, tofu_tensor *dst, const int *axes);

/**
 * @brief Apply Leaky ReLU activation function
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param negslope Slope for negative values (typically 0.01)
 * @return Result tensor with same shape as src (caller owns if dst was NULL)
 * @pre src must not be NULL
 * @note Computes: x if x >= 0, else negslope * x
 * @note Standard ReLU equivalent when negslope = 0
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_lrelu(const tofu_tensor *src, tofu_tensor *dst, float negslope);

/**
 * @brief Convert tensor to different data type
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param dtype_d Target data type (TOFU_FLOAT, TOFU_INT32, etc.)
 * @return Result tensor with same shape as src but different dtype (caller owns if dst was NULL)
 * @pre src must not be NULL
 * @note Converts each element to target type with appropriate casting
 * @note May lose precision (e.g., float to int truncates)
 * @note If dst is NULL, allocates new tensor with target dtype
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_convert(const tofu_tensor *src, tofu_tensor *dst, tofu_dtype dtype_d);

/**
 * @brief Resize tensor using interpolation (for image-like data)
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param new_dims Array of new dimension sizes
 * @param rtype Resize type (NEAREST, BILINEAR, etc.)
 * @return Result tensor with new dimensions (caller owns if dst was NULL)
 * @pre src and new_dims must not be NULL
 * @note Typically used for resizing image tensors
 * @note Supports various interpolation methods
 * @note If dst is NULL, allocates new tensor with new dimensions
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_resize(const tofu_tensor *src, tofu_tensor *dst, const int *new_dims,
                            tofu_resize_type rtype);

/**
 * @brief Subtract mean values from tensor (normalization)
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param mean Array of mean values to subtract, one per channel
 * @return Result tensor with same shape as src (caller owns if dst was NULL)
 * @pre src and mean must not be NULL
 * @note Common preprocessing step for neural networks
 * @note Subtracts mean[i] from each element in channel i
 * @note If dst is NULL, allocates new tensor
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_submean(const tofu_tensor *src, tofu_tensor *dst, const double *mean);

/**
 * @brief Broadcast tensor to specified shape (NumPy semantics)
 * @param src Source tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param ndim Number of dimensions for target shape
 * @param dims Target dimension sizes
 * @return Result tensor with target shape (caller owns if dst was NULL)
 * @pre src and dims must not be NULL; ndim > 0
 * @pre src must be broadcastable to target shape (NumPy rules)
 * @note Follows NumPy broadcasting rules
 * @note Size-1 dimensions are stretched to match target
 * @note If dst is NULL, allocates new tensor with target shape
 * @note Violating preconditions triggers assert() and crashes
 */
tofu_tensor *tofu_tensor_broadcast_to(const tofu_tensor *src, tofu_tensor *dst, int ndim, const int *dims);

/**
 * @brief Apply element-wise operation with automatic broadcasting
 * @param src1 First tensor (cannot be NULL)
 * @param src2 Second tensor (cannot be NULL)
 * @param dst Destination tensor (can be NULL to allocate new)
 * @param elew_op Operation to apply (ADD, SUB, MUL, DIV, etc.)
 * @return Result tensor with broadcast shape (caller owns if dst was NULL)
 * @pre src1 and src2 must not be NULL
 * @pre src1 and src2 must be broadcastable (NumPy rules)
 * @note Automatically broadcasts inputs to compatible shape
 * @note Equivalent to tofu_tensor_elew but with explicit broadcast handling
 * @note If dst is NULL, allocates new tensor with broadcast shape
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_elew for basic element-wise operations
 */
tofu_tensor *tofu_tensor_elew_broadcast(const tofu_tensor *src1, const tofu_tensor *src2, tofu_tensor *dst,
                               tofu_elew_op elew_op);


#ifdef __cplusplus
TOFU_CPPEND
#endif

#endif /* _TOFU_TENSOR_H_ */

/**
 * @brief Create tensor with heap-allocated copy of provided values
 * @param values Array of initial values (cannot be NULL)
 * @param ndim Number of dimensions (must be > 0 and <= TOFU_MAXDIM)
 * @param dims Array of dimension sizes, length must be ndim
 * @return Pointer to newly allocated tensor with copied data (caller owns, must call tofu_tensor_free_data_too)
 * @pre values and dims must not be NULL; ndim > 0 and <= TOFU_MAXDIM
 * @note IMPORTANT: Creates heap-allocated copy of values (safe for gradients)
 * @note DO NOT use compound literals like (float[]){1.0f} as they create stack memory
 * @note Number of values must match product of dims
 * @note Caller must call tofu_tensor_free_data_too to free both tensor and data
 * @note If tensor is passed to tofu_graph_param(), the graph takes ownership
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_create for wrapping existing data buffer
 * @see tofu_tensor_zeros for zero-initialized tensor
 */
TOFU_EXPORT tofu_tensor* tofu_tensor_create_with_values(const float* values, int ndim, const int* dims);

