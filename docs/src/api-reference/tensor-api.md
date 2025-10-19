# Tensor API Reference

The Tensor API provides the core data structures and operations for Tofu. Tensors are multi-dimensional arrays that support automatic differentiation when used with the Graph API.

## Table of Contents

- [Data Structures](#data-structures)
- [Creation Functions](#creation-functions)
- [Shape Operations](#shape-operations)
- [Mathematical Operations](#mathematical-operations)
- [Element-wise Operations](#element-wise-operations)
- [Reductions](#reductions)
- [Activation Functions](#activation-functions)
- [Utilities](#utilities)

---

## Data Structures

### `tofu_tensor`

The core tensor structure representing a multi-dimensional array.

```c
struct tofu_tensor {
    tofu_dtype dtype;           // Data type (TOFU_FLOAT, TOFU_INT32, etc.)
    int len;                    // Total number of elements
    int ndim;                   // Number of dimensions
    int *dims;                  // Array of dimension sizes
    void *data;                 // Pointer to data buffer
    struct tofu_tensor *owner;  // Data owner (NULL if self-owned)
    void *backend_data;         // Backend-specific data
};
```

### Data Types (`tofu_dtype`)

Supported tensor data types:

- `TOFU_FLOAT` - 32-bit floating point (most common for neural networks)
- `TOFU_DOUBLE` - 64-bit floating point
- `TOFU_INT32` - 32-bit signed integer
- `TOFU_INT64` - 64-bit signed integer
- `TOFU_INT16` - 16-bit signed integer
- `TOFU_INT8` - 8-bit signed integer
- `TOFU_UINT32` - 32-bit unsigned integer
- `TOFU_UINT64` - 64-bit unsigned integer
- `TOFU_UINT16` - 16-bit unsigned integer
- `TOFU_UINT8` - 8-bit unsigned integer
- `TOFU_BOOL` - Boolean type

### Element-wise Operations (`tofu_elew_op`)

- `TOFU_MUL` - Multiplication (*)
- `TOFU_DIV` - Division (/)
- `TOFU_SUM` - Addition (+)
- `TOFU_SUB` - Subtraction (-)
- `TOFU_MAX` - Element-wise maximum
- `TOFU_MIN` - Element-wise minimum
- `TOFU_POW` - Power (^)

---

## Creation Functions

### `tofu_tensor_create`

Create a tensor with an existing data buffer.

```c
tofu_tensor *tofu_tensor_create(void *data, int ndim, const int *dims, tofu_dtype dtype);
```

**Parameters:**
- `data` - Pointer to data buffer (cannot be NULL)
- `ndim` - Number of dimensions (must be > 0 and <= TOFU_MAXDIM = 8)
- `dims` - Array of dimension sizes, length must be `ndim`
- `dtype` - Data type (TOFU_FLOAT, TOFU_INT32, etc.)

**Returns:** Pointer to newly allocated tensor (caller owns, must call `tofu_tensor_free`)

**Ownership:**
- The tensor does NOT take ownership of the data buffer
- Caller must manage data lifetime and free both tensor and data separately
- Even when passed to `tofu_graph_param()`, caller still owns tensor

**Example:**
```c
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
int dims[] = {2, 2};
tofu_tensor *t = tofu_tensor_create(data, 2, dims, TOFU_FLOAT);

// Use tensor...

tofu_tensor_free(t);  // Free tensor structure
// data is still valid - free manually if needed
```

**Notes:**
- Typical pattern: create tensor → use in graph → `tofu_graph_free()` → `tofu_tensor_free()` → free data
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_tensor_zeros`, `tofu_tensor_create_with_values`

---

### `tofu_tensor_create_with_values`

Create a tensor with heap-allocated copy of provided values.

```c
tofu_tensor *tofu_tensor_create_with_values(const float *values, int ndim, const int *dims);
```

**Parameters:**
- `values` - Array of initial values (cannot be NULL)
- `ndim` - Number of dimensions (must be > 0 and <= TOFU_MAXDIM)
- `dims` - Array of dimension sizes, length must be `ndim`

**Returns:** Pointer to newly allocated tensor with copied data (caller owns, must call `tofu_tensor_free_data_too`)

**Important:**
- Creates heap-allocated copy of values (safe for gradients)
- DO NOT use compound literals like `(float[]){1.0f}` as they create stack memory
- Number of values must match product of dims
- Caller must call `tofu_tensor_free_data_too` to free both tensor and data

**Example:**
```c
float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
int dims[] = {2, 2};
tofu_tensor *t = tofu_tensor_create_with_values(values, 2, dims);

// Use tensor...

tofu_tensor_free_data_too(t);  // Free both tensor and data
```

---

### `tofu_tensor_zeros`

Create a zero-initialized tensor with allocated data buffer.

```c
tofu_tensor *tofu_tensor_zeros(int ndim, const int *dims, tofu_dtype dtype);
```

**Parameters:**
- `ndim` - Number of dimensions (must be > 0 and <= TOFU_MAXDIM)
- `dims` - Array of dimension sizes, length must be `ndim`
- `dtype` - Data type (TOFU_FLOAT, TOFU_INT32, etc.)

**Returns:** Pointer to newly allocated zero-filled tensor (caller owns, must call `tofu_tensor_free_data_too`)

**Ownership:**
- Allocates both tensor structure and data buffer
- Caller must call `tofu_tensor_free_data_too` to free both
- Even when passed to `tofu_graph_param()`, caller still owns tensor

**Example:**
```c
int dims[] = {3, 4};
tofu_tensor *t = tofu_tensor_zeros(2, dims, TOFU_FLOAT);

// All elements are 0.0f
// Use tensor...

tofu_tensor_free_data_too(t);  // Free both tensor and data
```

**See also:** `tofu_tensor_create`, `tofu_tensor_clone`

---

### `tofu_tensor_clone`

Create a deep copy of a tensor.

```c
tofu_tensor *tofu_tensor_clone(const tofu_tensor *src);
```

**Parameters:**
- `src` - Source tensor to clone (cannot be NULL)

**Returns:** Pointer to newly allocated tensor (caller owns, must call `tofu_tensor_free_data_too`)

**Behavior:**
- Creates both new tensor structure and new data buffer
- Copies all data from source to new tensor
- Preserves shape and data type

**Example:**
```c
tofu_tensor *original = tofu_tensor_zeros(2, (int[]){2, 3}, TOFU_FLOAT);
tofu_tensor *copy = tofu_tensor_clone(original);

// copy is independent of original
tofu_tensor_free_data_too(copy);
tofu_tensor_free_data_too(original);
```

---

### `tofu_tensor_repeat`

Create a tensor by repeating data multiple times.

```c
tofu_tensor *tofu_tensor_repeat(const tofu_tensor *src, int times);
```

**Parameters:**
- `src` - Source tensor to repeat (cannot be NULL)
- `times` - Number of repetitions (must be > 0)

**Returns:** Pointer to newly allocated tensor (caller owns, must call `tofu_tensor_free_data_too`)

**Behavior:**
- Creates new tensor with size = `src->len * times`
- Repeats source data sequentially

**Example:**
```c
float data[] = {1.0f, 2.0f};
tofu_tensor *t = tofu_tensor_create(data, 1, (int[]){2}, TOFU_FLOAT);
tofu_tensor *repeated = tofu_tensor_repeat(t, 3);
// repeated contains: [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

tofu_tensor_free_data_too(repeated);
tofu_tensor_free(t);
```

---

### `tofu_tensor_arange`

Create a 1-D tensor with evenly spaced values (similar to NumPy arange).

```c
tofu_tensor *tofu_tensor_arange(double start, double stop, double step, tofu_dtype dtype);
```

**Parameters:**
- `start` - Starting value (inclusive)
- `stop` - Ending value (exclusive)
- `step` - Step size between values
- `dtype` - Data type for the resulting tensor

**Returns:** Pointer to newly allocated 1-D tensor (caller owns, must call `tofu_tensor_free_data_too`)

**Behavior:**
- Creates values `[start, start+step, start+2*step, ..., stop)`
- Number of elements = `ceil((stop - start) / step)`

**Example:**
```c
tofu_tensor *t = tofu_tensor_arange(0.0, 5.0, 1.0, TOFU_FLOAT);
// t contains: [0.0, 1.0, 2.0, 3.0, 4.0]

tofu_tensor_free_data_too(t);
```

**See also:** `tofu_tensor_rearange` for in-place filling

---

### `tofu_tensor_rearange`

Fill existing tensor with evenly spaced values (in-place arange).

```c
void tofu_tensor_rearange(tofu_tensor *src, double start, double stop, double step);
```

**Parameters:**
- `src` - Tensor to fill (cannot be NULL)
- `start` - Starting value (inclusive)
- `stop` - Ending value (exclusive)
- `step` - Step size between values

**Behavior:**
- Fills tensor with `[start, start+step, start+2*step, ...]`
- Number of values written is `min(tensor size, ceil((stop-start)/step))`
- Modifies tensor data in-place

---

## Cleanup Functions

### `tofu_tensor_free`

Free tensor structure (does NOT free data buffer).

```c
void tofu_tensor_free(tofu_tensor *t);
```

**Parameters:**
- `t` - Tensor to free (can be NULL, no-op if NULL)

**Behavior:**
- Frees only the tensor structure and dims array
- Does NOT free the data buffer - caller must free data separately
- Safe to call even if tensor was used with `tofu_graph_param()`
- Call AFTER `tofu_graph_free()` if tensor was used with graph

**Example:**
```c
float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
tofu_tensor *t = tofu_tensor_create(data, 1, (int[]){4}, TOFU_FLOAT);

tofu_tensor_free(t);  // Free tensor structure only
// data is still valid
```

**See also:** `tofu_tensor_free_data_too`

---

### `tofu_tensor_free_data_too`

Free both tensor structure and data buffer.

```c
void tofu_tensor_free_data_too(tofu_tensor *t);
```

**Parameters:**
- `t` - Tensor to free (can be NULL, no-op if NULL)

**Behavior:**
- Frees both the tensor and its associated data buffer
- Only use if tensor owns its data (created with `tofu_tensor_zeros`, `tofu_tensor_clone`, etc.)
- Do NOT use if tensor was created with `tofu_tensor_create()` (use `tofu_tensor_free`)
- Safe to call if tensor was used with `tofu_graph_param()`
- Call AFTER `tofu_graph_free()` if tensor was used with graph

**Example:**
```c
tofu_tensor *t = tofu_tensor_zeros(2, (int[]){2, 3}, TOFU_FLOAT);

tofu_tensor_free_data_too(t);  // Free both tensor and data
```

**Warning:** Using this on tensors created with `tofu_tensor_create()` will cause undefined behavior!

---

## Shape Operations

### `tofu_tensor_size`

Get total number of elements in tensor.

```c
size_t tofu_tensor_size(tofu_tensor *t);
```

**Parameters:**
- `t` - Tensor (cannot be NULL)

**Returns:** Total element count (product of all dimensions)

**Example:**
```c
tofu_tensor *t = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
size_t size = tofu_tensor_size(t);  // Returns 12
```

---

### `tofu_tensor_reshape`

Reshape tensor to new dimensions (view operation, no data copy).

```c
tofu_tensor *tofu_tensor_reshape(tofu_tensor *src, int ndim, const int *dims);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `ndim` - Number of dimensions for reshaped tensor
- `dims` - Array of new dimension sizes

**Returns:** New tensor structure sharing data with source (caller owns, must call `tofu_tensor_free`)

**Behavior:**
- Does NOT copy data - result shares memory with source
- Only changes shape metadata, not data layout
- Source must outlive result tensor
- Product of dims must equal `tofu_tensor_size(src)`

**Example:**
```c
tofu_tensor *t = tofu_tensor_zeros(1, (int[]){12}, TOFU_FLOAT);
tofu_tensor *reshaped = tofu_tensor_reshape(t, 2, (int[]){3, 4});

// reshaped is a view of t with shape [3, 4]
tofu_tensor_free(reshaped);  // Free view
tofu_tensor_free_data_too(t);  // Free original
```

**See also:** `tofu_tensor_reshape_src` for in-place reshape

---

### `tofu_tensor_reshape_src`

Reshape tensor in-place (modifies source tensor metadata).

```c
void tofu_tensor_reshape_src(tofu_tensor *src, int ndim, const int *dims);
```

**Parameters:**
- `src` - Tensor to reshape (cannot be NULL)
- `ndim` - Number of dimensions for reshaped tensor
- `dims` - Array of new dimension sizes

**Behavior:**
- Modifies src tensor structure in-place
- Does NOT copy or reallocate data
- Only changes shape metadata
- Product of dims must equal `tofu_tensor_size(src)`

**Example:**
```c
tofu_tensor *t = tofu_tensor_zeros(1, (int[]){12}, TOFU_FLOAT);
tofu_tensor_reshape_src(t, 2, (int[]){3, 4});
// t now has shape [3, 4]
```

---

### `tofu_tensor_transpose`

Transpose tensor by permuting dimensions.

```c
tofu_tensor *tofu_tensor_transpose(const tofu_tensor *src, tofu_tensor *dst, const int *axes);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `axes` - Permutation array (can be NULL for reverse order)

**Returns:** Result tensor (caller owns if dst was NULL)

**Behavior:**
- If `axes` is NULL, reverses dimension order (e.g., `[2,3,4]` → `[4,3,2]`)
- If `axes` is non-NULL, permutes according to axes (e.g., `axes=[1,0]` swaps dims)
- For 2-D matrix, `axes=NULL` transposes (rows ↔ columns)

**Example:**
```c
// Matrix transpose
tofu_tensor *matrix = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
tofu_tensor *transposed = tofu_tensor_transpose(matrix, NULL, NULL);
// transposed has shape [4, 3]

// Custom permutation
int axes[] = {2, 0, 1};
tofu_tensor *t3d = tofu_tensor_zeros(3, (int[]){2, 3, 4}, TOFU_FLOAT);
tofu_tensor *permuted = tofu_tensor_transpose(t3d, NULL, axes);
// permuted has shape [4, 2, 3]
```

---

### `tofu_tensor_slice`

Extract slice from tensor (copies data).

```c
tofu_tensor *tofu_tensor_slice(const tofu_tensor *src, tofu_tensor *dst,
                               int axis, int start, int len);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `axis` - Axis along which to slice
- `start` - Starting index along axis
- `len` - Length of slice

**Returns:** Result tensor (caller owns if dst was NULL)

**Preconditions:**
- `axis < src->ndim`
- `start >= 0` and `start + len <= src->dims[axis]`
- If `dst` is non-NULL, it must have correct shape for slice

**Example:**
```c
tofu_tensor *t = tofu_tensor_arange(0.0, 10.0, 1.0, TOFU_FLOAT);
tofu_tensor *slice = tofu_tensor_slice(t, NULL, 0, 2, 5);
// slice contains: [2.0, 3.0, 4.0, 5.0, 6.0]
```

**See also:** `tofu_tensor_slice_nocopy` for view without copying

---

### `tofu_tensor_slice_nocopy`

Create view of tensor slice (no data copy).

```c
tofu_tensor *tofu_tensor_slice_nocopy(tofu_tensor *src, tofu_tensor *dst,
                                      int axis, int start, int len);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `axis` - Axis along which to slice
- `start` - Starting index along axis
- `len` - Length of slice

**Returns:** Result tensor sharing data with source (caller owns if dst was NULL)

**Behavior:**
- Does NOT copy data - result shares memory with source
- Modifying result will modify source tensor
- Source must outlive result tensor

**Warning:** This is a view operation - changes affect the original tensor!

---

### `tofu_tensor_concat`

Concatenate two tensors along specified axis.

```c
tofu_tensor *tofu_tensor_concat(const tofu_tensor *src1, const tofu_tensor *src2,
                                tofu_tensor *dst, int axis);
```

**Parameters:**
- `src1` - First tensor (cannot be NULL)
- `src2` - Second tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `axis` - Axis along which to concatenate

**Returns:** Result tensor (caller owns if dst was NULL)

**Preconditions:**
- All dimensions except `axis` must match between src1 and src2

**Behavior:**
- Result `dims[axis] = src1->dims[axis] + src2->dims[axis]`

**Example:**
```c
tofu_tensor *a = tofu_tensor_zeros(2, (int[]){2, 3}, TOFU_FLOAT);
tofu_tensor *b = tofu_tensor_zeros(2, (int[]){2, 3}, TOFU_FLOAT);
tofu_tensor *concat = tofu_tensor_concat(a, b, NULL, 0);
// concat has shape [4, 3]
```

---

## Mathematical Operations

### `tofu_tensor_matmul`

Compute matrix multiplication with broadcasting.

```c
tofu_tensor *tofu_tensor_matmul(const tofu_tensor *src1, const tofu_tensor *src2,
                                tofu_tensor *dst);
```

**Parameters:**
- `src1` - Left operand tensor (cannot be NULL)
- `src2` - Right operand tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)

**Returns:** Result tensor (caller owns if dst was NULL)

**Preconditions:**
- `src1->dims[src1->ndim-1]` must equal `src2->dims[src2->ndim-2]`

**Behavior:**
- **1-D @ 1-D**: Dot product → scalar
- **2-D @ 2-D**: Standard matrix multiplication
- **N-D @ 1-D**: Matrix-vector (drops last dim)
- **1-D @ N-D**: Vector-matrix (drops first dim)
- **N-D @ N-D**: Batch matmul with broadcasting

**Example:**
```c
// Matrix multiplication
tofu_tensor *A = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
tofu_tensor *B = tofu_tensor_zeros(2, (int[]){4, 5}, TOFU_FLOAT);
tofu_tensor *C = tofu_tensor_matmul(A, B, NULL);
// C has shape [3, 5]

// Batch matrix multiplication
tofu_tensor *batch_A = tofu_tensor_zeros(3, (int[]){2, 3, 4}, TOFU_FLOAT);
tofu_tensor *batch_B = tofu_tensor_zeros(3, (int[]){2, 4, 5}, TOFU_FLOAT);
tofu_tensor *batch_C = tofu_tensor_matmul(batch_A, batch_B, NULL);
// batch_C has shape [2, 3, 5]
```

**Notes:**
- Most commonly used operation for neural networks
- Broadcasts batch dimensions automatically

**See also:** `tofu_tensor_inner` for inner product

---

### `tofu_tensor_inner`

Compute inner product (sum-product over last axes).

```c
tofu_tensor *tofu_tensor_inner(const tofu_tensor *src1, const tofu_tensor *src2,
                               tofu_tensor *dst);
```

**Parameters:**
- `src1` - First tensor (cannot be NULL)
- `src2` - Second tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)

**Returns:** Result tensor (caller owns if dst was NULL)

**Preconditions:**
- `src1->dims[src1->ndim-1]` must equal `src2->dims[src2->ndim-1]`

**Behavior:**
- **1-D × 1-D**: Dot product → scalar
- **2-D × 2-D**: `result[i,j] = sum(a[i,:] * b[j,:])`
- **N-D × N-D**: Cartesian product of non-last dimensions
- Output shape: `(*a.shape[:-1], *b.shape[:-1])`

**Example:**
```c
tofu_tensor *a = tofu_tensor_arange(0.0, 3.0, 1.0, TOFU_FLOAT);  // [0, 1, 2]
tofu_tensor *b = tofu_tensor_arange(1.0, 4.0, 1.0, TOFU_FLOAT);  // [1, 2, 3]
tofu_tensor *result = tofu_tensor_inner(a, b, NULL);
// result = 0*1 + 1*2 + 2*3 = 8.0
```

**See also:** `tofu_tensor_matmul`, `tofu_tensor_outer`

---

### `tofu_tensor_outer`

Compute outer product (cartesian product without summation).

```c
tofu_tensor *tofu_tensor_outer(const tofu_tensor *src1, const tofu_tensor *src2,
                               tofu_tensor *dst);
```

**Parameters:**
- `src1` - First tensor (cannot be NULL)
- `src2` - Second tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)

**Returns:** Result tensor (caller owns if dst was NULL)

**Behavior:**
- Flattens both input tensors
- Computes: `result[i,j] = a[i] * b[j]`
- Always produces 2-D output
- Output shape: `[a.size, b.size]` where size is total element count

**Example:**
```c
tofu_tensor *a = tofu_tensor_arange(0.0, 3.0, 1.0, TOFU_FLOAT);  // [0, 1, 2]
tofu_tensor *b = tofu_tensor_arange(1.0, 3.0, 1.0, TOFU_FLOAT);  // [1, 2]
tofu_tensor *result = tofu_tensor_outer(a, b, NULL);
// result shape [3, 2]:
// [[0, 0],
//  [1, 2],
//  [2, 4]]
```

---

## Element-wise Operations

### `tofu_tensor_elew`

Apply element-wise binary operation with broadcasting.

```c
tofu_tensor *tofu_tensor_elew(const tofu_tensor *src1, const tofu_tensor *src2,
                              tofu_tensor *dst, tofu_elew_op elew_op);
```

**Parameters:**
- `src1` - First tensor (cannot be NULL)
- `src2` - Second tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `elew_op` - Operation to apply (TOFU_MUL, TOFU_DIV, TOFU_SUM, TOFU_SUB, TOFU_POW, etc.)

**Returns:** Result tensor (caller owns if dst was NULL)

**Preconditions:**
- src1 and src2 must be broadcastable (NumPy rules)

**Operations:**
- `TOFU_MUL` - Element-wise multiplication (*)
- `TOFU_DIV` - Element-wise division (/)
- `TOFU_SUM` - Element-wise addition (+)
- `TOFU_SUB` - Element-wise subtraction (-)
- `TOFU_POW` - Element-wise power (^)
- `TOFU_MAX` - Element-wise maximum
- `TOFU_MIN` - Element-wise minimum

**Example:**
```c
tofu_tensor *a = tofu_tensor_arange(1.0, 5.0, 1.0, TOFU_FLOAT);  // [1, 2, 3, 4]
tofu_tensor *b = tofu_tensor_arange(2.0, 6.0, 1.0, TOFU_FLOAT);  // [2, 3, 4, 5]

tofu_tensor *sum = tofu_tensor_elew(a, b, NULL, TOFU_SUM);
// sum = [3, 5, 7, 9]

tofu_tensor *prod = tofu_tensor_elew(a, b, NULL, TOFU_MUL);
// prod = [2, 6, 12, 20]

// Broadcasting example
tofu_tensor *matrix = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
float scalar_data[] = {2.0f};
tofu_tensor *scalar = tofu_tensor_create(scalar_data, 1, (int[]){1}, TOFU_FLOAT);
tofu_tensor *scaled = tofu_tensor_elew(matrix, scalar, NULL, TOFU_MUL);
// All elements of matrix multiplied by 2.0
```

**See also:** `tofu_tensor_elew_param`, `tofu_tensor_elew_broadcast`

---

### `tofu_tensor_elew_param`

Apply element-wise operation between tensor and scalar.

```c
tofu_tensor *tofu_tensor_elew_param(const tofu_tensor *src, double param,
                                    tofu_tensor *dst, tofu_elew_op elew_op);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `param` - Scalar parameter
- `dst` - Destination tensor (can be NULL to allocate new)
- `elew_op` - Operation to apply

**Returns:** Result tensor with same shape as src (caller owns if dst was NULL)

**Behavior:**
- Applies operation element-wise: `op(tensor_element, param)`

**Example:**
```c
tofu_tensor *t = tofu_tensor_arange(1.0, 5.0, 1.0, TOFU_FLOAT);  // [1, 2, 3, 4]

tofu_tensor *scaled = tofu_tensor_elew_param(t, 2.0, NULL, TOFU_MUL);
// scaled = [2, 4, 6, 8]

tofu_tensor *shifted = tofu_tensor_elew_param(t, 10.0, NULL, TOFU_SUM);
// shifted = [11, 12, 13, 14]

tofu_tensor *squared = tofu_tensor_elew_param(t, 2.0, NULL, TOFU_POW);
// squared = [1, 4, 9, 16]
```

---

### `tofu_tensor_elew_broadcast`

Apply element-wise operation with automatic broadcasting.

```c
tofu_tensor *tofu_tensor_elew_broadcast(const tofu_tensor *src1, const tofu_tensor *src2,
                                        tofu_tensor *dst, tofu_elew_op elew_op);
```

**Parameters:**
- `src1` - First tensor (cannot be NULL)
- `src2` - Second tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `elew_op` - Operation to apply

**Returns:** Result tensor with broadcast shape (caller owns if dst was NULL)

**Notes:**
- Automatically broadcasts inputs to compatible shape
- Equivalent to `tofu_tensor_elew` but with explicit broadcast handling
- Follows NumPy broadcasting rules

---

## Reductions

### `tofu_tensor_sumreduce`

Reduce tensor along axis using sum operation.

```c
tofu_tensor *tofu_tensor_sumreduce(const tofu_tensor *src, tofu_tensor *dst, int axis);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `axis` - Axis along which to reduce

**Returns:** Result tensor with `dims[axis]` removed (caller owns if dst was NULL)

**Behavior:**
- Output shape: `src->dims` with `dims[axis]` removed
- Computes sum of all elements along specified axis

**Example:**
```c
tofu_tensor *t = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
// Fill with 1.0
for (int i = 0; i < 12; i++) {
    float val = 1.0f;
    TOFU_TENSOR_DATA_FROM(t, i, val, TOFU_FLOAT);
}

tofu_tensor *row_sum = tofu_tensor_sumreduce(t, NULL, 1);
// row_sum has shape [3], each element = 4.0

tofu_tensor *col_sum = tofu_tensor_sumreduce(t, NULL, 0);
// col_sum has shape [4], each element = 3.0
```

**See also:** `tofu_tensor_meanreduce`, `tofu_tensor_maxreduce`

---

### `tofu_tensor_meanreduce`

Reduce tensor along axis using mean operation.

```c
tofu_tensor *tofu_tensor_meanreduce(const tofu_tensor *src, tofu_tensor *dst, int axis);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `axis` - Axis along which to reduce

**Returns:** Result tensor with `dims[axis]` removed (caller owns if dst was NULL)

**Behavior:**
- Output shape: `src->dims` with `dims[axis]` removed
- Computes arithmetic mean of all elements along specified axis

**Example:**
```c
tofu_tensor *t = tofu_tensor_arange(0.0, 12.0, 1.0, TOFU_FLOAT);
tofu_tensor_reshape_src(t, 2, (int[]){3, 4});

tofu_tensor *row_mean = tofu_tensor_meanreduce(t, NULL, 1);
// row_mean has shape [3]
// row_mean[0] = mean([0,1,2,3]) = 1.5
// row_mean[1] = mean([4,5,6,7]) = 5.5
// row_mean[2] = mean([8,9,10,11]) = 9.5
```

---

### `tofu_tensor_maxreduce`

Reduce tensor along axis using max operation.

```c
tofu_tensor *tofu_tensor_maxreduce(const tofu_tensor *src, tofu_tensor *dst,
                                   tofu_tensor *arg, int axis);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `arg` - Argmax indices tensor (can be NULL if indices not needed)
- `axis` - Axis along which to reduce

**Returns:** Result tensor with `dims[axis]` removed (caller owns if dst was NULL)

**Behavior:**
- Output shape: `src->dims` with `dims[axis]` removed
- If `arg` is non-NULL, fills it with indices of maximum values

**Example:**
```c
float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
tofu_tensor *t = tofu_tensor_create(data, 2, (int[]){2, 3}, TOFU_FLOAT);

tofu_tensor *max_vals = tofu_tensor_maxreduce(t, NULL, NULL, 1);
// max_vals = [4.0, 9.0]

tofu_tensor *indices = tofu_tensor_zeros(1, (int[]){2}, TOFU_INT32);
max_vals = tofu_tensor_maxreduce(t, NULL, indices, 1);
// indices = [2, 2] (position of max in each row)
```

---

### `tofu_tensor_sub_broadcast`

Subtract reduced tensor from source with broadcasting.

```c
tofu_tensor *tofu_tensor_sub_broadcast(const tofu_tensor *src, const tofu_tensor *reduced,
                                       tofu_tensor *dst, int axis);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `reduced` - Reduced tensor to subtract (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `axis` - Axis along which reduction was performed

**Returns:** Result tensor with same shape as src (caller owns if dst was NULL)

**Preconditions:**
- `reduced->ndim = src->ndim - 1` (one dimension removed)

**Behavior:**
- Broadcasts reduced tensor back along axis and subtracts
- Useful for normalization operations (subtract mean, etc.)

---

## Activation Functions

### `tofu_tensor_lrelu`

Apply Leaky ReLU activation function.

```c
tofu_tensor *tofu_tensor_lrelu(const tofu_tensor *src, tofu_tensor *dst, float negslope);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `negslope` - Slope for negative values (typically 0.01)

**Returns:** Result tensor with same shape as src (caller owns if dst was NULL)

**Behavior:**
- Computes: `x if x >= 0, else negslope * x`
- Standard ReLU equivalent when `negslope = 0`

**Example:**
```c
float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
tofu_tensor *t = tofu_tensor_create(data, 1, (int[]){5}, TOFU_FLOAT);

tofu_tensor *relu = tofu_tensor_lrelu(t, NULL, 0.0f);
// relu = [0.0, 0.0, 0.0, 1.0, 2.0]

tofu_tensor *leaky = tofu_tensor_lrelu(t, NULL, 0.01f);
// leaky = [-0.02, -0.01, 0.0, 1.0, 2.0]
```

**Note:** For use in computation graphs with automatic differentiation, use `tofu_graph_relu()` instead.

---

### `tofu_tensor_softmax`

Apply softmax activation along specified axis.

```c
tofu_tensor *tofu_tensor_softmax(const tofu_tensor *src, tofu_tensor *dst, int axis);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `axis` - Axis along which to apply softmax

**Returns:** Result tensor with same shape as src (caller owns if dst was NULL)

**Behavior:**
- Computes: `exp(x_i) / sum(exp(x_j))` along axis
- Uses numerically stable implementation (subtracts max before exp)
- Output values sum to 1.0 along specified axis

**Example:**
```c
float logits[] = {1.0f, 2.0f, 3.0f};
tofu_tensor *t = tofu_tensor_create(logits, 1, (int[]){3}, TOFU_FLOAT);
tofu_tensor *probs = tofu_tensor_softmax(t, NULL, 0);
// probs ≈ [0.09, 0.24, 0.67] (sums to 1.0)
```

**Note:** For use in computation graphs with automatic differentiation, use `tofu_graph_softmax()` instead.

---

### `tofu_tensor_layer_norm`

Apply layer normalization with learnable affine transform.

```c
tofu_tensor *tofu_tensor_layer_norm(const tofu_tensor *src, tofu_tensor *dst,
                                    const tofu_tensor *gamma, const tofu_tensor *beta,
                                    int axis, double eps);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `gamma` - Scale parameter tensor (can be NULL for no scaling)
- `beta` - Shift parameter tensor (can be NULL for no shift)
- `axis` - Axis along which to normalize
- `eps` - Small constant for numerical stability (typically 1e-5)

**Returns:** Result tensor with same shape as src (caller owns if dst was NULL)

**Behavior:**
- Normalizes: `(x - mean) / sqrt(variance + eps)`
- Then applies: `gamma * normalized + beta` (if gamma/beta non-NULL)
- If gamma/beta are NULL, only normalization is applied

**Example:**
```c
tofu_tensor *x = tofu_tensor_zeros(2, (int[]){2, 4}, TOFU_FLOAT);
float gamma_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
float beta_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
tofu_tensor *gamma = tofu_tensor_create(gamma_data, 1, (int[]){4}, TOFU_FLOAT);
tofu_tensor *beta = tofu_tensor_create(beta_data, 1, (int[]){4}, TOFU_FLOAT);

tofu_tensor *normalized = tofu_tensor_layer_norm(x, NULL, gamma, beta, 1, 1e-5);
```

---

## Utilities

### `tofu_tensor_issameshape`

Check if two tensors have the same shape.

```c
int tofu_tensor_issameshape(const tofu_tensor *t1, const tofu_tensor *t2);
```

**Parameters:**
- `t1` - First tensor (cannot be NULL)
- `t2` - Second tensor (cannot be NULL)

**Returns:** 1 if same shape, 0 otherwise

---

### `tofu_tensor_isbroadcastable`

Check if two tensors can be broadcast together (NumPy semantics).

```c
int tofu_tensor_isbroadcastable(const tofu_tensor *t1, const tofu_tensor *t2);
```

**Parameters:**
- `t1` - First tensor (cannot be NULL)
- `t2` - Second tensor (cannot be NULL)

**Returns:** 1 if broadcastable, 0 otherwise

**Broadcasting Rules:**
- Arrays with fewer dimensions are prepended with size-1 dimensions
- Size-1 dimensions are stretched to match the other array
- Dimensions must match or one must be 1

**Example:**
```c
tofu_tensor *a = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
tofu_tensor *b = tofu_tensor_zeros(1, (int[]){4}, TOFU_FLOAT);
int can_broadcast = tofu_tensor_isbroadcastable(a, b);  // Returns 1

tofu_tensor *c = tofu_tensor_zeros(1, (int[]){3}, TOFU_FLOAT);
can_broadcast = tofu_tensor_isbroadcastable(a, c);  // Returns 0
```

---

### `tofu_tensor_broadcast_to`

Broadcast tensor to specified shape (NumPy semantics).

```c
tofu_tensor *tofu_tensor_broadcast_to(const tofu_tensor *src, tofu_tensor *dst,
                                      int ndim, const int *dims);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `ndim` - Number of dimensions for target shape
- `dims` - Target dimension sizes

**Returns:** Result tensor with target shape (caller owns if dst was NULL)

**Preconditions:**
- src must be broadcastable to target shape (NumPy rules)

**Behavior:**
- Follows NumPy broadcasting rules
- Size-1 dimensions are stretched to match target

---

### `tofu_tensor_print`

Print tensor to stdout with custom format.

```c
void tofu_tensor_print(const tofu_tensor *t, const char *fmt);
```

**Parameters:**
- `t` - Tensor to print (cannot be NULL)
- `fmt` - Format string for each element (e.g., "%.6f", "%d")

**Example:**
```c
tofu_tensor *t = tofu_tensor_arange(0.0, 6.0, 1.0, TOFU_FLOAT);
tofu_tensor_reshape_src(t, 2, (int[]){2, 3});

tofu_tensor_print(t, "%.1f");
// Output:
// [[0.0, 1.0, 2.0],
//  [3.0, 4.0, 5.0]]
```

**See also:** `tofu_tensor_fprint` for printing to arbitrary stream, `tofu_tensor_save` for saving to file

---

### `tofu_tensor_fprint`

Print tensor to file stream with custom format.

```c
void tofu_tensor_fprint(FILE *stream, const tofu_tensor *t, const char *fmt);
```

**Parameters:**
- `stream` - File stream to write to (cannot be NULL)
- `t` - Tensor to print (cannot be NULL)
- `fmt` - Format string for each element

---

### `tofu_tensor_save`

Save tensor to file with custom format.

```c
int tofu_tensor_save(const char *file_name, const tofu_tensor *t, const char *fmt);
```

**Parameters:**
- `file_name` - Path to output file (cannot be NULL)
- `t` - Tensor to save (cannot be NULL)
- `fmt` - Format string for each element

**Returns:** 0 on success, non-zero on error

---

### `tofu_tensor_convert`

Convert tensor to different data type.

```c
tofu_tensor *tofu_tensor_convert(const tofu_tensor *src, tofu_tensor *dst,
                                 tofu_dtype dtype_d);
```

**Parameters:**
- `src` - Source tensor (cannot be NULL)
- `dst` - Destination tensor (can be NULL to allocate new)
- `dtype_d` - Target data type

**Returns:** Result tensor with same shape as src but different dtype (caller owns if dst was NULL)

**Behavior:**
- Converts each element to target type with appropriate casting
- May lose precision (e.g., float to int truncates)

**Example:**
```c
float data[] = {1.7f, 2.3f, 3.9f};
tofu_tensor *floats = tofu_tensor_create(data, 1, (int[]){3}, TOFU_FLOAT);
tofu_tensor *ints = tofu_tensor_convert(floats, NULL, TOFU_INT32);
// ints = [1, 2, 3]
```

---

### `tofu_tensor_index`

Convert multi-dimensional coordinates to flat index.

```c
int tofu_tensor_index(const tofu_tensor *t, int *coords);
```

**Parameters:**
- `t` - Tensor (cannot be NULL)
- `coords` - Array of coordinates, length must be `t->ndim`

**Returns:** Flat index into tensor data array

---

### `tofu_tensor_coords`

Convert flat index to multi-dimensional coordinates.

```c
void tofu_tensor_coords(const tofu_tensor *t, int index, int *coords);
```

**Parameters:**
- `t` - Tensor (cannot be NULL)
- `index` - Flat index into tensor data array
- `coords` - Output array for coordinates, length must be `t->ndim`

---

## Common Patterns

### Working with Tensor Memory

```c
// Pattern 1: User manages data buffer
float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
tofu_tensor *t = tofu_tensor_create(data, 1, (int[]){4}, TOFU_FLOAT);
// Use tensor...
tofu_tensor_free(t);
// data is still valid

// Pattern 2: Library manages data buffer
tofu_tensor *t = tofu_tensor_zeros(1, (int[]){4}, TOFU_FLOAT);
// Use tensor...
tofu_tensor_free_data_too(t);
// Both tensor and data are freed
```

### Accessing Tensor Elements

```c
tofu_tensor *t = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);

// Read element at index i
float value;
TOFU_TENSOR_DATA_TO(t, i, value, TOFU_FLOAT);

// Write element at index i
value = 42.0f;
TOFU_TENSOR_DATA_FROM(t, i, value, TOFU_FLOAT);

// Copy element from src[si] to dst[di]
TOFU_TENSOR_DATA_ASSIGN(dst, di, src, si);
```

### Broadcasting Example

```c
// Add scalar to matrix (broadcasting)
tofu_tensor *matrix = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
tofu_tensor *result = tofu_tensor_elew_param(matrix, 5.0, NULL, TOFU_SUM);

// Add vector to matrix rows (broadcasting)
tofu_tensor *row_vec = tofu_tensor_zeros(1, (int[]){4}, TOFU_FLOAT);
result = tofu_tensor_elew_broadcast(matrix, row_vec, NULL, TOFU_SUM);
```
