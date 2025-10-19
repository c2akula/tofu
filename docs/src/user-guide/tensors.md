# Tensors

Tensors are the fundamental data structure in Tofu, representing multi-dimensional arrays that flow through neural networks. This comprehensive guide covers everything you need to know about creating, manipulating, and operating on tensors.

## Introduction

Tensors are multi-dimensional arrays that generalize scalars (0-D), vectors (1-D), and matrices (2-D) to arbitrary dimensions. In neural networks, tensors represent:
- Input data (images, text, sensor readings)
- Model parameters (weights, biases)
- Intermediate activations
- Gradients during backpropagation

### Prerequisites

This guide assumes you've completed the Getting Started guide and understand:
- Basic tensor concepts (shape, dimensions)
- C programming and memory management
- How to compile and link against Tofu

### What This Guide Covers

- Tensor fundamentals and memory layout
- Creation methods for different use cases
- Data access and modification patterns
- Shape operations (reshape, transpose, slice)
- Mathematical operations (matmul, element-wise, broadcasting)
- Reduction operations (sum, mean, max)
- Activation functions
- Memory management and ownership

---

## Tensor Fundamentals

### Tensor Structure

A `tofu_tensor` represents a multi-dimensional array with the following key properties:

```c
struct tofu_tensor {
    tofu_dtype dtype;          // Data type (FLOAT, INT32, etc.)
    int len;                   // Total number of elements
    int ndim;                  // Number of dimensions
    int *dims;                 // Array of dimension sizes
    void *data;                // Pointer to data buffer
    struct tofu_tensor *owner; // For view tensors, points to data owner
    void *backend_data;        // Backend-specific data
};
```

Example of a 2×3 matrix:

```
ndim = 2
dims = [2, 3]
len = 6
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

Visual representation:
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]]
```

### Data Types

Tofu supports multiple data types via the `tofu_dtype` enum:

| Type | Description | Size | Use Cases |
|------|-------------|------|-----------|
| `TOFU_FLOAT` | 32-bit floating point | 4 bytes | Neural network weights, activations |
| `TOFU_DOUBLE` | 64-bit floating point | 8 bytes | High-precision computation |
| `TOFU_INT32` | 32-bit signed integer | 4 bytes | Labels, indices, counters |
| `TOFU_INT64` | 64-bit signed integer | 8 bytes | Large indices |
| `TOFU_INT8` | 8-bit signed integer | 1 byte | Quantized weights |
| `TOFU_INT16` | 16-bit signed integer | 2 bytes | Quantized activations |
| `TOFU_BOOL` | Boolean | 1 byte | Masks, conditions |

Most neural network operations use `TOFU_FLOAT` for weights and activations, while `TOFU_INT32` is common for labels and class predictions.

### Memory Layout

Tofu uses **row-major** (C-style) memory layout, where the last dimension varies fastest:

```c
// 2×3 matrix
float data[] = {1.0, 2.0, 3.0,   // Row 0
                4.0, 5.0, 6.0};  // Row 1

// 2×3×4 tensor (2 matrices, each 3×4)
// data[0..11] = first matrix (row-major)
// data[12..23] = second matrix (row-major)
```

This affects how you iterate and index tensors:

```c
// Iterating in memory order (efficient)
for (int i = 0; i < dims[0]; i++) {
    for (int j = 0; j < dims[1]; j++) {
        int index = i * dims[1] + j;
        // Access data[index]
    }
}
```

---

## Creating Tensors

### Wrapping Existing Data

Use `tofu_tensor_create()` to wrap existing data without copying:

```c
tofu_tensor *tofu_tensor_create(void *data, int ndim, const int *dims,
                                tofu_dtype dtype);
```

This is efficient when you already have data in memory:

```c
float weights[] = {0.1, 0.2, 0.3, 0.4};
tofu_tensor *W = tofu_tensor_create(weights, 2, (int[]){2, 2}, TOFU_FLOAT);

// Use W...

tofu_tensor_free(W);  // Frees structure, NOT data
// weights[] is still valid
```

**Use when**: Data is managed elsewhere (stack, static, or you handle malloc/free)

### Zero Initialization

`tofu_tensor_zeros()` allocates and zero-initializes a tensor:

```c
tofu_tensor *tofu_tensor_zeros(int ndim, const int *dims, tofu_dtype dtype);
```

Example:

```c
tofu_tensor *t = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
// t is a 3×4 matrix filled with 0.0

tofu_tensor_free_data_too(t);  // Frees both structure and data
```

**Use when**: You need a new tensor and will populate it later (common for parameters)

### Creating With Values

`tofu_tensor_create_with_values()` creates and initializes from an array:

```c
float values[] = {1.0, 2.0, 3.0, 4.0};
tofu_tensor *t = tofu_tensor_create_with_values(values, 1, (int[]){4}, TOFU_FLOAT);

tofu_tensor_free_data_too(t);
```

**Use when**: You have initial values ready (common for biases, small constants)

### Range of Values

`tofu_tensor_arange()` creates a tensor with evenly spaced values:

```c
tofu_tensor *tofu_tensor_arange(double start, double stop, double step,
                                tofu_dtype dtype);
```

Example:

```c
tofu_tensor *t = tofu_tensor_arange(0.0, 10.0, 2.0, TOFU_FLOAT);
// t = [0.0, 2.0, 4.0, 6.0, 8.0]

tofu_tensor_free_data_too(t);
```

**Use when**: Generating test data, indices, or sequences

### Deep Copy

`tofu_tensor_clone()` creates an independent copy:

```c
tofu_tensor *original = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
tofu_tensor *copy = tofu_tensor_clone(original);

// Modifying copy doesn't affect original

tofu_tensor_free_data_too(copy);
tofu_tensor_free_data_too(original);
```

**Use when**: You need to preserve original data while modifying a copy

---

## Accessing and Modifying Data

### Reading Values

Use the `TOFU_TENSOR_DATA_FROM()` macro to safely read values:

```c
tofu_tensor *t = tofu_tensor_arange(0.0, 5.0, 1.0, TOFU_FLOAT);

for (int i = 0; i < t->len; i++) {
    float value;
    TOFU_TENSOR_DATA_FROM(t, i, value, TOFU_FLOAT);
    printf("t[%d] = %.1f\n", i, value);
}

tofu_tensor_free_data_too(t);
```

### Writing Values

Use `TOFU_TENSOR_DATA_TO()` macro to safely write values:

```c
tofu_tensor *t = tofu_tensor_zeros(1, (int[]){4}, TOFU_FLOAT);

for (int i = 0; i < t->len; i++) {
    float value = i * 0.5;
    TOFU_TENSOR_DATA_TO(t, i, value, TOFU_FLOAT);
}

tofu_tensor_print(t, "%.1f");  // [0.0, 0.5, 1.0, 1.5]
tofu_tensor_free_data_too(t);
```

### Direct Pointer Access

For performance-critical code, access data directly:

```c
tofu_tensor *t = tofu_tensor_zeros(2, (int[]){100, 100}, TOFU_FLOAT);
float *data = (float*)t->data;

// Fast iteration
for (int i = 0; i < t->len; i++) {
    data[i] = i * 0.1;
}

tofu_tensor_free_data_too(t);
```

**Warning**: Ensure type safety - casting to wrong type causes undefined behavior.

### Iterating Multi-Dimensional Tensors

For 2-D tensors (matrices):

```c
tofu_tensor *matrix = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
float *data = (float*)matrix->data;

for (int i = 0; i < matrix->dims[0]; i++) {      // Rows
    for (int j = 0; j < matrix->dims[1]; j++) {  // Columns
        int index = i * matrix->dims[1] + j;
        data[index] = i + j;
    }
}

tofu_tensor_free_data_too(matrix);
```

For 3-D tensors:

```c
tofu_tensor *tensor = tofu_tensor_zeros(3, (int[]){2, 3, 4}, TOFU_FLOAT);
float *data = (float*)tensor->data;

for (int i = 0; i < tensor->dims[0]; i++) {
    for (int j = 0; j < tensor->dims[1]; j++) {
        for (int k = 0; k < tensor->dims[2]; k++) {
            int index = (i * tensor->dims[1] + j) * tensor->dims[2] + k;
            data[index] = i + j + k;
        }
    }
}

tofu_tensor_free_data_too(tensor);
```

---

## Shape Operations

### Reshape

`tofu_tensor_reshape()` changes tensor shape without copying data:

```c
tofu_tensor *tofu_tensor_reshape(const tofu_tensor *src, int ndim,
                                 const int *dims);
```

Example:

```c
tofu_tensor *t = tofu_tensor_arange(0.0, 12.0, 1.0, TOFU_FLOAT);
// t shape: [12]

tofu_tensor *matrix = tofu_tensor_reshape(t, 2, (int[]){3, 4});
// matrix shape: [3, 4], shares data with t

tofu_tensor_print(matrix, "%.1f");
// [[0.0, 1.0, 2.0, 3.0],
//  [4.0, 5.0, 6.0, 7.0],
//  [8.0, 9.0, 10.0, 11.0]]

tofu_tensor_free(matrix);  // View only
tofu_tensor_free_data_too(t);  // Original with data
```

**Important**: Product of new dimensions must equal original size.

### In-Place Reshape

`tofu_tensor_reshape_src()` reshapes a tensor in place:

```c
void tofu_tensor_reshape_src(tofu_tensor *t, int ndim, const int *new_dims);
```

Example:

```c
tofu_tensor *t = tofu_tensor_arange(0.0, 6.0, 1.0, TOFU_FLOAT);
tofu_tensor_reshape_src(t, 2, (int[]){2, 3});

tofu_tensor_print(t, "%.1f");
// [[0.0, 1.0, 2.0],
//  [3.0, 4.0, 5.0]]

tofu_tensor_free_data_too(t);
```

### Transpose

`tofu_tensor_transpose()` swaps dimensions:

```c
tofu_tensor *tofu_tensor_transpose(const tofu_tensor *src, tofu_tensor *dst);
```

For 2-D tensors, transposes rows and columns:

```c
float data[] = {1, 2, 3,
                4, 5, 6};
tofu_tensor *A = tofu_tensor_create(data, 2, (int[]){2, 3}, TOFU_FLOAT);
// [[1, 2, 3],
//  [4, 5, 6]]

tofu_tensor *AT = tofu_tensor_transpose(A, NULL);
// [[1, 4],
//  [2, 5],
//  [3, 6]]

tofu_tensor_free_data_too(AT);
tofu_tensor_free(A);
```

**Use cases**: Matrix operations, batch dimensions, image transformations

### Slice

`tofu_tensor_slice()` extracts a subtensor:

```c
tofu_tensor *tofu_tensor_slice(const tofu_tensor *src, tofu_tensor *dst,
                               int axis, int start, int len);
```

Example:

```c
tofu_tensor *t = tofu_tensor_arange(0.0, 10.0, 1.0, TOFU_FLOAT);
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

tofu_tensor *slice = tofu_tensor_slice(t, NULL, 0, 2, 5);
// [2, 3, 4, 5, 6]

tofu_tensor_free_data_too(slice);
tofu_tensor_free_data_too(t);
```

For matrices, slice rows:

```c
tofu_tensor *matrix = tofu_tensor_zeros(2, (int[]){10, 5}, TOFU_FLOAT);

tofu_tensor *rows = tofu_tensor_slice(matrix, NULL, 0, 2, 3);
// Extracts rows 2, 3, 4 → shape [3, 5]

tofu_tensor_free_data_too(rows);
tofu_tensor_free_data_too(matrix);
```

### Concatenate

`tofu_tensor_concat()` joins tensors along an axis:

```c
tofu_tensor *tofu_tensor_concat(const tofu_tensor *src1, const tofu_tensor *src2,
                                tofu_tensor *dst, int axis);
```

Example:

```c
tofu_tensor *a = tofu_tensor_arange(0.0, 3.0, 1.0, TOFU_FLOAT);
tofu_tensor_reshape_src(a, 2, (int[]){1, 3});  // [[0, 1, 2]]

tofu_tensor *b = tofu_tensor_arange(3.0, 6.0, 1.0, TOFU_FLOAT);
tofu_tensor_reshape_src(b, 2, (int[]){1, 3});  // [[3, 4, 5]]

tofu_tensor *c = tofu_tensor_concat(a, b, NULL, 0);  // Concatenate rows
// [[0, 1, 2],
//  [3, 4, 5]]

tofu_tensor_free_data_too(c);
tofu_tensor_free_data_too(b);
tofu_tensor_free_data_too(a);
```

---

## Mathematical Operations

### Matrix Multiplication

`tofu_tensor_matmul()` performs matrix multiplication with broadcasting:

```c
tofu_tensor *tofu_tensor_matmul(const tofu_tensor *src1, const tofu_tensor *src2,
                                tofu_tensor *dst);
```

Basic matrix multiplication:

```c
tofu_tensor *A = tofu_tensor_zeros(2, (int[]){2, 3}, TOFU_FLOAT);  // 2×3
tofu_tensor *B = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);  // 3×4

tofu_tensor *C = tofu_tensor_matmul(A, B, NULL);  // 2×4
// C[i,j] = Σ(A[i,k] * B[k,j])

tofu_tensor_free_data_too(C);
tofu_tensor_free_data_too(B);
tofu_tensor_free_data_too(A);
```

**Dimension rules**:
- For 1-D @ 1-D: `src1->dims[0]` must equal `src2->dims[0]` (dot product)
- For 2-D and higher: `src1->dims[src1->ndim-1]` must equal `src2->dims[src2->ndim-2]`

Batch matrix multiplication:

```c
tofu_tensor *batches = tofu_tensor_zeros(3, (int[]){10, 2, 3}, TOFU_FLOAT);
tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);

tofu_tensor *results = tofu_tensor_matmul(batches, weights, NULL);
// Shape: [10, 2, 4] - broadcasts weights across batch

tofu_tensor_free_data_too(results);
tofu_tensor_free_data_too(weights);
tofu_tensor_free_data_too(batches);
```

### Inner Product

`tofu_tensor_inner()` computes dot product (sum of element-wise products):

```c
tofu_tensor *a = tofu_tensor_arange(0.0, 3.0, 1.0, TOFU_FLOAT);  // [0, 1, 2]
tofu_tensor *b = tofu_tensor_arange(1.0, 4.0, 1.0, TOFU_FLOAT);  // [1, 2, 3]

tofu_tensor *result = tofu_tensor_inner(a, b, NULL);
// result = 0*1 + 1*2 + 2*3 = 8

tofu_tensor_free_data_too(result);
tofu_tensor_free_data_too(b);
tofu_tensor_free_data_too(a);
```

### Outer Product

`tofu_tensor_outer()` computes outer product:

```c
tofu_tensor *a = tofu_tensor_arange(1.0, 4.0, 1.0, TOFU_FLOAT);  // [1, 2, 3]
tofu_tensor *b = tofu_tensor_arange(1.0, 3.0, 1.0, TOFU_FLOAT);  // [1, 2]

tofu_tensor *result = tofu_tensor_outer(a, b, NULL);
// [[1, 2],
//  [2, 4],
//  [3, 6]]

tofu_tensor_free_data_too(result);
tofu_tensor_free_data_too(b);
tofu_tensor_free_data_too(a);
```

### Element-Wise Operations

`tofu_tensor_elew()` performs element-wise operations:

```c
tofu_tensor *tofu_tensor_elew(const tofu_tensor *src1, const tofu_tensor *src2,
                              tofu_tensor *dst, tofu_elew_op op);
```

Supported operations:

| Operation | Description | Example |
|-----------|-------------|---------|
| `TOFU_SUM` | Addition | a + b |
| `TOFU_SUB` | Subtraction | a - b |
| `TOFU_MUL` | Multiplication | a * b |
| `TOFU_DIV` | Division | a / b |
| `TOFU_MAX` | Maximum | max(a, b) |
| `TOFU_MIN` | Minimum | min(a, b) |

Example:

```c
tofu_tensor *a = tofu_tensor_arange(1.0, 5.0, 1.0, TOFU_FLOAT);  // [1, 2, 3, 4]
tofu_tensor *b = tofu_tensor_arange(2.0, 6.0, 1.0, TOFU_FLOAT);  // [2, 3, 4, 5]

tofu_tensor *sum = tofu_tensor_elew(a, b, NULL, TOFU_SUM);  // [3, 5, 7, 9]
tofu_tensor *prod = tofu_tensor_elew(a, b, NULL, TOFU_MUL);  // [2, 6, 12, 20]

tofu_tensor_free_data_too(prod);
tofu_tensor_free_data_too(sum);
tofu_tensor_free_data_too(b);
tofu_tensor_free_data_too(a);
```

### Element-Wise with Scalar

`tofu_tensor_elew_param()` applies operation with a scalar:

```c
tofu_tensor *tofu_tensor_elew_param(const tofu_tensor *src, double param,
                                    tofu_tensor *dst, tofu_elew_op op);
```

Example:

```c
tofu_tensor *t = tofu_tensor_arange(1.0, 5.0, 1.0, TOFU_FLOAT);  // [1, 2, 3, 4]

tofu_tensor *scaled = tofu_tensor_elew_param(t, 2.0, NULL, TOFU_MUL);  // [2, 4, 6, 8]
tofu_tensor *shifted = tofu_tensor_elew_param(t, 10.0, NULL, TOFU_SUM);  // [11, 12, 13, 14]

tofu_tensor_free_data_too(shifted);
tofu_tensor_free_data_too(scaled);
tofu_tensor_free_data_too(t);
```

### Broadcasting

Broadcasting allows operations on tensors with different but compatible shapes:

**Rules**:
1. Start from trailing dimensions
2. Dimensions are compatible if they're equal or one is 1
3. Missing dimensions are treated as 1

Examples:

```c
// Shape [3, 4] + Shape [4] → broadcasts to [3, 4]
tofu_tensor *matrix = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
tofu_tensor *bias = tofu_tensor_arange(1.0, 5.0, 1.0, TOFU_FLOAT);  // [4]

tofu_tensor *result = tofu_tensor_elew_broadcast(matrix, bias, NULL, TOFU_SUM);
// bias is added to each row of matrix

tofu_tensor_free_data_too(result);
tofu_tensor_free_data_too(bias);
tofu_tensor_free_data_too(matrix);
```

---

## Reduction Operations

### Sum Reduction

`tofu_tensor_sumreduce()` sums along an axis:

```c
tofu_tensor *t = tofu_tensor_arange(0.0, 12.0, 1.0, TOFU_FLOAT);
tofu_tensor_reshape_src(t, 2, (int[]){3, 4});

tofu_tensor *col_sums = tofu_tensor_sumreduce(t, NULL, 0);  // Sum rows
// [12, 15, 18, 21]

tofu_tensor *row_sums = tofu_tensor_sumreduce(t, NULL, 1);  // Sum columns
// [6, 22, 38]

tofu_tensor_free_data_too(row_sums);
tofu_tensor_free_data_too(col_sums);
tofu_tensor_free_data_too(t);
```

### Mean Reduction

`tofu_tensor_meanreduce()` computes mean along an axis:

```c
tofu_tensor *data = tofu_tensor_arange(0.0, 12.0, 1.0, TOFU_FLOAT);
tofu_tensor_reshape_src(data, 2, (int[]){3, 4});

tofu_tensor *row_means = tofu_tensor_meanreduce(data, NULL, 1);
// [1.5, 5.5, 9.5]

tofu_tensor_free_data_too(row_means);
tofu_tensor_free_data_too(data);
```

### Max Reduction

`tofu_tensor_maxreduce()` finds maximum values and optionally their indices:

```c
tofu_tensor *tofu_tensor_maxreduce(const tofu_tensor *src, tofu_tensor *dst,
                                   tofu_tensor *arg, int axis);
```

Example:

```c
float data[] = {3, 1, 4, 1, 5, 9, 2, 6, 5};
tofu_tensor *t = tofu_tensor_create(data, 2, (int[]){3, 3}, TOFU_FLOAT);

tofu_tensor *indices = tofu_tensor_zeros(1, (int[]){3}, TOFU_INT32);
tofu_tensor *max_vals = tofu_tensor_maxreduce(t, NULL, indices, 1);

// max_vals = [4, 9, 6]
// indices = [2, 2, 1]

tofu_tensor_free_data_too(max_vals);
tofu_tensor_free_data_too(indices);
tofu_tensor_free(t);
```

---

## Activation Functions

### Leaky ReLU

```c
tofu_tensor *tofu_tensor_lrelu(const tofu_tensor *src, tofu_tensor *dst,
                               float negslope);
```

Example:

```c
float data[] = {-2, -1, 0, 1, 2};
tofu_tensor *x = tofu_tensor_create(data, 1, (int[]){5}, TOFU_FLOAT);

tofu_tensor *relu = tofu_tensor_lrelu(x, NULL, 0.0f);  // Standard ReLU
// [0, 0, 0, 1, 2]

tofu_tensor *leaky = tofu_tensor_lrelu(x, NULL, 0.01f);  // Leaky ReLU
// [-0.02, -0.01, 0, 1, 2]

tofu_tensor_free_data_too(leaky);
tofu_tensor_free_data_too(relu);
tofu_tensor_free(x);
```

### Softmax

```c
tofu_tensor *tofu_tensor_softmax(const tofu_tensor *src, tofu_tensor *dst,
                                 int axis);
```

Example:

```c
float logits[] = {1, 2, 3};
tofu_tensor *t = tofu_tensor_create(logits, 1, (int[]){3}, TOFU_FLOAT);

tofu_tensor *probs = tofu_tensor_softmax(t, NULL, 0);
// Approximately [0.09, 0.24, 0.67]

tofu_tensor_free_data_too(probs);
tofu_tensor_free(t);
```

### Layer Normalization

```c
tofu_tensor *tofu_tensor_layer_norm(const tofu_tensor *src, tofu_tensor *dst,
                                    const tofu_tensor *gamma, const tofu_tensor *beta,
                                    int axis, double eps);
```

---

## Utility Functions

### Printing

```c
void tofu_tensor_print(const tofu_tensor *t, const char *fmt);
```

Example:

```c
tofu_tensor *t = tofu_tensor_arange(0.0, 6.0, 1.0, TOFU_FLOAT);
tofu_tensor_reshape_src(t, 2, (int[]){2, 3});

tofu_tensor_print(t, "%.1f");
// [[0.0, 1.0, 2.0],
//  [3.0, 4.0, 5.0]]

tofu_tensor_free_data_too(t);
```

### Size Queries

```c
size_t size = tofu_tensor_size(t);  // Total elements
int same_shape = tofu_tensor_issameshape(t1, t2);
int broadcastable = tofu_tensor_isbroadcastable(t1, t2);
```

### Type Conversion

```c
tofu_tensor *ints = tofu_tensor_convert(floats, NULL, TOFU_INT32);
```

---

## Memory Management

### Ownership Rules

**Rule 1**: Tensors created with `tofu_tensor_create()` don't own their data
```c
float data[4] = {1, 2, 3, 4};
tofu_tensor *t = tofu_tensor_create(data, 1, (int[]){4}, TOFU_FLOAT);
tofu_tensor_free(t);  // Only frees tensor structure
// data is still valid
```

**Rule 2**: Tensors created with `tofu_tensor_zeros()`, `tofu_tensor_clone()`, etc. own their data
```c
tofu_tensor *t = tofu_tensor_zeros(1, (int[]){4}, TOFU_FLOAT);
tofu_tensor_free_data_too(t);  // Frees both structure and data
```

**Rule 3**: View operations (reshape, transpose) share data
```c
tofu_tensor *original = tofu_tensor_zeros(1, (int[]){12}, TOFU_FLOAT);
tofu_tensor *view = tofu_tensor_reshape(original, 2, (int[]){3, 4});

// view shares data with original
tofu_tensor_free(view);  // Free view only
tofu_tensor_free_data_too(original);  // Free data with original
```

### Common Mistakes

**Mistake 1**: Using `free_data_too` on user-owned data
```c
// WRONG
float data[4];
tofu_tensor *t = tofu_tensor_create(data, 1, (int[]){4}, TOFU_FLOAT);
tofu_tensor_free_data_too(t);  // Tries to free stack memory!
```

**Mistake 2**: Memory leak from not freeing library-owned data
```c
// WRONG
tofu_tensor *t = tofu_tensor_zeros(1, (int[]){4}, TOFU_FLOAT);
tofu_tensor_free(t);  // Leaks data buffer!
```

**Mistake 3**: Freeing view data
```c
// WRONG
tofu_tensor *original = tofu_tensor_zeros(1, (int[]){12}, TOFU_FLOAT);
tofu_tensor *view = tofu_tensor_reshape(original, 2, (int[]){3, 4});
tofu_tensor_free_data_too(view);  // Frees shared data!
tofu_tensor_free_data_too(original);  // Double free!
```

---

## Best Practices

### Efficiency

1. **Reuse destination tensors** to avoid allocations
2. **Use views** when possible (reshape, not clone)
3. **Choose appropriate data types** (INT32 for labels, FLOAT for weights)
4. **Batch operations** for efficiency

### Debugging

1. **Validate shapes** before operations
2. **Print intermediate results** with `tofu_tensor_print()`
3. **Check for NaN/Inf** in numerical operations
4. **Track allocations** to find memory leaks

### Common Pitfalls

- Don't modify views expecting independence
- Don't use `free_data_too` on `tensor_create()` tensors
- Verify broadcast compatibility before operations
- Ensure consistent data types in operations

---

## Next Steps

Now that you understand tensors, continue to:
- **[Computation Graphs](graphs.md)** - Build neural networks with automatic differentiation
- **[Training Guide](training.md)** - Implement complete training loops
- **[API Reference](../api-reference/tensor-api.md)** - Detailed function specifications

For practical examples, see the tutorials section for complete neural network implementations.
