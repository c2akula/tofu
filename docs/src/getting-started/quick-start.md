# Quick Start

Get started with Tofu in 5 minutes.

## What You'll Build

In this quick-start guide, you'll create your first Tofu program that performs a basic tensor operation: matrix multiplication. This will introduce you to the fundamentals of working with tensors in Tofu.

## Prerequisites

You should have Tofu installed and built on your system. If not, please see the [Installation Guide](./installation.md) first.

## Your First Program

Create a file called `first.c` with the following code:

```c
#include <stdio.h>
#include "tofu/tofu_tensor.h"

int main() {
    // Create two matrices: a (2x3) and b (3x2)
    int dims_a[] = {2, 3};
    tofu_tensor *a = tofu_tensor_zeros(2, dims_a, TOFU_FLOAT);

    int dims_b[] = {3, 2};
    tofu_tensor *b = tofu_tensor_zeros(2, dims_b, TOFU_FLOAT);

    // Initialize matrix a with values [1, 2, 3, 4, 5, 6]
    float vals_a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (int i = 0; i < 6; i++) {
        TOFU_TENSOR_DATA_FROM(a, i, vals_a[i], TOFU_FLOAT);
    }

    // Initialize matrix b with values [1, 2, 3, 4, 5, 6]
    float vals_b[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (int i = 0; i < 6; i++) {
        TOFU_TENSOR_DATA_FROM(b, i, vals_b[i], TOFU_FLOAT);
    }

    // Perform matrix multiplication: result = a @ b
    tofu_tensor *result = tofu_tensor_matmul(a, b, NULL);

    // Print the result to stdout
    printf("Result of a @ b:\n");
    tofu_tensor_print(result, "%.1f");

    // Free all tensor memory
    tofu_tensor_free_data_too(a);
    tofu_tensor_free_data_too(b);
    tofu_tensor_free_data_too(result);

    return 0;
}
```

## Compiling Your Program

To compile this program, you'll need to link against the Tofu library:

```bash
gcc -o first first.c -I/path/to/tofu/build/include \
    -L/path/to/tofu/build/src -ltofu -lm
```

Or if you've installed Tofu to your system:

```bash
gcc -o first first.c -ltofu -lm
```

The `-lm` flag is needed for the math library, and `-ltofu` links against Tofu.

## Expected Output

When you run the program, you should see:

```
Result of a @ b:
[[22.0 28.0]
 [49.0 64.0]]
```

This is the result of multiplying:
- Matrix a: `[[1, 2, 3], [4, 5, 6]]` (shape 2×3)
- Matrix b: `[[1, 2], [3, 4], [5, 6]]` (shape 3×2)

## What Just Happened?

Let's break down the key components of your first tensor program:

### Tensor Creation

```c
tofu_tensor *a = tofu_tensor_zeros(2, dims_a, TOFU_FLOAT);
```

This creates a tensor (multi-dimensional array) with:
- `2` dimensions (a matrix)
- Shape defined by `dims_a` (2 rows, 3 columns)
- Data type `TOFU_FLOAT` (32-bit floating-point)
- All elements initialized to zero

### Tensor Data Access

```c
TOFU_TENSOR_DATA_FROM(a, i, vals_a[i], TOFU_FLOAT);
```

This macro writes a floating-point value into the tensor at flat index `i`. Tensors store data in a flat array, but Tofu handles the indexing for multi-dimensional access.

### Operations

```c
tofu_tensor *result = tofu_tensor_matmul(a, b, NULL);
```

Matrix multiplication is one of the fundamental operations in Tofu. Passing `NULL` for the destination tensor tells Tofu to allocate a new tensor for the result.

### Memory Management

```c
tofu_tensor_free_data_too(a);
```

Always clean up your tensors when done. Use `tofu_tensor_free_data_too()` when you created the tensor with `tofu_tensor_zeros()` (which allocates both the structure and data). This prevents memory leaks.

For deeper explanations of tensors, operations, and data types, see [Core Concepts](./concepts.md).

## Next Steps

Now that you've created your first tensor program, you're ready to explore more:

- **Build a neural network**: Learn how to create layers and train models in [Your First Network](./first-network.md)
- **Explore more operations**: Check out the API reference for all available tensor operations
- **Try more examples**: Look for example programs in the `examples/` directory

Happy tensor computing!
