# Tofu

![Version](https://img.shields.io/badge/version-0.3.0-blue)
![Tests](https://img.shields.io/badge/tests-13%2F13%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20ESP32-lightgrey)

**Tofu** is a lightweight deep learning framework for C, designed for embedded systems and environments where Python frameworks aren't available. It features automatic differentiation, dynamic computation graphs, and comprehensive validation across diverse neural network architectures.

**Key Features:**
- 🧠 **Automatic Differentiation**: Dynamic computation graphs with backward pass
- ✅ **Validated Operations**: All gradients numerically verified (13/13 tests passing)
- 🎯 **Production Ready**: Multi-class classification (100% accuracy), regression (MSE < 0.001)
- 📊 **NumPy-compatible API**: Familiar broadcasting semantics and tensor operations
- 🔧 **Zero Dependencies**: Pure C with no external libraries
- 📱 **ESP32 Support**: Cross-compilation for embedded deployment
- 🏗️ **Modern Architectures**: Residual networks, deep networks (10+ layers) validated

## Quick Start

```c
#include "tl_graph.h"
#include "tl_tensor.h"
#include "tl_optimizer.h"

// Create computation graph
tl_graph* g = tl_graph_create();

// Build simple neural network: [2] → [4] → [1]
float W1_data[8], W2_data[4], input_data[2] = {1.0f, 2.0f};
tl_tensor* t_input = tl_tensor_create(input_data, 1, (int[]){2}, TL_FLOAT);
tl_tensor* t_W1 = tl_tensor_create(W1_data, 2, (int[]){2, 4}, TL_FLOAT);
tl_tensor* t_W2 = tl_tensor_create(W2_data, 2, (int[]){4, 1}, TL_FLOAT);

// Forward pass
tl_graph_node* x = tl_graph_input(g, t_input);
tl_graph_node* W1 = tl_graph_param(g, t_W1);
tl_graph_node* h = tl_graph_matmul(g, x, W1);
tl_graph_node* h_act = tl_graph_relu(g, h);
tl_graph_node* W2 = tl_graph_param(g, t_W2);
tl_graph_node* output = tl_graph_matmul(g, h_act, W2);

// Backward pass (automatic differentiation)
tl_graph_backward(g, output);

// Gradients computed in W1->grad, W2->grad
tl_graph_free(g);
```

See [examples/](examples/) for complete training examples including MLP, ViT, and more.

## Prerequisites
The following steps have been tested for Ubuntu 16.04 but should work with
other distros as well. 
Required packages can be installed using the following command:

```
sudo apt-get install build-essential perl git pkg-config check
```

### ESP32 Cross-Compilation
To cross-compile for ESP32, you need the ESP32 toolchain installed. You can install it by:

1. Installing ESP-IDF: Follow the instructions at [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html)
2. Ensuring the ESP32 toolchain is in your PATH or specify its directory with the `--esp32-toolchain-dir` option

### Docker Testing Environment
For consistent testing across different environments, you can use the provided Docker setup:

1. Make sure you have Docker installed on your system
2. Run the provided script to build and test in a Docker container:
   ```
   ./run-tests-docker.sh
   ```

This will create an Ubuntu 22.04 container with all required dependencies and run the tests inside it. See the [Docker documentation](docker/README.md) for more information.

## Building and Installation
1.  Clone this repository to your local directory.

    ```
    cd <my_working_directory>
    git clone https://github.com/c2akula/tofu.git
    cd tofu
    ```

2.  Build and install

    First, configure your installation using:
    
    ```
    chmod +x configure
    ./configure
    ```
    There are options to customize your building and installation process.
    You can append them after `./configure`. 
    
    For cross-compiling for ESP32:
    ```
    ./configure --esp32=yes
    ```
    
    Or if your ESP32 toolchain is not in your PATH:
    ```
    ./configure --esp32=yes --esp32-toolchain-dir=/path/to/esp-idf/tools/xtensa-esp32-elf
    ```
    
    Detailed `./configure` options can be printed using `./configure -h`.

    After that, use `make` to compile the library and run the tests. Then `make install`
    to copy the library files and headers into the installation directory,
    or `sudo make install` if you don't have the permissions with that directory.
    
    ```
    make
    sudo make install
    ```

3.  Other `make` options

    Use `make info` to see other `make` options.
    Especially, you can use `make clean` to clean up the build directory and all
    object files, and `make uninstall` to remove library files and headers from
    the installation directory.

## Usage
Include `tl_tensor.h` in your project to use Tofu functions.

You can use the following command to get the compilation and linking flags when
building your project.

```
pkg-config --cflags --libs tofu
```

## Broadcasting Support

Tofu now supports broadcasting operations similar to NumPy. Broadcasting allows you to perform operations on arrays of different shapes. The rules for broadcasting are:

1. Arrays with fewer dimensions are prepended with dimensions of size 1.
2. Size-1 dimensions are stretched to match the corresponding dimension of the other array.
3. Arrays must be broadcastable in each dimension (they must have the same size, or one of them must have size 1).

### Broadcasting API

```c
// Check if two tensors can be broadcast together
int tl_tensor_isbroadcastable(const tl_tensor *t1, const tl_tensor *t2);

// Broadcast a tensor to a new shape
tl_tensor *tl_tensor_broadcast_to(const tl_tensor *src, tl_tensor *dst, int ndim, const int *dims);

// Element-wise operation with broadcasting
tl_tensor *tl_tensor_elew_broadcast(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst, tl_elew_op elew_op);
```

### Broadcasting Examples

```c
// Example 1: Broadcasting a scalar to a matrix
float scalar_val = 5.0f;
tl_tensor *scalar = tl_tensor_create(&scalar_val, 1, (int[]){1}, TL_FLOAT);
tl_tensor *result = tl_tensor_broadcast_to(scalar, NULL, 2, (int[]){3, 4});
// result will be a 3×4 matrix filled with 5.0

// Example 2: Element-wise multiplication with broadcasting
float arr1[] = {1, 2, 3};  // Shape: [3]
float arr2[] = {10, 20};   // Shape: [2, 1]
tl_tensor *t1 = tl_tensor_create(arr1, 1, (int[]){3}, TL_FLOAT);
tl_tensor *t2 = tl_tensor_create(arr2, 2, (int[]){2, 1}, TL_FLOAT);
tl_tensor *result = tl_tensor_elew_broadcast(t1, t2, NULL, TL_MUL);
// result will have shape [2, 3] and values [[10, 20, 30], [20, 40, 60]]
```

## Tensor Operations

Tofu provides NumPy-compatible tensor operations for linear algebra and tensor manipulation.

### Inner Product

Computes the inner product (sum-product over last axes) with cartesian product semantics.

```c
tl_tensor *tl_tensor_inner(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst);
```

**Behavior:**
- 1-D × 1-D: Dot product → scalar
- 2-D × 2-D: `result[i,j] = sum(a[i,:] * b[j,:])`
- N-D × N-D: Cartesian product of non-last dimensions

**Output shape:** `(*a.shape[:-1], *b.shape[:-1])`

**Example:**
```c
// Vector dot product
float a[] = {1, 2, 3};
float b[] = {4, 5, 6};
tl_tensor *v1 = tl_tensor_create(a, 1, (int[]){3}, TL_FLOAT);
tl_tensor *v2 = tl_tensor_create(b, 1, (int[]){3}, TL_FLOAT);
tl_tensor *result = tl_tensor_inner(v1, v2, NULL);
// result: scalar 32.0 (1*4 + 2*5 + 3*6)

// Matrix inner product [2,3] × [2,3] → [2,2]
float mat1[] = {1, 2, 3, 4, 5, 6};
float mat2[] = {1, 1, 1, 2, 2, 2};
tl_tensor *m1 = tl_tensor_create(mat1, 2, (int[]){2, 3}, TL_FLOAT);
tl_tensor *m2 = tl_tensor_create(mat2, 2, (int[]){2, 3}, TL_FLOAT);
result = tl_tensor_inner(m1, m2, NULL);
// result[i,j] = sum(m1[i,:] * m2[j,:])
```

### Matrix Multiplication (matmul)

Computes matrix multiplication with broadcasting on batch dimensions.

```c
tl_tensor *tl_tensor_matmul(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst);
```

**Behavior:**
- 1-D @ 1-D: Dot product → scalar
- 2-D @ 2-D: Standard matrix multiplication
- N-D @ 1-D: Matrix-vector, drops last dimension
- 1-D @ N-D: Vector-matrix, drops first dimension
- N-D @ N-D: Batch matrix multiplication with broadcasting

**Output shape:** Broadcasts batch dimensions, contracts last of a with second-to-last of b

**Example:**
```c
// Standard matrix multiplication [2,3] @ [3,2] → [2,2]
float a[] = {1, 2, 3, 4, 5, 6};
float b[] = {1, 1, 2, 2, 3, 3};
tl_tensor *m1 = tl_tensor_create(a, 2, (int[]){2, 3}, TL_FLOAT);
tl_tensor *m2 = tl_tensor_create(b, 2, (int[]){3, 2}, TL_FLOAT);
tl_tensor *result = tl_tensor_matmul(m1, m2, NULL);
// Standard matrix multiply: result[i,j] = sum(m1[i,:] * m2[:,j])

// Batch matrix multiplication with broadcasting [3,4] @ [2,4,5] → [2,3,5]
// The [3,4] matrix is broadcast across 2 batches
```

### Outer Product

Computes the outer product (cartesian product without summation).

```c
tl_tensor *tl_tensor_outer(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst);
```

**Behavior:**
- Flattens both input tensors
- Computes: `result[i,j] = a[i] * b[j]`
- Always produces 2-D output

**Output shape:** `[a.size, b.size]` where size is total element count

**Example:**
```c
// Vector outer product [3] outer [4] → [3,4]
float a[] = {1, 2, 3};
float b[] = {4, 5, 6, 7};
tl_tensor *v1 = tl_tensor_create(a, 1, (int[]){3}, TL_FLOAT);
tl_tensor *v2 = tl_tensor_create(b, 1, (int[]){4}, TL_FLOAT);
tl_tensor *result = tl_tensor_outer(v1, v2, NULL);
// result[i,j] = a[i] * b[j]
// [[4,  5,  6,  7],
//  [8, 10, 12, 14],
//  [12, 15, 18, 21]]

// Multi-dimensional inputs are flattened first
// [2,2] outer [2] → flatten to [4] outer [2] → [4,2]
```

### Comparison: inner vs matmul vs outer

| Operation | 1-D × 1-D | 2-D × 2-D | N-D Behavior | Use Case |
|-----------|-----------|-----------|--------------|----------|
| **inner** | Dot product (scalar) | Sum over last of both | Cartesian product | Generalized inner products |
| **matmul** | Dot product (scalar) | Matrix multiplication | Broadcasting batch dims | Deep learning, batch ops |
| **outer** | Outer product [m,n] | Flattens first [m²,n²] | Always flattens | Tensor products |

**Key Differences:**
- **inner**: Cartesian product of non-last dimensions (independent indices)
- **matmul**: Broadcasting of batch dimensions (shared indices) - most efficient for batches
- **outer**: No summation, always produces 2-D output

### Pre-allocated Destinations

All tensor operations accept an optional pre-allocated destination tensor:

```c
// Create destination tensor first
tl_tensor *dst = tl_tensor_zeros(2, (int[]){2, 2}, TL_FLOAT);

// Use pre-allocated destination
tl_tensor *result = tl_tensor_matmul(m1, m2, dst);
// result points to dst (reused)
```

This avoids memory allocation overhead in performance-critical loops.

Additional documentation is coming soon. But the API should be familiar if you have experience with `numpy` in Python.
