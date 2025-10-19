# Core Concepts

Understanding the fundamental concepts behind Tofu will help you build neural networks efficiently and write correct, safe code. This guide introduces the key ideas that make Tofu powerful.

## Introduction

Tofu is built around a small number of core concepts: **tensors**, **computation graphs**, and **automatic differentiation**. These three ideas work together to enable rapid development of machine learning systems.

If you come from a NumPy background, tensors will feel familiar—they're just multi-dimensional arrays. The difference is that Tofu tensors can be organized into computation graphs that automatically compute gradients for training. You won't need to manually derive derivative formulas or write error-prone gradient code.

This guide builds intuition for each concept with concrete examples. By the end, you'll understand how they fit together in a complete training loop.

## Tensors

A tensor is simply a multi-dimensional array of numbers. Tensors are the fundamental data structure in Tofu—all neural network operations work with tensors.

### Understanding Tensor Dimensions

Think of tensors as a natural extension of scalar numbers:

- **Scalar**: A single number. Shape: `[]` (0 dimensions). Example: `5.0`
- **Vector**: A 1-D list of numbers. Shape: `[3]` (1 dimension). Example: `[1.0, 2.0, 3.0]`
- **Matrix**: A 2-D grid of numbers. Shape: `[2, 3]` (2 dimensions). Example: `[[1, 2, 3], [4, 5, 6]]`
- **Tensor (3-D+)**: Higher-dimensional arrays. Shape: `[2, 3, 4]` (3 dimensions), etc.

In practice, machine learning uses tensors extensively:

- An image might be shape `[height, width, channels]` (e.g., `[28, 28, 1]` for 28x28 grayscale)
- A batch of images might be shape `[batch_size, height, width, channels]` (e.g., `[32, 28, 28, 1]`)
- Neural network weights are often 2-D matrices: shape `[input_dim, output_dim]`

### Tensor Shape and Size

Every tensor has a **shape**—a tuple of integer dimensions. The total number of elements is the product of all dimensions.

```
Tensor shape [2, 3, 4] contains 2 * 3 * 4 = 24 elements
```

Tofu's tensor structure stores both the shape and a flat data buffer:

```c
tofu_tensor {
    int ndim;        // Number of dimensions (e.g., 3)
    int *dims;       // Array of dimension sizes (e.g., [2, 3, 4])
    void *data;      // Flat buffer of 24 floating point numbers
    tofu_dtype dtype; // Data type (TOFU_FLOAT, TOFU_INT32, etc.)
}
```

### Data Types

Tensors can hold different numeric types depending on your needs:

- `TOFU_FLOAT` - 32-bit floating point (most common for neural networks)
- `TOFU_DOUBLE` - 64-bit floating point (higher precision)
- `TOFU_INT32`, `TOFU_INT64` - Integer types
- `TOFU_BOOL` - Boolean values

For machine learning, you'll typically use `TOFU_FLOAT` for efficiency and simplicity.

### Creating Tensors

Tofu provides several ways to create tensors:

```c
// Create tensor from existing data buffer (you manage the buffer)
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
int dims[] = {2, 2};
tofu_tensor *t = tofu_tensor_create(data, 2, dims, TOFU_FLOAT);

// Create tensor with values (library manages allocation)
float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
tofu_tensor *t = tofu_tensor_create_with_values(values, 2, dims);

// Create zero-filled tensor
int dims[] = {3, 4};
tofu_tensor *t = tofu_tensor_zeros(2, dims, TOFU_FLOAT);

// Create tensor with sequential values (like NumPy arange)
tofu_tensor *t = tofu_tensor_arange(0.0, 10.0, 1.0, TOFU_FLOAT);  // [0, 1, ..., 9]
```

### Tensor Operations

Tofu provides three main categories of tensor operations:

**Element-wise operations** apply an operation to each element independently:

```
Addition:  [1, 2] + [3, 4] = [4, 6]
Multiply:  [1, 2] * [3, 4] = [3, 8]
Power:     [2, 3] ^ 2 = [4, 9]
```

**Matrix operations** perform linear algebra calculations:

```
Matrix multiplication: [2,3] @ [3,4] = [2,4]
```

**Reduction operations** combine elements along an axis:

```
Sum reduction along axis 1:
[[1, 2, 3],     [6]        (1+2+3=6)
 [4, 5, 6]] --> [15]       (4+5+6=15)
```

These operations form the building blocks of neural networks. For example, a fully-connected layer performs: `output = matmul(input, weights) + bias`.

### Broadcasting: Implicit Dimension Expansion

A powerful feature of tensor operations is **broadcasting**—automatically expanding smaller tensors to match larger ones:

```
Matrix [3, 4] + Vector [4] broadcasts the vector to [3, 4]:
[[a, b, c, d],     [[a+x, b+y, c+z, d+w],
 [e, f, g, h],  +   [e+x, f+y, g+z, h+w],
 [i, j, k, l]]  x   [i+x, j+y, k+z, l+w]]

where vector [4] is implicitly treated as [x, y, z, w]
and repeated for each row
```

This allows you to add biases to layer outputs without manually replicating them.

## Computation Graphs

A computation graph is a way of representing how data flows through operations. Instead of computing results immediately, you describe the computation, then ask the graph to compute outputs and gradients.

### Why Use Computation Graphs?

There are two key advantages:

1. **Memory efficiency**: The graph knows all operations in advance, so it can optimize memory usage.
2. **Automatic differentiation**: Once you have the graph, computing gradients is automatic—no manual derivative math needed.

### Graph Structure: Directed Acyclic Graph (DAG)

A computation graph is a directed acyclic graph (DAG) where:

- **Nodes** represent tensors or operations
- **Edges** represent data flow between nodes
- **No cycles** (data only flows forward)

Here's a simple example:

```
      x (INPUT)  W (PARAM)
           |        |
           v        v
        matmul ←────+
           |
           | y
           v
          add ← bias (PARAM)
           |
           v
         relu
           |
           v
       output
```

This graph represents: `output = relu((x @ W) + bias)`

### Node Types

Nodes in a computation graph come in three flavors:

**Leaf nodes** have no inputs:
- `INPUT`: Data that doesn't need gradients (e.g., batch data)
- `PARAM`: Trainable parameters (e.g., weights, biases)

**Operation nodes** combine inputs:
- `MATMUL`: Matrix multiplication
- `ADD`: Element-wise addition
- `MUL`: Element-wise multiplication
- `RELU`: Activation function
- `SOFTMAX`: Softmax activation
- `MSE_LOSS`: Mean squared error loss
- `CE_LOSS`: Cross-entropy loss
- And many more...

**Important**: Graph nodes own their results (the tensors computed by operations), but the graph does NOT own INPUT or PARAM tensors. You create those tensors, add them to the graph, and you're responsible for freeing them later.

### Building a Graph

Creating a graph follows this pattern:

```c
// 1. Create tensors (you own these)
tofu_tensor *input = tofu_tensor_zeros(2, (int[]){3, 4}, TOFU_FLOAT);
tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){4, 5}, TOFU_FLOAT);
tofu_tensor *bias = tofu_tensor_zeros(1, (int[]){5}, TOFU_FLOAT);

// 2. Create graph and add leaf nodes
tofu_graph *g = tofu_graph_create();

tofu_graph_node *x = tofu_graph_input(g, input);
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *b = tofu_graph_param(g, bias);

// 3. Build computation by adding operations
tofu_graph_node *y = tofu_graph_matmul(g, x, W);
tofu_graph_node *z = tofu_graph_add(g, y, b);
tofu_graph_node *out = tofu_graph_relu(g, z);
```

The graph now contains all the computation. Each operation node automatically computes its value during construction.

### Forward Pass: Computing Outputs

When you add the first operation node to a graph, forward pass computation happens automatically:

```c
tofu_graph_node *y = tofu_graph_matmul(g, x, W);  // Forward pass computes matmul
tofu_tensor *result = tofu_graph_get_value(y);   // Get the computed result
```

Each node stores its result in `node->value`. You can inspect these at any time during or after building the graph.

## Automatic Differentiation

Automatic differentiation is the magic that lets you compute gradients without writing a single derivative formula. It works by building on the chain rule from calculus.

### The Chain Rule in Action

Recall from calculus: if `z = f(g(x))`, then `dz/dx = (df/dg) * (dg/dx)`.

In neural networks, this chains together:

```
Forward:  x → square → * 2 → y

Backward: ∂y/∂x = ∂(*2)/∂(square) * ∂(square)/∂x
                 = 2 * (2*x)
                 = 4x
```

For `x = 3`: Forward gives `y = (3^2) * 2 = 18`. Backward gives `dy/dx = 4*3 = 12`.

### Two Phases: Forward and Backward

Training has two phases:

**Forward pass**: Compute outputs by executing the graph. Each node records its operation and inputs for later use.

**Backward pass**: Starting from a loss scalar, compute gradients by working backwards through the graph, applying the chain rule at each node.

```
Forward Pass:
x → op1 → op2 → loss
(compute intermediate values)

Backward Pass (reverse):
loss → ∂op2 → ∂op1 → gradients for x
(compute gradients using chain rule)
```

### Reverse-Mode Autodiff: How Tofu Does It

Tofu uses reverse-mode automatic differentiation (also called backpropagation):

1. **Build the graph** by adding nodes (forward pass happens automatically)
2. **Call `tofu_graph_backward(g, loss)`** to compute gradients
3. **Gradients accumulate** in `node->grad` for each node

During backward:

```c
// Each node that requires gradients gets a ∂loss/∂node value in node->grad
tofu_graph_node *W = tofu_graph_param(g, weights);
// ... build graph and call backward ...
tofu_tensor *W_grad = tofu_graph_get_grad(W);  // Now contains ∂loss/∂W
```

The backward pass visits nodes in reverse topological order, so gradients flow correctly through the entire graph.

### Gradient Accumulation

Gradients accumulate by default. If you call backward twice without zeroing, gradients add up:

```c
tofu_graph_backward(g, loss1);  // node->grad = ∂loss1/∂node
tofu_graph_backward(g, loss2);  // node->grad += ∂loss2/∂node (accumulates!)
```

This is why you must call `tofu_graph_zero_grad()` before each training iteration:

```c
for (int epoch = 0; epoch < 100; epoch++) {
    tofu_graph_zero_grad(g);    // Clear old gradients

    // Forward pass (build graph)
    tofu_graph_node *loss = build_forward_pass(g);

    // Backward pass (compute new gradients)
    tofu_graph_backward(g, loss);
}
```

## Memory Management and Ownership

Proper memory management is critical in Tofu. Understanding ownership prevents memory leaks and use-after-free bugs.

### Ownership Rules

**Caller owns: INPUT and PARAM tensors**

When you create a tensor and pass it to `tofu_graph_input()` or `tofu_graph_param()`, the graph does NOT take ownership. You must free it yourself:

```c
tofu_tensor *input = tofu_tensor_create(data, 2, dims, TOFU_FLOAT);
tofu_graph_node *x = tofu_graph_input(g, input);

// After you're done with the graph...
tofu_graph_free(g);      // Graph is freed
tofu_tensor_free(input); // YOU must free the input tensor
```

**Graph owns: Operation results**

When an operation creates a result (like matmul, add, relu), the graph owns that tensor. You never free operation results—the graph does:

```c
tofu_graph_node *result = tofu_graph_matmul(g, a, b);
tofu_tensor *result_value = tofu_graph_get_value(result);

// When you're done:
tofu_graph_free(g);  // Frees result_value automatically
```

### View Operations: Sharing Memory

Some operations create "views"—new tensor structures that share memory with originals. No data is copied:

```c
tofu_tensor *t = tofu_tensor_zeros(1, (int[]){12}, TOFU_FLOAT);
tofu_tensor *reshaped = tofu_tensor_reshape(t, 2, (int[]){3, 4});

// reshaped shares data with t
// Both point to the same 12 floats, just interpreted differently
```

When using views, remember:
- Don't free the view with `free_data_too` (that would free shared memory)
- Use `tofu_tensor_free()` on views (just free the structure)
- The original must outlive the view

### Cleanup Order

Always clean up in this order:

```c
// 1. Free optimizer (if used)
tofu_optimizer_free(opt);

// 2. Free graph (frees operation results and nodes)
tofu_graph_free(g);

// 3. Free parameter tensors (you created these)
tofu_tensor_free_data_too(weights);
tofu_tensor_free_data_too(bias);

// 4. Free input tensors (you created these)
tofu_tensor_free(input);
```

### Common Mistakes

**Mistake 1: Freeing a view with `free_data_too`**
```c
// WRONG!
tofu_tensor *view = tofu_tensor_reshape(t, 2, (int[]){3, 4});
tofu_tensor_free_data_too(view);  // Frees shared memory!
```

**Correct:**
```c
tofu_tensor *view = tofu_tensor_reshape(t, 2, (int[]){3, 4});
tofu_tensor_free(view);            // Just free the structure
```

**Mistake 2: Using graph node results after freeing the graph**
```c
// WRONG!
tofu_graph_free(g);
tofu_tensor *result = tofu_graph_get_value(node);  // Dangling pointer!
```

**Correct:**
```c
tofu_tensor *result = tofu_graph_get_value(node);  // Get before freeing
tofu_graph_free(g);
```

**Mistake 3: Forgetting to free parameter tensors**
```c
// WRONG!
tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_free(g);  // Forgetting to free weights!
```

**Correct:**
```c
tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_free(g);
tofu_tensor_free_data_too(weights);  // Free the tensor you created
```

## Training Loop Pattern

A typical training loop follows a consistent pattern:

```
for each epoch:
    for each batch:
        1. Zero gradients
        2. Build forward graph and compute loss
        3. Backward pass (compute gradients)
        4. Optimizer step (update parameters using gradients)
        5. Clear operations (keep parameters, discard computation nodes)
```

Here's what this looks like in code:

```c
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);  // Learning rate

for (int epoch = 0; epoch < 100; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        // 1. Zero gradients
        tofu_optimizer_zero_grad(opt);

        // 2. Forward pass
        tofu_graph_node *x = tofu_graph_input(g, batch_data[batch]);
        tofu_graph_node *pred = tofu_graph_matmul(g, x, W);
        pred = tofu_graph_add(g, pred, b);
        tofu_graph_node *target = tofu_graph_input(g, batch_targets[batch]);

        // 3. Compute loss
        tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

        // 4. Backward pass (compute gradients)
        tofu_graph_backward(g, loss);

        // 5. Update parameters
        tofu_optimizer_step(opt);

        // 6. Clear operations for next batch (W and b are preserved)
        tofu_graph_clear_ops(g);
    }
}
```

The order is important:

- **Zero before forward**: Clean slate for new gradients
- **Forward then backward**: Must compute values before computing gradients
- **Backward before step**: Optimizer needs gradients to update
- **Clear after step**: Makes room for next batch while keeping parameters

## How These Concepts Work Together

Now you can see how everything fits together:

1. **Tensors** hold your data—inputs, parameters, and intermediate results
2. **Computation graphs** describe the structure of your model and automatically compute results
3. **Automatic differentiation** computes gradients by applying the chain rule through the graph
4. **Training loop** repeats: zero gradients, forward, backward, update parameters

The power of this design: you describe your model once, and Tofu automatically computes all the gradients. No manual derivative formulas. No gradient bugs. Just correct, efficient training.

## Summary

Understanding these core concepts will serve you well as you build neural networks with Tofu:

- **Tensors** are multi-dimensional arrays—the fundamental data structure
- **Computation graphs** organize operations in a way that enables automatic differentiation
- **Automatic differentiation** computes gradients automatically using the chain rule
- **Memory ownership** is explicit: you own inputs and parameters, graphs own operations
- **Training follows a pattern**: zero gradients, forward, backward, update, repeat

Next, check out the tutorials to see these concepts in action. The first tutorial will walk you through building a complete neural network from scratch.
