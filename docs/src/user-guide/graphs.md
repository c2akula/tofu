# Computation Graphs

Computation graphs are the foundation of automatic differentiation and neural network training in Tofu. This guide explains how to build, manage, and use computation graphs for deep learning applications.

## Introduction

A computation graph is a directed acyclic graph (DAG) that represents mathematical operations and their dependencies. Each node in the graph represents either:
- A leaf node: input data or trainable parameter
- An operation node: a mathematical operation (matmul, add, relu, etc.)

Tofu uses dynamic computation graphs (define-by-run), meaning the graph structure is built on-the-fly as operations execute. This provides flexibility for control flow and conditional computation.

### Key Concepts

**Forward Pass**: Computes values by flowing data through the graph from inputs to outputs. Each operation node stores its result in `node->value`.

**Backward Pass**: Computes gradients by flowing derivatives backward from the loss to parameters. Uses reverse-mode automatic differentiation (backpropagation). Each node stores its gradient in `node->grad`.

**Requires Gradient**: A flag indicating whether a node needs gradient computation. Parameters always require gradients, while inputs do not.

**Topological Order**: The graph automatically sorts nodes in reverse topological order for efficient backward pass execution.

### A Simple Example

```c
// Create graph
tofu_graph *g = tofu_graph_create();

// Add input and parameters
float x_data[] = {1.0f, 2.0f};
float w_data[] = {0.5f, -0.3f};

tofu_tensor *x_tensor = tofu_tensor_create(x_data, 1, (int[]){2}, TOFU_FLOAT);
tofu_tensor *w_tensor = tofu_tensor_create(w_data, 2, (int[]){2, 1}, TOFU_FLOAT);

tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *W = tofu_graph_param(g, w_tensor);

// Compute: y = x @ W
tofu_graph_node *y = tofu_graph_matmul(g, x, W);

// Backward pass
tofu_graph_backward(g, y);

// Access gradient
tofu_tensor *W_grad = tofu_graph_get_grad(W);  // Contains dL/dW

// Cleanup
tofu_graph_free(g);
tofu_tensor_free(x_tensor);
tofu_tensor_free(w_tensor);
```

### When to Use Graphs

Use computation graphs when you need:
- Automatic gradient computation for training
- Complex neural network architectures
- Efficient backpropagation through multiple layers
- Dynamic control flow in models

For simple tensor operations without gradients, you can use the Tensor API directly without graphs.

---

## Graph Fundamentals

### Directed Acyclic Graphs (DAGs)

Computation graphs must be acyclic to ensure well-defined forward and backward passes. Each operation creates a new node that depends on its input nodes, forming a directed edge from inputs to outputs.

```c
// This creates a DAG:
//   x ---\
//         matmul --> y --> relu --> z
//   W ---/
//   b -----------------------------> add --> out

tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *W = tofu_graph_param(g, W_tensor);
tofu_graph_node *b = tofu_graph_param(g, b_tensor);

tofu_graph_node *y = tofu_graph_matmul(g, x, W);
tofu_graph_node *z = tofu_graph_relu(g, y);
tofu_graph_node *out = tofu_graph_add(g, z, b);
```

The graph automatically tracks dependencies. When you call `tofu_graph_backward(g, out)`, it computes gradients for all parameters by traversing edges backward.

### Nodes and Their Roles

**Leaf Nodes** are the sources of data in the graph:
- `TOFU_OP_INPUT`: Non-trainable data (features, targets). Does not receive gradients.
- `TOFU_OP_PARAM`: Trainable parameters (weights, biases). Receives and accumulates gradients.

**Operation Nodes** perform computations:
- Binary operations: `matmul`, `add`, `mul`
- Activations: `relu`, `softmax`, `layer_norm`
- Shape operations: `reshape`, `transpose`
- Reductions: `mean`, `sum`
- Loss functions: `mse_loss`, `ce_loss`

Each operation node stores:
- `value`: Result of forward computation
- `grad`: Gradient from backward pass
- `inputs`: Pointer to input nodes
- `backward_fn`: Function to compute gradients for inputs
- `backward_ctx`: Saved tensors needed for backward (e.g., input values for ReLU)

### Forward Pass Execution

The forward pass computes outputs by executing operations in order:

```c
tofu_graph *g = tofu_graph_create();

// Create computation: y = relu(x @ W + b)
tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *W = tofu_graph_param(g, W_tensor);
tofu_graph_node *b = tofu_graph_param(g, b_tensor);

// Forward pass happens automatically as operations are added
tofu_graph_node *xW = tofu_graph_matmul(g, x, W);  // Computes x @ W immediately
tofu_graph_node *xWb = tofu_graph_add(g, xW, b);   // Computes xW + b immediately
tofu_graph_node *y = tofu_graph_relu(g, xWb);      // Computes relu(xWb) immediately

// At this point, y->value contains the final result
tofu_tensor *result = tofu_graph_get_value(y);
```

Each operation executes immediately when called, computing and storing the result in the new node's `value` field.

### Backward Pass Execution

The backward pass computes gradients using the chain rule:

```c
// Continuing from above...
tofu_graph_backward(g, y);

// Now all parameter nodes have gradients:
tofu_tensor *W_grad = tofu_graph_get_grad(W);  // dL/dW
tofu_tensor *b_grad = tofu_graph_get_grad(b);  // dL/db
```

The backward pass:
1. Initializes `loss->grad = 1.0` (derivative of loss w.r.t. itself)
2. Sorts nodes in reverse topological order
3. For each node (from loss back to inputs):
   - Calls its `backward_fn` to compute input gradients
   - Accumulates gradients for nodes that appear multiple times

**Important**: Gradients accumulate across backward passes. Always call `tofu_graph_zero_grad()` before each training iteration unless you explicitly want gradient accumulation.

### Gradient Flow and Requires Grad

The `requires_grad` flag determines whether a node needs gradient computation:

```c
tofu_graph_node *x = tofu_graph_input(g, x_tensor);     // requires_grad = 0
tofu_graph_node *W = tofu_graph_param(g, W_tensor);     // requires_grad = 1

tofu_graph_node *y = tofu_graph_matmul(g, x, W);        // requires_grad = 1 (because W does)
tofu_graph_node *z = tofu_graph_add(g, y, const_node);  // requires_grad = 1 (because y does)
```

An operation node requires gradients if ANY of its inputs require gradients. This propagates through the graph automatically.

Gradients only flow to nodes with `requires_grad = 1`:
- `PARAM` nodes always receive gradients (these are your trainable weights)
- `INPUT` nodes never receive gradients (they're just data)
- Operation nodes receive gradients if they're on a path from a parameter to the loss

---

## Creating and Managing Graphs

### Creating a Graph

Use `tofu_graph_create()` to create a new empty graph:

```c
tofu_graph *g = tofu_graph_create();

// Graph starts empty
// num_nodes = 0
// next_id = 0

// ... build your graph ...
```

The graph allocates memory dynamically and grows as you add nodes. It manages all nodes internally and frees them when `tofu_graph_free()` is called.

### Freeing a Graph

Use `tofu_graph_free()` to clean up a graph and all its nodes:

```c
tofu_graph *g = tofu_graph_create();

// Build graph...
tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *W = tofu_graph_param(g, W_tensor);
tofu_graph_node *y = tofu_graph_matmul(g, x, W);

// Free graph (but NOT tensors!)
tofu_graph_free(g);

// You must still free tensors separately
tofu_tensor_free(x_tensor);
tofu_tensor_free(W_tensor);
```

**Critical Memory Management Rule**: The graph does NOT take ownership of tensors passed to `tofu_graph_input()` or `tofu_graph_param()`. You must:
1. Free the graph first: `tofu_graph_free(g)`
2. Then free tensors: `tofu_tensor_free(tensor)`

The graph DOES own and free:
- All graph nodes
- All gradients (`node->grad`)
- All intermediate operation results (e.g., the result of `matmul`, `add`, etc.)

### Correct Cleanup Order

```c
// Create tensors
tofu_tensor *x_tensor = tofu_tensor_zeros(2, (int[]){1, 4}, TOFU_FLOAT);
tofu_tensor *W_tensor = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);

// Build graph
tofu_graph *g = tofu_graph_create();
tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *W = tofu_graph_param(g, W_tensor);
tofu_graph_node *y = tofu_graph_matmul(g, x, W);

// Training loop...

// CORRECT CLEANUP ORDER:
// 1. Free optimizer (if used)
tofu_optimizer_free(opt);

// 2. Free graph
tofu_graph_free(g);

// 3. Free tensors
tofu_tensor_free_data_too(x_tensor);
tofu_tensor_free_data_too(W_tensor);
```

### Clearing Operations

Use `tofu_graph_clear_ops()` to remove operation nodes while keeping parameters:

```c
tofu_graph *g = tofu_graph_create();

// Add parameters (persist across iterations)
tofu_graph_node *W = tofu_graph_param(g, W_tensor);
tofu_graph_node *b = tofu_graph_param(g, b_tensor);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Build forward graph for this batch
    tofu_graph_node *x = tofu_graph_input(g, batch_data);
    tofu_graph_node *y = tofu_graph_matmul(g, x, W);
    tofu_graph_node *out = tofu_graph_add(g, y, b);

    // Backward pass and optimization...
    tofu_graph_backward(g, loss);
    tofu_optimizer_step(opt);

    // Clear operations (W and b are preserved!)
    tofu_graph_clear_ops(g);
}
```

This is essential for training loops to prevent node accumulation and memory growth. After `clear_ops()`:
- All operation nodes are freed
- `PARAM` and `INPUT` nodes remain in the graph
- Parameter values and gradients are preserved
- The graph is ready for the next forward pass

**When to Use Clear Ops**:
- Between training iterations in a loop
- When reusing the same graph with different input data
- To prevent unbounded memory growth during training

**When NOT to Clear Ops**:
- If you need to access intermediate operation results after backward
- During a single forward/backward pass
- Before calling the optimizer (gradients would be lost!)

---

## Adding Leaf Nodes

Leaf nodes are the starting points of computation. They represent either input data or trainable parameters.

### Input Nodes

Input nodes represent non-trainable data like features or labels:

```c
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
tofu_tensor *x_tensor = tofu_tensor_create(data, 2, (int[]){1, 4}, TOFU_FLOAT);

tofu_graph *g = tofu_graph_create();
tofu_graph_node *x = tofu_graph_input(g, x_tensor);

// Input nodes do NOT compute gradients
// x->requires_grad == 0
// x->op == TOFU_OP_INPUT
```

Characteristics of input nodes:
- `requires_grad = 0` (no gradient computation)
- Used for features, labels, or other non-trainable data
- Graph does NOT own the tensor (caller must free it)
- Typically created fresh for each training iteration

### Parameter Nodes

Parameter nodes represent trainable weights or biases:

```c
float weights[] = {0.5f, -0.3f, 0.2f, 0.1f};
tofu_tensor *W_tensor = tofu_tensor_create(weights, 2, (int[]){2, 2}, TOFU_FLOAT);

tofu_graph *g = tofu_graph_create();
tofu_graph_node *W = tofu_graph_param(g, W_tensor);

// Parameter nodes DO compute gradients
// W->requires_grad == 1
// W->op == TOFU_OP_PARAM
```

Characteristics of parameter nodes:
- `requires_grad = 1` (gradient computation enabled)
- Used for weights, biases, or other learnable parameters
- Graph does NOT own the tensor (caller must free it)
- Typically created once and reused across iterations
- Preserved by `tofu_graph_clear_ops()`

### Ownership Rules

**Critical**: The graph does NOT take ownership of tensors passed to `tofu_graph_input()` or `tofu_graph_param()`.

```c
// Create tensor (you own this)
tofu_tensor *W = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);

// Add to graph (graph does NOT take ownership)
tofu_graph_node *W_node = tofu_graph_param(g, W);

// Later: free graph first, then tensor
tofu_graph_free(g);          // Frees the node, but NOT the tensor
tofu_tensor_free_data_too(W); // You must free the tensor
```

**Why this design?**
- Parameters persist across multiple training iterations
- You may need to save/load parameters independently
- Gives you full control over memory management

### Typical Usage Pattern

```c
// Setup phase (once)
tofu_tensor *W = tofu_tensor_zeros(2, (int[]){784, 10}, TOFU_FLOAT);
tofu_tensor *b = tofu_tensor_zeros(1, (int[]){10}, TOFU_FLOAT);

tofu_graph *g = tofu_graph_create();
tofu_graph_node *W_node = tofu_graph_param(g, W);
tofu_graph_node *b_node = tofu_graph_param(g, b);

// Training loop (many iterations)
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Create fresh input for each iteration
    float *batch_data = load_batch(epoch);
    tofu_tensor *x_tensor = tofu_tensor_create(batch_data, 2, (int[]){32, 784}, TOFU_FLOAT);

    // Add input node
    tofu_graph_node *x = tofu_graph_input(g, x_tensor);

    // Build forward graph using parameters
    tofu_graph_node *logits = tofu_graph_matmul(g, x, W_node);
    tofu_graph_node *out = tofu_graph_add(g, logits, b_node);

    // Training step...
    tofu_graph_backward(g, loss);
    tofu_optimizer_step(opt);

    // Clean up this iteration's input
    tofu_tensor_free(x_tensor);
    free(batch_data);

    // Clear operations (W_node and b_node are preserved)
    tofu_graph_clear_ops(g);
}

// Cleanup (once at end)
tofu_optimizer_free(opt);
tofu_graph_free(g);
tofu_tensor_free_data_too(W);
tofu_tensor_free_data_too(b);
```

### Common Pitfalls

**Pitfall 1: Freeing tensors before graph**
```c
// WRONG - this will crash!
tofu_tensor_free(W_tensor);  // Frees tensor
tofu_graph_free(g);          // Graph still references freed memory

// CORRECT
tofu_graph_free(g);          // Free graph first
tofu_tensor_free(W_tensor);  // Then free tensor
```

**Pitfall 2: Not freeing tensors**
```c
// WRONG - memory leak!
tofu_graph_free(g);
// Forgot to free tensors!

// CORRECT
tofu_graph_free(g);
tofu_tensor_free_data_too(W_tensor);
tofu_tensor_free_data_too(b_tensor);
```

**Pitfall 3: Clearing ops without re-adding params**
```c
// WRONG - params are gone after clear_ops if added in loop
for (int i = 0; i < 100; i++) {
    tofu_graph_node *W = tofu_graph_param(g, W_tensor);  // Re-adding param each time
    // ... training ...
    tofu_graph_clear_ops(g);  // Removes W!
}

// CORRECT - add params once before loop
tofu_graph_node *W = tofu_graph_param(g, W_tensor);
for (int i = 0; i < 100; i++) {
    // ... use W ...
    tofu_graph_clear_ops(g);  // W is preserved
}
```

---

## Mathematical Operations

Mathematical operations create new nodes that compute values during the forward pass and propagate gradients during the backward pass.

### Matrix Multiplication

Matrix multiplication is the workhorse of neural networks:

```c
// y = x @ W
// x: [batch, in_features]
// W: [in_features, out_features]
// y: [batch, out_features]

tofu_graph_node *x = tofu_graph_input(g, x_tensor);    // [32, 784]
tofu_graph_node *W = tofu_graph_param(g, W_tensor);    // [784, 10]
tofu_graph_node *y = tofu_graph_matmul(g, x, W);       // [32, 10]
```

The operation:
- Computes standard matrix multiplication with broadcasting
- Supports batched operations (3D, 4D tensors)
- Implements backward pass: `dL/dx = dL/dy @ W^T` and `dL/dW = x^T @ dL/dy`

**Precondition**: Inner dimensions must match: `a->dims[last] == b->dims[second-to-last]`

### Element-wise Addition

Addition is commonly used for adding biases:

```c
// out = x + b
// x: [batch, features]
// b: [features]
// out: [batch, features]

tofu_graph_node *x = tofu_graph_matmul(g, input, W);   // [32, 10]
tofu_graph_node *b = tofu_graph_param(g, b_tensor);    // [10]
tofu_graph_node *out = tofu_graph_add(g, x, b);        // [32, 10]
```

The operation:
- Performs element-wise addition with broadcasting
- Follows NumPy broadcasting rules
- Implements backward pass: gradients flow to both inputs

Broadcasting example:
```c
// Broadcasting [2, 3] + [3] -> [2, 3]
float a_data[] = {1, 2, 3, 4, 5, 6};
float b_data[] = {10, 20, 30};

tofu_tensor *a = tofu_tensor_create(a_data, 2, (int[]){2, 3}, TOFU_FLOAT);
tofu_tensor *b = tofu_tensor_create(b_data, 1, (int[]){3}, TOFU_FLOAT);

tofu_graph_node *a_node = tofu_graph_input(g, a);
tofu_graph_node *b_node = tofu_graph_input(g, b);
tofu_graph_node *c = tofu_graph_add(g, a_node, b_node);

// Result: [[11, 22, 33], [14, 25, 36]]
```

### Element-wise Multiplication

Multiplication is useful for attention mechanisms and gating:

```c
// out = x * y
tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *y = tofu_graph_input(g, y_tensor);
tofu_graph_node *out = tofu_graph_mul(g, x, y);
```

The operation:
- Performs element-wise multiplication with broadcasting
- Implements backward pass: `dL/dx = dL/dout * y` and `dL/dy = dL/dout * x`

Example: Attention scaling
```c
// Attention: scale * (Q @ K^T)
tofu_graph_node *qk = tofu_graph_matmul(g, Q, K_T);
tofu_graph_node *scale_tensor = tofu_graph_param(g, scale);
tofu_graph_node *scaled = tofu_graph_mul(g, qk, scale_tensor);
```

### Chaining Operations

Operations can be chained to build complex expressions:

```c
// Build: y = ReLU(x @ W + b)
tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *W = tofu_graph_param(g, W_tensor);
tofu_graph_node *b = tofu_graph_param(g, b_tensor);

// Chain operations
tofu_graph_node *xW = tofu_graph_matmul(g, x, W);     // Linear transformation
tofu_graph_node *xWb = tofu_graph_add(g, xW, b);      // Add bias
tofu_graph_node *y = tofu_graph_relu(g, xWb);         // Apply activation

// Each intermediate result is stored in the node's value
tofu_tensor *xW_value = tofu_graph_get_value(xW);     // Can inspect intermediates
```

### Multi-Layer Networks

Chaining creates deeper networks:

```c
// Two-layer network: h = ReLU(x @ W1 + b1), out = h @ W2 + b2
tofu_graph_node *x = tofu_graph_input(g, x_tensor);

// Layer 1: [batch, 784] -> [batch, 128]
tofu_graph_node *W1 = tofu_graph_param(g, W1_tensor);
tofu_graph_node *b1 = tofu_graph_param(g, b1_tensor);
tofu_graph_node *h1 = tofu_graph_matmul(g, x, W1);
tofu_graph_node *h1b = tofu_graph_add(g, h1, b1);
tofu_graph_node *h1a = tofu_graph_relu(g, h1b);

// Layer 2: [batch, 128] -> [batch, 10]
tofu_graph_node *W2 = tofu_graph_param(g, W2_tensor);
tofu_graph_node *b2 = tofu_graph_param(g, b2_tensor);
tofu_graph_node *h2 = tofu_graph_matmul(g, h1a, W2);
tofu_graph_node *out = tofu_graph_add(g, h2, b2);
```

The backward pass automatically computes gradients for all parameters (`W1`, `b1`, `W2`, `b2`) using the chain rule.

### Operation Results

Every operation stores its result immediately:

```c
tofu_graph_node *y = tofu_graph_matmul(g, x, W);

// Result is available immediately
tofu_tensor *result = tofu_graph_get_value(y);
tofu_tensor_print(result, "%.2f");

// The tensor is owned by the node - don't free it!
// It will be freed when you call tofu_graph_free(g)
```

---

## Activation Functions

Activation functions introduce non-linearity, enabling neural networks to learn complex patterns.

### ReLU (Rectified Linear Unit)

ReLU is the most common activation function:

```c
// y = max(0, x)
tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *y = tofu_graph_relu(g, x);

// Values: [-2, -1, 0, 1, 2] -> [0, 0, 0, 1, 2]
```

Properties:
- Simple and efficient: `y = (x > 0) ? x : 0`
- Gradient: 1 where `x > 0`, else 0
- Helps avoid vanishing gradients in deep networks
- Creates sparse activations (many zeros)

Usage pattern in networks:
```c
// Hidden layer with ReLU
tofu_graph_node *h = tofu_graph_matmul(g, x, W);
tofu_graph_node *h_bias = tofu_graph_add(g, h, b);
tofu_graph_node *h_act = tofu_graph_relu(g, h_bias);  // Apply ReLU after bias
```

### Softmax

Softmax converts logits to probabilities for classification:

```c
// Apply softmax along axis 1 (last dimension)
// Input:  [[1, 2, 3], [4, 5, 6]]  (logits)
// Output: [[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]]  (probabilities)

tofu_graph_node *logits = tofu_graph_matmul(g, x, W);
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);
```

Properties:
- Outputs sum to 1.0 along the specified axis
- Numerically stable (subtracts max before exp)
- Used in the final layer for classification
- Axis parameter specifies normalization dimension

Formula: `softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

Multi-class classification example:
```c
// 10-class classifier
tofu_graph_node *x = tofu_graph_input(g, x_tensor);      // [batch, features]
tofu_graph_node *W = tofu_graph_param(g, W_tensor);      // [features, 10]
tofu_graph_node *logits = tofu_graph_matmul(g, x, W);    // [batch, 10]
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);  // [batch, 10]

// probs now contains class probabilities for each sample
```

### Layer Normalization

Layer normalization stabilizes training in deep networks:

```c
// Normalize along axis 1
// out = gamma * (x - mean) / sqrt(var + eps) + beta

tofu_graph_node *x = tofu_graph_input(g, x_tensor);      // [batch, features]
tofu_graph_node *gamma = tofu_graph_param(g, gamma_tensor);  // [features]
tofu_graph_node *beta = tofu_graph_param(g, beta_tensor);    // [features]

tofu_graph_node *normalized = tofu_graph_layer_norm(g, x, gamma, beta, 1, 1e-5);
```

Parameters:
- `x`: Input tensor
- `gamma`: Scale parameter (can be NULL for no scaling)
- `beta`: Shift parameter (can be NULL for no shift)
- `axis`: Normalization axis (typically last dimension)
- `eps`: Small constant for numerical stability (typically 1e-5)

Properties:
- Normalizes activations to zero mean and unit variance
- Helps stabilize training and enables higher learning rates
- Common in transformers and deep networks

Typical usage in transformers:
```c
// Layer norm after self-attention
tofu_graph_node *attn_out = tofu_graph_matmul(g, attn_weights, V);
tofu_graph_node *normed = tofu_graph_layer_norm(g, attn_out, gamma, beta, 1, 1e-5);
```

### Combining Activations

Different activations serve different purposes:

```c
// Multi-layer network with different activations
tofu_graph_node *x = tofu_graph_input(g, x_tensor);

// Hidden layer 1: ReLU for non-linearity
tofu_graph_node *h1 = tofu_graph_matmul(g, x, W1);
tofu_graph_node *h1_bias = tofu_graph_add(g, h1, b1);
tofu_graph_node *h1_act = tofu_graph_relu(g, h1_bias);

// Hidden layer 2: ReLU + Layer Norm
tofu_graph_node *h2 = tofu_graph_matmul(g, h1_act, W2);
tofu_graph_node *h2_bias = tofu_graph_add(g, h2, b2);
tofu_graph_node *h2_act = tofu_graph_relu(g, h2_bias);
tofu_graph_node *h2_norm = tofu_graph_layer_norm(g, h2_act, gamma, beta, 1, 1e-5);

// Output layer: Softmax for classification
tofu_graph_node *logits = tofu_graph_matmul(g, h2_norm, W_out);
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);
```

---

## Shape Operations

Shape operations manipulate tensor dimensions without changing the underlying data.

### Reshape

Reshape changes tensor dimensions while preserving total elements:

```c
// Flatten: [batch, height, width, channels] -> [batch, height * width * channels]
int batch = 32;
int h = 28, w = 28, c = 1;
int flat_dim = h * w * c;  // 784

tofu_graph_node *img = tofu_graph_input(g, img_tensor);  // [32, 28, 28, 1]
tofu_graph_node *flat = tofu_graph_reshape(g, img, 2, (int[]){batch, flat_dim});  // [32, 784]
```

Properties:
- View operation (no data copy)
- Total number of elements must remain constant
- Useful for transitioning between convolutional and fully-connected layers

Common patterns:
```c
// Flatten for fully-connected layer
tofu_graph_node *flat = tofu_graph_reshape(g, x, 2, (int[]){batch, -1});

// Unflatten for visualization
tofu_graph_node *img = tofu_graph_reshape(g, flat, 4, (int[]){batch, 28, 28, 1});

// Prepare patches for Vision Transformer
tofu_graph_node *patches = tofu_graph_reshape(g, img, 3, (int[]){batch, num_patches, patch_dim});
```

### Transpose

Transpose permutes tensor dimensions:

```c
// Transpose matrix: [m, n] -> [n, m]
tofu_graph_node *W = tofu_graph_param(g, W_tensor);      // [784, 10]
tofu_graph_node *W_T = tofu_graph_transpose(g, W, NULL);  // [10, 784]

// NULL means reverse dimension order
```

With explicit axis permutation:
```c
// Permute: [batch, seq, features] -> [batch, features, seq]
int axes[] = {0, 2, 1};
tofu_graph_node *x = tofu_graph_input(g, x_tensor);       // [32, 100, 64]
tofu_graph_node *x_T = tofu_graph_transpose(g, x, axes);  // [32, 64, 100]
```

Common usage in attention:
```c
// Attention: Q @ K^T
tofu_graph_node *Q = tofu_graph_matmul(g, x, W_q);      // [batch, seq, dim]
tofu_graph_node *K = tofu_graph_matmul(g, x, W_k);      // [batch, seq, dim]
tofu_graph_node *K_T = tofu_graph_transpose(g, K, NULL);  // [batch, dim, seq]
tofu_graph_node *scores = tofu_graph_matmul(g, Q, K_T);   // [batch, seq, seq]
```

### Mean Reduction

Compute mean along specified axes (coming soon - API being finalized).

### Sum Reduction

Compute sum along specified axes (coming soon - API being finalized).

### Combining Shape Operations

Shape operations often work together:

```c
// Vision Transformer patch embedding
// Input: [batch, height, width, channels]
// Output: [batch, num_patches, embed_dim]

tofu_graph_node *img = tofu_graph_input(g, img_tensor);  // [32, 224, 224, 3]

// Step 1: Reshape to patches
int patch_size = 16;
int num_patches = (224 / patch_size) * (224 / patch_size);  // 196
int patch_dim = patch_size * patch_size * 3;  // 768

tofu_graph_node *patches = tofu_graph_reshape(g, img, 2,
    (int[]){32, num_patches, patch_dim});  // [32, 196, 768]

// Step 2: Project patches to embedding dimension
tofu_graph_node *W_proj = tofu_graph_param(g, W_proj_tensor);  // [768, 512]
tofu_graph_node *embeddings = tofu_graph_matmul(g, patches, W_proj);  // [32, 196, 512]
```

---

**(Part 1 of 2)**
--- (Part 2 of 2) ---

## 8. Loss Functions

Loss functions quantify how well your model performs by comparing predictions against ground truth. Tofu provides two essential loss functions optimized for different tasks.

### Mean Squared Error (MSE)

MSE measures the average squared difference between predictions and targets. Use it for regression tasks where you predict continuous values.

```c
tofu_graph_node* tofu_graph_mse_loss(tofu_graph* g,
                                      tofu_graph_node* pred,
                                      tofu_graph_node* target);
```

**Mathematical definition:**
```
MSE = mean((pred - target)²)
```

**When to use MSE:**
- Regression problems (predicting house prices, temperatures, etc.)
- When output values are continuous and unbounded
- When you want to penalize larger errors more heavily (squared term)

**Example: Linear regression**
```c
// Predict continuous values
tofu_graph_node* x = tofu_graph_input(g, input_tensor);
tofu_graph_node* W = tofu_graph_param(g, weights);
tofu_graph_node* b = tofu_graph_param(g, bias);

tofu_graph_node* pred = tofu_graph_add(g, tofu_graph_matmul(g, x, W), b);
tofu_graph_node* target = tofu_graph_input(g, target_tensor);

// MSE loss for regression
tofu_graph_node* loss = tofu_graph_mse_loss(g, pred, target);
tofu_graph_backward(g, loss);
```

**Key properties:**
- Output is a scalar (single value)
- Gradients scale linearly with error magnitude
- Sensitive to outliers due to squaring
- Always non-negative

### Cross-Entropy Loss

Cross-entropy measures the difference between predicted and true probability distributions. Use it for classification tasks.

```c
tofu_graph_node* tofu_graph_ce_loss(tofu_graph* g,
                                     tofu_graph_node* pred,
                                     tofu_graph_node* target);
```

**Mathematical definition:**
```
CE = -sum(target * log(pred))
```

**When to use cross-entropy:**
- Classification problems (image recognition, sentiment analysis)
- When outputs represent class probabilities
- Multi-class or binary classification tasks

**Example: Multi-class classification**
```c
// Predict class probabilities
tofu_graph_node* x = tofu_graph_input(g, input_tensor);
tofu_graph_node* W = tofu_graph_param(g, weights);
tofu_graph_node* b = tofu_graph_param(g, bias);

// Forward pass: logits -> softmax -> probabilities
tofu_graph_node* logits = tofu_graph_add(g, tofu_graph_matmul(g, x, W), b);
tofu_graph_node* probs = tofu_graph_softmax(g, logits, 1);  // axis=1 for batch

// Target should be one-hot encoded: [0, 1, 0, 0] for class 1
tofu_graph_node* target = tofu_graph_input(g, target_one_hot);

// Cross-entropy loss
tofu_graph_node* loss = tofu_graph_ce_loss(g, probs, target);
tofu_graph_backward(g, loss);
```

**Target format:**
Targets must be one-hot encoded vectors:
```c
// For batch_size=2, num_classes=4
// If sample 0 is class 2 and sample 1 is class 0:
float target_data[] = {
    0.0f, 0.0f, 1.0f, 0.0f,  // Sample 0: class 2
    1.0f, 0.0f, 0.0f, 0.0f   // Sample 1: class 0
};
tofu_tensor* target = tofu_tensor_create(target_data, 2,
                                          (int[]){2, 4}, TOFU_FLOAT);
```

**Key properties:**
- Numerically stable implementation (avoids log(0))
- Works well with softmax activation
- Penalizes confident wrong predictions heavily
- Output is a scalar (averaged over batch)

### Loss Function Comparison

| Property | MSE Loss | Cross-Entropy Loss |
|----------|----------|-------------------|
| Use case | Regression | Classification |
| Output type | Continuous values | Probabilities (0-1) |
| Activation | Linear or ReLU | Softmax |
| Gradient behavior | Linear with error | Exponential confidence penalty |
| Outlier sensitivity | High (squared) | Moderate (logarithmic) |

---

## 9. Forward Pass

The forward pass computes outputs by propagating data through your graph from inputs to loss. Results are automatically stored in each node.

### Accessing Results with `get_value`

After building your graph, each node contains its computed result in the `value` field. Access it using:

```c
tofu_tensor* tofu_graph_get_value(tofu_graph_node* node);
```

**Important:** The returned tensor is owned by the node. Never free it yourself.

**Example: Inspecting intermediate activations**
```c
// Build network
tofu_graph_node* x = tofu_graph_input(g, input_tensor);
tofu_graph_node* W1 = tofu_graph_param(g, weights1);
tofu_graph_node* h = tofu_graph_relu(g, tofu_graph_matmul(g, x, W1));

// Access hidden layer activations
tofu_tensor* hidden_values = tofu_graph_get_value(h);
printf("Hidden layer statistics:\n");
tofu_tensor_print(hidden_values, "%.4f");
```

### Typical Forward Pass Pattern

```c
// 1. Create inputs
tofu_graph_node* x = tofu_graph_input(g, input_data);
tofu_graph_node* target = tofu_graph_input(g, target_data);

// 2. Add parameters
tofu_graph_node* W = tofu_graph_param(g, weights);
tofu_graph_node* b = tofu_graph_param(g, bias);

// 3. Build computation graph
tofu_graph_node* logits = tofu_graph_add(g, tofu_graph_matmul(g, x, W), b);
tofu_graph_node* probs = tofu_graph_softmax(g, logits, 1);

// 4. Compute loss
tofu_graph_node* loss = tofu_graph_ce_loss(g, probs, target);

// 5. Access results
tofu_tensor* loss_value = tofu_graph_get_value(loss);
tofu_tensor* predictions = tofu_graph_get_value(probs);
```

### Reading Loss Values

Loss is typically a scalar tensor (single element):

```c
tofu_graph_node* loss = tofu_graph_mse_loss(g, pred, target);
tofu_tensor* loss_tensor = tofu_graph_get_value(loss);

// Extract scalar value
float loss_value;
TOFU_TENSOR_DATA_TO(loss_tensor, 0, loss_value, TOFU_FLOAT);
printf("Training loss: %.6f\n", loss_value);
```

### Making Predictions

For inference, forward pass without computing loss:

```c
// Inference mode (no target needed)
tofu_graph_node* x = tofu_graph_input(g, test_input);
tofu_graph_node* W = tofu_graph_param(g, trained_weights);
tofu_graph_node* b = tofu_graph_param(g, trained_bias);

tofu_graph_node* pred = tofu_graph_softmax(g,
    tofu_graph_add(g, tofu_graph_matmul(g, x, W), b), 1);

// Get predictions
tofu_tensor* predictions = tofu_graph_get_value(pred);

// Find class with highest probability
int pred_class = 0;
float max_prob = -1.0f;
for (int i = 0; i < num_classes; i++) {
    float prob;
    TOFU_TENSOR_DATA_TO(predictions, i, prob, TOFU_FLOAT);
    if (prob > max_prob) {
        max_prob = prob;
        pred_class = i;
    }
}
```

---

## 10. Backward Pass

The backward pass computes gradients using reverse-mode automatic differentiation (backpropagation). This enables training via gradient descent.

### Understanding Backpropagation

When you call `tofu_graph_backward()`, the graph computes how changes to each parameter affect the loss using the chain rule:

```
∂loss/∂W = ∂loss/∂output × ∂output/∂W
```

The algorithm:
1. Starts from the loss node (scalar)
2. Propagates gradients backward through operations
3. Accumulates gradients at parameter nodes
4. Stores results in `node->grad`

### Calling Backward

```c
void tofu_graph_backward(tofu_graph* g, tofu_graph_node* loss);
```

**Requirements:**
- `loss` must be a scalar (single element tensor)
- Call after forward pass completes
- Gradients accumulate with each call

**Example: Training iteration**
```c
// Forward pass
tofu_graph_node* x = tofu_graph_input(g, batch_data);
tofu_graph_node* W = tofu_graph_param(g, weights);
tofu_graph_node* pred = tofu_graph_matmul(g, x, W);
tofu_graph_node* target = tofu_graph_input(g, batch_targets);
tofu_graph_node* loss = tofu_graph_mse_loss(g, pred, target);

// Backward pass - computes all gradients
tofu_graph_backward(g, loss);

// Now W->grad contains ∂loss/∂W
```

### Accessing Gradients with `get_grad`

After backward pass, retrieve gradients from parameter nodes:

```c
tofu_tensor* tofu_graph_get_grad(tofu_graph_node* node);
```

**Returns:** Pointer to gradient tensor, or NULL if backward hasn't been called yet.

**Important:** The returned tensor is owned by the node. Never free it yourself.

**Example: Manual parameter update**
```c
tofu_graph_backward(g, loss);

// Get gradient
tofu_tensor* W_grad = tofu_graph_get_grad(W);
tofu_tensor* W_value = tofu_graph_get_value(W);

// Manual SGD update: W = W - learning_rate * grad
float lr = 0.01f;
for (int i = 0; i < W_value->len; i++) {
    float w, grad;
    TOFU_TENSOR_DATA_TO(W_value, i, w, TOFU_FLOAT);
    TOFU_TENSOR_DATA_TO(W_grad, i, grad, TOFU_FLOAT);

    float updated = w - lr * grad;
    TOFU_TENSOR_DATA_FROM(&updated, W_value, i, TOFU_FLOAT);
}
```

### Zeroing Gradients with `zero_grad`

Gradients accumulate by default. Always zero them before each training iteration:

```c
void tofu_graph_zero_grad(tofu_graph* g);
```

**Why this matters:**
```c
// WRONG: Gradients accumulate forever
for (int epoch = 0; epoch < 100; epoch++) {
    tofu_graph_backward(g, loss);  // Adds to existing gradients!
    tofu_optimizer_step(opt);
}

// CORRECT: Clear gradients each iteration
for (int epoch = 0; epoch < 100; epoch++) {
    tofu_graph_zero_grad(g);       // Start fresh
    tofu_graph_backward(g, loss);  // Compute gradients
    tofu_optimizer_step(opt);      // Update parameters
}
```

### Gradient Accumulation (Advanced)

Sometimes you intentionally want gradients to accumulate across multiple batches:

```c
// Simulate larger batch by accumulating gradients
int accumulation_steps = 4;

for (int step = 0; step < accumulation_steps; step++) {
    // Forward pass on mini-batch
    tofu_graph_node* loss = compute_loss(g, mini_batches[step]);

    // Accumulate gradients (don't zero between mini-batches)
    tofu_graph_backward(g, loss);

    if (step < accumulation_steps - 1) {
        tofu_graph_clear_ops(g);  // Clear graph but keep gradients
    }
}

// Update once with accumulated gradients
tofu_optimizer_step(opt);

// Now zero for next iteration
tofu_graph_zero_grad(g);
```

### Complete Backward Pass Example

```c
// Training loop with proper gradient handling
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // 1. Zero gradients from previous iteration
    tofu_graph_zero_grad(g);

    // 2. Forward pass
    tofu_graph_node* x = tofu_graph_input(g, train_data);
    tofu_graph_node* W = tofu_graph_param(g, weights);
    tofu_graph_node* pred = tofu_graph_matmul(g, x, W);
    tofu_graph_node* target = tofu_graph_input(g, train_targets);
    tofu_graph_node* loss = tofu_graph_mse_loss(g, pred, target);

    // 3. Backward pass
    tofu_graph_backward(g, loss);

    // 4. Check gradients (debugging)
    tofu_tensor* W_grad = tofu_graph_get_grad(W);
    if (W_grad) {
        float grad_norm = 0.0f;
        for (int i = 0; i < W_grad->len; i++) {
            float g;
            TOFU_TENSOR_DATA_TO(W_grad, i, g, TOFU_FLOAT);
            grad_norm += g * g;
        }
        printf("Gradient norm: %.6f\n", sqrtf(grad_norm));
    }

    // 5. Update parameters (use optimizer in practice)
    tofu_optimizer_step(optimizer);

    // 6. Clear operations for next iteration
    tofu_graph_clear_ops(g);
}
```

### Debugging Gradients

Common issues and solutions:

**Vanishing gradients** (gradients near zero):
```c
tofu_tensor* grad = tofu_graph_get_grad(W);
float max_grad = 0.0f;
for (int i = 0; i < grad->len; i++) {
    float g;
    TOFU_TENSOR_DATA_TO(grad, i, g, TOFU_FLOAT);
    if (fabsf(g) > max_grad) max_grad = fabsf(g);
}
if (max_grad < 1e-7f) {
    printf("WARNING: Vanishing gradients detected\n");
}
```

**Exploding gradients** (gradients too large):
```c
if (max_grad > 100.0f) {
    printf("WARNING: Exploding gradients detected\n");
    // Consider gradient clipping or reducing learning rate
}
```

---

## 11. Memory and Ownership

Understanding memory management is critical for correct and leak-free code.

### Ownership Rules (CRITICAL)

**Rule 1: Graph does NOT own input/parameter tensors**
```c
tofu_tensor* W = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_graph_node* W_node = tofu_graph_param(g, W);

// You still own W! Must free it after graph_free
tofu_graph_free(g);
tofu_tensor_free_data_too(W);  // Your responsibility
```

**Rule 2: Graph OWNS intermediate operation results**
```c
tofu_graph_node* result = tofu_graph_matmul(g, x, W);
// result->value is owned by the graph
// Don't free it - tofu_graph_free() handles it

tofu_graph_free(g);  // Frees result->value automatically
```

**Rule 3: Graph OWNS all nodes**
```c
tofu_graph_node* node = tofu_graph_relu(g, x);
// Don't free node - graph owns it

tofu_graph_free(g);  // Frees all nodes
```

**Rule 4: Never free tensors returned by get_value/get_grad**
```c
tofu_tensor* value = tofu_graph_get_value(node);   // Node owns this
tofu_tensor* grad = tofu_graph_get_grad(node);     // Node owns this

// WRONG: tofu_tensor_free(value);  // CRASH!
// CORRECT: Just use the tensor, don't free it
```

### Complete Cleanup Pattern

```c
// 1. Allocate parameter tensors
tofu_tensor* W = tofu_tensor_zeros(2, (int[]){64, 32}, TOFU_FLOAT);
tofu_tensor* b = tofu_tensor_zeros(1, (int[]){32}, TOFU_FLOAT);

// 2. Create graph
tofu_graph* g = tofu_graph_create();

// 3. Add parameters to graph
tofu_graph_node* W_node = tofu_graph_param(g, W);
tofu_graph_node* b_node = tofu_graph_param(g, b);

// 4. Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Allocate batch data (you manage this)
    float* batch_data = load_batch(epoch);
    tofu_tensor* x_tensor = tofu_tensor_create(batch_data, 2,
                                                (int[]){32, 64}, TOFU_FLOAT);

    // Build graph (operations owned by graph)
    tofu_graph_node* x = tofu_graph_input(g, x_tensor);
    tofu_graph_node* out = tofu_graph_add(g,
                                tofu_graph_matmul(g, x, W_node), b_node);
    // ... training ...

    // Free batch resources (you own these)
    tofu_tensor_free(x_tensor);
    free(batch_data);

    // Clear operations but keep parameters
    tofu_graph_clear_ops(g);
}

// 5. Cleanup (CRITICAL ORDER!)
tofu_graph_free(g);               // Graph owns: nodes, ops, gradients
tofu_tensor_free_data_too(W);     // You own: parameter tensors
tofu_tensor_free_data_too(b);
```

### Memory Management with Optimizers

```c
// Create optimizer (holds references to graph and parameters)
tofu_optimizer* opt = tofu_optimizer_adam_create(g, 0.001, 0.9, 0.999, 1e-8);

// Training...

// Cleanup order matters!
tofu_optimizer_free(opt);         // 1. Free optimizer first
tofu_graph_free(g);               // 2. Then graph
tofu_tensor_free_data_too(W);     // 3. Then parameter tensors
tofu_tensor_free_data_too(b);
```

### Common Memory Pitfalls

**Pitfall 1: Freeing parameter tensors too early**
```c
// WRONG
tofu_tensor* W = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_graph_node* W_node = tofu_graph_param(g, W);
tofu_tensor_free_data_too(W);  // DON'T DO THIS! Graph needs it

tofu_graph_backward(g, loss);  // CRASH: W is freed but graph uses it
```

**Pitfall 2: Freeing operation results**
```c
// WRONG
tofu_graph_node* result = tofu_graph_matmul(g, x, W);
tofu_tensor* result_value = tofu_graph_get_value(result);
tofu_tensor_free(result_value);  // CRASH! Graph owns this
```

**Pitfall 3: Forgetting to free parameter tensors**
```c
// Memory leak
tofu_tensor* W = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_graph_node* W_node = tofu_graph_param(g, W);

tofu_graph_free(g);  // Graph freed but W still allocated!
// Missing: tofu_tensor_free_data_too(W);
```

**Pitfall 4: Double-free via clear_ops**
```c
// WRONG
tofu_graph_node* x = tofu_graph_input(g, input_tensor);
tofu_graph_clear_ops(g);  // Removes x node
tofu_tensor_free(input_tensor);  // OK

tofu_tensor_free(input_tensor);  // CRASH: Double free!
```

### Batch Processing Memory Pattern

Efficient pattern for processing multiple batches:

```c
tofu_graph* g = tofu_graph_create();

// Parameters persist across batches
tofu_tensor* W = tofu_tensor_zeros(2, (int[]){64, 32}, TOFU_FLOAT);
tofu_graph_node* W_node = tofu_graph_param(g, W);

for (int batch = 0; batch < num_batches; batch++) {
    // Allocate batch-specific data
    float* batch_data = malloc(batch_size * 64 * sizeof(float));
    load_batch_data(batch_data, batch);

    tofu_tensor* x_tensor = tofu_tensor_create(batch_data, 2,
                                                (int[]){batch_size, 64}, TOFU_FLOAT);

    // Build graph for this batch
    tofu_graph_node* x = tofu_graph_input(g, x_tensor);
    tofu_graph_node* out = tofu_graph_matmul(g, x, W_node);
    // ... compute loss, backward, optimize ...

    // Free batch resources
    tofu_tensor_free(x_tensor);  // Free tensor wrapper
    free(batch_data);            // Free data buffer

    // Clear operations (keeps W_node!)
    tofu_graph_clear_ops(g);
}

// Final cleanup
tofu_graph_free(g);
tofu_tensor_free_data_too(W);
```

---

## 12. Building Complex Networks

Move beyond single layers to build sophisticated architectures.

### Multi-Layer Perceptron (MLP)

A complete MLP with multiple hidden layers:

```c
typedef struct {
    tofu_tensor *W1, *b1;  // Input -> Hidden1
    tofu_tensor *W2, *b2;  // Hidden1 -> Hidden2
    tofu_tensor *W3, *b3;  // Hidden2 -> Output
} MLP;

// Initialize weights with Xavier/He initialization
MLP* mlp_create(int input_dim, int hidden1, int hidden2, int output_dim) {
    MLP* mlp = malloc(sizeof(MLP));

    // Layer 1: input_dim -> hidden1
    mlp->W1 = tofu_tensor_zeros(2, (int[]){input_dim, hidden1}, TOFU_FLOAT);
    mlp->b1 = tofu_tensor_zeros(1, (int[]){hidden1}, TOFU_FLOAT);

    // Initialize W1 with Xavier: uniform(-sqrt(6/n), sqrt(6/n))
    float limit1 = sqrtf(6.0f / input_dim);
    for (int i = 0; i < mlp->W1->len; i++) {
        float val = (2.0f * rand() / RAND_MAX - 1.0f) * limit1;
        TOFU_TENSOR_DATA_FROM(&val, mlp->W1, i, TOFU_FLOAT);
    }

    // Layer 2: hidden1 -> hidden2
    mlp->W2 = tofu_tensor_zeros(2, (int[]){hidden1, hidden2}, TOFU_FLOAT);
    mlp->b2 = tofu_tensor_zeros(1, (int[]){hidden2}, TOFU_FLOAT);

    float limit2 = sqrtf(6.0f / hidden1);
    for (int i = 0; i < mlp->W2->len; i++) {
        float val = (2.0f * rand() / RAND_MAX - 1.0f) * limit2;
        TOFU_TENSOR_DATA_FROM(&val, mlp->W2, i, TOFU_FLOAT);
    }

    // Layer 3: hidden2 -> output_dim
    mlp->W3 = tofu_tensor_zeros(2, (int[]){hidden2, output_dim}, TOFU_FLOAT);
    mlp->b3 = tofu_tensor_zeros(1, (int[]){output_dim}, TOFU_FLOAT);

    float limit3 = sqrtf(6.0f / hidden2);
    for (int i = 0; i < mlp->W3->len; i++) {
        float val = (2.0f * rand() / RAND_MAX - 1.0f) * limit3;
        TOFU_TENSOR_DATA_FROM(&val, mlp->W3, i, TOFU_FLOAT);
    }

    return mlp;
}

// Forward pass
tofu_graph_node* mlp_forward(tofu_graph* g, tofu_graph_node* x, MLP* mlp) {
    // Add parameters to graph
    tofu_graph_node* W1 = tofu_graph_param(g, mlp->W1);
    tofu_graph_node* b1 = tofu_graph_param(g, mlp->b1);
    tofu_graph_node* W2 = tofu_graph_param(g, mlp->W2);
    tofu_graph_node* b2 = tofu_graph_param(g, mlp->b2);
    tofu_graph_node* W3 = tofu_graph_param(g, mlp->W3);
    tofu_graph_node* b3 = tofu_graph_param(g, mlp->b3);

    // Layer 1: x @ W1 + b1 -> ReLU
    tofu_graph_node* h1 = tofu_graph_matmul(g, x, W1);
    h1 = tofu_graph_add(g, h1, b1);
    h1 = tofu_graph_relu(g, h1);

    // Layer 2: h1 @ W2 + b2 -> ReLU
    tofu_graph_node* h2 = tofu_graph_matmul(g, h1, W2);
    h2 = tofu_graph_add(g, h2, b2);
    h2 = tofu_graph_relu(g, h2);

    // Layer 3: h2 @ W3 + b3 (logits)
    tofu_graph_node* out = tofu_graph_matmul(g, h2, W3);
    out = tofu_graph_add(g, out, b3);

    return out;
}

// Cleanup
void mlp_free(MLP* mlp) {
    tofu_tensor_free_data_too(mlp->W1);
    tofu_tensor_free_data_too(mlp->b1);
    tofu_tensor_free_data_too(mlp->W2);
    tofu_tensor_free_data_too(mlp->b2);
    tofu_tensor_free_data_too(mlp->W3);
    tofu_tensor_free_data_too(mlp->b3);
    free(mlp);
}

// Usage
MLP* model = mlp_create(784, 256, 128, 10);  // MNIST-style
tofu_graph* g = tofu_graph_create();

for (int epoch = 0; epoch < 100; epoch++) {
    tofu_graph_node* x = tofu_graph_input(g, batch_data);
    tofu_graph_node* logits = mlp_forward(g, x, model);
    tofu_graph_node* probs = tofu_graph_softmax(g, logits, 1);
    tofu_graph_node* target = tofu_graph_input(g, batch_targets);
    tofu_graph_node* loss = tofu_graph_ce_loss(g, probs, target);

    tofu_graph_backward(g, loss);
    tofu_optimizer_step(optimizer);
    tofu_graph_clear_ops(g);
}

mlp_free(model);
tofu_graph_free(g);
```

### Residual Connections

Residual connections (skip connections) help training deep networks:

```c
// Residual block: output = ReLU(x + F(x))
tofu_graph_node* residual_block(tofu_graph* g, tofu_graph_node* x,
                                 tofu_tensor* W1, tofu_tensor* b1,
                                 tofu_tensor* W2, tofu_tensor* b2) {
    // Main path F(x)
    tofu_graph_node* W1_node = tofu_graph_param(g, W1);
    tofu_graph_node* b1_node = tofu_graph_param(g, b1);
    tofu_graph_node* W2_node = tofu_graph_param(g, W2);
    tofu_graph_node* b2_node = tofu_graph_param(g, b2);

    // F(x) = W2 @ ReLU(W1 @ x + b1) + b2
    tofu_graph_node* h1 = tofu_graph_matmul(g, x, W1_node);
    h1 = tofu_graph_add(g, h1, b1_node);
    h1 = tofu_graph_relu(g, h1);

    tofu_graph_node* h2 = tofu_graph_matmul(g, h1, W2_node);
    h2 = tofu_graph_add(g, h2, b2_node);

    // Skip connection: x + F(x)
    tofu_graph_node* residual = tofu_graph_add(g, x, h2);

    // Final activation
    return tofu_graph_relu(g, residual);
}

// Stack multiple residual blocks
tofu_graph_node* x = tofu_graph_input(g, input_tensor);
x = residual_block(g, x, W1a, b1a, W1b, b1b);
x = residual_block(g, x, W2a, b2a, W2b, b2b);
x = residual_block(g, x, W3a, b3a, W3b, b3b);
tofu_graph_node* out = tofu_graph_matmul(g, x, W_out);
```

### Custom Layer Abstractions

Encapsulate common patterns:

```c
// Linear layer: y = x @ W + b
typedef struct {
    tofu_tensor* W;
    tofu_tensor* b;
} LinearLayer;

LinearLayer* linear_create(int in_features, int out_features) {
    LinearLayer* layer = malloc(sizeof(LinearLayer));
    layer->W = tofu_tensor_zeros(2, (int[]){in_features, out_features}, TOFU_FLOAT);
    layer->b = tofu_tensor_zeros(1, (int[]){out_features}, TOFU_FLOAT);

    // Initialize weights
    float limit = sqrtf(6.0f / in_features);
    for (int i = 0; i < layer->W->len; i++) {
        float val = (2.0f * rand() / RAND_MAX - 1.0f) * limit;
        TOFU_TENSOR_DATA_FROM(&val, layer->W, i, TOFU_FLOAT);
    }

    return layer;
}

tofu_graph_node* linear_forward(tofu_graph* g, tofu_graph_node* x, LinearLayer* layer) {
    tofu_graph_node* W = tofu_graph_param(g, layer->W);
    tofu_graph_node* b = tofu_graph_param(g, layer->b);
    tofu_graph_node* out = tofu_graph_matmul(g, x, W);
    return tofu_graph_add(g, out, b);
}

void linear_free(LinearLayer* layer) {
    tofu_tensor_free_data_too(layer->W);
    tofu_tensor_free_data_too(layer->b);
    free(layer);
}

// Build network with layer abstractions
LinearLayer* fc1 = linear_create(784, 256);
LinearLayer* fc2 = linear_create(256, 10);

tofu_graph_node* x = tofu_graph_input(g, input);
x = linear_forward(g, x, fc1);
x = tofu_graph_relu(g, x);
x = linear_forward(g, x, fc2);
tofu_graph_node* probs = tofu_graph_softmax(g, x, 1);
```

### Transformer-Style Attention (Simplified)

Basic attention mechanism pattern:

```c
// Simplified attention: softmax(Q @ K^T) @ V
tofu_graph_node* attention(tofu_graph* g,
                           tofu_graph_node* Q,  // Query
                           tofu_graph_node* K,  // Key
                           tofu_graph_node* V)  // Value
{
    // 1. Compute attention scores: Q @ K^T
    tofu_graph_node* K_T = tofu_graph_transpose(g, K, NULL);
    tofu_graph_node* scores = tofu_graph_matmul(g, Q, K_T);

    // 2. Softmax over last dimension
    tofu_graph_node* attn_weights = tofu_graph_softmax(g, scores, -1);

    // 3. Apply attention: attn_weights @ V
    tofu_graph_node* output = tofu_graph_matmul(g, attn_weights, V);

    return output;
}
```

---

## 13. Best Practices

Guidelines for robust and maintainable graph-based code.

### Graph Design Principles

**1. Separate model structure from training logic**
```c
// Good: Model is a reusable structure
typedef struct {
    tofu_tensor *W1, *b1, *W2, *b2;
} Model;

tofu_graph_node* model_forward(tofu_graph* g, tofu_graph_node* x, Model* m);

// Train the model
void train(Model* model, Dataset* data) {
    tofu_graph* g = tofu_graph_create();
    // Training loop uses model_forward()
    tofu_graph_free(g);
}
```

**2. Use clear_ops between iterations**
```c
// Efficient: Reuse graph structure
for (int batch = 0; batch < num_batches; batch++) {
    // Build graph for this batch
    tofu_graph_node* loss = build_forward_graph(g, batch);
    tofu_graph_backward(g, loss);
    tofu_optimizer_step(opt);

    // Clear ops but keep parameters
    tofu_graph_clear_ops(g);
}
```

**3. Always check tensor shapes during development**
```c
tofu_graph_node* result = tofu_graph_matmul(g, x, W);
tofu_tensor* result_tensor = tofu_graph_get_value(result);

printf("Result shape: [");
for (int i = 0; i < result_tensor->ndim; i++) {
    printf("%d%s", result_tensor->dims[i],
           i < result_tensor->ndim - 1 ? ", " : "");
}
printf("]\n");
```

### Debugging Strategies

**Monitor loss values**
```c
tofu_tensor* loss_tensor = tofu_graph_get_value(loss);
float loss_val;
TOFU_TENSOR_DATA_TO(loss_tensor, 0, loss_val, TOFU_FLOAT);

if (isnan(loss_val) || isinf(loss_val)) {
    printf("ERROR: Loss is NaN or Inf at epoch %d\n", epoch);
    // Check gradients, learning rate, or input data
}

if (loss_val > prev_loss * 2.0f) {
    printf("WARNING: Loss spiked at epoch %d\n", epoch);
    // Consider reducing learning rate
}
```

**Check gradient magnitudes**
```c
void print_gradient_stats(tofu_graph_node* param, const char* name) {
    tofu_tensor* grad = tofu_graph_get_grad(param);
    if (!grad) return;

    float min = 1e9f, max = -1e9f, sum = 0.0f;
    for (int i = 0; i < grad->len; i++) {
        float g;
        TOFU_TENSOR_DATA_TO(grad, i, g, TOFU_FLOAT);
        if (g < min) min = g;
        if (g > max) max = g;
        sum += g * g;
    }

    printf("%s grad: min=%.6f, max=%.6f, norm=%.6f\n",
           name, min, max, sqrtf(sum));
}
```

**Validate forward pass outputs**
```c
tofu_tensor* probs = tofu_graph_get_value(softmax_node);

// Check probabilities sum to 1
float sum = 0.0f;
for (int i = 0; i < probs->dims[1]; i++) {
    float p;
    TOFU_TENSOR_DATA_TO(probs, i, p, TOFU_FLOAT);
    sum += p;
}

if (fabsf(sum - 1.0f) > 1e-5f) {
    printf("WARNING: Probabilities don't sum to 1: %.6f\n", sum);
}
```

### Performance Tips

**1. Batch your data**
```c
// Slow: Process one sample at a time
for (int i = 0; i < 1000; i++) {
    tofu_graph_node* x = tofu_graph_input(g, single_samples[i]);
    // ... forward, backward, update ...
}

// Fast: Process batches
int batch_size = 32;
for (int i = 0; i < 1000; i += batch_size) {
    tofu_graph_node* x = tofu_graph_input(g, batched_samples[i/batch_size]);
    // ... forward, backward, update ...
}
```

**2. Reuse graph structure**
```c
// Less efficient: Create new graph each iteration
for (int epoch = 0; epoch < 100; epoch++) {
    tofu_graph* g = tofu_graph_create();
    // ... train ...
    tofu_graph_free(g);
}

// More efficient: Reuse graph
tofu_graph* g = tofu_graph_create();
for (int epoch = 0; epoch < 100; epoch++) {
    // ... train ...
    tofu_graph_clear_ops(g);
}
tofu_graph_free(g);
```

**3. Profile your code**
```c
#include <time.h>

clock_t start = clock();
tofu_graph_backward(g, loss);
clock_t end = clock();

double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
printf("Backward pass: %.2f ms\n", time_ms);
```

### Common Pitfalls to Avoid

1. **Forgetting to zero gradients**
   - Always call `tofu_graph_zero_grad()` before backward pass

2. **Freeing tensors too early**
   - Don't free parameter tensors until after `tofu_graph_free()`

3. **Wrong loss node**
   - Ensure loss is scalar before calling backward

4. **Shape mismatches**
   - Use `tofu_tensor_print()` to debug shape issues

5. **Learning rate too high**
   - Start with small values (0.001-0.01) and adjust

6. **No validation set**
   - Always evaluate on separate data to detect overfitting

### Complete Training Template

```c
// Complete training example with best practices
void train_model(Dataset* train_data, Dataset* val_data) {
    // Initialize
    tofu_graph* g = tofu_graph_create();
    Model* model = model_create(input_dim, hidden_dim, output_dim);
    tofu_optimizer* opt = tofu_optimizer_adam_create(g, 0.001, 0.9, 0.999, 1e-8);

    float best_val_loss = 1e9f;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Training phase
        float train_loss = 0.0f;
        for (int batch = 0; batch < train_data->num_batches; batch++) {
            tofu_graph_zero_grad(g);

            tofu_graph_node* x = tofu_graph_input(g, train_data->batches[batch].x);
            tofu_graph_node* pred = model_forward(g, x, model);
            tofu_graph_node* target = tofu_graph_input(g, train_data->batches[batch].y);
            tofu_graph_node* loss = tofu_graph_ce_loss(g, pred, target);

            tofu_tensor* loss_tensor = tofu_graph_get_value(loss);
            float batch_loss;
            TOFU_TENSOR_DATA_TO(loss_tensor, 0, batch_loss, TOFU_FLOAT);
            train_loss += batch_loss;

            tofu_graph_backward(g, loss);
            tofu_optimizer_step(opt);
            tofu_graph_clear_ops(g);
        }
        train_loss /= train_data->num_batches;

        // Validation phase (no gradient computation)
        float val_loss = 0.0f;
        for (int batch = 0; batch < val_data->num_batches; batch++) {
            tofu_graph_node* x = tofu_graph_input(g, val_data->batches[batch].x);
            tofu_graph_node* pred = model_forward(g, x, model);
            tofu_graph_node* target = tofu_graph_input(g, val_data->batches[batch].y);
            tofu_graph_node* loss = tofu_graph_ce_loss(g, pred, target);

            tofu_tensor* loss_tensor = tofu_graph_get_value(loss);
            float batch_loss;
            TOFU_TENSOR_DATA_TO(loss_tensor, 0, batch_loss, TOFU_FLOAT);
            val_loss += batch_loss;

            tofu_graph_clear_ops(g);
        }
        val_loss /= val_data->num_batches;

        // Logging
        printf("Epoch %3d: train_loss=%.4f, val_loss=%.4f",
               epoch, train_loss, val_loss);

        // Save best model
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            printf(" (best)");
            // Save model weights here
        }
        printf("\n");

        // Early stopping
        if (train_loss < 0.01f && val_loss > train_loss * 2.0f) {
            printf("Early stopping: overfitting detected\n");
            break;
        }
    }

    // Cleanup
    tofu_optimizer_free(opt);
    model_free(model);
    tofu_graph_free(g);
}
```

This completes the computation graphs user guide. You now have the knowledge to build, train, and debug neural networks using Tofu's graph API.
