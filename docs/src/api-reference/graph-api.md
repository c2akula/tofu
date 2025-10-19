# Graph API Reference

The Graph API provides computational graph construction and automatic differentiation for training neural networks. It implements reverse-mode automatic differentiation (backpropagation) for computing gradients.

## Table of Contents

- [Data Structures](#data-structures)
- [Graph Lifecycle](#graph-lifecycle)
- [Leaf Nodes](#leaf-nodes)
- [Operations](#operations)
- [Loss Functions](#loss-functions)
- [Backward Pass](#backward-pass)
- [Utilities](#utilities)
- [Usage Patterns](#usage-patterns)

---

## Data Structures

### `tofu_graph`

The computation graph structure that manages all nodes and their relationships.

```c
struct tofu_graph {
    tofu_graph_node** nodes;         // All nodes in graph
    int num_nodes;                   // Number of nodes
    int capacity;                    // Allocated capacity

    tofu_graph_node** topo_order;    // Nodes in reverse topological order
    int topo_size;                   // Size of topo_order
    int topo_capacity;               // Allocated capacity

    int next_id;                     // Next available node ID
};
```

### `tofu_graph_node`

A node in the computation graph representing an operation or leaf value.

```c
struct tofu_graph_node {
    int id;                          // Unique node ID within graph
    tofu_op_type op;                 // Operation type

    tofu_tensor* value;              // Forward pass result
    tofu_tensor* grad;               // Gradient (∂L/∂value)

    tofu_graph_node** inputs;        // Input nodes
    int num_inputs;                  // Number of inputs
    int capacity_inputs;             // Allocated capacity for inputs

    tofu_backward_fn backward_fn;    // Backward pass function
    void* backward_ctx;              // Context for backward (saved tensors, etc.)

    int requires_grad;               // Does this need gradient computation?
    int visited;                     // For topological sort

    tofu_graph* graph;               // Parent graph
};
```

### Operation Types (`tofu_op_type`)

Enumeration of all supported operations:

- **Leaf nodes:**
  - `TOFU_OP_INPUT` - Input node (no gradient)
  - `TOFU_OP_PARAM` - Trainable parameter (requires gradient)

- **Binary operations:**
  - `TOFU_OP_MATMUL` - Matrix multiplication
  - `TOFU_OP_ADD` - Element-wise addition
  - `TOFU_OP_MUL` - Element-wise multiplication

- **Activations:**
  - `TOFU_OP_RELU` - ReLU activation
  - `TOFU_OP_SOFTMAX` - Softmax activation
  - `TOFU_OP_LAYER_NORM` - Layer normalization

- **Shape operations:**
  - `TOFU_OP_RESHAPE` - Reshape operation
  - `TOFU_OP_TRANSPOSE` - Transpose operation

- **Reductions:**
  - `TOFU_OP_MEAN` - Mean reduction
  - `TOFU_OP_SUM` - Sum reduction

- **Loss functions:**
  - `TOFU_OP_MSE_LOSS` - Mean squared error loss
  - `TOFU_OP_CE_LOSS` - Cross-entropy loss

---

## Graph Lifecycle

### `tofu_graph_create`

Create a new empty computation graph.

```c
tofu_graph* tofu_graph_create(void);
```

**Returns:** Pointer to newly allocated graph (caller owns, must call `tofu_graph_free`)

**Behavior:**
- Graph starts empty - add nodes via `tofu_graph_input`, `tofu_graph_param`, etc.
- Graph does NOT take ownership of tensors passed to `tofu_graph_param`
- Caller must call `tofu_graph_free` to free graph and all nodes

**Example:**
```c
tofu_graph *g = tofu_graph_create();

// Build graph...

tofu_graph_free(g);
```

---

### `tofu_graph_free`

Free computation graph and all nodes.

```c
void tofu_graph_free(tofu_graph* g);
```

**Parameters:**
- `g` - Graph to free (can be NULL, no-op if NULL)

**Behavior:**
- Frees all graph nodes and their gradients
- Frees intermediate operation results (matmul, add, etc.)
- Does NOT free INPUT or PARAM tensors (caller owns them)
- Caller must separately free tensors passed to input/param functions
- Safe to call multiple times (idempotent)

**Ownership Pattern:**
```c
// Create tensors
tofu_tensor *input = tofu_tensor_zeros(2, (int[]){1, 4}, TOFU_FLOAT);
tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);

// Build graph
tofu_graph *g = tofu_graph_create();
tofu_graph_node *x = tofu_graph_input(g, input);
tofu_graph_node *W = tofu_graph_param(g, weights);
// ... more operations ...

// Cleanup (important order!)
tofu_graph_free(g);               // 1. Free graph first
tofu_tensor_free_data_too(input);  // 2. Then free tensors
tofu_tensor_free_data_too(weights);
```

**See also:** `tofu_graph_clear_ops`

---

### `tofu_graph_clear_ops`

Clear all operation nodes but keep parameter nodes.

```c
void tofu_graph_clear_ops(tofu_graph* g);
```

**Parameters:**
- `g` - Graph to clear (cannot be NULL)

**Behavior:**
- Frees all nodes except PARAM and INPUT nodes
- Preserves trainable parameters for next forward pass
- Use between training iterations to reset computation graph

**Use Case:**
```c
tofu_graph *g = tofu_graph_create();

// Add parameters (preserved across iterations)
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *b = tofu_graph_param(g, bias);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Build forward graph for this batch
    tofu_graph_node *x = tofu_graph_input(g, batch_data);
    tofu_graph_node *y = tofu_graph_matmul(g, x, W);
    tofu_graph_node *out = tofu_graph_add(g, y, b);

    // Backward and optimize...

    // Clear operations for next iteration (W and b are preserved)
    tofu_graph_clear_ops(g);
}
```

**Notes:**
- More efficient than creating new graph each iteration
- Parameters maintain their values and gradients
- Violating preconditions triggers `assert()` and crashes

---

## Leaf Nodes

Leaf nodes are the starting points of computation - they have no inputs and represent either data or learnable parameters.

### `tofu_graph_input`

Create input node (non-trainable data source).

```c
tofu_graph_node* tofu_graph_input(tofu_graph* g, tofu_tensor* data);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `data` - Input tensor data (cannot be NULL)

**Returns:** Pointer to newly created graph node (graph owns node, caller owns tensor)

**Behavior:**
- Input nodes do NOT compute gradients
- IMPORTANT: Graph does NOT take ownership of data tensor
- Caller must free data tensor separately after `tofu_graph_free()`
- Use for input data that doesn't require backpropagation

**Example:**
```c
float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
tofu_tensor *x_tensor = tofu_tensor_create(input_data, 2, (int[]){1, 4}, TOFU_FLOAT);

tofu_graph *g = tofu_graph_create();
tofu_graph_node *x = tofu_graph_input(g, x_tensor);

// Use x in graph operations...

tofu_graph_free(g);
tofu_tensor_free(x_tensor);  // Caller must free tensor
```

**Notes:**
- Typical pattern: create tensor → input node → use → graph_free → free tensor
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_graph_param` for trainable parameters

---

### `tofu_graph_param`

Create parameter node (trainable weights/biases).

```c
tofu_graph_node* tofu_graph_param(tofu_graph* g, tofu_tensor* data);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `data` - Parameter tensor data (cannot be NULL)

**Returns:** Pointer to newly created graph node (graph owns node, caller owns tensor)

**Behavior:**
- IMPORTANT: Graph does NOT take ownership of data tensor
- Caller must free data tensor separately after `tofu_graph_free()`
- Parameter nodes compute gradients during backward pass
- Use for trainable weights, biases, etc.

**Example:**
```c
// Create trainable weights
tofu_tensor *W = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_tensor *b = tofu_tensor_zeros(1, (int[]){3}, TOFU_FLOAT);

tofu_graph *g = tofu_graph_create();
tofu_graph_node *W_node = tofu_graph_param(g, W);
tofu_graph_node *b_node = tofu_graph_param(g, b);

// Build network...
// Training loop with backward pass computes W->grad and b->grad

// Cleanup
tofu_graph_free(g);
tofu_tensor_free_data_too(W);  // Caller must free tensors
tofu_tensor_free_data_too(b);
```

**Notes:**
- Typical pattern: create tensor → param node → free tensor after graph_free
- Gradients are stored in the node, accessible via `tofu_graph_get_grad()`
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_graph_input` for non-trainable inputs, `tofu_graph_get_grad` to access gradients

---

## Operations

Operations create new nodes in the graph that compute values during forward pass and gradients during backward pass.

### `tofu_graph_matmul`

Add matrix multiplication node to graph.

```c
tofu_graph_node* tofu_graph_matmul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `a` - Left operand node (cannot be NULL)
- `b` - Right operand node (cannot be NULL)

**Returns:** Pointer to result node (graph owns, freed by `tofu_graph_free`)

**Preconditions:**
- `a->value->dims[last]` must equal `b->value->dims[second-to-last]`

**Behavior:**
- Computes matrix multiplication with broadcasting
- Implements backward pass for gradient computation
- Result node requires gradient if any input requires gradient

**Example:**
```c
// Neural network layer: y = x @ W
tofu_graph_node *x = tofu_graph_input(g, input_tensor);
tofu_graph_node *W = tofu_graph_param(g, weights_tensor);
tofu_graph_node *y = tofu_graph_matmul(g, x, W);
```

**Notes:**
- Most commonly used operation for neural networks
- Follows same semantics as `tofu_tensor_matmul` (see Tensor API)
- Violating preconditions triggers `assert()` and crashes

---

### `tofu_graph_add`

Add element-wise addition node to graph.

```c
tofu_graph_node* tofu_graph_add(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `a` - First operand node (cannot be NULL)
- `b` - Second operand node (cannot be NULL)

**Returns:** Pointer to result node (graph owns, freed by `tofu_graph_free`)

**Preconditions:**
- a and b must be broadcastable (NumPy rules)

**Behavior:**
- Computes element-wise addition with broadcasting
- Implements backward pass for gradient computation

**Example:**
```c
// Add bias: y = x + b
tofu_graph_node *x = tofu_graph_matmul(g, input, weights);
tofu_graph_node *b = tofu_graph_param(g, bias_tensor);
tofu_graph_node *y = tofu_graph_add(g, x, b);
```

**Notes:**
- Supports NumPy-style broadcasting
- Common for adding biases to layer outputs

---

### `tofu_graph_mul`

Add element-wise multiplication node to graph.

```c
tofu_graph_node* tofu_graph_mul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `a` - First operand node (cannot be NULL)
- `b` - Second operand node (cannot be NULL)

**Returns:** Pointer to result node (graph owns, freed by `tofu_graph_free`)

**Preconditions:**
- a and b must be broadcastable (NumPy rules)

**Behavior:**
- Computes element-wise multiplication with broadcasting
- Implements backward pass for gradient computation

**Example:**
```c
// Attention mechanism: scaled dot product
tofu_graph_node *qk = tofu_graph_matmul(g, q, k);
tofu_graph_node *scale = tofu_graph_param(g, scale_tensor);
tofu_graph_node *scaled = tofu_graph_mul(g, qk, scale);
```

---

### `tofu_graph_relu`

Add ReLU activation node to graph.

```c
tofu_graph_node* tofu_graph_relu(tofu_graph* g, tofu_graph_node* x);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `x` - Input node (cannot be NULL)

**Returns:** Pointer to result node (graph owns, freed by `tofu_graph_free`)

**Behavior:**
- Computes ReLU: `max(0, x)`
- Implements backward pass for gradient computation
- Gradient is 1 where `x > 0`, else 0

**Example:**
```c
// Hidden layer with ReLU
tofu_graph_node *h1 = tofu_graph_matmul(g, x, W1);
tofu_graph_node *h1_bias = tofu_graph_add(g, h1, b1);
tofu_graph_node *h1_relu = tofu_graph_relu(g, h1_bias);
```

---

### `tofu_graph_softmax`

Add softmax activation node to graph.

```c
tofu_graph_node* tofu_graph_softmax(tofu_graph* g, tofu_graph_node* x, int axis);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `x` - Input node (cannot be NULL)
- `axis` - Axis along which to apply softmax

**Returns:** Pointer to result node (graph owns, freed by `tofu_graph_free`)

**Preconditions:**
- `axis < x->value->ndim`

**Behavior:**
- Computes softmax along specified axis (exp normalization)
- Implements backward pass for gradient computation
- Numerically stable (subtracts max before exp)

**Example:**
```c
// Classification output layer
tofu_graph_node *logits = tofu_graph_matmul(g, h, W_out);
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);
```

**Notes:**
- Typically used for classification tasks
- Output values sum to 1.0 along specified axis

---

### `tofu_graph_layer_norm`

Add layer normalization node to graph.

```c
tofu_graph_node* tofu_graph_layer_norm(tofu_graph* g, tofu_graph_node* x,
                                       tofu_graph_node* gamma, tofu_graph_node* beta,
                                       int axis, double eps);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `x` - Input node (cannot be NULL)
- `gamma` - Scale parameter node (can be NULL for no scaling)
- `beta` - Shift parameter node (can be NULL for no shift)
- `axis` - Axis along which to normalize
- `eps` - Small constant for numerical stability (typically 1e-5)

**Returns:** Pointer to result node (graph owns, freed by `tofu_graph_free`)

**Preconditions:**
- `axis < x->value->ndim`
- `eps > 0`

**Behavior:**
- Normalizes: `(x - mean) / sqrt(variance + eps)`
- Then applies: `gamma * normalized + beta` (if gamma/beta non-NULL)
- Implements backward pass for gradient computation

**Example:**
```c
// Transformer-style layer norm
tofu_graph_node *gamma = tofu_graph_param(g, gamma_tensor);
tofu_graph_node *beta = tofu_graph_param(g, beta_tensor);
tofu_graph_node *normalized = tofu_graph_layer_norm(g, x, gamma, beta, 1, 1e-5);
```

**Notes:**
- Common in transformer architectures
- Helps stabilize training of deep networks

---

### `tofu_graph_reshape`

Add reshape node to graph.

```c
tofu_graph_node* tofu_graph_reshape(tofu_graph* g, tofu_graph_node* x, int ndim, const int* dims);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `x` - Input node (cannot be NULL)
- `ndim` - Number of dimensions for reshaped tensor
- `dims` - Array of new dimension sizes

**Returns:** Pointer to result node (graph owns, freed by `tofu_graph_free`)

**Preconditions:**
- Product of dims must equal `x->value` total elements

**Behavior:**
- View operation (no data copy)
- Implements backward pass for gradient computation

**Example:**
```c
// Flatten for fully connected layer
// Input: [batch, channels, height, width]
// Output: [batch, channels * height * width]
int flat_dim = channels * height * width;
tofu_graph_node *flat = tofu_graph_reshape(g, x, 2, (int[]){batch_size, flat_dim});
```

---

### `tofu_graph_transpose`

Add transpose node to graph.

```c
tofu_graph_node* tofu_graph_transpose(tofu_graph* g, tofu_graph_node* x, const int* axes);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `x` - Input node (cannot be NULL)
- `axes` - Permutation array (can be NULL for reverse order)

**Returns:** Pointer to result node (graph owns, freed by `tofu_graph_free`)

**Preconditions:**
- If axes is non-NULL, it must be valid permutation of `[0, ..., ndim-1]`

**Behavior:**
- If axes is NULL, reverses dimension order
- Implements backward pass for gradient computation

**Example:**
```c
// Transpose matrix for weight matrix
tofu_graph_node *W_T = tofu_graph_transpose(g, W, NULL);
```

---

## Loss Functions

Loss functions compute scalar values representing model error. They're typically the final nodes in a computation graph before calling `tofu_graph_backward()`.

### `tofu_graph_mse_loss`

Add mean squared error loss node to graph.

```c
tofu_graph_node* tofu_graph_mse_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `pred` - Prediction node (cannot be NULL)
- `target` - Target/ground truth node (cannot be NULL)

**Returns:** Pointer to scalar loss node (graph owns, freed by `tofu_graph_free`)

**Preconditions:**
- pred and target must have same shape

**Behavior:**
- Computes: `mean((pred - target)^2)`
- Returns scalar (average over all elements)
- Use for regression tasks
- Implements backward pass for gradient computation

**Example:**
```c
// Regression task
tofu_graph_node *pred = tofu_graph_matmul(g, x, W);
tofu_graph_node *target = tofu_graph_input(g, target_tensor);
tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

// Compute gradients
tofu_graph_backward(g, loss);
```

**See also:** `tofu_graph_ce_loss` for classification

---

### `tofu_graph_ce_loss`

Add cross-entropy loss node to graph.

```c
tofu_graph_node* tofu_graph_ce_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target);
```

**Parameters:**
- `g` - Graph to add node to (cannot be NULL)
- `pred` - Prediction node (softmax probabilities) (cannot be NULL)
- `target` - Target/ground truth node (class indices or one-hot) (cannot be NULL)

**Returns:** Pointer to scalar loss node (graph owns, freed by `tofu_graph_free`)

**Behavior:**
- Computes: `-sum(target * log(pred))`
- Returns scalar (average over batch)
- Use for classification tasks
- Numerically stable implementation
- Implements backward pass for gradient computation

**Example:**
```c
// Classification task
tofu_graph_node *logits = tofu_graph_matmul(g, x, W);
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);
tofu_graph_node *target = tofu_graph_input(g, target_tensor);
tofu_graph_node *loss = tofu_graph_ce_loss(g, probs, target);

// Compute gradients
tofu_graph_backward(g, loss);
```

**See also:** `tofu_graph_mse_loss` for regression

---

## Backward Pass

### `tofu_graph_backward`

Perform backward pass (backpropagation) from loss node.

```c
void tofu_graph_backward(tofu_graph* g, tofu_graph_node* loss);
```

**Parameters:**
- `g` - Graph containing loss node (cannot be NULL)
- `loss` - Loss node to backpropagate from (cannot be NULL)

**Preconditions:**
- loss must be scalar (single element tensor)

**Behavior:**
- Computes gradients for all nodes requiring gradient
- Populates `node->grad` for all PARAM nodes
- Uses reverse-mode automatic differentiation
- Call after forward pass, before optimizer step
- Gradients accumulate - call `tofu_graph_zero_grad` first if needed

**Example:**
```c
// Training iteration
tofu_graph *g = tofu_graph_create();

// Forward pass
tofu_graph_node *x = tofu_graph_input(g, input_data);
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *pred = tofu_graph_matmul(g, x, W);
tofu_graph_node *target = tofu_graph_input(g, target_data);
tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

// Backward pass
tofu_graph_backward(g, loss);

// Now W->grad contains ∂loss/∂W
tofu_tensor *W_grad = tofu_graph_get_grad(W);
```

**Notes:**
- Automatically builds topological sort for efficient gradient computation
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_graph_zero_grad` to clear gradients, `tofu_graph_get_grad` to access gradients

---

## Utilities

### `tofu_graph_get_value`

Get forward pass result from graph node.

```c
tofu_tensor* tofu_graph_get_value(tofu_graph_node* node);
```

**Parameters:**
- `node` - Graph node (cannot be NULL)

**Returns:** Pointer to result tensor (node owns, do NOT free)

**Behavior:**
- Returns tensor computed during forward pass
- Do NOT free returned tensor (node owns it)

**Example:**
```c
tofu_graph_node *pred = tofu_graph_matmul(g, x, W);
tofu_tensor *pred_value = tofu_graph_get_value(pred);

// Print predictions
tofu_tensor_print(pred_value, "%.6f");
```

**Warning:** Do not free the returned tensor!

---

### `tofu_graph_get_grad`

Get gradient from graph node.

```c
tofu_tensor* tofu_graph_get_grad(tofu_graph_node* node);
```

**Parameters:**
- `node` - Graph node (cannot be NULL)

**Returns:** Pointer to gradient tensor (node owns, do NOT free), or NULL if no gradient

**Behavior:**
- Returns gradient computed during backward pass
- Returns NULL if backward hasn't been called yet
- Do NOT free returned tensor (node owns it)

**Example:**
```c
tofu_graph_node *W = tofu_graph_param(g, weights);
// ... build graph and backward pass ...

tofu_tensor *W_grad = tofu_graph_get_grad(W);
if (W_grad) {
    // Use gradient for parameter update
    // (or use optimizer which handles this automatically)
}
```

**Warning:** Do not free the returned tensor!

---

### `tofu_graph_zero_grad`

Zero out all gradients in graph.

```c
void tofu_graph_zero_grad(tofu_graph* g);
```

**Parameters:**
- `g` - Graph to zero gradients for (cannot be NULL)

**Behavior:**
- Sets all `node->grad` tensors to zero
- Call before each training iteration to prevent gradient accumulation
- Does NOT free gradient tensors, just zeros values

**Example:**
```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Zero gradients before forward pass
    tofu_graph_zero_grad(g);

    // Forward pass
    // ... build graph ...

    // Backward pass
    tofu_graph_backward(g, loss);

    // Update parameters
    tofu_optimizer_step(optimizer);
}
```

**Notes:**
- Essential for correct training - prevents gradient accumulation
- Typically called by optimizer's `zero_grad()` function

**See also:** `tofu_optimizer_zero_grad`

---

## Usage Patterns

### Basic Training Loop

```c
// Setup
tofu_graph *g = tofu_graph_create();
tofu_tensor *W = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_tensor *b = tofu_tensor_zeros(1, (int[]){3}, TOFU_FLOAT);

tofu_graph_node *W_node = tofu_graph_param(g, W);
tofu_graph_node *b_node = tofu_graph_param(g, b);

tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        // Zero gradients
        tofu_optimizer_zero_grad(opt);

        // Forward pass
        tofu_graph_node *x = tofu_graph_input(g, batch_data[batch]);
        tofu_graph_node *y = tofu_graph_matmul(g, x, W_node);
        tofu_graph_node *out = tofu_graph_add(g, y, b_node);
        tofu_graph_node *target = tofu_graph_input(g, batch_targets[batch]);
        tofu_graph_node *loss = tofu_graph_mse_loss(g, out, target);

        // Backward pass
        tofu_graph_backward(g, loss);

        // Update parameters
        tofu_optimizer_step(opt);

        // Clear operations for next batch (keeps parameters)
        tofu_graph_clear_ops(g);
    }
}

// Cleanup
tofu_optimizer_free(opt);
tofu_graph_free(g);
tofu_tensor_free_data_too(W);
tofu_tensor_free_data_too(b);
```

### Multi-Layer Neural Network

```c
// Define network architecture
typedef struct {
    tofu_tensor *W1, *b1;
    tofu_tensor *W2, *b2;
    tofu_tensor *W3, *b3;
} Network;

// Forward pass function
tofu_graph_node* forward_pass(tofu_graph *g, tofu_graph_node *x, Network *net) {
    // Layer 1
    tofu_graph_node *W1 = tofu_graph_param(g, net->W1);
    tofu_graph_node *b1 = tofu_graph_param(g, net->b1);
    tofu_graph_node *h1 = tofu_graph_matmul(g, x, W1);
    h1 = tofu_graph_add(g, h1, b1);
    h1 = tofu_graph_relu(g, h1);

    // Layer 2
    tofu_graph_node *W2 = tofu_graph_param(g, net->W2);
    tofu_graph_node *b2 = tofu_graph_param(g, net->b2);
    tofu_graph_node *h2 = tofu_graph_matmul(g, h1, W2);
    h2 = tofu_graph_add(g, h2, b2);
    h2 = tofu_graph_relu(g, h2);

    // Output layer
    tofu_graph_node *W3 = tofu_graph_param(g, net->W3);
    tofu_graph_node *b3 = tofu_graph_param(g, net->b3);
    tofu_graph_node *out = tofu_graph_matmul(g, h2, W3);
    out = tofu_graph_add(g, out, b3);

    return out;
}

// Usage
tofu_graph *g = tofu_graph_create();
tofu_graph_node *x = tofu_graph_input(g, input_data);
tofu_graph_node *pred = forward_pass(g, x, &network);
tofu_graph_node *target = tofu_graph_input(g, target_data);
tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

tofu_graph_backward(g, loss);
```

### Classification with Softmax and Cross-Entropy

```c
// Classification network
tofu_graph_node *x = tofu_graph_input(g, input_data);
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *b = tofu_graph_param(g, bias);

// Logits
tofu_graph_node *logits = tofu_graph_matmul(g, x, W);
logits = tofu_graph_add(g, logits, b);

// Softmax (probabilities)
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);

// Cross-entropy loss
tofu_graph_node *target = tofu_graph_input(g, target_data);
tofu_graph_node *loss = tofu_graph_ce_loss(g, probs, target);

// Backward and optimize
tofu_graph_backward(g, loss);
tofu_optimizer_step(optimizer);
```

### Memory Management Best Practices

```c
// 1. Create tensors for parameters (library manages data)
tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){784, 10}, TOFU_FLOAT);
tofu_tensor *bias = tofu_tensor_zeros(1, (int[]){10}, TOFU_FLOAT);

// 2. Create graph and add parameters
tofu_graph *g = tofu_graph_create();
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *b = tofu_graph_param(g, bias);

// 3. Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Create input tensors (user manages data)
    float *batch_data = load_batch(epoch);
    tofu_tensor *x_tensor = tofu_tensor_create(batch_data, 2, (int[]){32, 784}, TOFU_FLOAT);

    // Build graph
    tofu_graph_node *x = tofu_graph_input(g, x_tensor);
    // ... forward pass ...

    // Training step
    tofu_graph_backward(g, loss);
    tofu_optimizer_step(opt);

    // Clean up batch resources
    tofu_tensor_free(x_tensor);  // Free tensor structure
    free(batch_data);            // Free data buffer

    // Clear operations (keeps parameters)
    tofu_graph_clear_ops(g);
}

// 4. Cleanup (IMPORTANT ORDER!)
tofu_optimizer_free(opt);           // Free optimizer first
tofu_graph_free(g);                 // Then free graph
tofu_tensor_free_data_too(weights);  // Finally free parameter tensors
tofu_tensor_free_data_too(bias);
```

### Efficient Batch Processing

```c
tofu_graph *g = tofu_graph_create();

// Add parameters once (persists across batches)
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *b = tofu_graph_param(g, bias);

for (int batch = 0; batch < num_batches; batch++) {
    tofu_optimizer_zero_grad(opt);

    // Add input for this batch
    tofu_graph_node *x = tofu_graph_input(g, batch_data[batch]);

    // Build forward graph
    tofu_graph_node *out = tofu_graph_add(g, tofu_graph_matmul(g, x, W), b);
    tofu_graph_node *loss = tofu_graph_mse_loss(g, out, batch_targets[batch]);

    // Backward and update
    tofu_graph_backward(g, loss);
    tofu_optimizer_step(opt);

    // Clear operations for next batch (W and b are preserved)
    tofu_graph_clear_ops(g);
}
```

---

## Notes

### Gradient Accumulation

Gradients accumulate by default. Always call `tofu_graph_zero_grad()` or `tofu_optimizer_zero_grad()` before each training iteration:

```c
// CORRECT: Zero gradients before each iteration
for (int i = 0; i < num_iterations; i++) {
    tofu_optimizer_zero_grad(opt);  // Clear previous gradients
    // ... forward and backward ...
}

// INCORRECT: Gradients accumulate indefinitely
for (int i = 0; i < num_iterations; i++) {
    // ... forward and backward ...
    // Gradients from all iterations accumulate!
}
```

### Dynamic Graphs

Tofu uses dynamic computation graphs (define-by-run). The graph structure can change between iterations:

```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    if (epoch < 10) {
        // Simple network
        out = tofu_graph_matmul(g, x, W1);
    } else {
        // More complex network
        h = tofu_graph_relu(g, tofu_graph_matmul(g, x, W1));
        out = tofu_graph_matmul(g, h, W2);
    }

    tofu_graph_backward(g, loss);
    tofu_graph_clear_ops(g);  // Clear for next iteration
}
```

### Error Checking

Most functions use `assert()` for precondition checking. In release builds with assertions disabled, violating preconditions leads to undefined behavior. Always ensure:
- Pointers are non-NULL
- Shapes are compatible
- Tensors are broadcastable
- Loss is scalar before calling backward
