# Optimizer API Reference

The Optimizer API provides algorithms for updating trainable parameters based on computed gradients. Optimizers automatically collect parameters from the computation graph and apply update rules during training.

## Table of Contents

- [Data Structures](#data-structures)
- [Creating Optimizers](#creating-optimizers)
- [Training Operations](#training-operations)
- [Parameter Management](#parameter-management)
- [Usage Patterns](#usage-patterns)
- [Hyperparameter Guidance](#hyperparameter-guidance)

---

## Data Structures

### `tofu_optimizer`

The optimizer structure that manages parameters and their update strategy.

```c
struct tofu_optimizer {
    tofu_optim_type type;            // Optimizer type
    tofu_graph* graph;               // Associated computation graph

    tofu_graph_node** params;        // Array of parameter nodes
    int num_params;                  // Number of parameters
    int capacity_params;             // Allocated capacity

    double learning_rate;            // Learning rate

    void* state;                     // Optimizer state (momentum buffers, etc.)

    tofu_optim_step_fn step_fn;      // Parameter update function
};
```

### Optimizer Types (`tofu_optim_type`)

Available optimization algorithms:

- `TOFU_OPTIM_SGD` - Vanilla Stochastic Gradient Descent
- `TOFU_OPTIM_SGD_MOMENTUM` - SGD with momentum
- `TOFU_OPTIM_ADAM` - Adam optimizer (future)

---

## Creating Optimizers

### `tofu_optimizer_sgd_create`

Create SGD (Stochastic Gradient Descent) optimizer.

```c
tofu_optimizer* tofu_optimizer_sgd_create(tofu_graph* g, double learning_rate);
```

**Parameters:**
- `g` - Computation graph containing parameters (cannot be NULL)
- `learning_rate` - Learning rate (step size) (must be > 0)

**Returns:** Pointer to newly allocated optimizer (caller owns, must call `tofu_optimizer_free`)

**Preconditions:**
- g must not be NULL
- learning_rate > 0

**Behavior:**
- Implements vanilla SGD: `param = param - learning_rate * grad`
- Automatically collects all PARAM nodes from graph
- Caller must call `tofu_optimizer_free` to free optimizer

**Algorithm:**
```
for each parameter θ:
    θ ← θ - η * ∇θL
where:
    η = learning_rate
    ∇θL = gradient of loss w.r.t. parameter
```

**Example:**
```c
tofu_graph *g = tofu_graph_create();

// Add parameters to graph
tofu_tensor *W = tofu_tensor_zeros(2, (int[]){784, 10}, TOFU_FLOAT);
tofu_graph_node *W_node = tofu_graph_param(g, W);

// Create optimizer
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    tofu_optimizer_zero_grad(opt);
    // ... forward and backward pass ...
    tofu_optimizer_step(opt);
}

// Cleanup
tofu_optimizer_free(opt);
tofu_graph_free(g);
tofu_tensor_free_data_too(W);
```

**Notes:**
- Simple and robust, good baseline optimizer
- No momentum or adaptive learning rates
- May converge slowly on complex problems
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_optimizer_sgd_momentum_create` for SGD with momentum

---

### `tofu_optimizer_sgd_momentum_create`

Create SGD optimizer with momentum.

```c
tofu_optimizer* tofu_optimizer_sgd_momentum_create(tofu_graph* g, double learning_rate, double momentum);
```

**Parameters:**
- `g` - Computation graph containing parameters (cannot be NULL)
- `learning_rate` - Learning rate (step size) (must be > 0)
- `momentum` - Momentum coefficient (typically 0.9) (must be >= 0 and < 1)

**Returns:** Pointer to newly allocated optimizer (caller owns, must call `tofu_optimizer_free`)

**Preconditions:**
- g must not be NULL
- learning_rate > 0
- 0 <= momentum < 1

**Behavior:**
- Implements SGD with momentum:
  - `velocity = momentum * velocity + grad`
  - `param = param - learning_rate * velocity`
- Momentum helps accelerate training and reduces oscillations
- Automatically collects all PARAM nodes from graph
- Caller must call `tofu_optimizer_free` to free optimizer

**Algorithm:**
```
for each parameter θ:
    v ← μ * v + ∇θL
    θ ← θ - η * v
where:
    η = learning_rate
    μ = momentum
    v = velocity (accumulated gradients)
    ∇θL = gradient of loss w.r.t. parameter
```

**Example:**
```c
tofu_graph *g = tofu_graph_create();

// Add parameters
tofu_tensor *W = tofu_tensor_zeros(2, (int[]){784, 10}, TOFU_FLOAT);
tofu_graph_node *W_node = tofu_graph_param(g, W);

// Create optimizer with momentum
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    tofu_optimizer_zero_grad(opt);
    // ... forward and backward pass ...
    tofu_optimizer_step(opt);
}

// Cleanup
tofu_optimizer_free(opt);
```

**Notes:**
- Momentum helps escape local minima and speeds up convergence
- Typical momentum values: 0.9 (standard), 0.99 (high momentum)
- More effective than vanilla SGD for deep networks
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_optimizer_sgd_create` for vanilla SGD

---

## Cleanup

### `tofu_optimizer_free`

Free optimizer and its state.

```c
void tofu_optimizer_free(tofu_optimizer* opt);
```

**Parameters:**
- `opt` - Optimizer to free (can be NULL, no-op if NULL)

**Behavior:**
- Frees optimizer structure and internal state (momentum buffers, etc.)
- Does NOT free the graph or parameters (graph owns them)
- Safe to call multiple times (idempotent)

**Cleanup Order:**
```c
// CORRECT order:
tofu_optimizer_free(opt);           // 1. Free optimizer
tofu_graph_free(g);                 // 2. Free graph
tofu_tensor_free_data_too(weights);  // 3. Free tensors

// INCORRECT order (may crash):
tofu_graph_free(g);                 // DON'T free graph before optimizer!
tofu_optimizer_free(opt);           // Optimizer may access freed memory
```

---

## Training Operations

### `tofu_optimizer_step`

Perform one optimization step (update parameters).

```c
void tofu_optimizer_step(tofu_optimizer* opt);
```

**Parameters:**
- `opt` - Optimizer (cannot be NULL)

**Preconditions:**
- opt must not be NULL
- Gradients must be computed (call `tofu_graph_backward` first)

**Behavior:**
- Updates all parameters using computed gradients
- Algorithm depends on optimizer type (SGD, SGD+momentum, etc.)
- Call after backward pass: forward → backward → step
- Does NOT zero gradients - call `tofu_optimizer_zero_grad` if needed

**Training Sequence:**
```c
for (int iteration = 0; iteration < num_iterations; iteration++) {
    // 1. Zero gradients
    tofu_optimizer_zero_grad(opt);

    // 2. Forward pass
    tofu_graph_node *x = tofu_graph_input(g, input_data);
    tofu_graph_node *pred = forward_pass(g, x);
    tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

    // 3. Backward pass
    tofu_graph_backward(g, loss);

    // 4. Update parameters
    tofu_optimizer_step(opt);

    // 5. Clear operations for next iteration
    tofu_graph_clear_ops(g);
}
```

**Notes:**
- Must call `tofu_graph_backward()` before this function
- Modifies parameter tensors in-place
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_graph_backward`, `tofu_optimizer_zero_grad`

---

### `tofu_optimizer_zero_grad`

Zero out all parameter gradients.

```c
void tofu_optimizer_zero_grad(tofu_optimizer* opt);
```

**Parameters:**
- `opt` - Optimizer (cannot be NULL)

**Preconditions:**
- opt must not be NULL

**Behavior:**
- Sets gradients to zero for all tracked parameters
- Call before each training iteration to prevent gradient accumulation
- Equivalent to `tofu_graph_zero_grad` but works via optimizer

**Example:**
```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Zero gradients before forward pass
    tofu_optimizer_zero_grad(opt);

    // Forward pass
    tofu_graph_node *pred = forward_pass(g, input);
    tofu_graph_node *loss = compute_loss(g, pred, target);

    // Backward pass
    tofu_graph_backward(g, loss);

    // Update parameters
    tofu_optimizer_step(opt);
}
```

**Notes:**
- Essential for correct training - prevents gradient accumulation
- Must call before each training iteration
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_graph_zero_grad`

---

## Parameter Management

Most users won't need these functions - parameters are automatically collected during optimizer creation. These are useful for advanced use cases like dynamic network architectures.

### `tofu_optimizer_add_param`

Manually add parameter node to optimizer.

```c
int tofu_optimizer_add_param(tofu_optimizer* opt, tofu_graph_node* param);
```

**Parameters:**
- `opt` - Optimizer (cannot be NULL)
- `param` - Parameter node to track (cannot be NULL)

**Returns:** 0 on success, non-zero on error

**Preconditions:**
- opt and param must not be NULL
- param must be a PARAM node (requires gradient)

**Behavior:**
- Usually not needed - optimizer auto-collects params at creation
- Use if you need to add parameters dynamically

**Example:**
```c
// Create optimizer
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// Add parameters dynamically (rare use case)
tofu_tensor *new_weight = tofu_tensor_zeros(2, (int[]){10, 5}, TOFU_FLOAT);
tofu_graph_node *W_new = tofu_graph_param(g, new_weight);
tofu_optimizer_add_param(opt, W_new);
```

**Notes:**
- Rarely needed - use only for dynamic architectures
- Violating preconditions triggers `assert()` and crashes

**See also:** `tofu_optimizer_collect_params` to scan graph for all params

---

### `tofu_optimizer_collect_params`

Collect all parameter nodes from graph.

```c
void tofu_optimizer_collect_params(tofu_optimizer* opt);
```

**Parameters:**
- `opt` - Optimizer (cannot be NULL)

**Preconditions:**
- opt must not be NULL

**Behavior:**
- Scans graph and adds all PARAM nodes to optimizer
- Called automatically during optimizer creation
- Use if graph structure changes and you need to rescan
- Clears existing parameter list before collecting

**Example:**
```c
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// Add more parameters to graph later
tofu_tensor *W2 = tofu_tensor_zeros(2, (int[]){10, 5}, TOFU_FLOAT);
tofu_graph_node *W2_node = tofu_graph_param(g, W2);

// Rescan graph to include new parameters
tofu_optimizer_collect_params(opt);
```

**Notes:**
- Rarely needed - parameters auto-collected at creation
- Use only if network structure changes dynamically
- Violating preconditions triggers `assert()` and crashes

---

## Usage Patterns

### Basic Training Loop

```c
// Setup
tofu_graph *g = tofu_graph_create();

// Create parameters
tofu_tensor *W = tofu_tensor_zeros(2, (int[]){784, 10}, TOFU_FLOAT);
tofu_tensor *b = tofu_tensor_zeros(1, (int[]){10}, TOFU_FLOAT);

// Add to graph
tofu_graph_node *W_node = tofu_graph_param(g, W);
tofu_graph_node *b_node = tofu_graph_param(g, b);

// Create optimizer
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        // 1. Zero gradients
        tofu_optimizer_zero_grad(opt);

        // 2. Forward pass
        tofu_graph_node *x = tofu_graph_input(g, batch_data[batch]);
        tofu_graph_node *h = tofu_graph_matmul(g, x, W_node);
        tofu_graph_node *pred = tofu_graph_add(g, h, b_node);

        // 3. Compute loss
        tofu_graph_node *target = tofu_graph_input(g, batch_targets[batch]);
        tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

        // 4. Backward pass
        tofu_graph_backward(g, loss);

        // 5. Update parameters
        tofu_optimizer_step(opt);

        // 6. Clear operations for next batch
        tofu_graph_clear_ops(g);
    }
}

// Cleanup
tofu_optimizer_free(opt);
tofu_graph_free(g);
tofu_tensor_free_data_too(W);
tofu_tensor_free_data_too(b);
```

### Training with Momentum

```c
// Setup with momentum optimizer
tofu_graph *g = tofu_graph_create();

// Network parameters
tofu_tensor *W1 = tofu_tensor_zeros(2, (int[]){784, 128}, TOFU_FLOAT);
tofu_tensor *b1 = tofu_tensor_zeros(1, (int[]){128}, TOFU_FLOAT);
tofu_tensor *W2 = tofu_tensor_zeros(2, (int[]){128, 10}, TOFU_FLOAT);
tofu_tensor *b2 = tofu_tensor_zeros(1, (int[]){10}, TOFU_FLOAT);

// Add to graph
tofu_graph_node *W1_node = tofu_graph_param(g, W1);
tofu_graph_node *b1_node = tofu_graph_param(g, b1);
tofu_graph_node *W2_node = tofu_graph_param(g, W2);
tofu_graph_node *b2_node = tofu_graph_param(g, b2);

// Create optimizer with momentum
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        tofu_optimizer_zero_grad(opt);

        // Forward pass
        tofu_graph_node *x = tofu_graph_input(g, batch_data[batch]);

        // Layer 1
        tofu_graph_node *h1 = tofu_graph_matmul(g, x, W1_node);
        h1 = tofu_graph_add(g, h1, b1_node);
        h1 = tofu_graph_relu(g, h1);

        // Layer 2
        tofu_graph_node *h2 = tofu_graph_matmul(g, h1, W2_node);
        h2 = tofu_graph_add(g, h2, b2_node);

        // Loss
        tofu_graph_node *target = tofu_graph_input(g, batch_targets[batch]);
        tofu_graph_node *loss = tofu_graph_mse_loss(g, h2, target);

        // Backward and update
        tofu_graph_backward(g, loss);
        tofu_optimizer_step(opt);

        tofu_graph_clear_ops(g);
    }
}

// Cleanup
tofu_optimizer_free(opt);
tofu_graph_free(g);
tofu_tensor_free_data_too(W1);
tofu_tensor_free_data_too(b1);
tofu_tensor_free_data_too(W2);
tofu_tensor_free_data_too(b2);
```

### Learning Rate Scheduling

Manual learning rate adjustment during training:

```c
// Create optimizer
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.1);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Reduce learning rate every 10 epochs
    if (epoch % 10 == 0 && epoch > 0) {
        opt->learning_rate *= 0.5;
        printf("Epoch %d: Reduced learning rate to %.6f\n", epoch, opt->learning_rate);
    }

    // Training loop for this epoch
    for (int batch = 0; batch < num_batches; batch++) {
        tofu_optimizer_zero_grad(opt);
        // ... forward, backward, step ...
    }
}
```

### Monitoring Gradients

Useful for debugging and understanding training dynamics:

```c
// After backward pass, before optimizer step
tofu_tensor *W_grad = tofu_graph_get_grad(W_node);

// Compute gradient statistics
double grad_sum = 0.0;
double grad_max = -INFINITY;
for (int i = 0; i < W_grad->len; i++) {
    float val;
    TOFU_TENSOR_DATA_TO(W_grad, i, val, TOFU_FLOAT);
    grad_sum += fabs(val);
    if (fabs(val) > grad_max) grad_max = fabs(val);
}

printf("Gradient mean: %.6f, max: %.6f\n",
       grad_sum / W_grad->len, grad_max);

// Now update parameters
tofu_optimizer_step(opt);
```

### Gradient Clipping (Manual)

Prevent exploding gradients:

```c
void clip_gradients(tofu_optimizer *opt, double max_norm) {
    for (int i = 0; i < opt->num_params; i++) {
        tofu_tensor *grad = tofu_graph_get_grad(opt->params[i]);
        if (!grad) continue;

        // Compute gradient norm
        double norm = 0.0;
        for (int j = 0; j < grad->len; j++) {
            float val;
            TOFU_TENSOR_DATA_TO(grad, j, val, TOFU_FLOAT);
            norm += val * val;
        }
        norm = sqrt(norm);

        // Clip if necessary
        if (norm > max_norm) {
            double scale = max_norm / norm;
            for (int j = 0; j < grad->len; j++) {
                float val;
                TOFU_TENSOR_DATA_TO(grad, j, val, TOFU_FLOAT);
                val *= scale;
                TOFU_TENSOR_DATA_FROM(grad, j, val, TOFU_FLOAT);
            }
        }
    }
}

// Usage in training loop
tofu_graph_backward(g, loss);
clip_gradients(opt, 1.0);  // Clip to max norm of 1.0
tofu_optimizer_step(opt);
```

---

## Hyperparameter Guidance

### Learning Rate

The learning rate is the most important hyperparameter. It controls the step size of parameter updates.

**Guidelines:**

| Problem Type | Recommended Range | Notes |
|-------------|------------------|-------|
| Small networks | 0.01 - 0.1 | Can use larger learning rates |
| Deep networks | 0.001 - 0.01 | Need smaller learning rates |
| Fine-tuning | 0.0001 - 0.001 | Very small to preserve learned features |

**Common values:**
- **0.1** - Starting point for small networks
- **0.01** - Default safe choice for most problems
- **0.001** - Deep networks, complex problems
- **0.0001** - Fine-tuning pre-trained models

**Signs of incorrect learning rate:**
- **Too high**: Loss diverges (increases), NaN values, training unstable
- **Too low**: Very slow convergence, loss decreases too slowly

**Example - Finding good learning rate:**
```c
// Try multiple learning rates
double learning_rates[] = {0.001, 0.01, 0.1};

for (int lr_idx = 0; lr_idx < 3; lr_idx++) {
    printf("\n=== Testing LR: %.4f ===\n", learning_rates[lr_idx]);

    // Reset parameters
    reinitialize_parameters(W, b);

    // Create optimizer with this learning rate
    tofu_optimizer *opt = tofu_optimizer_sgd_create(g, learning_rates[lr_idx]);

    // Train for a few epochs
    for (int epoch = 0; epoch < 10; epoch++) {
        // ... training loop ...
        printf("Epoch %d, Loss: %.6f\n", epoch, loss_value);
    }

    tofu_optimizer_free(opt);
}
```

### Momentum

Momentum helps accelerate convergence and dampen oscillations.

**Guidelines:**

| Scenario | Recommended Value | Effect |
|----------|------------------|--------|
| Default | 0.9 | Good balance for most problems |
| High momentum | 0.95 - 0.99 | Faster convergence, may overshoot |
| Low momentum | 0.5 - 0.8 | More stable, slower convergence |
| No momentum | 0.0 | Vanilla SGD, most stable but slowest |

**Common values:**
- **0.9** - Standard choice for most problems
- **0.95** - Deep networks, when convergence is slow
- **0.99** - Very deep networks (ResNet, Transformers)
- **0.5** - Noisy gradients, unstable training

**Example:**
```c
// Standard momentum for deep network
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);

// Higher momentum for very deep network
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.99);

// Low momentum for noisy gradients
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.5);
```

### Batch Size Considerations

Batch size affects effective learning rate:

```c
// Larger batches → more stable gradients → can use higher learning rate
int batch_size = 128;
double lr = 0.01;

tofu_optimizer *opt = tofu_optimizer_sgd_create(g, lr);

// If you increase batch size, consider increasing learning rate proportionally
// batch_size = 256 → lr = 0.02
// batch_size = 512 → lr = 0.04
```

### Learning Rate Schedules

Common strategies for adjusting learning rate during training:

**Step Decay:**
```c
// Reduce learning rate every N epochs
if (epoch % 30 == 0 && epoch > 0) {
    opt->learning_rate *= 0.1;  // Reduce by 10x
}
```

**Exponential Decay:**
```c
// Decay gradually every epoch
double initial_lr = 0.1;
double decay_rate = 0.96;
opt->learning_rate = initial_lr * pow(decay_rate, epoch);
```

**Cosine Annealing:**
```c
// Smooth decay following cosine curve
double initial_lr = 0.1;
double min_lr = 0.001;
opt->learning_rate = min_lr + (initial_lr - min_lr) *
                     (1 + cos(M_PI * epoch / num_epochs)) / 2;
```

### Training Tips

**1. Start with a reasonable learning rate:**
```c
// Good defaults:
tofu_optimizer *opt_sgd = tofu_optimizer_sgd_create(g, 0.01);
tofu_optimizer *opt_momentum = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);
```

**2. Monitor loss and adjust:**
```c
double prev_loss = INFINITY;

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // ... training ...

    // Check if loss is improving
    if (loss_value > prev_loss * 1.1) {
        printf("Loss increased! Consider reducing learning rate.\n");
    }

    prev_loss = loss_value;
}
```

**3. Use learning rate warmup for large learning rates:**
```c
double target_lr = 0.1;
int warmup_epochs = 5;

for (int epoch = 0; epoch < num_epochs; epoch++) {
    if (epoch < warmup_epochs) {
        // Gradually increase learning rate
        opt->learning_rate = target_lr * (epoch + 1) / warmup_epochs;
    } else {
        opt->learning_rate = target_lr;
    }

    // ... training ...
}
```

**4. Weight decay (L2 regularization) - manual implementation:**
```c
double weight_decay = 0.0001;

void apply_weight_decay(tofu_optimizer *opt, double weight_decay) {
    for (int i = 0; i < opt->num_params; i++) {
        tofu_tensor *param = tofu_graph_get_value(opt->params[i]);

        for (int j = 0; j < param->len; j++) {
            float val;
            TOFU_TENSOR_DATA_TO(param, j, val, TOFU_FLOAT);
            val *= (1.0 - weight_decay * opt->learning_rate);
            TOFU_TENSOR_DATA_FROM(param, j, val, TOFU_FLOAT);
        }
    }
}

// Use before optimizer step
tofu_graph_backward(g, loss);
apply_weight_decay(opt, 0.0001);
tofu_optimizer_step(opt);
```

---

## Common Pitfalls

### Forgetting to Zero Gradients

**Problem:**
```c
// WRONG: Gradients accumulate indefinitely
for (int i = 0; i < num_iterations; i++) {
    // forward, backward, step...
    // Gradients keep accumulating!
}
```

**Solution:**
```c
// CORRECT: Zero gradients each iteration
for (int i = 0; i < num_iterations; i++) {
    tofu_optimizer_zero_grad(opt);  // Clear gradients
    // forward, backward, step...
}
```

### Incorrect Cleanup Order

**Problem:**
```c
// WRONG: Freeing graph before optimizer
tofu_graph_free(g);        // Graph freed
tofu_optimizer_free(opt);  // Optimizer tries to access freed graph!
```

**Solution:**
```c
// CORRECT: Free optimizer before graph
tofu_optimizer_free(opt);  // Free optimizer first
tofu_graph_free(g);        // Then free graph
```

### Learning Rate Too High

**Symptoms:**
- Loss becomes NaN
- Loss diverges (increases)
- Training unstable

**Solution:**
```c
// Reduce learning rate by 10x
double new_lr = opt->learning_rate * 0.1;
tofu_optimizer_free(opt);
opt = tofu_optimizer_sgd_create(g, new_lr);
```

### Learning Rate Too Low

**Symptoms:**
- Loss decreases very slowly
- Training takes many epochs
- No progress after many iterations

**Solution:**
```c
// Increase learning rate by 10x
double new_lr = opt->learning_rate * 10.0;
tofu_optimizer_free(opt);
opt = tofu_optimizer_sgd_create(g, new_lr);
```

---

## Notes

### Optimizer State Persistence

Optimizer state (like momentum buffers) persists across training iterations:

```c
// Momentum accumulates across iterations
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Momentum from previous epochs affects current updates
    // ... training ...
}
```

### Parameter Collection

Optimizers automatically collect parameters when created:

```c
// All PARAM nodes are collected automatically
tofu_graph_node *W1 = tofu_graph_param(g, weights1);
tofu_graph_node *W2 = tofu_graph_param(g, weights2);

tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);
// opt now tracks both W1 and W2
```

### Memory Management

Optimizer owns its internal state but not the graph or parameters:

```c
// Optimizer allocates momentum buffers (if using momentum)
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);

// When freed, optimizer releases momentum buffers
tofu_optimizer_free(opt);

// Graph and parameters remain valid
// (must be freed separately)
```
