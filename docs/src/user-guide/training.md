# Training Neural Networks

This guide covers how to train neural networks using TOFU's automatic differentiation and optimization capabilities. We'll walk through the complete training process with practical examples.

## Introduction

Training a neural network in TOFU follows a standard pattern familiar to users of modern frameworks like PyTorch or TensorFlow. The key difference is that TOFU is designed for resource-constrained environments like microcontrollers, so we emphasize memory efficiency and explicit resource management.

### What You'll Learn

In this guide you'll learn:
- How to structure a complete training loop
- Data preparation and batching strategies
- Forward and backward pass mechanics
- Loss computation and monitoring
- Parameter optimization
- Training strategies for embedded systems
- Debugging and evaluation techniques

### Prerequisites

Before starting, ensure you're familiar with:
- Basic tensor operations (see [Tensors guide](tensors.md))
- Computation graphs (see [Graphs guide](graphs.md))
- Loss functions (see [Loss Functions guide](loss-functions.md))
- Optimizers (see [Optimizers guide](optimizers.md))

### The Training Paradigm

Neural network training is an iterative process of:
1. Making predictions (forward pass)
2. Measuring error (loss computation)
3. Computing gradients (backward pass)
4. Updating weights (optimization step)

TOFU provides all the primitives needed for this cycle through its computation graph API and automatic differentiation engine.

### Memory Considerations

Training on microcontrollers requires careful memory management. TOFU helps by:
- Allowing graph reuse across iterations via `tofu_graph_clear_ops()`
- Minimizing allocations during training
- Providing explicit control over tensor lifetimes
- Supporting in-place operations where possible

## The Training Loop

Every training loop in TOFU follows a consistent five-step pattern. Understanding this pattern is essential for successful training.

### The Five-Step Pattern

```c
for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        /* Step 1: Zero gradients */
        tofu_graph_zero_grad(g);

        /* Step 2: Forward pass */
        tofu_graph_node* prediction = forward_pass(g, input, params);

        /* Step 3: Compute loss */
        tofu_graph_node* loss = tofu_graph_mse_loss(g, prediction, target);

        /* Step 4: Backward pass */
        tofu_graph_backward(g, loss);

        /* Step 5: Update parameters */
        tofu_optimizer_step(optimizer);

        /* Cleanup: Clear operations but keep parameters */
        tofu_graph_clear_ops(g);
    }
}
```

Let's examine each step in detail.

### Step 1: Zero Gradients

Before computing new gradients, clear any gradients from the previous iteration:

```c
tofu_graph_zero_grad(g);
```

**Why?** Gradients accumulate by default. If you don't zero them, new gradients add to old ones, producing incorrect updates.

**When to skip:** Only when you explicitly want gradient accumulation (advanced technique).

### Step 2: Forward Pass

Build the computation graph and compute predictions:

```c
/* Create input node */
tofu_graph_node* x = tofu_graph_input(g, input_tensor);

/* Build network */
tofu_graph_node* h1 = tofu_graph_matmul(g, x, w1);
tofu_graph_node* h1_bias = tofu_graph_add(g, h1, b1);
tofu_graph_node* h1_act = tofu_graph_relu(g, h1_bias);

/* Output layer */
tofu_graph_node* output = tofu_graph_matmul(g, h1_act, w2);
tofu_graph_node* pred = tofu_graph_add(g, output, b2);
```

**Key principle:** The forward pass constructs the computational graph that defines your model.

### Step 3: Compute Loss

Compare predictions to targets:

```c
tofu_graph_node* target = tofu_graph_input(g, target_tensor);
tofu_graph_node* loss = tofu_graph_mse_loss(g, pred, target);
```

The loss node becomes the starting point for backpropagation.

### Step 4: Backward Pass

Compute gradients via automatic differentiation:

```c
tofu_graph_backward(g, loss);
```

This populates the `grad` field of all parameter nodes with gradients.

### Step 5: Update Parameters

Apply gradients to update trainable parameters:

```c
tofu_optimizer_step(optimizer);
```

The optimizer uses computed gradients and its internal algorithm (SGD, momentum, etc.) to update parameter values.

### Graph Cleanup

After each iteration, clear operations while preserving parameters:

```c
tofu_graph_clear_ops(g);
```

This frees intermediate computation nodes but keeps parameter nodes, allowing the graph to be reused in the next iteration.

## Data Preparation

Proper data preparation is crucial for successful training. This section covers batching, normalization, and memory-efficient data handling.

### Dataset Structure

Organize your data to facilitate batch processing:

```c
typedef struct {
    float* images;      /* [num_samples, feature_dims] */
    int* labels;        /* [num_samples] */
    int num_samples;
    int feature_dims;
} dataset;
```

For the XOR problem, data preparation is simple:

```c
float xor_inputs[4][2] = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
};

float xor_targets[4][1] = {
    {0.0f},
    {1.0f},
    {1.0f},
    {0.0f}
};
```

### Batching Strategies

For larger datasets, process data in batches:

```c
const int BATCH_SIZE = 4;

for (int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
    int batch_end = (batch_start + BATCH_SIZE < num_samples)
                    ? batch_start + BATCH_SIZE
                    : num_samples;
    int actual_batch_size = batch_end - batch_start;

    /* Prepare batch data */
    float* batch_data = (float*)malloc(actual_batch_size * feature_dims * sizeof(float));
    int* batch_labels = (int*)malloc(actual_batch_size * sizeof(int));

    for (int i = 0; i < actual_batch_size; i++) {
        memcpy(batch_data + i * feature_dims,
               dataset->images + (batch_start + i) * feature_dims,
               feature_dims * sizeof(float));
        batch_labels[i] = dataset->labels[batch_start + i];
    }

    /* Create batch tensor */
    tofu_tensor* t_batch = tofu_tensor_create(batch_data, 2,
                                              (int[]){actual_batch_size, feature_dims},
                                              TOFU_FLOAT);

    /* ... training step ... */

    /* Cleanup */
    tofu_tensor_free(t_batch);
    free(batch_data);
    free(batch_labels);
}
```

**Batch size considerations:**
- Larger batches: More stable gradients, better hardware utilization
- Smaller batches: Less memory usage, more frequent updates
- For microcontrollers: Start with batch_size=1 or very small batches

### Data Normalization

Normalize inputs for better training stability:

```c
/* Compute mean and std from training data */
void compute_statistics(const float* data, int num_samples, int dims,
                       float* mean, float* std) {
    /* Zero initialize */
    for (int d = 0; d < dims; d++) {
        mean[d] = 0.0f;
        std[d] = 0.0f;
    }

    /* Compute mean */
    for (int i = 0; i < num_samples; i++) {
        for (int d = 0; d < dims; d++) {
            mean[d] += data[i * dims + d];
        }
    }
    for (int d = 0; d < dims; d++) {
        mean[d] /= num_samples;
    }

    /* Compute std */
    for (int i = 0; i < num_samples; i++) {
        for (int d = 0; d < dims; d++) {
            float diff = data[i * dims + d] - mean[d];
            std[d] += diff * diff;
        }
    }
    for (int d = 0; d < dims; d++) {
        std[d] = sqrtf(std[d] / num_samples);
    }
}

/* Normalize data */
void normalize_data(float* data, int num_samples, int dims,
                   const float* mean, const float* std) {
    for (int i = 0; i < num_samples; i++) {
        for (int d = 0; d < dims; d++) {
            data[i * dims + d] = (data[i * dims + d] - mean[d]) / (std[d] + 1e-8f);
        }
    }
}
```

**Common normalization strategies:**
- **Z-score normalization:** `(x - mean) / std` (shown above)
- **Min-max scaling:** `(x - min) / (max - min)` to [0, 1]
- **Simple scaling:** Divide by 255 for image data
- **No normalization:** For binary inputs like XOR

### One-Hot Encoding

For classification, encode labels as one-hot vectors:

```c
/* Convert integer labels to one-hot encoding */
void create_one_hot(int* labels, int batch_size, int num_classes, float* one_hot) {
    memset(one_hot, 0, batch_size * num_classes * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        one_hot[i * num_classes + labels[i]] = 1.0f;
    }
}

/* Usage in training loop */
float* target_data = (float*)calloc(batch_size * num_classes, sizeof(float));
create_one_hot(batch_labels, batch_size, num_classes, target_data);

tofu_tensor* t_target = tofu_tensor_create(target_data, 2,
                                           (int[]){batch_size, num_classes},
                                           TOFU_FLOAT);
```

## Forward Pass

The forward pass computes predictions by propagating input through the network. Understanding graph construction and reuse is key to efficient training.

### Building the Computation Graph

For a simple feedforward network:

```c
tofu_graph_node* forward_pass(tofu_graph* g,
                              tofu_tensor* input_data,
                              tofu_graph_node* w1, tofu_graph_node* b1,
                              tofu_graph_node* w2, tofu_graph_node* b2) {
    /* Input layer */
    tofu_graph_node* x = tofu_graph_input(g, input_data);

    /* Hidden layer: x @ w1 + b1 */
    tofu_graph_node* h1_matmul = tofu_graph_matmul(g, x, w1);
    tofu_graph_node* h1_bias = tofu_graph_add(g, h1_matmul, b1);
    tofu_graph_node* h1 = tofu_graph_relu(g, h1_bias);

    /* Output layer: h1 @ w2 + b2 */
    tofu_graph_node* out_matmul = tofu_graph_matmul(g, h1, w2);
    tofu_graph_node* prediction = tofu_graph_add(g, out_matmul, b2);

    return prediction;
}
```

### Reusing Graphs with clear_ops

Instead of creating a new graph each iteration, reuse it:

```c
/* Initialize graph once */
tofu_graph* g = tofu_graph_create();

/* Create parameters once (persist across iterations) */
tofu_graph_node* w1 = tofu_graph_param(g, t_w1);
tofu_graph_node* b1 = tofu_graph_param(g, t_b1);
tofu_graph_node* w2 = tofu_graph_param(g, t_w2);
tofu_graph_node* b2 = tofu_graph_param(g, t_b2);

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    tofu_graph_zero_grad(g);

    /* Build forward pass (creates new operation nodes) */
    tofu_graph_node* pred = forward_pass(g, input, w1, b1, w2, b2);
    tofu_graph_node* loss = tofu_graph_mse_loss(g, pred, target);

    tofu_graph_backward(g, loss);
    tofu_optimizer_step(optimizer);

    /* Clear operations but keep parameters */
    tofu_graph_clear_ops(g);  /* This is crucial! */
}
```

**Why clear_ops?**
- Frees intermediate operation nodes
- Preserves parameter nodes and their values
- Allows graph reuse without memory leaks
- Essential for embedded systems with limited memory

### Activation Functions

TOFU supports several activation functions:

```c
/* ReLU: max(0, x) */
tofu_graph_node* relu_out = tofu_graph_relu(g, input);

/* Softmax: exp(x) / sum(exp(x)) */
tofu_graph_node* softmax_out = tofu_graph_softmax(g, logits, 1);  /* axis=1 */
```

Choose activation based on your task:
- **ReLU:** Hidden layers, default choice
- **Softmax:** Output layer for multi-class classification
- **None:** Output layer for regression

## Computing Loss

The loss function measures prediction error and drives learning. Choosing the right loss function is critical for your task.

### Mean Squared Error (MSE)

Use MSE for regression tasks:

```c
tofu_graph_node* loss = tofu_graph_mse_loss(g, prediction, target);
```

**Formula:** `L = mean((pred - target)^2)`

**When to use:**
- Regression problems (predicting continuous values)
- Output layer without activation (raw values)
- Examples: XOR (as regression), price prediction, temperature estimation

**Example from XOR training:**

```c
/* Prediction is continuous output */
tofu_graph_node* y_pred = tofu_graph_add(g, y_matmul, b2);

/* Target is also continuous */
float* target_data = (float*)malloc(OUTPUT_SIZE * sizeof(float));
target_data[0] = xor_targets[sample][0];  /* 0.0 or 1.0 */
tofu_tensor* t_target = tofu_tensor_create(target_data, 1,
                                           (int[]){OUTPUT_SIZE}, TOFU_FLOAT);
tofu_graph_node* y_target = tofu_graph_input(g, t_target);

/* MSE loss */
tofu_graph_node* loss_node = tofu_graph_mse_loss(g, y_pred, y_target);
```

### Cross-Entropy (CE) Loss

Use cross-entropy for classification:

```c
/* Apply softmax first */
tofu_graph_node* probs = tofu_graph_softmax(g, logits, 1);

/* Compute CE loss with one-hot targets */
tofu_graph_node* loss = tofu_graph_ce_loss(g, probs, one_hot_target);
```

**Formula:** `L = -mean(sum(target * log(pred)))`

**When to use:**
- Multi-class classification
- Softmax output layer (probabilities)
- One-hot encoded targets
- Examples: MNIST digit classification, CNN pattern recognition

**Example from CNN training:**

```c
/* Forward pass with softmax */
tofu_graph_node* probs = cnn_forward_probs(g, input, params);

/* One-hot encode targets */
float* target_data = (float*)calloc(batch_size * num_classes, sizeof(float));
for (int i = 0; i < batch_size; i++) {
    target_data[i * num_classes + labels[i]] = 1.0f;
}
tofu_tensor* t_target = tofu_tensor_create(target_data, 2,
                                           (int[]){batch_size, num_classes},
                                           TOFU_FLOAT);
tofu_graph_node* target = tofu_graph_input(g, t_target);

/* Cross-entropy loss */
tofu_graph_node* loss_node = tofu_graph_ce_loss(g, probs, target);
```

### Extracting Loss Values

Get the scalar loss value for monitoring:

```c
tofu_tensor* loss_tensor = tofu_graph_get_value(loss_node);
float loss_value = 0.0f;
if (loss_tensor && loss_tensor->len > 0) {
    TOFU_TENSOR_DATA_TO(loss_tensor, 0, loss_value, TOFU_FLOAT);
}
```

### Monitoring Loss

Track loss to ensure training progresses:

```c
float epoch_loss = 0.0f;
int num_batches = 0;

for (int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
    /* ... forward pass and loss computation ... */

    float batch_loss = 0.0f;
    tofu_tensor* loss_tensor = tofu_graph_get_value(loss_node);
    if (loss_tensor && loss_tensor->len > 0) {
        TOFU_TENSOR_DATA_TO(loss_tensor, 0, batch_loss, TOFU_FLOAT);
    }

    epoch_loss += batch_loss;
    num_batches++;

    /* ... backward pass and update ... */
}

float avg_loss = epoch_loss / num_batches;
printf("Epoch %d: avg_loss = %.6f\n", epoch, avg_loss);
```

### Loss Patterns

**Healthy training:**
- Loss decreases steadily
- Eventually plateaus at a low value
- May have small oscillations

**Problem signs:**
- Loss increases: Learning rate too high or gradient explosion
- Loss stuck: Learning rate too low or poor initialization
- Loss = NaN: Numerical instability (try lower learning rate)

## Backward Pass

The backward pass computes gradients through automatic differentiation. TOFU handles the complexity; you just call one function.

### Invoking Backpropagation

```c
tofu_graph_backward(g, loss_node);
```

This single call:
1. Traverses the computation graph in reverse topological order
2. Applies the chain rule at each operation
3. Accumulates gradients in each node's `grad` field
4. Populates gradients for all parameter nodes

### How Automatic Differentiation Works

TOFU implements reverse-mode automatic differentiation (backpropagation):

```
Forward pass:  Input → Op1 → Op2 → ... → Loss
Backward pass: Loss → ∂Op2 → ∂Op1 → ... → ∂Input
```

Each operation knows how to compute its local gradient:
- **Matmul:** `∂L/∂A = ∂L/∂C @ B^T` and `∂L/∂B = A^T @ ∂L/∂C`
- **Add:** `∂L/∂A = ∂L/∂C` and `∂L/∂B = ∂L/∂C` (with broadcasting)
- **ReLU:** `∂L/∂x = ∂L/∂y * (x > 0)`
- **MSE:** `∂L/∂pred = 2 * (pred - target) / n`

### Gradient Flow Example

For a simple network `y = relu(x @ w + b)`:

```c
/* Forward pass builds graph */
tofu_graph_node* xw = tofu_graph_matmul(g, x, w);
tofu_graph_node* xw_b = tofu_graph_add(g, xw, b);
tofu_graph_node* y = tofu_graph_relu(g, xw_b);
tofu_graph_node* loss = tofu_graph_mse_loss(g, y, target);

/* Backward pass computes gradients */
tofu_graph_backward(g, loss);

/* Now gradients are available:
 * loss->grad: Always 1.0 (starting point)
 * y->grad: ∂L/∂y
 * xw_b->grad: ∂L/∂y * relu_grad
 * xw->grad: ∂L/∂(xw+b)
 * w->grad: x^T @ ∂L/∂(xw)  <- Used by optimizer
 * b->grad: sum(∂L/∂(xw+b))  <- Used by optimizer
 */
```

### Accessing Gradients

Check gradient values (useful for debugging):

```c
tofu_tensor* w_grad = tofu_graph_get_grad(w1);
if (w_grad) {
    printf("w1 gradient norm: ");
    float grad_sum = 0.0f;
    for (int i = 0; i < w_grad->len; i++) {
        float val;
        TOFU_TENSOR_DATA_TO(w_grad, i, val, TOFU_FLOAT);
        grad_sum += val * val;
    }
    printf("%.6f\n", sqrtf(grad_sum));
}
```

### Gradient Checking (Debug Tool)

Verify backprop implementation with numerical gradients:

```c
float numerical_gradient(tofu_graph* g, tofu_graph_node* param,
                        tofu_graph_node* loss, int param_idx) {
    const float epsilon = 1e-4f;

    /* Get parameter value */
    float original;
    TOFU_TENSOR_DATA_TO(param->value, param_idx, original, TOFU_FLOAT);

    /* Compute f(x + epsilon) */
    float perturbed = original + epsilon;
    TOFU_TENSOR_DATA_FROM(param->value, param_idx, perturbed, TOFU_FLOAT);
    tofu_graph_zero_grad(g);
    /* ... rerun forward pass ... */
    float loss_plus;
    TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_plus, TOFU_FLOAT);

    /* Compute f(x - epsilon) */
    perturbed = original - epsilon;
    TOFU_TENSOR_DATA_FROM(param->value, param_idx, perturbed, TOFU_FLOAT);
    tofu_graph_zero_grad(g);
    /* ... rerun forward pass ... */
    float loss_minus;
    TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_minus, TOFU_FLOAT);

    /* Restore original value */
    TOFU_TENSOR_DATA_FROM(param->value, param_idx, original, TOFU_FLOAT);

    /* Numerical gradient: (f(x+ε) - f(x-ε)) / (2ε) */
    return (loss_plus - loss_minus) / (2.0f * epsilon);
}

/* Compare with analytical gradient */
float analytical_grad;
TOFU_TENSOR_DATA_TO(tofu_graph_get_grad(param), param_idx, analytical_grad, TOFU_FLOAT);

float numerical_grad = numerical_gradient(g, param, loss, param_idx);
float relative_error = fabsf(analytical_grad - numerical_grad) /
                       fmaxf(fabsf(analytical_grad), fabsf(numerical_grad));

if (relative_error < 1e-5f) {
    printf("Gradient check PASSED (error: %.2e)\n", relative_error);
} else {
    printf("Gradient check FAILED (error: %.2e)\n", relative_error);
}
```

**Use gradient checking sparingly** - it's expensive (requires multiple forward passes per parameter).

### Common Gradient Issues

**Vanishing gradients:**
- Gradients become very small (near zero)
- Common with deep networks or saturating activations
- Solutions: Better initialization (Xavier), ReLU activations, batch normalization

**Exploding gradients:**
- Gradients become very large
- Loss becomes NaN
- Solutions: Lower learning rate, gradient clipping, better initialization

## Parameter Updates

After computing gradients, update parameters using the optimizer. This is where learning actually happens.

### Optimizer Step

```c
tofu_optimizer_step(optimizer);
```

This updates all parameters according to the optimizer's algorithm:

**SGD:** `param = param - learning_rate * grad`

**SGD with momentum:**
```
velocity = momentum * velocity + grad
param = param - learning_rate * velocity
```

### Choosing an Optimizer

TOFU provides two optimizers:

```c
/* Vanilla SGD */
tofu_optimizer* sgd = tofu_optimizer_sgd_create(g, 0.01);

/* SGD with momentum (recommended for most tasks) */
tofu_optimizer* sgd_momentum = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);
```

**Vanilla SGD:**
- Simplest algorithm
- Good for convex problems
- Can oscillate in ravines
- Use when: Memory is tight, simple problem

**SGD with Momentum:**
- Accumulates velocity
- Faster convergence
- Less oscillation
- Use when: Default choice for most problems

### Learning Rate Selection

The learning rate is the most important hyperparameter:

```c
/* Too high (0.5): May diverge or oscillate */
tofu_optimizer* opt_high = tofu_optimizer_sgd_create(g, 0.5);

/* Too low (0.0001): Very slow convergence */
tofu_optimizer* opt_low = tofu_optimizer_sgd_create(g, 0.0001);

/* Just right (0.01 - 0.1): Task-dependent */
tofu_optimizer* opt_good = tofu_optimizer_sgd_create(g, 0.01);
```

**Guidelines:**
- Start with 0.01 or 0.1
- If loss diverges: Reduce by 10x
- If convergence is slow: Increase by 2-3x
- Smaller networks often need larger learning rates
- Batch size matters: Larger batches → higher learning rate

### Parameter Update Timing

The order matters:

```c
/* Correct order */
tofu_graph_zero_grad(g);          /* 1. Clear old gradients */
/* forward pass */                /* 2. Compute predictions */
/* loss computation */             /* 3. Measure error */
tofu_graph_backward(g, loss);     /* 4. Compute gradients */
tofu_optimizer_step(optimizer);   /* 5. Update parameters */

/* WRONG: Update before backward */
tofu_optimizer_step(optimizer);   /* Updates with old/zero gradients! */
tofu_graph_backward(g, loss);
```

### Monitoring Parameter Changes

Track how much parameters change each step:

```c
/* Before update */
float w_before;
TOFU_TENSOR_DATA_TO(w1->value, 0, w_before, TOFU_FLOAT);

/* Update */
tofu_optimizer_step(optimizer);

/* After update */
float w_after;
TOFU_TENSOR_DATA_TO(w1->value, 0, w_after, TOFU_FLOAT);

float change = fabsf(w_after - w_before);
printf("Parameter change: %.6f\n", change);
```

**Healthy training:**
- Parameters change gradually
- Change magnitude decreases over time
- No sudden jumps

## Training Strategies

Effective training requires more than just the basic loop. Here are strategies for better results.

### Mini-Batch Training

Process data in batches instead of one sample at a time:

```c
const int BATCH_SIZE = 16;

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    /* Shuffle data (optional but recommended) */
    shuffle_dataset(dataset);

    for (int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
        int batch_end = (batch_start + BATCH_SIZE < num_samples)
                        ? batch_start + BATCH_SIZE
                        : num_samples;
        int actual_batch_size = batch_end - batch_start;

        /* Prepare batch */
        float* batch_data = create_batch(dataset, batch_start, actual_batch_size);
        tofu_tensor* t_batch = tofu_tensor_create(batch_data, 2,
                                                  (int[]){actual_batch_size, feature_dims},
                                                  TOFU_FLOAT);

        /* Train on batch */
        tofu_graph_zero_grad(g);
        tofu_graph_node* input = tofu_graph_input(g, t_batch);
        /* ... rest of training step ... */

        /* Cleanup */
        tofu_tensor_free(t_batch);
        free(batch_data);
        tofu_graph_clear_ops(g);
    }
}
```

**Batch size trade-offs:**
- **Larger (32-128):** More stable gradients, better GPU utilization, higher memory
- **Smaller (1-8):** Less memory, more updates, noisier gradients
- **Microcontroller:** Often limited to 1-4 due to memory constraints

### Epoch Management

An epoch is one complete pass through the training data:

```c
const int NUM_EPOCHS = 100;
float best_loss = INFINITY;

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    float epoch_loss = 0.0f;
    int num_batches = 0;

    /* Train on all batches */
    for (int batch = 0; batch < num_batches_total; batch++) {
        /* ... training step ... */
        epoch_loss += batch_loss;
        num_batches++;
    }

    /* Average loss over epoch */
    float avg_loss = epoch_loss / num_batches;

    /* Track best model */
    if (avg_loss < best_loss) {
        best_loss = avg_loss;
        /* Optionally save parameters */
    }

    /* Report progress */
    if (epoch % 10 == 0) {
        printf("Epoch %d: loss = %.6f\n", epoch, avg_loss);
    }
}
```

**How many epochs?**
- Too few: Underfitting (model hasn't learned)
- Too many: Overfitting (model memorizes training data)
- Monitor validation loss to determine when to stop

### Learning Rate Scheduling

Adjust learning rate during training:

```c
float initial_lr = 0.1f;

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    /* Step decay: Reduce by 10x every 50 epochs */
    float lr = initial_lr;
    if (epoch >= 50) lr *= 0.1f;
    if (epoch >= 100) lr *= 0.1f;

    /* Recreate optimizer with new learning rate */
    tofu_optimizer_free(optimizer);
    optimizer = tofu_optimizer_sgd_create(g, lr);

    /* ... training for this epoch ... */
}
```

**Common schedules:**
- **Step decay:** Reduce by constant factor at fixed intervals
- **Exponential decay:** `lr = lr0 * exp(-k * epoch)`
- **Warmup:** Start with low lr, gradually increase
- **Manual:** Reduce when loss plateaus

### Data Augmentation

Increase effective dataset size by transforming inputs:

```c
void augment_image(float* image, int width, int height) {
    /* Random flip */
    if (rand() % 2) {
        horizontal_flip(image, width, height);
    }

    /* Random noise */
    for (int i = 0; i < width * height; i++) {
        image[i] += 0.01f * ((float)rand() / RAND_MAX - 0.5f);
    }
}

/* Apply during training */
augment_image(batch_data, 8, 8);
tofu_tensor* t_input = tofu_tensor_create(batch_data, 2, ...);
```

**Augmentation techniques:**
- Rotation, flipping, cropping (images)
- Noise injection (signals)
- Time shifting (sequences)
- **Caution:** Some augmentations may not make sense for your data

### Early Stopping

Stop training when validation loss stops improving:

```c
float best_val_loss = INFINITY;
int patience = 10;  /* Number of epochs to wait */
int wait = 0;

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    /* Train */
    float train_loss = train_epoch(g, optimizer, train_data);

    /* Validate */
    float val_loss = evaluate(g, params, val_data);

    /* Check improvement */
    if (val_loss < best_val_loss) {
        best_val_loss = val_loss;
        wait = 0;
        /* Save best model */
    } else {
        wait++;
        if (wait >= patience) {
            printf("Early stopping at epoch %d\n", epoch);
            break;
        }
    }
}
```

## Monitoring and Debugging

Track training progress and diagnose issues.

### Loss Curves

Plot loss over time to understand training dynamics:

```c
#define MAX_EPOCHS 200
float train_losses[MAX_EPOCHS];
float val_losses[MAX_EPOCHS];

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    train_losses[epoch] = train_epoch(...);
    val_losses[epoch] = evaluate(...);

    printf("Epoch %3d: train_loss=%.4f, val_loss=%.4f\n",
           epoch, train_losses[epoch], val_losses[epoch]);
}

/* Analyze curves */
save_losses("losses.txt", train_losses, val_losses, NUM_EPOCHS);
```

**What to look for:**
- **Both decreasing:** Healthy training
- **Train decreases, val increases:** Overfitting
- **Both plateau:** Underfitting or need lower learning rate
- **Both increasing:** Learning rate too high

### Gradient Monitoring

Check gradient magnitudes:

```c
void check_gradients(tofu_graph* g, tofu_graph_node** params, int num_params) {
    for (int i = 0; i < num_params; i++) {
        tofu_tensor* grad = tofu_graph_get_grad(params[i]);
        if (!grad) continue;

        float grad_norm = 0.0f;
        for (int j = 0; j < grad->len; j++) {
            float val;
            TOFU_TENSOR_DATA_TO(grad, j, val, TOFU_FLOAT);
            grad_norm += val * val;
        }
        grad_norm = sqrtf(grad_norm);

        printf("Param %d gradient norm: %.6f\n", i, grad_norm);

        /* Warning signs */
        if (grad_norm < 1e-7f) {
            printf("  WARNING: Vanishing gradient!\n");
        }
        if (grad_norm > 1e3f) {
            printf("  WARNING: Exploding gradient!\n");
        }
    }
}

/* Call after backward pass */
tofu_graph_backward(g, loss);
check_gradients(g, params, num_params);
```

### Activation Statistics

Monitor activation distributions:

```c
void check_activations(tofu_graph_node* node) {
    tofu_tensor* act = tofu_graph_get_value(node);
    if (!act) return;

    float min_val = INFINITY, max_val = -INFINITY, mean = 0.0f;

    for (int i = 0; i < act->len; i++) {
        float val;
        TOFU_TENSOR_DATA_TO(act, i, val, TOFU_FLOAT);

        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        mean += val;
    }
    mean /= act->len;

    printf("Activation stats: min=%.4f, max=%.4f, mean=%.4f\n",
           min_val, max_val, mean);

    /* Warning signs */
    if (max_val - min_val < 1e-6f) {
        printf("  WARNING: Dead activations (all same)!\n");
    }
}
```

### Debugging Checklist

When training fails, check:

1. **Loss is NaN:**
   - Reduce learning rate
   - Check for division by zero
   - Verify input data normalization

2. **Loss doesn't decrease:**
   - Increase learning rate
   - Check gradient flow (print gradients)
   - Verify data/labels are correct
   - Try better initialization

3. **Training is slow:**
   - Increase learning rate
   - Use momentum
   - Check batch size
   - Verify network is not too large

4. **Overfitting:**
   - Add more training data
   - Reduce network size
   - Use validation set for early stopping

## Evaluation

After training, evaluate model performance on test data.

### Computing Accuracy

For classification tasks:

```c
float compute_accuracy(tofu_graph* g, cnn_params* params,
                      float* test_data, int* test_labels, int num_samples) {
    int correct = 0;

    for (int i = 0; i < num_samples; i++) {
        tofu_graph_zero_grad(g);

        /* Forward pass */
        float* input_data = &test_data[i * INPUT_SIZE];
        tofu_tensor* t_input = tofu_tensor_create(input_data, 1,
                                                  (int[]){INPUT_SIZE}, TOFU_FLOAT);
        tofu_graph_node* input = tofu_graph_input(g, t_input);
        tofu_graph_node* probs = cnn_forward_probs(g, input, params);

        /* Get prediction */
        tofu_tensor* probs_tensor = tofu_graph_get_value(probs);
        int pred_class = argmax(probs_tensor);

        if (pred_class == test_labels[i]) {
            correct++;
        }

        tofu_tensor_free(t_input);
        tofu_graph_clear_ops(g);
    }

    return (float)correct / num_samples;
}

/* Helper function */
int argmax(tofu_tensor* tensor) {
    int max_idx = 0;
    float max_val = -INFINITY;

    for (int i = 0; i < tensor->len; i++) {
        float val;
        TOFU_TENSOR_DATA_TO(tensor, i, val, TOFU_FLOAT);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    return max_idx;
}
```

### Regression Metrics

For regression tasks:

```c
float compute_mse(tofu_graph* g, tofu_graph_node* w1, tofu_graph_node* b1,
                 tofu_graph_node* w2, tofu_graph_node* b2,
                 float test_inputs[][2], float test_targets[][1], int num_samples) {
    float total_error = 0.0f;

    for (int i = 0; i < num_samples; i++) {
        /* Forward pass */
        float* input_data = (float*)malloc(2 * sizeof(float));
        input_data[0] = test_inputs[i][0];
        input_data[1] = test_inputs[i][1];

        tofu_tensor* t_input = tofu_tensor_create(input_data, 1, (int[]){2}, TOFU_FLOAT);
        tofu_graph_node* x = tofu_graph_input(g, t_input);

        /* Network computation */
        tofu_graph_node* h1 = tofu_graph_relu(g, tofu_graph_add(g,
                                               tofu_graph_matmul(g, x, w1), b1));
        tofu_graph_node* pred = tofu_graph_add(g, tofu_graph_matmul(g, h1, w2), b2);

        /* Get prediction */
        float pred_val;
        TOFU_TENSOR_DATA_TO(tofu_graph_get_value(pred), 0, pred_val, TOFU_FLOAT);

        /* Compute error */
        float error = pred_val - test_targets[i][0];
        total_error += error * error;

        tofu_tensor_free(t_input);
        free(input_data);
        tofu_graph_clear_ops(g);
    }

    return total_error / num_samples;
}
```

### Confusion Matrix

For detailed classification analysis:

```c
void compute_confusion_matrix(tofu_graph* g, cnn_params* params,
                             float* test_data, int* test_labels,
                             int num_samples, int num_classes,
                             int confusion[4][4]) {
    /* Initialize matrix */
    memset(confusion, 0, num_classes * num_classes * sizeof(int));

    for (int i = 0; i < num_samples; i++) {
        /* Get prediction */
        int pred_class = predict_sample(g, params, &test_data[i * INPUT_SIZE]);
        int true_class = test_labels[i];

        /* Update confusion matrix */
        confusion[true_class][pred_class]++;
    }

    /* Print matrix */
    printf("\nConfusion Matrix:\n");
    printf("      ");
    for (int i = 0; i < num_classes; i++) printf("%4d ", i);
    printf("\n");

    for (int i = 0; i < num_classes; i++) {
        printf("True %d: ", i);
        for (int j = 0; j < num_classes; j++) {
            printf("%4d ", confusion[i][j]);
        }
        printf("\n");
    }
}
```

## Complete Example

Here's a complete XOR training example bringing everything together:

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tofu_tensor.h"
#include "tofu_graph.h"
#include "tofu_optimizer.h"

/* Xavier initialization */
float xavier_init(int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    return limit * (2.0f * (float)rand() / RAND_MAX - 1.0f);
}

int main() {
    /* Configuration */
    const int INPUT_SIZE = 2, HIDDEN_SIZE = 4, OUTPUT_SIZE = 1;
    const int NUM_EPOCHS = 2000;
    const float LEARNING_RATE = 0.1f;

    /* XOR dataset */
    float inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float targets[4][1] = {{0}, {1}, {1}, {0}};

    /* Create graph */
    tofu_graph* g = tofu_graph_create();

    /* Initialize parameters */
    float* w1_data = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
        w1_data[i] = xavier_init(INPUT_SIZE);
    tofu_tensor* t_w1 = tofu_tensor_create(w1_data, 2,
                                           (int[]){INPUT_SIZE, HIDDEN_SIZE}, TOFU_FLOAT);
    tofu_graph_node* w1 = tofu_graph_param(g, t_w1);

    float* b1_data = calloc(HIDDEN_SIZE, sizeof(float));
    tofu_tensor* t_b1 = tofu_tensor_create(b1_data, 1, (int[]){HIDDEN_SIZE}, TOFU_FLOAT);
    tofu_graph_node* b1 = tofu_graph_param(g, t_b1);

    float* w2_data = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
        w2_data[i] = xavier_init(HIDDEN_SIZE);
    tofu_tensor* t_w2 = tofu_tensor_create(w2_data, 2,
                                           (int[]){HIDDEN_SIZE, OUTPUT_SIZE}, TOFU_FLOAT);
    tofu_graph_node* w2 = tofu_graph_param(g, t_w2);

    float* b2_data = calloc(OUTPUT_SIZE, sizeof(float));
    tofu_tensor* t_b2 = tofu_tensor_create(b2_data, 1, (int[]){OUTPUT_SIZE}, TOFU_FLOAT);
    tofu_graph_node* b2 = tofu_graph_param(g, t_b2);

    /* Create optimizer */
    tofu_optimizer* optimizer = tofu_optimizer_sgd_create(g, LEARNING_RATE);

    /* Training loop */
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        for (int sample = 0; sample < 4; sample++) {
            /* Zero gradients */
            tofu_graph_zero_grad(g);

            /* Create input */
            float* in_data = malloc(INPUT_SIZE * sizeof(float));
            in_data[0] = inputs[sample][0];
            in_data[1] = inputs[sample][1];
            tofu_tensor* t_in = tofu_tensor_create(in_data, 1, (int[]){INPUT_SIZE}, TOFU_FLOAT);
            tofu_graph_node* x = tofu_graph_input(g, t_in);

            /* Forward pass */
            tofu_graph_node* h1 = tofu_graph_relu(g, tofu_graph_add(g,
                                                   tofu_graph_matmul(g, x, w1), b1));
            tofu_graph_node* pred = tofu_graph_add(g, tofu_graph_matmul(g, h1, w2), b2);

            /* Create target */
            float* tgt_data = malloc(OUTPUT_SIZE * sizeof(float));
            tgt_data[0] = targets[sample][0];
            tofu_tensor* t_tgt = tofu_tensor_create(tgt_data, 1, (int[]){OUTPUT_SIZE}, TOFU_FLOAT);
            tofu_graph_node* tgt = tofu_graph_input(g, t_tgt);

            /* Compute loss */
            tofu_graph_node* loss = tofu_graph_mse_loss(g, pred, tgt);
            float loss_val;
            TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_val, TOFU_FLOAT);
            epoch_loss += loss_val;

            /* Backward pass */
            tofu_graph_backward(g, loss);

            /* Update parameters */
            tofu_optimizer_step(optimizer);

            /* Cleanup */
            tofu_tensor_free(t_in);
            tofu_tensor_free(t_tgt);
            free(in_data);
            free(tgt_data);
            tofu_graph_clear_ops(g);
        }

        /* Report progress */
        if (epoch % 200 == 0) {
            printf("Epoch %4d: loss = %.6f\n", epoch, epoch_loss / 4);
        }
    }

    /* Evaluate */
    printf("\nFinal predictions:\n");
    for (int i = 0; i < 4; i++) {
        float* in_data = malloc(INPUT_SIZE * sizeof(float));
        in_data[0] = inputs[i][0];
        in_data[1] = inputs[i][1];
        tofu_tensor* t_in = tofu_tensor_create(in_data, 1, (int[]){INPUT_SIZE}, TOFU_FLOAT);
        tofu_graph_node* x = tofu_graph_input(g, t_in);

        tofu_graph_node* h1 = tofu_graph_relu(g, tofu_graph_add(g,
                                               tofu_graph_matmul(g, x, w1), b1));
        tofu_graph_node* pred = tofu_graph_add(g, tofu_graph_matmul(g, h1, w2), b2);

        float pred_val;
        TOFU_TENSOR_DATA_TO(tofu_graph_get_value(pred), 0, pred_val, TOFU_FLOAT);

        printf("[%.0f, %.0f] -> %.4f (target: %.0f)\n",
               inputs[i][0], inputs[i][1], pred_val, targets[i][0]);

        tofu_tensor_free(t_in);
        free(in_data);
        tofu_graph_clear_ops(g);
    }

    /* Cleanup */
    tofu_optimizer_free(optimizer);
    tofu_graph_free(g);
    tofu_tensor_free_data_too(t_w1);
    tofu_tensor_free_data_too(t_b1);
    tofu_tensor_free_data_too(t_w2);
    tofu_tensor_free_data_too(t_b2);

    return 0;
}
```

This example demonstrates:
- Parameter initialization with Xavier method
- Complete training loop with all five steps
- Proper memory management (malloc/free)
- Graph reuse via clear_ops
- Loss monitoring during training
- Final evaluation on the dataset

## Best Practices

### Memory Management

**Always free tensors in correct order:**
```c
/* Correct order */
tofu_optimizer_free(optimizer);  /* 1. Free optimizer first */
tofu_graph_free(g);              /* 2. Free graph second */
tofu_tensor_free_data_too(t_w1); /* 3. Free parameter tensors last */
tofu_tensor_free_data_too(t_b1);
```

**Use clear_ops between iterations:**
```c
for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    /* ... training step ... */
    tofu_graph_clear_ops(g);  /* Prevents memory leaks */
}
```

### Initialization

**Use Xavier/He initialization:**
```c
/* Xavier: Good for tanh/sigmoid */
float xavier = sqrtf(6.0f / fan_in) * (2.0f * rand() / RAND_MAX - 1.0f);

/* He: Better for ReLU */
float he = sqrtf(2.0f / fan_in) * (2.0f * rand() / RAND_MAX - 1.0f);
```

**Never initialize to all zeros:**
```c
/* WRONG: Breaks symmetry, prevents learning */
float* w_data = calloc(size, sizeof(float));

/* CORRECT: Random initialization */
for (int i = 0; i < size; i++)
    w_data[i] = xavier_init(fan_in);
```

### Hyperparameter Tuning

**Start with these defaults:**
- Learning rate: 0.01 - 0.1
- Batch size: 1 - 16 (for microcontrollers)
- Hidden layer size: 2x - 4x input size
- Epochs: 100 - 1000

**Tune systematically:**
1. Get the model working at all (reduce problem size if needed)
2. Tune learning rate (most important)
3. Tune architecture (layer sizes)
4. Tune batch size (memory permitting)

### Debugging

**Print everything during development:**
```c
printf("Loss: %.6f\n", loss_val);
printf("Grad norm: %.6f\n", grad_norm);
printf("Prediction: %.4f, Target: %.4f\n", pred, target);
```

**Check intermediate values:**
```c
tofu_tensor* h1_val = tofu_graph_get_value(h1);
printf("Hidden layer stats: ");
print_tensor_stats(h1_val);
```

**Start simple, scale up:**
1. Verify on tiny dataset (4 samples)
2. Check on small network (few parameters)
3. Scale to full problem once working

### Resource Constraints

For microcontrollers, minimize memory usage:
- Use batch_size=1 if memory is tight
- Keep networks small (< 10k parameters)
- Reuse graph with clear_ops
- Consider quantization (future work)
- Profile memory usage regularly

With these best practices, you're ready to train neural networks on TOFU. See the examples directory for more complete training scripts.
