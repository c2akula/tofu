# Optimizers

Optimizers update neural network parameters using computed gradients. Understanding how optimizers work and how to tune them is essential for training models effectively.

## Introduction

Training a neural network means finding parameter values that minimize a loss function. This is an optimization problem: start with random parameters, compute gradients that indicate how to adjust them, and iteratively update parameters to reduce loss.

Optimizers automate this process. They take computed gradients and apply update rules to parameters. Different optimizers use different strategies—some use only the current gradient (like SGD), while others accumulate information from previous steps (like momentum-based methods).

This guide explains optimizer fundamentals, shows you how to create and use optimizers, describes the algorithms available in Tofu, and provides practical guidance for tuning hyperparameters and troubleshooting training issues.

## Optimizer Fundamentals

Understanding how optimizers work requires grasping two key concepts: gradient descent and the learning rate.

### Gradient Descent: Following the Slope Downhill

Imagine you're standing on a mountain in fog, trying to reach the lowest point. You can't see far, but you can feel which direction slopes downward beneath your feet. Gradient descent works the same way: at each step, compute which direction reduces the loss function, then take a small step in that direction.

Mathematically, for a parameter theta and loss L:

```
theta_new = theta_old - learning_rate * gradient
```

Where `gradient = dL/dtheta` (the derivative of loss with respect to the parameter).

The gradient points in the direction of steepest ascent (uphill). By subtracting it, we move downhill toward lower loss.

### Learning Rate: Step Size Matters

The learning rate controls how large a step to take. This is the single most important hyperparameter in training neural networks.

**Too large**: You'll overshoot the minimum, potentially making loss worse or causing training to diverge completely.

```
Loss landscape:  \    /
                  \__/
With large steps: --> X <-- (overshoot back and forth)
```

**Too small**: Training converges slowly. You'll make progress, but it might take 10x or 100x more iterations than necessary.

```
Loss landscape:  \    /
                  \__/
With tiny steps:  . . . . . (very slow progress)
```

**Just right**: Training converges efficiently without instability.

```
Loss landscape:  \    /
                  \__/
Good step size:   -> -> -> (steady progress to minimum)
```

Typical learning rates range from 0.0001 to 0.1. Start with 0.01 and adjust based on training behavior.

### Stochastic Gradient Descent (SGD)

Classical gradient descent computes gradients using the entire training dataset. This is expensive and slow. Stochastic Gradient Descent (SGD) uses small batches of data instead—typically 32, 64, or 128 examples at a time.

The "stochastic" (random) part means each batch gives a noisy estimate of the true gradient. But averaging over many batches gives the correct direction, and computing on small batches is much faster than using the entire dataset.

In practice, when people say "SGD," they usually mean "mini-batch SGD"—computing gradients on small batches rather than single examples or the full dataset.

### Why Multiple Optimizer Types?

If vanilla SGD works, why do we need other optimizers? Because SGD has limitations:

1. **Slow convergence** on complex loss landscapes
2. **Oscillation** in narrow valleys (moves back and forth rather than forward)
3. **Sensitivity** to learning rate choice

Advanced optimizers like SGD with momentum address these issues by accumulating information about previous gradients. This helps accelerate training and dampen oscillations.

## Creating Optimizers

Optimizers in Tofu are tied to computation graphs. When you create an optimizer, it automatically collects all trainable parameters (nodes created with `tofu_graph_param`) from the graph.

### Basic Setup

Creating an optimizer follows this pattern:

```c
// 1. Create graph and add parameters
tofu_graph *g = tofu_graph_create();

tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){784, 10}, TOFU_FLOAT);
tofu_tensor *bias = tofu_tensor_zeros(1, (int[]){10}, TOFU_FLOAT);

tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *b = tofu_graph_param(g, bias);

// 2. Create optimizer (automatically finds W and b)
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// 3. Use in training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    tofu_optimizer_zero_grad(opt);
    // ... forward pass, compute loss ...
    tofu_graph_backward(g, loss);
    tofu_optimizer_step(opt);
}

// 4. Cleanup (optimizer before graph)
tofu_optimizer_free(opt);
tofu_graph_free(g);
tofu_tensor_free_data_too(weights);
tofu_tensor_free_data_too(bias);
```

Key points:

- **Automatic parameter collection**: The optimizer scans the graph and finds all PARAM nodes when created
- **One optimizer per graph**: Each optimizer manages parameters from a single graph
- **Cleanup order matters**: Always free the optimizer before the graph

### Choosing a Learning Rate

Start with these defaults:

- **0.01** - Safe starting point for most problems
- **0.001** - Deep networks, complex problems
- **0.1** - Small networks, simple problems

After a few iterations, check if loss is decreasing. If not, reduce the learning rate by 10x. If loss decreases very slowly, try increasing by 2x-5x.

### Memory Considerations

Different optimizers have different memory requirements:

- **SGD**: No extra memory (just the parameters themselves)
- **SGD with momentum**: One velocity buffer per parameter (doubles memory)

For large networks on memory-constrained devices, vanilla SGD may be the only option. For everything else, momentum is usually worth the extra memory.

## SGD: Stochastic Gradient Descent

Vanilla SGD is the simplest optimizer. It updates parameters by directly subtracting the scaled gradient.

### The Algorithm

For each parameter theta:

```
theta = theta - learning_rate * gradient
```

That's it. Compute the gradient, scale it by the learning rate, subtract from the parameter.

In code:

```c
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);
```

The optimizer applies this update rule to every parameter automatically when you call `tofu_optimizer_step()`.

### Implementation Example

Here's a complete training loop using SGD:

```c
// Setup
tofu_graph *g = tofu_graph_create();

// Network: linear layer (input_dim=4, output_dim=3)
tofu_tensor *W = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_tensor *b = tofu_tensor_zeros(1, (int[]){3}, TOFU_FLOAT);

tofu_graph_node *W_node = tofu_graph_param(g, W);
tofu_graph_node *b_node = tofu_graph_param(g, b);

// Create SGD optimizer with learning rate 0.01
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    double epoch_loss = 0.0;

    for (int batch = 0; batch < num_batches; batch++) {
        // Zero gradients before forward pass
        tofu_optimizer_zero_grad(opt);

        // Forward pass: pred = input @ W + b
        tofu_graph_node *x = tofu_graph_input(g, batch_data[batch]);
        tofu_graph_node *h = tofu_graph_matmul(g, x, W_node);
        tofu_graph_node *pred = tofu_graph_add(g, h, b_node);

        // Compute loss
        tofu_graph_node *target = tofu_graph_input(g, batch_targets[batch]);
        tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

        // Get loss value for logging
        float loss_val;
        tofu_tensor *loss_tensor = tofu_graph_get_value(loss);
        TOFU_TENSOR_DATA_TO(loss_tensor, 0, loss_val, TOFU_FLOAT);
        epoch_loss += loss_val;

        // Backward pass: compute gradients
        tofu_graph_backward(g, loss);

        // Update parameters using gradients
        tofu_optimizer_step(opt);

        // Clear operations for next batch
        tofu_graph_clear_ops(g);
    }

    printf("Epoch %d: Loss = %.6f\n", epoch, epoch_loss / num_batches);
}

// Cleanup
tofu_optimizer_free(opt);
tofu_graph_free(g);
tofu_tensor_free_data_too(W);
tofu_tensor_free_data_too(b);
```

### When to Use SGD

SGD works well when:

- **Memory is tight**: SGD has no extra memory overhead
- **Loss landscape is smooth**: Few local minima, well-conditioned gradients
- **You have time to tune**: SGD is sensitive to learning rate, so you'll need to experiment

SGD struggles when:

- **Loss landscape is complex**: Many local minima or saddle points
- **Gradients are noisy**: High variance in gradient estimates
- **Convergence needs to be fast**: SGD converges slower than momentum-based methods

### Tuning SGD

The learning rate is the only hyperparameter for vanilla SGD. Here's how to tune it:

**Start with 0.01**:
```c
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);
```

**Watch the first few iterations**:

- Loss decreasing steadily: Good sign, continue training
- Loss increasing or NaN: Learning rate too high, reduce by 10x
- Loss barely changing: Learning rate too low, increase by 2x-5x

**Common learning rate values**:

- 0.1 - Aggressive, works for simple problems
- 0.01 - Conservative, good default
- 0.001 - Very conservative, deep networks
- 0.0001 - Fine-tuning pretrained models

**Monitoring training**:

```c
for (int epoch = 0; epoch < max_epochs; epoch++) {
    // ... training loop ...

    if (epoch_loss < best_loss) {
        best_loss = epoch_loss;
        no_improvement_count = 0;
    } else {
        no_improvement_count++;
    }

    // Reduce learning rate if stuck
    if (no_improvement_count > 10) {
        opt->learning_rate *= 0.5;
        printf("Reducing learning rate to %.6f\n", opt->learning_rate);
        no_improvement_count = 0;
    }
}
```

## SGD with Momentum

Momentum helps SGD converge faster and more smoothly by accumulating a velocity term that averages gradients over time. This dampens oscillations and accelerates progress in consistent directions.

### The Algorithm

Instead of directly using the current gradient, momentum maintains a velocity vector that accumulates gradients exponentially:

```
v = momentum * v - learning_rate * gradient
theta = theta + v
```

Where:
- `v` is the velocity (initialized to zero)
- `momentum` is a coefficient (typically 0.9)
- `learning_rate` scales the gradient contribution
- `gradient` is the current parameter gradient

This differs from classical momentum formulations but is mathematically equivalent. The key insight: multiply the velocity by momentum (typically 0.9), then subtract the scaled gradient and add the result to the parameter.

### Why Momentum Works

Think of momentum as a ball rolling downhill. When the slope consistently points in one direction, the ball accelerates (velocity builds up). When the slope changes direction, the accumulated velocity smooths out oscillations.

**Without momentum** (vanilla SGD):
```
Narrow valley:  |        |
Path taken:     | -> <- ->| (oscillates back and forth)
                | -> <- ->|
```

**With momentum**:
```
Narrow valley:  |        |
Path taken:     |  -->   | (smooth progress forward)
                |   -->  |
```

Momentum provides two benefits:

1. **Acceleration**: Builds up speed in consistent directions
2. **Dampening**: Reduces oscillations in directions that change frequently

### Implementation Example

Creating an SGD optimizer with momentum requires one additional parameter:

```c
// Create optimizer with learning_rate=0.01, momentum=0.9
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);
```

The rest of the training loop is identical to vanilla SGD:

```c
// Setup
tofu_graph *g = tofu_graph_create();

// Network: two-layer MLP
tofu_tensor *W1 = tofu_tensor_zeros(2, (int[]){784, 128}, TOFU_FLOAT);
tofu_tensor *b1 = tofu_tensor_zeros(1, (int[]){128}, TOFU_FLOAT);
tofu_tensor *W2 = tofu_tensor_zeros(2, (int[]){128, 10}, TOFU_FLOAT);
tofu_tensor *b2 = tofu_tensor_zeros(1, (int[]){10}, TOFU_FLOAT);

tofu_graph_node *W1_node = tofu_graph_param(g, W1);
tofu_graph_node *b1_node = tofu_graph_param(g, b1);
tofu_graph_node *W2_node = tofu_graph_param(g, W2);
tofu_graph_node *b2_node = tofu_graph_param(g, b2);

// Create optimizer with momentum
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        tofu_optimizer_zero_grad(opt);

        // Forward pass
        tofu_graph_node *x = tofu_graph_input(g, batch_data[batch]);

        // Layer 1: h1 = relu(x @ W1 + b1)
        tofu_graph_node *h1 = tofu_graph_matmul(g, x, W1_node);
        h1 = tofu_graph_add(g, h1, b1_node);
        h1 = tofu_graph_relu(g, h1);

        // Layer 2: output = h1 @ W2 + b2
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

### Tuning Momentum

Momentum has two hyperparameters: learning rate and momentum coefficient.

**Learning Rate**: Start with the same values as vanilla SGD (0.01 is a good default). Momentum often allows slightly higher learning rates because it dampens oscillations.

**Momentum Coefficient**: Controls how much past gradients influence current updates.

Common values:

- **0.9** - Standard choice, works well for most problems
- **0.95** - High momentum, use for slow convergence
- **0.99** - Very high momentum, use for very deep networks
- **0.5-0.8** - Low momentum, use if training is unstable

The momentum coefficient is easier to tune than learning rate. Start with 0.9 and adjust if needed.

### When to Use Momentum

Use momentum when:

- **Training is slow**: Momentum accelerates convergence
- **Gradients are noisy**: Momentum smooths out noise
- **Deep networks**: Momentum helps propagate gradients through many layers
- **Memory is available**: Momentum requires one velocity buffer per parameter

Stick with vanilla SGD when:

- **Memory is very tight**: Momentum doubles memory requirements
- **Loss landscape is simple**: Vanilla SGD may be sufficient

In practice, momentum is the default choice for most problems. The memory cost is usually worth the faster convergence.

## Using Optimizers in Training

Now that you understand optimizer algorithms, let's look at the mechanics of using them in training loops.

### The Training Cycle

Every training iteration follows the same four-step pattern:

```
1. Zero gradients    (tofu_optimizer_zero_grad)
2. Forward pass      (build computation graph)
3. Backward pass     (tofu_graph_backward)
4. Update parameters (tofu_optimizer_step)
```

This cycle repeats for every batch in every epoch.

### Step-by-Step Breakdown

**Step 1: Zero Gradients**

Gradients accumulate by default in Tofu. If you don't zero them, they'll keep adding up across iterations, leading to incorrect updates.

```c
tofu_optimizer_zero_grad(opt);
```

This clears all gradient buffers for parameters tracked by the optimizer. Always call this before the forward pass.

**Step 2: Forward Pass**

Build the computation graph by adding operations. Each operation automatically computes its value:

```c
tofu_graph_node *x = tofu_graph_input(g, input_data);
tofu_graph_node *h = tofu_graph_matmul(g, x, W);
tofu_graph_node *pred = tofu_graph_add(g, h, b);
tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);
```

At this point, `loss` contains the computed loss value, but gradients haven't been computed yet.

**Step 3: Backward Pass**

Compute gradients by calling backward on the loss node:

```c
tofu_graph_backward(g, loss);
```

This triggers reverse-mode automatic differentiation. Tofu walks the graph backwards, computing gradients for every parameter using the chain rule. After this call, every parameter has its gradient stored in `node->grad`.

**Step 4: Update Parameters**

Apply the optimizer's update rule to adjust parameters:

```c
tofu_optimizer_step(opt);
```

This uses the computed gradients to update parameters. For SGD, it subtracts `learning_rate * gradient` from each parameter. For momentum, it updates velocity buffers and then parameters.

**Step 5: Clear Operations**

Before the next iteration, clear operation nodes from the graph while preserving parameters:

```c
tofu_graph_clear_ops(g);
```

This frees memory used by intermediate computations (matmul results, activations, etc.) but keeps parameters and their gradients intact.

### Complete Training Loop

Here's a full training loop with all the pieces:

```c
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    double total_loss = 0.0;

    for (int batch = 0; batch < num_batches; batch++) {
        // 1. Zero gradients
        tofu_optimizer_zero_grad(opt);

        // 2. Forward pass
        tofu_graph_node *x = tofu_graph_input(g, batch_data[batch]);
        tofu_graph_node *pred = forward_pass(g, x);  // Your model
        tofu_graph_node *target = tofu_graph_input(g, batch_targets[batch]);
        tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

        // Track loss for logging
        float loss_val;
        TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_val, TOFU_FLOAT);
        total_loss += loss_val;

        // 3. Backward pass
        tofu_graph_backward(g, loss);

        // 4. Update parameters
        tofu_optimizer_step(opt);

        // 5. Clear operations
        tofu_graph_clear_ops(g);
    }

    // Log epoch statistics
    printf("Epoch %d: Avg Loss = %.6f\n", epoch, total_loss / num_batches);
}
```

### Common Mistakes

**Mistake 1: Forgetting to zero gradients**

```c
// WRONG: Gradients accumulate indefinitely
for (int i = 0; i < iterations; i++) {
    // No zero_grad call!
    // ... forward, backward, step ...
}
```

This causes gradients to grow without bound. Updates become incorrect after the first iteration.

**Correct:**
```c
for (int i = 0; i < iterations; i++) {
    tofu_optimizer_zero_grad(opt);  // Clear old gradients
    // ... forward, backward, step ...
}
```

**Mistake 2: Calling step before backward**

```c
// WRONG: No gradients computed yet!
tofu_optimizer_step(opt);
tofu_graph_backward(g, loss);
```

The optimizer needs gradients to update parameters. Always call backward before step.

**Correct:**
```c
tofu_graph_backward(g, loss);   // Compute gradients
tofu_optimizer_step(opt);        // Use gradients to update
```

**Mistake 3: Not clearing operations**

```c
// WRONG: Memory grows indefinitely
for (int batch = 0; batch < num_batches; batch++) {
    // ... training ...
    // No clear_ops call!
}
```

Each batch adds nodes to the graph. Without clearing, memory usage grows until the program crashes.

**Correct:**
```c
for (int batch = 0; batch < num_batches; batch++) {
    // ... training ...
    tofu_graph_clear_ops(g);  // Clear after each batch
}
```

### Monitoring Training

Track key metrics to understand training progress:

```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    double epoch_loss = 0.0;
    int num_correct = 0;

    for (int batch = 0; batch < num_batches; batch++) {
        // ... training loop ...

        // Track loss
        float loss_val;
        TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_val, TOFU_FLOAT);
        epoch_loss += loss_val;

        // Track accuracy (for classification)
        num_correct += count_correct_predictions(pred, target);
    }

    double avg_loss = epoch_loss / num_batches;
    double accuracy = (double)num_correct / (num_batches * batch_size);

    printf("Epoch %d: Loss = %.6f, Accuracy = %.2f%%\n",
           epoch, avg_loss, accuracy * 100);
}
```

## Learning Rate Strategies

The learning rate often needs adjustment during training. Starting with a fixed rate works for simple problems, but complex models benefit from learning rate schedules.

### Fixed Learning Rate

The simplest strategy: use the same learning rate throughout training.

```c
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// Training loop uses 0.01 for all epochs
for (int epoch = 0; epoch < 100; epoch++) {
    // ... training ...
}
```

This works well when:
- The problem is simple
- You've found a good learning rate through experimentation
- Training converges in a reasonable number of epochs

### Step Decay

Reduce the learning rate by a fixed factor every N epochs:

```c
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.1);

for (int epoch = 0; epoch < 100; epoch++) {
    // Reduce learning rate by 10x every 30 epochs
    if (epoch % 30 == 0 && epoch > 0) {
        opt->learning_rate *= 0.1;
        printf("Epoch %d: Learning rate reduced to %.6f\n",
               epoch, opt->learning_rate);
    }

    // ... training loop ...
}
```

Common schedules:
- Divide by 10 every 30 epochs (0.1 -> 0.01 -> 0.001)
- Divide by 2 every 10 epochs (0.1 -> 0.05 -> 0.025)

Step decay is simple and effective for many problems.

### Exponential Decay

Gradually reduce the learning rate every epoch:

```c
double initial_lr = 0.1;
double decay_rate = 0.95;

tofu_optimizer *opt = tofu_optimizer_sgd_create(g, initial_lr);

for (int epoch = 0; epoch < 100; epoch++) {
    // Update learning rate
    opt->learning_rate = initial_lr * pow(decay_rate, epoch);

    if (epoch % 10 == 0) {
        printf("Epoch %d: Learning rate = %.6f\n", epoch, opt->learning_rate);
    }

    // ... training loop ...
}
```

This provides smooth, gradual decay. The decay rate controls how quickly the learning rate decreases (0.95 is typical).

### Cosine Annealing

Reduce the learning rate following a cosine curve:

```c
#include <math.h>

double initial_lr = 0.1;
double min_lr = 0.001;
int num_epochs = 100;

tofu_optimizer *opt = tofu_optimizer_sgd_create(g, initial_lr);

for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Cosine annealing formula
    double progress = (double)epoch / num_epochs;
    opt->learning_rate = min_lr + (initial_lr - min_lr) *
                         (1.0 + cos(M_PI * progress)) / 2.0;

    // ... training loop ...
}
```

Cosine annealing provides smooth decay that starts fast and slows down near the end.

### Learning Rate Warmup

For very high initial learning rates, gradually increase from a small value:

```c
double target_lr = 0.1;
int warmup_epochs = 5;

tofu_optimizer *opt = tofu_optimizer_sgd_create(g, target_lr / warmup_epochs);

for (int epoch = 0; epoch < 100; epoch++) {
    // Warmup phase
    if (epoch < warmup_epochs) {
        opt->learning_rate = target_lr * (epoch + 1) / warmup_epochs;
    }

    // ... training loop ...
}
```

Warmup prevents instability in the first few epochs when using aggressive learning rates.

### Adaptive Scheduling

Reduce the learning rate when progress stalls:

```c
double best_loss = INFINITY;
int patience = 5;
int no_improvement_count = 0;

tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

for (int epoch = 0; epoch < 100; epoch++) {
    // ... training loop ...
    double epoch_loss = compute_epoch_loss();

    // Track progress
    if (epoch_loss < best_loss) {
        best_loss = epoch_loss;
        no_improvement_count = 0;
    } else {
        no_improvement_count++;
    }

    // Reduce learning rate if stuck
    if (no_improvement_count >= patience) {
        opt->learning_rate *= 0.5;
        printf("Reducing learning rate to %.6f\n", opt->learning_rate);
        no_improvement_count = 0;
    }
}
```

This adapts to training dynamics automatically, reducing the learning rate only when needed.

### Choosing a Strategy

**Start simple**: Use a fixed learning rate first. Only add scheduling if training plateaus.

**Step decay**: Good default for most problems. Easy to understand and implement.

**Exponential/Cosine**: Use for long training runs (100+ epochs) where smooth decay is beneficial.

**Adaptive**: Best when you're not sure how many epochs you need or when progress is unpredictable.

**Warmup**: Use when starting with very high learning rates (0.1+) to prevent early instability.

## Choosing an Optimizer

With multiple optimizers available, how do you choose? Here's a practical decision guide.

### Start with SGD + Momentum

For most problems, SGD with momentum (0.01 learning rate, 0.9 momentum) is the best starting point:

```c
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);
```

This provides:
- Good convergence speed
- Reasonable memory overhead
- Robustness to hyperparameter choices

### Decision Tree

```
Is memory extremely tight? (< 2x parameter memory available)
├─ YES: Use vanilla SGD
└─ NO: Continue

Is the problem very simple? (linear model, small dataset)
├─ YES: Use vanilla SGD (momentum won't help much)
└─ NO: Continue

Is the network deep? (> 5 layers)
├─ YES: Use SGD with momentum 0.9 or higher
└─ NO: Use SGD with momentum 0.9
```

### Comparison Table

| Optimizer | Memory | Convergence | Tuning | Best For |
|-----------|--------|-------------|---------|----------|
| SGD | Minimal | Slower | Difficult | Memory-constrained, simple problems |
| SGD+Momentum | 2x params | Faster | Moderate | General purpose, deep networks |

### Network Depth Considerations

**Shallow networks (1-3 layers)**:
- Vanilla SGD often sufficient
- Momentum helps but not essential

**Medium networks (4-10 layers)**:
- Momentum recommended
- Use momentum 0.9

**Deep networks (10+ layers)**:
- Momentum essential
- Use momentum 0.95-0.99

### Problem Type Recommendations

**Regression (MSE loss)**:
```c
// Start here
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);
```

**Classification (cross-entropy loss)**:
```c
// May need higher learning rate
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.05, 0.9);
```

**Fine-tuning pretrained models**:
```c
// Very small learning rate to preserve learned features
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.0001);
```

### When in Doubt

Default configuration for new problems:

```c
tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);
```

This works well for most cases. Adjust based on training behavior:
- Loss diverges: Reduce learning rate by 10x
- Convergence too slow: Increase learning rate by 2-5x
- Still slow: Increase momentum to 0.95

## Troubleshooting

Training neural networks is often an iterative process of diagnosing and fixing issues. Here are common problems and their solutions.

### Loss is NaN or Infinite

**Symptoms**: Loss becomes NaN or infinity after a few iterations.

**Causes**:
1. Learning rate too high
2. Gradient explosion (very large gradients)
3. Numerical instability in loss function

**Solutions**:

Reduce learning rate dramatically:
```c
// If using 0.01, try 0.001
opt->learning_rate = 0.001;
```

Check gradients for extreme values:
```c
tofu_graph_backward(g, loss);

// Before optimizer step, check gradient magnitudes
for (int i = 0; i < opt->num_params; i++) {
    tofu_tensor *grad = tofu_graph_get_grad(opt->params[i]);
    double max_grad = find_max_abs_value(grad);

    if (max_grad > 1000.0) {
        printf("Warning: Large gradient detected: %.2f\n", max_grad);
    }
}

tofu_optimizer_step(opt);
```

Implement gradient clipping:
```c
void clip_gradients(tofu_optimizer *opt, double max_norm) {
    for (int i = 0; i < opt->num_params; i++) {
        tofu_tensor *grad = tofu_graph_get_grad(opt->params[i]);
        if (!grad) continue;

        // Compute L2 norm
        double norm = 0.0;
        for (int j = 0; j < grad->len; j++) {
            float val;
            TOFU_TENSOR_DATA_TO(grad, j, val, TOFU_FLOAT);
            norm += val * val;
        }
        norm = sqrt(norm);

        // Clip if too large
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

// Use before optimizer step
tofu_graph_backward(g, loss);
clip_gradients(opt, 1.0);  // Clip to max norm of 1.0
tofu_optimizer_step(opt);
```

### Loss Not Decreasing

**Symptoms**: Loss stays constant or decreases very slowly.

**Causes**:
1. Learning rate too low
2. Model stuck in poor initialization
3. Gradient vanishing
4. Wrong loss function or labels

**Solutions**:

Increase learning rate:
```c
opt->learning_rate *= 10.0;  // Try 10x higher
```

Check if gradients are flowing:
```c
tofu_graph_backward(g, loss);

// Check if gradients are non-zero
for (int i = 0; i < opt->num_params; i++) {
    tofu_tensor *grad = tofu_graph_get_grad(opt->params[i]);
    double sum_abs = 0.0;

    for (int j = 0; j < grad->len; j++) {
        float val;
        TOFU_TENSOR_DATA_TO(grad, j, val, TOFU_FLOAT);
        sum_abs += fabs(val);
    }

    double mean_abs = sum_abs / grad->len;

    if (mean_abs < 1e-7) {
        printf("Warning: Very small gradients (%.2e) for parameter %d\n",
               mean_abs, i);
    }
}
```

Try momentum if using vanilla SGD:
```c
// Replace vanilla SGD with momentum
tofu_optimizer_free(opt);
opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);
```

### Loss Oscillates

**Symptoms**: Loss goes up and down rather than steadily decreasing.

**Causes**:
1. Learning rate too high
2. Batch size too small (noisy gradients)
3. Wrong momentum setting

**Solutions**:

Reduce learning rate:
```c
opt->learning_rate *= 0.5;  // Try half the current rate
```

Use or increase momentum:
```c
// If using vanilla SGD, add momentum
tofu_optimizer_free(opt);
opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.9);

// If already using momentum, increase it
tofu_optimizer_free(opt);
opt = tofu_optimizer_sgd_momentum_create(g, 0.01, 0.95);
```

### Training Slows Down Over Time

**Symptoms**: Loss decreases quickly at first, then stalls.

**Causes**:
1. Learning rate too high for fine-tuning
2. Converging to local minimum
3. Need learning rate schedule

**Solutions**:

Implement step decay:
```c
for (int epoch = 0; epoch < 100; epoch++) {
    // Reduce learning rate when progress slows
    if (epoch == 30 || epoch == 60 || epoch == 90) {
        opt->learning_rate *= 0.1;
        printf("Reduced learning rate to %.6f\n", opt->learning_rate);
    }

    // ... training loop ...
}
```

Use adaptive scheduling:
```c
double best_loss = INFINITY;
int no_improvement = 0;

for (int epoch = 0; epoch < 100; epoch++) {
    // ... training ...

    if (epoch_loss < best_loss) {
        best_loss = epoch_loss;
        no_improvement = 0;
    } else {
        no_improvement++;
    }

    if (no_improvement > 5) {
        opt->learning_rate *= 0.5;
        no_improvement = 0;
    }
}
```

### Memory Issues

**Symptoms**: Program crashes with allocation errors or runs out of memory.

**Causes**:
1. Not clearing operations between batches
2. Momentum optimizer on large networks
3. Accumulating tensors unintentionally

**Solutions**:

Always clear operations:
```c
for (int batch = 0; batch < num_batches; batch++) {
    // ... training ...
    tofu_graph_clear_ops(g);  // Essential for memory management
}
```

Use vanilla SGD if momentum exhausts memory:
```c
// Switch from momentum to vanilla SGD
tofu_optimizer_free(opt);
opt = tofu_optimizer_sgd_create(g, 0.01);
```

## Best Practices

Here are guidelines to help you train models effectively and avoid common pitfalls.

### Always Zero Gradients

Make this the first line of every training iteration:

```c
for (int batch = 0; batch < num_batches; batch++) {
    tofu_optimizer_zero_grad(opt);  // Never forget this!
    // ... rest of training loop ...
}
```

Without this, gradients accumulate across batches, leading to incorrect updates.

### Monitor Multiple Metrics

Don't rely on loss alone. Track additional metrics:

```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    double total_loss = 0.0;
    double total_l2_norm = 0.0;
    double max_grad = 0.0;

    for (int batch = 0; batch < num_batches; batch++) {
        // ... training ...

        // Track loss
        float loss_val;
        TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_val, TOFU_FLOAT);
        total_loss += loss_val;

        // Track parameter norm
        total_l2_norm += compute_parameter_norm(opt);

        // Track max gradient
        max_grad = fmax(max_grad, compute_max_gradient(opt));
    }

    printf("Epoch %d: Loss=%.4f, Param_Norm=%.4f, Max_Grad=%.4f\n",
           epoch, total_loss / num_batches,
           total_l2_norm / num_batches, max_grad);
}
```

### Save Checkpoints

Periodically save model parameters during training:

```c
void save_parameters(tofu_optimizer *opt, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;

    for (int i = 0; i < opt->num_params; i++) {
        tofu_tensor *param = tofu_graph_get_value(opt->params[i]);
        fwrite(param->data, 1, param->len * sizeof(float), f);
    }

    fclose(f);
}

// Use during training
for (int epoch = 0; epoch < 100; epoch++) {
    // ... training loop ...

    // Save every 10 epochs
    if (epoch % 10 == 0) {
        char filename[256];
        snprintf(filename, sizeof(filename), "model_epoch_%d.bin", epoch);
        save_parameters(opt, filename);
    }
}
```

### Start Conservative

Begin with conservative hyperparameters and increase aggressiveness only if needed:

```c
// Conservative defaults
double learning_rate = 0.01;  // Not too high
double momentum = 0.9;         // Standard momentum

tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, learning_rate, momentum);
```

It's easier to increase the learning rate if training is too slow than to recover from instability caused by too-high rates.

### Test on Small Data First

Before training on the full dataset, verify your setup on a small subset:

```c
// Test with 10 batches first
int test_batches = 10;

for (int epoch = 0; epoch < 5; epoch++) {
    for (int batch = 0; batch < test_batches; batch++) {
        // ... training loop ...
    }
}

// If loss decreases on small data, scale to full dataset
```

This quickly reveals issues with the model, loss function, or optimizer configuration.

### Use Learning Rate Warmup for High Rates

When using aggressive learning rates (> 0.05), warm up gradually:

```c
double target_lr = 0.1;
int warmup_epochs = 5;

for (int epoch = 0; epoch < 100; epoch++) {
    if (epoch < warmup_epochs) {
        opt->learning_rate = target_lr * (epoch + 1) / warmup_epochs;
    } else {
        opt->learning_rate = target_lr;
    }

    // ... training loop ...
}
```

### Document Your Configuration

Keep track of hyperparameters that work:

```c
// Document successful configurations
printf("Configuration:\n");
printf("  Optimizer: SGD with Momentum\n");
printf("  Learning Rate: %.6f\n", opt->learning_rate);
printf("  Momentum: %.2f\n", 0.9);
printf("  Batch Size: %d\n", batch_size);
printf("  Schedule: Step decay by 0.1 every 30 epochs\n");
```

This helps when you need to replicate results or adjust for similar problems.

## Complete Example

Here's a complete training example that demonstrates all the concepts from this guide.

### Problem: Binary Classification

We'll train a two-layer neural network to classify binary data.

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tofu.h"

// Network architecture: input(10) -> hidden(20) -> output(1)

int main(void) {
    // Hyperparameters
    const int input_dim = 10;
    const int hidden_dim = 20;
    const int output_dim = 1;
    const int batch_size = 32;
    const int num_batches = 100;
    const int num_epochs = 50;
    const double learning_rate = 0.01;
    const double momentum = 0.9;

    // Create graph
    tofu_graph *g = tofu_graph_create();

    // Initialize parameters
    tofu_tensor *W1 = tofu_tensor_zeros(2, (int[]){input_dim, hidden_dim}, TOFU_FLOAT);
    tofu_tensor *b1 = tofu_tensor_zeros(1, (int[]){hidden_dim}, TOFU_FLOAT);
    tofu_tensor *W2 = tofu_tensor_zeros(2, (int[]){hidden_dim, output_dim}, TOFU_FLOAT);
    tofu_tensor *b2 = tofu_tensor_zeros(1, (int[]){output_dim}, TOFU_FLOAT);

    // Add parameters to graph
    tofu_graph_node *W1_node = tofu_graph_param(g, W1);
    tofu_graph_node *b1_node = tofu_graph_param(g, b1);
    tofu_graph_node *W2_node = tofu_graph_param(g, W2);
    tofu_graph_node *b2_node = tofu_graph_param(g, b2);

    // Create optimizer
    tofu_optimizer *opt = tofu_optimizer_sgd_momentum_create(g, learning_rate, momentum);

    printf("Training Configuration:\n");
    printf("  Architecture: %d -> %d -> %d\n", input_dim, hidden_dim, output_dim);
    printf("  Optimizer: SGD with Momentum\n");
    printf("  Learning Rate: %.4f\n", learning_rate);
    printf("  Momentum: %.2f\n", momentum);
    printf("  Batch Size: %d\n", batch_size);
    printf("  Epochs: %d\n\n", num_epochs);

    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double epoch_loss = 0.0;

        // Learning rate schedule: reduce by 0.1 every 20 epochs
        if (epoch % 20 == 0 && epoch > 0) {
            opt->learning_rate *= 0.1;
            printf("Reduced learning rate to %.6f\n", opt->learning_rate);
        }

        for (int batch = 0; batch < num_batches; batch++) {
            // 1. Zero gradients
            tofu_optimizer_zero_grad(opt);

            // 2. Generate synthetic batch data (normally loaded from dataset)
            tofu_tensor *batch_x = generate_batch_data(batch_size, input_dim);
            tofu_tensor *batch_y = generate_batch_labels(batch_size, output_dim);

            // 3. Forward pass
            tofu_graph_node *x = tofu_graph_input(g, batch_x);

            // Layer 1: h = relu(x @ W1 + b1)
            tofu_graph_node *h1 = tofu_graph_matmul(g, x, W1_node);
            h1 = tofu_graph_add(g, h1, b1_node);
            h1 = tofu_graph_relu(g, h1);

            // Layer 2: pred = h @ W2 + b2
            tofu_graph_node *pred = tofu_graph_matmul(g, h1, W2_node);
            pred = tofu_graph_add(g, pred, b2_node);

            // Loss
            tofu_graph_node *target = tofu_graph_input(g, batch_y);
            tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, target);

            // Track loss
            float loss_val;
            TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_val, TOFU_FLOAT);
            epoch_loss += loss_val;

            // 4. Backward pass
            tofu_graph_backward(g, loss);

            // 5. Update parameters
            tofu_optimizer_step(opt);

            // 6. Clear operations
            tofu_graph_clear_ops(g);

            // Free batch data
            tofu_tensor_free_data_too(batch_x);
            tofu_tensor_free_data_too(batch_y);
        }

        // Log progress
        double avg_loss = epoch_loss / num_batches;
        printf("Epoch %2d: Loss = %.6f\n", epoch, avg_loss);

        // Early stopping
        if (avg_loss < 0.001) {
            printf("Converged! Stopping early.\n");
            break;
        }
    }

    // Cleanup
    tofu_optimizer_free(opt);
    tofu_graph_free(g);
    tofu_tensor_free_data_too(W1);
    tofu_tensor_free_data_too(b1);
    tofu_tensor_free_data_too(W2);
    tofu_tensor_free_data_too(b2);

    printf("\nTraining complete!\n");

    return 0;
}

// Helper function to generate synthetic data (replace with real data loading)
tofu_tensor* generate_batch_data(int batch_size, int input_dim) {
    float *data = (float*)malloc(batch_size * input_dim * sizeof(float));
    for (int i = 0; i < batch_size * input_dim; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random in [-1, 1]
    }
    tofu_tensor *t = tofu_tensor_create_with_values(data, 2,
                                                     (int[]){batch_size, input_dim});
    free(data);
    return t;
}

tofu_tensor* generate_batch_labels(int batch_size, int output_dim) {
    float *data = (float*)malloc(batch_size * output_dim * sizeof(float));
    for (int i = 0; i < batch_size * output_dim; i++) {
        data[i] = ((float)rand() / RAND_MAX) > 0.5f ? 1.0f : 0.0f;
    }
    tofu_tensor *t = tofu_tensor_create_with_values(data, 2,
                                                     (int[]){batch_size, output_dim});
    free(data);
    return t;
}
```

This example demonstrates:

1. Creating a computation graph with parameters
2. Building a multi-layer network
3. Setting up an optimizer with momentum
4. Implementing a learning rate schedule
5. Proper training loop structure
6. Monitoring loss over time
7. Correct cleanup order

Adapt this template for your specific problem by:
- Changing network architecture (layer sizes, activations)
- Loading real data instead of synthetic batches
- Adding validation/test evaluation
- Saving model checkpoints
- Implementing gradient clipping if needed

## Summary

Optimizers are the engine of neural network training. They take computed gradients and update parameters to minimize loss. Here are the key takeaways:

**Core Concepts:**
- Gradient descent follows gradients downhill toward lower loss
- Learning rate controls step size (most important hyperparameter)
- Momentum accumulates velocity to accelerate convergence and dampen oscillations

**Choosing an Optimizer:**
- Start with SGD + momentum (learning_rate=0.01, momentum=0.9)
- Use vanilla SGD only for memory-constrained or very simple problems
- Deep networks benefit from higher momentum (0.95-0.99)

**Using Optimizers:**
- Always follow the pattern: zero_grad, forward, backward, step
- Clear operations between batches to manage memory
- Monitor training metrics (loss, gradients, parameter norms)

**Tuning:**
- Start with learning_rate=0.01
- Increase if training is too slow, decrease if unstable
- Use learning rate schedules for long training runs
- Implement gradient clipping for unstable gradients

**Troubleshooting:**
- NaN loss: Reduce learning rate, clip gradients
- No progress: Increase learning rate, add momentum
- Oscillations: Reduce learning rate, increase momentum
- Slow convergence: Use learning rate schedule, higher momentum

With these principles, you can train neural networks effectively and debug issues when they arise. Experiment with different configurations, monitor training carefully, and adjust based on what you observe. The best optimizer and hyperparameters depend on your specific problem, so be prepared to iterate.
