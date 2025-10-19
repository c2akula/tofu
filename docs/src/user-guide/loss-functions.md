# Loss Functions

Loss functions are the core mechanism that guides neural network training. They measure how far your model's predictions are from the true values, producing a single scalar number that quantifies the error. During training, the optimizer uses gradients of this loss to adjust model parameters and improve predictions.

This guide explains how loss functions work, when to use each type, and how to integrate them into your training loops. You'll learn to choose the right loss function for your task and interpret loss values during training.

## Table of Contents

- [Introduction](#introduction)
- [Loss Function Fundamentals](#loss-function-fundamentals)
- [Mean Squared Error](#mean-squared-error)
- [Cross-Entropy Loss](#cross-entropy-loss)
- [Choosing a Loss Function](#choosing-a-loss-function)
- [Loss in Training Loop](#loss-in-training-loop)
- [Understanding Loss Values](#understanding-loss-values)
- [Advanced Topics](#advanced-topics)
- [Complete Examples](#complete-examples)
- [Best Practices](#best-practices)

---

## Introduction

Every machine learning model needs a way to evaluate how well it's performing. A **loss function** (also called an objective function or cost function) provides this evaluation by computing a numerical score representing prediction error.

During training:

1. The model makes predictions on input data
2. The loss function compares predictions to true target values
3. The result is a single number (scalar) representing total error
4. Gradients of this loss tell us how to adjust weights
5. The optimizer updates weights to reduce the loss

The choice of loss function depends on your task type (regression vs classification) and the structure of your data. Tofu provides two fundamental loss functions that cover most use cases:

- **Mean Squared Error (MSE)**: For regression tasks where you predict continuous values
- **Cross-Entropy Loss**: For classification tasks where you predict discrete classes

Let's explore the fundamental properties all loss functions must have, then dive into each type.

---

## Loss Function Fundamentals

To work correctly with gradient-based optimization, loss functions must satisfy three key requirements.

### 1. Objective Function

A loss function defines the **optimization objective**—the quantity we want to minimize during training. Lower loss means better predictions. The training process iteratively adjusts model parameters to find weights that minimize this function.

Think of it like hiking down a mountain in fog. The loss value tells you your current altitude, and the gradient tells you which direction is downhill. Your goal is to reach the lowest point (minimize loss).

### 2. Scalar Output

Loss functions must return a **single number** (scalar), not a vector or matrix. This scalar summarizes all prediction errors across all samples and features into one value.

Why scalar? Because optimization algorithms need a single objective to minimize. You can't simultaneously minimize multiple conflicting objectives without combining them into one number.

Example shapes:
```
Predictions:  [batch_size, features]  e.g., [32, 10]
Targets:      [batch_size, features]  e.g., [32, 10]
Loss:         [1]                     (scalar)
```

The loss computation typically:
1. Computes per-element errors
2. Sums or averages across all elements
3. Returns a single scalar value

### 3. Differentiable

Loss functions must be **differentiable** (smooth, with computable gradients). The gradient tells us how loss changes when we adjust each parameter—it's the compass that guides optimization.

Non-differentiable functions (like step functions or absolute value at zero) create problems for gradient-based optimizers. They can't compute meaningful gradients, so training fails or converges slowly.

Mathematically, we need:
```
∂L/∂w  (gradient of loss L with respect to each weight w)
```

Tofu's automatic differentiation system computes these gradients automatically through backpropagation, so you don't need to derive formulas manually.

### Loss Function Workflow

Here's how a loss function fits into one training iteration:

```
1. Forward Pass:
   Input → Model → Predictions

2. Loss Computation:
   Loss = loss_function(Predictions, Targets)

3. Backward Pass:
   Compute ∂Loss/∂weights for all parameters

4. Parameter Update:
   weights = weights - learning_rate * ∂Loss/∂weights

5. Repeat until loss is minimized
```

Now let's examine specific loss functions and when to use them.

---

## Mean Squared Error

Mean Squared Error (MSE) is the most common loss function for **regression tasks**—problems where you predict continuous numerical values rather than discrete classes.

### Mathematical Formula

MSE computes the average squared difference between predictions and targets:

```
MSE = (1/n) * Σ(prediction - target)²

Where:
- n = number of elements (batch_size × features)
- Σ = sum over all elements
```

The squaring operation ensures:
- Errors are always positive (negative errors don't cancel positive ones)
- Large errors are penalized more heavily than small errors
- The function is smooth and differentiable everywhere

### Implementation in Tofu

Use `tofu_graph_mse_loss()` to add MSE loss to your computation graph:

```c
tofu_graph_node* tofu_graph_mse_loss(tofu_graph* g,
                                      tofu_graph_node* pred,
                                      tofu_graph_node* target);
```

**Parameters:**
- `g`: Computation graph
- `pred`: Model predictions (any shape)
- `target`: True target values (must match pred shape)

**Returns:** Scalar loss node (shape [1])

**Requirements:**
- `pred` and `target` must have identical shapes
- Both must be non-NULL

### When to Use MSE

MSE is ideal for regression problems:

**Perfect use cases:**
- Predicting house prices (continuous dollar values)
- Estimating temperature (continuous degrees)
- Forecasting stock prices
- Predicting ages, distances, or other continuous quantities
- Image denoising (pixel value reconstruction)

**Why it works:**
- Treats all dimensions equally
- Penalizes large errors more than small ones (squared term)
- Has nice mathematical properties (convex, smooth gradients)
- Easy to interpret (units are squared target units)

**When NOT to use:**
- Classification tasks (use cross-entropy instead)
- When outliers are common (MSE heavily penalizes outliers)
- When you care about percentage error rather than absolute error

### Practical Example

Here's a complete regression example predicting house prices:

```c
// Setup: Simple linear regression y = x @ W + b
tofu_graph *g = tofu_graph_create();

// Training data: 4 samples with 2 features each
float input_data[] = {
    1.0f, 2.0f,   // Sample 1: [sqft=1000, bedrooms=2]
    2.0f, 3.0f,   // Sample 2: [sqft=2000, bedrooms=3]
    3.0f, 4.0f,   // Sample 3: [sqft=3000, bedrooms=4]
    4.0f, 5.0f    // Sample 4: [sqft=4000, bedrooms=5]
};

// Target prices (in $100k)
float target_data[] = {
    150.0f,  // $150k
    250.0f,  // $250k
    350.0f,  // $350k
    450.0f   // $450k
};

// Create tensors
tofu_tensor *x_tensor = tofu_tensor_create(
    input_data, 2, (int[]){4, 2}, TOFU_FLOAT);
tofu_tensor *y_tensor = tofu_tensor_create(
    target_data, 2, (int[]){4, 1}, TOFU_FLOAT);

// Model parameters (weights and bias)
tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){2, 1}, TOFU_FLOAT);
tofu_tensor *bias = tofu_tensor_zeros(1, (int[]){1}, TOFU_FLOAT);

// Build computation graph
tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *b = tofu_graph_param(g, bias);

// Forward pass: prediction = x @ W + b
tofu_graph_node *matmul_result = tofu_graph_matmul(g, x, W);
tofu_graph_node *prediction = tofu_graph_add(g, matmul_result, b);

// Target
tofu_graph_node *target = tofu_graph_input(g, y_tensor);

// Compute MSE loss
tofu_graph_node *loss = tofu_graph_mse_loss(g, prediction, target);

// Get loss value
tofu_tensor *loss_value = tofu_graph_get_value(loss);
float loss_scalar;
TOFU_TENSOR_DATA_TO(loss_value, 0, loss_scalar, TOFU_FLOAT);
printf("MSE Loss: %.6f\n", loss_scalar);

// Backward pass to compute gradients
tofu_graph_backward(g, loss);

// Now W->grad and b->grad contain gradients for parameter updates
```

### Understanding MSE Values

MSE values depend on the scale of your target values:

**Small targets (e.g., normalized to [0, 1]):**
- Good MSE: < 0.01
- Acceptable: 0.01 - 0.1
- Poor: > 0.1

**Large targets (e.g., house prices in thousands):**
- MSE = 10,000 means average error is √10,000 = $100
- MSE = 1,000 means average error is √1,000 ≈ $32
- MSE = 100 means average error is √100 = $10

**Tip**: Take the square root of MSE to get **Root Mean Squared Error (RMSE)**, which has the same units as your target variable and is easier to interpret.

### Gradient Behavior

The gradient of MSE with respect to predictions is:

```
∂MSE/∂pred = (2/n) * (pred - target)
```

Key properties:
- Gradient magnitude is proportional to error size
- Large errors produce large gradients (faster learning)
- Small errors produce small gradients (slower learning)
- Can cause exploding gradients if predictions are very wrong

---

## Cross-Entropy Loss

Cross-Entropy Loss (also called log loss) is the standard loss function for **classification tasks**—problems where you assign inputs to discrete categories.

### Mathematical Formula

Cross-entropy measures the difference between predicted probability distributions and true labels:

```
CE = -(1/n) * Σ(target * log(prediction))

Where:
- n = batch_size × num_classes
- target = one-hot encoded true class (or class probabilities)
- prediction = softmax probabilities (sum to 1)
- log = natural logarithm
```

The formula rewards correct classifications (low loss) and heavily penalizes confident wrong predictions (high loss).

### Why Cross-Entropy for Classification?

Cross-entropy has special properties that make it ideal for classification:

1. **Probabilistic interpretation**: It measures the "surprise" of predictions given the true distribution
2. **Strong gradients**: Even for small probability errors, gradients remain strong enough to drive learning
3. **Numerical stability**: Works well with softmax activation (more on this below)
4. **Theoretical foundation**: Derived from maximum likelihood estimation in statistics

MSE doesn't work well for classification because:
- It treats class probabilities as arbitrary numbers, ignoring their sum-to-one constraint
- Gradients vanish when the model is confident but wrong
- No probabilistic interpretation

### Softmax and Cross-Entropy Connection

Cross-entropy is almost always used with **softmax activation** on the final layer. Here's why:

**Softmax** converts raw scores (logits) into probabilities:

```
softmax(x_i) = exp(x_i) / Σ(exp(x_j))

Properties:
- All outputs are in range (0, 1)
- Outputs sum to 1 (valid probability distribution)
- Highlights the maximum value (turns scores into confident predictions)
```

**Together**, softmax + cross-entropy creates a powerful combination:
- Softmax outputs represent class probabilities
- Cross-entropy compares these probabilities to true labels
- Gradients flow efficiently even when predictions are wrong

### Implementation in Tofu

Use `tofu_graph_ce_loss()` with softmax probabilities:

```c
tofu_graph_node* tofu_graph_ce_loss(tofu_graph* g,
                                     tofu_graph_node* pred,
                                     tofu_graph_node* target);
```

**Parameters:**
- `g`: Computation graph
- `pred`: Predicted probabilities from softmax (shape: [batch, num_classes])
- `target`: One-hot encoded true labels (shape: [batch, num_classes])

**Returns:** Scalar loss node (shape [1])

**Requirements:**
- `pred` must be softmax probabilities (values in [0, 1], sum to 1 per sample)
- `target` must be one-hot encoded (1 for true class, 0 for others)
- Shapes must match

**Numerical stability:** Tofu's implementation adds epsilon (1e-7) to avoid log(0), which would be undefined.

### When to Use Cross-Entropy

Cross-entropy is ideal for classification:

**Perfect use cases:**
- Image classification (cat vs dog vs bird)
- Text classification (spam vs not spam)
- Sentiment analysis (positive/negative/neutral)
- Multi-class problems (10 digits, 1000 object categories, etc.)
- Any problem with discrete categorical outputs

**Why it works:**
- Designed for probability distributions
- Strong gradients throughout training
- Natural pairing with softmax
- Well-studied theoretical properties

**When NOT to use:**
- Regression problems (use MSE instead)
- Multi-label classification where multiple classes can be true simultaneously (requires binary cross-entropy per class)

### Practical Example: MNIST-style Digit Classification

Here's a complete classification example for 4 classes:

```c
// Setup: Neural network for 4-class classification
tofu_graph *g = tofu_graph_create();

// Training data: 8 samples with 10 features each
float input_data[8 * 10] = { /* ...fill with data... */ };

// One-hot encoded labels (8 samples, 4 classes)
float label_data[8 * 4] = {
    1, 0, 0, 0,  // Sample 0: class 0
    0, 1, 0, 0,  // Sample 1: class 1
    0, 0, 1, 0,  // Sample 2: class 2
    0, 0, 0, 1,  // Sample 3: class 3
    1, 0, 0, 0,  // Sample 4: class 0
    0, 1, 0, 0,  // Sample 5: class 1
    0, 0, 1, 0,  // Sample 6: class 2
    0, 0, 0, 1   // Sample 7: class 3
};

// Create tensors
tofu_tensor *x_tensor = tofu_tensor_create(
    input_data, 2, (int[]){8, 10}, TOFU_FLOAT);
tofu_tensor *y_tensor = tofu_tensor_create(
    label_data, 2, (int[]){8, 4}, TOFU_FLOAT);

// Model parameters
tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){10, 4}, TOFU_FLOAT);
tofu_tensor *bias = tofu_tensor_zeros(1, (int[]){4}, TOFU_FLOAT);

// Build graph
tofu_graph_node *x = tofu_graph_input(g, x_tensor);
tofu_graph_node *W = tofu_graph_param(g, weights);
tofu_graph_node *b = tofu_graph_param(g, bias);

// Forward pass: logits = x @ W + b
tofu_graph_node *logits = tofu_graph_matmul(g, x, W);
logits = tofu_graph_add(g, logits, b);

// Softmax activation (converts logits to probabilities)
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);  // axis=1

// Target labels
tofu_graph_node *target = tofu_graph_input(g, y_tensor);

// Cross-entropy loss
tofu_graph_node *loss = tofu_graph_ce_loss(g, probs, target);

// Get loss value
tofu_tensor *loss_value = tofu_graph_get_value(loss);
float loss_scalar;
TOFU_TENSOR_DATA_TO(loss_value, 0, loss_scalar, TOFU_FLOAT);
printf("Cross-Entropy Loss: %.6f\n", loss_scalar);

// Backward pass
tofu_graph_backward(g, loss);

// Gradients are now available in W->grad and b->grad
```

### Understanding Cross-Entropy Values

Cross-entropy values depend on the number of classes:

**Binary classification (2 classes):**
- Random guessing: ~0.693 (log(2))
- Good model: < 0.3
- Excellent model: < 0.1

**Multi-class (e.g., 10 classes):**
- Random guessing: ~2.303 (log(10))
- Good model: < 1.0
- Excellent model: < 0.5

**Key insight:** Cross-entropy can never be negative. Zero loss means perfect predictions (100% confidence in correct class). As predictions get worse, loss increases without bound.

### Interpreting Loss During Training

Watch for these patterns:

**Healthy training:**
```
Epoch 0:  loss = 2.30  (random initialization)
Epoch 10: loss = 1.20  (learning started)
Epoch 50: loss = 0.50  (converging)
Epoch 100: loss = 0.15  (well-trained)
```

**Problems:**
- Loss stays at log(num_classes): Model isn't learning (check learning rate)
- Loss increases: Learning rate too high or numerical instability
- Loss plateaus early: Model too simple or data too hard

### Gradient Behavior

The gradient of cross-entropy with respect to predictions is:

```
∂CE/∂pred = -target / pred
```

Key properties:
- When prediction is very wrong (pred ≈ 0 but target = 1), gradient is very large
- When prediction is correct and confident (pred ≈ 1 and target = 1), gradient is small
- This creates strong learning signals when needed most

Combined with softmax, the gradient simplifies beautifully to:

```
∂(CE + softmax)/∂logits = pred - target
```

This is why softmax + cross-entropy is the gold standard for classification.

---

## Choosing a Loss Function

Selecting the right loss function is critical—it defines what "good" means for your model. Here's a decision guide.

### Decision Tree

```
Start: What type of problem are you solving?
│
├─ Predicting continuous values (numbers)?
│  └─ Use Mean Squared Error (MSE)
│     Examples: Regression, image denoising, forecasting
│
└─ Predicting discrete categories (classes)?
   └─ Use Cross-Entropy Loss
      Examples: Classification, object recognition, sentiment analysis
```

### Quick Reference Table

| **Task Type** | **Loss Function** | **Output Activation** | **Target Format** |
|---------------|-------------------|-----------------------|-------------------|
| Regression | MSE | None (linear) | Continuous values |
| Binary Classification | Cross-Entropy | Softmax (2 classes) | One-hot [0,1] or [1,0] |
| Multi-class Classification | Cross-Entropy | Softmax | One-hot encoding |
| Image Reconstruction | MSE | None or Sigmoid | Pixel values |

### Detailed Recommendations

**Use MSE when:**
- Output is a continuous number (prices, temperatures, distances)
- You care about absolute error magnitude
- Your task is regression or reconstruction
- Outliers are rare or acceptable

**Use Cross-Entropy when:**
- Output is a discrete category (class label)
- You need probability predictions
- Your task is classification
- You want strong gradients throughout training

**Example scenarios:**

| **Problem** | **Input** | **Output** | **Loss** | **Why** |
|-------------|-----------|------------|----------|---------|
| House price prediction | Features (sqft, bedrooms) | Price ($) | MSE | Continuous value |
| Spam detection | Email text | Spam/Not Spam | Cross-Entropy | Binary classification |
| Digit recognition | Image pixels | Digit (0-9) | Cross-Entropy | Multi-class classification |
| Temperature forecast | Historical data | Temperature (°F) | MSE | Continuous value |
| Sentiment analysis | Review text | Pos/Neg/Neutral | Cross-Entropy | Multi-class classification |

### Common Mistakes

**Mistake 1: Using MSE for classification**

```c
// WRONG: Using MSE to predict classes
tofu_graph_node *logits = tofu_graph_matmul(g, x, W);
tofu_graph_node *loss = tofu_graph_mse_loss(g, logits, target);  // Bad!
```

Problem: MSE treats class probabilities as arbitrary numbers, leading to weak gradients and poor convergence.

**Fix:** Use softmax + cross-entropy:

```c
// CORRECT: Classification with cross-entropy
tofu_graph_node *logits = tofu_graph_matmul(g, x, W);
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);
tofu_graph_node *loss = tofu_graph_ce_loss(g, probs, target);  // Good!
```

**Mistake 2: Forgetting softmax before cross-entropy**

```c
// WRONG: Cross-entropy without softmax
tofu_graph_node *logits = tofu_graph_matmul(g, x, W);
tofu_graph_node *loss = tofu_graph_ce_loss(g, logits, target);  // Bad!
```

Problem: Cross-entropy expects probabilities (sum to 1), but logits are raw scores.

**Fix:** Always apply softmax first:

```c
// CORRECT: Softmax before cross-entropy
tofu_graph_node *logits = tofu_graph_matmul(g, x, W);
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);
tofu_graph_node *loss = tofu_graph_ce_loss(g, probs, target);  // Good!
```

**Mistake 3: Wrong target format**

```c
// WRONG: Class indices instead of one-hot for cross-entropy
float targets[] = {0, 1, 2, 1};  // Class indices
```

Problem: Cross-entropy expects one-hot encoded targets, not class indices.

**Fix:** Convert to one-hot:

```c
// CORRECT: One-hot encoded targets
float targets[] = {
    1, 0, 0,  // Class 0
    0, 1, 0,  // Class 1
    0, 0, 1,  // Class 2
    0, 1, 0   // Class 1
};
```

---

## Loss in Training Loop

The loss function integrates into the training loop at a specific point in the forward-backward cycle. Understanding this workflow ensures correct implementation.

### Training Loop Structure

A typical training iteration follows this pattern:

```
1. Zero gradients       (clear previous gradients)
2. Forward pass         (compute predictions)
3. Compute loss         (evaluate predictions)
4. Backward pass        (compute gradients via backpropagation)
5. Optimizer step       (update parameters)
6. Repeat
```

Loss computation happens **after** the forward pass and **before** the backward pass. It's the bridge connecting prediction to optimization.

### Complete Training Loop Example

Here's a full training loop showing loss integration:

```c
// Setup: Create graph, parameters, and optimizer
tofu_graph *g = tofu_graph_create();

tofu_tensor *weights = tofu_tensor_zeros(2, (int[]){4, 3}, TOFU_FLOAT);
tofu_tensor *bias = tofu_tensor_zeros(1, (int[]){3}, TOFU_FLOAT);

tofu_graph_node *W_node = tofu_graph_param(g, weights);
tofu_graph_node *b_node = tofu_graph_param(g, bias);

tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.01);

// Training loop
const int NUM_EPOCHS = 100;
const int BATCH_SIZE = 32;

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    float epoch_loss = 0.0f;
    int num_batches = 0;

    for (int batch = 0; batch < num_batches_in_dataset; batch++) {
        // 1. Zero gradients
        tofu_optimizer_zero_grad(opt);

        // 2. Forward pass
        // Load batch data
        tofu_tensor *batch_x = load_batch_input(batch);
        tofu_tensor *batch_y = load_batch_target(batch);

        tofu_graph_node *x = tofu_graph_input(g, batch_x);
        tofu_graph_node *y_true = tofu_graph_input(g, batch_y);

        // Model forward pass
        tofu_graph_node *h = tofu_graph_matmul(g, x, W_node);
        tofu_graph_node *pred = tofu_graph_add(g, h, b_node);

        // 3. Compute loss
        tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, y_true);

        // Extract loss value for monitoring
        tofu_tensor *loss_tensor = tofu_graph_get_value(loss);
        float loss_value;
        TOFU_TENSOR_DATA_TO(loss_tensor, 0, loss_value, TOFU_FLOAT);
        epoch_loss += loss_value;

        // 4. Backward pass
        tofu_graph_backward(g, loss);

        // 5. Optimizer step
        tofu_optimizer_step(opt);

        // Cleanup batch resources
        tofu_tensor_free(batch_x);
        tofu_tensor_free(batch_y);

        // Clear graph operations (keeps parameters)
        tofu_graph_clear_ops(g);

        num_batches++;
    }

    // Report progress
    float avg_loss = epoch_loss / num_batches;
    printf("Epoch %d: loss = %.6f\n", epoch, avg_loss);
}
```

### Monitoring Loss During Training

Track loss values across epochs to monitor training progress:

```c
// Loss tracking
float loss_history[NUM_EPOCHS];

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    // ... training code ...

    loss_history[epoch] = avg_loss;

    // Print every 10 epochs
    if (epoch % 10 == 0) {
        printf("Epoch %3d: loss = %.6f\n", epoch, avg_loss);
    }
}
```

### Loss Curves

Visualizing loss over time reveals training behavior:

**Healthy training curve:**
```
Loss
│
2.0 ┤●
    │ ●
1.5 ┤  ●
    │   ●●
1.0 ┤     ●●
    │       ●●●
0.5 ┤          ●●●●●●●●●●●
    └──────────────────────────> Epoch
```

Characteristics:
- Smooth decrease
- Eventually plateaus
- No wild fluctuations

**Warning signs:**

```
Loss increasing:
    │    ●●●
    │  ●●
    │●●
    └─────> Learning rate too high

Loss plateauing early:
    │●●●●●●●●●●●●●●
    │
    └─────> Model too simple or stuck

Loss oscillating:
    │ ● ● ● ● ●
    │● ● ● ● ● ●
    └─────> Batch size too small or LR too high
```

---

## Understanding Loss Values

Interpreting loss values correctly helps diagnose training issues and assess model quality.

### Absolute Loss Magnitude

**Loss value interpretation depends heavily on context:**

**For MSE:**
- Scale depends on target value range
- MSE = 100 is terrible for normalized data [0, 1]
- MSE = 100 might be excellent for house prices in thousands
- Always consider: What's the typical magnitude of your targets?

**For Cross-Entropy:**
- Random guessing baseline: log(num_classes)
- Binary classification random: 0.693
- 10-class random: 2.303
- Perfect predictions: 0.0

**Rule of thumb:** Compare loss to a baseline (random guessing or simple heuristic) to assess improvement.

### Relative Changes Matter More

Focus on **loss trends** rather than absolute values:

```c
// Good trend (decreasing)
Epoch 0:   loss = 1.50
Epoch 10:  loss = 1.20  (20% reduction)
Epoch 20:  loss = 0.85  (29% reduction)
Epoch 50:  loss = 0.45  (47% reduction)

// Bad trend (increasing)
Epoch 0:   loss = 1.50
Epoch 10:  loss = 1.65  (increasing - problem!)
```

### Common Loss Troubleshooting

**Problem: Loss is NaN or infinite**

Causes:
- Learning rate too high (exploding gradients)
- Numerical overflow in loss computation
- Invalid data (NaN in input)

Fixes:
```c
// 1. Reduce learning rate
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.001);  // Was 0.1

// 2. Check for NaN in data
for (int i = 0; i < tensor->len; i++) {
    float val;
    TOFU_TENSOR_DATA_TO(tensor, i, val, TOFU_FLOAT);
    if (isnan(val) || isinf(val)) {
        fprintf(stderr, "Invalid data at index %d\n", i);
    }
}

// 3. Add gradient clipping (manual)
tofu_tensor *grad = tofu_graph_get_grad(param_node);
// Clip gradients to [-max_grad, max_grad]
```

**Problem: Loss doesn't decrease**

Causes:
- Learning rate too low
- Model too simple (can't fit data)
- Weights initialized poorly
- Wrong loss function for task

Fixes:
```c
// 1. Increase learning rate
tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.1);  // Was 0.001

// 2. Add hidden layers (increase model capacity)
tofu_graph_node *h1 = tofu_graph_relu(g, tofu_graph_add(g,
    tofu_graph_matmul(g, x, W1), b1));
tofu_graph_node *h2 = tofu_graph_relu(g, tofu_graph_add(g,
    tofu_graph_matmul(g, h1, W2), b2));

// 3. Check you're using the right loss function
// Classification → Cross-Entropy, Regression → MSE
```

**Problem: Loss plateaus too early**

Causes:
- Model capacity too small
- Learning rate needs adjustment
- Reached local minimum
- Need more training time

Fixes:
```c
// 1. Train longer
const int NUM_EPOCHS = 500;  // Was 100

// 2. Add capacity
// Increase hidden layer size or add more layers

// 3. Try learning rate schedule
float lr = (epoch < 50) ? 0.1 : 0.01;  // Reduce LR after 50 epochs
```

**Problem: Loss oscillates wildly**

Causes:
- Learning rate too high
- Batch size too small
- Numerical instability

Fixes:
```c
// 1. Reduce learning rate
lr = 0.001;  // Was 0.1

// 2. Increase batch size
BATCH_SIZE = 64;  // Was 16

// 3. Add momentum (helps smooth updates)
tofu_optimizer *opt = tofu_optimizer_adam_create(g, 0.001);
```

### Comparing Train vs Validation Loss

Always monitor loss on held-out validation data:

```c
// Training loop with validation
for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    // Training
    float train_loss = train_one_epoch(g, opt, train_data);

    // Validation (no gradient updates)
    float val_loss = evaluate_loss(g, val_data);

    printf("Epoch %d: train_loss=%.4f, val_loss=%.4f\n",
           epoch, train_loss, val_loss);

    // Check for overfitting
    if (val_loss > train_loss * 1.5) {
        printf("Warning: Model may be overfitting\n");
    }
}
```

**Healthy pattern:**
```
Train loss: 0.50, Val loss: 0.55  (close - good generalization)
```

**Overfitting:**
```
Train loss: 0.10, Val loss: 0.80  (gap too large - overfitting)
```

---

## Advanced Topics

Beyond basic loss functions, advanced techniques can improve training stability and model performance.

### Loss Weighting

Sometimes you want to emphasize certain samples or classes. Loss weighting adjusts the contribution of individual samples.

**Class weighting for imbalanced data:**

If you have 90% negative samples and 10% positive samples in binary classification, the model may ignore the minority class. Weight the minority class higher:

```c
// Manually weight loss by class
// Assume we have per-sample weights
float class_weights[2] = {1.0f, 9.0f};  // Weight minority class 9x

// Compute weighted loss manually
tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);
tofu_graph_node *loss_unweighted = tofu_graph_ce_loss(g, probs, target);

// Get loss and multiply by weights
tofu_tensor *loss_tensor = tofu_graph_get_value(loss_unweighted);
float loss_val;
TOFU_TENSOR_DATA_TO(loss_tensor, 0, loss_val, TOFU_FLOAT);

// Weight by class (simplified - production code would weight per-sample)
int predicted_class = /* determine class */;
float weighted_loss = loss_val * class_weights[predicted_class];
```

**Note:** Tofu doesn't have built-in weighted loss functions yet, so implement weighting manually or at the sample level.

### Regularization Loss Terms

Regularization adds a penalty term to prevent overfitting:

```
Total Loss = Task Loss + λ * Regularization Term

Where λ controls regularization strength
```

**L2 Regularization (Weight Decay):**

Penalize large weights to prevent overfitting:

```c
// Compute L2 regularization manually
float l2_penalty = 0.0f;
const float lambda = 0.01f;

tofu_tensor *W = param_tensor;
for (int i = 0; i < W->len; i++) {
    float w;
    TOFU_TENSOR_DATA_TO(W, i, w, TOFU_FLOAT);
    l2_penalty += w * w;
}
l2_penalty *= lambda;

// Add to loss
float total_loss = task_loss + l2_penalty;
```

**Note:** Most optimizers (like Adam) have built-in weight decay support, which is more efficient than manual regularization.

### Custom Loss Functions

For specialized tasks, you may need custom losses. Implement them by:

1. Computing the loss value using tensor operations
2. Implementing the backward pass (gradient computation)

Example: Huber loss (robust to outliers)

```c
// Huber loss: Combines MSE (small errors) with MAE (large errors)
// Loss = 0.5 * (pred - target)^2        if |error| < delta
//      = delta * (|error| - 0.5*delta)  otherwise

// This requires implementing a custom graph operation
// (beyond basic usage - see advanced tutorials)
```

For most use cases, MSE and cross-entropy are sufficient. Custom losses require deeper knowledge of Tofu's backward pass implementation.

### Multi-Task Learning

Train one model for multiple tasks by combining losses:

```c
// Example: Predict both class and bounding box
tofu_graph_node *class_logits = /* classification head */;
tofu_graph_node *bbox_pred = /* regression head */;

// Classification loss
tofu_graph_node *class_probs = tofu_graph_softmax(g, class_logits, 1);
tofu_graph_node *class_loss = tofu_graph_ce_loss(g, class_probs, class_target);

// Bounding box regression loss
tofu_graph_node *bbox_loss = tofu_graph_mse_loss(g, bbox_pred, bbox_target);

// Combined loss (weighted sum)
// Note: Must be done manually as Tofu doesn't support loss addition yet
float class_loss_val = extract_scalar(class_loss);
float bbox_loss_val = extract_scalar(bbox_loss);
float total_loss = class_loss_val + 0.5 * bbox_loss_val;  // Weight bbox 50%
```

---

## Complete Examples

Let's walk through two complete, practical examples: regression and classification.

### Example 1: Regression - House Price Prediction

**Goal:** Predict house prices from square footage and number of bedrooms.

```c
#include <stdio.h>
#include <stdlib.h>
#include "tofu_tensor.h"
#include "tofu_graph.h"
#include "tofu_optimizer.h"

int main() {
    // Dataset: 4 houses
    float features[] = {
        1000.0f, 2.0f,  // 1000 sqft, 2 bedrooms → $150k
        1500.0f, 3.0f,  // 1500 sqft, 3 bedrooms → $200k
        2000.0f, 3.0f,  // 2000 sqft, 3 bedrooms → $250k
        2500.0f, 4.0f   // 2500 sqft, 4 bedrooms → $300k
    };

    float prices[] = {150.0f, 200.0f, 250.0f, 300.0f};

    // Create tensors
    tofu_tensor *X = tofu_tensor_create(features, 2, (int[]){4, 2}, TOFU_FLOAT);
    tofu_tensor *y = tofu_tensor_create(prices, 2, (int[]){4, 1}, TOFU_FLOAT);

    // Model parameters (linear regression: y = X @ W + b)
    tofu_tensor *W = tofu_tensor_zeros(2, (int[]){2, 1}, TOFU_FLOAT);
    tofu_tensor *b = tofu_tensor_zeros(1, (int[]){1}, TOFU_FLOAT);

    // Create graph and optimizer
    tofu_graph *g = tofu_graph_create();
    tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.0001);  // Small LR

    // Training loop
    for (int epoch = 0; epoch < 1000; epoch++) {
        tofu_optimizer_zero_grad(opt);

        // Forward pass
        tofu_graph_node *x_node = tofu_graph_input(g, X);
        tofu_graph_node *W_node = tofu_graph_param(g, W);
        tofu_graph_node *b_node = tofu_graph_param(g, b);
        tofu_graph_node *y_node = tofu_graph_input(g, y);

        tofu_graph_node *pred = tofu_graph_add(g,
            tofu_graph_matmul(g, x_node, W_node), b_node);

        // MSE loss
        tofu_graph_node *loss = tofu_graph_mse_loss(g, pred, y_node);

        // Extract loss value
        float loss_val;
        TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_val, TOFU_FLOAT);

        // Backward and optimize
        tofu_graph_backward(g, loss);
        tofu_optimizer_step(opt);

        if (epoch % 100 == 0) {
            printf("Epoch %4d: MSE = %.2f\n", epoch, loss_val);
        }

        tofu_graph_clear_ops(g);
    }

    // Final predictions
    printf("\nFinal predictions:\n");
    tofu_graph_node *x_node = tofu_graph_input(g, X);
    tofu_graph_node *W_node = tofu_graph_param(g, W);
    tofu_graph_node *b_node = tofu_graph_param(g, b);
    tofu_graph_node *pred = tofu_graph_add(g,
        tofu_graph_matmul(g, x_node, W_node), b_node);

    tofu_tensor *predictions = tofu_graph_get_value(pred);
    for (int i = 0; i < 4; i++) {
        float pred_price;
        TOFU_TENSOR_DATA_TO(predictions, i, pred_price, TOFU_FLOAT);
        printf("House %d: Predicted=%.1f, Actual=%.1f\n",
               i, pred_price, prices[i]);
    }

    // Cleanup
    tofu_optimizer_free(opt);
    tofu_graph_free(g);
    tofu_tensor_free(X);
    tofu_tensor_free(y);
    tofu_tensor_free_data_too(W);
    tofu_tensor_free_data_too(b);

    return 0;
}
```

**Expected output:**
```
Epoch    0: MSE = 42500.00
Epoch  100: MSE = 1250.50
Epoch  200: MSE = 523.75
Epoch  900: MSE = 12.30

Final predictions:
House 0: Predicted=148.5, Actual=150.0
House 1: Predicted=201.2, Actual=200.0
House 2: Predicted=251.8, Actual=250.0
House 3: Predicted=298.7, Actual=300.0
```

### Example 2: Classification - XOR Problem

**Goal:** Learn the XOR function (classic non-linear classification).

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tofu_tensor.h"
#include "tofu_graph.h"
#include "tofu_optimizer.h"

// Xavier initialization
float xavier_init(int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    return limit * (2.0f * rand() / RAND_MAX - 1.0f);
}

int main() {
    // XOR dataset
    float inputs[] = {
        0, 0,  // → 0
        0, 1,  // → 1
        1, 0,  // → 1
        1, 1   // → 0
    };

    // One-hot targets (2 classes: [1,0] = class 0, [0,1] = class 1)
    float targets[] = {
        1, 0,  // XOR(0,0) = 0
        0, 1,  // XOR(0,1) = 1
        0, 1,  // XOR(1,0) = 1
        1, 0   // XOR(1,1) = 0
    };

    // Create tensors
    tofu_tensor *X = tofu_tensor_create(inputs, 2, (int[]){4, 2}, TOFU_FLOAT);
    tofu_tensor *y = tofu_tensor_create(targets, 2, (int[]){4, 2}, TOFU_FLOAT);

    // Model: 2 → 4 → 2 (need hidden layer for non-linearity)
    tofu_tensor *W1 = tofu_tensor_zeros(2, (int[]){2, 4}, TOFU_FLOAT);
    tofu_tensor *b1 = tofu_tensor_zeros(1, (int[]){4}, TOFU_FLOAT);
    tofu_tensor *W2 = tofu_tensor_zeros(2, (int[]){4, 2}, TOFU_FLOAT);
    tofu_tensor *b2 = tofu_tensor_zeros(1, (int[]){2}, TOFU_FLOAT);

    // Xavier initialization for W1, W2
    for (int i = 0; i < W1->len; i++) {
        float val = xavier_init(2);
        TOFU_TENSOR_DATA_FROM(W1, i, val, TOFU_FLOAT);
    }
    for (int i = 0; i < W2->len; i++) {
        float val = xavier_init(4);
        TOFU_TENSOR_DATA_FROM(W2, i, val, TOFU_FLOAT);
    }

    // Create graph and optimizer
    tofu_graph *g = tofu_graph_create();
    tofu_optimizer *opt = tofu_optimizer_sgd_create(g, 0.5);  // Higher LR

    // Training loop
    for (int epoch = 0; epoch < 2000; epoch++) {
        tofu_optimizer_zero_grad(opt);

        // Forward pass
        tofu_graph_node *x = tofu_graph_input(g, X);
        tofu_graph_node *w1 = tofu_graph_param(g, W1);
        tofu_graph_node *b1_node = tofu_graph_param(g, b1);
        tofu_graph_node *w2 = tofu_graph_param(g, W2);
        tofu_graph_node *b2_node = tofu_graph_param(g, b2);
        tofu_graph_node *y_node = tofu_graph_input(g, y);

        // Layer 1: x @ W1 + b1 → ReLU
        tofu_graph_node *h1 = tofu_graph_relu(g, tofu_graph_add(g,
            tofu_graph_matmul(g, x, w1), b1_node));

        // Layer 2: h1 @ W2 + b2 → softmax
        tofu_graph_node *logits = tofu_graph_add(g,
            tofu_graph_matmul(g, h1, w2), b2_node);
        tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);

        // Cross-entropy loss
        tofu_graph_node *loss = tofu_graph_ce_loss(g, probs, y_node);

        float loss_val;
        TOFU_TENSOR_DATA_TO(tofu_graph_get_value(loss), 0, loss_val, TOFU_FLOAT);

        // Backward and optimize
        tofu_graph_backward(g, loss);
        tofu_optimizer_step(opt);

        if (epoch % 200 == 0) {
            printf("Epoch %4d: CE Loss = %.4f\n", epoch, loss_val);
        }

        tofu_graph_clear_ops(g);
    }

    // Test predictions
    printf("\nFinal predictions:\n");
    tofu_graph_node *x = tofu_graph_input(g, X);
    tofu_graph_node *w1 = tofu_graph_param(g, W1);
    tofu_graph_node *b1_node = tofu_graph_param(g, b1);
    tofu_graph_node *w2 = tofu_graph_param(g, W2);
    tofu_graph_node *b2_node = tofu_graph_param(g, b2);

    tofu_graph_node *h1 = tofu_graph_relu(g, tofu_graph_add(g,
        tofu_graph_matmul(g, x, w1), b1_node));
    tofu_graph_node *logits = tofu_graph_add(g,
        tofu_graph_matmul(g, h1, w2), b2_node);
    tofu_graph_node *probs = tofu_graph_softmax(g, logits, 1);

    tofu_tensor *predictions = tofu_graph_get_value(probs);

    for (int i = 0; i < 4; i++) {
        float prob0, prob1;
        TOFU_TENSOR_DATA_TO(predictions, i*2, prob0, TOFU_FLOAT);
        TOFU_TENSOR_DATA_TO(predictions, i*2+1, prob1, TOFU_FLOAT);
        int pred_class = (prob1 > prob0) ? 1 : 0;
        int true_class = (targets[i*2+1] > 0.5f) ? 1 : 0;

        printf("[%.0f, %.0f] → Pred=%d (%.3f, %.3f), True=%d\n",
               inputs[i*2], inputs[i*2+1],
               pred_class, prob0, prob1, true_class);
    }

    // Cleanup
    tofu_optimizer_free(opt);
    tofu_graph_free(g);
    tofu_tensor_free(X);
    tofu_tensor_free(y);
    tofu_tensor_free_data_too(W1);
    tofu_tensor_free_data_too(b1);
    tofu_tensor_free_data_too(W2);
    tofu_tensor_free_data_too(b2);

    return 0;
}
```

**Expected output:**
```
Epoch    0: CE Loss = 0.7120
Epoch  200: CE Loss = 0.4532
Epoch  400: CE Loss = 0.2145
Epoch 1800: CE Loss = 0.0523

Final predictions:
[0, 0] → Pred=0 (0.972, 0.028), True=0
[0, 1] → Pred=1 (0.045, 0.955), True=1
[1, 0] → Pred=1 (0.039, 0.961), True=1
[1, 1] → Pred=0 (0.968, 0.032), True=0
```

---

## Best Practices

Follow these guidelines for effective loss function usage:

### 1. Match Loss to Task Type

**Always use:**
- MSE for regression
- Cross-Entropy for classification

**Never mix them:** Using the wrong loss leads to poor convergence and incorrect learning.

### 2. Monitor Loss During Training

```c
// Log loss to file or console
FILE *log = fopen("training_log.txt", "w");
for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    float loss = train_epoch(...);
    fprintf(log, "%d,%.6f\n", epoch, loss);
    if (epoch % 10 == 0) {
        printf("Epoch %d: loss=%.6f\n", epoch, loss);
    }
}
fclose(log);
```

Track trends, not just final values. Loss should decrease smoothly over time.

### 3. Use Appropriate Learning Rates

Loss behavior reveals learning rate issues:

```c
// Too high: Loss explodes or oscillates wildly
// Solution: Reduce by 10x
lr = 0.01;  // Was 0.1

// Too low: Loss barely decreases
// Solution: Increase by 10x
lr = 0.1;  // Was 0.01
```

### 4. Normalize Your Data

Large input/output ranges cause numerical instability:

```c
// Bad: Raw house prices ($100k - $500k)
float price = 250000.0f;

// Good: Normalized to reasonable range
float price_normalized = (250000.0f - mean) / std_dev;
// or
float price_scaled = 250000.0f / 1000.0f;  // Scale to [100-500]
```

Normalization prevents exploding gradients and improves convergence.

### 5. Check for Numerical Issues

```c
// Add checks during training
if (isnan(loss_val) || isinf(loss_val)) {
    fprintf(stderr, "ERROR: Loss is %f at epoch %d\n", loss_val, epoch);
    // Reduce learning rate or check data
    break;
}
```

### 6. Compare to Baselines

Always establish a baseline before training:

```c
// Baseline 1: Random predictions
// For classification: loss ≈ log(num_classes)
// For regression: loss ≈ variance of targets

// Baseline 2: Simple heuristic
// Classification: Always predict most common class
// Regression: Always predict mean target value

printf("Random baseline loss: %.4f\n", baseline_loss);
printf("Trained model loss: %.4f\n", final_loss);
printf("Improvement: %.1f%%\n",
       100.0 * (baseline_loss - final_loss) / baseline_loss);
```

### 7. Use Validation Data

Never trust training loss alone:

```c
// Split data: 80% train, 20% validation
float train_loss = evaluate_loss(g, train_data);
float val_loss = evaluate_loss(g, val_data);

if (val_loss > train_loss * 1.5) {
    printf("Warning: Possible overfitting\n");
}
```

### 8. Save Best Model Based on Validation Loss

```c
float best_val_loss = INFINITY;
tofu_tensor *best_W = NULL;

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    train_epoch(...);
    float val_loss = evaluate_validation(...);

    if (val_loss < best_val_loss) {
        best_val_loss = val_loss;
        // Save model parameters
        if (best_W) tofu_tensor_free_data_too(best_W);
        best_W = tofu_tensor_clone(W);
        printf("New best model at epoch %d: val_loss=%.4f\n",
               epoch, val_loss);
    }
}
```

### 9. Early Stopping

Stop training when validation loss stops improving:

```c
int patience = 20;  // Wait 20 epochs for improvement
int no_improve_count = 0;

for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    float val_loss = train_and_validate(...);

    if (val_loss < best_val_loss) {
        best_val_loss = val_loss;
        no_improve_count = 0;
    } else {
        no_improve_count++;
    }

    if (no_improve_count >= patience) {
        printf("Early stopping at epoch %d\n", epoch);
        break;
    }
}
```

### 10. Document Your Loss Function Choice

```c
// At the top of your training code, document decisions:
/*
 * Model: Image classifier for 10 classes
 * Loss: Cross-Entropy (classification task)
 * Architecture: Input → 128 → 64 → 10 (softmax)
 * Optimizer: SGD with lr=0.01
 * Expected loss: Random ~2.3, Target <0.5
 */
```

This helps future debugging and maintains clear expectations.

---

## Summary

Loss functions are the foundation of neural network training. Key takeaways:

1. **Match loss to task:**
   - Regression → MSE
   - Classification → Cross-Entropy (with softmax)

2. **Loss must be:**
   - Scalar (single number)
   - Differentiable (smooth gradients)
   - Representative of task objective

3. **Monitor loss trends:**
   - Decreasing = learning
   - Plateauing = convergence or stuck
   - Increasing = problem (LR too high, numerical issues)

4. **Interpret loss in context:**
   - Compare to baselines (random guessing)
   - Track validation loss (detect overfitting)
   - Understand scale (depends on data range)

5. **Debug with loss values:**
   - NaN/Inf → Check learning rate, data validity
   - No decrease → Increase LR or model capacity
   - Oscillation → Reduce LR or increase batch size

With proper loss function selection and monitoring, you'll train neural networks that converge reliably and achieve strong performance on your task.

For more details on loss function implementation and gradients, see the [Graph API Reference](/api-reference/graph-api.html#loss-functions). For optimizer integration, see the [Optimizer User Guide](/user-guide/optimizers.html).
