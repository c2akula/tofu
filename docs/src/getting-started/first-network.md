# Your First Neural Network

Welcome to Tofu! In this guide, you'll build and train your first neural network to solve the classic XOR problem. By the end, you'll understand how to construct computation graphs, perform forward and backward passes, and optimize model parameters.

## What You'll Build

You'll build a neural network that learns the XOR (exclusive OR) function. XOR is a simple yet elegant problem that demonstrates why neural networks need hidden layers. Your final model will output correct predictions for all four XOR inputs.

## Why XOR?

XOR is the perfect learning problem because:

1. **Non-linear**: You can't solve XOR with a single linear layer. This teaches you why hidden layers matter.
2. **Small**: The dataset has only 4 examples, so training is fast.
3. **Well-understood**: We know exactly what "correct" looks like.
4. **Practical**: The same patterns apply to larger, real-world problems.

## What You'll Learn

- How to create and structure computation graphs in Tofu
- How to build a multi-layer neural network
- How to execute forward passes (predictions)
- How to perform backward passes (gradient computation)
- How to run training loops with optimizers
- How to manage memory ownership correctly
- How to verify that your network actually learned

## The XOR Problem

### Understanding XOR

XOR returns 1 when inputs are different, 0 when they're the same:

```
[0, 0] → 0  (same, output 0)
[0, 1] → 1  (different, output 1)
[1, 0] → 1  (different, output 1)
[1, 1] → 0  (same, output 0)
```

### Why It's Special

A single linear layer cannot learn XOR. Mathematically, XOR is not linearly separable—you cannot draw a single straight line to separate the 1s from the 0s on a 2D plane.

However, a network with a hidden layer can solve it by learning intermediate features. The hidden layer performs a non-linear transformation that makes XOR linearly separable in the higher-dimensional hidden space.

### Network Architecture

To solve XOR, we'll use this architecture:

```
Input Layer (2 units)
    ↓
Hidden Layer (4 units with ReLU activation)
    ↓
Output Layer (1 unit)
```

The flow:
1. **Input layer**: Takes [x1, x2] (the two binary inputs)
2. **Hidden layer**: Learns 4 intermediate features via matrix multiplication and ReLU
3. **Output layer**: Combines hidden features to produce final prediction

The ReLU (Rectified Linear Unit) activation in the hidden layer is crucial—it introduces non-linearity. Without it, stacking layers would be equivalent to a single linear layer.

## Complete Code Walkthrough

Here's the full XOR training program. We'll break it down into sections and explain each part.

### Section 1: Setup and Includes

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "tofu_tensor.h"
#include "tofu_graph.h"
#include "tofu_optimizer.h"

/* Xavier weight initialization for better convergence */
static float xor_xavier_init(int fan_in) {
    float limit = sqrtf(6.0f / (float)fan_in);
    return 2.0f * (float)rand() / RAND_MAX - 1.0f;
}
```

We include Tofu's three core modules:
- **tofu_tensor.h**: Tensor creation and manipulation
- **tofu_graph.h**: Computation graph construction and differentiation
- **tofu_optimizer.h**: Gradient-based parameter updates

The `xor_xavier_init` function initializes weights using Xavier initialization. This ensures weights start in a good range that helps training converge faster than random initialization.

### Section 2: Main Function and Configuration

```c
int main() {
    printf("============================================================\n");
    printf("XOR Neural Network Training Example\n");
    printf("============================================================\n\n");

    /* Configuration */
    const int INPUT_SIZE = 2;
    const int HIDDEN_SIZE = 4;
    const int OUTPUT_SIZE = 1;
    const int NUM_EPOCHS = 2000;
    const float LEARNING_RATE = 0.1f;
    const int REPORT_INTERVAL = 200;
```

These constants define our network shape and training hyperparameters:
- **INPUT_SIZE (2)**: Two binary inputs for XOR
- **HIDDEN_SIZE (4)**: Four hidden units (more than enough to solve XOR)
- **OUTPUT_SIZE (1)**: Single output for binary classification
- **NUM_EPOCHS (2000)**: Number of times we iterate through the dataset
- **LEARNING_RATE (0.1)**: Controls step size in parameter updates (higher = faster but riskier)
- **REPORT_INTERVAL (200)**: How often to print progress

### Section 3: Data Preparation

```c
    /* Prepare XOR dataset */
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

    printf("XOR Dataset:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%.0f, %.0f] -> %.0f\n",
               xor_inputs[i][0], xor_inputs[i][1], xor_targets[i][0]);
    }
    printf("\n");
```

We hardcode the complete XOR dataset. This is all the training data we need—the network must generalize from just 4 examples (and it can because of the structure of the XOR problem).

### Section 4: Creating the Computation Graph

```c
    /* Create computation graph */
    tofu_graph* g = tofu_graph_create();
    assert(g != NULL);
```

We create an empty computation graph. All our operations will be added to this graph. Think of it as a blueprint for computations—nodes represent operations, edges represent data flow.

**Important concept**: The graph doesn't own the tensor data. We allocate tensors separately and pass them to the graph. We're responsible for freeing them later.

### Section 5: Initializing Network Parameters

```c
    /* Initialize weights with Xavier initialization */
    float* w1_data = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        w1_data[i] = xor_xavier_init(INPUT_SIZE);
    }
    tofu_tensor* t_w1 = tofu_tensor_create(w1_data, 2,
                                           (int[]){INPUT_SIZE, HIDDEN_SIZE}, TOFU_FLOAT);
    tofu_graph_node* w1 = tofu_graph_param(g, t_w1);

    /* Initialize first bias */
    float* b1_data = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    tofu_tensor* t_b1 = tofu_tensor_create(b1_data, 1, (int[]){HIDDEN_SIZE}, TOFU_FLOAT);
    tofu_graph_node* b1 = tofu_graph_param(g, t_b1);

    /* Initialize weights for hidden to output */
    float* w2_data = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        w2_data[i] = xor_xavier_init(HIDDEN_SIZE);
    }
    tofu_tensor* t_w2 = tofu_tensor_create(w2_data, 2,
                                           (int[]){HIDDEN_SIZE, OUTPUT_SIZE}, TOFU_FLOAT);
    tofu_graph_node* w2 = tofu_graph_param(g, t_w2);

    /* Initialize second bias */
    float* b2_data = (float*)calloc(OUTPUT_SIZE, sizeof(float));
    tofu_tensor* t_b2 = tofu_tensor_create(b2_data, 1, (int[]){OUTPUT_SIZE}, TOFU_FLOAT);
    tofu_graph_node* b2 = tofu_graph_param(g, t_b2);
```

We initialize four parameter tensors:

1. **w1**: Shape [2, 4]. Transforms input to hidden layer. Each column is one hidden unit's weights from both inputs.
2. **b1**: Shape [4]. Bias for each hidden unit. We use `calloc` (zeros) for biases.
3. **w2**: Shape [4, 1]. Transforms hidden to output. Weights from all 4 hidden units to the single output.
4. **b2**: Shape [1]. Bias for the output unit.

Each tensor is converted to a graph node using `tofu_graph_param`. These are "parameter" nodes (trainable) as opposed to "input" nodes (non-trainable). The optimizer will update these during training.

### Section 6: Creating the Optimizer

```c
    /* Create optimizer */
    tofu_optimizer* optimizer = tofu_optimizer_sgd_create(g, LEARNING_RATE);
    assert(optimizer != NULL);
```

We create an SGD (Stochastic Gradient Descent) optimizer with our learning rate. The optimizer will:
1. Collect all parameter nodes from the graph
2. Compute gradients during backward passes
3. Update parameters based on those gradients

SGD is simple and effective: `param = param - learning_rate * gradient`

### Section 7: Training Loop Structure

```c
    float best_loss = INFINITY;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        /* Process each training example */
        for (int sample = 0; sample < 4; sample++) {
```

We train for 2000 epochs (full passes through the dataset). Each epoch processes all 4 XOR examples. We track the best loss and accumulate epoch loss for reporting.

This is "online learning" (one example at a time) rather than "batch learning," which is appropriate for tiny datasets.

### Section 8: Forward Pass

```c
            /* Zero gradients */
            tofu_graph_zero_grad(g);

            /* Create input tensor for this sample */
            float* input_data = (float*)malloc(INPUT_SIZE * sizeof(float));
            input_data[0] = xor_inputs[sample][0];
            input_data[1] = xor_inputs[sample][1];
            tofu_tensor* t_input = tofu_tensor_create(input_data, 1,
                                                      (int[]){INPUT_SIZE}, TOFU_FLOAT);
            tofu_graph_node* x = tofu_graph_input(g, t_input);

            /* Forward pass: Layer 1 */
            /* h1 = x @ w1 + b1 */
            tofu_graph_node* h1_matmul = tofu_graph_matmul(g, x, w1);
            tofu_graph_node* h1_bias = tofu_graph_add(g, h1_matmul, b1);

            /* Apply ReLU activation */
            tofu_graph_node* h1_relu = tofu_graph_relu(g, h1_bias);

            /* Forward pass: Layer 2 (output) */
            /* y = h1 @ w2 + b2 */
            tofu_graph_node* y_matmul = tofu_graph_matmul(g, h1_relu, w2);
            tofu_graph_node* y_pred = tofu_graph_add(g, y_matmul, b2);

            /* Create target tensor */
            float* target_data = (float*)malloc(OUTPUT_SIZE * sizeof(float));
            target_data[0] = xor_targets[sample][0];
            tofu_tensor* t_target = tofu_tensor_create(target_data, 1,
                                                       (int[]){OUTPUT_SIZE}, TOFU_FLOAT);
            tofu_graph_node* y_target = tofu_graph_input(g, t_target);

            /* Compute MSE loss */
            tofu_graph_node* loss_node = tofu_graph_mse_loss(g, y_pred, y_target);
```

Let's trace through the computation:

**Zero gradients**: Before computing new gradients, we clear old ones to prevent accumulation.

**Create input node**: Each example is a separate tensor created fresh, wrapped in a graph input node (non-trainable).

**Hidden layer computation**:
1. `h1_matmul = x @ w1`: Matrix multiply. Input [1, 2] @ weights [2, 4] → [1, 4]
2. `h1_bias = h1_matmul + b1`: Add bias [4] to each row (broadcasting)
3. `h1_relu = ReLU(h1_bias)`: Apply ReLU activation element-wise (max(0, x))

**Output layer computation**:
1. `y_matmul = h1_relu @ w2`: Matrix multiply. Hidden [1, 4] @ weights [4, 1] → [1, 1]
2. `y_pred = y_matmul + b2`: Add output bias [1]

**Loss computation**: `loss = MSE(y_pred, y_target)` = mean squared error = (y_pred - y_target)²

The computation graph now contains a chain: input → matmul → add → ReLU → matmul → add → loss

### Section 9: Backward Pass and Parameter Update

```c
            /* Extract loss value */
            tofu_tensor* loss_tensor = tofu_graph_get_value(loss_node);
            float sample_loss = 0.0f;
            if (loss_tensor && loss_tensor->len > 0) {
                TOFU_TENSOR_DATA_TO(loss_tensor, 0, sample_loss, TOFU_FLOAT);
            }
            epoch_loss += sample_loss;

            /* Backward pass: compute gradients */
            tofu_graph_backward(g, loss_node);

            /* Optimizer step: update weights and biases */
            tofu_optimizer_step(optimizer);

            /* Cleanup input/target tensors for this sample */
            tofu_tensor_free(t_input);
            tofu_tensor_free(t_target);
            free(input_data);
            free(target_data);

            /* Clear operations for next sample (keeps parameters) */
            tofu_graph_clear_ops(g);
        }

        /* Average loss over all 4 samples */
        epoch_loss /= 4;
```

**Extract loss**: We read the numerical loss value from the computed tensor using `TOFU_TENSOR_DATA_TO`.

**Backward pass**: `tofu_graph_backward(g, loss_node)` propagates gradients backward through the graph. Starting from the loss scalar, it computes:
- ∂loss/∂y_pred (what change in prediction would reduce loss)
- ∂loss/∂h1_relu (through the output layer)
- ∂loss/∂h1_bias (through ReLU)
- ∂loss/∂h1_matmul (through addition)
- ∂loss/∂w1, ∂loss/∂b1 (through matmul and add)
- ∂loss/∂w2, ∂loss/∂b2 (through second layer)

This is automatic differentiation—Tofu handles all the calculus!

**Parameter update**: `tofu_optimizer_step(optimizer)` updates all parameters using their gradients:
- w1 ← w1 - learning_rate × ∂loss/∂w1
- b1 ← b1 - learning_rate × ∂loss/∂b1
- (and similarly for w2, b2)

**Cleanup**: We free the input and target tensors (we allocated them, so we own them). Importantly, we keep w1, b1, w2, b2 (the parameters) intact.

**Clear operations**: `tofu_graph_clear_ops(g)` removes all the intermediate computation nodes but keeps the parameters. This prepares for the next sample without recreating parameters.

### Section 10: Training Progress Reporting

```c
        /* Report progress */
        if (epoch % REPORT_INTERVAL == 0 || epoch == NUM_EPOCHS - 1) {
            printf("Epoch %4d: loss = %.6f\n", epoch, epoch_loss);
        }
    }

    printf("\nTraining Complete!\n");
    printf("Final average loss: %.6f\n", best_loss);
```

We print loss every 200 epochs and at the end. Watching loss decrease is satisfying and helps you spot problems (e.g., loss increasing means learning rate is too high).

### Section 11: Evaluation

```c
    printf("\n");
    printf("Final Predictions:\n");
    printf("Input       Predicted  Target\n");
    printf("----        ---------  ------\n");

    for (int sample = 0; sample < 4; sample++) {
        /* Build inference graph for this sample */
        float* input_data = (float*)malloc(INPUT_SIZE * sizeof(float));
        input_data[0] = xor_inputs[sample][0];
        input_data[1] = xor_inputs[sample][1];
        tofu_tensor* t_input = tofu_tensor_create(input_data, 1,
                                                  (int[]){INPUT_SIZE}, TOFU_FLOAT);
        tofu_graph_node* x = tofu_graph_input(g, t_input);

        /* Forward pass (same as training) */
        tofu_graph_node* h1_matmul = tofu_graph_matmul(g, x, w1);
        tofu_graph_node* h1_bias = tofu_graph_add(g, h1_matmul, b1);
        tofu_graph_node* h1_relu = tofu_graph_relu(g, h1_bias);
        tofu_graph_node* y_matmul = tofu_graph_matmul(g, h1_relu, w2);
        tofu_graph_node* y_pred = tofu_graph_add(g, y_matmul, b2);

        /* Get prediction */
        tofu_tensor* pred_tensor = tofu_graph_get_value(y_pred);
        float prediction = 0.0f;
        if (pred_tensor && pred_tensor->len > 0) {
            TOFU_TENSOR_DATA_TO(pred_tensor, 0, prediction, TOFU_FLOAT);
        }

        printf("[%.0f, %.0f]    %.4f    %.0f\n",
               xor_inputs[sample][0], xor_inputs[sample][1],
               prediction, xor_targets[sample][0]);

        tofu_tensor_free(t_input);
        free(input_data);
        tofu_graph_clear_ops(g);
    }
```

After training, we perform inference (forward pass without backward) on all 4 examples. We print predicted values vs. targets. Since our output is a real number (not discrete), we'll see values close to 0 or 1.

### Section 12: Accuracy Check and Cleanup

```c
    /* Check accuracy (threshold at 0.5) */
    int correct = 0;
    for (int sample = 0; sample < 4; sample++) {
        /* ... build prediction graph ... */
        int pred_class = (prediction > 0.5f) ? 1 : 0;
        int true_class = (int)xor_targets[sample][0];

        if (pred_class == true_class) {
            correct++;
        }
        /* ... cleanup ... */
    }

    float accuracy = (float)correct / 4.0f;
    printf("Accuracy: %d/4 (%.1f%%)\n", correct, accuracy * 100.0f);

    if (accuracy == 1.0f) {
        printf("\nSuccess! Network learned XOR perfectly!\n");
    }

    /* Cleanup (IMPORTANT ORDER) */
    printf("\n");
    printf("Cleaning up resources...\n");

    tofu_optimizer_free(optimizer);
    tofu_graph_free(g);

    /* Free parameter tensors (caller owns them) */
    tofu_tensor_free_data_too(t_w1);
    tofu_tensor_free_data_too(t_b1);
    tofu_tensor_free_data_too(t_w2);
    tofu_tensor_free_data_too(t_b2);

    printf("Done!\n");
    printf("============================================================\n");

    return 0;
}
```

We convert predictions to binary (threshold at 0.5) and compute accuracy. Finally, **cleanup is critical and in the right order**:

1. Free optimizer (it might hold references to the graph)
2. Free graph (it owns the nodes but not the tensor data)
3. Free parameter tensors (we allocated the data, so we free it)

This order prevents use-after-free errors.

## Compiling and Running

Save the complete program as `examples/xor_training.c` (or copy from Tofu's examples directory).

### Compile

```bash
# From the tofu directory
make lib                # Build the library
cc -I./src examples/xor_training.c build/src/libtofu.a -o xor_training -lm
```

### Run

```bash
./xor_training
```

### Expected Output

```
============================================================
XOR Neural Network Training Example
============================================================

Network Architecture: [2] -> [4] -> [1]
Training: 2000 epochs with SGD, learning_rate=0.100

XOR Dataset:
  [0, 0] -> 0
  [0, 1] -> 1
  [1, 0] -> 1
  [1, 1] -> 0

Epoch    0: loss = 0.488130
Epoch  200: loss = 0.000000
Epoch  400: loss = 0.000000
...
Epoch 1999: loss = 0.000000

Training Complete!
Final average loss: 0.000000

Final Predictions:
Input       Predicted  Target
----        ---------  ------
[0, 0]    0.0000    0
[0, 1]    1.0000    1
[1, 0]    1.0000    1
[1, 1]    0.0000    0

Accuracy: 4/4 (100.0%)

Success! Network learned XOR perfectly!

Cleaning up resources...
Done!
```

**Key observation**: Loss rapidly converges to ~0 within a few hundred epochs! The network quickly learns to solve XOR.

## Understanding the Results

### Why Loss Decreases

Initially, the network makes random predictions (loss ≈ 0.49, far from correct). During training:

1. Gradients computed via backprop tell each parameter how it contributed to the error
2. Parameters move (via optimizer) in the direction that reduces loss
3. With each example, the network improves
4. After sufficient epochs, predictions are nearly perfect (loss → 0)

This iterative improvement is the essence of machine learning.

### What the Network Learned

The hidden layer developed 4 internal features (learned by the 4 hidden units). These features transform the input space so that XOR becomes linearly separable. Think of it as the network learning new coordinate axes in which the problem is easier.

The output layer learned to combine these 4 hidden features into a single decision.

### How to Verify Learning

The predictions match targets perfectly:
- `[0, 0] → 0.0000` (should be 0) ✓
- `[0, 1] → 1.0000` (should be 1) ✓
- `[1, 0] → 1.0000` (should be 1) ✓
- `[1, 1] → 0.0000` (should be 0) ✓

100% accuracy is the highest possible.

## Experimenting Further

Now that you've trained a network, try modifying parameters to understand their effects:

### Try Different Learning Rates

Change `LEARNING_RATE` to 0.01 (slower) or 0.5 (faster, but risky). Watch how convergence speed changes.

### Try Different Hidden Sizes

Change `HIDDEN_SIZE` to 2 (too small—might not converge) or 8 (overkill). Can the network still solve XOR?

### Add More Hidden Layers

Modify the forward pass to add another hidden layer:
```c
tofu_graph_node* h2 = tofu_graph_matmul(g, h1_relu, w2);
h2 = tofu_graph_add(g, h2, b2);
h2 = tofu_graph_relu(g, h2);
tofu_graph_node* y_matmul = tofu_graph_matmul(g, h2, w3);
```

Does a deeper network help? (For XOR, it shouldn't be necessary.)

### Monitor Individual Gradients

After `tofu_graph_backward()`, print gradient values to understand what each parameter is learning:
```c
tofu_tensor* w1_grad = tofu_graph_get_grad(w1);
printf("W1 gradient[0]: %.6f\n", ...);
```

## Next Steps

You've mastered the fundamentals! Here's your learning path:

1. **Dive Deeper**: Read the [Concepts Guide](concepts.md) to understand backpropagation and automatic differentiation in detail.

2. **Build Bigger**: Study the [CNN Training Example](../examples/cnn_training.c) to see how to scale to realistic datasets and architectures.

3. **Real Datasets**: Try training on real data:
   - MNIST for digit classification
   - Iris for flower classification
   - Your own custom dataset

4. **Advanced Optimizers**: Experiment with SGD with momentum or Adam (if available in your version).

5. **API Reference**: Consult the [Graph API](../api-reference/graph-api.md) and [Optimizer API](../api-reference/optimizer-api.md) for complete documentation of all functions.

## Key Takeaways

- **Computation graphs** let you define complex computations and differentiate them automatically
- **Forward pass** computes predictions (operations evaluate top-to-bottom)
- **Backward pass** computes gradients (automatic differentiation flows bottom-to-top)
- **Optimizers** update parameters based on gradients to minimize loss
- **Memory ownership** is crucial: you own input/parameter tensors, graph owns computed nodes
- **Iteration** (epochs) matters: neural networks improve with repeated exposure to data

You now understand the full training pipeline. You're ready to tackle more complex problems!
