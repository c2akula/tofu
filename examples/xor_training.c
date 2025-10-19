/*
 * XOR Neural Network Training Example
 *
 * Demonstrates building and training a simple neural network to solve the XOR problem.
 * This is a classic machine learning problem requiring a non-linear (hidden layer) solution.
 *
 * Architecture:
 *   Input (2 dims) -> Hidden (4 hidden units, ReLU) -> Output (1 dim)
 *
 * Dataset: XOR truth table
 *   [0,0] -> 0
 *   [0,1] -> 1
 *   [1,0] -> 1
 *   [1,1] -> 0
 *
 * Training: SGD with learning rate 0.1, 2000 epochs
 * Expected: Loss converges to near 0, network learns XOR function
 */

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
    return limit * (2.0f * (float)rand() / RAND_MAX - 1.0f);
}

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

    printf("Network Architecture: [%d] -> [%d] -> [%d]\n",
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Training: %d epochs with SGD, learning_rate=%.3f\n\n",
           NUM_EPOCHS, LEARNING_RATE);

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

    /* Create computation graph */
    tofu_graph* g = tofu_graph_create();
    assert(g != NULL);

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

    /* Create optimizer */
    tofu_optimizer* optimizer = tofu_optimizer_sgd_create(g, LEARNING_RATE);
    assert(optimizer != NULL);

    /* Training loop */
    float best_loss = INFINITY;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        /* Process each training example */
        for (int sample = 0; sample < 4; sample++) {
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

        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
        }

        /* Report progress */
        if (epoch % REPORT_INTERVAL == 0 || epoch == NUM_EPOCHS - 1) {
            printf("Epoch %4d: loss = %.6f\n", epoch, epoch_loss);
        }
    }

    printf("\nTraining Complete!\n");
    printf("Final average loss: %.6f\n", best_loss);

    /* Evaluation: Test on XOR dataset */
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

        /* Forward pass */
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

    printf("\n");

    /* Check accuracy (threshold at 0.5) */
    int correct = 0;
    for (int sample = 0; sample < 4; sample++) {
        float* input_data = (float*)malloc(INPUT_SIZE * sizeof(float));
        input_data[0] = xor_inputs[sample][0];
        input_data[1] = xor_inputs[sample][1];
        tofu_tensor* t_input = tofu_tensor_create(input_data, 1,
                                                  (int[]){INPUT_SIZE}, TOFU_FLOAT);
        tofu_graph_node* x = tofu_graph_input(g, t_input);

        tofu_graph_node* h1_matmul = tofu_graph_matmul(g, x, w1);
        tofu_graph_node* h1_bias = tofu_graph_add(g, h1_matmul, b1);
        tofu_graph_node* h1_relu = tofu_graph_relu(g, h1_bias);
        tofu_graph_node* y_matmul = tofu_graph_matmul(g, h1_relu, w2);
        tofu_graph_node* y_pred = tofu_graph_add(g, y_matmul, b2);

        tofu_tensor* pred_tensor = tofu_graph_get_value(y_pred);
        float prediction = 0.0f;
        if (pred_tensor && pred_tensor->len > 0) {
            TOFU_TENSOR_DATA_TO(pred_tensor, 0, prediction, TOFU_FLOAT);
        }

        int pred_class = (prediction > 0.5f) ? 1 : 0;
        int true_class = (int)xor_targets[sample][0];

        if (pred_class == true_class) {
            correct++;
        }

        tofu_tensor_free(t_input);
        free(input_data);
        tofu_graph_clear_ops(g);
    }

    float accuracy = (float)correct / 4.0f;
    printf("Accuracy: %d/4 (%.1f%%)\n", correct, accuracy * 100.0f);

    if (accuracy == 1.0f) {
        printf("\nSuccess! Network learned XOR perfectly!\n");
    } else if (accuracy >= 0.75f) {
        printf("\nGood progress! Network learned most of XOR.\n");
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
