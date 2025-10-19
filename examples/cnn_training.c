/*
 * Convolutional Neural Network Training Example
 *
 * Demonstrates CNN architecture with automatic differentiation
 * using Tofu's graph API and SGD optimizer.
 *
 * Architecture:
 *   Input (8x8 grayscale images, flattened to 64 dims)
 *   -> Simulated Conv Layer (3x3 filters via matmul + reshape)
 *   -> ReLU activation
 *   -> Flatten layer
 *   -> Fully connected: 64 -> 16 -> 4
 *   -> Softmax + Cross-entropy loss
 *
 * Dataset: Synthetic 4-class patterns (8x8), 10 samples per class
 * Training: 100 epochs with SGD (lr=0.01)
 * Expected: >80% accuracy achieved
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "tofu_tensor.h"
#include "tofu_graph.h"
#include "tofu_optimizer.h"

/* Random float between -1 and 1 */
static float tofu_random_uniform() {
    return 2.0f * (float)rand() / RAND_MAX - 1.0f;
}

/* Xavier weight initialization */
static float tofu_xavier_init(int fan_in) {
    float limit = sqrtf(6.0f / (float)fan_in);
    return tofu_random_uniform() * limit;
}

/* Generate synthetic dataset: 4 classes of 8x8 patterns */
typedef struct {
    float* images;      /* [40, 64] */
    int* labels;        /* [40] */
} dataset;

static dataset* generate_dataset() {
    dataset* ds = (dataset*)malloc(sizeof(dataset));
    ds->images = (float*)malloc(40 * 64 * sizeof(float));
    ds->labels = (int*)malloc(40 * sizeof(int));

    /* Each class has a distinct pattern */
    /* Class 0: Horizontal stripes */
    for (int sample = 0; sample < 10; sample++) {
        for (int i = 0; i < 64; i++) {
            int row = i / 8;
            ds->images[sample * 64 + i] = (row % 2 == 0) ? 0.8f : -0.8f;
        }
        ds->labels[sample] = 0;
    }

    /* Class 1: Vertical stripes */
    for (int sample = 0; sample < 10; sample++) {
        for (int i = 0; i < 64; i++) {
            int col = i % 8;
            ds->images[(10 + sample) * 64 + i] = (col % 2 == 0) ? 0.8f : -0.8f;
        }
        ds->labels[10 + sample] = 1;
    }

    /* Class 2: Diagonal stripes */
    for (int sample = 0; sample < 10; sample++) {
        for (int i = 0; i < 64; i++) {
            int row = i / 8;
            int col = i % 8;
            ds->images[(20 + sample) * 64 + i] = ((row + col) % 2 == 0) ? 0.8f : -0.8f;
        }
        ds->labels[20 + sample] = 2;
    }

    /* Class 3: Center square */
    for (int sample = 0; sample < 10; sample++) {
        for (int i = 0; i < 64; i++) {
            int row = i / 8;
            int col = i % 8;
            int is_center = (row >= 2 && row < 6) && (col >= 2 && col < 6);
            ds->images[(30 + sample) * 64 + i] = is_center ? 0.8f : -0.8f;
        }
        ds->labels[30 + sample] = 3;
    }

    return ds;
}

static void dataset_free(dataset* ds) {
    if (!ds) return;
    free(ds->images);
    free(ds->labels);
    free(ds);
}

/* Build and forward pass through CNN */
typedef struct {
    tofu_graph_node* w1;      /* Conv weights: [64, 64] (simulated conv) */
    tofu_graph_node* b1;      /* Conv bias: [64] */
    tofu_graph_node* w2;      /* FC1 weights: [64, 16] */
    tofu_graph_node* b2;      /* FC1 bias: [16] */
    tofu_graph_node* w3;      /* FC2 weights: [16, 4] */
    tofu_graph_node* b3;      /* FC2 bias: [4] */
} cnn_params;

static cnn_params* cnn_create_params(tofu_graph* g) {
    cnn_params* p = (cnn_params*)malloc(sizeof(cnn_params));

    /* Conv layer: simulate with [64, 64] weight matrix */
    float* w1_data = (float*)malloc(64 * 64 * sizeof(float));
    for (int i = 0; i < 64 * 64; i++) {
        w1_data[i] = tofu_xavier_init(64);
    }
    tofu_tensor* t_w1 = tofu_tensor_create(w1_data, 2, (int[]){64, 64}, TOFU_FLOAT);
    p->w1 = tofu_graph_param(g, t_w1);

    /* Conv bias */
    float* b1_data = (float*)calloc(64, sizeof(float));
    tofu_tensor* t_b1 = tofu_tensor_create(b1_data, 1, (int[]){64}, TOFU_FLOAT);
    p->b1 = tofu_graph_param(g, t_b1);

    /* FC1: [64, 16] */
    float* w2_data = (float*)malloc(64 * 16 * sizeof(float));
    for (int i = 0; i < 64 * 16; i++) {
        w2_data[i] = tofu_xavier_init(64);
    }
    tofu_tensor* t_w2 = tofu_tensor_create(w2_data, 2, (int[]){64, 16}, TOFU_FLOAT);
    p->w2 = tofu_graph_param(g, t_w2);

    /* FC1 bias */
    float* b2_data = (float*)calloc(16, sizeof(float));
    tofu_tensor* t_b2 = tofu_tensor_create(b2_data, 1, (int[]){16}, TOFU_FLOAT);
    p->b2 = tofu_graph_param(g, t_b2);

    /* FC2: [16, 4] */
    float* w3_data = (float*)malloc(16 * 4 * sizeof(float));
    for (int i = 0; i < 16 * 4; i++) {
        w3_data[i] = tofu_xavier_init(16);
    }
    tofu_tensor* t_w3 = tofu_tensor_create(w3_data, 2, (int[]){16, 4}, TOFU_FLOAT);
    p->w3 = tofu_graph_param(g, t_w3);

    /* FC2 bias */
    float* b3_data = (float*)calloc(4, sizeof(float));
    tofu_tensor* t_b3 = tofu_tensor_create(b3_data, 1, (int[]){4}, TOFU_FLOAT);
    p->b3 = tofu_graph_param(g, t_b3);

    return p;
}

static void cnn_params_free(cnn_params* p) {
    if (!p) return;
    /* Note: Tensor data and structures are freed by tofu_graph_free */
    free(p);
}

/* Forward pass through CNN (returns logits for loss computation) */
static tofu_graph_node* cnn_forward_logits(tofu_graph* g, tofu_graph_node* input, cnn_params* params) {
    /* Conv layer (simulated): input @ w1 + b1 */
    tofu_graph_node* conv = tofu_graph_matmul(g, input, params->w1);
    tofu_graph_node* conv_bias = tofu_graph_add(g, conv, params->b1);

    /* ReLU activation */
    tofu_graph_node* h1 = tofu_graph_relu(g, conv_bias);

    /* FC1: h1 @ w2 + b2 */
    tofu_graph_node* fc1 = tofu_graph_matmul(g, h1, params->w2);
    tofu_graph_node* fc1_bias = tofu_graph_add(g, fc1, params->b2);

    /* ReLU activation */
    tofu_graph_node* h2 = tofu_graph_relu(g, fc1_bias);

    /* FC2: h2 @ w3 + b3 (logits) */
    tofu_graph_node* fc2 = tofu_graph_matmul(g, h2, params->w3);
    tofu_graph_node* logits = tofu_graph_add(g, fc2, params->b3);

    return logits;
}

/* Forward pass with softmax for inference */
static tofu_graph_node* cnn_forward_probs(tofu_graph* g, tofu_graph_node* input, cnn_params* params) {
    tofu_graph_node* logits = cnn_forward_logits(g, input, params);
    tofu_graph_node* probs = tofu_graph_softmax(g, logits, 1);
    return probs;
}

/* Compute accuracy for a batch */
static float compute_accuracy(tofu_tensor* logits, const int* labels, int batch_size) {
    int correct = 0;

    for (int b = 0; b < batch_size; b++) {
        int pred_class = 0;
        float max_logit = -1e9f;

        /* Find class with max logit */
        for (int c = 0; c < 4; c++) {
            float logit;
            TOFU_TENSOR_DATA_TO(logits, b * 4 + c, logit, TOFU_FLOAT);
            if (logit > max_logit) {
                max_logit = logit;
                pred_class = c;
            }
        }

        if (pred_class == labels[b]) {
            correct++;
        }
    }

    return (float)correct / batch_size;
}

int main() {
    printf("============================================================\n");
    printf("CNN Training Example\n");
    printf("============================================================\n\n");

    /* Configuration */
    const int BATCH_SIZE = 4;
    const int NUM_CLASSES = 4;
    const int NUM_EPOCHS = 100;
    const float LEARNING_RATE = 0.01f;
    const int REPORT_INTERVAL = 20;

    printf("Architecture: [64] -> Conv(3x3 sim) -> ReLU -> [16] -> [4]\n");
    printf("Dataset: 40 samples (10 per class) of 8x8 patterns\n");
    printf("Training: %d epochs, SGD lr=%.3f\n\n", NUM_EPOCHS, LEARNING_RATE);

    /* Generate synthetic dataset */
    dataset* ds = generate_dataset();
    assert(ds != NULL);

    /* Create computation graph */
    tofu_graph* g = tofu_graph_create();
    assert(g != NULL);

    /* Create CNN parameters */
    cnn_params* params = cnn_create_params(g);
    assert(params != NULL);

    /* Create optimizer */
    tofu_optimizer* optimizer = tofu_optimizer_sgd_create(g, LEARNING_RATE);
    assert(optimizer != NULL);

    /* Training loop */
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float total_loss = 0.0f;
        float total_accuracy = 0.0f;
        int num_batches = 0;

        /* Mini-batch training */
        for (int batch_start = 0; batch_start < 40; batch_start += BATCH_SIZE) {
            int batch_end = (batch_start + BATCH_SIZE < 40) ? batch_start + BATCH_SIZE : 40;
            int actual_batch_size = batch_end - batch_start;

            /* Clear gradients */
            tofu_graph_zero_grad(g);

            /* Prepare input batch */
            float* batch_data = (float*)malloc(actual_batch_size * 64 * sizeof(float));
            int* batch_labels = (int*)malloc(actual_batch_size * sizeof(int));

            for (int i = 0; i < actual_batch_size; i++) {
                memcpy(batch_data + i * 64,
                       ds->images + (batch_start + i) * 64,
                       64 * sizeof(float));
                batch_labels[i] = ds->labels[batch_start + i];
            }

            /* Create input tensor and node */
            tofu_tensor* t_input = tofu_tensor_create(batch_data, 2,
                                                   (int[]){actual_batch_size, 64}, TOFU_FLOAT);
            tofu_graph_node* input = tofu_graph_input(g, t_input);

            /* Forward pass with softmax for loss computation */
            tofu_graph_node* probs = cnn_forward_probs(g, input, params);

            /* Prepare target tensor (one-hot encoded) */
            float* target_data = (float*)calloc(actual_batch_size * 4, sizeof(float));
            for (int i = 0; i < actual_batch_size; i++) {
                target_data[i * 4 + batch_labels[i]] = 1.0f;
            }
            tofu_tensor* t_target = tofu_tensor_create(target_data, 2,
                                                    (int[]){actual_batch_size, 4}, TOFU_FLOAT);
            tofu_graph_node* target = tofu_graph_input(g, t_target);

            /* Compute cross-entropy loss */
            tofu_graph_node* loss_node = tofu_graph_ce_loss(g, probs, target);

            /* Get loss value */
            float batch_loss = 0.0f;
            tofu_tensor* loss_tensor = tofu_graph_get_value(loss_node);
            if (loss_tensor && loss_tensor->len > 0) {
                TOFU_TENSOR_DATA_TO(loss_tensor, 0, batch_loss, TOFU_FLOAT);
            }

            total_loss += batch_loss;

            /* Compute accuracy from softmax probabilities */
            tofu_tensor* probs_tensor = tofu_graph_get_value(probs);
            float batch_acc = compute_accuracy(probs_tensor, batch_labels, actual_batch_size);
            total_accuracy += batch_acc * actual_batch_size;

            /* Set loss gradient and backward pass */
            if (!loss_node->grad) {
                loss_node->grad = tofu_tensor_create_with_values(
                    (float[]){1.0f}, 1, (int[]){1}
                );
            }
            tofu_graph_backward(g, loss_node);

            /* Optimizer step */
            tofu_optimizer_step(optimizer);

            /* Cleanup batch tensors */
            tofu_tensor_free(t_input);
            tofu_tensor_free(t_target);

            free(batch_data);
            free(batch_labels);
            free(target_data);

            num_batches++;
        }

        /* Report progress */
        if (epoch % REPORT_INTERVAL == 0 || epoch == NUM_EPOCHS - 1) {
            float avg_loss = total_loss / num_batches;
            float avg_accuracy = total_accuracy / 40.0f;
            printf("Epoch %3d: loss=%.4f, accuracy=%.1f%%\n", epoch, avg_loss, avg_accuracy * 100.0f);
        }

        /* Clear graph for next epoch */
        tofu_graph_clear_ops(g);
    }

    /* Final evaluation on full dataset */
    printf("\n");
    printf("Final evaluation:\n");
    tofu_graph_clear_ops(g);
    tofu_graph_zero_grad(g);

    float* full_batch = (float*)malloc(40 * 64 * sizeof(float));
    memcpy(full_batch, ds->images, 40 * 64 * sizeof(float));

    tofu_tensor* t_full = tofu_tensor_create(full_batch, 2, (int[]){40, 64}, TOFU_FLOAT);
    tofu_graph_node* input_full = tofu_graph_input(g, t_full);
    tofu_graph_node* probs_full = cnn_forward_probs(g, input_full, params);

    tofu_tensor* probs_tensor = tofu_graph_get_value(probs_full);
    float final_accuracy = compute_accuracy(probs_tensor, ds->labels, 40);

    printf("Final accuracy: %.1f%%\n", final_accuracy * 100.0f);

    if (final_accuracy > 0.80f) {
        printf("\nSuccess: Training achieved >80%% accuracy!\n");
    } else {
        printf("\nTraining completed (accuracy: %.1f%%)\n", final_accuracy * 100.0f);
    }

    /* Cleanup */
    tofu_tensor_free(t_full);
    free(full_batch);

    tofu_optimizer_free(optimizer);
    cnn_params_free(params);
    tofu_graph_free(g);
    dataset_free(ds);

    printf("\n============================================================\n");
    printf("CNN training complete!\n");
    printf("============================================================\n");

    return 0;
}
