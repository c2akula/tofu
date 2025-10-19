/*
 * Tofu Framework - Validation Test Suite Phase 3: Multi-Class Classification
 *
 * This test suite validates the framework's ability to perform multi-class
 * classification on linearly separable data using a 2-layer neural network.
 *
 * Test 3.1: Multi-Class Classification
 * - Dataset: 3 classes of 2D points (10 samples per class)
 * - Network: [2] -> [8] -> [3] with ReLU + softmax
 * - Loss: Cross-entropy loss
 * - Optimizer: SGD with lr=0.01
 * - Training: 200 epochs
 * - Success: Accuracy > 90%
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "tl_graph.h"
#include "tl_tensor.h"
#include "tl_optimizer.h"

#define NUM_CLASSES 3
#define NUM_SAMPLES_PER_CLASS 10
#define NUM_SAMPLES (NUM_SAMPLES_PER_CLASS * NUM_CLASSES)  /* 30 total */
#define INPUT_DIM 2
#define HIDDEN_DIM 8
#define OUTPUT_DIM 3
#define EPOCHS 200
#define LEARNING_RATE 0.01f
#define NOISE_STDDEV 0.1f
#define ACCURACY_THRESHOLD 0.90f

/* Class centers for 3-class problem */
static const float class_centers[NUM_CLASSES][INPUT_DIM] = {
    {0.0f, 0.0f},      /* Class 0 */
    {1.0f, 0.0f},      /* Class 1 */
    {0.5f, 1.0f}       /* Class 2 */
};

/*
 * Generate random Gaussian sample using Box-Muller transform
 * Returns value sampled from N(0, 1)
 */
static float gaussian_random() {
    static int has_spare = 0;
    static float spare;

    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    has_spare = 1;
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;

    /* Avoid log(0) */
    if (u1 < 1e-6f) u1 = 1e-6f;

    float mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * 3.14159265359f * u2);
    spare = mag * sinf(2.0f * 3.14159265359f * u2);

    return z0;
}

/*
 * Helper: Generate dataset for 3-class classification
 * Outputs:
 *   - X: (30, 2) matrix of input features
 *   - Y: (30, 3) one-hot encoded labels
 */
static void generate_dataset(float* X, float* Y) {
    /* Initialize Y to all zeros */
    memset(Y, 0, NUM_SAMPLES * OUTPUT_DIM * sizeof(float));

    /* Generate samples for each class */
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
        for (int sample_idx = 0; sample_idx < NUM_SAMPLES_PER_CLASS; sample_idx++) {
            int global_idx = class_idx * NUM_SAMPLES_PER_CLASS + sample_idx;

            /* Generate 2D point near class center with Gaussian noise */
            float x0 = class_centers[class_idx][0] + NOISE_STDDEV * gaussian_random();
            float x1 = class_centers[class_idx][1] + NOISE_STDDEV * gaussian_random();

            X[global_idx * INPUT_DIM + 0] = x0;
            X[global_idx * INPUT_DIM + 1] = x1;

            /* Set one-hot label */
            Y[global_idx * OUTPUT_DIM + class_idx] = 1.0f;
        }
    }
}

/*
 * Helper: Initialize weights using Xavier initialization
 * limit = sqrt(6 / (fan_in + fan_out))
 */
static void init_weights_xavier(float* data, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
    }
}

/*
 * Helper: Initialize biases to zero
 */
static void init_bias_zero(float* data, int dim) {
    memset(data, 0, dim * sizeof(float));
}

/*
 * Helper: Get predicted class (argmax of softmax output)
 */
static int get_predicted_class(float* logits, int num_classes) {
    int pred_class = 0;
    float max_val = logits[0];

    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            pred_class = i;
        }
    }

    return pred_class;
}

/*
 * Helper: Get true class from one-hot label
 */
static int get_true_class(float* label, int num_classes) {
    for (int i = 0; i < num_classes; i++) {
        if (label[i] > 0.5f) {
            return i;
        }
    }
    return 0;  /* Should not happen with valid one-hot encoding */
}

/*
 * Test 3.1: Multi-Class Classification
 *
 * Validates that the network can classify 3 classes of 2D points
 * with accuracy > 90% after 200 epochs of training.
 */
static void test_multi_class_classification() {
    printf("\n");
    printf("================================================================================\n");
    printf("Test 3.1: Multi-Class Classification\n");
    printf("================================================================================\n");
    printf("Dataset: 30 samples (10 per class) of 2D points with Gaussian noise σ=0.1\n");
    printf("Network: [2] -> [8] -> [3] with ReLU + softmax\n");
    printf("Loss: Cross-entropy loss\n");
    printf("Optimizer: SGD with lr=0.01\n");
    printf("Training: %d epochs\n", EPOCHS);
    printf("Success Criterion: Accuracy > %.1f%%\n", ACCURACY_THRESHOLD * 100);
    printf("================================================================================\n\n");

    /* Allocate dataset */
    float* X = (float*)malloc(NUM_SAMPLES * INPUT_DIM * sizeof(float));
    float* Y = (float*)malloc(NUM_SAMPLES * OUTPUT_DIM * sizeof(float));

    assert(X != NULL && Y != NULL);

    /* Generate dataset */
    printf("Generating dataset...\n");
    srand(42);  /* Reproducibility */
    generate_dataset(X, Y);

    /* Print sample points from each class */
    printf("Sample points from each class:\n");
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
        int sample_idx = class_idx * NUM_SAMPLES_PER_CLASS;
        printf("  Class %d (center ~[%.1f, %.1f]): ", class_idx,
               class_centers[class_idx][0], class_centers[class_idx][1]);
        printf("[%.3f, %.3f]\n", X[sample_idx * INPUT_DIM], X[sample_idx * INPUT_DIM + 1]);
    }
    printf("\n");

    /* Allocate weight and bias tensors */
    float* W1_data = (float*)malloc(INPUT_DIM * HIDDEN_DIM * sizeof(float));
    float* b1_data = (float*)malloc(HIDDEN_DIM * sizeof(float));
    float* W2_data = (float*)malloc(HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    float* b2_data = (float*)malloc(OUTPUT_DIM * sizeof(float));

    assert(W1_data && b1_data && W2_data && b2_data);

    /* Initialize weights and biases */
    init_weights_xavier(W1_data, INPUT_DIM, HIDDEN_DIM);
    init_bias_zero(b1_data, HIDDEN_DIM);
    init_weights_xavier(W2_data, HIDDEN_DIM, OUTPUT_DIM);
    init_bias_zero(b2_data, OUTPUT_DIM);

    printf("Network initialized with Xavier weights and zero biases\n\n");

    /* Create computation graph */
    tl_graph* g = tl_graph_create();
    assert(g != NULL);

    /* Create parameter tensors */
    tl_tensor* t_W1 = tl_tensor_create(W1_data, 2, (int[]){INPUT_DIM, HIDDEN_DIM}, TL_FLOAT);
    tl_tensor* t_b1 = tl_tensor_create(b1_data, 1, (int[]){HIDDEN_DIM}, TL_FLOAT);
    tl_tensor* t_W2 = tl_tensor_create(W2_data, 2, (int[]){HIDDEN_DIM, OUTPUT_DIM}, TL_FLOAT);
    tl_tensor* t_b2 = tl_tensor_create(b2_data, 1, (int[]){OUTPUT_DIM}, TL_FLOAT);

    assert(t_W1 && t_b1 && t_W2 && t_b2);

    /* Create trainable parameters */
    tl_graph_node* p_W1 = tl_graph_param(g, t_W1);
    tl_graph_node* p_b1 = tl_graph_param(g, t_b1);
    tl_graph_node* p_W2 = tl_graph_param(g, t_W2);
    tl_graph_node* p_b2 = tl_graph_param(g, t_b2);

    assert(p_W1 && p_b1 && p_W2 && p_b2);

    /* Create optimizer */
    tl_optimizer* opt = tl_optimizer_sgd_create(g, LEARNING_RATE);
    assert(opt != NULL);

    /* Manually add parameters to optimizer */
    tl_optimizer_add_param(opt, p_W1);
    tl_optimizer_add_param(opt, p_b1);
    tl_optimizer_add_param(opt, p_W2);
    tl_optimizer_add_param(opt, p_b2);

    printf("Starting training...\n");
    printf("Epoch | Loss      | Train Accuracy\n");
    printf("------|-----------|----------------\n");

    /* Training loop */
    float min_loss = 1e9f;
    int best_epoch = 0;
    float best_accuracy = 0.0f;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;

        /* Iterate over all samples */
        for (int sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) {
            /* Clear the graph for new forward pass */
            tl_graph_clear_ops(g);

            /* Get single sample and label */
            float* x_sample = &X[sample_idx * INPUT_DIM];
            float* y_sample = &Y[sample_idx * OUTPUT_DIM];

            /* Create input tensor: reshape to (1, 2) */
            tl_tensor* t_x = tl_tensor_create(x_sample, 1, (int[]){INPUT_DIM}, TL_FLOAT);
            tl_tensor* t_y = tl_tensor_create(y_sample, 1, (int[]){OUTPUT_DIM}, TL_FLOAT);

            assert(t_x && t_y);

            /* Create input nodes */
            tl_graph_node* x = tl_graph_input(g, t_x);
            tl_graph_node* y_true = tl_graph_input(g, t_y);

            /* Forward pass: h = relu(x @ W1 + b1) */
            tl_graph_node* xW1 = tl_graph_matmul(g, x, p_W1);
            tl_graph_node* xW1_b1 = tl_graph_add(g, xW1, p_b1);
            tl_graph_node* h = tl_graph_relu(g, xW1_b1);

            /* Output: logits = h @ W2 + b2 */
            tl_graph_node* hW2 = tl_graph_matmul(g, h, p_W2);
            tl_graph_node* logits = tl_graph_add(g, hW2, p_b2);

            /* Apply softmax for probabilities */
            tl_graph_node* probs = tl_graph_softmax(g, logits, 0);

            /* Compute cross-entropy loss */
            tl_graph_node* loss_node = tl_graph_ce_loss(g, probs, y_true);

            /* Extract loss value */
            float loss_val = 0.0f;
            TL_TENSOR_DATA_TO(loss_node->value, 0, loss_val, TL_FLOAT);

            epoch_loss += loss_val;

            /* Check prediction accuracy */
            float* logits_data = (float*)logits->value->data;
            int pred_class = get_predicted_class(logits_data, OUTPUT_DIM);
            int true_class = get_true_class(y_sample, OUTPUT_DIM);

            if (pred_class == true_class) {
                correct_predictions++;
            }

            /* Backward pass */
            tl_graph_zero_grad(g);
            tl_graph_backward(g, loss_node);

            /* Optimizer step */
            tl_optimizer_step(opt);

            /* Cleanup for this iteration */
            tl_tensor_free(t_x);
            tl_tensor_free(t_y);
        }

        /* Print training progress */
        float avg_loss = epoch_loss / NUM_SAMPLES;
        float train_accuracy = (float)correct_predictions / NUM_SAMPLES;

        if ((epoch + 1) % 20 == 0 || epoch < 5) {
            printf("%5d | %9.6f | %.4f\n", epoch + 1, avg_loss, train_accuracy);
        }

        /* Track best model */
        if (train_accuracy > best_accuracy) {
            best_accuracy = train_accuracy;
            best_epoch = epoch + 1;
        }

        if (avg_loss < min_loss) {
            min_loss = avg_loss;
        }
    }

    printf("\n");
    printf("Training Summary:\n");
    printf("  Best Accuracy: %.4f at epoch %d\n", best_accuracy, best_epoch);
    printf("  Final Loss: %.6f\n", min_loss);
    printf("\n");

    /* Evaluate on full dataset */
    printf("Final Evaluation on Full Dataset:\n");
    printf("---------------------------------\n");

    int final_correct = 0;
    int class_correct[NUM_CLASSES] = {0};
    int class_total[NUM_CLASSES] = {0};

    for (int sample_idx = 0; sample_idx < NUM_SAMPLES; sample_idx++) {
        tl_graph_clear_ops(g);

        float* x_sample = &X[sample_idx * INPUT_DIM];
        float* y_sample = &Y[sample_idx * OUTPUT_DIM];

        tl_tensor* t_x = tl_tensor_create(x_sample, 1, (int[]){INPUT_DIM}, TL_FLOAT);
        tl_tensor* t_y = tl_tensor_create(y_sample, 1, (int[]){OUTPUT_DIM}, TL_FLOAT);

        tl_graph_node* x = tl_graph_input(g, t_x);
        tl_graph_node* y_true = tl_graph_input(g, t_y);

        /* Forward pass: h = relu(x @ W1 + b1) */
        tl_graph_node* xW1 = tl_graph_matmul(g, x, p_W1);
        tl_graph_node* xW1_b1 = tl_graph_add(g, xW1, p_b1);
        tl_graph_node* h = tl_graph_relu(g, xW1_b1);

        /* Output: logits = h @ W2 + b2 */
        tl_graph_node* hW2 = tl_graph_matmul(g, h, p_W2);
        tl_graph_node* logits = tl_graph_add(g, hW2, p_b2);

        float* logits_data = (float*)logits->value->data;
        int pred_class = get_predicted_class(logits_data, OUTPUT_DIM);
        int true_class = get_true_class(y_sample, OUTPUT_DIM);

        class_total[true_class]++;

        if (pred_class == true_class) {
            final_correct++;
            class_correct[true_class]++;
        }

        tl_tensor_free(t_x);
        tl_tensor_free(t_y);
    }

    float final_accuracy = (float)final_correct / NUM_SAMPLES;

    printf("Overall Accuracy: %.4f (%d/%d correct)\n", final_accuracy, final_correct, NUM_SAMPLES);
    printf("\nPer-Class Accuracy:\n");
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
        float class_acc = class_total[class_idx] > 0 ? (float)class_correct[class_idx] / class_total[class_idx] : 0.0f;
        printf("  Class %d: %.4f (%d/%d)\n", class_idx, class_acc, class_correct[class_idx], class_total[class_idx]);
    }

    printf("\n");
    printf("================================================================================\n");

    /* Verify success criterion */
    if (final_accuracy > ACCURACY_THRESHOLD) {
        printf("PASS: Accuracy %.4f > %.4f\n", final_accuracy, ACCURACY_THRESHOLD);
        printf("================================================================================\n\n");
    } else {
        printf("FAIL: Accuracy %.4f <= %.4f\n", final_accuracy, ACCURACY_THRESHOLD);
        printf("================================================================================\n\n");
    }

    /* Cleanup */
    tl_graph_free(g);
    tl_tensor_free(t_W1);
    tl_tensor_free(t_b1);
    tl_tensor_free(t_W2);
    tl_tensor_free(t_b2);
    tl_optimizer_free(opt);

    free(X);
    free(Y);
    free(W1_data);
    free(b1_data);
    free(W2_data);
    free(b2_data);
}

int main() {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║           Tofu Validation Test Suite Phase 3: Multi-Class Classification       ║\n");
    printf("╚════════════════════════════════════════════════════════════════════════════════╝\n");

    test_multi_class_classification();

    printf("Test suite completed.\n");

    return 0;
}
