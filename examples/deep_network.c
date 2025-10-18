/*
 * Deep neural network with batch processing
 * Tests: batch matmul, multiple layers, larger matrices
 *
 * Network: input(10) -> 64 -> 32 -> 16 -> output(5)
 * Batch size: 8
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tl_tensor.h"

/* Load binary weights */
float* load_weights(const char* filename, int size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }

    float* data = (float*)malloc(size * sizeof(float));
    size_t read = fread(data, sizeof(float), size, f);
    fclose(f);

    if (read != size) {
        fprintf(stderr, "Error: Expected %d floats in %s, read %zu\n", size, filename, read);
        free(data);
        return NULL;
    }

    return data;
}

/* ReLU activation */
void relu(tl_tensor* t) {
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        if (val < 0) val = 0;
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

/* Add bias using broadcasting-like behavior */
tl_tensor* add_bias_batch(tl_tensor* x, tl_tensor* bias) {
    /* x: [batch, features], bias: [features] */
    tl_tensor* result = tl_tensor_clone(x);

    int batch_size = x->dims[0];
    int features = x->dims[1];

    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < features; f++) {
            float x_val, b_val;
            int idx = b * features + f;
            TL_TENSOR_DATA_TO(result, idx, x_val, TL_FLOAT);
            TL_TENSOR_DATA_TO(bias, f, b_val, TL_FLOAT);
            x_val += b_val;
            TL_TENSOR_DATA_FROM(result, idx, x_val, TL_FLOAT);
        }
    }

    return result;
}

int main() {
    printf("============================================================\n");
    printf("Deep Neural Network with Batch Processing\n");
    printf("============================================================\n\n");

    /* Architecture */
    const int BATCH_SIZE = 8;
    const int INPUT_SIZE = 10;
    const int HIDDEN1_SIZE = 64;
    const int HIDDEN2_SIZE = 32;
    const int HIDDEN3_SIZE = 16;
    const int OUTPUT_SIZE = 5;

    printf("Architecture:\n");
    printf("  Input:   [%d, %d]\n", BATCH_SIZE, INPUT_SIZE);
    printf("  Hidden1: [%d, %d] with ReLU\n", BATCH_SIZE, HIDDEN1_SIZE);
    printf("  Hidden2: [%d, %d] with ReLU\n", BATCH_SIZE, HIDDEN2_SIZE);
    printf("  Hidden3: [%d, %d] with ReLU\n", BATCH_SIZE, HIDDEN3_SIZE);
    printf("  Output:  [%d, %d]\n\n", BATCH_SIZE, OUTPUT_SIZE);

    /* Load weights */
    printf("Loading weights...\n");
    float* W1_data = load_weights("examples/deep_W1.bin", INPUT_SIZE * HIDDEN1_SIZE);
    float* b1_data = load_weights("examples/deep_b1.bin", HIDDEN1_SIZE);
    float* W2_data = load_weights("examples/deep_W2.bin", HIDDEN1_SIZE * HIDDEN2_SIZE);
    float* b2_data = load_weights("examples/deep_b2.bin", HIDDEN2_SIZE);
    float* W3_data = load_weights("examples/deep_W3.bin", HIDDEN2_SIZE * HIDDEN3_SIZE);
    float* b3_data = load_weights("examples/deep_b3.bin", HIDDEN3_SIZE);
    float* W4_data = load_weights("examples/deep_W4.bin", HIDDEN3_SIZE * OUTPUT_SIZE);
    float* b4_data = load_weights("examples/deep_b4.bin", OUTPUT_SIZE);
    float* input_data = load_weights("examples/deep_input.bin", BATCH_SIZE * INPUT_SIZE);

    if (!W1_data || !b1_data || !W2_data || !b2_data ||
        !W3_data || !b3_data || !W4_data || !b4_data || !input_data) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    /* Create tensors */
    tl_tensor* X = tl_tensor_create(input_data, 2, (int[]){BATCH_SIZE, INPUT_SIZE}, TL_FLOAT);
    tl_tensor* W1 = tl_tensor_create(W1_data, 2, (int[]){INPUT_SIZE, HIDDEN1_SIZE}, TL_FLOAT);
    tl_tensor* b1 = tl_tensor_create(b1_data, 1, (int[]){HIDDEN1_SIZE}, TL_FLOAT);
    tl_tensor* W2 = tl_tensor_create(W2_data, 2, (int[]){HIDDEN1_SIZE, HIDDEN2_SIZE}, TL_FLOAT);
    tl_tensor* b2 = tl_tensor_create(b2_data, 1, (int[]){HIDDEN2_SIZE}, TL_FLOAT);
    tl_tensor* W3 = tl_tensor_create(W3_data, 2, (int[]){HIDDEN2_SIZE, HIDDEN3_SIZE}, TL_FLOAT);
    tl_tensor* b3 = tl_tensor_create(b3_data, 1, (int[]){HIDDEN3_SIZE}, TL_FLOAT);
    tl_tensor* W4 = tl_tensor_create(W4_data, 2, (int[]){HIDDEN3_SIZE, OUTPUT_SIZE}, TL_FLOAT);
    tl_tensor* b4 = tl_tensor_create(b4_data, 1, (int[]){OUTPUT_SIZE}, TL_FLOAT);

    printf("Weights loaded successfully.\n\n");

    /* Forward pass */
    printf("Forward pass:\n");

    /* Layer 1 */
    printf("  Layer 1: [%d,%d] @ [%d,%d] -> [%d,%d]\n",
           X->dims[0], X->dims[1], W1->dims[0], W1->dims[1],
           BATCH_SIZE, HIDDEN1_SIZE);
    tl_tensor* h1 = tl_tensor_matmul(X, W1, NULL);
    if (!h1) {
        fprintf(stderr, "Layer 1 matmul failed!\n");
        return 1;
    }
    tl_tensor* h1_bias = add_bias_batch(h1, b1);
    tl_tensor_free_data_too(h1);
    relu(h1_bias);

    /* Layer 2 */
    printf("  Layer 2: [%d,%d] @ [%d,%d] -> [%d,%d]\n",
           h1_bias->dims[0], h1_bias->dims[1], W2->dims[0], W2->dims[1],
           BATCH_SIZE, HIDDEN2_SIZE);
    tl_tensor* h2 = tl_tensor_matmul(h1_bias, W2, NULL);
    if (!h2) {
        fprintf(stderr, "Layer 2 matmul failed!\n");
        return 1;
    }
    tl_tensor* h2_bias = add_bias_batch(h2, b2);
    tl_tensor_free_data_too(h2);
    relu(h2_bias);

    /* Layer 3 */
    printf("  Layer 3: [%d,%d] @ [%d,%d] -> [%d,%d]\n",
           h2_bias->dims[0], h2_bias->dims[1], W3->dims[0], W3->dims[1],
           BATCH_SIZE, HIDDEN3_SIZE);
    tl_tensor* h3 = tl_tensor_matmul(h2_bias, W3, NULL);
    if (!h3) {
        fprintf(stderr, "Layer 3 matmul failed!\n");
        return 1;
    }
    tl_tensor* h3_bias = add_bias_batch(h3, b3);
    tl_tensor_free_data_too(h3);
    relu(h3_bias);

    /* Layer 4 (output) */
    printf("  Layer 4: [%d,%d] @ [%d,%d] -> [%d,%d]\n",
           h3_bias->dims[0], h3_bias->dims[1], W4->dims[0], W4->dims[1],
           BATCH_SIZE, OUTPUT_SIZE);
    tl_tensor* output = tl_tensor_matmul(h3_bias, W4, NULL);
    if (!output) {
        fprintf(stderr, "Layer 4 matmul failed!\n");
        return 1;
    }
    tl_tensor* final_output = add_bias_batch(output, b4);
    tl_tensor_free_data_too(output);

    printf("\nOutput shape: [%d, %d]\n\n", final_output->dims[0], final_output->dims[1]);

    /* Print first 2 samples */
    printf("Output (first 2 samples):\n");
    for (int b = 0; b < 2; b++) {
        printf("  Sample %d: [", b);
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float val;
            int idx = b * OUTPUT_SIZE + i;
            TL_TENSOR_DATA_TO(final_output, idx, val, TL_FLOAT);
            printf("%.8f", val);
            if (i < OUTPUT_SIZE - 1) printf(", ");
        }
        printf("]\n");
    }

    /* Compare with expected */
    float* expected_data = load_weights("examples/deep_expected.bin", BATCH_SIZE * OUTPUT_SIZE);
    if (expected_data) {
        printf("\nExpected (first 2 samples):\n");
        for (int b = 0; b < 2; b++) {
            printf("  Sample %d: [", b);
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                int idx = b * OUTPUT_SIZE + i;
                printf("%.8f", expected_data[idx]);
                if (i < OUTPUT_SIZE - 1) printf(", ");
            }
            printf("]\n");
        }

        /* Check accuracy */
        float max_diff = 0;
        int total_elements = BATCH_SIZE * OUTPUT_SIZE;
        for (int i = 0; i < total_elements; i++) {
            float val;
            TL_TENSOR_DATA_TO(final_output, i, val, TL_FLOAT);
            float diff = fabsf(val - expected_data[i]);
            if (diff > max_diff) max_diff = diff;
        }

        printf("\nMax difference: %.10f\n", max_diff);
        if (max_diff < 1e-5) {
            printf("✓ Output matches NumPy reference!\n");
        } else {
            printf("✗ Output differs from NumPy reference (diff: %.10f)\n", max_diff);
        }

        free(expected_data);
    }

    /* Cleanup */
    tl_tensor_free(X);
    tl_tensor_free(W1); tl_tensor_free(b1);
    tl_tensor_free(W2); tl_tensor_free(b2);
    tl_tensor_free(W3); tl_tensor_free(b3);
    tl_tensor_free(W4); tl_tensor_free(b4);
    tl_tensor_free_data_too(h1_bias);
    tl_tensor_free_data_too(h2_bias);
    tl_tensor_free_data_too(h3_bias);
    tl_tensor_free_data_too(final_output);

    free(W1_data); free(b1_data);
    free(W2_data); free(b2_data);
    free(W3_data); free(b3_data);
    free(W4_data); free(b4_data);
    free(input_data);

    printf("\n============================================================\n");
    printf("Deep network inference complete!\n");
    printf("============================================================\n");

    return 0;
}
