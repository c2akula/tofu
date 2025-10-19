/*
 * Simple 2-layer neural network inference example
 * Demonstrates real-world usage of Tofu tensor operations
 *
 * Network: input(4) -> hidden(8) -> output(3)
 * Operations: matmul, element-wise add, ReLU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tofu_tensor.h"

/* Load binary weights from file */
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
        fprintf(stderr, "Error: Expected %d floats, read %zu\n", size, read);
        free(data);
        return NULL;
    }

    return data;
}

/* ReLU activation: max(0, x) */
void relu(tofu_tensor* t) {
    for (int i = 0; i < t->len; i++) {
        float val;
        TOFU_TENSOR_DATA_TO(t, i, val, TOFU_FLOAT);
        if (val < 0) val = 0;
        TOFU_TENSOR_DATA_FROM(t, i, val, TOFU_FLOAT);
    }
}

/* Add bias vector to each row of a matrix */
tofu_tensor* add_bias(tofu_tensor* x, tofu_tensor* bias) {
    /* x: [batch, features], bias: [features] */
    /* Using element-wise broadcast would be ideal, but for simplicity: */
    tofu_tensor* result = tofu_tensor_clone(x);

    for (int i = 0; i < x->dims[0]; i++) {  /* for each batch */
        for (int j = 0; j < x->dims[1]; j++) {  /* for each feature */
            float x_val, b_val;
            int idx = i * x->dims[1] + j;
            TOFU_TENSOR_DATA_TO(result, idx, x_val, TOFU_FLOAT);
            TOFU_TENSOR_DATA_TO(bias, j, b_val, TOFU_FLOAT);
            x_val += b_val;
            TOFU_TENSOR_DATA_FROM(result, idx, x_val, TOFU_FLOAT);
        }
    }

    return result;
}

int main() {
    printf("============================================================\n");
    printf("Neural Network Inference Example\n");
    printf("============================================================\n\n");

    /* Network architecture */
    const int INPUT_SIZE = 4;
    const int HIDDEN_SIZE = 8;
    const int OUTPUT_SIZE = 3;

    /* Load weights */
    printf("Loading weights...\n");
    float* W1_data = load_weights("examples/W1.bin", INPUT_SIZE * HIDDEN_SIZE);
    float* b1_data = load_weights("examples/b1.bin", HIDDEN_SIZE);
    float* W2_data = load_weights("examples/W2.bin", HIDDEN_SIZE * OUTPUT_SIZE);
    float* b2_data = load_weights("examples/b2.bin", OUTPUT_SIZE);
    float* input_data = load_weights("examples/test_input.bin", INPUT_SIZE);

    if (!W1_data || !b1_data || !W2_data || !b2_data || !input_data) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    /* Create tensors */
    tofu_tensor* X = tofu_tensor_create(input_data, 2, (int[]){1, INPUT_SIZE}, TOFU_FLOAT);
    tofu_tensor* W1 = tofu_tensor_create(W1_data, 2, (int[]){INPUT_SIZE, HIDDEN_SIZE}, TOFU_FLOAT);
    tofu_tensor* b1 = tofu_tensor_create(b1_data, 1, (int[]){HIDDEN_SIZE}, TOFU_FLOAT);
    tofu_tensor* W2 = tofu_tensor_create(W2_data, 2, (int[]){HIDDEN_SIZE, OUTPUT_SIZE}, TOFU_FLOAT);
    tofu_tensor* b2 = tofu_tensor_create(b2_data, 1, (int[]){OUTPUT_SIZE}, TOFU_FLOAT);

    printf("Input shape: [%d, %d]\n", X->dims[0], X->dims[1]);
    printf("W1 shape: [%d, %d]\n", W1->dims[0], W1->dims[1]);
    printf("W2 shape: [%d, %d]\n\n", W2->dims[0], W2->dims[1]);

    /* Forward pass: Layer 1 */
    printf("Layer 1: X @ W1 + b1\n");
    tofu_tensor* h1 = tofu_tensor_matmul(X, W1, NULL);
    printf("  After matmul: [%d, %d]\n", h1->dims[0], h1->dims[1]);

    tofu_tensor* h1_bias = add_bias(h1, b1);
    tofu_tensor_free_data_too(h1);
    printf("  After bias: [%d, %d]\n", h1_bias->dims[0], h1_bias->dims[1]);

    relu(h1_bias);
    printf("  After ReLU: [%d, %d]\n\n", h1_bias->dims[0], h1_bias->dims[1]);

    /* Forward pass: Layer 2 */
    printf("Layer 2: hidden @ W2 + b2\n");
    tofu_tensor* output = tofu_tensor_matmul(h1_bias, W2, NULL);
    printf("  After matmul: [%d, %d]\n", output->dims[0], output->dims[1]);

    tofu_tensor* final_output = add_bias(output, b2);
    tofu_tensor_free_data_too(output);
    printf("  After bias: [%d, %d]\n\n", final_output->dims[0], final_output->dims[1]);

    /* Print results */
    printf("Output:\n[");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float val;
        TOFU_TENSOR_DATA_TO(final_output, i, val, TOFU_FLOAT);
        printf("%.6f", val);
        if (i < OUTPUT_SIZE - 1) printf(", ");
    }
    printf("]\n\n");

    /* Load and compare with expected output */
    float* expected_data = load_weights("examples/expected_output.bin", OUTPUT_SIZE);
    if (expected_data) {
        printf("Expected output:\n[");
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            printf("%.6f", expected_data[i]);
            if (i < OUTPUT_SIZE - 1) printf(", ");
        }
        printf("]\n\n");

        /* Check accuracy */
        float max_diff = 0;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float val;
            TOFU_TENSOR_DATA_TO(final_output, i, val, TOFU_FLOAT);
            float diff = fabsf(val - expected_data[i]);
            if (diff > max_diff) max_diff = diff;
        }

        printf("Max difference: %.8f\n", max_diff);
        if (max_diff < 1e-5) {
            printf("✓ Output matches NumPy reference!\n");
        } else {
            printf("✗ Output differs from NumPy reference\n");
        }

        free(expected_data);
    }

    /* Cleanup */
    tofu_tensor_free(X);
    tofu_tensor_free(W1);
    tofu_tensor_free(b1);
    tofu_tensor_free(W2);
    tofu_tensor_free(b2);
    tofu_tensor_free_data_too(h1_bias);
    tofu_tensor_free_data_too(final_output);

    free(W1_data);
    free(b1_data);
    free(W2_data);
    free(b2_data);
    free(input_data);

    printf("\n============================================================\n");
    printf("Inference complete!\n");
    printf("============================================================\n");

    return 0;
}
