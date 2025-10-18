/*
 * Stress test with larger input sizes
 * Tests memory handling, numerical stability with large tensors
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "tl_tensor.h"

/* Simple ReLU */
void relu(tl_tensor* t) {
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        if (val < 0) val = 0;
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

/* Check for NaN/Inf */
int has_invalid_values(tl_tensor* t) {
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        if (isnan(val) || isinf(val)) {
            return 1;
        }
    }
    return 0;
}

/* Initialize tensor with random values */
void init_random(tl_tensor* t, float scale) {
    for (int i = 0; i < t->len; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * scale;
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

int main() {
    printf("============================================================\n");
    printf("Stress Test with Large Input Sizes\n");
    printf("============================================================\n\n");

    srand(42);

    /* Test 1: Large batch size */
    printf("Test 1: Large batch processing\n");
    const int BATCH_SIZE = 128;
    const int INPUT_SIZE = 256;
    const int HIDDEN_SIZE = 512;
    const int OUTPUT_SIZE = 128;

    printf("  Batch: %d samples\n", BATCH_SIZE);
    printf("  Input: %d features\n", INPUT_SIZE);
    printf("  Hidden: %d neurons\n", HIDDEN_SIZE);
    printf("  Output: %d classes\n\n", OUTPUT_SIZE);

    /* Create tensors */
    printf("  Allocating tensors...\n");
    tl_tensor* X = tl_tensor_zeros(2, (int[]){BATCH_SIZE, INPUT_SIZE}, TL_FLOAT);
    tl_tensor* W1 = tl_tensor_zeros(2, (int[]){INPUT_SIZE, HIDDEN_SIZE}, TL_FLOAT);
    tl_tensor* W2 = tl_tensor_zeros(2, (int[]){HIDDEN_SIZE, OUTPUT_SIZE}, TL_FLOAT);

    if (!X || !W1 || !W2) {
        fprintf(stderr, "  ✗ Failed to allocate tensors\n");
        return 1;
    }
    printf("  ✓ Allocated %.2f MB\n",
           (X->len + W1->len + W2->len) * sizeof(float) / 1024.0 / 1024.0);

    /* Initialize */
    printf("  Initializing with random values...\n");
    init_random(X, 2.0f);
    init_random(W1, 0.1f);
    init_random(W2, 0.1f);

    /* Forward pass */
    printf("  Forward pass: [%d,%d] @ [%d,%d] -> [%d,%d]\n",
           X->dims[0], X->dims[1], W1->dims[0], W1->dims[1],
           BATCH_SIZE, HIDDEN_SIZE);

    clock_t start = clock();
    tl_tensor* h1 = tl_tensor_matmul(X, W1, NULL);
    clock_t end = clock();

    if (!h1) {
        fprintf(stderr, "  ✗ Layer 1 matmul failed\n");
        return 1;
    }

    double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  ✓ Layer 1 completed in %.2f ms\n", time_ms);

    /* Check for numerical issues */
    if (has_invalid_values(h1)) {
        printf("  ✗ Found NaN/Inf in layer 1 output\n");
        return 1;
    }
    printf("  ✓ No NaN/Inf detected\n");

    relu(h1);

    printf("  Forward pass: [%d,%d] @ [%d,%d] -> [%d,%d]\n",
           h1->dims[0], h1->dims[1], W2->dims[0], W2->dims[1],
           BATCH_SIZE, OUTPUT_SIZE);

    start = clock();
    tl_tensor* output = tl_tensor_matmul(h1, W2, NULL);
    end = clock();

    if (!output) {
        fprintf(stderr, "  ✗ Layer 2 matmul failed\n");
        return 1;
    }

    time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  ✓ Layer 2 completed in %.2f ms\n", time_ms);

    if (has_invalid_values(output)) {
        printf("  ✗ Found NaN/Inf in output\n");
        return 1;
    }
    printf("  ✓ Output is valid\n");

    /* Compute statistics */
    float sum = 0, min = INFINITY, max = -INFINITY;
    for (int i = 0; i < output->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(output, i, val, TL_FLOAT);
        sum += val;
        if (val < min) min = val;
        if (val > max) max = val;
    }
    float mean = sum / output->len;
    printf("  Output stats: min=%.6f, max=%.6f, mean=%.6f\n\n", min, max, mean);

    /* Cleanup */
    tl_tensor_free_data_too(X);
    tl_tensor_free_data_too(W1);
    tl_tensor_free_data_too(W2);
    tl_tensor_free_data_too(h1);
    tl_tensor_free_data_too(output);

    /* Test 2: Very deep network (memory stress) */
    printf("Test 2: Very deep network (10 layers)\n");
    const int DEPTH = 10;
    const int LAYER_SIZE = 64;

    printf("  Network: 10 layers, %d neurons each\n", LAYER_SIZE);
    printf("  Total parameters: %d\n\n", DEPTH * LAYER_SIZE * LAYER_SIZE);

    tl_tensor* layers[DEPTH + 1];
    layers[0] = tl_tensor_zeros(2, (int[]){1, LAYER_SIZE}, TL_FLOAT);
    init_random(layers[0], 1.0f);

    printf("  Creating %d weight matrices...\n", DEPTH);
    tl_tensor* weights[DEPTH];
    for (int i = 0; i < DEPTH; i++) {
        weights[i] = tl_tensor_zeros(2, (int[]){LAYER_SIZE, LAYER_SIZE}, TL_FLOAT);
        if (!weights[i]) {
            fprintf(stderr, "  ✗ Failed to allocate weight matrix %d\n", i);
            return 1;
        }
        init_random(weights[i], 0.1f);
    }
    printf("  ✓ Allocated %.2f MB for weights\n",
           DEPTH * LAYER_SIZE * LAYER_SIZE * sizeof(float) / 1024.0 / 1024.0);

    printf("  Forward pass through %d layers...\n", DEPTH);
    start = clock();
    for (int i = 0; i < DEPTH; i++) {
        layers[i + 1] = tl_tensor_matmul(layers[i], weights[i], NULL);
        if (!layers[i + 1]) {
            fprintf(stderr, "  ✗ Layer %d matmul failed\n", i);
            return 1;
        }
        relu(layers[i + 1]);

        if (has_invalid_values(layers[i + 1])) {
            fprintf(stderr, "  ✗ NaN/Inf detected at layer %d\n", i);
            return 1;
        }
    }
    end = clock();
    time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("  ✓ Completed in %.2f ms\n", time_ms);

    /* Final output stats */
    tl_tensor* final = layers[DEPTH];
    sum = 0;
    min = INFINITY;
    max = -INFINITY;
    for (int i = 0; i < final->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(final, i, val, TL_FLOAT);
        sum += val;
        if (val < min) min = val;
        if (val > max) max = val;
    }
    mean = sum / final->len;
    printf("  Final output: min=%.6f, max=%.6f, mean=%.6f\n", min, max, mean);

    /* Cleanup */
    for (int i = 0; i <= DEPTH; i++) {
        tl_tensor_free_data_too(layers[i]);
    }
    for (int i = 0; i < DEPTH; i++) {
        tl_tensor_free_data_too(weights[i]);
    }

    printf("\n============================================================\n");
    printf("Stress tests complete!\n");
    printf("✓ Large batch processing works\n");
    printf("✓ Very deep networks work\n");
    printf("✓ No memory or numerical issues detected\n");
    printf("============================================================\n");

    return 0;
}
