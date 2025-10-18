/*
 * Diagnostic test for very deep networks
 * Tracks layer-by-layer statistics to identify issues
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tl_tensor.h"

void relu(tl_tensor* t) {
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        if (val < 0) val = 0;
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

void compute_stats(tl_tensor* t, float* min, float* max, float* mean, int* zero_count) {
    *min = INFINITY;
    *max = -INFINITY;
    float sum = 0;
    *zero_count = 0;

    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        sum += val;
        if (val < *min) *min = val;
        if (val > *max) *max = val;
        if (val == 0.0f) (*zero_count)++;
    }
    *mean = sum / t->len;
}

void init_random(tl_tensor* t, float scale) {
    for (int i = 0; i < t->len; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * scale;
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

int main() {
    printf("============================================================\n");
    printf("Deep Network Diagnostic\n");
    printf("============================================================\n\n");

    srand(42);

    const int DEPTH = 10;
    const int LAYER_SIZE = 64;

    printf("Network: 10 layers, %d neurons each\n\n", LAYER_SIZE);

    /* Initialize */
    tl_tensor* layers[DEPTH + 1];
    tl_tensor* weights[DEPTH];

    layers[0] = tl_tensor_zeros(2, (int[]){1, LAYER_SIZE}, TL_FLOAT);
    init_random(layers[0], 1.0f);

    for (int i = 0; i < DEPTH; i++) {
        weights[i] = tl_tensor_zeros(2, (int[]){LAYER_SIZE, LAYER_SIZE}, TL_FLOAT);
        init_random(weights[i], 0.1f);
    }

    /* Track statistics through layers */
    printf("Layer-by-layer statistics:\n");
    printf("%-8s %-12s %-12s %-12s %-12s\n", "Layer", "Min", "Max", "Mean", "Zeros");
    printf("----------------------------------------------------------------\n");

    float min, max, mean;
    int zero_count;

    compute_stats(layers[0], &min, &max, &mean, &zero_count);
    printf("%-8s %-12.6f %-12.6f %-12.6f %-12d\n", "Input", min, max, mean, zero_count);

    for (int i = 0; i < DEPTH; i++) {
        /* Before activation */
        layers[i + 1] = tl_tensor_matmul(layers[i], weights[i], NULL);
        if (!layers[i + 1]) {
            fprintf(stderr, "Layer %d matmul failed\n", i);
            return 1;
        }

        compute_stats(layers[i + 1], &min, &max, &mean, &zero_count);
        printf("L%-7d %-12.6f %-12.6f %-12.6f %-12d (before ReLU)\n",
               i + 1, min, max, mean, zero_count);

        /* After activation */
        relu(layers[i + 1]);
        compute_stats(layers[i + 1], &min, &max, &mean, &zero_count);
        printf("L%-7d %-12.6f %-12.6f %-12.6f %-12d (after ReLU)\n",
               i + 1, min, max, mean, zero_count);
    }

    printf("\n================================================================\n");
    printf("Analysis:\n");

    compute_stats(layers[DEPTH], &min, &max, &mean, &zero_count);
    float zero_percent = (float)zero_count / layers[DEPTH]->len * 100.0f;

    printf("Final layer: %.1f%% zeros (%d / %d elements)\n",
           zero_percent, zero_count, layers[DEPTH]->len);

    if (zero_percent > 99.0f) {
        printf("\n⚠️  WARNING: Dying ReLU detected!\n");
        printf("   - Most values became negative and were zeroed by ReLU\n");
        printf("   - This is common in very deep networks without:\n");
        printf("     • Batch normalization\n");
        printf("     • Residual connections (skip connections)\n");
        printf("     • Better weight initialization (Xavier/He)\n");
        printf("     • Alternative activations (LeakyReLU, ELU)\n");
    } else if (zero_percent > 50.0f) {
        printf("\n⚠️  CAUTION: High sparsity detected (%.1f%% zeros)\n", zero_percent);
    } else {
        printf("\n✓ Network appears healthy\n");
    }

    /* Cleanup */
    for (int i = 0; i <= DEPTH; i++) {
        tl_tensor_free_data_too(layers[i]);
    }
    for (int i = 0; i < DEPTH; i++) {
        tl_tensor_free_data_too(weights[i]);
    }

    printf("================================================================\n");

    return 0;
}
