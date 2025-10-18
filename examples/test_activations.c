/*
 * Test different activation functions
 * Tests numerical stability with different ranges
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tl_tensor.h"

/* Activation functions */
void relu(tl_tensor* t) {
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        if (val < 0) val = 0;
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

void sigmoid(tl_tensor* t) {
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        val = 1.0f / (1.0f + expf(-val));
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

void tanh_activation(tl_tensor* t) {
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        val = tanhf(val);
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

void softmax(tl_tensor* t) {
    /* Simple softmax for 1D tensor */
    float max_val = -INFINITY;
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        if (val > max_val) max_val = val;
    }

    float sum = 0;
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        val = expf(val - max_val);  /* Numerical stability trick */
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
        sum += val;
    }

    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        val /= sum;
        TL_TENSOR_DATA_FROM(t, i, val, TL_FLOAT);
    }
}

void print_tensor(const char* name, tl_tensor* t) {
    printf("%s: [", name);
    for (int i = 0; i < t->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t, i, val, TL_FLOAT);
        printf("%.6f", val);
        if (i < t->len - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    printf("============================================================\n");
    printf("Testing Activation Functions\n");
    printf("============================================================\n\n");

    /* Test data with various ranges */
    float test_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 10.0f, -10.0f, 100.0f};
    int n = sizeof(test_data) / sizeof(test_data[0]);

    /* Test ReLU */
    printf("ReLU (max(0, x)):\n");
    tl_tensor* t_relu = tl_tensor_create(test_data, 1, (int[]){n}, TL_FLOAT);
    print_tensor("  Input ", t_relu);
    relu(t_relu);
    print_tensor("  Output", t_relu);
    printf("\n");

    /* Test Sigmoid */
    printf("Sigmoid (1 / (1 + e^-x)):\n");
    tl_tensor* t_sigmoid = tl_tensor_create(test_data, 1, (int[]){n}, TL_FLOAT);
    print_tensor("  Input ", t_sigmoid);
    sigmoid(t_sigmoid);
    print_tensor("  Output", t_sigmoid);
    printf("\n");

    /* Test Tanh */
    printf("Tanh:\n");
    tl_tensor* t_tanh = tl_tensor_create(test_data, 1, (int[]){n}, TL_FLOAT);
    print_tensor("  Input ", t_tanh);
    tanh_activation(t_tanh);
    print_tensor("  Output", t_tanh);
    printf("\n");

    /* Test Softmax */
    printf("Softmax (normalized exponentials):\n");
    float softmax_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    tl_tensor* t_softmax = tl_tensor_create(softmax_data, 1, (int[]){5}, TL_FLOAT);
    print_tensor("  Input ", t_softmax);
    softmax(t_softmax);
    print_tensor("  Output", t_softmax);

    /* Verify softmax sums to 1 */
    float sum = 0;
    for (int i = 0; i < t_softmax->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t_softmax, i, val, TL_FLOAT);
        sum += val;
    }
    printf("  Sum: %.10f (should be 1.0)\n", sum);
    printf("\n");

    /* Test numerical stability with extreme values */
    printf("Numerical Stability Test:\n");
    float extreme_data[] = {-1000.0f, 1000.0f, 0.00001f, -0.00001f};
    tl_tensor* t_extreme = tl_tensor_create(extreme_data, 1, (int[]){4}, TL_FLOAT);
    print_tensor("  Extreme input", t_extreme);
    sigmoid(t_extreme);
    print_tensor("  After sigmoid", t_extreme);
    printf("  Checking for NaN/Inf...\n");

    int has_nan_inf = 0;
    for (int i = 0; i < t_extreme->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(t_extreme, i, val, TL_FLOAT);
        if (isnan(val) || isinf(val)) {
            printf("  ✗ Found NaN/Inf at index %d: %.6f\n", i, val);
            has_nan_inf = 1;
        }
    }
    if (!has_nan_inf) {
        printf("  ✓ No NaN/Inf detected\n");
    }

    /* Cleanup */
    tl_tensor_free(t_relu);
    tl_tensor_free(t_sigmoid);
    tl_tensor_free(t_tanh);
    tl_tensor_free(t_softmax);
    tl_tensor_free(t_extreme);

    printf("\n============================================================\n");
    printf("Activation function tests complete!\n");
    printf("============================================================\n");

    return 0;
}
