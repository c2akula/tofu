/*
 * Test new operations: softmax, sum/mean reduce, layer norm
 */

#include <stdio.h>
#include <math.h>
#include "tl_tensor.h"

#define EPSILON 1e-5

int test_sumreduce() {
    printf("Testing sumreduce...\n");

    /* Test data: [2, 3] */
    float data[] = {1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f};
    tl_tensor* t = tl_tensor_create(data, 2, (int[]){2, 3}, TL_FLOAT);

    /* Sum along axis 1 (columns): expect [6, 15] */
    tl_tensor* sum = tl_tensor_sumreduce(t, NULL, 1);
    float expected[] = {6.0f, 15.0f};

    for (int i = 0; i < 2; i++) {
        float val;
        TL_TENSOR_DATA_TO(sum, i, val, TL_FLOAT);
        if (fabsf(val - expected[i]) > EPSILON) {
            printf("  ✗ FAILED: expected %.2f, got %.2f\n", expected[i], val);
            return 0;
        }
    }

    tl_tensor_free(t);  /* t uses stack data, don't free data */
    tl_tensor_free_data_too(sum);
    printf("  ✓ PASSED\n\n");
    return 1;
}

int test_meanreduce() {
    printf("Testing meanreduce...\n");

    /* Test data: [2, 3] */
    float data[] = {1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f};
    tl_tensor* t = tl_tensor_create(data, 2, (int[]){2, 3}, TL_FLOAT);

    /* Mean along axis 1 (columns): expect [2, 5] */
    tl_tensor* mean = tl_tensor_meanreduce(t, NULL, 1);
    float expected[] = {2.0f, 5.0f};

    for (int i = 0; i < 2; i++) {
        float val;
        TL_TENSOR_DATA_TO(mean, i, val, TL_FLOAT);
        if (fabsf(val - expected[i]) > EPSILON) {
            printf("  ✗ FAILED: expected %.2f, got %.2f\n", expected[i], val);
            return 0;
        }
    }

    tl_tensor_free(t);  /* t uses stack data, don't free data */
    tl_tensor_free_data_too(mean);
    printf("  ✓ PASSED\n\n");
    return 1;
}

int test_softmax() {
    printf("Testing softmax...\n");

    /* Test data: [2, 3] */
    float data[] = {1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f};
    tl_tensor* t = tl_tensor_create(data, 2, (int[]){2, 3}, TL_FLOAT);

    /* Softmax along axis 1 */
    tl_tensor* softmax = tl_tensor_softmax(t, NULL, 1);

    /* Check that each row sums to 1 */
    for (int row = 0; row < 2; row++) {
        float sum = 0.0f;
        for (int col = 0; col < 3; col++) {
            float val;
            TL_TENSOR_DATA_TO(softmax, row * 3 + col, val, TL_FLOAT);
            sum += val;
        }
        if (fabsf(sum - 1.0f) > EPSILON) {
            printf("  ✗ FAILED: row %d sum is %.6f, expected 1.0\n", row, sum);
            return 0;
        }
    }

    /* Check numerical stability (no NaN/Inf) */
    for (int i = 0; i < softmax->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(softmax, i, val, TL_FLOAT);
        if (isnan(val) || isinf(val)) {
            printf("  ✗ FAILED: found NaN/Inf at index %d\n", i);
            return 0;
        }
    }

    tl_tensor_free(t);  /* t uses stack data, don't free data */
    tl_tensor_free_data_too(softmax);
    printf("  ✓ PASSED\n\n");
    return 1;
}

int test_layer_norm() {
    printf("Testing layer_norm...\n");

    /* Test data: [2, 4] */
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f};
    tl_tensor* t = tl_tensor_create(data, 2, (int[]){2, 4}, TL_FLOAT);

    /* Gamma and beta for scaling/shifting */
    float gamma_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    tl_tensor* gamma = tl_tensor_create(gamma_data, 1, (int[]){4}, TL_FLOAT);
    tl_tensor* beta = tl_tensor_create(beta_data, 1, (int[]){4}, TL_FLOAT);

    /* Layer norm along axis 1 */
    tl_tensor* normed = tl_tensor_layer_norm(t, NULL, gamma, beta, 1, 1e-5);

    /* Check that each row has mean ≈ 0 and std ≈ 1 */
    for (int row = 0; row < 2; row++) {
        float sum = 0.0f;
        for (int col = 0; col < 4; col++) {
            float val;
            TL_TENSOR_DATA_TO(normed, row * 4 + col, val, TL_FLOAT);
            sum += val;
        }
        float mean = sum / 4.0f;

        if (fabsf(mean) > EPSILON) {
            printf("  ✗ FAILED: row %d mean is %.6f, expected ~0.0\n", row, mean);
            return 0;
        }

        /* Check std */
        float var_sum = 0.0f;
        for (int col = 0; col < 4; col++) {
            float val;
            TL_TENSOR_DATA_TO(normed, row * 4 + col, val, TL_FLOAT);
            var_sum += val * val;
        }
        float std = sqrtf(var_sum / 4.0f);

        if (fabsf(std - 1.0f) > EPSILON) {
            printf("  ✗ FAILED: row %d std is %.6f, expected ~1.0\n", row, std);
            return 0;
        }
    }

    tl_tensor_free(t);  /* t uses stack data */
    tl_tensor_free(gamma);  /* gamma uses stack data */
    tl_tensor_free(beta);  /* beta uses stack data */
    tl_tensor_free_data_too(normed);
    printf("  ✓ PASSED\n\n");
    return 1;
}

int main() {
    printf("============================================================\n");
    printf("Testing New Operations\n");
    printf("============================================================\n\n");

    int passed = 0;
    int total = 4;

    passed += test_sumreduce();
    passed += test_meanreduce();
    passed += test_softmax();
    passed += test_layer_norm();

    printf("============================================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("============================================================\n");

    return (passed == total) ? 0 : 1;
}
