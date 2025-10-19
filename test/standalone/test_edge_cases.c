/*
 * Tofu Edge Case Test Suite
 *
 * Purpose: Document behavior for edge cases and catch regressions.
 * Note: Some tests may crash (assert failures) - this is expected and documents limitations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "tl_graph.h"
#include "tl_tensor.h"

#define TEST_PASS 0
#define TEST_FAIL 1
#define TEST_SKIP -1

/* Track test results */
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;
int tests_skipped = 0;

void report_test(const char* name, int result) {
    tests_run++;
    printf("  ");
    if (result == TEST_PASS) {
        printf("✓ %s\n", name);
        tests_passed++;
    } else if (result == TEST_FAIL) {
        printf("✗ %s (FAILED)\n", name);
        tests_failed++;
    } else {
        printf("⊘ %s (SKIPPED)\n", name);
        tests_skipped++;
    }
}

/* ============================================================
 * Category 1: Zero Value Tests
 * ============================================================ */

int test_zero_matrix_matmul() {
    /* Test: Matmul with all-zero matrices should produce zero result */
    float A_data[4] = {0, 0, 0, 0};  /* 2x2 zero matrix */
    float B_data[4] = {1, 2, 3, 4};  /* 2x2 non-zero matrix */

    tl_tensor* A = tl_tensor_create(A_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* B = tl_tensor_create(B_data, 2, (int[]){2, 2}, TL_FLOAT);

    tl_tensor* C = tl_tensor_matmul(A, B, NULL);

    /* Result should be all zeros */
    int all_zero = 1;
    for (int i = 0; i < 4; i++) {
        float val;
        TL_TENSOR_DATA_TO(C, i, val, TL_FLOAT);
        if (fabsf(val) > 1e-7f) {
            all_zero = 0;
            break;
        }
    }

    tl_tensor_free(A);
    tl_tensor_free(B);
    tl_tensor_free_data_too(C);

    return all_zero ? TEST_PASS : TEST_FAIL;
}

int test_zero_gradient() {
    /* Test: Backward pass with zero gradients */
    tl_graph* g = tl_graph_create();

    float x_data[4] = {1, 2, 3, 4};
    float W_data[4] = {1, 0, 0, 1};

    tl_tensor* t_x = tl_tensor_create(x_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* t_W = tl_tensor_create(W_data, 2, (int[]){2, 2}, TL_FLOAT);

    tl_graph_node* x = tl_graph_input(g, t_x);
    tl_graph_node* W = tl_graph_param(g, t_W);
    tl_graph_node* y = tl_graph_matmul(g, x, W);

    /* Manually set output gradient to zero */
    if (y->grad) {
        for (int i = 0; i < y->grad->len; i++) {
            float zero = 0.0f;
            TL_TENSOR_DATA_FROM(y->grad, i, zero, TL_FLOAT);
        }
    }

    tl_graph_backward(g, y);

    /* Gradients should be zero or NULL */
    int pass = (W->grad == NULL || W->grad->len == 0);

    tl_tensor_free(t_x);
    tl_tensor_free(t_W);
    tl_graph_free(g);

    return pass ? TEST_PASS : TEST_FAIL;
}

/* ============================================================
 * Category 2: NaN/Inf Tests
 * ============================================================ */

int test_nan_detection() {
    /* Test: NaN propagation through matmul */
    float A_data[4] = {NAN, 1, 2, 3};
    float B_data[4] = {1, 2, 3, 4};

    tl_tensor* A = tl_tensor_create(A_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* B = tl_tensor_create(B_data, 2, (int[]){2, 2}, TL_FLOAT);

    tl_tensor* C = tl_tensor_matmul(A, B, NULL);

    /* Check if NaN propagated */
    int has_nan = 0;
    for (int i = 0; i < C->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(C, i, val, TL_FLOAT);
        if (isnan(val)) {
            has_nan = 1;
            break;
        }
    }

    tl_tensor_free(A);
    tl_tensor_free(B);
    tl_tensor_free_data_too(C);

    /* Document behavior: NaN propagates */
    return has_nan ? TEST_PASS : TEST_FAIL;
}

int test_inf_handling() {
    /* Test: Infinity handling in operations */
    float A_data[4] = {INFINITY, 1, 2, 3};
    float B_data[4] = {1, 2, 3, 4};

    tl_tensor* A = tl_tensor_create(A_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* B = tl_tensor_create(B_data, 2, (int[]){2, 2}, TL_FLOAT);

    tl_tensor* C = tl_tensor_matmul(A, B, NULL);

    /* Check if Inf exists in output */
    int has_inf = 0;
    for (int i = 0; i < C->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(C, i, val, TL_FLOAT);
        if (isinf(val)) {
            has_inf = 1;
            break;
        }
    }

    tl_tensor_free(A);
    tl_tensor_free(B);
    tl_tensor_free_data_too(C);

    /* Document: Inf propagates */
    return has_inf ? TEST_PASS : TEST_FAIL;
}

/* ============================================================
 * Category 3: Extreme Value Tests
 * ============================================================ */

int test_large_values() {
    /* Test: Very large values (near FLT_MAX) */
    float large = FLT_MAX / 10.0f;  /* Avoid overflow */
    float A_data[4] = {large, 0, 0, large};
    float B_data[4] = {1, 0, 0, 1};  /* Identity-like */

    tl_tensor* A = tl_tensor_create(A_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* B = tl_tensor_create(B_data, 2, (int[]){2, 2}, TL_FLOAT);

    tl_tensor* C = tl_tensor_matmul(A, B, NULL);

    /* Check result is still finite */
    int all_finite = 1;
    for (int i = 0; i < C->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(C, i, val, TL_FLOAT);
        if (!isfinite(val)) {
            all_finite = 0;
            break;
        }
    }

    tl_tensor_free(A);
    tl_tensor_free(B);
    tl_tensor_free_data_too(C);

    return all_finite ? TEST_PASS : TEST_FAIL;
}

int test_small_values() {
    /* Test: Very small values (near zero, denormals) */
    float small = FLT_MIN * 10.0f;
    float A_data[4] = {small, small, small, small};
    float B_data[4] = {1, 1, 1, 1};

    tl_tensor* A = tl_tensor_create(A_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* B = tl_tensor_create(B_data, 2, (int[]){2, 2}, TL_FLOAT);

    tl_tensor* C = tl_tensor_matmul(A, B, NULL);

    /* Result should be finite (possibly zero due to underflow) */
    int all_finite = 1;
    for (int i = 0; i < C->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(C, i, val, TL_FLOAT);
        if (!isfinite(val)) {
            all_finite = 0;
            break;
        }
    }

    tl_tensor_free(A);
    tl_tensor_free(B);
    tl_tensor_free_data_too(C);

    return all_finite ? TEST_PASS : TEST_FAIL;
}

/* ============================================================
 * Category 4: Dimension Edge Cases
 * ============================================================ */

int test_1x1_matrix() {
    /* Test: 1x1 matrix operations (edge case for many algorithms) */
    float A_data[1] = {5.0f};
    float B_data[1] = {3.0f};

    tl_tensor* A = tl_tensor_create(A_data, 2, (int[]){1, 1}, TL_FLOAT);
    tl_tensor* B = tl_tensor_create(B_data, 2, (int[]){1, 1}, TL_FLOAT);

    tl_tensor* C = tl_tensor_matmul(A, B, NULL);

    float result;
    TL_TENSOR_DATA_TO(C, 0, result, TL_FLOAT);

    int correct = (fabsf(result - 15.0f) < 1e-5f);  /* 5 * 3 = 15 */

    tl_tensor_free(A);
    tl_tensor_free(B);
    tl_tensor_free_data_too(C);

    return correct ? TEST_PASS : TEST_FAIL;
}

/* ============================================================
 * Main Test Runner
 * ============================================================ */

int main() {
    printf("============================================================\n");
    printf("Tofu Edge Case Test Suite\n");
    printf("============================================================\n");
    printf("Purpose: Document behavior and catch regressions\n");
    printf("Note: Some failures are expected (known limitations)\n");
    printf("============================================================\n\n");

    /* Category 1: Zero Values */
    printf("Category: Zero Value Tests\n");
    report_test("test_zero_matrix_matmul", test_zero_matrix_matmul());
    report_test("test_zero_gradient", test_zero_gradient());
    printf("\n");

    /* Category 2: NaN/Inf */
    printf("Category: NaN/Inf Handling\n");
    report_test("test_nan_detection", test_nan_detection());
    report_test("test_inf_handling", test_inf_handling());
    printf("\n");

    /* Category 3: Extreme Values */
    printf("Category: Extreme Value Tests\n");
    report_test("test_large_values", test_large_values());
    report_test("test_small_values", test_small_values());
    printf("\n");

    /* Category 4: Dimension Edge Cases */
    printf("Category: Dimension Edge Cases\n");
    report_test("test_1x1_matrix", test_1x1_matrix());
    printf("\n");

    /* Summary */
    printf("============================================================\n");
    printf("Test Summary\n");
    printf("============================================================\n");
    printf("Total:   %d tests\n", tests_run);
    printf("Passed:  %d tests\n", tests_passed);
    printf("Failed:  %d tests\n", tests_failed);
    printf("Skipped: %d tests\n", tests_skipped);
    printf("============================================================\n");

    if (tests_failed > 0) {
        printf("\nNote: Failures document known limitations for v1.1.0 fixes\n");
    }

    return (tests_failed > 0) ? 1 : 0;
}
