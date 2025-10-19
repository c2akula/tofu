/*
 * Test Suite: Broadcasting Gradient Computation
 *
 * Purpose: Verify that gradients are correctly reduced when broadcasting occurs
 * in element-wise operations (ADD, MUL).
 *
 * Broadcasting Rule: When forward pass broadcasts (e.g., [3,1] + [3,4] -> [3,4]),
 * backward pass must reduce gradient back to original shape by summing over
 * broadcast dimensions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tofu_graph.h"
#include "tofu_tensor.h"

#define TOLERANCE 1e-5f

#define TEST_PASS 0
#define TEST_FAIL 1

/* Track test results */
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

void report_test(const char* name, int result) {
    tests_run++;
    printf("  ");
    if (result == TEST_PASS) {
        printf("✓ %s\n", name);
        tests_passed++;
    } else {
        printf("✗ %s (FAILED)\n", name);
        tests_failed++;
    }
}

/* ============================================================
 * Test 1: MUL with Broadcasting - Simple Case
 * ============================================================
 * Forward: x[3,1] * y[3,4] -> z[3,4]
 * Backward: grad_x should be [3,1] (sum over dim 1)
 */
int test_mul_broadcast_simple() {
    tofu_graph* g = tofu_graph_create();

    /* Create tensors */
    float x_data[3] = {2.0f, 3.0f, 4.0f};  /* Will be reshaped to [3,1] */
    float y_data[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f};  /* [3,4] */

    tofu_tensor* t_x = tofu_tensor_create(x_data, 2, (int[]){3, 1}, TOFU_FLOAT);
    tofu_tensor* t_y = tofu_tensor_create(y_data, 2, (int[]){3, 4}, TOFU_FLOAT);

    /* Build graph: z = x * y */
    tofu_graph_node* x = tofu_graph_param(g, t_x);
    tofu_graph_node* y = tofu_graph_param(g, t_y);
    tofu_graph_node* z = tofu_graph_mul(g, x, y);

    /* Initialize gradient at output (all ones) */
    tofu_graph_zero_grad(g);
    tofu_graph_backward(g, z);

    /* Check gradient shape */
    if (!x->grad || x->grad->ndim != 2 || x->grad->dims[0] != 3 || x->grad->dims[1] != 1) {
        printf("      Error: grad_x shape incorrect\n");
        tofu_tensor_free(t_x);
        tofu_tensor_free(t_y);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    /* Check gradient values
     * grad_x[i,0] = sum(grad_z[i,:] * y[i,:]) = sum(1.0 * y[i,:])
     * grad_x[0,0] = 1+2+3+4 = 10
     * grad_x[1,0] = 5+6+7+8 = 26
     * grad_x[2,0] = 9+10+11+12 = 42
     */
    float expected_grad_x[3] = {10.0f, 26.0f, 42.0f};
    for (int i = 0; i < 3; i++) {
        float grad_val;
        TOFU_TENSOR_DATA_TO(x->grad, i, grad_val, TOFU_FLOAT);
        if (fabsf(grad_val - expected_grad_x[i]) > TOLERANCE) {
            printf("      Error: grad_x[%d,0] = %f, expected %f\n",
                   i, grad_val, expected_grad_x[i]);
            tofu_tensor_free(t_x);
            tofu_tensor_free(t_y);
            tofu_graph_free(g);
            return TEST_FAIL;
        }
    }

    tofu_tensor_free(t_x);
    tofu_tensor_free(t_y);
    tofu_graph_free(g);
    return TEST_PASS;
}

/* ============================================================
 * Test 2: ADD with Broadcasting - Scalar + Matrix
 * ============================================================
 * Forward: x[1] + y[3,4] -> z[3,4]
 * Backward: grad_x should be [1] (sum all elements)
 */
int test_add_broadcast_scalar() {
    tofu_graph* g = tofu_graph_create();

    /* Create tensors */
    float x_data[1] = {5.0f};  /* Scalar [1] */
    float y_data[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f};  /* [3,4] */

    tofu_tensor* t_x = tofu_tensor_create(x_data, 1, (int[]){1}, TOFU_FLOAT);
    tofu_tensor* t_y = tofu_tensor_create(y_data, 2, (int[]){3, 4}, TOFU_FLOAT);

    /* Build graph: z = x + y */
    tofu_graph_node* x = tofu_graph_param(g, t_x);
    tofu_graph_node* y = tofu_graph_param(g, t_y);
    tofu_graph_node* z = tofu_graph_add(g, x, y);

    /* Initialize gradient at output (all ones) */
    tofu_graph_zero_grad(g);
    tofu_graph_backward(g, z);

    /* Check gradient shape */
    if (!x->grad) {
        printf("      Error: grad_x is NULL\n");
        tofu_tensor_free(t_x);
        tofu_tensor_free(t_y);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    if (x->grad->ndim != 1 || x->grad->dims[0] != 1) {
        printf("      Error: grad_x shape incorrect - got [");
        for (int i = 0; i < x->grad->ndim; i++) {
            printf("%d%s", x->grad->dims[i], (i < x->grad->ndim - 1) ? "," : "");
        }
        printf("], expected [1]\n");
        tofu_tensor_free(t_x);
        tofu_tensor_free(t_y);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    /* Check gradient value
     * grad_x[0] = sum(grad_z[:,:]) = sum of all ones = 12
     */
    float grad_val;
    TOFU_TENSOR_DATA_TO(x->grad, 0, grad_val, TOFU_FLOAT);
    if (fabsf(grad_val - 12.0f) > TOLERANCE) {
        printf("      Error: grad_x[0] = %f, expected 12.0\n", grad_val);
        tofu_tensor_free(t_x);
        tofu_tensor_free(t_y);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    tofu_tensor_free(t_x);
    tofu_tensor_free(t_y);
    tofu_graph_free(g);
    return TEST_PASS;
}

/* ============================================================
 * Test 3: MUL with Broadcasting - Multiple Dimensions
 * ============================================================
 * Forward: x[2,1,3] * y[1,4,3] -> z[2,4,3]
 * Backward: grad_x should be [2,1,3], grad_y should be [1,4,3]
 */
int test_mul_broadcast_multidim() {
    tofu_graph* g = tofu_graph_create();

    /* Create tensors */
    float x_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  /* [2,1,3] */
    float y_data[12] = {1.0f, 1.0f, 1.0f,
                        2.0f, 2.0f, 2.0f,
                        3.0f, 3.0f, 3.0f,
                        4.0f, 4.0f, 4.0f};  /* [1,4,3] */

    tofu_tensor* t_x = tofu_tensor_create(x_data, 3, (int[]){2, 1, 3}, TOFU_FLOAT);
    tofu_tensor* t_y = tofu_tensor_create(y_data, 3, (int[]){1, 4, 3}, TOFU_FLOAT);

    /* Build graph: z = x * y */
    tofu_graph_node* x = tofu_graph_param(g, t_x);
    tofu_graph_node* y = tofu_graph_param(g, t_y);
    tofu_graph_node* z = tofu_graph_mul(g, x, y);

    /* Initialize gradient at output (all ones) */
    tofu_graph_zero_grad(g);
    tofu_graph_backward(g, z);

    /* Check gradient shapes */
    if (!x->grad || x->grad->ndim != 3 ||
        x->grad->dims[0] != 2 || x->grad->dims[1] != 1 || x->grad->dims[2] != 3) {
        printf("      Error: grad_x shape incorrect\n");
        tofu_tensor_free(t_x);
        tofu_tensor_free(t_y);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    if (!y->grad || y->grad->ndim != 3 ||
        y->grad->dims[0] != 1 || y->grad->dims[1] != 4 || y->grad->dims[2] != 3) {
        printf("      Error: grad_y shape incorrect\n");
        tofu_tensor_free(t_x);
        tofu_tensor_free(t_y);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    /* Check grad_x values: each [i,0,:] should be sum over y[0,:,:]
     * grad_x[0,0,:] = sum(y[0,:,:]) = [1+2+3+4, 1+2+3+4, 1+2+3+4] = [10, 10, 10]
     * grad_x[1,0,:] = sum(y[0,:,:]) = [10, 10, 10]
     */
    for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 3; k++) {
            float grad_val;
            TOFU_TENSOR_DATA_TO(x->grad, i * 3 + k, grad_val, TOFU_FLOAT);
            if (fabsf(grad_val - 10.0f) > TOLERANCE) {
                printf("      Error: grad_x[%d,0,%d] = %f, expected 10.0\n",
                       i, k, grad_val);
                tofu_tensor_free(t_x);
                tofu_tensor_free(t_y);
                tofu_graph_free(g);
                return TEST_FAIL;
            }
        }
    }

    tofu_tensor_free(t_x);
    tofu_tensor_free(t_y);
    tofu_graph_free(g);
    return TEST_PASS;
}

/* ============================================================
 * Test 4: ADD with No Broadcasting - Same Shape
 * ============================================================
 * Forward: x[3,4] + y[3,4] -> z[3,4]
 * Backward: gradients should have same shape (no reduction)
 */
int test_add_no_broadcast() {
    tofu_graph* g = tofu_graph_create();

    /* Create tensors */
    float x_data[12] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float y_data[12] = {12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                        6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    tofu_tensor* t_x = tofu_tensor_create(x_data, 2, (int[]){3, 4}, TOFU_FLOAT);
    tofu_tensor* t_y = tofu_tensor_create(y_data, 2, (int[]){3, 4}, TOFU_FLOAT);

    /* Build graph: z = x + y */
    tofu_graph_node* x = tofu_graph_param(g, t_x);
    tofu_graph_node* y = tofu_graph_param(g, t_y);
    tofu_graph_node* z = tofu_graph_add(g, x, y);

    /* Initialize gradient at output (all ones) */
    tofu_graph_zero_grad(g);
    tofu_graph_backward(g, z);

    /* Check gradient shapes - should match input shapes */
    if (!x->grad || x->grad->ndim != 2 || x->grad->dims[0] != 3 || x->grad->dims[1] != 4) {
        printf("      Error: grad_x shape incorrect\n");
        tofu_tensor_free(t_x);
        tofu_tensor_free(t_y);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    if (!y->grad || y->grad->ndim != 2 || y->grad->dims[0] != 3 || y->grad->dims[1] != 4) {
        printf("      Error: grad_y shape incorrect\n");
        tofu_tensor_free(t_x);
        tofu_tensor_free(t_y);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    /* Check gradient values - all should be 1.0 */
    for (int i = 0; i < 12; i++) {
        float grad_x_val, grad_y_val;
        TOFU_TENSOR_DATA_TO(x->grad, i, grad_x_val, TOFU_FLOAT);
        TOFU_TENSOR_DATA_TO(y->grad, i, grad_y_val, TOFU_FLOAT);

        if (fabsf(grad_x_val - 1.0f) > TOLERANCE || fabsf(grad_y_val - 1.0f) > TOLERANCE) {
            printf("      Error: gradient values incorrect at index %d\n", i);
            tofu_tensor_free(t_x);
            tofu_tensor_free(t_y);
            tofu_graph_free(g);
            return TEST_FAIL;
        }
    }

    tofu_tensor_free(t_x);
    tofu_tensor_free(t_y);
    tofu_graph_free(g);
    return TEST_PASS;
}

/* ============================================================
 * Test 5: MATMUL with Batch Broadcasting
 * ============================================================
 * Forward: A[1,3,4] @ B[2,4,5] -> C[2,3,5]
 * Backward: grad_A should be [1,3,4] (sum over batch dim 0)
 *           grad_B should be [2,4,5] (no reduction needed)
 */
int test_matmul_batch_broadcast() {
    tofu_graph* g = tofu_graph_create();

    /* Create tensors with batch broadcasting */
    float A_data[12];  /* [1,3,4] - will broadcast to [2,3,4] */
    float B_data[40];  /* [2,4,5] - no broadcasting */

    /* Initialize with simple values */
    for (int i = 0; i < 12; i++) {
        A_data[i] = (float)(i + 1);  /* 1, 2, 3, ..., 12 */
    }
    for (int i = 0; i < 40; i++) {
        B_data[i] = 1.0f;  /* All ones for easier gradient calculation */
    }

    tofu_tensor* t_A = tofu_tensor_create(A_data, 3, (int[]){1, 3, 4}, TOFU_FLOAT);
    tofu_tensor* t_B = tofu_tensor_create(B_data, 3, (int[]){2, 4, 5}, TOFU_FLOAT);

    /* Build graph: C = A @ B */
    tofu_graph_node* A = tofu_graph_param(g, t_A);
    tofu_graph_node* B = tofu_graph_param(g, t_B);
    tofu_graph_node* C = tofu_graph_matmul(g, A, B);

    /* Check forward pass output shape */
    if (!C->value || C->value->ndim != 3 ||
        C->value->dims[0] != 2 || C->value->dims[1] != 3 || C->value->dims[2] != 5) {
        printf("      Error: Forward pass output shape incorrect\n");
        tofu_tensor_free(t_A);
        tofu_tensor_free(t_B);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    /* Initialize gradient at output (all ones) */
    tofu_graph_zero_grad(g);
    tofu_graph_backward(g, C);

    /* Check gradient shape for A - should be [1,3,4] */
    if (!A->grad) {
        printf("      Error: grad_A is NULL\n");
        tofu_tensor_free(t_A);
        tofu_tensor_free(t_B);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    if (A->grad->ndim != 3 || A->grad->dims[0] != 1 ||
        A->grad->dims[1] != 3 || A->grad->dims[2] != 4) {
        printf("      Error: grad_A shape incorrect - got [");
        for (int i = 0; i < A->grad->ndim; i++) {
            printf("%d%s", A->grad->dims[i], (i < A->grad->ndim - 1) ? "," : "");
        }
        printf("], expected [1,3,4]\n");
        tofu_tensor_free(t_A);
        tofu_tensor_free(t_B);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    /* Check gradient shape for B - should be [2,4,5] (no change) */
    if (!B->grad || B->grad->ndim != 3 ||
        B->grad->dims[0] != 2 || B->grad->dims[1] != 4 || B->grad->dims[2] != 5) {
        printf("      Error: grad_B shape incorrect\n");
        tofu_tensor_free(t_A);
        tofu_tensor_free(t_B);
        tofu_graph_free(g);
        return TEST_FAIL;
    }

    /* Verify gradient values
     * Since B is all ones and grad_C is all ones:
     * grad_A[0,i,j] should be sum over batches of: grad_C[b,i,:] @ B[b,j,:].T
     * With B=1 and grad_C=1, grad_A[0,i,j] should be 2*5 = 10 (sum over 2 batches, 5 columns)
     */
    for (int i = 0; i < 12; i++) {
        float grad_val;
        TOFU_TENSOR_DATA_TO(A->grad, i, grad_val, TOFU_FLOAT);
        if (fabsf(grad_val - 10.0f) > TOLERANCE) {
            printf("      Error: grad_A[%d] = %f, expected 10.0\n", i, grad_val);
            tofu_tensor_free(t_A);
            tofu_tensor_free(t_B);
            tofu_graph_free(g);
            return TEST_FAIL;
        }
    }

    tofu_tensor_free(t_A);
    tofu_tensor_free(t_B);
    tofu_graph_free(g);
    return TEST_PASS;
}

int main() {
    printf("============================================================\n");
    printf("Tofu Broadcasting Gradient Test Suite\n");
    printf("============================================================\n");
    printf("Testing gradient reduction for broadcast operations\n");
    printf("============================================================\n\n");

    printf("Category: MUL Broadcasting Tests\n");
    report_test("test_mul_broadcast_simple", test_mul_broadcast_simple());
    report_test("test_mul_broadcast_multidim", test_mul_broadcast_multidim());

    printf("\nCategory: ADD Broadcasting Tests\n");
    report_test("test_add_broadcast_scalar", test_add_broadcast_scalar());
    report_test("test_add_no_broadcast", test_add_no_broadcast());

    printf("\nCategory: MATMUL Batch Broadcasting Tests\n");
    report_test("test_matmul_batch_broadcast", test_matmul_batch_broadcast());

    printf("\n============================================================\n");
    printf("Test Summary\n");
    printf("============================================================\n");
    printf("Total:   %d tests\n", tests_run);
    printf("Passed:  %d tests\n", tests_passed);
    printf("Failed:  %d tests\n", tests_failed);
    printf("============================================================\n");

    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
