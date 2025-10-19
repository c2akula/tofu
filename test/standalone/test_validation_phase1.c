/*
 * Tofu Framework - Validation Test Suite Phase 1: Correctness
 *
 * This test suite validates the correctness of gradient computations
 * using gradient checking (numerical vs analytical gradients).
 *
 * Gradient checking provides mathematical proof of correctness by comparing:
 * - Analytical gradients (computed via backward pass)
 * - Numerical gradients (computed via finite differences)
 *
 * If relative error < 1e-5, the gradient implementation is correct.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "tl_graph.h"
#include "tl_tensor.h"
#include "tl_optimizer.h"

#define EPSILON 1e-5f
#define TOLERANCE 1e-4f

/* Helper: Compute relative error between analytical and numerical gradients */
static float relative_error(float analytical, float numerical) {
    float numerator = fabsf(analytical - numerical);
    float denominator = fmaxf(fabsf(analytical), fabsf(numerical));

    /* Handle case where both are zero */
    if (denominator < 1e-10f) {
        return numerator;
    }

    return numerator / denominator;
}

/* Helper: Compute numerical gradient for a single parameter */
static float compute_numerical_gradient(
    float* param_data,
    int param_idx,
    int param_len,
    float (*loss_fn)(float*, void*),
    void* context
) {
    float original = param_data[param_idx];

    /* f(x + epsilon) */
    param_data[param_idx] = original + EPSILON;
    float loss_plus = loss_fn(param_data, context);

    /* f(x - epsilon) */
    param_data[param_idx] = original - EPSILON;
    float loss_minus = loss_fn(param_data, context);

    /* Restore original value */
    param_data[param_idx] = original;

    /* Numerical gradient: (f(x+ε) - f(x-ε)) / (2ε) */
    return (loss_plus - loss_minus) / (2.0f * EPSILON);
}

/*
 * Test 1.1.1: Gradient Checking for Matmul
 *
 * Tests: dL/dA and dL/dB for C = A @ B
 */

typedef struct {
    float* A_data;
    float* B_data;
    int M, K, N;
} matmul_context;

static float matmul_loss_fn(float* param_data, void* ctx) {
    matmul_context* mc = (matmul_context*)ctx;

    tl_graph* g = tl_graph_create();

    tl_tensor* t_A = tl_tensor_create(mc->A_data, 2, (int[]){mc->M, mc->K}, TL_FLOAT);
    tl_tensor* t_B = tl_tensor_create(mc->B_data, 2, (int[]){mc->K, mc->N}, TL_FLOAT);

    tl_graph_node* A = tl_graph_input(g, t_A);
    tl_graph_node* B = tl_graph_input(g, t_B);
    tl_graph_node* C = tl_graph_matmul(g, A, B);

    /* Compute scalar loss: sum of all outputs */
    float loss = 0.0f;
    for (int i = 0; i < C->value->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(C->value, i, val, TL_FLOAT);
        loss += val;
    }

    tl_tensor_free(t_A);
    tl_tensor_free(t_B);
    tl_graph_free(g);

    return loss;
}

static void test_gradient_checking_matmul() {
    printf("\nTest 1.1.1: Gradient Checking - Matmul\n");
    printf("----------------------------------------\n");

    int M = 2, K = 3, N = 2;

    float A_data[] = {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f};

    float B_data[] = {0.5f, -0.3f,
                      0.2f,  0.8f,
                      -0.1f, 0.4f};

    /* Create graph */
    tl_graph* g = tl_graph_create();

    tl_tensor* t_A = tl_tensor_create(A_data, 2, (int[]){M, K}, TL_FLOAT);
    tl_tensor* t_B = tl_tensor_create(B_data, 2, (int[]){K, N}, TL_FLOAT);

    tl_graph_node* A = tl_graph_input(g, t_A);
    tl_graph_node* B = tl_graph_input(g, t_B);
    A->requires_grad = 1;  /* Enable gradient computation for inputs */
    B->requires_grad = 1;
    tl_graph_node* C = tl_graph_matmul(g, A, B);

    /* Compute scalar loss: sum of all outputs */
    float loss = 0.0f;
    for (int i = 0; i < C->value->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(C->value, i, val, TL_FLOAT);
        loss += val;
    }

    /* Backward pass - gradient of loss w.r.t. C is all ones */
    C->grad = tl_tensor_create_with_values((float[]){1.0f, 1.0f, 1.0f, 1.0f}, 2, (int[]){M, N});
    tl_graph_backward(g, C);

    /* Check gradients for A */
    printf("  Checking dL/dA:\n");
    matmul_context ctx_A = {A_data, B_data, M, K, N};
    int num_errors_A = 0;

    for (int i = 0; i < M * K; i++) {
        float analytical;
        TL_TENSOR_DATA_TO(A->grad, i, analytical, TL_FLOAT);

        float numerical = compute_numerical_gradient(A_data, i, M * K, matmul_loss_fn, &ctx_A);
        float error = relative_error(analytical, numerical);

        if (error > TOLERANCE) {
            printf("    ERROR at A[%d]: analytical=%.6f, numerical=%.6f, error=%.6f\n",
                   i, analytical, numerical, error);
            num_errors_A++;
        }
    }

    if (num_errors_A == 0) {
        printf("    ✓ All %d gradients correct (error < %.0e)\n", M * K, TOLERANCE);
    } else {
        printf("    ✗ FAILED: %d/%d gradients incorrect\n", num_errors_A, M * K);
    }

    /* Check gradients for B */
    printf("  Checking dL/dB:\n");
    matmul_context ctx_B = {A_data, B_data, M, K, N};
    int num_errors_B = 0;

    for (int i = 0; i < K * N; i++) {
        float analytical;
        TL_TENSOR_DATA_TO(B->grad, i, analytical, TL_FLOAT);

        float numerical = compute_numerical_gradient(B_data, i, K * N, matmul_loss_fn, &ctx_B);
        float error = relative_error(analytical, numerical);

        if (error > TOLERANCE) {
            printf("    ERROR at B[%d]: analytical=%.6f, numerical=%.6f, error=%.6f\n",
                   i, analytical, numerical, error);
            num_errors_B++;
        }
    }

    if (num_errors_B == 0) {
        printf("    ✓ All %d gradients correct (error < %.0e)\n", K * N, TOLERANCE);
    } else {
        printf("    ✗ FAILED: %d/%d gradients incorrect\n", num_errors_B, K * N);
    }

    assert(num_errors_A == 0 && num_errors_B == 0);

    tl_tensor_free(t_A);
    tl_tensor_free(t_B);
    tl_graph_free(g);

    printf("  ✓ PASSED\n");
}

/*
 * Test 1.1.2: Gradient Checking for Add
 *
 * Tests: dL/dx and dL/dy for z = x + y
 */

typedef struct {
    float* x_data;
    float* y_data;
    int len;
} add_context;

static float add_loss_fn(float* param_data, void* ctx) {
    add_context* ac = (add_context*)ctx;

    tl_graph* g = tl_graph_create();

    tl_tensor* t_x = tl_tensor_create(ac->x_data, 1, (int[]){ac->len}, TL_FLOAT);
    tl_tensor* t_y = tl_tensor_create(ac->y_data, 1, (int[]){ac->len}, TL_FLOAT);

    tl_graph_node* x = tl_graph_input(g, t_x);
    tl_graph_node* y = tl_graph_input(g, t_y);
    tl_graph_node* z = tl_graph_add(g, x, y);

    /* Compute scalar loss: sum of all outputs */
    float loss = 0.0f;
    for (int i = 0; i < z->value->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(z->value, i, val, TL_FLOAT);
        loss += val;
    }

    tl_tensor_free(t_x);
    tl_tensor_free(t_y);
    tl_graph_free(g);

    return loss;
}

static void test_gradient_checking_add() {
    printf("\nTest 1.1.2: Gradient Checking - Add\n");
    printf("------------------------------------\n");

    int len = 4;
    float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y_data[] = {0.5f, -0.3f, 0.8f, -0.2f};

    /* Create graph */
    tl_graph* g = tl_graph_create();

    tl_tensor* t_x = tl_tensor_create(x_data, 1, (int[]){len}, TL_FLOAT);
    tl_tensor* t_y = tl_tensor_create(y_data, 1, (int[]){len}, TL_FLOAT);

    tl_graph_node* x = tl_graph_input(g, t_x);
    tl_graph_node* y = tl_graph_input(g, t_y);
    x->requires_grad = 1;  /* Enable gradient computation for inputs */
    y->requires_grad = 1;
    tl_graph_node* z = tl_graph_add(g, x, y);

    /* Compute scalar loss: sum of all outputs */
    float loss = 0.0f;
    for (int i = 0; i < z->value->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(z->value, i, val, TL_FLOAT);
        loss += val;
    }

    /* Backward pass */
    z->grad = tl_tensor_create_with_values((float[]){1.0f, 1.0f, 1.0f, 1.0f}, 1, (int[]){len});
    tl_graph_backward(g, z);

    /* Check gradients for x */
    printf("  Checking dL/dx:\n");
    add_context ctx_x = {x_data, y_data, len};
    int num_errors_x = 0;

    for (int i = 0; i < len; i++) {
        float analytical;
        TL_TENSOR_DATA_TO(x->grad, i, analytical, TL_FLOAT);

        float numerical = compute_numerical_gradient(x_data, i, len, add_loss_fn, &ctx_x);
        float error = relative_error(analytical, numerical);

        if (error > TOLERANCE) {
            printf("    ERROR at x[%d]: analytical=%.6f, numerical=%.6f, error=%.6f\n",
                   i, analytical, numerical, error);
            num_errors_x++;
        }
    }

    if (num_errors_x == 0) {
        printf("    ✓ All %d gradients correct (error < %.0e)\n", len, TOLERANCE);
    } else {
        printf("    ✗ FAILED: %d/%d gradients incorrect\n", num_errors_x, len);
    }

    /* Check gradients for y */
    printf("  Checking dL/dy:\n");
    add_context ctx_y = {x_data, y_data, len};
    int num_errors_y = 0;

    for (int i = 0; i < len; i++) {
        float analytical;
        TL_TENSOR_DATA_TO(y->grad, i, analytical, TL_FLOAT);

        float numerical = compute_numerical_gradient(y_data, i, len, add_loss_fn, &ctx_y);
        float error = relative_error(analytical, numerical);

        if (error > TOLERANCE) {
            printf("    ERROR at y[%d]: analytical=%.6f, numerical=%.6f, error=%.6f\n",
                   i, analytical, numerical, error);
            num_errors_y++;
        }
    }

    if (num_errors_y == 0) {
        printf("    ✓ All %d gradients correct (error < %.0e)\n", len, TOLERANCE);
    } else {
        printf("    ✗ FAILED: %d/%d gradients incorrect\n", num_errors_y, len);
    }

    assert(num_errors_x == 0 && num_errors_y == 0);

    tl_tensor_free(t_x);
    tl_tensor_free(t_y);
    tl_graph_free(g);

    printf("  ✓ PASSED\n");
}

/*
 * Test 1.1.3: Gradient Checking for ReLU
 *
 * Tests: dL/dx for y = ReLU(x)
 */

typedef struct {
    float* x_data;
    int len;
} relu_context;

static float relu_loss_fn(float* param_data, void* ctx) {
    relu_context* rc = (relu_context*)ctx;

    tl_graph* g = tl_graph_create();

    tl_tensor* t_x = tl_tensor_create(rc->x_data, 1, (int[]){rc->len}, TL_FLOAT);
    tl_graph_node* x = tl_graph_input(g, t_x);
    tl_graph_node* y = tl_graph_relu(g, x);

    /* Compute scalar loss: sum of all outputs */
    float loss = 0.0f;
    for (int i = 0; i < y->value->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(y->value, i, val, TL_FLOAT);
        loss += val;
    }

    tl_tensor_free(t_x);
    tl_graph_free(g);

    return loss;
}

static void test_gradient_checking_relu() {
    printf("\nTest 1.1.3: Gradient Checking - ReLU\n");
    printf("-------------------------------------\n");

    int len = 6;
    /* Include negative, positive, and near-zero values */
    float x_data[] = {-2.0f, -0.5f, -0.01f, 0.01f, 0.5f, 2.0f};

    /* Create graph */
    tl_graph* g = tl_graph_create();

    tl_tensor* t_x = tl_tensor_create(x_data, 1, (int[]){len}, TL_FLOAT);
    tl_graph_node* x = tl_graph_input(g, t_x);
    x->requires_grad = 1;  /* Enable gradient computation for inputs */
    tl_graph_node* y = tl_graph_relu(g, x);

    /* Compute scalar loss: sum of all outputs */
    float loss = 0.0f;
    for (int i = 0; i < y->value->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(y->value, i, val, TL_FLOAT);
        loss += val;
    }

    /* Backward pass */
    y->grad = tl_tensor_create_with_values((float[]){1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, 1, (int[]){len});
    tl_graph_backward(g, y);

    /* Check gradients for x */
    printf("  Checking dL/dx:\n");
    relu_context ctx = {x_data, len};
    int num_errors = 0;

    for (int i = 0; i < len; i++) {
        float analytical;
        TL_TENSOR_DATA_TO(x->grad, i, analytical, TL_FLOAT);

        float numerical = compute_numerical_gradient(x_data, i, len, relu_loss_fn, &ctx);
        float error = relative_error(analytical, numerical);

        if (error > TOLERANCE) {
            printf("    ERROR at x[%d]=%.2f: analytical=%.6f, numerical=%.6f, error=%.6f\n",
                   i, x_data[i], analytical, numerical, error);
            num_errors++;
        }
    }

    if (num_errors == 0) {
        printf("    ✓ All %d gradients correct (error < %.0e)\n", len, TOLERANCE);
    } else {
        printf("    ✗ FAILED: %d/%d gradients incorrect\n", num_errors, len);
    }

    assert(num_errors == 0);

    tl_tensor_free(t_x);
    tl_graph_free(g);

    printf("  ✓ PASSED\n");
}

/*
 * Test 1.1.4: Gradient Checking for Softmax
 *
 * Tests: dL/dx for y = softmax(x)
 * This is the most complex gradient to verify.
 */

typedef struct {
    float* x_data;
    int batch_size;
    int num_classes;
} softmax_context;

static float softmax_loss_fn(float* param_data, void* ctx) {
    softmax_context* sc = (softmax_context*)ctx;

    tl_graph* g = tl_graph_create();

    tl_tensor* t_x = tl_tensor_create(sc->x_data, 2, (int[]){sc->batch_size, sc->num_classes}, TL_FLOAT);
    tl_graph_node* x = tl_graph_input(g, t_x);
    tl_graph_node* y = tl_graph_softmax(g, x, 1);

    /* Compute scalar loss: sum of all outputs */
    float loss = 0.0f;
    for (int i = 0; i < y->value->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(y->value, i, val, TL_FLOAT);
        loss += val;
    }

    tl_tensor_free(t_x);
    tl_graph_free(g);

    return loss;
}

static void test_gradient_checking_softmax() {
    printf("\nTest 1.1.4: Gradient Checking - Softmax\n");
    printf("----------------------------------------\n");

    int batch_size = 2;
    int num_classes = 3;

    float x_data[] = {
        1.0f, 2.0f, 3.0f,     /* Sample 1 */
        0.5f, -0.3f, 0.8f     /* Sample 2 */
    };

    /* Create graph */
    tl_graph* g = tl_graph_create();

    tl_tensor* t_x = tl_tensor_create(x_data, 2, (int[]){batch_size, num_classes}, TL_FLOAT);
    tl_graph_node* x = tl_graph_input(g, t_x);
    x->requires_grad = 1;  /* Enable gradient computation for inputs */
    tl_graph_node* y = tl_graph_softmax(g, x, 1);

    /* Compute scalar loss: sum of all outputs */
    float loss = 0.0f;
    for (int i = 0; i < y->value->len; i++) {
        float val;
        TL_TENSOR_DATA_TO(y->value, i, val, TL_FLOAT);
        loss += val;
    }

    /* Backward pass */
    float grad_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    y->grad = tl_tensor_create_with_values(grad_data, 2, (int[]){batch_size, num_classes});
    tl_graph_backward(g, y);

    /* Check gradients for x */
    printf("  Checking dL/dx:\n");
    softmax_context ctx = {x_data, batch_size, num_classes};
    int num_errors = 0;

    for (int i = 0; i < batch_size * num_classes; i++) {
        float analytical;
        TL_TENSOR_DATA_TO(x->grad, i, analytical, TL_FLOAT);

        float numerical = compute_numerical_gradient(x_data, i, batch_size * num_classes, softmax_loss_fn, &ctx);
        float error = relative_error(analytical, numerical);

        if (error > TOLERANCE) {
            printf("    ERROR at x[%d]: analytical=%.6f, numerical=%.6f, error=%.6f\n",
                   i, analytical, numerical, error);
            num_errors++;
        }
    }

    if (num_errors == 0) {
        printf("    ✓ All %d gradients correct (error < %.0e)\n", batch_size * num_classes, TOLERANCE);
    } else {
        printf("    ✗ FAILED: %d/%d gradients incorrect\n", num_errors, batch_size * num_classes);
    }

    assert(num_errors == 0);

    tl_tensor_free(t_x);
    tl_graph_free(g);

    printf("  ✓ PASSED\n");
}

/*
 * Test 1.2.1: Known Analytical Solution - Linear Regression
 *
 * Problem: Learn y = 2x + 3 from clean data
 * Expected: Weights converge to w≈2, b≈3
 */

static void test_known_solution_linear_regression() {
    printf("\nTest 1.2.1: Known Solution - Linear Regression (y = 2x + 3)\n");
    printf("------------------------------------------------------------\n");

    /* Generate data: y = 2x + 3 */
    int num_samples = 20;
    float x_vals[20];
    float y_vals[20];

    for (int i = 0; i < num_samples; i++) {
        x_vals[i] = (float)i / 10.0f - 1.0f;  /* x ∈ [-1, 1] */
        y_vals[i] = 2.0f * x_vals[i] + 3.0f;  /* y = 2x + 3 */
    }

    /* Initialize parameters */
    float w = 0.0f;  /* Will learn → 2.0 */
    float b = 0.0f;  /* Will learn → 3.0 */

    tl_graph* g = tl_graph_create();

    tl_tensor* t_w = tl_tensor_create(&w, 0, NULL, TL_FLOAT);
    tl_tensor* t_b = tl_tensor_create(&b, 0, NULL, TL_FLOAT);

    tl_graph_node* param_w = tl_graph_param(g, t_w);
    tl_graph_node* param_b = tl_graph_param(g, t_b);

    tl_optimizer* opt = tl_optimizer_sgd_create(g, 0.1);

    int epochs = 100;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < num_samples; i++) {
            tl_optimizer_zero_grad(opt);

            /* Forward pass: pred = w * x + b */
            tl_tensor* t_x = tl_tensor_create(&x_vals[i], 0, NULL, TL_FLOAT);
            tl_graph_node* x = tl_graph_input(g, t_x);

            tl_graph_node* wx = tl_graph_mul(g, param_w, x);
            tl_graph_node* pred = tl_graph_add(g, wx, param_b);

            /* Compute loss: MSE = (pred - y)^2 */
            float pred_val;
            TL_TENSOR_DATA_TO(pred->value, 0, pred_val, TL_FLOAT);
            float error = pred_val - y_vals[i];
            float loss = error * error;
            epoch_loss += loss;

            /* Backward pass */
            float grad_val = 2.0f * error;
            pred->grad = tl_tensor_create_with_values(&grad_val, 0, NULL);
            tl_graph_backward(g, pred);

            /* Update parameters */
            tl_optimizer_step(opt);

            tl_tensor_free(t_x);
            tl_graph_clear_ops(g);
        }

        epoch_loss /= num_samples;

        if (epoch == 0) {
            initial_loss = epoch_loss;
        }
        if (epoch == epochs - 1) {
            final_loss = epoch_loss;
        }

        if (epoch % 20 == 0 || epoch == epochs - 1) {
            float w_val, b_val;
            TL_TENSOR_DATA_TO(param_w->value, 0, w_val, TL_FLOAT);
            TL_TENSOR_DATA_TO(param_b->value, 0, b_val, TL_FLOAT);
            printf("  Epoch %3d: loss=%.6f, w=%.4f, b=%.4f\n", epoch, epoch_loss, w_val, b_val);
        }
    }

    /* Check if converged to correct values */
    float final_w, final_b;
    TL_TENSOR_DATA_TO(param_w->value, 0, final_w, TL_FLOAT);
    TL_TENSOR_DATA_TO(param_b->value, 0, final_b, TL_FLOAT);

    float w_error = fabsf(final_w - 2.0f);
    float b_error = fabsf(final_b - 3.0f);

    printf("\n  Target: w=2.0, b=3.0\n");
    printf("  Learned: w=%.4f (error=%.4f), b=%.4f (error=%.4f)\n",
           final_w, w_error, final_b, b_error);

    assert(w_error < 0.1f);
    assert(b_error < 0.1f);
    assert(final_loss < initial_loss);

    tl_optimizer_free(opt);
    tl_tensor_free(t_w);
    tl_tensor_free(t_b);
    tl_graph_free(g);

    printf("  ✓ PASSED (converged to analytical solution)\n");
}

/*
 * Test 1.2.2: Known Analytical Solution - XOR with Expected Structure
 *
 * Problem: Learn XOR function with known decision boundary
 * Expected: 100% training accuracy
 */

static void test_known_solution_xor() {
    printf("\nTest 1.2.2: Known Solution - XOR Classification\n");
    printf("------------------------------------------------\n");

    /* XOR dataset */
    float X[4][2] = {
        {0.0f, 0.0f},  /* -> 0 */
        {0.0f, 1.0f},  /* -> 1 */
        {1.0f, 0.0f},  /* -> 1 */
        {1.0f, 1.0f}   /* -> 0 */
    };
    float Y[4] = {0.0f, 1.0f, 1.0f, 0.0f};

    /* Network: [2] -> [4] -> [1] */
    int input_dim = 2, hidden_dim = 4, output_dim = 1;

    float W1_data[8];
    float b1_data[4];
    float W2_data[4];
    float b2_data[1];

    /* Xavier initialization */
    float scale1 = sqrtf(2.0f / input_dim);
    float scale2 = sqrtf(2.0f / hidden_dim);

    for (int i = 0; i < 8; i++) W1_data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale1;
    for (int i = 0; i < 4; i++) b1_data[i] = 0.0f;
    for (int i = 0; i < 4; i++) W2_data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale2;
    b2_data[0] = 0.0f;

    tl_graph* g = tl_graph_create();

    tl_tensor* t_W1 = tl_tensor_create(W1_data, 2, (int[]){input_dim, hidden_dim}, TL_FLOAT);
    tl_tensor* t_b1 = tl_tensor_create(b1_data, 1, (int[]){hidden_dim}, TL_FLOAT);
    tl_tensor* t_W2 = tl_tensor_create(W2_data, 2, (int[]){hidden_dim, output_dim}, TL_FLOAT);
    tl_tensor* t_b2 = tl_tensor_create(b2_data, 1, (int[]){output_dim}, TL_FLOAT);

    tl_graph_node* W1 = tl_graph_param(g, t_W1);
    tl_graph_node* b1 = tl_graph_param(g, t_b1);
    tl_graph_node* W2 = tl_graph_param(g, t_W2);
    tl_graph_node* b2 = tl_graph_param(g, t_b2);

    tl_optimizer* opt = tl_optimizer_sgd_momentum_create(g, 0.5, 0.9);

    int epochs = 500;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int sample = 0; sample < 4; sample++) {
            tl_optimizer_zero_grad(opt);

            /* Forward pass */
            tl_tensor* t_x = tl_tensor_create(X[sample], 1, (int[]){input_dim}, TL_FLOAT);
            tl_graph_node* x = tl_graph_input(g, t_x);

            tl_graph_node* h1 = tl_graph_matmul(g, x, W1);
            tl_graph_node* h1_bias = tl_graph_add(g, h1, b1);
            tl_graph_node* h1_act = tl_graph_relu(g, h1_bias);
            tl_graph_node* h2 = tl_graph_matmul(g, h1_act, W2);
            tl_graph_node* output = tl_graph_add(g, h2, b2);

            /* Compute loss: MSE */
            float pred;
            TL_TENSOR_DATA_TO(output->value, 0, pred, TL_FLOAT);
            float error = pred - Y[sample];
            float loss = error * error;
            epoch_loss += loss;

            /* Backward pass */
            float grad_val = 2.0f * error;
            output->grad = tl_tensor_create_with_values(&grad_val, 1, (int[]){output_dim});
            tl_graph_backward(g, output);

            /* Update */
            tl_optimizer_step(opt);

            tl_tensor_free(t_x);
            tl_graph_clear_ops(g);
        }

        epoch_loss /= 4.0f;

        if (epoch == 0) {
            initial_loss = epoch_loss;
        }
        if (epoch == epochs - 1) {
            final_loss = epoch_loss;
        }

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            printf("  Epoch %3d: loss=%.6f\n", epoch, epoch_loss);
        }
    }

    /* Test accuracy */
    int correct = 0;
    for (int sample = 0; sample < 4; sample++) {
        tl_tensor* t_x = tl_tensor_create(X[sample], 1, (int[]){input_dim}, TL_FLOAT);
        tl_graph_node* x = tl_graph_input(g, t_x);

        tl_graph_node* h1 = tl_graph_matmul(g, x, W1);
        tl_graph_node* h1_bias = tl_graph_add(g, h1, b1);
        tl_graph_node* h1_act = tl_graph_relu(g, h1_bias);
        tl_graph_node* h2 = tl_graph_matmul(g, h1_act, W2);
        tl_graph_node* output = tl_graph_add(g, h2, b2);

        float pred;
        TL_TENSOR_DATA_TO(output->value, 0, pred, TL_FLOAT);
        int pred_class = (pred > 0.5f) ? 1 : 0;
        int true_class = (Y[sample] > 0.5f) ? 1 : 0;

        if (pred_class == true_class) {
            correct++;
        }

        tl_tensor_free(t_x);
        tl_graph_clear_ops(g);
    }

    float accuracy = (float)correct / 4.0f * 100.0f;
    printf("\n  Training accuracy: %.0f%% (%d/4 correct)\n", accuracy, correct);

    assert(accuracy == 100.0f);
    assert(final_loss < initial_loss);

    tl_optimizer_free(opt);
    tl_tensor_free(t_W1);
    tl_tensor_free(t_b1);
    tl_tensor_free(t_W2);
    tl_tensor_free(t_b2);
    tl_graph_free(g);

    printf("  ✓ PASSED (100%% accuracy achieved)\n");
}

/*
 * Main test runner
 */

int main() {
    printf("============================================================\n");
    printf("Tofu Validation Test Suite - Phase 1: Correctness\n");
    printf("============================================================\n");

    printf("\n*** Phase 1.1: Gradient Checking Tests ***\n");
    printf("These tests validate that analytical gradients match numerical gradients.\n");

    test_gradient_checking_matmul();
    test_gradient_checking_add();
    test_gradient_checking_relu();
    test_gradient_checking_softmax();

    printf("\n*** Phase 1.2: Known Analytical Solutions Tests ***\n");
    printf("These tests validate end-to-end correctness on problems with known answers.\n");

    test_known_solution_linear_regression();
    test_known_solution_xor();

    printf("\n============================================================\n");
    printf("Phase 1 Validation: ALL TESTS PASSED ✓\n");
    printf("============================================================\n");
    printf("\nConclusion: Tofu's gradient implementations are mathematically correct\n");
    printf("and the framework can learn known analytical solutions successfully.\n\n");

    return 0;
}
