/*
 * Double-precision gradient checking to validate analytical gradients
 *
 * This test performs gradient checking using double precision to eliminate
 * floating-point errors that affect float-based numerical differentiation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tl_graph.h"
#include "tl_tensor.h"

#define EPSILON 1e-7
#define TOLERANCE 1e-5  /* Much stricter tolerance with double precision */

/*
 * Double-precision matmul: C = A @ B
 * Computes the result in double precision for high-accuracy numerical gradients
 */
void matmul_double(const double* A, const double* B, double* C,
                   int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
 * Compute loss in double precision: sum of all matmul outputs
 */
double compute_loss_double(const double* A, const double* B, int M, int K, int N) {
    double* C = (double*)malloc(M * N * sizeof(double));
    matmul_double(A, B, C, M, K, N);

    double loss = 0.0;
    for (int i = 0; i < M * N; i++) {
        loss += C[i];
    }

    free(C);
    return loss;
}

/*
 * Compute numerical gradient using double precision
 */
double compute_numerical_gradient_double(double* params, int idx, int M, int K, int N,
                                         const double* B_data) {
    double original = params[idx];

    /* f(x + epsilon) */
    params[idx] = original + EPSILON;
    double loss_plus = compute_loss_double(params, B_data, M, K, N);

    /* f(x - epsilon) */
    params[idx] = original - EPSILON;
    double loss_minus = compute_loss_double(params, B_data, M, K, N);

    /* Restore */
    params[idx] = original;

    /* Numerical gradient */
    return (loss_plus - loss_minus) / (2.0 * EPSILON);
}

int main() {
    printf("============================================================\n");
    printf("Double-Precision Gradient Checking Validation\n");
    printf("============================================================\n");
    printf("\nObjective: Validate analytical gradients using high-precision\n");
    printf("           numerical differentiation with epsilon=%.0e\n\n", EPSILON);

    /* Test case: Same as validation test */
    int M = 2, K = 3, N = 2;

    float A_data_f[] = {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f};

    float B_data_f[] = {0.5f, -0.3f,
                        0.2f,  0.8f,
                        -0.1f, 0.4f};

    /* Convert to double precision */
    double A_data[6], B_data[6];
    for (int i = 0; i < M * K; i++) A_data[i] = (double)A_data_f[i];
    for (int i = 0; i < K * N; i++) B_data[i] = (double)B_data_f[i];

    printf("Test Data:\n");
    printf("  A [%d×%d]: [%.1f, %.1f, %.1f; %.1f, %.1f, %.1f]\n",
           M, K, A_data[0], A_data[1], A_data[2], A_data[3], A_data[4], A_data[5]);
    printf("  B [%d×%d]: [%.1f, %.1f; %.1f, %.1f; %.1f, %.1f]\n\n",
           K, N, B_data[0], B_data[1], B_data[2], B_data[3], B_data[4], B_data[5]);

    /* Compute analytical gradients using Tofu */
    printf("Computing analytical gradients using Tofu (float)...\n");
    tl_graph* g = tl_graph_create();

    tl_tensor* t_A = tl_tensor_create(A_data_f, 2, (int[]){M, K}, TL_FLOAT);
    tl_tensor* t_B = tl_tensor_create(B_data_f, 2, (int[]){K, N}, TL_FLOAT);

    tl_graph_node* A = tl_graph_input(g, t_A);
    tl_graph_node* B = tl_graph_input(g, t_B);
    A->requires_grad = 1;
    B->requires_grad = 1;

    tl_graph_node* C = tl_graph_matmul(g, A, B);

    /* Set gradient to ones */
    C->grad = tl_tensor_create_with_values((float[]){1.0f, 1.0f, 1.0f, 1.0f},
                                            2, (int[]){M, N});
    tl_graph_backward(g, C);

    printf("Analytical gradients (from Tofu):\n");
    printf("  dL/dA:\n");
    for (int i = 0; i < M * K; i++) {
        float grad_val;
        TL_TENSOR_DATA_TO(A->grad, i, grad_val, TL_FLOAT);
        printf("    A[%d]: %.6f\n", i, grad_val);
    }

    printf("\n");
    printf("Computing numerical gradients using double precision...\n");
    printf("Numerical gradients (double precision, epsilon=%.0e):\n", EPSILON);
    printf("  dL/dA:\n");

    int num_passed = 0;
    int num_failed = 0;
    double max_error = 0.0;

    for (int i = 0; i < M * K; i++) {
        double numerical = compute_numerical_gradient_double(A_data, i, M, K, N, B_data);

        float analytical_f;
        TL_TENSOR_DATA_TO(A->grad, i, analytical_f, TL_FLOAT);
        double analytical = (double)analytical_f;

        double error = fabs(analytical - numerical);
        double rel_error = error / fmax(fabs(analytical), fabs(numerical));

        printf("    A[%d]: numerical=%.10f, analytical=%.10f, error=%.2e, rel_error=%.2e",
               i, numerical, analytical, error, rel_error);

        if (rel_error < TOLERANCE) {
            printf(" ✓\n");
            num_passed++;
        } else {
            printf(" ✗\n");
            num_failed++;
        }

        if (rel_error > max_error) {
            max_error = rel_error;
        }
    }

    printf("\n");
    printf("Results:\n");
    printf("  Passed: %d/%d\n", num_passed, M * K);
    printf("  Failed: %d/%d\n", num_failed, M * K);
    printf("  Max relative error: %.2e\n", max_error);
    printf("  Tolerance: %.2e\n\n", TOLERANCE);

    if (num_failed == 0) {
        printf("✓ SUCCESS: All gradients match within tolerance!\n");
        printf("  This confirms that Tofu's analytical gradients are mathematically correct.\n");
        printf("  The errors seen with float precision were due to numerical precision limits.\n");
    } else {
        printf("✗ FAILED: Some gradients do not match.\n");
        printf("  This suggests a potential issue in the backward pass implementation.\n");
    }

    /* Cleanup */
    tl_tensor_free(t_A);
    tl_tensor_free(t_B);
    tl_graph_free(g);

    printf("\n============================================================\n");

    return (num_failed == 0) ? 0 : 1;
}
