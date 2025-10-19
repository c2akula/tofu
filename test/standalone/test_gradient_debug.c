/*
 * Simple debug test for gradient checking
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tl_graph.h"
#include "tl_tensor.h"

int main() {
    printf("=== Simple Matmul Gradient Debug ===\n\n");

    /* Simple 2x2 @ 2x1 case */
    float A_data[] = {1.0f, 2.0f,
                      3.0f, 4.0f};  /* [2, 2] */
    float B_data[] = {0.5f,
                      0.3f};        /* [2, 1] */

    printf("A = [[%.1f, %.1f], [%.1f, %.1f]]\n", A_data[0], A_data[1], A_data[2], A_data[3]);
    printf("B = [[%.1f], [%.1f]]\n", B_data[0], B_data[1]);

    /* Forward pass */
    tl_graph* g = tl_graph_create();

    tl_tensor* t_A = tl_tensor_create(A_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* t_B = tl_tensor_create(B_data, 2, (int[]){2, 1}, TL_FLOAT);

    tl_graph_node* A = tl_graph_input(g, t_A);
    tl_graph_node* B = tl_graph_input(g, t_B);
    A->requires_grad = 1;
    B->requires_grad = 1;

    tl_graph_node* C = tl_graph_matmul(g, A, B);

    printf("\nC = A @ B:\n");
    float C0, C1;
    TL_TENSOR_DATA_TO(C->value, 0, C0, TL_FLOAT);
    TL_TENSOR_DATA_TO(C->value, 1, C1, TL_FLOAT);
    printf("C[0] = %.4f (expected: 1.0*0.5 + 2.0*0.3 = 1.1)\n", C0);
    printf("C[1] = %.4f (expected: 3.0*0.5 + 4.0*0.3 = 2.7)\n", C1);

    float loss = C0 + C1;
    printf("\nLoss = %.4f (expected: 3.8)\n", loss);

    /* Backward pass */
    C->grad = tl_tensor_create_with_values((float[]){1.0f, 1.0f}, 2, (int[]){2, 1});
    tl_graph_backward(g, C);

    printf("\nAnalytical gradients:\n");
    printf("dL/dA:\n");
    for (int i = 0; i < 4; i++) {
        float grad_val;
        TL_TENSOR_DATA_TO(A->grad, i, grad_val, TL_FLOAT);
        printf("  A[%d]: %.4f\n", i, grad_val);
    }
    printf("dL/dB:\n");
    for (int i = 0; i < 2; i++) {
        float grad_val;
        TL_TENSOR_DATA_TO(B->grad, i, grad_val, TL_FLOAT);
        printf("  B[%d]: %.4f\n", i, grad_val);
    }

    /* Manual calculation for dL/dA */
    printf("\nExpected dL/dA = dL/dC @ B^T:\n");
    printf("dL/dC = [[1.0], [1.0]], B^T = [[0.5, 0.3]]\n");
    printf("dL/dA = [[1.0], [1.0]] @ [[0.5, 0.3]] = [[0.5, 0.3], [0.5, 0.3]]\n");
    printf("Expected: dL/dA[0]=0.5, dL/dA[1]=0.3, dL/dA[2]=0.5, dL/dA[3]=0.3\n");

    /* Numerical gradient for A[0] */
    printf("\nNumerical gradient for A[0]:\n");
    float epsilon = 1e-5f;
    float orig = A_data[0];

    /* f(x + eps) */
    A_data[0] = orig + epsilon;
    tl_graph* g_plus = tl_graph_create();
    tl_tensor* t_A_plus = tl_tensor_create(A_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* t_B_plus = tl_tensor_create(B_data, 2, (int[]){2, 1}, TL_FLOAT);
    tl_graph_node* A_plus = tl_graph_input(g_plus, t_A_plus);
    tl_graph_node* B_plus = tl_graph_input(g_plus, t_B_plus);
    tl_graph_node* C_plus = tl_graph_matmul(g_plus, A_plus, B_plus);
    float Cp0, Cp1;
    TL_TENSOR_DATA_TO(C_plus->value, 0, Cp0, TL_FLOAT);
    TL_TENSOR_DATA_TO(C_plus->value, 1, Cp1, TL_FLOAT);
    float loss_plus = Cp0 + Cp1;
    printf("  A_data[0] = %.6f, loss = %.6f\n", A_data[0], loss_plus);

    /* f(x - eps) */
    A_data[0] = orig - epsilon;
    tl_graph* g_minus = tl_graph_create();
    tl_tensor* t_A_minus = tl_tensor_create(A_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_tensor* t_B_minus = tl_tensor_create(B_data, 2, (int[]){2, 1}, TL_FLOAT);
    tl_graph_node* A_minus = tl_graph_input(g_minus, t_A_minus);
    tl_graph_node* B_minus = tl_graph_input(g_minus, t_B_minus);
    tl_graph_node* C_minus = tl_graph_matmul(g_minus, A_minus, B_minus);
    float Cm0, Cm1;
    TL_TENSOR_DATA_TO(C_minus->value, 0, Cm0, TL_FLOAT);
    TL_TENSOR_DATA_TO(C_minus->value, 1, Cm1, TL_FLOAT);
    float loss_minus = Cm0 + Cm1;
    printf("  A_data[0] = %.6f, loss = %.6f\n", A_data[0], loss_minus);

    /* Restore */
    A_data[0] = orig;

    float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
    float analytical;
    TL_TENSOR_DATA_TO(A->grad, 0, analytical, TL_FLOAT);
    printf("  Numerical gradient: %.6f\n", numerical);
    printf("  Analytical gradient: %.6f\n", analytical);
    printf("  Difference: %.6f\n", fabsf(numerical - analytical));

    /* Cleanup */
    tl_tensor_free(t_A);
    tl_tensor_free(t_B);
    tl_graph_free(g);
    tl_tensor_free(t_A_plus);
    tl_tensor_free(t_B_plus);
    tl_graph_free(g_plus);
    tl_tensor_free(t_A_minus);
    tl_tensor_free(t_B_minus);
    tl_graph_free(g_minus);

    return 0;
}
