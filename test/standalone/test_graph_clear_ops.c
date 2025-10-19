/*
 * Test to verify graph cleanup strategy prevents node accumulation
 */

#include <stdio.h>
#include <stdlib.h>
#include "src/tofu_graph.h"
#include "src/tofu_tensor.h"

int main(void)
{
    printf("Testing tofu_graph_clear_ops() to prevent node accumulation...\n\n");

    /* Create input and parameters once */
    float x_data[] = {1.0f, 2.0f};
    float w_data[] = {0.5f, 0.3f};
    float b_data[] = {0.1f};

    tofu_tensor* x_tensor = tofu_tensor_create(x_data, 1, (int[]){2}, TOFU_FLOAT);
    tofu_tensor* w_tensor = tofu_tensor_create(w_data, 1, (int[]){2}, TOFU_FLOAT);
    tofu_tensor* b_tensor = tofu_tensor_create(b_data, 1, (int[]){1}, TOFU_FLOAT);

    /* Create graph ONCE - will be reused across iterations */
    tofu_graph* g = tofu_graph_create();

    /* Create INPUT and PARAM nodes ONCE */
    tofu_graph_node* x = tofu_graph_input(g, x_tensor);
    tofu_graph_node* w = tofu_graph_param(g, w_tensor);
    tofu_graph_node* b = tofu_graph_param(g, b_tensor);

    printf("Initial graph: %d nodes (3 = 1 INPUT + 2 PARAM)\n", g->num_nodes);

    printf("\nRunning 10 training iterations with tofu_graph_clear_ops()...\n");
    for (int iter = 0; iter < 10; iter++) {
        /* Forward pass: y = sum(x * w) + b
         * This creates NEW operation nodes each iteration */
        tofu_graph_node* xw = tofu_graph_mul(g, x, w);
        tofu_graph_node* y = tofu_graph_add(g, xw, b);

        printf("  Iteration %d: graph has %d nodes", iter, g->num_nodes);

        /* Backward pass */
        float grad_data[] = {1.0f};
        y->grad = tofu_tensor_create_with_values(grad_data, 1, (int[]){1});
        tofu_graph_backward(g, y);

        printf(" (before clear_ops)\n");

        /* IMPORTANT: Clear operation nodes, keeping INPUT/PARAM nodes */
        tofu_graph_clear_ops(g);

        /* After clearing, should be back to 3 nodes */
        if (g->num_nodes != 3) {
            printf("  ERROR: Expected 3 nodes after clear_ops, got %d\n", g->num_nodes);
            return 1;
        }
    }

    printf("\nFinal graph: %d nodes (should still be 3)\n", g->num_nodes);
    printf("\nSUCCESS! Graph size remains constant across iterations.\n");
    printf("This prevents node accumulation and memory growth.\n");

    /* Cleanup */
    tofu_graph_free(g);
    tofu_tensor_free_data_too(x_tensor);
    tofu_tensor_free_data_too(w_tensor);
    tofu_tensor_free_data_too(b_tensor);

    return 0;
}
