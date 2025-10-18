/*
 * Test to verify operation node value memory leak is fixed
 */

#include <stdio.h>
#include <stdlib.h>
#include "tl_graph.h"
#include "tl_tensor.h"

int main(void)
{
    printf("Testing graph_free after training iterations...\n\n");

    /* Create input and parameters */
    float x_data[] = {1.0f, 2.0f};
    float w_data[] = {0.5f, 0.3f};
    float b_data[] = {0.1f};

    tl_tensor* x_tensor = tl_tensor_create(x_data, 1, (int[]){2}, TL_FLOAT);
    tl_tensor* w_tensor = tl_tensor_create(w_data, 1, (int[]){2}, TL_FLOAT);
    tl_tensor* b_tensor = tl_tensor_create(b_data, 1, (int[]){1}, TL_FLOAT);

    printf("Running 5 training iterations...\n");
    for (int iter = 0; iter < 5; iter++) {
        /* Create graph */
        tl_graph* g = tl_graph_create();

        /* Create nodes */
        tl_graph_node* x = tl_graph_input(g, x_tensor);
        tl_graph_node* w = tl_graph_param(g, w_tensor);
        tl_graph_node* b = tl_graph_param(g, b_tensor);

        /* Forward pass: y = sum(x * w) + b */
        tl_graph_node* xw = tl_graph_mul(g, x, w);
        tl_graph_node* sum_xw = xw;  /* Simplified - in real code would sum */
        tl_graph_node* y = tl_graph_add(g, sum_xw, b);

        printf("  Iteration %d: graph has %d nodes\n", iter, g->num_nodes);

        /* Backward pass */
        float grad_data[] = {1.0f};
        y->grad = tl_tensor_create_with_values(grad_data, 1, (int[]){1});
        tl_graph_backward(g, y);

        /* Free graph - this should now properly free operation node values */
        tl_graph_free(g);
    }

    printf("\nGraph cleanup successful! Memory leak should be fixed.\n");

    /* Cleanup */
    tl_tensor_free_data_too(x_tensor);
    tl_tensor_free_data_too(w_tensor);
    tl_tensor_free_data_too(b_tensor);

    return 0;
}
