#include <stdio.h>
#include "src/tofu_graph.h"
#include "src/tofu_tensor.h"

int main() {
    printf("Creating minimal test case...\n");
    
    tofu_graph* g = tofu_graph_create();
    
    /* Create a single operation with backward */
    float x_data[2] = {1.0f, 2.0f};
    tofu_tensor* t_x = tofu_tensor_create(x_data, 1, (int[]){2}, TOFU_FLOAT);
    tofu_graph_node* x = tofu_graph_input(g, t_x);
    tofu_graph_node* out = tofu_graph_relu(g, x);
    
    /* Run backward */
    out->grad = tofu_tensor_create((float[]){1.0f, 1.0f}, 1, (int[]){2}, TOFU_FLOAT);
    tofu_graph_backward(g, out);
    
    printf("Graph has %d nodes\n", g->num_nodes);
    printf("Attempting tofu_graph_free...\n");
    fflush(stdout);
    
    tofu_graph_free(g);
    
    printf("Success!\n");
    tofu_tensor_free(t_x);
    return 0;
}
