#include <stdio.h>
#include "src/tl_graph.h"
#include "src/tl_tensor.h"

int main() {
    printf("Creating minimal test case...\n");
    
    tl_graph* g = tl_graph_create();
    
    /* Create a single operation with backward */
    float x_data[2] = {1.0f, 2.0f};
    tl_tensor* t_x = tl_tensor_create(x_data, 1, (int[]){2}, TL_FLOAT);
    tl_graph_node* x = tl_graph_input(g, t_x);
    tl_graph_node* out = tl_graph_relu(g, x);
    
    /* Run backward */
    out->grad = tl_tensor_create((float[]){1.0f, 1.0f}, 1, (int[]){2}, TL_FLOAT);
    tl_graph_backward(g, out);
    
    printf("Graph has %d nodes\n", g->num_nodes);
    printf("Attempting tl_graph_free...\n");
    fflush(stdout);
    
    tl_graph_free(g);
    
    printf("Success!\n");
    tl_tensor_free(t_x);
    return 0;
}
