#include <stdio.h>
#include "src/tl_graph.h"
#include "src/tl_tensor.h"

int main() {
    printf("Testing with properly allocated gradient...\n");
    
    tl_graph* g = tl_graph_create();
    
    float x_data[2] = {1.0f, 2.0f};
    tl_tensor* t_x = tl_tensor_create(x_data, 1, (int[]){2}, TL_FLOAT);
    tl_graph_node* x = tl_graph_input(g, t_x);
    tl_graph_node* out = tl_graph_relu(g, x);
    
    /* Create gradient on HEAP, not stack */
    float* grad_data = (float*)malloc(2 * sizeof(float));
    grad_data[0] = 1.0f;
    grad_data[1] = 1.0f;
    out->grad = tl_tensor_create(grad_data, 1, (int[]){2}, TL_FLOAT);
    
    tl_graph_backward(g, out);
    
    printf("Backward pass completed\n");
    printf("Attempting tl_graph_free...\n");
    fflush(stdout);
    
    tl_graph_free(g);
    
    printf("Success!\n");
    tl_tensor_free(t_x);
    free(grad_data);
    return 0;
}
