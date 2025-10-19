#include <stdio.h>
#include "src/tofu_graph.h"
#include "src/tofu_tensor.h"

int main() {
    printf("Testing with properly allocated gradient...\n");
    
    tofu_graph* g = tofu_graph_create();
    
    float x_data[2] = {1.0f, 2.0f};
    tofu_tensor* t_x = tofu_tensor_create(x_data, 1, (int[]){2}, TOFU_FLOAT);
    tofu_graph_node* x = tofu_graph_input(g, t_x);
    tofu_graph_node* out = tofu_graph_relu(g, x);
    
    /* Create gradient on HEAP, not stack */
    float* grad_data = (float*)malloc(2 * sizeof(float));
    grad_data[0] = 1.0f;
    grad_data[1] = 1.0f;
    out->grad = tofu_tensor_create(grad_data, 1, (int[]){2}, TOFU_FLOAT);
    
    tofu_graph_backward(g, out);
    
    printf("Backward pass completed\n");
    printf("Attempting tofu_graph_free...\n");
    fflush(stdout);
    
    tofu_graph_free(g);
    
    printf("Success!\n");
    tofu_tensor_free(t_x);
    free(grad_data);
    return 0;
}
