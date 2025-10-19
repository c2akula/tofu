#include <stdio.h>
#include "tofu_graph.h"
#include "tofu_tensor.h"

int main() {
    printf("Testing simple graph with forward pass and free...\n");
    
    tofu_graph* g = tofu_graph_create();
    
    float data[4] = {1,2,3,4};
    tofu_tensor* t = tofu_tensor_create(data, 1, (int[]){4}, TOFU_FLOAT);
    tofu_graph_node* input = tofu_graph_input(g, t);
    
    float W_data[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    tofu_tensor* t_W = tofu_tensor_create(W_data, 2, (int[]){4, 2}, TOFU_FLOAT);
    tofu_graph_node* W = tofu_graph_param(g, t_W);
    
    printf("  Creating matmul node...\n");
    tofu_graph_node* output = tofu_graph_matmul(g, input, W);
    
    printf("  Output shape: [%d, %d]\n", output->value->dims[0], output->value->dims[1]);
    printf("  Graph has %d nodes\n", g->num_nodes);
    
    printf("  Freeing tensors...\n");
    tofu_tensor_free(t);
    tofu_tensor_free(t_W);
    
    printf("  Freeing graph...\n");
    tofu_graph_free(g);
    
    printf("  ✓ SUCCESS - graph freed without hanging!\n");
    return 0;
}
