#include <stdio.h>
#include "tl_graph.h"
#include "tl_tensor.h"

int main() {
    printf("Testing simple graph with forward pass and free...\n");
    
    tl_graph* g = tl_graph_create();
    
    float data[4] = {1,2,3,4};
    tl_tensor* t = tl_tensor_create(data, 1, (int[]){4}, TL_FLOAT);
    tl_graph_node* input = tl_graph_input(g, t);
    
    float W_data[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    tl_tensor* t_W = tl_tensor_create(W_data, 2, (int[]){4, 2}, TL_FLOAT);
    tl_graph_node* W = tl_graph_param(g, t_W);
    
    printf("  Creating matmul node...\n");
    tl_graph_node* output = tl_graph_matmul(g, input, W);
    
    printf("  Output shape: [%d, %d]\n", output->value->dims[0], output->value->dims[1]);
    printf("  Graph has %d nodes\n", g->num_nodes);
    
    printf("  Freeing tensors...\n");
    tl_tensor_free(t);
    tl_tensor_free(t_W);
    
    printf("  Freeing graph...\n");
    tl_graph_free(g);
    
    printf("  ✓ SUCCESS - graph freed without hanging!\n");
    return 0;
}
