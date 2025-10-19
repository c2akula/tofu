#include <stdio.h>
#include "tl_graph.h"
#include "tl_tensor.h"

int main() {
    printf("Testing graph with reshape and free...\n");
    
    tl_graph* g = tl_graph_create();
    
    float img_data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    tl_tensor* t_img = tl_tensor_create(img_data, 1, (int[]){16}, TL_FLOAT);
    tl_graph_node* img = tl_graph_input(g, t_img);
    
    printf("  Reshaping [16] to [4, 4]...\n");
    tl_graph_node* reshaped = tl_graph_reshape(g, img, 2, (int[]){4, 4});
    printf("  Reshaped shape: [%d, %d]\n", reshaped->value->dims[0], reshaped->value->dims[1]);
    
    float W_data[32];
    for (int i = 0; i < 32; i++) W_data[i] = 0.1f;
    tl_tensor* t_W = tl_tensor_create(W_data, 2, (int[]){4, 8}, TL_FLOAT);
    tl_graph_node* W = tl_graph_param(g, t_W);
    
    printf("  Creating matmul...\n");
    tl_graph_node* output = tl_graph_matmul(g, reshaped, W);
    printf("  Output shape: [%d, %d]\n", output->value->dims[0], output->value->dims[1]);
    
    printf("  Freeing tensors...\n");
    tl_tensor_free(t_img);
    tl_tensor_free(t_W);
    
    printf("  Freeing graph...\n");
    tl_graph_free(g);
    
    printf("  ✓ SUCCESS!\n");
    return 0;
}
