#include <stdio.h>
#include "tofu_graph.h"
#include "tofu_tensor.h"

int main() {
    printf("Testing graph with reshape and free...\n");
    
    tofu_graph* g = tofu_graph_create();
    
    float img_data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    tofu_tensor* t_img = tofu_tensor_create(img_data, 1, (int[]){16}, TOFU_FLOAT);
    tofu_graph_node* img = tofu_graph_input(g, t_img);
    
    printf("  Reshaping [16] to [4, 4]...\n");
    tofu_graph_node* reshaped = tofu_graph_reshape(g, img, 2, (int[]){4, 4});
    printf("  Reshaped shape: [%d, %d]\n", reshaped->value->dims[0], reshaped->value->dims[1]);
    
    float W_data[32];
    for (int i = 0; i < 32; i++) W_data[i] = 0.1f;
    tofu_tensor* t_W = tofu_tensor_create(W_data, 2, (int[]){4, 8}, TOFU_FLOAT);
    tofu_graph_node* W = tofu_graph_param(g, t_W);
    
    printf("  Creating matmul...\n");
    tofu_graph_node* output = tofu_graph_matmul(g, reshaped, W);
    printf("  Output shape: [%d, %d]\n", output->value->dims[0], output->value->dims[1]);
    
    printf("  Freeing tensors...\n");
    tofu_tensor_free(t_img);
    tofu_tensor_free(t_W);
    
    printf("  Freeing graph...\n");
    tofu_graph_free(g);
    
    printf("  ✓ SUCCESS!\n");
    return 0;
}
