#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/tl_graph.h"
#include "src/tl_tensor.h"
#include "src/tl_optimizer.h"

void test_correct_order() {
    printf("Testing CORRECT cleanup order...\n\n");
    
    tl_graph* g = tl_graph_create();
    
    float W_data[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    tl_tensor* t_W = tl_tensor_create(W_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_graph_node* W = tl_graph_param(g, t_W);
    
    tl_optimizer* opt = tl_optimizer_sgd_create(g, 0.1);
    
    /* Run training iterations */
    for (int iter = 0; iter < 5; iter++) {
        tl_optimizer_zero_grad(opt);
        
        float x_data[2] = {1.0f, 2.0f};
        tl_tensor* t_x = tl_tensor_create(x_data, 1, (int[]){2}, TL_FLOAT);
        tl_graph_node* x = tl_graph_input(g, t_x);
        
        tl_graph_node* out = tl_graph_matmul(g, x, W);
        out->grad = tl_tensor_create((float[]){1.0f, 1.0f}, 1, (int[]){2}, TL_FLOAT);
        tl_graph_backward(g, out);
        tl_optimizer_step(opt);
        
        tl_tensor_free(t_x);
    }
    
    printf("Graph has %d nodes after training\n", g->num_nodes);
    
    /* CORRECT ORDER: Free graph BEFORE freeing parameter tensors */
    printf("\nFreeing in correct order:\n");
    printf("1. Free optimizer...\n");
    tl_optimizer_free(opt);
    
    printf("2. Free graph...\n");
    clock_t start = clock();
    tl_graph_free(g);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("   graph_free took %.2f ms\n", time_ms);
    
    printf("3. Free parameter tensors...\n");
    tl_tensor_free(t_W);
    
    printf("\n✓ Success! Cleanup completed without hanging\n");
}

int main() {
    test_correct_order();
    return 0;
}
