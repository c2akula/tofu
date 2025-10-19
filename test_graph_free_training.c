#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/tl_graph.h"
#include "src/tl_tensor.h"
#include "src/tl_optimizer.h"

void test_training_graph_free() {
    printf("Testing graph_free after training iterations...\n\n");
    
    tl_graph* g = tl_graph_create();
    
    /* Create simple network */
    float W_data[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    tl_tensor* t_W = tl_tensor_create(W_data, 2, (int[]){2, 2}, TL_FLOAT);
    tl_graph_node* W = tl_graph_param(g, t_W);
    
    tl_optimizer* opt = tl_optimizer_sgd_create(g, 0.1);
    
    /* Run a few training iterations */
    printf("Running 5 training iterations...\n");
    for (int iter = 0; iter < 5; iter++) {
        tl_optimizer_zero_grad(opt);
        
        float x_data[2] = {1.0f, 2.0f};
        tl_tensor* t_x = tl_tensor_create(x_data, 1, (int[]){2}, TL_FLOAT);
        tl_graph_node* x = tl_graph_input(g, t_x);
        
        /* Forward */
        tl_graph_node* out = tl_graph_matmul(g, x, W);
        
        /* Backward */
        out->grad = tl_tensor_create((float[]){1.0f, 1.0f}, 1, (int[]){2}, TL_FLOAT);
        tl_graph_backward(g, out);
        
        /* Update */
        tl_optimizer_step(opt);
        
        tl_tensor_free(t_x);
        
        printf("  Iteration %d: graph has %d nodes\n", iter, g->num_nodes);
    }
    
    printf("\nAttempting to free graph with %d nodes...\n", g->num_nodes);
    clock_t start = clock();
    
    tl_tensor_free(t_W);
    tl_optimizer_free(opt);
    tl_graph_free(g);
    
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    printf("Cleanup took %.2f ms\n", time_ms);
    printf("✓ Success!\n");
}

int main() {
    test_training_graph_free();
    return 0;
}
