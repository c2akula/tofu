#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/tl_graph.h"
#include "src/tl_tensor.h"

void test_graph_free_performance() {
    printf("Testing graph_free performance with different node counts...\n\n");
    
    int test_sizes[] = {10, 50, 100, 200};
    
    for (int t = 0; t < 4; t++) {
        int num_ops = test_sizes[t];
        
        tl_graph* g = tl_graph_create();
        
        /* Create input */
        float data[4] = {1, 2, 3, 4};
        tl_tensor* t_x = tl_tensor_create(data, 1, (int[]){4}, TL_FLOAT);
        tl_graph_node* x = tl_graph_input(g, t_x);
        
        /* Create many operations */
        tl_graph_node* result = x;
        for (int i = 0; i < num_ops; i++) {
            result = tl_graph_relu(g, result);
        }
        
        printf("Created graph with %d nodes (input + %d ops)\n", g->num_nodes, num_ops);
        
        /* Time the free operation */
        clock_t start = clock();
        tl_graph_free(g);
        clock_t end = clock();
        
        double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
        printf("  graph_free took %.2f ms\n\n", time_ms);
        
        tl_tensor_free(t_x);
    }
}

int main() {
    test_graph_free_performance();
    return 0;
}
