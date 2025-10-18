/*
 * Test computation graph infrastructure
 */

#include <stdio.h>
#include <assert.h>
#include "tl_graph.h"

/* Sprint 1 Tests: Core Infrastructure */

void test_graph_create_free()
{
    printf("Test: Graph create/free...\n");

    tl_graph* g = tl_graph_create();
    assert(g != NULL);
    assert(g->num_nodes == 0);
    assert(g->next_id == 0);

    tl_graph_free(g);
    printf("  ✓ PASSED\n");
}

void test_graph_input_node()
{
    printf("Test: Input node creation...\n");

    tl_graph* g = tl_graph_create();

    /* Create input data */
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tl_tensor* t = tl_tensor_create(data, 2, (int[]){2, 2}, TL_FLOAT);

    /* Add input node */
    tl_graph_node* x = tl_graph_input(g, t);
    assert(x != NULL);
    assert(x->op == TL_OP_INPUT);
    assert(x->value == t);
    assert(x->requires_grad == 0);
    assert(x->id == 0);
    assert(g->num_nodes == 1);

    /* Verify value access */
    tl_tensor* val = tl_graph_get_value(x);
    assert(val == t);

    tl_tensor_free(t);
    tl_graph_free(g);
    printf("  ✓ PASSED\n");
}

void test_graph_param_node()
{
    printf("Test: Parameter node creation...\n");

    tl_graph* g = tl_graph_create();

    /* Create parameter data */
    float data[] = {0.5f, -0.3f, 0.2f, 0.1f};
    tl_tensor* t = tl_tensor_create(data, 2, (int[]){2, 2}, TL_FLOAT);

    /* Add parameter node */
    tl_graph_node* W = tl_graph_param(g, t);
    assert(W != NULL);
    assert(W->op == TL_OP_PARAM);
    assert(W->value == t);
    assert(W->requires_grad == 1);  /* Parameters require gradients */
    assert(W->id == 0);
    assert(g->num_nodes == 1);

    tl_tensor_free(t);
    tl_graph_free(g);
    printf("  ✓ PASSED\n");
}

void test_graph_multiple_nodes()
{
    printf("Test: Multiple nodes...\n");

    tl_graph* g = tl_graph_create();

    float data1[] = {1.0f, 2.0f};
    float data2[] = {3.0f, 4.0f};
    float data3[] = {5.0f, 6.0f};

    tl_tensor* t1 = tl_tensor_create(data1, 1, (int[]){2}, TL_FLOAT);
    tl_tensor* t2 = tl_tensor_create(data2, 1, (int[]){2}, TL_FLOAT);
    tl_tensor* t3 = tl_tensor_create(data3, 1, (int[]){2}, TL_FLOAT);

    tl_graph_node* x = tl_graph_input(g, t1);
    tl_graph_node* W = tl_graph_param(g, t2);
    tl_graph_node* b = tl_graph_param(g, t3);

    assert(g->num_nodes == 3);
    assert(x->id == 0);
    assert(W->id == 1);
    assert(b->id == 2);

    assert(x->requires_grad == 0);
    assert(W->requires_grad == 1);
    assert(b->requires_grad == 1);

    tl_tensor_free(t1);
    tl_tensor_free(t2);
    tl_tensor_free(t3);
    tl_graph_free(g);
    printf("  ✓ PASSED\n");
}

void test_graph_capacity_expansion()
{
    printf("Test: Graph capacity expansion...\n");

    tl_graph* g = tl_graph_create();
    int initial_capacity = g->capacity;

    /* Add nodes beyond initial capacity */
    tl_tensor* dummy = tl_tensor_zeros(1, (int[]){1}, TL_FLOAT);

    for (int i = 0; i < initial_capacity + 10; i++) {
        tl_graph_node* node = tl_graph_input(g, dummy);
        assert(node != NULL);
        assert(node->id == i);
    }

    assert(g->num_nodes == initial_capacity + 10);
    assert(g->capacity > initial_capacity);

    tl_tensor_free_data_too(dummy);
    tl_graph_free(g);
    printf("  ✓ PASSED\n");
}

void test_graph_zero_grad()
{
    printf("Test: Zero gradients...\n");

    tl_graph* g = tl_graph_create();

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    tl_tensor* t = tl_tensor_create(data, 1, (int[]){4}, TL_FLOAT);

    tl_graph_node* W = tl_graph_param(g, t);

    /* Manually set gradient */
    W->grad = tl_tensor_zeros(1, (int[]){4}, TL_FLOAT);
    float grad_val = 5.0f;
    for (int i = 0; i < 4; i++) {
        TL_TENSOR_DATA_FROM(W->grad, i, grad_val, TL_FLOAT);
    }

    /* Verify gradient is set */
    float check_val;
    TL_TENSOR_DATA_TO(W->grad, 0, check_val, TL_FLOAT);
    assert(check_val == 5.0f);

    /* Zero gradients */
    tl_graph_zero_grad(g);

    /* Verify gradient is zeroed */
    TL_TENSOR_DATA_TO(W->grad, 0, check_val, TL_FLOAT);
    assert(check_val == 0.0f);

    tl_tensor_free(t);
    tl_graph_free(g);
    printf("  ✓ PASSED\n");
}

int main()
{
    printf("============================================================\n");
    printf("Sprint 1: Core Graph Infrastructure Tests\n");
    printf("============================================================\n\n");

    test_graph_create_free();
    test_graph_input_node();
    test_graph_param_node();
    test_graph_multiple_nodes();
    test_graph_capacity_expansion();
    test_graph_zero_grad();

    printf("\n============================================================\n");
    printf("Sprint 1: All tests passed! ✓\n");
    printf("============================================================\n");

    return 0;
}
