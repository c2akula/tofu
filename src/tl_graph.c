/*
 * Copyright (c) 2018-2020 Zhixu Zhao
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tl_graph.h"

#define TL_GRAPH_INITIAL_CAPACITY 32

/* Create a new computation graph */
TL_EXPORT tl_graph* tl_graph_create(void)
{
    tl_graph* g = (tl_graph*)malloc(sizeof(tl_graph));
    if (!g)
        return NULL;

    g->capacity = TL_GRAPH_INITIAL_CAPACITY;
    g->nodes = (tl_graph_node**)malloc(g->capacity * sizeof(tl_graph_node*));
    if (!g->nodes) {
        free(g);
        return NULL;
    }

    g->num_nodes = 0;
    g->next_id = 0;

    /* Initialize topological order arrays */
    g->topo_capacity = TL_GRAPH_INITIAL_CAPACITY;
    g->topo_order = (tl_graph_node**)malloc(g->topo_capacity * sizeof(tl_graph_node*));
    if (!g->topo_order) {
        free(g->nodes);
        free(g);
        return NULL;
    }
    g->topo_size = 0;

    return g;
}

/* Free a graph node */
static void tl_graph_node_free(tl_graph_node* node)
{
    if (!node)
        return;

    /* Free gradient if it exists */
    if (node->grad) {
        tl_tensor_free_data_too(node->grad);
        node->grad = NULL;
    }

    /* Free inputs array */
    if (node->inputs) {
        free(node->inputs);
        node->inputs = NULL;
    }

    /* Free backward context if it exists */
    if (node->backward_ctx) {
        free(node->backward_ctx);
        node->backward_ctx = NULL;
    }

    /* Note: We don't free node->value here because:
     * - For INPUT/PARAM nodes, value is owned by user
     * - For operation nodes, value will be freed by operations
     */

    free(node);
}

/* Free a computation graph */
TL_EXPORT void tl_graph_free(tl_graph* g)
{
    if (!g)
        return;

    /* Free all nodes */
    for (int i = 0; i < g->num_nodes; i++) {
        tl_graph_node_free(g->nodes[i]);
    }

    /* Free arrays */
    free(g->nodes);
    free(g->topo_order);
    free(g);
}

/* Add a node to the graph */
static tl_graph_node* tl_graph_add_node(tl_graph* g, tl_op_type op)
{
    assert(g);

    /* Expand capacity if needed */
    if (g->num_nodes >= g->capacity) {
        int new_capacity = g->capacity * 2;
        tl_graph_node** new_nodes = (tl_graph_node**)realloc(
            g->nodes, new_capacity * sizeof(tl_graph_node*));
        if (!new_nodes)
            return NULL;
        g->nodes = new_nodes;
        g->capacity = new_capacity;
    }

    /* Create node */
    tl_graph_node* node = (tl_graph_node*)malloc(sizeof(tl_graph_node));
    if (!node)
        return NULL;

    /* Initialize node */
    node->id = g->next_id++;
    node->op = op;
    node->value = NULL;
    node->grad = NULL;
    node->inputs = NULL;
    node->num_inputs = 0;
    node->capacity_inputs = 0;
    node->backward_fn = NULL;
    node->backward_ctx = NULL;
    node->requires_grad = 0;
    node->visited = 0;
    node->graph = g;

    /* Add to graph */
    g->nodes[g->num_nodes++] = node;

    return node;
}

/* Create input node (no gradient) */
TL_EXPORT tl_graph_node* tl_graph_input(tl_graph* g, tl_tensor* data)
{
    assert(g && data);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_INPUT);
    if (!node)
        return NULL;

    node->value = data;
    node->requires_grad = 0;  /* Inputs don't require gradients */

    return node;
}

/* Create parameter node (requires gradient) */
TL_EXPORT tl_graph_node* tl_graph_param(tl_graph* g, tl_tensor* data)
{
    assert(g && data);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_PARAM);
    if (!node)
        return NULL;

    node->value = data;
    node->requires_grad = 1;  /* Parameters require gradients */

    return node;
}

/* Get value tensor from node */
TL_EXPORT tl_tensor* tl_graph_get_value(tl_graph_node* node)
{
    assert(node);
    return node->value;
}

/* Get gradient tensor from node */
TL_EXPORT tl_tensor* tl_graph_get_grad(tl_graph_node* node)
{
    assert(node);
    return node->grad;
}

/* Zero all gradients in graph */
TL_EXPORT void tl_graph_zero_grad(tl_graph* g)
{
    assert(g);

    for (int i = 0; i < g->num_nodes; i++) {
        tl_graph_node* node = g->nodes[i];
        if (node->grad) {
            /* Zero out gradient tensor */
            for (int j = 0; j < node->grad->len; j++) {
                double zero = 0.0;
                TL_TENSOR_DATA_FROM(node->grad, j, zero, TL_DOUBLE);
            }
        }
    }
}

/* Helper: Add input to node */
static int tl_graph_node_add_input(tl_graph_node* node, tl_graph_node* input)
{
    assert(node && input);

    /* Expand capacity if needed */
    if (node->num_inputs >= node->capacity_inputs) {
        int new_capacity = node->capacity_inputs == 0 ? 2 : node->capacity_inputs * 2;
        tl_graph_node** new_inputs = (tl_graph_node**)realloc(
            node->inputs, new_capacity * sizeof(tl_graph_node*));
        if (!new_inputs)
            return -1;
        node->inputs = new_inputs;
        node->capacity_inputs = new_capacity;
    }

    node->inputs[node->num_inputs++] = input;
    return 0;
}

/* Helper: Check if any input requires gradient */
static int tl_graph_any_requires_grad(tl_graph_node** inputs, int num_inputs)
{
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i]->requires_grad)
            return 1;
    }
    return 0;
}

/* Matrix multiplication: y = a @ b */
TL_EXPORT tl_graph_node* tl_graph_matmul(tl_graph* g, tl_graph_node* a, tl_graph_node* b)
{
    assert(g && a && b);
    assert(a->value && b->value);

    /* Create node */
    tl_graph_node* node = tl_graph_add_node(g, TL_OP_MATMUL);
    if (!node)
        return NULL;

    /* Add inputs */
    if (tl_graph_node_add_input(node, a) < 0 || tl_graph_node_add_input(node, b) < 0) {
        return NULL;
    }

    /* Compute forward pass */
    node->value = tl_tensor_matmul(a->value, b->value, NULL);
    if (!node->value)
        return NULL;

    /* Determine if gradient is required */
    node->requires_grad = tl_graph_any_requires_grad(node->inputs, node->num_inputs);

    return node;
}

/* Element-wise addition: z = a + b */
TL_EXPORT tl_graph_node* tl_graph_add(tl_graph* g, tl_graph_node* a, tl_graph_node* b)
{
    assert(g && a && b);
    assert(a->value && b->value);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_ADD);
    if (!node)
        return NULL;

    if (tl_graph_node_add_input(node, a) < 0 || tl_graph_node_add_input(node, b) < 0) {
        return NULL;
    }

    /* Use broadcasting if shapes don't match exactly */
    node->value = tl_tensor_elew_broadcast(a->value, b->value, NULL, TL_SUM);
    if (!node->value)
        return NULL;

    node->requires_grad = tl_graph_any_requires_grad(node->inputs, node->num_inputs);

    return node;
}

/* Element-wise multiplication: z = a * b */
TL_EXPORT tl_graph_node* tl_graph_mul(tl_graph* g, tl_graph_node* a, tl_graph_node* b)
{
    assert(g && a && b);
    assert(a->value && b->value);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_MUL);
    if (!node)
        return NULL;

    if (tl_graph_node_add_input(node, a) < 0 || tl_graph_node_add_input(node, b) < 0) {
        return NULL;
    }

    node->value = tl_tensor_elew_broadcast(a->value, b->value, NULL, TL_MUL);
    if (!node->value)
        return NULL;

    node->requires_grad = tl_graph_any_requires_grad(node->inputs, node->num_inputs);

    return node;
}

/* ReLU activation: y = max(0, x) */
TL_EXPORT tl_graph_node* tl_graph_relu(tl_graph* g, tl_graph_node* x)
{
    assert(g && x);
    assert(x->value);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_RELU);
    if (!node)
        return NULL;

    if (tl_graph_node_add_input(node, x) < 0) {
        return NULL;
    }

    /* ReLU: max(0, x) - using lrelu with negslope=0 */
    node->value = tl_tensor_lrelu(x->value, NULL, 0.0f);
    if (!node->value)
        return NULL;

    node->requires_grad = x->requires_grad;

    return node;
}

/* Softmax activation: y = softmax(x) along axis */
TL_EXPORT tl_graph_node* tl_graph_softmax(tl_graph* g, tl_graph_node* x, int axis)
{
    assert(g && x);
    assert(x->value);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_SOFTMAX);
    if (!node)
        return NULL;

    if (tl_graph_node_add_input(node, x) < 0) {
        return NULL;
    }

    /* Store axis in backward context for later */
    int* axis_ctx = (int*)malloc(sizeof(int));
    if (!axis_ctx)
        return NULL;
    *axis_ctx = axis;
    node->backward_ctx = axis_ctx;

    node->value = tl_tensor_softmax(x->value, NULL, axis);
    if (!node->value)
        return NULL;

    node->requires_grad = x->requires_grad;

    return node;
}

/* Layer normalization */
TL_EXPORT tl_graph_node* tl_graph_layer_norm(tl_graph* g, tl_graph_node* x,
                                             tl_graph_node* gamma, tl_graph_node* beta,
                                             int axis, double eps)
{
    assert(g && x);
    assert(x->value);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_LAYER_NORM);
    if (!node)
        return NULL;

    if (tl_graph_node_add_input(node, x) < 0)
        return NULL;
    if (gamma && tl_graph_node_add_input(node, gamma) < 0)
        return NULL;
    if (beta && tl_graph_node_add_input(node, beta) < 0)
        return NULL;

    /* Store axis and eps in backward context */
    typedef struct {
        int axis;
        double eps;
    } layer_norm_ctx;

    layer_norm_ctx* ctx = (layer_norm_ctx*)malloc(sizeof(layer_norm_ctx));
    if (!ctx)
        return NULL;
    ctx->axis = axis;
    ctx->eps = eps;
    node->backward_ctx = ctx;

    node->value = tl_tensor_layer_norm(x->value, NULL,
                                       gamma ? gamma->value : NULL,
                                       beta ? beta->value : NULL,
                                       axis, eps);
    if (!node->value)
        return NULL;

    node->requires_grad = tl_graph_any_requires_grad(node->inputs, node->num_inputs);

    return node;
}

/* Reshape operation */
TL_EXPORT tl_graph_node* tl_graph_reshape(tl_graph* g, tl_graph_node* x, int ndim, const int* dims)
{
    assert(g && x);
    assert(x->value);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_RESHAPE);
    if (!node)
        return NULL;

    if (tl_graph_node_add_input(node, x) < 0)
        return NULL;

    /* Store original shape for backward pass */
    int* shape_ctx = (int*)malloc((x->value->ndim + 1) * sizeof(int));
    if (!shape_ctx)
        return NULL;
    shape_ctx[0] = x->value->ndim;
    for (int i = 0; i < x->value->ndim; i++) {
        shape_ctx[i + 1] = x->value->dims[i];
    }
    node->backward_ctx = shape_ctx;

    node->value = tl_tensor_reshape(x->value, ndim, dims);
    if (!node->value)
        return NULL;

    node->requires_grad = x->requires_grad;

    return node;
}

/* Transpose operation */
TL_EXPORT tl_graph_node* tl_graph_transpose(tl_graph* g, tl_graph_node* x, const int* axes)
{
    assert(g && x);
    assert(x->value);

    tl_graph_node* node = tl_graph_add_node(g, TL_OP_TRANSPOSE);
    if (!node)
        return NULL;

    if (tl_graph_node_add_input(node, x) < 0)
        return NULL;

    /* Store axes for backward pass */
    int* axes_ctx = NULL;
    if (axes) {
        axes_ctx = (int*)malloc(x->value->ndim * sizeof(int));
        if (!axes_ctx)
            return NULL;
        for (int i = 0; i < x->value->ndim; i++) {
            axes_ctx[i] = axes[i];
        }
    }
    node->backward_ctx = axes_ctx;

    node->value = tl_tensor_transpose(x->value, NULL, axes);
    if (!node->value)
        return NULL;

    node->requires_grad = x->requires_grad;

    return node;
}
