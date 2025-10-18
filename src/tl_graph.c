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
#include <math.h>
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

    /* Free value tensor for operation nodes only
     * - For INPUT/PARAM nodes, value is owned by user
     * - For operation nodes, value is allocated by the operation and must be freed
     * - For view operations (reshape, transpose), only free tensor struct, not data
     */
    if (node->value && node->op != TL_OP_INPUT && node->op != TL_OP_PARAM) {
        if (node->value->owner) {
            /* This is a view (reshape/transpose) - only free tensor struct */
            tl_tensor_free(node->value);
        } else {
            /* This allocates its own data - free everything */
            tl_tensor_free_data_too(node->value);
        }
        node->value = NULL;
    }

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

/* Clear operation nodes from graph, keeping INPUT and PARAM nodes
 * This is useful in training loops to avoid node accumulation
 */
TL_EXPORT void tl_graph_clear_ops(tl_graph* g)
{
    if (!g)
        return;

    /* Keep INPUT and PARAM nodes, remove all operation nodes */
    int write_idx = 0;
    for (int read_idx = 0; read_idx < g->num_nodes; read_idx++) {
        tl_graph_node* node = g->nodes[read_idx];
        if (node->op == TL_OP_INPUT || node->op == TL_OP_PARAM) {
            /* Zero out gradients for reused nodes */
            if (node->grad) {
                for (int j = 0; j < node->grad->len; j++) {
                    double zero = 0.0;
                    TL_TENSOR_DATA_FROM(node->grad, j, zero, TL_DOUBLE);
                }
            }
            g->nodes[write_idx++] = node;
        } else {
            /* Free operation node */
            tl_graph_node_free(node);
        }
    }
    g->num_nodes = write_idx;

    /* Clear topological order */
    g->topo_size = 0;
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

/* ========================================================================
 * Backward Pass Implementation
 * ======================================================================== */

/* Topological sort using DFS */
static void topo_sort_dfs(tl_graph_node* node, tl_graph* g)
{
    if (node->visited)
        return;

    node->visited = 1;

    /* Visit all inputs first */
    for (int i = 0; i < node->num_inputs; i++) {
        topo_sort_dfs(node->inputs[i], g);
    }

    /* Add to topological order */
    if (g->topo_size >= g->topo_capacity) {
        int new_capacity = g->topo_capacity * 2;
        tl_graph_node** new_order = (tl_graph_node**)realloc(
            g->topo_order, new_capacity * sizeof(tl_graph_node*));
        if (!new_order)
            return;
        g->topo_order = new_order;
        g->topo_capacity = new_capacity;
    }

    g->topo_order[g->topo_size++] = node;
}

/* Build topological order for backward pass */
static void tl_graph_build_topo(tl_graph* g, tl_graph_node* root)
{
    /* Reset visited flags */
    for (int i = 0; i < g->num_nodes; i++) {
        g->nodes[i]->visited = 0;
    }

    /* Build topological order */
    g->topo_size = 0;
    topo_sort_dfs(root, g);
}

/* Helper: Accumulate gradient (add to existing gradient) */
static void accumulate_grad(tl_graph_node* node, tl_tensor* grad_contrib)
{
    if (!node->requires_grad)
        return;

    if (!node->grad) {
        /* First gradient contribution - just store it */
        node->grad = tl_tensor_clone(grad_contrib);
    } else {
        /* Accumulate (add to existing gradient) */
        tl_tensor* new_grad = tl_tensor_elew_broadcast(node->grad, grad_contrib, NULL, TL_SUM);
        tl_tensor_free_data_too(node->grad);
        node->grad = new_grad;
    }
}

/* Backward functions for each operation */

/* Matmul backward: y = A @ B */
static void matmul_backward(tl_graph_node* node)
{
    assert(node->op == TL_OP_MATMUL);
    assert(node->num_inputs == 2);

    tl_graph_node* A = node->inputs[0];
    tl_graph_node* B = node->inputs[1];
    tl_tensor* grad_y = node->grad;

    if (!grad_y)
        return;

    /* Handle 1D tensors by reshaping to 2D for gradient computation */
    int is_A_1d = (A->value->ndim == 1);
    int is_grad_y_1d = (grad_y->ndim == 1);

    tl_tensor* grad_y_2d = NULL;
    if (is_grad_y_1d && grad_y->dims[0] == 1) {
        /* Reshape [1] to [1, 1] */
        int new_dims[] = {1, 1};
        grad_y_2d = tl_tensor_reshape(grad_y, 2, new_dims);
    } else if (is_grad_y_1d) {
        /* Reshape [n] to [1, n] */
        int new_dims[] = {1, grad_y->dims[0]};
        grad_y_2d = tl_tensor_reshape(grad_y, 2, new_dims);
    } else {
        grad_y_2d = grad_y;
    }

    /* ∂L/∂A = (∂L/∂y) @ B^T */
    if (A->requires_grad) {
        tl_tensor* B_T = tl_tensor_transpose(B->value, NULL, NULL);
        tl_tensor* grad_A = tl_tensor_matmul(grad_y_2d, B_T, NULL);

        /* Reshape back to original shape if needed */
        if (is_A_1d && grad_A->ndim == 2) {
            if (grad_A->dims[0] == 1) {
                /* Squeeze [1, n] to [n] */
                int new_dims[] = {grad_A->dims[1]};
                tl_tensor* grad_A_1d = tl_tensor_reshape(grad_A, 1, new_dims);
                tl_tensor_free(grad_A);
                grad_A = grad_A_1d;
            }
        }

        accumulate_grad(A, grad_A);
        tl_tensor_free_data_too(B_T);
        tl_tensor_free_data_too(grad_A);
    }

    /* ∂L/∂B = A^T @ (∂L/∂y) */
    if (B->requires_grad) {
        tl_tensor* A_val = A->value;
        tl_tensor* A_2d = NULL;

        /* If A is 1D, reshape to row vector [1, n] (matching forward behavior) */
        if (is_A_1d) {
            int new_dims[] = {1, A->value->dims[0]};
            A_2d = tl_tensor_reshape(A->value, 2, new_dims);
            A_val = A_2d;
        }

        /* Transpose: [1, n] → [n, 1] */
        tl_tensor* A_T = tl_tensor_transpose(A_val, NULL, NULL);

        /* Reshape grad_y if needed for matmul with A_T */
        tl_tensor* grad_y_for_B = grad_y;
        tl_tensor* grad_y_B_reshaped = NULL;
        if (is_grad_y_1d) {
            /* Reshape [n] to [1, n] to match matmul semantics */
            int new_dims[] = {1, grad_y->dims[0]};
            grad_y_B_reshaped = tl_tensor_reshape(grad_y, 2, new_dims);
            grad_y_for_B = grad_y_B_reshaped;
        }

        tl_tensor* grad_B = tl_tensor_matmul(A_T, grad_y_for_B, NULL);

        accumulate_grad(B, grad_B);

        if (A_2d)
            tl_tensor_free(A_2d);
        if (grad_y_B_reshaped)
            tl_tensor_free(grad_y_B_reshaped);
        tl_tensor_free_data_too(A_T);
        tl_tensor_free_data_too(grad_B);
    }

    /* Free temporary grad_y_2d if we created it */
    if (grad_y_2d != grad_y && grad_y_2d) {
        tl_tensor_free(grad_y_2d);
    }
}

/* Add backward: z = x + y */
static void add_backward(tl_graph_node* node)
{
    assert(node->op == TL_OP_ADD);
    assert(node->num_inputs == 2);

    tl_graph_node* x = node->inputs[0];
    tl_graph_node* y = node->inputs[1];
    tl_tensor* grad_z = node->grad;

    if (!grad_z)
        return;

    /* ∂L/∂x = ∂L/∂z (sum over broadcast dimensions if needed) */
    if (x->requires_grad) {
        /* TODO: Handle broadcasting properly - for now assume same shape */
        accumulate_grad(x, grad_z);
    }

    /* ∂L/∂y = ∂L/∂z */
    if (y->requires_grad) {
        accumulate_grad(y, grad_z);
    }
}

/* Mul backward: z = x * y */
static void mul_backward(tl_graph_node* node)
{
    assert(node->op == TL_OP_MUL);
    assert(node->num_inputs == 2);

    tl_graph_node* x = node->inputs[0];
    tl_graph_node* y = node->inputs[1];
    tl_tensor* grad_z = node->grad;

    if (!grad_z)
        return;

    /* ∂L/∂x = ∂L/∂z * y */
    if (x->requires_grad) {
        tl_tensor* grad_x = tl_tensor_elew_broadcast(grad_z, y->value, NULL, TL_MUL);
        accumulate_grad(x, grad_x);
        tl_tensor_free_data_too(grad_x);
    }

    /* ∂L/∂y = ∂L/∂z * x */
    if (y->requires_grad) {
        tl_tensor* grad_y = tl_tensor_elew_broadcast(grad_z, x->value, NULL, TL_MUL);
        accumulate_grad(y, grad_y);
        tl_tensor_free_data_too(grad_y);
    }
}

/* ReLU backward: y = max(0, x) */
static void relu_backward(tl_graph_node* node)
{
    assert(node->op == TL_OP_RELU);
    assert(node->num_inputs == 1);

    tl_graph_node* x = node->inputs[0];
    tl_tensor* grad_y = node->grad;

    if (!grad_y || !x->requires_grad)
        return;

    /* ∂L/∂x = ∂L/∂y * (x > 0) */
    tl_tensor* grad_x = tl_tensor_zeros(x->value->ndim, x->value->dims, x->value->dtype);

    for (int i = 0; i < x->value->len; i++) {
        float x_val, grad_y_val;
        TL_TENSOR_DATA_TO(x->value, i, x_val, TL_FLOAT);
        TL_TENSOR_DATA_TO(grad_y, i, grad_y_val, TL_FLOAT);

        /* Gradient passes through only if x > 0 */
        float grad = (x_val > 0.0f) ? grad_y_val : 0.0f;
        TL_TENSOR_DATA_FROM(grad_x, i, grad, TL_FLOAT);
    }

    accumulate_grad(x, grad_x);
    tl_tensor_free_data_too(grad_x);
}

/* Softmax backward: y = softmax(x, axis) */
static void softmax_backward(tl_graph_node* node)
{
    assert(node->op == TL_OP_SOFTMAX);
    assert(node->num_inputs == 1);

    tl_graph_node* x = node->inputs[0];
    tl_tensor* grad_y = node->grad;
    tl_tensor* y = node->value;  /* softmax output */

    if (!grad_y || !x->requires_grad)
        return;

    int axis = *(int*)node->backward_ctx;

    /* ∂L/∂x_i = y_i * (∂L/∂y_i - Σ_j(y_j * ∂L/∂y_j)) */
    tl_tensor* grad_x = tl_tensor_zeros(x->value->ndim, x->value->dims, x->value->dtype);

    /* Compute sum along axis: Σ_j(y_j * ∂L/∂y_j) */
    tl_tensor* y_grad_prod = tl_tensor_zeros(y->ndim, y->dims, y->dtype);
    for (int i = 0; i < y->len; i++) {
        float y_val, grad_y_val;
        TL_TENSOR_DATA_TO(y, i, y_val, TL_FLOAT);
        TL_TENSOR_DATA_TO(grad_y, i, grad_y_val, TL_FLOAT);
        float prod = y_val * grad_y_val;
        TL_TENSOR_DATA_FROM(y_grad_prod, i, prod, TL_FLOAT);
    }

    tl_tensor* sum_y_grad = tl_tensor_sumreduce(y_grad_prod, NULL, axis);

    /* Now compute gradient for each element */
    int axis_size = x->value->dims[axis];
    int outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= x->value->dims[i];
    }
    int inner_size = 1;
    for (int i = axis + 1; i < x->value->ndim; i++) {
        inner_size *= x->value->dims[i];
    }

    for (int outer = 0; outer < outer_size; outer++) {
        for (int inner = 0; inner < inner_size; inner++) {
            float sum_val;
            int sum_idx = outer * inner_size + inner;
            TL_TENSOR_DATA_TO(sum_y_grad, sum_idx, sum_val, TL_FLOAT);

            for (int j = 0; j < axis_size; j++) {
                int idx = outer * axis_size * inner_size + j * inner_size + inner;
                float y_val, grad_y_val;
                TL_TENSOR_DATA_TO(y, idx, y_val, TL_FLOAT);
                TL_TENSOR_DATA_TO(grad_y, idx, grad_y_val, TL_FLOAT);

                float grad = y_val * (grad_y_val - sum_val);
                TL_TENSOR_DATA_FROM(grad_x, idx, grad, TL_FLOAT);
            }
        }
    }

    accumulate_grad(x, grad_x);
    tl_tensor_free_data_too(grad_x);
    tl_tensor_free_data_too(y_grad_prod);
    tl_tensor_free_data_too(sum_y_grad);
}

/* Layer norm backward: y = layer_norm(x, gamma, beta, axis, eps) */
static void layer_norm_backward(tl_graph_node* node)
{
    assert(node->op == TL_OP_LAYER_NORM);
    assert(node->num_inputs >= 1);

    tl_graph_node* x = node->inputs[0];
    tl_graph_node* gamma = (node->num_inputs >= 2) ? node->inputs[1] : NULL;
    tl_graph_node* beta = (node->num_inputs >= 3) ? node->inputs[2] : NULL;
    tl_tensor* grad_y = node->grad;

    if (!grad_y)
        return;

    typedef struct {
        int axis;
        double eps;
    } layer_norm_ctx;
    layer_norm_ctx* ctx = (layer_norm_ctx*)node->backward_ctx;
    int axis = ctx->axis;
    double eps = ctx->eps;

    /* Compute mean and variance (same as forward pass) */
    tl_tensor* mean = tl_tensor_meanreduce(x->value, NULL, axis);
    tl_tensor* x_centered = tl_tensor_sub_broadcast(x->value, mean, NULL, axis);

    /* var = mean((x - mean)^2) */
    tl_tensor* x_centered_sq = tl_tensor_elew(x_centered, x_centered, NULL, TL_MUL);
    tl_tensor* var = tl_tensor_meanreduce(x_centered_sq, NULL, axis);

    /* Normalize: x_norm = (x - mean) / sqrt(var + eps) */
    int N = x->value->dims[axis];

    /* Compute dimension strides for gradient calculations */
    int axis_size = x->value->dims[axis];
    int outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= x->value->dims[i];
    }
    int inner_size = 1;
    for (int i = axis + 1; i < x->value->ndim; i++) {
        inner_size *= x->value->dims[i];
    }

    /* Gradient w.r.t. input */
    if (x->requires_grad) {
        tl_tensor* grad_x = tl_tensor_zeros(x->value->ndim, x->value->dims, x->value->dtype);

        /* This is a simplified version - full layer norm backward is complex */
        /* For now, just pass gradient through assuming normalization is identity-like */
        int axis_size = x->value->dims[axis];
        int outer_size = 1;
        for (int i = 0; i < axis; i++) {
            outer_size *= x->value->dims[i];
        }
        int inner_size = 1;
        for (int i = axis + 1; i < x->value->ndim; i++) {
            inner_size *= x->value->dims[i];
        }

        for (int outer = 0; outer < outer_size; outer++) {
            for (int inner = 0; inner < inner_size; inner++) {
                float var_val;
                int reduce_idx = outer * inner_size + inner;
                TL_TENSOR_DATA_TO(var, reduce_idx, var_val, TL_FLOAT);
                float std = sqrtf(var_val + (float)eps);

                /* Compute gradient contributions */
                float sum_grad_y = 0.0f;
                float sum_grad_y_xnorm = 0.0f;

                for (int j = 0; j < axis_size; j++) {
                    int idx = outer * axis_size * inner_size + j * inner_size + inner;
                    float x_centered_val, grad_y_val;
                    TL_TENSOR_DATA_TO(x_centered, idx, x_centered_val, TL_FLOAT);
                    TL_TENSOR_DATA_TO(grad_y, idx, grad_y_val, TL_FLOAT);

                    float gamma_val = 1.0f;
                    if (gamma && gamma->value) {
                        /* gamma has reduced dimensions along axis */
                        int gamma_idx = (gamma->value->ndim == 1) ? j : idx;
                        TL_TENSOR_DATA_TO(gamma->value, gamma_idx, gamma_val, TL_FLOAT);
                    }

                    sum_grad_y += grad_y_val * gamma_val;
                    sum_grad_y_xnorm += grad_y_val * gamma_val * (x_centered_val / std);
                }

                /* Apply gradient formula */
                for (int j = 0; j < axis_size; j++) {
                    int idx = outer * axis_size * inner_size + j * inner_size + inner;
                    float x_centered_val, grad_y_val;
                    TL_TENSOR_DATA_TO(x_centered, idx, x_centered_val, TL_FLOAT);
                    TL_TENSOR_DATA_TO(grad_y, idx, grad_y_val, TL_FLOAT);

                    float gamma_val = 1.0f;
                    if (gamma && gamma->value) {
                        int gamma_idx = (gamma->value->ndim == 1) ? j : idx;
                        TL_TENSOR_DATA_TO(gamma->value, gamma_idx, gamma_val, TL_FLOAT);
                    }

                    float grad = (gamma_val * grad_y_val - sum_grad_y / N - (x_centered_val / std) * sum_grad_y_xnorm / N) / std;
                    TL_TENSOR_DATA_FROM(grad_x, idx, grad, TL_FLOAT);
                }
            }
        }

        accumulate_grad(x, grad_x);
        tl_tensor_free_data_too(grad_x);
    }

    /* Gradient w.r.t. gamma (scale parameter) */
    if (gamma && gamma->requires_grad) {
        tl_tensor* grad_gamma = tl_tensor_zeros(gamma->value->ndim, gamma->value->dims, gamma->value->dtype);

        /* ∂L/∂gamma = Σ(∂L/∂y * x_normalized) */
        for (int i = 0; i < x->value->len; i++) {
            float var_val, x_centered_val, grad_y_val;

            /* Map to reduce index for var */
            int axis_size = x->value->dims[axis];
            int inner_size = 1;
            for (int k = axis + 1; k < x->value->ndim; k++) {
                inner_size *= x->value->dims[k];
            }
            int reduce_idx = (i / inner_size) / axis_size * inner_size + (i % inner_size);

            TL_TENSOR_DATA_TO(var, reduce_idx, var_val, TL_FLOAT);
            TL_TENSOR_DATA_TO(x_centered, i, x_centered_val, TL_FLOAT);
            TL_TENSOR_DATA_TO(grad_y, i, grad_y_val, TL_FLOAT);

            float std = sqrtf(var_val + (float)eps);
            float x_norm = x_centered_val / std;

            int gamma_idx = (gamma->value->ndim == 1) ? ((i / inner_size) % axis_size) : i;
            float grad_contrib = grad_y_val * x_norm;

            float current_grad;
            TL_TENSOR_DATA_TO(grad_gamma, gamma_idx, current_grad, TL_FLOAT);
            current_grad += grad_contrib;
            TL_TENSOR_DATA_FROM(grad_gamma, gamma_idx, current_grad, TL_FLOAT);
        }

        accumulate_grad(gamma, grad_gamma);
        tl_tensor_free_data_too(grad_gamma);
    }

    /* Gradient w.r.t. beta (shift parameter) */
    if (beta && beta->requires_grad) {
        /* ∂L/∂beta = Σ(∂L/∂y) - sum over all dimensions except the normalized axis */
        tl_tensor* grad_beta = tl_tensor_zeros(beta->value->ndim, beta->value->dims, beta->value->dtype);

        /* Sum grad_y over all dimensions except axis */
        for (int i = 0; i < x->value->len; i++) {
            float grad_y_val;
            TL_TENSOR_DATA_TO(grad_y, i, grad_y_val, TL_FLOAT);

            /* Map to beta index (only the axis dimension matters) */
            int inner_size = 1;
            for (int k = axis + 1; k < x->value->ndim; k++) {
                inner_size *= x->value->dims[k];
            }
            int beta_idx = (beta->value->ndim == 1) ? ((i / inner_size) % axis_size) : i;

            float current_grad;
            TL_TENSOR_DATA_TO(grad_beta, beta_idx, current_grad, TL_FLOAT);
            current_grad += grad_y_val;
            TL_TENSOR_DATA_FROM(grad_beta, beta_idx, current_grad, TL_FLOAT);
        }

        accumulate_grad(beta, grad_beta);
        tl_tensor_free_data_too(grad_beta);
    }

    tl_tensor_free_data_too(mean);
    tl_tensor_free_data_too(x_centered);
    tl_tensor_free_data_too(x_centered_sq);
    tl_tensor_free_data_too(var);
}

/* Assign backward functions to nodes */
static void assign_backward_fn(tl_graph_node* node)
{
    switch (node->op) {
        case TL_OP_MATMUL:
            node->backward_fn = matmul_backward;
            break;
        case TL_OP_ADD:
            node->backward_fn = add_backward;
            break;
        case TL_OP_MUL:
            node->backward_fn = mul_backward;
            break;
        case TL_OP_RELU:
            node->backward_fn = relu_backward;
            break;
        case TL_OP_SOFTMAX:
            node->backward_fn = softmax_backward;
            break;
        case TL_OP_LAYER_NORM:
            node->backward_fn = layer_norm_backward;
            break;
        case TL_OP_INPUT:
        case TL_OP_PARAM:
            /* Leaf nodes have no backward */
            node->backward_fn = NULL;
            break;
        default:
            /* Not implemented yet */
            node->backward_fn = NULL;
            break;
    }
}

/* Main backward pass */
TL_EXPORT void tl_graph_backward(tl_graph* g, tl_graph_node* loss)
{
    assert(g && loss);

    /* Build topological order */
    tl_graph_build_topo(g, loss);

    /* Assign backward functions */
    for (int i = 0; i < g->topo_size; i++) {
        assign_backward_fn(g->topo_order[i]);
    }

    /* Initialize loss gradient to 1.0 (dL/dL = 1) */
    if (!loss->grad) {
        loss->grad = tl_tensor_zeros(loss->value->ndim, loss->value->dims, loss->value->dtype);
        for (int i = 0; i < loss->grad->len; i++) {
            float one = 1.0f;
            TL_TENSOR_DATA_FROM(loss->grad, i, one, TL_FLOAT);
        }
    }

    /* Traverse in reverse topological order */
    for (int i = g->topo_size - 1; i >= 0; i--) {
        tl_graph_node* node = g->topo_order[i];
        if (node->backward_fn && node->grad) {
            node->backward_fn(node);
        }
    }
}
