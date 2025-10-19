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
#include "tofu_graph.h"

#define TOFU_GRAPH_INITIAL_CAPACITY 32

/* Create a new computation graph */
TOFU_EXPORT tofu_graph* tofu_graph_create(void)
{
    tofu_graph* g = (tofu_graph*)malloc(sizeof(tofu_graph));
    if (!g)
        return NULL;

    g->capacity = TOFU_GRAPH_INITIAL_CAPACITY;
    g->nodes = (tofu_graph_node**)malloc(g->capacity * sizeof(tofu_graph_node*));
    if (!g->nodes) {
        free(g);
        return NULL;
    }

    g->num_nodes = 0;
    g->next_id = 0;

    /* Initialize topological order arrays */
    g->topo_capacity = TOFU_GRAPH_INITIAL_CAPACITY;
    g->topo_order = (tofu_graph_node**)malloc(g->topo_capacity * sizeof(tofu_graph_node*));
    if (!g->topo_order) {
        free(g->nodes);
        free(g);
        return NULL;
    }
    g->topo_size = 0;

    return g;
}

/* Free a graph node */
static void tofu_graph_node_free(tofu_graph_node* node)
{
    if (!node)
        return;

    /* Free gradient if it exists */
    if (node->grad) {
        tofu_tensor_free_data_too(node->grad);
        node->grad = NULL;
    }

    /* Free inputs array */
    if (node->inputs) {
        free(node->inputs);
        node->inputs = NULL;
    }

    /* Free backward context if it exists */
    if (node->backward_ctx) {
        /* Special handling for layer norm context which contains tensors */
        if (node->op == TOFU_OP_LAYER_NORM) {
            typedef struct {
                int axis;
                double eps;
                tofu_tensor* x_norm;
                tofu_tensor* mean;
                tofu_tensor* inv_std;
                int n;
            } layer_norm_ctx;

            layer_norm_ctx* ctx = (layer_norm_ctx*)node->backward_ctx;
            if (ctx->x_norm) {
                tofu_tensor_free_data_too(ctx->x_norm);
            }
            if (ctx->mean) {
                tofu_tensor_free_data_too(ctx->mean);
            }
            if (ctx->inv_std) {
                tofu_tensor_free_data_too(ctx->inv_std);
            }
        }
        free(node->backward_ctx);
        node->backward_ctx = NULL;
    }

    /* Free value tensor for operation nodes only
     * - For INPUT/PARAM nodes, value is owned by user
     * - For operation nodes, value is allocated by the operation and must be freed
     * - For view operations (reshape, transpose), only free tensor struct, not data
     */
    if (node->value && node->op != TOFU_OP_INPUT && node->op != TOFU_OP_PARAM) {
        if (node->value->owner) {
            /* This is a view (reshape/transpose) - only free tensor struct */
            tofu_tensor_free(node->value);
        } else {
            /* This allocates its own data - free everything */
            tofu_tensor_free_data_too(node->value);
        }
        node->value = NULL;
    }

    free(node);
}

/* Free a computation graph */
TOFU_EXPORT void tofu_graph_free(tofu_graph* g)
{
    if (!g)
        return;

    /* Free all nodes */
    for (int i = 0; i < g->num_nodes; i++) {
        tofu_graph_node_free(g->nodes[i]);
    }

    /* Free arrays */
    free(g->nodes);
    free(g->topo_order);
    free(g);
}

/* Clear operation nodes from graph, keeping INPUT and PARAM nodes
 * This is useful in training loops to avoid node accumulation
 */
TOFU_EXPORT void tofu_graph_clear_ops(tofu_graph* g)
{
    if (!g)
        return;

    /* Keep INPUT and PARAM nodes, remove all operation nodes */
    int write_idx = 0;
    for (int read_idx = 0; read_idx < g->num_nodes; read_idx++) {
        tofu_graph_node* node = g->nodes[read_idx];
        if (node->op == TOFU_OP_INPUT || node->op == TOFU_OP_PARAM) {
            /* Zero out gradients for reused nodes */
            if (node->grad) {
                for (int j = 0; j < node->grad->len; j++) {
                    double zero = 0.0;
                    TOFU_TENSOR_DATA_FROM(node->grad, j, zero, TOFU_DOUBLE);
                }
            }
            g->nodes[write_idx++] = node;
        } else {
            /* Free operation node */
            tofu_graph_node_free(node);
        }
    }
    g->num_nodes = write_idx;

    /* Clear topological order */
    g->topo_size = 0;
}

/* Add a node to the graph */
static tofu_graph_node* tofu_graph_add_node(tofu_graph* g, tofu_op_type op)
{
    assert(g);

    /* Expand capacity if needed */
    if (g->num_nodes >= g->capacity) {
        int new_capacity = g->capacity * 2;
        tofu_graph_node** new_nodes = (tofu_graph_node**)realloc(
            g->nodes, new_capacity * sizeof(tofu_graph_node*));
        if (!new_nodes)
            return NULL;
        g->nodes = new_nodes;
        g->capacity = new_capacity;
    }

    /* Create node */
    tofu_graph_node* node = (tofu_graph_node*)malloc(sizeof(tofu_graph_node));
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
TOFU_EXPORT tofu_graph_node* tofu_graph_input(tofu_graph* g, tofu_tensor* data)
{
    assert(g && data);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_INPUT);
    if (!node)
        return NULL;

    node->value = data;
    node->requires_grad = 0;  /* Inputs don't require gradients */

    return node;
}

/* Create parameter node (requires gradient) */
TOFU_EXPORT tofu_graph_node* tofu_graph_param(tofu_graph* g, tofu_tensor* data)
{
    assert(g && data);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_PARAM);
    if (!node)
        return NULL;

    node->value = data;
    node->requires_grad = 1;  /* Parameters require gradients */

    return node;
}

/* Get value tensor from node */
TOFU_EXPORT tofu_tensor* tofu_graph_get_value(tofu_graph_node* node)
{
    assert(node);
    return node->value;
}

/* Get gradient tensor from node */
TOFU_EXPORT tofu_tensor* tofu_graph_get_grad(tofu_graph_node* node)
{
    assert(node);
    return node->grad;
}

/* Zero all gradients in graph */
TOFU_EXPORT void tofu_graph_zero_grad(tofu_graph* g)
{
    assert(g);

    for (int i = 0; i < g->num_nodes; i++) {
        tofu_graph_node* node = g->nodes[i];
        if (node->grad) {
            /* Zero out gradient tensor */
            for (int j = 0; j < node->grad->len; j++) {
                double zero = 0.0;
                TOFU_TENSOR_DATA_FROM(node->grad, j, zero, TOFU_DOUBLE);
            }
        }
    }
}

/* Helper: Add input to node */
static int tofu_graph_node_add_input(tofu_graph_node* node, tofu_graph_node* input)
{
    assert(node && input);

    /* Expand capacity if needed */
    if (node->num_inputs >= node->capacity_inputs) {
        int new_capacity = node->capacity_inputs == 0 ? 2 : node->capacity_inputs * 2;
        tofu_graph_node** new_inputs = (tofu_graph_node**)realloc(
            node->inputs, new_capacity * sizeof(tofu_graph_node*));
        if (!new_inputs)
            return -1;
        node->inputs = new_inputs;
        node->capacity_inputs = new_capacity;
    }

    node->inputs[node->num_inputs++] = input;
    return 0;
}

/* Helper: Check if any input requires gradient */
static int tofu_graph_any_requires_grad(tofu_graph_node** inputs, int num_inputs)
{
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i]->requires_grad)
            return 1;
    }
    return 0;
}

/* Matrix multiplication: y = a @ b */
TOFU_EXPORT tofu_graph_node* tofu_graph_matmul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b)
{
    assert(g && a && b);
    assert(a->value && b->value);

    /* Create node */
    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_MATMUL);
    if (!node)
        return NULL;

    /* Add inputs */
    if (tofu_graph_node_add_input(node, a) < 0 || tofu_graph_node_add_input(node, b) < 0) {
        return NULL;
    }

    /* Compute forward pass */
    node->value = tofu_tensor_matmul(a->value, b->value, NULL);
    if (!node->value)
        return NULL;

    /* Determine if gradient is required */
    node->requires_grad = tofu_graph_any_requires_grad(node->inputs, node->num_inputs);

    return node;
}

/* Element-wise addition: z = a + b */
TOFU_EXPORT tofu_graph_node* tofu_graph_add(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b)
{
    assert(g && a && b);
    assert(a->value && b->value);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_ADD);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, a) < 0 || tofu_graph_node_add_input(node, b) < 0) {
        return NULL;
    }

    /* Use broadcasting if shapes don't match exactly */
    node->value = tofu_tensor_elew_broadcast(a->value, b->value, NULL, TOFU_SUM);
    if (!node->value)
        return NULL;

    node->requires_grad = tofu_graph_any_requires_grad(node->inputs, node->num_inputs);

    return node;
}

/* Element-wise multiplication: z = a * b */
TOFU_EXPORT tofu_graph_node* tofu_graph_mul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b)
{
    assert(g && a && b);
    assert(a->value && b->value);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_MUL);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, a) < 0 || tofu_graph_node_add_input(node, b) < 0) {
        return NULL;
    }

    node->value = tofu_tensor_elew_broadcast(a->value, b->value, NULL, TOFU_MUL);
    if (!node->value)
        return NULL;

    node->requires_grad = tofu_graph_any_requires_grad(node->inputs, node->num_inputs);

    return node;
}

/* ReLU activation: y = max(0, x) */
TOFU_EXPORT tofu_graph_node* tofu_graph_relu(tofu_graph* g, tofu_graph_node* x)
{
    assert(g && x);
    assert(x->value);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_RELU);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, x) < 0) {
        return NULL;
    }

    /* ReLU: max(0, x) - using lrelu with negslope=0 */
    node->value = tofu_tensor_lrelu(x->value, NULL, 0.0f);
    if (!node->value)
        return NULL;

    node->requires_grad = x->requires_grad;

    return node;
}

/* Softmax activation: y = softmax(x) along axis */
TOFU_EXPORT tofu_graph_node* tofu_graph_softmax(tofu_graph* g, tofu_graph_node* x, int axis)
{
    assert(g && x);
    assert(x->value);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_SOFTMAX);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, x) < 0) {
        return NULL;
    }

    /* Store axis in backward context for later */
    int* axis_ctx = (int*)malloc(sizeof(int));
    if (!axis_ctx)
        return NULL;
    *axis_ctx = axis;
    node->backward_ctx = axis_ctx;

    node->value = tofu_tensor_softmax(x->value, NULL, axis);
    if (!node->value)
        return NULL;

    node->requires_grad = x->requires_grad;

    return node;
}

/* Layer normalization */
TOFU_EXPORT tofu_graph_node* tofu_graph_layer_norm(tofu_graph* g, tofu_graph_node* x,
                                             tofu_graph_node* gamma, tofu_graph_node* beta,
                                             int axis, double eps)
{
    assert(g && x);
    assert(x->value);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_LAYER_NORM);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, x) < 0)
        return NULL;
    if (gamma && tofu_graph_node_add_input(node, gamma) < 0)
        return NULL;
    if (beta && tofu_graph_node_add_input(node, beta) < 0)
        return NULL;

    /* Store axis and eps in backward context */
    typedef struct {
        int axis;
        double eps;
        tofu_tensor* x_norm;      /* Normalized input: (x - mean) / sqrt(var + eps) */
        tofu_tensor* mean;        /* Mean values */
        tofu_tensor* inv_std;     /* 1 / sqrt(var + eps) */
        int n;                  /* Feature dimension size (number of elements normalized) */
    } layer_norm_ctx;

    layer_norm_ctx* ctx = (layer_norm_ctx*)malloc(sizeof(layer_norm_ctx));
    if (!ctx)
        return NULL;
    ctx->axis = axis;
    ctx->eps = eps;
    ctx->n = x->value->dims[axis];

    /* Compute layer norm and cache intermediate values */
    node->value = tofu_tensor_layer_norm(x->value, NULL,
                                       gamma ? gamma->value : NULL,
                                       beta ? beta->value : NULL,
                                       axis, eps);
    if (!node->value) {
        free(ctx);
        return NULL;
    }

    /* Compute and cache intermediate values for backward pass */
    /* We need: x_norm, mean, and inv_std */

    /* Compute mean */
    ctx->mean = tofu_tensor_meanreduce(x->value, NULL, axis);
    if (!ctx->mean) {
        tofu_tensor_free_data_too(node->value);
        free(ctx);
        return NULL;
    }

    /* Compute x_centered = x - mean (broadcast) */
    tofu_tensor* x_centered = tofu_tensor_sub_broadcast(x->value, ctx->mean, NULL, axis);
    if (!x_centered) {
        tofu_tensor_free_data_too(ctx->mean);
        tofu_tensor_free_data_too(node->value);
        free(ctx);
        return NULL;
    }

    /* Compute variance = mean((x - mean)^2) */
    tofu_tensor* x_centered_sq = tofu_tensor_elew(x_centered, x_centered, NULL, TOFU_MUL);
    if (!x_centered_sq) {
        tofu_tensor_free_data_too(x_centered);
        tofu_tensor_free_data_too(ctx->mean);
        tofu_tensor_free_data_too(node->value);
        free(ctx);
        return NULL;
    }

    tofu_tensor* var = tofu_tensor_meanreduce(x_centered_sq, NULL, axis);
    if (!var) {
        tofu_tensor_free_data_too(x_centered_sq);
        tofu_tensor_free_data_too(x_centered);
        tofu_tensor_free_data_too(ctx->mean);
        tofu_tensor_free_data_too(node->value);
        free(ctx);
        return NULL;
    }

    /* Compute inv_std = 1 / sqrt(var + eps) and x_norm = x_centered / sqrt(var + eps) */
    ctx->inv_std = tofu_tensor_zeros(var->ndim, var->dims, TOFU_FLOAT);
    if (!ctx->inv_std) {
        tofu_tensor_free_data_too(var);
        tofu_tensor_free_data_too(x_centered_sq);
        tofu_tensor_free_data_too(x_centered);
        tofu_tensor_free_data_too(ctx->mean);
        tofu_tensor_free_data_too(node->value);
        free(ctx);
        return NULL;
    }

    ctx->x_norm = tofu_tensor_zeros(x->value->ndim, x->value->dims, TOFU_FLOAT);
    if (!ctx->x_norm) {
        tofu_tensor_free_data_too(ctx->inv_std);
        tofu_tensor_free_data_too(var);
        tofu_tensor_free_data_too(x_centered_sq);
        tofu_tensor_free_data_too(x_centered);
        tofu_tensor_free_data_too(ctx->mean);
        tofu_tensor_free_data_too(node->value);
        free(ctx);
        return NULL;
    }

    /* Populate inv_std and x_norm */
    for (int i = 0; i < var->len; i++) {
        float var_val;
        TOFU_TENSOR_DATA_TO(var, i, var_val, TOFU_FLOAT);
        float std = sqrtf(var_val + (float)eps);
        float inv_std_val = 1.0f / std;
        TOFU_TENSOR_DATA_FROM(ctx->inv_std, i, inv_std_val, TOFU_FLOAT);
    }

    /* Compute x_norm = x_centered / std (using broadcast) */
    int axis_size = x->value->dims[axis];
    int outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= x->value->dims[i];
    }
    int inner_size = 1;
    for (int i = axis + 1; i < x->value->ndim; i++) {
        inner_size *= x->value->dims[i];
    }

    for (int i = 0; i < x->value->len; i++) {
        /* Map element index to reduce index for var/mean */
        int outer = i / (axis_size * inner_size);
        int inner = i % inner_size;
        int reduce_idx = outer * inner_size + inner;

        float x_centered_val;
        TOFU_TENSOR_DATA_TO(x_centered, i, x_centered_val, TOFU_FLOAT);
        float inv_std_val;
        TOFU_TENSOR_DATA_TO(ctx->inv_std, reduce_idx, inv_std_val, TOFU_FLOAT);

        float x_norm_val = x_centered_val * inv_std_val;
        TOFU_TENSOR_DATA_FROM(ctx->x_norm, i, x_norm_val, TOFU_FLOAT);
    }

    /* Clean up temporary tensors */
    tofu_tensor_free_data_too(var);
    tofu_tensor_free_data_too(x_centered_sq);
    tofu_tensor_free_data_too(x_centered);

    node->backward_ctx = ctx;
    node->requires_grad = tofu_graph_any_requires_grad(node->inputs, node->num_inputs);

    return node;
}

/* Reshape operation */
TOFU_EXPORT tofu_graph_node* tofu_graph_reshape(tofu_graph* g, tofu_graph_node* x, int ndim, const int* dims)
{
    assert(g && x);
    assert(x->value);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_RESHAPE);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, x) < 0)
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

    node->value = tofu_tensor_reshape(x->value, ndim, dims);
    if (!node->value)
        return NULL;

    node->requires_grad = x->requires_grad;

    return node;
}

/* Transpose operation */
TOFU_EXPORT tofu_graph_node* tofu_graph_transpose(tofu_graph* g, tofu_graph_node* x, const int* axes)
{
    assert(g && x);
    assert(x->value);

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_TRANSPOSE);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, x) < 0)
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

    node->value = tofu_tensor_transpose(x->value, NULL, axes);
    if (!node->value)
        return NULL;

    node->requires_grad = x->requires_grad;

    return node;
}

/* Mean Squared Error Loss: L = (1/n) * Σ(pred - target)² */
TOFU_EXPORT tofu_graph_node* tofu_graph_mse_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target)
{
    assert(g && pred && target);
    assert(pred->value && target->value);
    assert(tofu_tensor_issameshape(pred->value, target->value));

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_MSE_LOSS);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, pred) < 0 || tofu_graph_node_add_input(node, target) < 0) {
        return NULL;
    }

    /* Compute loss: (1/n) * Σ(pred - target)² */
    tofu_tensor* diff = tofu_tensor_elew_broadcast(pred->value, target->value, NULL, TOFU_SUB);
    if (!diff)
        return NULL;

    tofu_tensor* diff_sq = tofu_tensor_elew_param(diff, 2.0, NULL, TOFU_POW);
    tofu_tensor_free_data_too(diff);
    if (!diff_sq)
        return NULL;

    /* Sum all elements manually */
    int n = pred->value->len;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float val;
        TOFU_TENSOR_DATA_TO(diff_sq, i, val, TOFU_FLOAT);
        sum += val;
    }

    /* Average: sum / n */
    float loss_val = sum / n;

    /* Create scalar loss tensor */
    float* loss_data = (float*)malloc(sizeof(float));
    if (!loss_data) {
        tofu_tensor_free_data_too(diff_sq);
        return NULL;
    }
    *loss_data = loss_val;

    tofu_tensor* loss = tofu_tensor_create(loss_data, 1, (int[]){1}, TOFU_FLOAT);
    tofu_tensor_free_data_too(diff_sq);
    if (!loss) {
        free(loss_data);
        return NULL;
    }

    node->value = loss;
    node->requires_grad = pred->requires_grad;

    return node;
}

/* Cross-Entropy Loss: L = -(1/n) * Σ(target * log(pred)) */
TOFU_EXPORT tofu_graph_node* tofu_graph_ce_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target)
{
    assert(g && pred && target);
    assert(pred->value && target->value);
    assert(tofu_tensor_issameshape(pred->value, target->value));

    tofu_graph_node* node = tofu_graph_add_node(g, TOFU_OP_CE_LOSS);
    if (!node)
        return NULL;

    if (tofu_graph_node_add_input(node, pred) < 0 || tofu_graph_node_add_input(node, target) < 0) {
        return NULL;
    }

    /* Compute loss: -(1/n) * Σ(target * log(pred + epsilon))
     * Add epsilon for numerical stability to avoid log(0) */
    const float epsilon = 1e-7f;

    tofu_tensor* pred_safe = tofu_tensor_elew_param(pred->value, epsilon, NULL, TOFU_SUM);
    if (!pred_safe)
        return NULL;

    /* This is not ideal - we need log operation which may not exist
     * For now, compute loss element-wise using available operations */
    tofu_tensor* log_pred = tofu_tensor_clone(pred_safe);
    if (!log_pred) {
        tofu_tensor_free_data_too(pred_safe);
        return NULL;
    }

    /* Compute log(pred) element-wise */
    for (int i = 0; i < log_pred->len; i++) {
        float val;
        TOFU_TENSOR_DATA_TO(log_pred, i, val, TOFU_FLOAT);
        val = logf(val);
        TOFU_TENSOR_DATA_FROM(log_pred, i, val, TOFU_FLOAT);
    }

    tofu_tensor_free_data_too(pred_safe);

    /* Compute target * log(pred) */
    tofu_tensor* target_log_pred = tofu_tensor_elew_broadcast(target->value, log_pred, NULL, TOFU_MUL);
    tofu_tensor_free_data_too(log_pred);
    if (!target_log_pred)
        return NULL;

    /* Sum all elements manually */
    int n = pred->value->len;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float val;
        TOFU_TENSOR_DATA_TO(target_log_pred, i, val, TOFU_FLOAT);
        sum += val;
    }
    tofu_tensor_free_data_too(target_log_pred);

    /* Compute loss: -(1/n) * sum */
    float loss_val = -(1.0f / n) * sum;

    /* Create scalar loss tensor */
    float* loss_data = (float*)malloc(sizeof(float));
    if (!loss_data)
        return NULL;
    *loss_data = loss_val;

    tofu_tensor* loss = tofu_tensor_create(loss_data, 1, (int[]){1}, TOFU_FLOAT);
    if (!loss) {
        free(loss_data);
        return NULL;
    }

    node->value = loss;
    node->requires_grad = pred->requires_grad;

    return node;
}

/* ========================================================================
 * Backward Pass Implementation
 * ======================================================================== */

/* Topological sort using DFS */
static void topo_sort_dfs(tofu_graph_node* node, tofu_graph* g)
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
        tofu_graph_node** new_order = (tofu_graph_node**)realloc(
            g->topo_order, new_capacity * sizeof(tofu_graph_node*));
        if (!new_order)
            return;
        g->topo_order = new_order;
        g->topo_capacity = new_capacity;
    }

    g->topo_order[g->topo_size++] = node;
}

/* Build topological order for backward pass */
static void tofu_graph_build_topo(tofu_graph* g, tofu_graph_node* root)
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
static void accumulate_grad(tofu_graph_node* node, tofu_tensor* grad_contrib)
{
    if (!node->requires_grad)
        return;

    if (!node->grad) {
        /* First gradient contribution - just store it */
        node->grad = tofu_tensor_clone(grad_contrib);
    } else {
        /* Accumulate (add to existing gradient) */
        tofu_tensor* new_grad = tofu_tensor_elew_broadcast(node->grad, grad_contrib, NULL, TOFU_SUM);
        tofu_tensor_free_data_too(node->grad);
        node->grad = new_grad;
    }
}

/* Helper: Reduce gradient to match input shape when broadcasting occurred
 * When broadcasting happens in forward pass (e.g., [3,1] * [3,4] -> [3,4]),
 * the gradient has the output shape [3,4], but we need to reduce it back to [3,1]
 * by summing over the broadcast dimensions.
 */
static tofu_tensor* reduce_grad_for_broadcast(tofu_tensor* grad, const tofu_tensor* input)
{
    int i;
    tofu_tensor* reduced_grad = grad;

    /* Identify dimensions that need reduction by comparing shapes from the right */
    int grad_ndim = grad->ndim;
    int input_ndim = input->ndim;

    /* Process dimensions from right to left (align from trailing dimensions) */
    for (i = 0; i < grad_ndim; i++) {
        int grad_axis = grad_ndim - 1 - i;  /* Current axis in grad */
        int input_axis = input_ndim - 1 - i;  /* Corresponding axis in input */

        /* Determine if this dimension needs reduction */
        int needs_reduction = 0;

        if (input_axis < 0) {
            /* Grad has more dimensions - this is a prepended dimension, needs reduction */
            needs_reduction = 1;
        } else if (input->dims[input_axis] == 1 && reduced_grad->dims[grad_axis] > 1) {
            /* Input had size 1 but grad has size > 1 - broadcast dimension, needs reduction */
            needs_reduction = 1;
        }

        if (needs_reduction) {
            tofu_tensor* temp = tofu_tensor_sumreduce(reduced_grad, NULL, grad_axis);
            if (reduced_grad != grad) {
                tofu_tensor_free_data_too(reduced_grad);
            }
            reduced_grad = temp;
            /* Note: sumreduce doesn't change ndim, just sets dims[grad_axis] = 1 */
        }
    }

    /* If reduced_grad has more dimensions than input, reshape to match
     * (sumreduce keeps dims as size 1, but we need to match input's ndim) */
    if (reduced_grad->ndim != input_ndim) {
        /* Reshape returns a view, so we need to clone to avoid dangling references */
        tofu_tensor* temp_reshape = tofu_tensor_reshape(reduced_grad, input_ndim, input->dims);
        tofu_tensor* cloned = tofu_tensor_clone(temp_reshape);
        tofu_tensor_free(temp_reshape);  /* Free the view */
        if (reduced_grad != grad) {
            tofu_tensor_free_data_too(reduced_grad);  /* Free the intermediate */
        }
        reduced_grad = cloned;
    }

    return reduced_grad;
}

/* Backward functions for each operation */

/* Matmul backward: y = A @ B */
static void matmul_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_MATMUL);
    assert(node->num_inputs == 2);

    tofu_graph_node* A = node->inputs[0];
    tofu_graph_node* B = node->inputs[1];
    tofu_tensor* grad_y = node->grad;

    if (!grad_y)
        return;

    /* Handle 1D tensors by reshaping to 2D for gradient computation */
    int is_A_1d = (A->value->ndim == 1);
    int is_grad_y_1d = (grad_y->ndim == 1);

    tofu_tensor* grad_y_2d = NULL;
    if (is_grad_y_1d && grad_y->dims[0] == 1) {
        /* Reshape [1] to [1, 1] */
        int new_dims[] = {1, 1};
        grad_y_2d = tofu_tensor_reshape(grad_y, 2, new_dims);
    } else if (is_grad_y_1d) {
        /* Reshape [n] to [1, n] */
        int new_dims[] = {1, grad_y->dims[0]};
        grad_y_2d = tofu_tensor_reshape(grad_y, 2, new_dims);
    } else {
        grad_y_2d = grad_y;
    }

    /* ∂L/∂A = (∂L/∂y) @ B^T */
    if (A->requires_grad) {
        /* For batch matmul, transpose only the last 2 dimensions (matrix dims), not batch dims */
        tofu_tensor* B_T = NULL;
        if (B->value->ndim == 2) {
            /* Simple 2D case */
            B_T = tofu_tensor_transpose(B->value, NULL, NULL);
        } else {
            /* N-D case: transpose only last 2 dims */
            int* axes = (int*)malloc(B->value->ndim * sizeof(int));
            for (int i = 0; i < B->value->ndim - 2; i++) {
                axes[i] = i;  /* Keep batch dimensions in order */
            }
            axes[B->value->ndim - 2] = B->value->ndim - 1;  /* Swap last two */
            axes[B->value->ndim - 1] = B->value->ndim - 2;
            B_T = tofu_tensor_transpose(B->value, NULL, axes);
            free(axes);
        }
        tofu_tensor* grad_A_full = tofu_tensor_matmul(grad_y_2d, B_T, NULL);

        /* Reduce gradient if broadcasting occurred in batch dimensions */
        tofu_tensor* grad_A = reduce_grad_for_broadcast(grad_A_full, A->value);

        /* Reshape back to original shape if needed (for 1D inputs) */
        if (is_A_1d && grad_A->ndim == 2) {
            if (grad_A->dims[0] == 1) {
                /* Squeeze [1, n] to [n] */
                int new_dims[] = {grad_A->dims[1]};
                tofu_tensor* grad_A_1d = tofu_tensor_reshape(grad_A, 1, new_dims);
                if (grad_A != grad_A_full) {
                    tofu_tensor_free_data_too(grad_A);
                }
                grad_A = grad_A_1d;
            }
        }

        accumulate_grad(A, grad_A);
        tofu_tensor_free_data_too(B_T);
        tofu_tensor_free_data_too(grad_A_full);
        if (grad_A != grad_A_full) {
            tofu_tensor_free_data_too(grad_A);
        }
    }

    /* ∂L/∂B = A^T @ (∂L/∂y) */
    if (B->requires_grad) {
        tofu_tensor* A_val = A->value;
        tofu_tensor* A_2d = NULL;

        /* If A is 1D, reshape to row vector [1, n] (matching forward behavior) */
        if (is_A_1d) {
            int new_dims[] = {1, A->value->dims[0]};
            A_2d = tofu_tensor_reshape(A->value, 2, new_dims);
            A_val = A_2d;
        }

        /* Transpose only matrix dimensions (last 2) */
        tofu_tensor* A_T = NULL;
        if (A_val->ndim == 2) {
            /* Simple 2D case */
            A_T = tofu_tensor_transpose(A_val, NULL, NULL);
        } else {
            /* N-D case: transpose only last 2 dims */
            int* axes = (int*)malloc(A_val->ndim * sizeof(int));
            for (int i = 0; i < A_val->ndim - 2; i++) {
                axes[i] = i;  /* Keep batch dimensions in order */
            }
            axes[A_val->ndim - 2] = A_val->ndim - 1;  /* Swap last two */
            axes[A_val->ndim - 1] = A_val->ndim - 2;
            A_T = tofu_tensor_transpose(A_val, NULL, axes);
            free(axes);
        }

        /* Reshape grad_y if needed for matmul with A_T */
        tofu_tensor* grad_y_for_B = grad_y;
        tofu_tensor* grad_y_B_reshaped = NULL;
        if (is_grad_y_1d) {
            /* Reshape [n] to [1, n] to match matmul semantics */
            int new_dims[] = {1, grad_y->dims[0]};
            grad_y_B_reshaped = tofu_tensor_reshape(grad_y, 2, new_dims);
            grad_y_for_B = grad_y_B_reshaped;
        }

        tofu_tensor* grad_B_full = tofu_tensor_matmul(A_T, grad_y_for_B, NULL);

        /* Reduce gradient if broadcasting occurred in batch dimensions */
        tofu_tensor* grad_B = reduce_grad_for_broadcast(grad_B_full, B->value);

        accumulate_grad(B, grad_B);

        if (A_2d)
            tofu_tensor_free(A_2d);
        if (grad_y_B_reshaped)
            tofu_tensor_free(grad_y_B_reshaped);
        tofu_tensor_free_data_too(A_T);
        tofu_tensor_free_data_too(grad_B_full);
        if (grad_B != grad_B_full) {
            tofu_tensor_free_data_too(grad_B);
        }
    }

    /* Free temporary grad_y_2d if we created it */
    if (grad_y_2d != grad_y && grad_y_2d) {
        tofu_tensor_free(grad_y_2d);
    }
}

/* Add backward: z = x + y */
static void add_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_ADD);
    assert(node->num_inputs == 2);

    tofu_graph_node* x = node->inputs[0];
    tofu_graph_node* y = node->inputs[1];
    tofu_tensor* grad_z = node->grad;

    if (!grad_z)
        return;

    /* ∂L/∂x = ∂L/∂z (sum over broadcast dimensions if needed) */
    if (x->requires_grad) {
        tofu_tensor* grad_x = reduce_grad_for_broadcast(grad_z, x->value);
        accumulate_grad(x, grad_x);
        if (grad_x != grad_z) {
            tofu_tensor_free_data_too(grad_x);
        }
    }

    /* ∂L/∂y = ∂L/∂z (sum over broadcast dimensions if needed) */
    if (y->requires_grad) {
        tofu_tensor* grad_y = reduce_grad_for_broadcast(grad_z, y->value);
        accumulate_grad(y, grad_y);
        if (grad_y != grad_z) {
            tofu_tensor_free_data_too(grad_y);
        }
    }
}

/* Mul backward: z = x * y */
static void mul_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_MUL);
    assert(node->num_inputs == 2);

    tofu_graph_node* x = node->inputs[0];
    tofu_graph_node* y = node->inputs[1];
    tofu_tensor* grad_z = node->grad;

    if (!grad_z)
        return;

    /* ∂L/∂x = ∂L/∂z * y (sum over broadcast dimensions if needed) */
    if (x->requires_grad) {
        tofu_tensor* grad_x_full = tofu_tensor_elew_broadcast(grad_z, y->value, NULL, TOFU_MUL);
        tofu_tensor* grad_x = reduce_grad_for_broadcast(grad_x_full, x->value);
        accumulate_grad(x, grad_x);
        tofu_tensor_free_data_too(grad_x_full);
        if (grad_x != grad_x_full) {
            tofu_tensor_free_data_too(grad_x);
        }
    }

    /* ∂L/∂y = ∂L/∂z * x (sum over broadcast dimensions if needed) */
    if (y->requires_grad) {
        tofu_tensor* grad_y_full = tofu_tensor_elew_broadcast(grad_z, x->value, NULL, TOFU_MUL);
        tofu_tensor* grad_y = reduce_grad_for_broadcast(grad_y_full, y->value);
        accumulate_grad(y, grad_y);
        tofu_tensor_free_data_too(grad_y_full);
        if (grad_y != grad_y_full) {
            tofu_tensor_free_data_too(grad_y);
        }
    }
}

/* ReLU backward: y = max(0, x) */
static void relu_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_RELU);
    assert(node->num_inputs == 1);

    tofu_graph_node* x = node->inputs[0];
    tofu_tensor* grad_y = node->grad;

    if (!grad_y || !x->requires_grad)
        return;

    /* ∂L/∂x = ∂L/∂y * (x > 0) */
    tofu_tensor* grad_x = tofu_tensor_zeros(x->value->ndim, x->value->dims, x->value->dtype);

    for (int i = 0; i < x->value->len; i++) {
        float x_val, grad_y_val;
        TOFU_TENSOR_DATA_TO(x->value, i, x_val, TOFU_FLOAT);
        TOFU_TENSOR_DATA_TO(grad_y, i, grad_y_val, TOFU_FLOAT);

        /* Gradient passes through only if x > 0 */
        float grad = (x_val > 0.0f) ? grad_y_val : 0.0f;
        TOFU_TENSOR_DATA_FROM(grad_x, i, grad, TOFU_FLOAT);
    }

    accumulate_grad(x, grad_x);
    tofu_tensor_free_data_too(grad_x);
}

/* Softmax backward: y = softmax(x, axis) */
static void softmax_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_SOFTMAX);
    assert(node->num_inputs == 1);

    tofu_graph_node* x = node->inputs[0];
    tofu_tensor* grad_y = node->grad;
    tofu_tensor* y = node->value;  /* softmax output */

    if (!grad_y || !x->requires_grad)
        return;

    int axis = *(int*)node->backward_ctx;

    /* ∂L/∂x_i = y_i * (∂L/∂y_i - Σ_j(y_j * ∂L/∂y_j)) */
    tofu_tensor* grad_x = tofu_tensor_zeros(x->value->ndim, x->value->dims, x->value->dtype);

    /* Compute sum along axis: Σ_j(y_j * ∂L/∂y_j) */
    tofu_tensor* y_grad_prod = tofu_tensor_zeros(y->ndim, y->dims, y->dtype);
    for (int i = 0; i < y->len; i++) {
        float y_val, grad_y_val;
        TOFU_TENSOR_DATA_TO(y, i, y_val, TOFU_FLOAT);
        TOFU_TENSOR_DATA_TO(grad_y, i, grad_y_val, TOFU_FLOAT);
        float prod = y_val * grad_y_val;
        TOFU_TENSOR_DATA_FROM(y_grad_prod, i, prod, TOFU_FLOAT);
    }

    tofu_tensor* sum_y_grad = tofu_tensor_sumreduce(y_grad_prod, NULL, axis);

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
            TOFU_TENSOR_DATA_TO(sum_y_grad, sum_idx, sum_val, TOFU_FLOAT);

            for (int j = 0; j < axis_size; j++) {
                int idx = outer * axis_size * inner_size + j * inner_size + inner;
                float y_val, grad_y_val;
                TOFU_TENSOR_DATA_TO(y, idx, y_val, TOFU_FLOAT);
                TOFU_TENSOR_DATA_TO(grad_y, idx, grad_y_val, TOFU_FLOAT);

                float grad = y_val * (grad_y_val - sum_val);
                TOFU_TENSOR_DATA_FROM(grad_x, idx, grad, TOFU_FLOAT);
            }
        }
    }

    accumulate_grad(x, grad_x);
    tofu_tensor_free_data_too(grad_x);
    tofu_tensor_free_data_too(y_grad_prod);
    tofu_tensor_free_data_too(sum_y_grad);
}

/* Layer norm backward: y = layer_norm(x, gamma, beta, axis, eps)
 *
 * Mathematical formulas (given ∂L/∂y):
 *
 * ∂L/∂γ = Σ(∂L/∂y ⊙ x_norm)   (sum over all dims except feature dim)
 * ∂L/∂β = Σ(∂L/∂y)             (sum over all dims except feature dim)
 *
 * ∂L/∂x_norm = ∂L/∂y ⊙ γ
 *
 * ∂L/∂x = (1 / (n * σ)) * [
 *     n * ∂L/∂x_norm - Σ(∂L/∂x_norm) - x_norm * Σ(∂L/∂x_norm ⊙ x_norm)
 * ]
 *
 * where:
 *   n = feature dimension size
 *   σ = sqrt(var + eps) = 1 / inv_std (standard deviation)
 *   ⊙ = element-wise multiplication
 *   Σ = sum over feature dimension
 */
static void layer_norm_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_LAYER_NORM);
    assert(node->num_inputs >= 1);

    tofu_graph_node* x = node->inputs[0];
    tofu_graph_node* gamma = (node->num_inputs >= 2) ? node->inputs[1] : NULL;
    tofu_graph_node* beta = (node->num_inputs >= 3) ? node->inputs[2] : NULL;
    tofu_tensor* grad_y = node->grad;

    if (!grad_y)
        return;

    /* Retrieve cached values from forward pass */
    typedef struct {
        int axis;
        double eps;
        tofu_tensor* x_norm;
        tofu_tensor* mean;
        tofu_tensor* inv_std;
        int n;
    } layer_norm_ctx;

    layer_norm_ctx* ctx = (layer_norm_ctx*)node->backward_ctx;
    int axis = ctx->axis;
    double eps = ctx->eps;
    tofu_tensor* x_norm = ctx->x_norm;
    tofu_tensor* inv_std = ctx->inv_std;
    int n = ctx->n;

    /* Compute dimension parameters for reductions */
    int axis_size = x->value->dims[axis];
    int outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= x->value->dims[i];
    }
    int inner_size = 1;
    for (int i = axis + 1; i < x->value->ndim; i++) {
        inner_size *= x->value->dims[i];
    }

    /* ========== STEP 1: Compute ∂L/∂γ (scale parameter gradient) ==========
     * ∂L/∂γ = Σ(∂L/∂y ⊙ x_norm)
     * Sum over all dimensions except feature dimension
     */
    if (gamma && gamma->requires_grad) {
        tofu_tensor* grad_gamma = tofu_tensor_zeros(gamma->value->ndim, gamma->value->dims, gamma->value->dtype);

        /* Accumulate contributions for each feature dimension */
        for (int i = 0; i < x->value->len; i++) {
            /* Extract gamma index (feature dimension index) */
            int axis_idx = (i / inner_size) % axis_size;
            int gamma_idx = (gamma->value->ndim == 1) ? axis_idx : i;

            float grad_y_val;
            TOFU_TENSOR_DATA_TO(grad_y, i, grad_y_val, TOFU_FLOAT);

            float x_norm_val;
            TOFU_TENSOR_DATA_TO(x_norm, i, x_norm_val, TOFU_FLOAT);

            /* Contribution: ∂L/∂y[i] * x_norm[i] */
            float contrib = grad_y_val * x_norm_val;

            float current_grad;
            TOFU_TENSOR_DATA_TO(grad_gamma, gamma_idx, current_grad, TOFU_FLOAT);
            current_grad += contrib;
            TOFU_TENSOR_DATA_FROM(grad_gamma, gamma_idx, current_grad, TOFU_FLOAT);
        }

        accumulate_grad(gamma, grad_gamma);
        tofu_tensor_free_data_too(grad_gamma);
    }

    /* ========== STEP 2: Compute ∂L/∂β (bias parameter gradient) ==========
     * ∂L/∂β = Σ(∂L/∂y)
     * Sum over all dimensions except feature dimension
     */
    if (beta && beta->requires_grad) {
        tofu_tensor* grad_beta = tofu_tensor_zeros(beta->value->ndim, beta->value->dims, beta->value->dtype);

        /* Accumulate contributions for each feature dimension */
        for (int i = 0; i < x->value->len; i++) {
            /* Extract beta index (feature dimension index) */
            int axis_idx = (i / inner_size) % axis_size;
            int beta_idx = (beta->value->ndim == 1) ? axis_idx : i;

            float grad_y_val;
            TOFU_TENSOR_DATA_TO(grad_y, i, grad_y_val, TOFU_FLOAT);

            float current_grad;
            TOFU_TENSOR_DATA_TO(grad_beta, beta_idx, current_grad, TOFU_FLOAT);
            current_grad += grad_y_val;
            TOFU_TENSOR_DATA_FROM(grad_beta, beta_idx, current_grad, TOFU_FLOAT);
        }

        accumulate_grad(beta, grad_beta);
        tofu_tensor_free_data_too(grad_beta);
    }

    /* ========== STEP 3: Compute ∂L/∂x_norm (intermediate gradient) ==========
     * ∂L/∂x_norm = ∂L/∂y ⊙ γ
     */
    tofu_tensor* grad_x_norm = tofu_tensor_zeros(x->value->ndim, x->value->dims, x->value->dtype);
    for (int i = 0; i < x->value->len; i++) {
        float grad_y_val;
        TOFU_TENSOR_DATA_TO(grad_y, i, grad_y_val, TOFU_FLOAT);

        float gamma_val = 1.0f;
        if (gamma && gamma->value) {
            int axis_idx = (i / inner_size) % axis_size;
            int gamma_idx = (gamma->value->ndim == 1) ? axis_idx : i;
            TOFU_TENSOR_DATA_TO(gamma->value, gamma_idx, gamma_val, TOFU_FLOAT);
        }

        float grad_x_norm_val = grad_y_val * gamma_val;
        TOFU_TENSOR_DATA_FROM(grad_x_norm, i, grad_x_norm_val, TOFU_FLOAT);
    }

    /* ========== STEP 4: Compute ∂L/∂x (input gradient) - MOST COMPLEX ==========
     * This is the most complex part: chain rule through normalization
     *
     * ∂L/∂x = (1 / (n * σ)) * [
     *     n * ∂L/∂x_norm - Σ(∂L/∂x_norm) - x_norm * Σ(∂L/∂x_norm ⊙ x_norm)
     * ]
     *
     * where Σ denotes sum over the feature dimension
     */
    if (x->requires_grad) {
        tofu_tensor* grad_x = tofu_tensor_zeros(x->value->ndim, x->value->dims, x->value->dtype);

        /* For each normalization group (outer_size * inner_size total) */
        for (int outer = 0; outer < outer_size; outer++) {
            for (int inner = 0; inner < inner_size; inner++) {
                int reduce_idx = outer * inner_size + inner;

                /* Get sigma (std dev) = 1 / inv_std */
                float inv_std_val;
                TOFU_TENSOR_DATA_TO(inv_std, reduce_idx, inv_std_val, TOFU_FLOAT);
                float sigma = 1.0f / inv_std_val;

                /* Compute Σ(∂L/∂x_norm) over feature dimension */
                float sum_grad_x_norm = 0.0f;
                for (int j = 0; j < axis_size; j++) {
                    int idx = outer * axis_size * inner_size + j * inner_size + inner;
                    float grad_x_norm_val;
                    TOFU_TENSOR_DATA_TO(grad_x_norm, idx, grad_x_norm_val, TOFU_FLOAT);
                    sum_grad_x_norm += grad_x_norm_val;
                }

                /* Compute Σ(∂L/∂x_norm ⊙ x_norm) = Σ(∂L/∂x_norm * x_norm) */
                float sum_grad_x_norm_times_x_norm = 0.0f;
                for (int j = 0; j < axis_size; j++) {
                    int idx = outer * axis_size * inner_size + j * inner_size + inner;
                    float grad_x_norm_val;
                    TOFU_TENSOR_DATA_TO(grad_x_norm, idx, grad_x_norm_val, TOFU_FLOAT);
                    float x_norm_val;
                    TOFU_TENSOR_DATA_TO(x_norm, idx, x_norm_val, TOFU_FLOAT);
                    sum_grad_x_norm_times_x_norm += grad_x_norm_val * x_norm_val;
                }

                /* Compute gradient for each element in this normalization group */
                for (int j = 0; j < axis_size; j++) {
                    int idx = outer * axis_size * inner_size + j * inner_size + inner;

                    float grad_x_norm_val;
                    TOFU_TENSOR_DATA_TO(grad_x_norm, idx, grad_x_norm_val, TOFU_FLOAT);

                    float x_norm_val;
                    TOFU_TENSOR_DATA_TO(x_norm, idx, x_norm_val, TOFU_FLOAT);

                    /* Formula: grad_x = (1 / (n * σ)) * [
                     *     n * grad_x_norm - sum_grad_x_norm - x_norm * sum_grad_x_norm_times_x_norm
                     * ]
                     */
                    float numerator = n * grad_x_norm_val - sum_grad_x_norm - x_norm_val * sum_grad_x_norm_times_x_norm;
                    float grad_x_val = numerator / (n * sigma);

                    TOFU_TENSOR_DATA_FROM(grad_x, idx, grad_x_val, TOFU_FLOAT);
                }
            }
        }

        accumulate_grad(x, grad_x);
        tofu_tensor_free_data_too(grad_x);
    }

    /* Clean up intermediate tensors */
    tofu_tensor_free_data_too(grad_x_norm);
}

/* MSE Loss backward: L = (1/n) * Σ(pred - target)²
 * ∂L/∂pred = (2/n) * (pred - target) */
static void mse_loss_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_MSE_LOSS);
    assert(node->num_inputs == 2);

    tofu_graph_node* pred = node->inputs[0];
    tofu_graph_node* target = node->inputs[1];
    tofu_tensor* grad_loss = node->grad;

    if (!grad_loss)
        return;

    /* Compute gradient: ∂L/∂pred = (2/n) * (pred - target)
     * Since grad_loss is a scalar (1.0), we need to:
     * grad_pred = grad_loss * (2/n) * (pred - target) */

    if (pred->requires_grad) {
        /* Compute (pred - target) */
        tofu_tensor* diff = tofu_tensor_elew_broadcast(pred->value, target->value, NULL, TOFU_SUB);
        if (!diff)
            return;

        /* Scale by (2/n) */
        int n = pred->value->len;
        tofu_tensor* grad_pred = tofu_tensor_elew_param(diff, 2.0f / n, NULL, TOFU_MUL);
        tofu_tensor_free_data_too(diff);
        if (!grad_pred)
            return;

        /* Scale by grad_loss (chain rule) - grad_loss is scalar */
        if (grad_loss->len == 1) {
            float loss_grad_val;
            TOFU_TENSOR_DATA_TO(grad_loss, 0, loss_grad_val, TOFU_FLOAT);
            if (loss_grad_val != 1.0f) {
                tofu_tensor* scaled_grad = tofu_tensor_elew_param(grad_pred, loss_grad_val, NULL, TOFU_MUL);
                tofu_tensor_free_data_too(grad_pred);
                grad_pred = scaled_grad;
            }
        }

        accumulate_grad(pred, grad_pred);
        tofu_tensor_free_data_too(grad_pred);
    }

    /* Target is not trainable, so no gradient accumulation needed */
}

/* Cross-Entropy Loss backward: L = -(1/n) * Σ(target * log(pred))
 * ∂L/∂pred = -(1/n) * (target / (pred + epsilon)) */
static void ce_loss_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_CE_LOSS);
    assert(node->num_inputs == 2);

    tofu_graph_node* pred = node->inputs[0];
    tofu_graph_node* target = node->inputs[1];
    tofu_tensor* grad_loss = node->grad;

    if (!grad_loss)
        return;

    if (pred->requires_grad) {
        const float epsilon = 1e-7f;

        /* Compute gradient: ∂L/∂pred = -(1/n) * (target / (pred + epsilon))
         * Since we computed loss as -(1/n) * Σ(target * log(pred + epsilon)),
         * gradient is -(target / (pred + epsilon)) / n */

        /* Compute (pred + epsilon) for numerical stability */
        tofu_tensor* pred_safe = tofu_tensor_elew_param(pred->value, epsilon, NULL, TOFU_SUM);
        if (!pred_safe)
            return;

        /* Compute target / (pred + epsilon)
         * Note: tofu_tensor_elew doesn't have a DIV operation directly
         * We'll do this element-wise */
        tofu_tensor* grad_pred = tofu_tensor_zeros(pred->value->ndim, pred->value->dims, TOFU_FLOAT);
        if (!grad_pred) {
            tofu_tensor_free_data_too(pred_safe);
            return;
        }

        int n = pred->value->len;
        for (int i = 0; i < n; i++) {
            float target_val;
            float pred_safe_val;
            TOFU_TENSOR_DATA_TO(target->value, i, target_val, TOFU_FLOAT);
            TOFU_TENSOR_DATA_TO(pred_safe, i, pred_safe_val, TOFU_FLOAT);

            /* grad = -(1/n) * (target / pred_safe) */
            float grad_val = -(1.0f / n) * (target_val / pred_safe_val);

            TOFU_TENSOR_DATA_FROM(grad_pred, i, grad_val, TOFU_FLOAT);
        }

        tofu_tensor_free_data_too(pred_safe);

        /* Scale by grad_loss (chain rule) - grad_loss is scalar */
        if (grad_loss->len == 1) {
            float loss_grad_val;
            TOFU_TENSOR_DATA_TO(grad_loss, 0, loss_grad_val, TOFU_FLOAT);
            if (loss_grad_val != 1.0f) {
                tofu_tensor* scaled_grad = tofu_tensor_elew_param(grad_pred, loss_grad_val, NULL, TOFU_MUL);
                tofu_tensor_free_data_too(grad_pred);
                grad_pred = scaled_grad;
            }
        }

        accumulate_grad(pred, grad_pred);
        tofu_tensor_free_data_too(grad_pred);
    }

    /* Target is not trainable, so no gradient accumulation needed */
}

/* Transpose backward: y = transpose(x, axes) */
static void transpose_backward(tofu_graph_node* node)
{
    assert(node->op == TOFU_OP_TRANSPOSE);
    assert(node->num_inputs == 1);

    tofu_graph_node* x = node->inputs[0];
    tofu_tensor* grad_y = node->grad;

    if (!grad_y || !x->requires_grad)
        return;

    /* Get the axes permutation from backward context */
    int* axes = (int*)node->backward_ctx;
    int ndim = x->value->ndim;

    /* Compute inverse permutation
     * If axes[i] = j, then inv_axes[j] = i
     * Example: axes = [2, 0, 1] -> inv_axes = [1, 2, 0]
     */
    int* inv_axes = (int*)malloc(ndim * sizeof(int));
    if (!inv_axes)
        return;

    if (axes) {
        for (int i = 0; i < ndim; i++) {
            inv_axes[axes[i]] = i;
        }
    } else {
        /* If axes is NULL, it means default transpose (reverse all dims)
         * For 3D: [0,1,2] -> [2,1,0], inverse is [2,1,0] */
        for (int i = 0; i < ndim; i++) {
            inv_axes[i] = ndim - 1 - i;
        }
    }

    /* Apply inverse transpose to gradient
     * ∂L/∂x = transpose(∂L/∂y, inv_axes) */
    tofu_tensor* grad_x = tofu_tensor_transpose(grad_y, NULL, inv_axes);
    if (!grad_x) {
        free(inv_axes);
        return;
    }

    accumulate_grad(x, grad_x);
    tofu_tensor_free_data_too(grad_x);
    free(inv_axes);
}

/* Assign backward functions to nodes */
static void assign_backward_fn(tofu_graph_node* node)
{
    switch (node->op) {
        case TOFU_OP_MATMUL:
            node->backward_fn = matmul_backward;
            break;
        case TOFU_OP_ADD:
            node->backward_fn = add_backward;
            break;
        case TOFU_OP_MUL:
            node->backward_fn = mul_backward;
            break;
        case TOFU_OP_RELU:
            node->backward_fn = relu_backward;
            break;
        case TOFU_OP_SOFTMAX:
            node->backward_fn = softmax_backward;
            break;
        case TOFU_OP_LAYER_NORM:
            node->backward_fn = layer_norm_backward;
            break;
        case TOFU_OP_MSE_LOSS:
            node->backward_fn = mse_loss_backward;
            break;
        case TOFU_OP_CE_LOSS:
            node->backward_fn = ce_loss_backward;
            break;
        case TOFU_OP_TRANSPOSE:
            node->backward_fn = transpose_backward;
            break;
        case TOFU_OP_INPUT:
        case TOFU_OP_PARAM:
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
TOFU_EXPORT void tofu_graph_backward(tofu_graph* g, tofu_graph_node* loss)
{
    assert(g && loss);

    /* Build topological order */
    tofu_graph_build_topo(g, loss);

    /* Assign backward functions */
    for (int i = 0; i < g->topo_size; i++) {
        assign_backward_fn(g->topo_order[i]);
    }

    /* Initialize loss gradient to 1.0 (dL/dL = 1) */
    if (!loss->grad) {
        loss->grad = tofu_tensor_zeros(loss->value->ndim, loss->value->dims, loss->value->dtype);
        for (int i = 0; i < loss->grad->len; i++) {
            float one = 1.0f;
            TOFU_TENSOR_DATA_FROM(loss->grad, i, one, TOFU_FLOAT);
        }
    }

    /* Traverse in reverse topological order */
    for (int i = g->topo_size - 1; i >= 0; i--) {
        tofu_graph_node* node = g->topo_order[i];
        if (node->backward_fn && node->grad) {
            node->backward_fn(node);
        }
    }
}
