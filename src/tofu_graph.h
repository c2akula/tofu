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

#ifndef _TOFU_GRAPH_H_
#define _TOFU_GRAPH_H_

#include "tofu_tensor.h"

/* Operation types for computation graph */
typedef enum {
    TOFU_OP_INPUT,       /* Leaf node (input data, no gradient) */
    TOFU_OP_PARAM,       /* Trainable parameter (requires gradient) */
    TOFU_OP_MATMUL,      /* Matrix multiplication */
    TOFU_OP_ADD,         /* Element-wise addition */
    TOFU_OP_MUL,         /* Element-wise multiplication */
    TOFU_OP_RELU,        /* ReLU activation */
    TOFU_OP_SOFTMAX,     /* Softmax activation */
    TOFU_OP_LAYER_NORM,  /* Layer normalization */
    TOFU_OP_RESHAPE,     /* Reshape operation */
    TOFU_OP_TRANSPOSE,   /* Transpose operation */
    TOFU_OP_MEAN,        /* Mean reduction */
    TOFU_OP_SUM,         /* Sum reduction */
    TOFU_OP_MSE_LOSS,    /* Mean squared error loss */
    TOFU_OP_CE_LOSS      /* Cross-entropy loss */
} tofu_op_type;

/* Forward declaration */
typedef struct tofu_graph tofu_graph;
typedef struct tofu_graph_node tofu_graph_node;

/* Backward function signature */
typedef void (*tofu_backward_fn)(tofu_graph_node* node);

/* Computation graph node */
struct tofu_graph_node {
    int id;                        /* Unique node ID within graph */
    tofu_op_type op;                 /* Operation type */

    /* Values */
    tofu_tensor* value;              /* Forward pass result */
    tofu_tensor* grad;               /* Gradient (∂L/∂value) */

    /* Topology */
    tofu_graph_node** inputs;        /* Input nodes */
    int num_inputs;                /* Number of inputs */
    int capacity_inputs;           /* Allocated capacity for inputs */

    /* Backward pass */
    tofu_backward_fn backward_fn;    /* Backward pass function */
    void* backward_ctx;            /* Context for backward (saved tensors, etc.) */

    /* Flags */
    int requires_grad;             /* Does this need gradient computation? */
    int visited;                   /* For topological sort */

    /* Reference to graph */
    tofu_graph* graph;               /* Parent graph */
};

/* Computation graph */
struct tofu_graph {
    tofu_graph_node** nodes;         /* All nodes in graph */
    int num_nodes;                 /* Number of nodes */
    int capacity;                  /* Allocated capacity */

    /* Topological order for backward pass */
    tofu_graph_node** topo_order;    /* Nodes in reverse topological order */
    int topo_size;                 /* Size of topo_order */
    int topo_capacity;             /* Allocated capacity */

    /* ID counter */
    int next_id;                   /* Next available node ID */
};

#ifdef __cplusplus
TOFU_CPPSTART
#endif

/* Graph lifecycle */
TOFU_EXPORT tofu_graph* tofu_graph_create(void);
TOFU_EXPORT void tofu_graph_free(tofu_graph* g);
TOFU_EXPORT void tofu_graph_clear_ops(tofu_graph* g);

/* Build graph - leaf nodes */
TOFU_EXPORT tofu_graph_node* tofu_graph_input(tofu_graph* g, tofu_tensor* data);
TOFU_EXPORT tofu_graph_node* tofu_graph_param(tofu_graph* g, tofu_tensor* data);

/* Forward pass operations */
TOFU_EXPORT tofu_graph_node* tofu_graph_matmul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
TOFU_EXPORT tofu_graph_node* tofu_graph_add(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
TOFU_EXPORT tofu_graph_node* tofu_graph_mul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
TOFU_EXPORT tofu_graph_node* tofu_graph_relu(tofu_graph* g, tofu_graph_node* x);
TOFU_EXPORT tofu_graph_node* tofu_graph_softmax(tofu_graph* g, tofu_graph_node* x, int axis);
TOFU_EXPORT tofu_graph_node* tofu_graph_layer_norm(tofu_graph* g, tofu_graph_node* x,
                                             tofu_graph_node* gamma, tofu_graph_node* beta,
                                             int axis, double eps);
TOFU_EXPORT tofu_graph_node* tofu_graph_reshape(tofu_graph* g, tofu_graph_node* x, int ndim, const int* dims);
TOFU_EXPORT tofu_graph_node* tofu_graph_transpose(tofu_graph* g, tofu_graph_node* x, const int* axes);

/* Loss functions */
TOFU_EXPORT tofu_graph_node* tofu_graph_mse_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target);
TOFU_EXPORT tofu_graph_node* tofu_graph_ce_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target);

/* Backward pass */
TOFU_EXPORT void tofu_graph_backward(tofu_graph* g, tofu_graph_node* loss);

/* Utility functions */
TOFU_EXPORT tofu_tensor* tofu_graph_get_value(tofu_graph_node* node);
TOFU_EXPORT tofu_tensor* tofu_graph_get_grad(tofu_graph_node* node);
TOFU_EXPORT void tofu_graph_zero_grad(tofu_graph* g);

#ifdef __cplusplus
TOFU_CPPEND
#endif

#endif /* _TOFU_GRAPH_H_ */
