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

#ifndef _TL_GRAPH_H_
#define _TL_GRAPH_H_

#include "tl_tensor.h"

/* Operation types for computation graph */
typedef enum {
    TL_OP_INPUT,       /* Leaf node (input data, no gradient) */
    TL_OP_PARAM,       /* Trainable parameter (requires gradient) */
    TL_OP_MATMUL,      /* Matrix multiplication */
    TL_OP_ADD,         /* Element-wise addition */
    TL_OP_MUL,         /* Element-wise multiplication */
    TL_OP_RELU,        /* ReLU activation */
    TL_OP_SOFTMAX,     /* Softmax activation */
    TL_OP_LAYER_NORM,  /* Layer normalization */
    TL_OP_RESHAPE,     /* Reshape operation */
    TL_OP_TRANSPOSE,   /* Transpose operation */
    TL_OP_MEAN,        /* Mean reduction */
    TL_OP_SUM,         /* Sum reduction */
    TL_OP_MSE_LOSS,    /* Mean squared error loss */
    TL_OP_CE_LOSS      /* Cross-entropy loss */
} tl_op_type;

/* Forward declaration */
typedef struct tl_graph tl_graph;
typedef struct tl_graph_node tl_graph_node;

/* Backward function signature */
typedef void (*tl_backward_fn)(tl_graph_node* node);

/* Computation graph node */
struct tl_graph_node {
    int id;                        /* Unique node ID within graph */
    tl_op_type op;                 /* Operation type */

    /* Values */
    tl_tensor* value;              /* Forward pass result */
    tl_tensor* grad;               /* Gradient (∂L/∂value) */

    /* Topology */
    tl_graph_node** inputs;        /* Input nodes */
    int num_inputs;                /* Number of inputs */
    int capacity_inputs;           /* Allocated capacity for inputs */

    /* Backward pass */
    tl_backward_fn backward_fn;    /* Backward pass function */
    void* backward_ctx;            /* Context for backward (saved tensors, etc.) */

    /* Flags */
    int requires_grad;             /* Does this need gradient computation? */
    int visited;                   /* For topological sort */

    /* Reference to graph */
    tl_graph* graph;               /* Parent graph */
};

/* Computation graph */
struct tl_graph {
    tl_graph_node** nodes;         /* All nodes in graph */
    int num_nodes;                 /* Number of nodes */
    int capacity;                  /* Allocated capacity */

    /* Topological order for backward pass */
    tl_graph_node** topo_order;    /* Nodes in reverse topological order */
    int topo_size;                 /* Size of topo_order */
    int topo_capacity;             /* Allocated capacity */

    /* ID counter */
    int next_id;                   /* Next available node ID */
};

#ifdef __cplusplus
TL_CPPSTART
#endif

/* Graph lifecycle */
TL_EXPORT tl_graph* tl_graph_create(void);
TL_EXPORT void tl_graph_free(tl_graph* g);
TL_EXPORT void tl_graph_clear_ops(tl_graph* g);

/* Build graph - leaf nodes */
TL_EXPORT tl_graph_node* tl_graph_input(tl_graph* g, tl_tensor* data);
TL_EXPORT tl_graph_node* tl_graph_param(tl_graph* g, tl_tensor* data);

/* Forward pass operations */
TL_EXPORT tl_graph_node* tl_graph_matmul(tl_graph* g, tl_graph_node* a, tl_graph_node* b);
TL_EXPORT tl_graph_node* tl_graph_add(tl_graph* g, tl_graph_node* a, tl_graph_node* b);
TL_EXPORT tl_graph_node* tl_graph_mul(tl_graph* g, tl_graph_node* a, tl_graph_node* b);
TL_EXPORT tl_graph_node* tl_graph_relu(tl_graph* g, tl_graph_node* x);
TL_EXPORT tl_graph_node* tl_graph_softmax(tl_graph* g, tl_graph_node* x, int axis);
TL_EXPORT tl_graph_node* tl_graph_layer_norm(tl_graph* g, tl_graph_node* x,
                                             tl_graph_node* gamma, tl_graph_node* beta,
                                             int axis, double eps);
TL_EXPORT tl_graph_node* tl_graph_reshape(tl_graph* g, tl_graph_node* x, int ndim, const int* dims);
TL_EXPORT tl_graph_node* tl_graph_transpose(tl_graph* g, tl_graph_node* x, const int* axes);

/* Loss functions */
TL_EXPORT tl_graph_node* tl_graph_mse_loss(tl_graph* g, tl_graph_node* pred, tl_graph_node* target);
TL_EXPORT tl_graph_node* tl_graph_ce_loss(tl_graph* g, tl_graph_node* pred, tl_graph_node* target);

/* Backward pass */
TL_EXPORT void tl_graph_backward(tl_graph* g, tl_graph_node* loss);

/* Utility functions */
TL_EXPORT tl_tensor* tl_graph_get_value(tl_graph_node* node);
TL_EXPORT tl_tensor* tl_graph_get_grad(tl_graph_node* node);
TL_EXPORT void tl_graph_zero_grad(tl_graph* g);

#ifdef __cplusplus
TL_CPPEND
#endif

#endif /* _TL_GRAPH_H_ */
