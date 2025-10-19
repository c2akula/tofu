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

/**
 * @brief Create a new empty computation graph
 * @return Pointer to newly allocated graph (caller owns, must call tofu_graph_free)
 * @note Graph starts empty - add nodes via tofu_graph_input, tofu_graph_param, etc.
 * @note Graph takes ownership of tensors passed to tofu_graph_param
 * @note Caller must call tofu_graph_free to free graph and all nodes
 */
TOFU_EXPORT tofu_graph* tofu_graph_create(void);

/**
 * @brief Free computation graph and all nodes
 * @param g Graph to free (can be NULL, no-op if NULL)
 * @note Frees all nodes, gradients, and owned tensors
 * @note Tensors passed to tofu_graph_param are freed
 * @note Tensors passed to tofu_graph_input are NOT freed (caller owns)
 * @note Safe to call multiple times (idempotent)
 */
TOFU_EXPORT void tofu_graph_free(tofu_graph* g);

/**
 * @brief Clear all operation nodes but keep parameter nodes
 * @param g Graph to clear (cannot be NULL)
 * @pre g must not be NULL
 * @note Frees all nodes except PARAM and INPUT nodes
 * @note Preserves trainable parameters for next forward pass
 * @note Use between training iterations to reset computation graph
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT void tofu_graph_clear_ops(tofu_graph* g);

/* Build graph - leaf nodes */

/**
 * @brief Create input node (non-trainable data source)
 * @param g Graph to add node to (cannot be NULL)
 * @param data Input tensor data (cannot be NULL)
 * @return Pointer to newly created graph node (graph owns, freed by tofu_graph_free)
 * @pre g and data must not be NULL
 * @note Input nodes do NOT compute gradients
 * @note Graph does NOT take ownership of data tensor (caller must manage)
 * @note Use for input data that doesn't require backpropagation
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_graph_param for trainable parameters
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_input(tofu_graph* g, tofu_tensor* data);

/**
 * @brief Create parameter node (trainable weights/biases)
 * @param g Graph to add node to (cannot be NULL)
 * @param data Parameter tensor data (cannot be NULL)
 * @return Pointer to newly created graph node (graph owns, freed by tofu_graph_free)
 * @pre g and data must not be NULL
 * @note IMPORTANT: Graph takes ownership of data tensor
 * @note Parameter nodes compute gradients during backward pass
 * @note Use for trainable weights, biases, etc.
 * @note data tensor will be freed when graph is freed
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_graph_input for non-trainable inputs
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_param(tofu_graph* g, tofu_tensor* data);

/* Forward pass operations */

/**
 * @brief Add matrix multiplication node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param a Left operand node (cannot be NULL)
 * @param b Right operand node (cannot be NULL)
 * @return Pointer to result node (graph owns, freed by tofu_graph_free)
 * @pre g, a, and b must not be NULL
 * @pre a->value->dims[last] must equal b->value->dims[second-to-last]
 * @note Computes matrix multiplication with broadcasting
 * @note Implements backward pass for gradient computation
 * @note Result node requires gradient if any input requires gradient
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_tensor_matmul for operation semantics
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_matmul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);

/**
 * @brief Add element-wise addition node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param a First operand node (cannot be NULL)
 * @param b Second operand node (cannot be NULL)
 * @return Pointer to result node (graph owns, freed by tofu_graph_free)
 * @pre g, a, and b must not be NULL
 * @pre a and b must be broadcastable (NumPy rules)
 * @note Computes element-wise addition with broadcasting
 * @note Implements backward pass for gradient computation
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_add(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);

/**
 * @brief Add element-wise multiplication node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param a First operand node (cannot be NULL)
 * @param b Second operand node (cannot be NULL)
 * @return Pointer to result node (graph owns, freed by tofu_graph_free)
 * @pre g, a, and b must not be NULL
 * @pre a and b must be broadcastable (NumPy rules)
 * @note Computes element-wise multiplication with broadcasting
 * @note Implements backward pass for gradient computation
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_mul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);

/**
 * @brief Add ReLU activation node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param x Input node (cannot be NULL)
 * @return Pointer to result node (graph owns, freed by tofu_graph_free)
 * @pre g and x must not be NULL
 * @note Computes ReLU: max(0, x)
 * @note Implements backward pass for gradient computation
 * @note Gradient is 1 where x > 0, else 0
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_relu(tofu_graph* g, tofu_graph_node* x);

/**
 * @brief Add softmax activation node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param x Input node (cannot be NULL)
 * @param axis Axis along which to apply softmax
 * @return Pointer to result node (graph owns, freed by tofu_graph_free)
 * @pre g and x must not be NULL; axis < x->value->ndim
 * @note Computes softmax along specified axis (exp normalization)
 * @note Implements backward pass for gradient computation
 * @note Numerically stable (subtracts max before exp)
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_softmax(tofu_graph* g, tofu_graph_node* x, int axis);

/**
 * @brief Add layer normalization node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param x Input node (cannot be NULL)
 * @param gamma Scale parameter node (can be NULL for no scaling)
 * @param beta Shift parameter node (can be NULL for no shift)
 * @param axis Axis along which to normalize
 * @param eps Small constant for numerical stability (typically 1e-5)
 * @return Pointer to result node (graph owns, freed by tofu_graph_free)
 * @pre g and x must not be NULL; axis < x->value->ndim; eps > 0
 * @note Normalizes: (x - mean) / sqrt(variance + eps)
 * @note Then applies: gamma * normalized + beta (if gamma/beta non-NULL)
 * @note Implements backward pass for gradient computation
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_layer_norm(tofu_graph* g, tofu_graph_node* x,
                                             tofu_graph_node* gamma, tofu_graph_node* beta,
                                             int axis, double eps);

/**
 * @brief Add reshape node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param x Input node (cannot be NULL)
 * @param ndim Number of dimensions for reshaped tensor
 * @param dims Array of new dimension sizes
 * @return Pointer to result node (graph owns, freed by tofu_graph_free)
 * @pre g, x, and dims must not be NULL; ndim > 0
 * @pre Product of dims must equal x->value total elements
 * @note View operation (no data copy)
 * @note Implements backward pass for gradient computation
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_reshape(tofu_graph* g, tofu_graph_node* x, int ndim, const int* dims);

/**
 * @brief Add transpose node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param x Input node (cannot be NULL)
 * @param axes Permutation array (can be NULL for reverse order)
 * @return Pointer to result node (graph owns, freed by tofu_graph_free)
 * @pre g and x must not be NULL
 * @pre If axes is non-NULL, it must be valid permutation of [0, ..., ndim-1]
 * @note If axes is NULL, reverses dimension order
 * @note Implements backward pass for gradient computation
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_transpose(tofu_graph* g, tofu_graph_node* x, const int* axes);

/* Loss functions */

/**
 * @brief Add mean squared error loss node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param pred Prediction node (cannot be NULL)
 * @param target Target/ground truth node (cannot be NULL)
 * @return Pointer to scalar loss node (graph owns, freed by tofu_graph_free)
 * @pre g, pred, and target must not be NULL
 * @pre pred and target must have same shape
 * @note Computes: mean((pred - target)^2)
 * @note Returns scalar (average over all elements)
 * @note Use for regression tasks
 * @note Implements backward pass for gradient computation
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_graph_ce_loss for classification
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_mse_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target);

/**
 * @brief Add cross-entropy loss node to graph
 * @param g Graph to add node to (cannot be NULL)
 * @param pred Prediction node (softmax probabilities) (cannot be NULL)
 * @param target Target/ground truth node (class indices or one-hot) (cannot be NULL)
 * @return Pointer to scalar loss node (graph owns, freed by tofu_graph_free)
 * @pre g, pred, and target must not be NULL
 * @note Computes: -sum(target * log(pred))
 * @note Returns scalar (average over batch)
 * @note Use for classification tasks
 * @note Numerically stable implementation
 * @note Implements backward pass for gradient computation
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_graph_mse_loss for regression
 */
TOFU_EXPORT tofu_graph_node* tofu_graph_ce_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target);

/* Backward pass */

/**
 * @brief Perform backward pass (backpropagation) from loss node
 * @param g Graph containing loss node (cannot be NULL)
 * @param loss Loss node to backpropagate from (cannot be NULL)
 * @pre g and loss must not be NULL
 * @pre loss must be scalar (single element tensor)
 * @note Computes gradients for all nodes requiring gradient
 * @note Populates node->grad for all PARAM nodes
 * @note Uses reverse-mode automatic differentiation
 * @note Call after forward pass, before optimizer step
 * @note Gradients accumulate - call tofu_graph_zero_grad first if needed
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_graph_zero_grad to clear gradients
 */
TOFU_EXPORT void tofu_graph_backward(tofu_graph* g, tofu_graph_node* loss);

/* Utility functions */

/**
 * @brief Get forward pass result from graph node
 * @param node Graph node (cannot be NULL)
 * @return Pointer to result tensor (node owns, do NOT free)
 * @pre node must not be NULL
 * @note Returns tensor computed during forward pass
 * @note Do NOT free returned tensor (node owns it)
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_tensor* tofu_graph_get_value(tofu_graph_node* node);

/**
 * @brief Get gradient from graph node
 * @param node Graph node (cannot be NULL)
 * @return Pointer to gradient tensor (node owns, do NOT free), or NULL if no gradient
 * @pre node must not be NULL
 * @note Returns gradient computed during backward pass
 * @note Returns NULL if backward hasn't been called yet
 * @note Do NOT free returned tensor (node owns it)
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT tofu_tensor* tofu_graph_get_grad(tofu_graph_node* node);

/**
 * @brief Zero out all gradients in graph
 * @param g Graph to zero gradients for (cannot be NULL)
 * @pre g must not be NULL
 * @note Sets all node->grad tensors to zero
 * @note Call before each training iteration to prevent gradient accumulation
 * @note Does NOT free gradient tensors, just zeros values
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT void tofu_graph_zero_grad(tofu_graph* g);

#ifdef __cplusplus
TOFU_CPPEND
#endif

#endif /* _TOFU_GRAPH_H_ */
