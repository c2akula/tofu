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

#ifndef _TOFU_OPTIMIZER_H_
#define _TOFU_OPTIMIZER_H_

#include "tofu_graph.h"

/* Optimizer types */
typedef enum {
    TOFU_OPTIM_SGD,
    TOFU_OPTIM_SGD_MOMENTUM,
    TOFU_OPTIM_ADAM  /* Future */
} tofu_optim_type;

/* Forward declaration */
typedef struct tofu_optimizer tofu_optimizer;

/* Optimizer step function signature */
typedef void (*tofu_optim_step_fn)(tofu_optimizer* opt);

/* Optimizer base structure */
struct tofu_optimizer {
    tofu_optim_type type;           /* Optimizer type */
    tofu_graph* graph;              /* Associated computation graph */

    /* Parameters */
    tofu_graph_node** params;       /* Array of parameter nodes */
    int num_params;               /* Number of parameters */
    int capacity_params;          /* Allocated capacity */

    /* Hyperparameters */
    double learning_rate;         /* Learning rate */

    /* Optimizer-specific state */
    void* state;                  /* Optimizer state (e.g., momentum buffers) */

    /* Step function */
    tofu_optim_step_fn step_fn;     /* Parameter update function */
};

#ifdef __cplusplus
TOFU_CPPSTART
#endif

/* Optimizer lifecycle */

/**
 * @brief Create SGD (Stochastic Gradient Descent) optimizer
 * @param g Computation graph containing parameters (cannot be NULL)
 * @param learning_rate Learning rate (step size) (must be > 0)
 * @return Pointer to newly allocated optimizer (caller owns, must call tofu_optimizer_free)
 * @pre g must not be NULL; learning_rate > 0
 * @note Implements vanilla SGD: param = param - learning_rate * grad
 * @note Automatically collects all PARAM nodes from graph
 * @note Caller must call tofu_optimizer_free to free optimizer
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_optimizer_sgd_momentum_create for SGD with momentum
 */
TOFU_EXPORT tofu_optimizer* tofu_optimizer_sgd_create(tofu_graph* g, double learning_rate);

/**
 * @brief Create SGD optimizer with momentum
 * @param g Computation graph containing parameters (cannot be NULL)
 * @param learning_rate Learning rate (step size) (must be > 0)
 * @param momentum Momentum coefficient (typically 0.9) (must be >= 0 and < 1)
 * @return Pointer to newly allocated optimizer (caller owns, must call tofu_optimizer_free)
 * @pre g must not be NULL; learning_rate > 0; 0 <= momentum < 1
 * @note Implements SGD with momentum:
 *       velocity = momentum * velocity + grad
 *       param = param - learning_rate * velocity
 * @note Momentum helps accelerate training and reduces oscillations
 * @note Automatically collects all PARAM nodes from graph
 * @note Caller must call tofu_optimizer_free to free optimizer
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_optimizer_sgd_create for vanilla SGD
 */
TOFU_EXPORT tofu_optimizer* tofu_optimizer_sgd_momentum_create(tofu_graph* g, double learning_rate, double momentum);

/**
 * @brief Free optimizer and its state
 * @param opt Optimizer to free (can be NULL, no-op if NULL)
 * @note Frees optimizer structure and internal state (momentum buffers, etc.)
 * @note Does NOT free the graph or parameters (graph owns them)
 * @note Safe to call multiple times (idempotent)
 */
TOFU_EXPORT void tofu_optimizer_free(tofu_optimizer* opt);

/* Optimizer operations */

/**
 * @brief Perform one optimization step (update parameters)
 * @param opt Optimizer (cannot be NULL)
 * @pre opt must not be NULL
 * @pre Gradients must be computed (call tofu_graph_backward first)
 * @note Updates all parameters using computed gradients
 * @note Algorithm depends on optimizer type (SGD, SGD+momentum, etc.)
 * @note Call after backward pass: forward → backward → step
 * @note Does NOT zero gradients - call tofu_optimizer_zero_grad if needed
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_graph_backward to compute gradients
 * @see tofu_optimizer_zero_grad to clear gradients
 */
TOFU_EXPORT void tofu_optimizer_step(tofu_optimizer* opt);

/**
 * @brief Zero out all parameter gradients
 * @param opt Optimizer (cannot be NULL)
 * @pre opt must not be NULL
 * @note Sets gradients to zero for all tracked parameters
 * @note Call before each training iteration to prevent gradient accumulation
 * @note Equivalent to tofu_graph_zero_grad but works via optimizer
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_graph_zero_grad for graph-level gradient zeroing
 */
TOFU_EXPORT void tofu_optimizer_zero_grad(tofu_optimizer* opt);

/* Utility functions */

/**
 * @brief Manually add parameter node to optimizer
 * @param opt Optimizer (cannot be NULL)
 * @param param Parameter node to track (cannot be NULL)
 * @return 0 on success, non-zero on error
 * @pre opt and param must not be NULL
 * @pre param must be a PARAM node (requires gradient)
 * @note Usually not needed - optimizer auto-collects params at creation
 * @note Use if you need to add parameters dynamically
 * @note Violating preconditions triggers assert() and crashes
 * @see tofu_optimizer_collect_params to scan graph for all params
 */
TOFU_EXPORT int tofu_optimizer_add_param(tofu_optimizer* opt, tofu_graph_node* param);

/**
 * @brief Collect all parameter nodes from graph
 * @param opt Optimizer (cannot be NULL)
 * @pre opt must not be NULL
 * @note Scans graph and adds all PARAM nodes to optimizer
 * @note Called automatically during optimizer creation
 * @note Use if graph structure changes and you need to rescan
 * @note Clears existing parameter list before collecting
 * @note Violating preconditions triggers assert() and crashes
 */
TOFU_EXPORT void tofu_optimizer_collect_params(tofu_optimizer* opt);

#ifdef __cplusplus
TOFU_CPPEND
#endif

#endif /* _TOFU_OPTIMIZER_H_ */
