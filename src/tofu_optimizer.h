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
TOFU_EXPORT tofu_optimizer* tofu_optimizer_sgd_create(tofu_graph* g, double learning_rate);
TOFU_EXPORT tofu_optimizer* tofu_optimizer_sgd_momentum_create(tofu_graph* g, double learning_rate, double momentum);
TOFU_EXPORT void tofu_optimizer_free(tofu_optimizer* opt);

/* Optimizer operations */
TOFU_EXPORT void tofu_optimizer_step(tofu_optimizer* opt);
TOFU_EXPORT void tofu_optimizer_zero_grad(tofu_optimizer* opt);

/* Utility functions */
TOFU_EXPORT int tofu_optimizer_add_param(tofu_optimizer* opt, tofu_graph_node* param);
TOFU_EXPORT void tofu_optimizer_collect_params(tofu_optimizer* opt);

#ifdef __cplusplus
TOFU_CPPEND
#endif

#endif /* _TOFU_OPTIMIZER_H_ */
