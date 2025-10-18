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

#ifndef _TL_OPTIMIZER_H_
#define _TL_OPTIMIZER_H_

#include "tl_graph.h"

/* Optimizer types */
typedef enum {
    TL_OPTIM_SGD,
    TL_OPTIM_SGD_MOMENTUM,
    TL_OPTIM_ADAM  /* Future */
} tl_optim_type;

/* Forward declaration */
typedef struct tl_optimizer tl_optimizer;

/* Optimizer step function signature */
typedef void (*tl_optim_step_fn)(tl_optimizer* opt);

/* Optimizer base structure */
struct tl_optimizer {
    tl_optim_type type;           /* Optimizer type */
    tl_graph* graph;              /* Associated computation graph */

    /* Parameters */
    tl_graph_node** params;       /* Array of parameter nodes */
    int num_params;               /* Number of parameters */
    int capacity_params;          /* Allocated capacity */

    /* Hyperparameters */
    double learning_rate;         /* Learning rate */

    /* Optimizer-specific state */
    void* state;                  /* Optimizer state (e.g., momentum buffers) */

    /* Step function */
    tl_optim_step_fn step_fn;     /* Parameter update function */
};

#ifdef __cplusplus
TL_CPPSTART
#endif

/* Optimizer lifecycle */
TL_EXPORT tl_optimizer* tl_optimizer_sgd_create(tl_graph* g, double learning_rate);
TL_EXPORT tl_optimizer* tl_optimizer_sgd_momentum_create(tl_graph* g, double learning_rate, double momentum);
TL_EXPORT void tl_optimizer_free(tl_optimizer* opt);

/* Optimizer operations */
TL_EXPORT void tl_optimizer_step(tl_optimizer* opt);
TL_EXPORT void tl_optimizer_zero_grad(tl_optimizer* opt);

/* Utility functions */
TL_EXPORT int tl_optimizer_add_param(tl_optimizer* opt, tl_graph_node* param);
TL_EXPORT void tl_optimizer_collect_params(tl_optimizer* opt);

#ifdef __cplusplus
TL_CPPEND
#endif

#endif /* _TL_OPTIMIZER_H_ */
