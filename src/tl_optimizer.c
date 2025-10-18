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
#include "tl_optimizer.h"

#define TL_OPTIMIZER_INITIAL_CAPACITY 32

/* SGD momentum state */
typedef struct {
    double momentum;              /* Momentum coefficient */
    tl_tensor** velocity;         /* Velocity buffers for each parameter */
} sgd_momentum_state;

/* Create base optimizer */
static tl_optimizer* tl_optimizer_create_base(tl_graph* g, tl_optim_type type, double learning_rate)
{
    assert(g);
    assert(learning_rate > 0.0);

    tl_optimizer* opt = (tl_optimizer*)malloc(sizeof(tl_optimizer));
    if (!opt)
        return NULL;

    opt->type = type;
    opt->graph = g;
    opt->learning_rate = learning_rate;

    opt->capacity_params = TL_OPTIMIZER_INITIAL_CAPACITY;
    opt->params = (tl_graph_node**)malloc(opt->capacity_params * sizeof(tl_graph_node*));
    if (!opt->params) {
        free(opt);
        return NULL;
    }
    opt->num_params = 0;

    opt->state = NULL;
    opt->step_fn = NULL;

    return opt;
}

/* Collect all parameter nodes from graph */
TL_EXPORT void tl_optimizer_collect_params(tl_optimizer* opt)
{
    assert(opt && opt->graph);

    opt->num_params = 0;

    for (int i = 0; i < opt->graph->num_nodes; i++) {
        tl_graph_node* node = opt->graph->nodes[i];
        if (node->op == TL_OP_PARAM) {
            tl_optimizer_add_param(opt, node);
        }
    }
}

/* Add parameter to optimizer */
TL_EXPORT int tl_optimizer_add_param(tl_optimizer* opt, tl_graph_node* param)
{
    assert(opt && param);
    assert(param->op == TL_OP_PARAM);

    /* Expand capacity if needed */
    if (opt->num_params >= opt->capacity_params) {
        int new_capacity = opt->capacity_params * 2;
        tl_graph_node** new_params = (tl_graph_node**)realloc(
            opt->params, new_capacity * sizeof(tl_graph_node*));
        if (!new_params)
            return -1;
        opt->params = new_params;
        opt->capacity_params = new_capacity;
    }

    opt->params[opt->num_params++] = param;
    return 0;
}

/* SGD step: θ = θ - lr * ∇θ */
static void sgd_step(tl_optimizer* opt)
{
    assert(opt);

    for (int i = 0; i < opt->num_params; i++) {
        tl_graph_node* param = opt->params[i];

        if (!param->grad)
            continue;

        /* Update: param -= learning_rate * grad */
        for (int j = 0; j < param->value->len; j++) {
            double param_val, grad_val;
            TL_TENSOR_DATA_TO(param->value, j, param_val, TL_DOUBLE);
            TL_TENSOR_DATA_TO(param->grad, j, grad_val, TL_DOUBLE);

            param_val -= opt->learning_rate * grad_val;

            TL_TENSOR_DATA_FROM(param->value, j, param_val, TL_DOUBLE);
        }
    }
}

/* SGD with momentum step: v = μ*v - lr*∇θ, θ = θ + v */
static void sgd_momentum_step(tl_optimizer* opt)
{
    assert(opt && opt->state);

    sgd_momentum_state* state = (sgd_momentum_state*)opt->state;

    for (int i = 0; i < opt->num_params; i++) {
        tl_graph_node* param = opt->params[i];

        if (!param->grad)
            continue;

        tl_tensor* velocity = state->velocity[i];

        /* Update velocity: v = momentum * v - lr * grad */
        for (int j = 0; j < param->value->len; j++) {
            double vel_val, grad_val, param_val;
            TL_TENSOR_DATA_TO(velocity, j, vel_val, TL_DOUBLE);
            TL_TENSOR_DATA_TO(param->grad, j, grad_val, TL_DOUBLE);
            TL_TENSOR_DATA_TO(param->value, j, param_val, TL_DOUBLE);

            vel_val = state->momentum * vel_val - opt->learning_rate * grad_val;
            param_val += vel_val;

            TL_TENSOR_DATA_FROM(velocity, j, vel_val, TL_DOUBLE);
            TL_TENSOR_DATA_FROM(param->value, j, param_val, TL_DOUBLE);
        }
    }
}

/* Create SGD optimizer */
TL_EXPORT tl_optimizer* tl_optimizer_sgd_create(tl_graph* g, double learning_rate)
{
    tl_optimizer* opt = tl_optimizer_create_base(g, TL_OPTIM_SGD, learning_rate);
    if (!opt)
        return NULL;

    opt->step_fn = sgd_step;

    /* Collect parameters from graph */
    tl_optimizer_collect_params(opt);

    return opt;
}

/* Create SGD optimizer with momentum */
TL_EXPORT tl_optimizer* tl_optimizer_sgd_momentum_create(tl_graph* g, double learning_rate, double momentum)
{
    assert(momentum >= 0.0 && momentum < 1.0);

    tl_optimizer* opt = tl_optimizer_create_base(g, TL_OPTIM_SGD_MOMENTUM, learning_rate);
    if (!opt)
        return NULL;

    /* Collect parameters first to know how many we have */
    tl_optimizer_collect_params(opt);

    /* Initialize momentum state */
    sgd_momentum_state* state = (sgd_momentum_state*)malloc(sizeof(sgd_momentum_state));
    if (!state) {
        tl_optimizer_free(opt);
        return NULL;
    }

    state->momentum = momentum;
    state->velocity = (tl_tensor**)malloc(opt->num_params * sizeof(tl_tensor*));
    if (!state->velocity) {
        free(state);
        tl_optimizer_free(opt);
        return NULL;
    }

    /* Initialize velocity buffers to zero */
    for (int i = 0; i < opt->num_params; i++) {
        tl_graph_node* param = opt->params[i];
        state->velocity[i] = tl_tensor_zeros(param->value->ndim,
                                             param->value->dims,
                                             param->value->dtype);
    }

    opt->state = state;
    opt->step_fn = sgd_momentum_step;

    return opt;
}

/* Free optimizer */
TL_EXPORT void tl_optimizer_free(tl_optimizer* opt)
{
    if (!opt)
        return;

    /* Free optimizer-specific state */
    if (opt->state) {
        if (opt->type == TL_OPTIM_SGD_MOMENTUM) {
            sgd_momentum_state* state = (sgd_momentum_state*)opt->state;
            if (state->velocity) {
                for (int i = 0; i < opt->num_params; i++) {
                    tl_tensor_free_data_too(state->velocity[i]);
                }
                free(state->velocity);
            }
            free(state);
        }
    }

    free(opt->params);
    free(opt);
}

/* Perform optimizer step */
TL_EXPORT void tl_optimizer_step(tl_optimizer* opt)
{
    assert(opt && opt->step_fn);
    opt->step_fn(opt);
}

/* Zero all gradients */
TL_EXPORT void tl_optimizer_zero_grad(tl_optimizer* opt)
{
    assert(opt);
    tl_graph_zero_grad(opt->graph);
}
