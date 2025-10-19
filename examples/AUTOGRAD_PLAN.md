# Computation Graph + Autograd Implementation Plan

## Agile Sprint-Based Approach

### Sprint 0: Design & Architecture (Current)
**Goal**: Define interfaces and data structures before coding
**Duration**: Design session

**Deliverables**:
1. Core data structures defined
2. API design documented
3. Memory management strategy
4. Test strategy outlined

---

## Core Design

### Data Structures

```c
/* Operation types */
typedef enum {
    TOFU_OP_INPUT,      /* Leaf node (data) */
    TOFU_OP_PARAM,      /* Trainable parameter */
    TOFU_OP_MATMUL,     /* Matrix multiplication */
    TOFU_OP_ADD,        /* Element-wise addition */
    TOFU_OP_MUL,        /* Element-wise multiplication */
    TOFU_OP_RELU,       /* ReLU activation */
    TOFU_OP_SOFTMAX,    /* Softmax */
    TOFU_OP_LAYER_NORM, /* Layer normalization */
    TOFU_OP_RESHAPE,    /* Reshape operation */
    TOFU_OP_TRANSPOSE,  /* Transpose */
    TOFU_OP_MEAN,       /* Mean reduction */
    TOFU_OP_MSE_LOSS,   /* Mean squared error loss */
    TOFU_OP_CE_LOSS     /* Cross-entropy loss */
} tofu_op_type;

/* Computation graph node */
typedef struct tofu_graph_node {
    int id;                           /* Unique node ID */
    tofu_op_type op;                    /* Operation type */
    tofu_tensor* value;                 /* Forward pass result */
    tofu_tensor* grad;                  /* Gradient (∂L/∂value) */

    /* Topology */
    struct tofu_graph_node** inputs;    /* Input nodes */
    int num_inputs;                   /* Number of inputs */

    /* Backward pass function */
    void (*backward_fn)(struct tofu_graph_node* node);

    /* Operation-specific context */
    void* backward_ctx;               /* e.g., saved tensors for backward */

    /* Memory management */
    int ref_count;                    /* Reference counting */
    int requires_grad;                /* Does this need gradient? */
} tofu_graph_node;

/* Computation graph */
typedef struct tofu_graph {
    tofu_graph_node** nodes;            /* All nodes */
    int num_nodes;                    /* Number of nodes */
    int capacity;                     /* Allocated capacity */

    /* Topological order for backward pass */
    tofu_graph_node** topo_order;
    int topo_size;

    /* Memory arena for efficient allocation */
    void* arena;
} tofu_graph;
```

### API Design

```c
/* Graph lifecycle */
tofu_graph* tofu_graph_create();
void tofu_graph_free(tofu_graph* g);

/* Build graph (record operations) */
tofu_graph_node* tofu_graph_input(tofu_graph* g, tofu_tensor* data);
tofu_graph_node* tofu_graph_param(tofu_graph* g, tofu_tensor* data);

/* Operations (return new nodes) */
tofu_graph_node* tofu_graph_matmul(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
tofu_graph_node* tofu_graph_add(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
tofu_graph_node* tofu_graph_relu(tofu_graph* g, tofu_graph_node* x);
tofu_graph_node* tofu_graph_softmax(tofu_graph* g, tofu_graph_node* x, int axis);
tofu_graph_node* tofu_graph_layer_norm(tofu_graph* g, tofu_graph_node* x,
                                   tofu_graph_node* gamma, tofu_graph_node* beta,
                                   int axis, double eps);

/* Loss functions */
tofu_graph_node* tofu_graph_mse_loss(tofu_graph* g, tofu_graph_node* pred, tofu_graph_node* target);
tofu_graph_node* tofu_graph_cross_entropy(tofu_graph* g, tofu_graph_node* logits, tofu_graph_node* labels);

/* Backward pass */
void tofu_graph_backward(tofu_graph* g, tofu_graph_node* loss);

/* Access gradients */
tofu_tensor* tofu_graph_get_grad(tofu_graph_node* node);
```

---

## Sprint 1: Core Infrastructure
**Goal**: Basic graph building without gradients
**Duration**: 1-2 hours

### Tasks
- [ ] Implement `tofu_graph` structure
- [ ] Implement `tofu_graph_create()` and `tofu_graph_free()`
- [ ] Implement `tofu_graph_node` creation
- [ ] Implement `tofu_graph_input()` and `tofu_graph_param()`
- [ ] Add dynamic array for node storage
- [ ] Memory management (ref counting basics)

### Test Strategy
```c
/* Test: Create graph and add nodes */
tofu_graph* g = tofu_graph_create();
tofu_tensor* data = tofu_tensor_zeros(2, (int[]){2, 3}, TOFU_FLOAT);
tofu_graph_node* x = tofu_graph_input(g, data);
assert(x->op == TOFU_OP_INPUT);
assert(x->value == data);
tofu_graph_free(g);
```

### Success Criteria
- ✅ Can create and destroy graphs
- ✅ Can add input/param nodes
- ✅ No memory leaks (valgrind)

---

## Sprint 2: Forward Pass Operations
**Goal**: Build graph by recording operations
**Duration**: 2-3 hours

### Tasks
- [ ] Implement `tofu_graph_matmul()`
- [ ] Implement `tofu_graph_add()` (element-wise)
- [ ] Implement `tofu_graph_relu()`
- [ ] Implement `tofu_graph_softmax()`
- [ ] Implement `tofu_graph_layer_norm()`
- [ ] Implement topological sort for execution order

### Test Strategy
```c
/* Test: Build and execute forward pass */
tofu_graph* g = tofu_graph_create();
tofu_graph_node* x = tofu_graph_input(g, input_data);
tofu_graph_node* W = tofu_graph_param(g, weights);
tofu_graph_node* y = tofu_graph_matmul(g, x, W);
tofu_graph_node* z = tofu_graph_relu(g, y);

/* Verify forward values are computed */
assert(z->value != NULL);
assert(z->value->ndim == 2);
```

### Success Criteria
- ✅ Forward pass executes correctly
- ✅ Node topology is valid
- ✅ Values match direct tensor operations

---

## Sprint 3: Backward Pass - Basic Ops
**Goal**: Implement gradients for matmul, add, relu
**Duration**: 2-3 hours

### Tasks
- [ ] Implement backward pass infrastructure
- [ ] Implement `matmul_backward()`
- [ ] Implement `add_backward()`
- [ ] Implement `relu_backward()`
- [ ] Implement `tofu_graph_backward()`
- [ ] Gradient accumulation (for nodes with multiple consumers)

### Gradient Formulas

**Matmul**: `y = A @ B`
- `∂L/∂A = (∂L/∂y) @ B^T`
- `∂L/∂B = A^T @ (∂L/∂y)`

**Add**: `z = x + y`
- `∂L/∂x = ∂L/∂z` (possibly summed over broadcast dims)
- `∂L/∂y = ∂L/∂z`

**ReLU**: `y = max(0, x)`
- `∂L/∂x = ∂L/∂y * (x > 0)`

### Test Strategy
```c
/* Test: Gradient through simple network */
tofu_graph* g = tofu_graph_create();
tofu_graph_node* x = tofu_graph_param(g, input_data);  /* requires_grad=true */
tofu_graph_node* W = tofu_graph_param(g, weights);
tofu_graph_node* y = tofu_graph_matmul(g, x, W);
tofu_graph_node* loss = tofu_graph_mse_loss(g, y, target);

tofu_graph_backward(g, loss);

/* Verify gradients computed */
assert(W->grad != NULL);
assert(x->grad != NULL);

/* Compare with numerical gradient */
float numerical_grad = compute_numerical_gradient(x, loss);
float computed_grad = get_grad_element(x->grad, 0);
assert(fabs(numerical_grad - computed_grad) < 1e-4);
```

### Success Criteria
- ✅ Backward pass executes
- ✅ Gradients match numerical gradients (finite differences)
- ✅ Gradients accumulate correctly for shared nodes

---

## Sprint 4: Backward Pass - Advanced Ops
**Goal**: Complete gradient implementation for all ops
**Duration**: 2-3 hours

### Tasks
- [ ] Implement `softmax_backward()`
- [ ] Implement `layer_norm_backward()`
- [ ] Implement `cross_entropy_backward()`
- [ ] Handle reshape/transpose gradients
- [ ] Implement gradient clipping (optional)

### Gradient Formulas

**Softmax**: `y = softmax(x)`
- Jacobian-vector product: `∂L/∂x = softmax(x) ⊙ (∂L/∂y - <∂L/∂y, softmax(x)>)`

**Layer Norm**: Complex, requires saved mean/variance
- See: https://arxiv.org/abs/1607.06450 for detailed derivation

**Cross-Entropy**: `L = -Σ y_i log(ŷ_i)`
- `∂L/∂ŷ = -y/ŷ` (simplified for one-hot labels: `ŷ - y`)

### Test Strategy
```c
/* Test: Full transformer block backward */
tofu_graph* g = tofu_graph_create();
// Build: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
// ... (simplified)
tofu_graph_node* loss = tofu_graph_cross_entropy(g, output, labels);
tofu_graph_backward(g, loss);

/* Verify all parameters have gradients */
assert(all_params_have_grads(g));

/* Verify gradients are not NaN/Inf */
assert(no_nan_or_inf_grads(g));
```

### Success Criteria
- ✅ All operations have backward pass
- ✅ Complex graphs (attention, transformer) work
- ✅ Numerical gradient checks pass

---

## Sprint 5: Optimizer & Training Loop
**Goal**: SGD optimizer and weight updates
**Duration**: 1-2 hours

### Tasks
- [ ] Implement `tofu_optimizer_sgd()`
- [ ] Implement parameter update
- [ ] Zero gradients between steps
- [ ] Optional: momentum, weight decay

### API Design
```c
typedef struct {
    tofu_graph_node** params;
    int num_params;
    float lr;
    float momentum;
    float weight_decay;
    tofu_tensor** velocity;  /* For momentum */
} tofu_optimizer_sgd;

tofu_optimizer_sgd* tofu_optimizer_sgd_create(tofu_graph* g, float lr);
void tofu_optimizer_step(tofu_optimizer_sgd* opt);
void tofu_optimizer_zero_grad(tofu_graph* g);
void tofu_optimizer_free(tofu_optimizer_sgd* opt);
```

### Test Strategy
```c
/* Test: Training loop reduces loss */
float initial_loss = evaluate_loss(model, data);
for (int i = 0; i < 100; i++) {
    forward_pass();
    tofu_graph_backward(g, loss);
    tofu_optimizer_step(opt);
    tofu_optimizer_zero_grad(g);
}
float final_loss = evaluate_loss(model, data);
assert(final_loss < initial_loss);
```

### Success Criteria
- ✅ Optimizer updates parameters
- ✅ Loss decreases over iterations
- ✅ Gradients are zeroed correctly

---

## Sprint 6: Integration Test - Simple MLP
**Goal**: Train a 2-layer MLP on synthetic data
**Duration**: 1-2 hours

### Tasks
- [ ] Create synthetic classification dataset
- [ ] Build 2-layer MLP with autograd
- [ ] Training loop (forward, backward, update)
- [ ] Evaluate accuracy

### Test
```c
/* XOR problem or simple binary classification */
// Input: [N, 2], Output: [N, 1]
// Network: input -> 8 hidden (ReLU) -> 1 output (sigmoid)
// Loss: Binary cross-entropy
// Target: >90% accuracy
```

### Success Criteria
- ✅ MLP trains to >90% accuracy
- ✅ Training is stable (no NaN/exploding gradients)
- ✅ Predictions are correct on test data

---

## Sprint 7: Vision Transformer - Forward Only
**Goal**: Build ViT forward pass using autograd
**Duration**: 2-3 hours

### Tasks
- [ ] Implement patch embedding layer
- [ ] Implement multi-head attention using graph ops
- [ ] Implement FFN block
- [ ] Implement full transformer encoder
- [ ] Load MNIST test data
- [ ] Run inference (no training yet)

### Success Criteria
- ✅ ViT forward pass executes
- ✅ Output shape is correct [batch, 10]
- ✅ No crashes or memory leaks

---

## Sprint 8: Vision Transformer - Training
**Goal**: Train ViT on MNIST
**Duration**: 3-4 hours

### Tasks
- [ ] Load MNIST training data
- [ ] Implement training loop
- [ ] Add mini-batch processing
- [ ] Track loss and accuracy
- [ ] Compare with PyTorch implementation

### Success Criteria
- ✅ Training completes without errors
- ✅ Loss decreases
- ✅ Test accuracy >95%
- ✅ Results match PyTorch (within tolerance)

---

## Sprint 9: Validation & Documentation
**Goal**: Final validation and docs
**Duration**: 1-2 hours

### Tasks
- [ ] Run all tests
- [ ] Memory leak check (valgrind)
- [ ] Performance profiling
- [ ] Write usage examples
- [ ] Document API
- [ ] Update FINDINGS.md

### Success Criteria
- ✅ All tests pass
- ✅ No memory leaks
- ✅ Documentation complete
- ✅ Examples work

---

## Risk Management

### High-Risk Areas
1. **Memory management** - Graph with many nodes
   - Mitigation: Careful ref counting, arena allocator
2. **Numerical stability** - Gradient computation
   - Mitigation: Gradient clipping, numerical checks
3. **Complexity** - Large codebase
   - Mitigation: Sprint-based, incremental testing

### Dependencies
- Sprint 2 depends on Sprint 1 (infrastructure)
- Sprint 3 depends on Sprint 2 (forward ops)
- Sprint 4 depends on Sprint 3 (basic backward)
- Sprint 5 depends on Sprint 4 (gradients)
- Sprints 6-8 depend on Sprint 5 (optimizer)

### Rollback Strategy
- Each sprint is independently committable
- If a sprint fails, rollback to previous sprint
- Can deploy forward-only graph if backward fails

---

## Definition of Done (for entire project)

1. ✅ All sprints completed
2. ✅ All tests pass
3. ✅ ViT trains on MNIST to >95% accuracy
4. ✅ Results match PyTorch implementation
5. ✅ No memory leaks (valgrind clean)
6. ✅ API documentation complete
7. ✅ Performance benchmarked
8. ✅ Code committed and tagged

---

## Estimated Timeline

- **Sprints 1-2**: 3-5 hours (infrastructure + forward)
- **Sprints 3-4**: 4-6 hours (backward pass)
- **Sprint 5**: 1-2 hours (optimizer)
- **Sprint 6**: 1-2 hours (MLP test)
- **Sprints 7-8**: 5-7 hours (ViT)
- **Sprint 9**: 1-2 hours (validation)

**Total**: 15-24 hours of focused work

**Recommendation**: Work in 2-3 hour sessions, completing 1-2 sprints per session.

---

## Next Immediate Action

**Start Sprint 1**: Implement core graph infrastructure
- Create `src/tofu_graph.h` and `src/tofu_graph.c`
- Implement basic data structures
- Write initial tests in `test/test_tofu_graph.c`
