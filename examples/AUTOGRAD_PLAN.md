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
    TL_OP_INPUT,      /* Leaf node (data) */
    TL_OP_PARAM,      /* Trainable parameter */
    TL_OP_MATMUL,     /* Matrix multiplication */
    TL_OP_ADD,        /* Element-wise addition */
    TL_OP_MUL,        /* Element-wise multiplication */
    TL_OP_RELU,       /* ReLU activation */
    TL_OP_SOFTMAX,    /* Softmax */
    TL_OP_LAYER_NORM, /* Layer normalization */
    TL_OP_RESHAPE,    /* Reshape operation */
    TL_OP_TRANSPOSE,  /* Transpose */
    TL_OP_MEAN,       /* Mean reduction */
    TL_OP_MSE_LOSS,   /* Mean squared error loss */
    TL_OP_CE_LOSS     /* Cross-entropy loss */
} tl_op_type;

/* Computation graph node */
typedef struct tl_graph_node {
    int id;                           /* Unique node ID */
    tl_op_type op;                    /* Operation type */
    tl_tensor* value;                 /* Forward pass result */
    tl_tensor* grad;                  /* Gradient (∂L/∂value) */

    /* Topology */
    struct tl_graph_node** inputs;    /* Input nodes */
    int num_inputs;                   /* Number of inputs */

    /* Backward pass function */
    void (*backward_fn)(struct tl_graph_node* node);

    /* Operation-specific context */
    void* backward_ctx;               /* e.g., saved tensors for backward */

    /* Memory management */
    int ref_count;                    /* Reference counting */
    int requires_grad;                /* Does this need gradient? */
} tl_graph_node;

/* Computation graph */
typedef struct tl_graph {
    tl_graph_node** nodes;            /* All nodes */
    int num_nodes;                    /* Number of nodes */
    int capacity;                     /* Allocated capacity */

    /* Topological order for backward pass */
    tl_graph_node** topo_order;
    int topo_size;

    /* Memory arena for efficient allocation */
    void* arena;
} tl_graph;
```

### API Design

```c
/* Graph lifecycle */
tl_graph* tl_graph_create();
void tl_graph_free(tl_graph* g);

/* Build graph (record operations) */
tl_graph_node* tl_graph_input(tl_graph* g, tl_tensor* data);
tl_graph_node* tl_graph_param(tl_graph* g, tl_tensor* data);

/* Operations (return new nodes) */
tl_graph_node* tl_graph_matmul(tl_graph* g, tl_graph_node* a, tl_graph_node* b);
tl_graph_node* tl_graph_add(tl_graph* g, tl_graph_node* a, tl_graph_node* b);
tl_graph_node* tl_graph_relu(tl_graph* g, tl_graph_node* x);
tl_graph_node* tl_graph_softmax(tl_graph* g, tl_graph_node* x, int axis);
tl_graph_node* tl_graph_layer_norm(tl_graph* g, tl_graph_node* x,
                                   tl_graph_node* gamma, tl_graph_node* beta,
                                   int axis, double eps);

/* Loss functions */
tl_graph_node* tl_graph_mse_loss(tl_graph* g, tl_graph_node* pred, tl_graph_node* target);
tl_graph_node* tl_graph_cross_entropy(tl_graph* g, tl_graph_node* logits, tl_graph_node* labels);

/* Backward pass */
void tl_graph_backward(tl_graph* g, tl_graph_node* loss);

/* Access gradients */
tl_tensor* tl_graph_get_grad(tl_graph_node* node);
```

---

## Sprint 1: Core Infrastructure
**Goal**: Basic graph building without gradients
**Duration**: 1-2 hours

### Tasks
- [ ] Implement `tl_graph` structure
- [ ] Implement `tl_graph_create()` and `tl_graph_free()`
- [ ] Implement `tl_graph_node` creation
- [ ] Implement `tl_graph_input()` and `tl_graph_param()`
- [ ] Add dynamic array for node storage
- [ ] Memory management (ref counting basics)

### Test Strategy
```c
/* Test: Create graph and add nodes */
tl_graph* g = tl_graph_create();
tl_tensor* data = tl_tensor_zeros(2, (int[]){2, 3}, TL_FLOAT);
tl_graph_node* x = tl_graph_input(g, data);
assert(x->op == TL_OP_INPUT);
assert(x->value == data);
tl_graph_free(g);
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
- [ ] Implement `tl_graph_matmul()`
- [ ] Implement `tl_graph_add()` (element-wise)
- [ ] Implement `tl_graph_relu()`
- [ ] Implement `tl_graph_softmax()`
- [ ] Implement `tl_graph_layer_norm()`
- [ ] Implement topological sort for execution order

### Test Strategy
```c
/* Test: Build and execute forward pass */
tl_graph* g = tl_graph_create();
tl_graph_node* x = tl_graph_input(g, input_data);
tl_graph_node* W = tl_graph_param(g, weights);
tl_graph_node* y = tl_graph_matmul(g, x, W);
tl_graph_node* z = tl_graph_relu(g, y);

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
- [ ] Implement `tl_graph_backward()`
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
tl_graph* g = tl_graph_create();
tl_graph_node* x = tl_graph_param(g, input_data);  /* requires_grad=true */
tl_graph_node* W = tl_graph_param(g, weights);
tl_graph_node* y = tl_graph_matmul(g, x, W);
tl_graph_node* loss = tl_graph_mse_loss(g, y, target);

tl_graph_backward(g, loss);

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
tl_graph* g = tl_graph_create();
// Build: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
// ... (simplified)
tl_graph_node* loss = tl_graph_cross_entropy(g, output, labels);
tl_graph_backward(g, loss);

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
- [ ] Implement `tl_optimizer_sgd()`
- [ ] Implement parameter update
- [ ] Zero gradients between steps
- [ ] Optional: momentum, weight decay

### API Design
```c
typedef struct {
    tl_graph_node** params;
    int num_params;
    float lr;
    float momentum;
    float weight_decay;
    tl_tensor** velocity;  /* For momentum */
} tl_optimizer_sgd;

tl_optimizer_sgd* tl_optimizer_sgd_create(tl_graph* g, float lr);
void tl_optimizer_step(tl_optimizer_sgd* opt);
void tl_optimizer_zero_grad(tl_graph* g);
void tl_optimizer_free(tl_optimizer_sgd* opt);
```

### Test Strategy
```c
/* Test: Training loop reduces loss */
float initial_loss = evaluate_loss(model, data);
for (int i = 0; i < 100; i++) {
    forward_pass();
    tl_graph_backward(g, loss);
    tl_optimizer_step(opt);
    tl_optimizer_zero_grad(g);
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
- Create `src/tl_graph.h` and `src/tl_graph.c`
- Implement basic data structures
- Write initial tests in `test/test_tl_graph.c`
