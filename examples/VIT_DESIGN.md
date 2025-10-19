# Vision Transformer for MNIST - Design Document

## Goal
Implement a complete Vision Transformer (ViT) for MNIST digit classification to validate Tofu library with a modern, complex architecture.

## Architecture Overview

### Input Processing
```
MNIST image: 28×28 grayscale
├─> Split into patches: 7×7 grid of 4×4 patches
├─> Flatten patches: 49 patches × 16 pixels = [49, 16]
├─> Linear projection: [49, 16] @ [16, 64] = [49, 64]
└─> Add positional embeddings: [49, 64] + [49, 64]
```

### Transformer Encoder (1-2 blocks)
```
For each block:
├─> Layer Norm 1
├─> Multi-Head Self-Attention (4 heads, 64-dim)
│   ├─> Q = X @ W_q  [49, 64] @ [64, 64] = [49, 64]
│   ├─> K = X @ W_k  [49, 64] @ [64, 64] = [49, 64]
│   ├─> V = X @ W_v  [49, 64] @ [64, 64] = [49, 64]
│   ├─> Reshape to [4, 49, 16] for multi-head
│   ├─> Attention = softmax(Q @ K^T / sqrt(16))  [4, 49, 49]
│   ├─> Output = Attention @ V  [4, 49, 16]
│   └─> Reshape back to [49, 64]
├─> Residual connection
├─> Layer Norm 2
├─> FFN: [49, 64] -> [49, 256] -> [49, 64]
└─> Residual connection
```

### Classification Head
```
├─> Average pool over sequence: [49, 64] -> [64]
└─> Linear classifier: [64] @ [64, 10] = [10]
```

## Parameters

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch embedding | [16, 64] | 1,024 |
| Position embedding | [49, 64] | 3,136 |
| **Per Transformer Block:** | | |
| LayerNorm 1 (γ, β) | [64] × 2 | 128 |
| W_q, W_k, W_v | [64, 64] × 3 | 12,288 |
| W_out | [64, 64] | 4,096 |
| LayerNorm 2 (γ, β) | [64] × 2 | 128 |
| FFN W1 | [64, 256] | 16,384 |
| FFN W2 | [256, 64] | 16,384 |
| **Classification Head:** | | |
| W_class | [64, 10] | 640 |
| **Total (1 block):** | | **~54K** |
| **Total (2 blocks):** | | **~103K** |

## Operations Required

### Already in Tofu ✅
1. `tofu_tensor_matmul` - Q/K/V projections, FFN
2. `tofu_tensor_transpose` - K^T for attention scores
3. `tofu_tensor_reshape` - Multi-head reshaping
4. `tofu_tensor_elew_broadcast` - Residual connections (add)
5. `tofu_tensor_elew_param` - Scaling (divide by sqrt(d_k))

### Need to Add - Forward Pass 🔨
1. **Softmax** (have in examples, need in library)
   - `tofu_tensor_softmax(src, dst, axis)` - Apply along axis

2. **Reduction Operations**
   - `tofu_tensor_sum(src, dst, axis)` - Sum along axis
   - `tofu_tensor_mean(src, dst, axis)` - Mean along axis

3. **Layer Normalization**
   - `tofu_tensor_layer_norm(src, dst, gamma, beta, eps)` - Normalize + scale/shift

### Need to Add - Backward Pass 🔨
1. **Softmax Gradient**
   - `tofu_tensor_softmax_backward(grad_out, softmax_out, grad_in, axis)`

2. **Layer Norm Gradient**
   - `tofu_tensor_layer_norm_backward(grad_out, x, gamma, grad_in, grad_gamma, grad_beta)`

3. **Loss Functions**
   - `tofu_tensor_cross_entropy(logits, labels, loss)` - Forward
   - `tofu_tensor_cross_entropy_backward(logits, labels, grad)` - Backward

4. **Optimizer**
   - `tofu_optimizer_sgd(params[], grads[], lr, momentum)`
   - Or: `tofu_optimizer_adam(params[], grads[], lr, beta1, beta2, eps)`

### Nice to Have (Can Implement in User Code)
- Argmax (can use `tofu_tensor_maxreduce` with arg output)
- Gradient clipping
- Learning rate scheduling

## Implementation Plan

### Phase 1: Add Missing Operations
1. Add `tofu_tensor_softmax` to library
2. Add `tofu_tensor_sum` reduction
3. Add `tofu_tensor_mean` reduction
4. Add `tofu_tensor_layer_norm` (or implement with existing ops)

### Phase 2: Add Backward Pass Operations
1. Implement gradient computation for all forward ops
2. Add cross-entropy loss + gradient
3. Add SGD or Adam optimizer
4. Test backward pass against PyTorch autograd

### Phase 3: Training in C with Tofu
1. Implement ViT forward pass in C
2. Implement ViT backward pass in C
3. Load MNIST training data
4. Training loop with mini-batches
5. Evaluate on test set
6. Compare with PyTorch implementation

### Phase 4: Validation
1. Verify loss curves match PyTorch
2. Verify final accuracy (target: >95%)
3. Compare gradients at each layer
4. Profile memory usage during training
5. Measure training time
6. Document findings and API usability feedback

## Expected Challenges

1. **Tensor Reshaping for Multi-Head**
   - Need to reshape [49, 64] -> [49, 4, 16] -> [4, 49, 16]
   - Then back after attention

2. **Softmax Over Sequence**
   - Need softmax along axis=2 for [4, 49, 49] attention scores
   - Numerical stability is critical

3. **Layer Normalization**
   - Compute mean and variance along feature dimension
   - Normalize, then scale/shift

4. **Memory Management**
   - Many intermediate tensors in attention
   - Need careful cleanup to avoid leaks

5. **Debugging**
   - Complex architecture makes issues harder to trace
   - Layer-by-layer validation essential

## Success Criteria

1. ✅ **Correctness**: C predictions match Python exactly
2. ✅ **Accuracy**: >95% on MNIST test set
3. ✅ **Robustness**: No NaN/Inf during inference
4. ✅ **Memory**: No leaks, reasonable usage
5. ✅ **Performance**: Inference time measured and documented
6. ✅ **API Usability**: Code is readable and maintainable

## Why This Validates Tofu

- **Modern Architecture**: ViT is state-of-the-art (2020+)
- **Complex Tensor Ops**: Attention requires sophisticated manipulation
- **Real Application**: Actual digit classification
- **Research-Level**: Proves library can handle cutting-edge models
- **Rich Feedback**: Will reveal any API shortcomings or missing operations

This is the most comprehensive validation possible short of training the model in C!
