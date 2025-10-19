# Documentation Fixes Required

**Date**: 2025-10-19
**Status**: Verification complete, fixes in progress

## ✅ Fixed

### 1. **CRITICAL: src/tofu_graph.h:104**
- **Issue**: Doxygen comment incorrectly stated "Graph takes ownership"
- **Fix Applied**: Changed to "Graph does NOT take ownership"
- **Status**: ✅ FIXED
- **Commit**: Needs to be committed

### 2. **docs/src/api-reference/tensor-api.md** - Matmul Precondition

**Location**: Line 586

**Issue**: Precondition formula invalid for 1-D tensors (out of bounds access)

**Fix Applied**: Split precondition into 1-D case and 2-D+ case:
```markdown
**Preconditions:**
- For 1-D @ 1-D: `src1->dims[0]` must equal `src2->dims[0]`
- For 2-D and higher: `src1->dims[src1->ndim-1]` must equal `src2->dims[src2->ndim-2]`
```

**Status**: ✅ FIXED

---

### 3. **docs/src/api-reference/optimizer-api.md** - SGD Momentum Formula

**Location**: Lines 138-154

**Issue**: Formula didn't match actual implementation in `tofu_optimizer.c:139-151`

**Fix Applied**: Updated to match implementation with explanatory note:
```markdown
velocity = momentum * velocity - learning_rate * grad
param = param + velocity

Mathematical notation:
v ← μ * v - η * ∇θL
θ ← θ + v

Note: This is mathematically equivalent to classical momentum but incorporates
the learning rate into the velocity update rather than the parameter update.
```

**Status**: ✅ FIXED

---

## ❌ Pending Fixes

None - all critical issues resolved!

---

## ✅ Optional Improvements (Completed)

### 4. **docs/src/api-reference/tensor-api.md** - View Operation Warning

**Location**: Lines 389-395 (tofu_tensor_reshape)

**Improvement Applied**: Added explicit warning about NOT calling `tofu_tensor_free_data_too` on views:
```markdown
**Warning:** Do NOT call `tofu_tensor_free_data_too` on the reshaped view - this would free
the shared data while the source tensor still references it! Only use `tofu_tensor_free` on views.
```

**Status**: ✅ APPLIED

---

### 5. **docs/src/api-reference/graph-api.md** - Reshape as View Operation

**Location**: Line 540

**Improvement Applied**: Clarified that reshaped tensors share data with input:
```markdown
View operation (no data copy) - reshaped tensor shares data with input
```

**Status**: ✅ APPLIED

---

### 6. **docs/src/api-reference/graph-api.md** - Gradient Accumulation

**Location**: Lines 687-688

**Improvement Applied**: Expanded explanation of gradient accumulation behavior:
```markdown
Gradients accumulate across multiple backward passes and from multiple computational paths.
Always call tofu_graph_zero_grad before each training iteration unless you intentionally
want gradient accumulation (e.g., for gradient accumulation across mini-batches).
```

**Status**: ✅ APPLIED

---

## Verification Summary

### Status by Document

| Document | Critical Errors | Minor Issues | Overall |
|----------|----------------|--------------|---------|
| **graph-api.md** (1,055 lines) | 0 | 2 | ✅ 98% |
| **tensor-api.md** (1,288 lines) | 1 | 1 | ⚠️ 95% |
| **optimizer-api.md** (907 lines) | 1 | 0 | ⚠️ 95% |
| **Source: tofu_graph.h** | 1 (FIXED) | 0 | ✅ 100% |

### Priority

1. **P0 (Critical)**: Fix matmul precondition, SGD momentum formula
2. **P1 (Important)**: Add view operation warnings
3. **P2 (Nice-to-have)**: Clarify gradient accumulation behavior

---

## Next Steps

1. Apply fixes #2 and #3 to documentation
2. Consider optional improvements (#4-6)
3. Commit all changes together
4. Rebuild mdBook documentation
5. Preview and verify fixes
