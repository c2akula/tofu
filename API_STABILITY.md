# API Stability Guarantee

**Status**: Effective from v1.0.0
**Last Updated**: 2025-10-19

## Overview

Starting with v1.0.0, Tofu's public API is **frozen** and follows strict semantic versioning. This document defines what constitutes the public API, our stability guarantees, and the process for handling breaking changes.

---

## Semantic Versioning Policy

Tofu follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

### Version Number Meanings

- **MAJOR** (e.g., 1.x.x → 2.0.0): Breaking API changes
  - Function signature changes
  - Function/type renames
  - Behavior changes that break existing code
  - Removal of deprecated features

- **MINOR** (e.g., 1.0.x → 1.1.0): Backward-compatible additions
  - New functions
  - New operation types
  - New optional parameters (with defaults)
  - Performance improvements

- **PATCH** (e.g., 1.0.0 → 1.0.1): Backward-compatible bug fixes
  - Bug fixes
  - Documentation corrections
  - Internal implementation improvements

---

## What is the Public API?

The **public API** consists of all exported symbols in public header files:

### Core Headers (Stable API)
- `tofu.h` - Library version and configuration
- `tofu_tensor.h` - Tensor operations and data structures
- `tofu_graph.h` - Computation graph and automatic differentiation
- `tofu_optimizer.h` - Optimization algorithms

### Included in Public API
- ✅ Function signatures (`TOFU_EXPORT` functions)
- ✅ Public type definitions (`tofu_tensor`, `tofu_graph`, etc.)
- ✅ Public constants and enums (`TOFU_FLOAT`, operation types, etc.)
- ✅ Macro definitions (`TOFU_MAXDIM`, etc.)
- ✅ Documented behavior and semantics

### Excluded from Public API (May Change)
- ❌ Internal implementation details
- ❌ Private functions (no `TOFU_EXPORT` macro)
- ❌ Internal data structure layouts (opaque pointers)
- ❌ Undocumented behavior
- ❌ Performance characteristics (can improve in MINOR/PATCH)

---

## API Stability Guarantees

### What Won't Change Without MAJOR Version Bump

1. **Function Signatures**
   ```c
   // These will remain stable:
   tofu_tensor* tofu_tensor_matmul(const tofu_tensor* a, const tofu_tensor* b, tofu_tensor* dst);
   tofu_graph_node* tofu_graph_add(tofu_graph* g, tofu_graph_node* a, tofu_graph_node* b);
   ```

2. **Function Names**
   - All `tofu_*` prefixed functions
   - All `TOFU_*` prefixed macros and constants

3. **Type Names**
   - `tofu_tensor`, `tofu_graph`, `tofu_optimizer`, etc.
   - Enum values (`TOFU_OP_MATMUL`, `TOFU_FLOAT`, etc.)

4. **Behavior**
   - Documented semantics (e.g., NumPy broadcasting rules)
   - Error handling patterns (assert-based precondition checking)
   - Ownership semantics (documented in Doxygen comments)

### What Can Change in MINOR/PATCH Versions

1. **Internal Implementation**
   - Algorithm optimizations
   - Memory layout changes (opaque structures)
   - Internal helper functions

2. **Performance**
   - Speed improvements
   - Memory usage improvements
   - Better cache utilization

3. **Bug Fixes**
   - Fixes for incorrect behavior (PATCH)
   - Fixes that make behavior match documentation (PATCH)

4. **Additions**
   - New functions (MINOR)
   - New operation types (MINOR)
   - New optimizers (MINOR)

---

## Deprecation Policy

When we need to remove or change public API functions, we follow a **one major version warning period**:

### Deprecation Process

1. **Version N.x.x** (Current)
   - Feature is fully supported
   - No warnings

2. **Version (N+1).0.0** (Deprecation)
   - Feature is marked `TOFU_DEPRECATED`
   - Compiler warnings issued
   - Documentation updated with migration guide
   - Feature still works correctly

3. **Version (N+2).0.0** (Removal)
   - Feature is removed
   - Code using feature will not compile

### Example Timeline

```c
// v1.5.0: Feature is added
tofu_tensor* tofu_tensor_old_func(void);

// v2.0.0: Feature is deprecated
TOFU_DEPRECATED tofu_tensor* tofu_tensor_old_func(void);  // Warning issued
tofu_tensor* tofu_tensor_new_func(void);  // Replacement available

// v3.0.0: Feature is removed
// tofu_tensor_old_func() no longer exists
```

---

## Breaking Change Policy

### Requesting Breaking Changes

Breaking changes require strong justification. To propose a breaking change:

1. **Open a GitHub Issue**
   - Title: `[Breaking Change] <Description>`
   - Label: `breaking-change`, `api`
   - Template:
     ```markdown
     ## Motivation
     Why is this breaking change necessary?

     ## Current API
     What exists today?

     ## Proposed API
     What should it become?

     ## Migration Path
     How will users update their code?

     ## Alternatives Considered
     What non-breaking alternatives were explored?
     ```

2. **Review Process**
   - Core maintainers review
   - Community discussion (minimum 14 days)
   - Vote by core maintainers (requires consensus)

3. **Approval Requirements**
   - Must have compelling justification (security, correctness, major usability issue)
   - Must provide clear migration guide
   - Must be batched for next MAJOR version
   - Must follow deprecation policy (one version warning)

### Migration Guide Requirements

Every breaking change must include:

1. **Before/After Code Examples**
   ```c
   // Before (v1.x)
   tofu_tensor_old_api(x, y);

   // After (v2.x)
   tofu_tensor_new_api(x, y, default_value);
   ```

2. **Automated Migration** (if possible)
   - Regex patterns for find-and-replace
   - Script to update code automatically

3. **Rationale**
   - Why the change was necessary
   - What problems it solves

---

## Stability Commitment

### v1.0.0 and Beyond

Starting with v1.0.0, we commit to:

1. **No Breaking Changes in MINOR/PATCH versions**
   - Your code that works with v1.0.0 will work with v1.x.x

2. **Minimal Breaking Changes in MAJOR versions**
   - Breaking changes are batched for major versions
   - Each breaking change has a migration guide
   - Deprecation warnings given one major version in advance

3. **Long-Term Support (LTS)**
   - Each MAJOR version is supported for at least 1 year after next MAJOR release
   - Security fixes backported to previous MAJOR version
   - Critical bug fixes backported (at maintainer discretion)

### Pre-1.0 Stability

**Before v1.0.0** (including v0.x versions):
- Breaking changes may occur in MINOR versions
- We follow best effort for stability but prioritize getting API right
- Breaking changes are documented in CHANGELOG.md

---

## Contact

For questions about API stability:
- Open a GitHub Issue with label `api`
- Mention `@maintainers` for core team attention

For security issues, see SECURITY.md (to be created).

---

## Version History

- **2025-10-19**: Initial version for v1.0.0 release
