# Milestone 4: API Stabilization (v0.9.0 - Release Candidate)

**Goal**: Freeze API for v1.0.0, ensure stability and consistency
**Duration**: 5-7 days
**Current Version**: v0.5.0

---

## Philosophy: What Does "API Freeze" Mean?

After v0.9.0, the public API is **frozen** for v1.0.0:
- No breaking changes without major version bump (v2.0.0)
- Functions can be added (minor version bump)
- Bugs can be fixed (patch version bump)
- Internal implementation can change (as long as API remains stable)

**This is our last chance to make breaking changes before v1.0.0!**

---

## Phase 1: API Audit (Days 1-2)

### Task 1.1: Inventory Public API
**Goal**: Create complete list of all public-facing APIs

**Actions**:
1. List all exported headers (from `EXPORT_HEADERS` in config)
2. For each header, extract all:
   - Function declarations
   - Struct definitions
   - Enum definitions
   - Macro definitions
3. Create `API_INVENTORY.md` with categorization

**Deliverable**: `API_INVENTORY.md` listing all public APIs

**Acceptance Criteria**:
- All exported symbols documented
- Categorized by module (tensor, graph, optimizer)
- Current usage notes (where used in examples)

---

### Task 1.2: Naming Consistency Audit
**Goal**: Ensure all public APIs follow consistent naming conventions

**Criteria to Check**:
- ✓ All functions use `snake_case`
- ✓ All functions prefixed with `tofu_` (namespace)
- ✓ Struct names follow `tofu_<name>` pattern
- ✓ Enum values use `TOFU_<NAME>` pattern
- ✓ No Hungarian notation (no `pFoo`, `iCount`)
- ✓ Clear, descriptive names (avoid abbreviations)

**Actions**:
1. Run automated scan for naming violations
2. Document any inconsistencies
3. Propose renames (if any breaking changes needed)

**Deliverable**: `API_NAMING_AUDIT.md`

**Acceptance Criteria**:
- List of any naming violations
- Proposed fixes for inconsistencies
- Decision: accept current names or rename before v1.0

---

### Task 1.3: API Usability Review
**Goal**: Ensure APIs are intuitive and well-designed

**Questions to Answer**:
1. **Parameter Order**: Is it consistent?
   - Example: `tofu_tensor_create(data, ndim, dims, dtype)` - is order logical?
   - Do similar functions follow same parameter order?

2. **NULL Handling**: Is it consistent?
   - Which parameters can be NULL?
   - Is behavior documented?
   - Example: `tofu_tensor_matmul(src1, src2, dst)` - can `dst` be NULL?

3. **Ownership Semantics**: Who owns what?
   - When does caller need to free memory?
   - When does function take ownership?
   - Example: `tofu_graph_input(g, tensor)` - does graph own tensor?

4. **Error Handling**: Is it consistent?
   - Current: mostly `assert()` for errors
   - Should any functions return error codes?
   - Decision: stay with assert or add return codes?

**Actions**:
1. Review top 20 most-used functions
2. Check examples for common usage patterns
3. Identify pain points or confusion

**Deliverable**: `API_USABILITY_REVIEW.md`

**Acceptance Criteria**:
- Document ownership semantics
- Document NULL handling rules
- List any confusing APIs
- Propose simplifications (if needed)

---

## Phase 2: API Documentation (Days 3-4)

### Task 2.1: Add Function-Level Comments
**Goal**: Every public function has basic documentation

**Format** (Doxygen-compatible):
```c
/**
 * @brief Brief one-line description
 * @param param1 Description of param1
 * @param param2 Description of param2 (can be NULL)
 * @return Description of return value
 * @note Additional notes (ownership, thread-safety, etc.)
 */
TOFU_EXPORT tofu_tensor* tofu_tensor_create(...);
```

**Priority Order**:
1. **High**: Core tensor operations (create, free, matmul, etc.)
2. **Medium**: Graph operations (input, param, backward)
3. **Low**: Utility functions

**Actions**:
1. Add comments to `tofu_tensor.h` (30+ functions)
2. Add comments to `tofu_graph.h` (20+ functions)
3. Add comments to `tofu_optimizer.h` (10+ functions)

**Deliverable**: Updated header files with inline documentation

**Acceptance Criteria**:
- All public functions have `@brief` comment
- All parameters documented
- Ownership semantics noted
- NULL handling documented

---

### Task 2.2: Create API Stability Document
**Goal**: Document API guarantees and versioning policy

**Document**: `API_STABILITY.md`

**Contents**:
1. **Semantic Versioning Policy**:
   - Major (v2.0.0): Breaking changes allowed
   - Minor (v1.1.0): New features, no breaking changes
   - Patch (v1.0.1): Bug fixes only

2. **API Stability Guarantees**:
   - What's stable: function signatures, struct layouts
   - What can change: internal implementation, performance
   - Deprecation policy: 1 major version warning period

3. **Breaking Change Policy**:
   - How to request breaking changes (GitHub issue)
   - Review process
   - Migration guide requirements

**Deliverable**: `API_STABILITY.md`

**Acceptance Criteria**:
- Clear versioning policy
- Explicit stability guarantees
- Deprecation process defined

---

### Task 2.3: Update README with API Stability Announcement
**Goal**: Communicate API freeze to users

**Changes to README**:
1. Add "API Status" section:
   ```markdown
   ## API Status

   **Current Version**: v0.9.0 (Release Candidate)
   **API Status**: Frozen for v1.0.0

   The public API is now frozen. No breaking changes will be made before v1.0.0.
   After v1.0.0, we follow semantic versioning (see [API_STABILITY.md](API_STABILITY.md)).
   ```

2. Link to API_STABILITY.md

**Deliverable**: Updated README.md

---

## Phase 3: Performance Baseline (Day 5)

**Note**: Performance optimization is **optional** for v1.0.0, but establishing a baseline helps track progress.

### Task 3.1: Create Micro-Benchmarks
**Goal**: Measure performance of core operations

**Benchmarks to Create**:
1. **Matmul Performance**:
   - Sizes: [64×64], [128×128], [256×256], [512×512]
   - Measure: GFLOPS, time per operation

2. **Memory Allocation**:
   - Tensor creation/destruction rate
   - Graph creation/destruction overhead

3. **Backward Pass**:
   - Time for forward vs backward pass
   - Gradient computation overhead

**Implementation**:
- Create `benchmarks/micro_benchmarks.c`
- Use `clock_gettime()` for timing
- Run 100 iterations, report average

**Deliverable**: `benchmarks/micro_benchmarks.c` + `PERFORMANCE_BASELINE.md`

**Acceptance Criteria**:
- Benchmarks run on CI/CD
- Results documented in PERFORMANCE_BASELINE.md
- Establishes baseline for future optimization

---

### Task 3.2: Profile Examples (Optional)
**Goal**: Identify hotspots in real-world usage

**Actions**:
1. Profile CNN training example with Instruments (macOS) or perf (Linux)
2. Identify top 10 functions by CPU time
3. Document findings

**Deliverable**: `PROFILING_REPORT.md` (optional)

**Acceptance Criteria**:
- Top 10 hotspots identified
- No obvious low-hanging fruit (if found, optionally optimize)

---

## Phase 4: Build System Verification (Day 6)

### Task 4.1: Verify macOS Build
**Goal**: Ensure clean builds on macOS

**Actions**:
1. Test on macOS 13+ (already done in CI/CD)
2. Test with Xcode Command Line Tools
3. Test with Homebrew gcc/clang
4. Verify pkg-config integration

**Deliverable**: Clean CI/CD pass (already passing)

---

### Task 4.2: Test ESP32 Cross-Compilation
**Goal**: Ensure ESP32 builds still work

**Actions**:
1. Run `./configure --esp32=yes`
2. Build library with ESP32 toolchain
3. Verify no compilation errors

**Deliverable**: Successful ESP32 build

**Acceptance Criteria**:
- Library compiles for ESP32 without errors
- Document any limitations (e.g., no examples for ESP32)

---

### Task 4.3: Document Supported Platforms
**Goal**: Clearly state what platforms are supported

**Update README**:
```markdown
## Supported Platforms

| Platform | Build | Tests | Status |
|----------|-------|-------|--------|
| Linux (Ubuntu 20.04+) | ✅ | ✅ | Fully supported |
| macOS (13+) | ✅ | ✅ | Fully supported |
| ESP32 | ✅ | ⚠️ | Build only (no test suite) |
| Windows | ❌ | ❌ | Not supported (v1.1.0+) |

**Build Requirements**:
- GCC 7+ or Clang 10+
- CMake or GNU Make
- pkg-config
```

**Deliverable**: Updated README with platform support table

---

## Phase 5: Release Preparation (Day 7)

### Task 5.1: Draft Release Notes
**Goal**: Summarize changes from v0.1.0 → v0.9.0

**Document**: `RELEASE_NOTES_v0.9.0.md`

**Contents**:
1. **Highlights**:
   - API freeze announcement
   - All gradients implemented and validated
   - CI/CD with multi-platform testing
   - Memory safety verified (AddressSanitizer)

2. **Breaking Changes** (if any):
   - List any renamed functions
   - Migration guide

3. **New Features**:
   - Edge case test suite
   - Enhanced examples (CNN, ResNet)

4. **Bug Fixes**:
   - Memory leaks fixed
   - View operations corrected

5. **Known Limitations**:
   - Assert-based error handling
   - No Windows support

**Deliverable**: `RELEASE_NOTES_v0.9.0.md`

---

### Task 5.2: Update Version Numbers
**Goal**: Bump version to v0.9.0

**Files to Update**:
1. `README.md` - Badge: `version-0.9.0`
2. `CHANGELOG.md` - Add v0.9.0 entry
3. `configure` script - Version string
4. Any version constants in code

**Deliverable**: Version bump commit

---

### Task 5.3: Final Testing Pass
**Goal**: Ensure everything works before tagging v0.9.0

**Checklist**:
- [ ] All CI/CD tests pass (Phase 1, Phase 3, edge cases)
- [ ] Examples compile and run (CNN, ResNet, MLP, ViT)
- [ ] ESP32 cross-compilation works
- [ ] No memory leaks (AddressSanitizer clean)
- [ ] Documentation up to date

**Deliverable**: All tests passing

---

### Task 5.4: Tag v0.9.0 Release
**Goal**: Create release candidate tag

**Actions**:
1. Commit all changes
2. Tag: `git tag -a v0.9.0 -m "v0.9.0: API Freeze - Release Candidate"`
3. Push tag: `git push origin v0.9.0`
4. Create GitHub release with RELEASE_NOTES_v0.9.0.md

**Deliverable**: v0.9.0 tagged and released

---

## Success Criteria

Milestone 4 is complete when:

1. ✅ **API Inventory**: All public APIs documented in API_INVENTORY.md
2. ✅ **API Consistency**: Naming audit complete, no major violations
3. ✅ **API Documentation**: All public functions have inline comments
4. ✅ **API Stability**: API_STABILITY.md published with guarantees
5. ✅ **Performance Baseline**: Micro-benchmarks created and results documented
6. ✅ **Build Verification**: macOS and ESP32 builds verified
7. ✅ **Platform Documentation**: Supported platforms documented in README
8. ✅ **Release Notes**: v0.9.0 release notes drafted
9. ✅ **Version Bump**: All version numbers updated to v0.9.0
10. ✅ **CI/CD Passing**: All tests pass on all platforms
11. ✅ **Release Tagged**: v0.9.0 tagged and pushed

---

## Deferred to v1.1.0+

The following items are **not critical** for v1.0.0:

1. **Windows Support**: No Windows machine available for testing
2. **SIMD Optimization**: Performance is acceptable, can optimize later
3. **Memory Pool Allocator**: Nice-to-have, not required
4. **Operation Fusion**: Advanced optimization, defer
5. **Comprehensive API Docs**: mdBook documentation deferred to post-v1.0.0

---

## Timeline

| Day | Phase | Tasks | Time |
|-----|-------|-------|------|
| 1 | API Audit | 1.1 Inventory, 1.2 Naming | 6-8h |
| 2 | API Audit | 1.3 Usability Review | 4-6h |
| 3 | API Docs | 2.1 Function Comments (tofu_tensor.h) | 6-8h |
| 4 | API Docs | 2.1 Function Comments (tofu_graph.h, tofu_optimizer.h), 2.2 Stability Doc | 6-8h |
| 5 | Performance | 3.1 Micro-benchmarks, 3.2 Profiling (optional) | 6-8h |
| 6 | Build System | 4.1 macOS, 4.2 ESP32, 4.3 Platform Docs | 4-6h |
| 7 | Release Prep | 5.1 Release Notes, 5.2 Version Bump, 5.3 Testing, 5.4 Tag | 6-8h |

**Total**: 38-52 hours (~5-7 days)

---

## Open Questions

1. **Breaking Changes**: Should we rename any APIs before freezing?
2. **Error Handling**: Stay with assert() or add return codes?
3. **Performance Targets**: What's acceptable for v1.0.0?
4. **ESP32 Testing**: Can we test on actual hardware?

These should be discussed before starting Phase 1.
