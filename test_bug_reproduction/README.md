# Bug Reproduction Test Suite

This directory contains standalone test programs that reproduce and verify the critical bugs identified in the comprehensive codebase review.

## Purpose

These tests serve to:
1. **Confirm** that the identified bugs actually exist
2. **Demonstrate** the impact and behavior of each bug
3. **Validate** that proposed fixes resolve the issues
4. **Document** expected vs actual behavior

## Test Files

### 1. `test_stride_bug.c` - Critical Bug #1
**Bug:** Incorrect stride calculation in `tofu_get_strides()`
**Location:** `src/tofu_tensor_internal.h:61-62`
**Issue:** Loop uses `t->dims[t->ndim - 2]` (a VALUE) instead of `t->ndim - 2` (an INDEX)
**Impact:** Out-of-bounds array access, incorrect stride values, affects ALL tensor operations

### 2. `test_fprintf_bug.c` - Critical Bug #2
**Bug:** Type cast errors in fprintf functions
**Location:** `src/tofu_type.c:256, 264`
**Issue:** `fprintf_uint64` and `fprintf_uint32` cast to `uint16_t` instead of correct types
**Impact:** Data loss (48 bits for uint64, 16 bits for uint32), incorrect output

### 3. `test_comparison_bug.c` - Critical Bug #3
**Bug:** Floating-point comparison functions return incorrect values
**Location:** `src/tofu_type.c:309-316`
**Issue:** `cmp_double` and `cmp_float` cast float subtraction to int, losing precision
**Impact:**
- Fractional differences (< 1.0) return 0 (equal)
- Large differences overflow
- Sorting algorithms fail
- Any comparison-based algorithm fails

### 4. `test_integer_overflow_bug.c` - Critical Bug #4
**Bug:** Integer comparison functions overflow/underflow
**Location:** `src/tofu_type.c:339-361`
**Issue:** Unsigned comparisons underflow when p1 < p2, signed comparisons overflow
**Impact:** Wrong comparison results, sorting fails, undefined behavior

### 5. `test_ndebug_logic.c` - Bug #5
**Bug:** Backwards NDEBUG conditional logic
**Location:** `src/tofu_tensor.c:29-32`
**Issue:** Uses `#ifdef NDEBUG` with `assert()`, but asserts are disabled when NDEBUG is defined
**Impact:** Bounds checking never runs (in debug OR release mode)

## Building and Running

### Build All Tests
```bash
make
```

### Run All Tests
```bash
make run
```

### Run Individual Tests
```bash
make run-stride      # Test stride calculation bug
make run-fprintf     # Test fprintf bugs
make run-comparison  # Test comparison bugs
make run-overflow    # Test integer overflow bugs
make run-ndebug      # Test NDEBUG logic bug
```

### Build and Run Specific Test
```bash
./test_stride_bug
./test_fprintf_bug
./test_comparison_bug
./test_integer_overflow_bug
./test_ndebug_logic
```

### Test NDEBUG Logic in Release Mode
```bash
make run-ndebug-release
```

## Expected Output

Each test will:
1. Display a header with the bug description
2. Show the buggy behavior with detailed analysis
3. Show the correct behavior for comparison
4. Confirm the bug with assertions and status messages
5. Provide a conclusion with fix recommendations

Example output format:
```
╔════════════════════════════════════════════════════════════╗
║  BUG REPRODUCTION TEST: <Test Name>                       ║
║  Critical Bug #N from Code Review                         ║
╚════════════════════════════════════════════════════════════╝

Bug Description:
  <Description of the bug>

Impact:
  <Impact on the system>

=== Test 1: <Test Name> ===

<Test execution and analysis>

✓ Bug confirmed: <Summary>

╔════════════════════════════════════════════════════════════╗
║  CONCLUSION: <Conclusion>                                  ║
║  <Fix recommendation>                                      ║
╚════════════════════════════════════════════════════════════╝
```

## Test Methodology

Each test follows this pattern:

1. **Reproduce** - Implement both buggy and correct versions of the code
2. **Demonstrate** - Show concrete examples where the bug manifests
3. **Compare** - Run both versions and compare results
4. **Verify** - Use assertions to confirm the bug exists
5. **Document** - Explain why the bug occurs and what the correct behavior should be

## Integration with Main Codebase

These are **standalone tests** that do NOT require building the main Tofu library. They:
- Are self-contained (no external dependencies except libc and libm)
- Reproduce the bugs in isolation
- Can be compiled and run independently
- Serve as documentation of the bugs

## Using Tests to Verify Fixes

After fixing bugs in the main codebase:

1. Modify the test to use the actual library functions instead of the standalone versions
2. Link against the Tofu library
3. Verify that the test passes with the fixed code
4. Ensure the test still fails with the original buggy code (for regression testing)

## Clean Up

```bash
make clean
```

This removes all test executables.

## Notes

- Tests are designed to be safe and not crash (they demonstrate bugs without causing actual failures)
- Some tests show what WOULD happen with the buggy code rather than actually running it (e.g., stride bug would cause segfault)
- All tests include detailed comments explaining the bug mechanism
- Tests use vivid output formatting for easy readability

## Contributing

If you find additional bugs or want to add more test cases:

1. Create a new test file: `test_<bug_name>.c`
2. Follow the existing test structure and formatting
3. Add the test to the Makefile
4. Update this README
5. Submit a pull request

## Contact

See `CODEBASE_REVIEW_REPORT.md` in the root directory for the full code review report.
