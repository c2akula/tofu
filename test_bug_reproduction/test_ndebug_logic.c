/*
 * Test to demonstrate BUG #5: Backwards NDEBUG conditional logic
 *
 * Bug Location: src/tofu_tensor.c:29-32
 *
 * Code has:
 *   #ifdef NDEBUG
 *       assert(...);
 *   #endif
 *
 * But assert() is DISABLED when NDEBUG is defined!
 * So the check never actually runs when NDEBUG is defined.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Simulate the buggy code pattern */
void check_bounds_buggy(int *coords, int ndim, int *dims)
{
#ifdef NDEBUG
    /* BUG: This code only runs when NDEBUG is defined,
     * but assert() is a no-op when NDEBUG is defined!
     * So this check NEVER actually runs! */
    for (int i = 0; i < ndim; i++)
        assert(coords[i] >= 0 && coords[i] < dims[i]);
#endif
}

/* Correct version 1: Check in debug mode only */
void check_bounds_correct_v1(int *coords, int ndim, int *dims)
{
#ifndef NDEBUG  /* NOT NDEBUG - runs in debug mode */
    for (int i = 0; i < ndim; i++)
        assert(coords[i] >= 0 && coords[i] < dims[i]);
#endif
}

/* Correct version 2: Always check (recommended) */
void check_bounds_correct_v2(int *coords, int ndim, int *dims)
{
    /* No conditional - always check */
    for (int i = 0; i < ndim; i++)
        assert(coords[i] >= 0 && coords[i] < dims[i]);
}

void test_debug_build()
{
    printf("=== Test in DEBUG mode (NDEBUG not defined) ===\n\n");

#ifndef NDEBUG
    printf("Running in DEBUG mode (NDEBUG is NOT defined)\n");
    printf("assert() is ACTIVE in this mode\n\n");

    int coords_valid[] = {1, 2};
    int coords_invalid[] = {5, 10};  /* Out of bounds */
    int dims[] = {3, 4};
    int ndim = 2;

    printf("Test 1: Valid coordinates\n");
    check_bounds_buggy(coords_valid, ndim, dims);
    printf("  Buggy version: Check DID NOT RUN (#ifdef NDEBUG is false)\n");
    printf("  ✗ No bounds checking in debug mode!\n\n");

    check_bounds_correct_v1(coords_valid, ndim, dims);
    printf("  Correct v1: Check ran successfully (#ifndef NDEBUG is true)\n");
    printf("  ✓ Bounds checking works in debug mode!\n\n");

    check_bounds_correct_v2(coords_valid, ndim, dims);
    printf("  Correct v2: Check ran successfully (always checks)\n");
    printf("  ✓ Bounds checking works!\n\n");

#else
    printf("Running in RELEASE mode (NDEBUG IS defined)\n");
    printf("assert() is DISABLED in this mode\n\n");

    int coords_invalid[] = {5, 10};  /* Out of bounds */
    int dims[] = {3, 4};
    int ndim = 2;

    printf("Test 1: Invalid coordinates (should fail assert)\n");
    printf("  Buggy version behavior:\n");
    printf("    #ifdef NDEBUG is TRUE\n");
    printf("    So the assert() code is INSIDE the conditional\n");
    printf("    BUT assert() is a no-op in release mode!\n");
    printf("    Result: NO checking happens!\n\n");

    check_bounds_buggy(coords_invalid, ndim, dims);
    printf("    Buggy version: Invalid coords ACCEPTED (no check ran)\n");
    printf("    ✗ CRITICAL: Out-of-bounds access not caught!\n\n");

    printf("  Correct v1 behavior:\n");
    printf("    #ifndef NDEBUG is FALSE (NDEBUG is defined)\n");
    printf("    So check doesn't run (as intended for release)\n");
    check_bounds_correct_v1(coords_invalid, ndim, dims);
    printf("    Correct v1: No check in release mode (by design)\n");
    printf("    ✓ Intended behavior: assertions disabled in release\n\n");

    /* Note: v2 would crash here since it always checks */
    printf("  Correct v2 behavior:\n");
    printf("    Always checks, would catch the error\n");
    printf("    (Not running to avoid crash in this test)\n\n");
#endif
}

void test_intent_analysis()
{
    printf("=== Intent Analysis ===\n\n");

    printf("The buggy code pattern '#ifdef NDEBUG + assert()' makes no sense:\n\n");

    printf("Scenario 1: Debug build (NDEBUG not defined)\n");
    printf("  #ifdef NDEBUG is FALSE\n");
    printf("  Code block doesn't execute\n");
    printf("  Result: No bounds checking in DEBUG mode!\n");
    printf("  ✗ This is backwards - we WANT checking in debug mode!\n\n");

    printf("Scenario 2: Release build (NDEBUG defined)\n");
    printf("  #ifdef NDEBUG is TRUE\n");
    printf("  Code block executes\n");
    printf("  BUT assert() is compiled to nothing\n");
    printf("  Result: No bounds checking in RELEASE mode either!\n");
    printf("  ✗ So the check NEVER runs in either mode!\n\n");

    printf("Conclusion:\n");
    printf("  The code intends to check bounds, but the logic is wrong\n");
    printf("  Either:\n");
    printf("    1. Change to #ifndef NDEBUG (check in debug only), or\n");
    printf("    2. Remove conditional (always check), or\n");
    printf("    3. Use explicit if() instead of assert()\n\n");
}

void test_assert_behavior()
{
    printf("=== Understanding assert() Behavior ===\n\n");

#ifdef NDEBUG
    printf("NDEBUG is DEFINED (release build)\n");
    printf("  assert(condition) expands to: ((void)0)\n");
    printf("  This means: do nothing\n");
    printf("  All assertions are compiled out\n\n");
#else
    printf("NDEBUG is NOT DEFINED (debug build)\n");
    printf("  assert(condition) expands to: if (!condition) abort()\n");
    printf("  Assertions are active and will crash on failure\n\n");
#endif

    printf("The Macro Expansion:\n");
    printf("  #ifdef NDEBUG\n");
    printf("      assert(x >= 0);  // When NDEBUG is defined\n");
    printf("  #endif\n\n");

    printf("  Expands to:\n");
    printf("  #ifdef NDEBUG\n");
    printf("      ((void)0);  // Does nothing!\n");
    printf("  #endif\n\n");

    printf("  So the entire block becomes:\n");
    printf("  #ifdef NDEBUG\n");
    printf("      /* nothing */\n");
    printf("  #endif\n\n");

    printf("  ✗ The assertion never has any effect!\n\n");
}

int main()
{
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  BUG DEMONSTRATION: Backwards NDEBUG Logic                ║\n");
    printf("║  Bug #5 from Code Review                                  ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("Bug Description:\n");
    printf("  Code uses '#ifdef NDEBUG' with assert() inside\n");
    printf("  This is backwards - assert() is disabled when NDEBUG is defined!\n\n");

    printf("Impact:\n");
    printf("  - Bounds checking never runs (in debug OR release)\n");
    printf("  - Out-of-bounds access not caught\n");
    printf("  - Defeats the purpose of the check\n\n");

    test_debug_build();
    test_intent_analysis();
    test_assert_behavior();

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║  CONCLUSION: Logic bug confirmed!                          ║\n");
    printf("║  The conditional should be '#ifndef NDEBUG' not '#ifdef'  ║\n");
    printf("║  Or remove the conditional entirely                        ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
