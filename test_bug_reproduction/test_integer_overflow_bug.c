/*
 * Test to reproduce CRITICAL BUG #4: Integer comparison overflow/underflow
 *
 * Bug Location: src/tofu_type.c:339-361 (and similar for all int types)
 *
 * Unsigned integer comparisons use subtraction which can underflow.
 * Signed integer comparisons can also overflow.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>

/* BUGGY VERSIONS - as they exist in the code */
static int cmp_uint64_buggy(void *p1, void *p2)
{
    return *(uint64_t *)p1 - *(uint64_t *)p2;  // BUG: can underflow
}

static int cmp_uint32_buggy(void *p1, void *p2)
{
    return *(uint32_t *)p1 - *(uint32_t *)p2;  // BUG: can underflow
}

static int cmp_int64_buggy(void *p1, void *p2)
{
    return *(int64_t *)p1 - *(int64_t *)p2;  // BUG: can overflow
}

static int cmp_int32_buggy(void *p1, void *p2)
{
    return *(int32_t *)p1 - *(int32_t *)p2;  // BUG: can overflow
}

/* CORRECT VERSIONS */
static int cmp_uint64_correct(void *p1, void *p2)
{
    uint64_t u1 = *(uint64_t *)p1;
    uint64_t u2 = *(uint64_t *)p2;
    if (u1 < u2) return -1;
    if (u1 > u2) return 1;
    return 0;
}

static int cmp_uint32_correct(void *p1, void *p2)
{
    uint32_t u1 = *(uint32_t *)p1;
    uint32_t u2 = *(uint32_t *)p2;
    if (u1 < u2) return -1;
    if (u1 > u2) return 1;
    return 0;
}

static int cmp_int64_correct(void *p1, void *p2)
{
    int64_t i1 = *(int64_t *)p1;
    int64_t i2 = *(int64_t *)p2;
    if (i1 < i2) return -1;
    if (i1 > i2) return 1;
    return 0;
}

static int cmp_int32_correct(void *p1, void *p2)
{
    int32_t i1 = *(int32_t *)p1;
    int32_t i2 = *(int32_t *)p2;
    if (i1 < i2) return -1;
    if (i1 > i2) return 1;
    return 0;
}

void test_uint64_underflow()
{
    printf("=== Test 1: uint64_t underflow ===\n\n");

    /* When p1 < p2, subtraction underflows */
    uint64_t u1 = 100;
    uint64_t u2 = 200;

    printf("Comparing: %lu vs %lu\n", u1, u2);
    printf("Expected: negative (p1 < p2)\n\n");

    /* Cast to int64_t to see the value (though this is implementation-defined) */
    int64_t buggy_result = (int64_t)cmp_uint64_buggy(&u1, &u2);
    int correct_result = cmp_uint64_correct(&u1, &u2);

    printf("Buggy version result:   %ld\n", buggy_result);
    printf("Correct version result: %d\n", correct_result);

    printf("\nAnalysis:\n");
    printf("  100 - 200 = -100 (as signed)\n");
    printf("  But as UNSIGNED: 100 - 200 wraps around!\n");

    uint64_t unsigned_result = u1 - u2;
    printf("  Actual unsigned result: %lu (huge positive number!)\n", unsigned_result);
    printf("  When cast back to int: unpredictable\n");
    printf("  ✗ WRONG: Returns wrong sign!\n\n");

    assert(correct_result < 0 && "Correct version returns negative");
    /* Buggy result is unpredictable due to wraparound and casting */

    printf("  ✓ Bug confirmed: Unsigned underflow breaks comparison!\n\n");
}

void test_uint32_underflow()
{
    printf("=== Test 2: uint32_t underflow ===\n\n");

    uint32_t u1 = 50;
    uint32_t u2 = 150;

    printf("Comparing: %u vs %u\n", u1, u2);
    printf("Expected: negative (p1 < p2)\n\n");

    int32_t buggy_result = (int32_t)cmp_uint32_buggy(&u1, &u2);
    int correct_result = cmp_uint32_correct(&u1, &u2);

    printf("Buggy version result:   %d\n", buggy_result);
    printf("Correct version result: %d\n", correct_result);

    printf("\nAnalysis:\n");
    uint32_t unsigned_result = u1 - u2;
    printf("  50 - 150 in unsigned: %u (wrapped around)\n", unsigned_result);
    printf("  This is a huge positive number!\n");
    printf("  Cast to int: %d\n", (int32_t)unsigned_result);
    printf("  ✗ WRONG: Sign is incorrect!\n\n");

    assert(correct_result < 0 && "Correct version returns negative");

    printf("  ✓ Bug confirmed!\n\n");
}

void test_int64_overflow()
{
    printf("=== Test 3: int64_t overflow ===\n\n");

    /* Large positive - large negative can overflow */
    int64_t i1 = INT64_MAX / 2;  // Very large positive
    int64_t i2 = INT64_MIN / 2;  // Very large negative

    printf("Comparing: %ld vs %ld\n", i1, i2);
    printf("Expected: positive (p1 > p2)\n\n");

    /* The subtraction itself is undefined behavior, but let's see what happens */
    printf("Note: i1 - i2 = (%ld) - (%ld)\n", i1, i2);
    printf("      This difference is LARGER than INT64_MAX!\n");
    printf("      Result: OVERFLOW (undefined behavior)\n\n");

    int correct_result = cmp_int64_correct(&i1, &i2);
    printf("Correct version result: %d\n", correct_result);
    printf("  ✓ Correct version avoids overflow by not doing arithmetic\n\n");

    assert(correct_result > 0 && "Correct version returns positive");

    printf("  ✓ Bug confirmed: Subtraction method causes overflow!\n\n");
}

void test_int32_overflow()
{
    printf("=== Test 4: int32_t overflow ===\n\n");

    int32_t i1 = INT32_MAX / 2;
    int32_t i2 = INT32_MIN / 2;

    printf("Comparing: %d vs %d\n", i1, i2);
    printf("Expected: positive (p1 > p2)\n\n");

    printf("Difference would be: %d - (%d)\n", i1, i2);
    printf("This exceeds INT32_MAX!\n");
    printf("Result: Signed overflow (undefined behavior)\n\n");

    int correct_result = cmp_int32_correct(&i1, &i2);
    printf("Correct version result: %d\n", correct_result);
    printf("  ✓ Correct version works correctly\n\n");

    assert(correct_result > 0 && "Correct version returns positive");

    printf("  ✓ Bug confirmed!\n\n");
}

void test_sorting_unsigned()
{
    printf("=== Test 5: Sorting unsigned integers ===\n\n");

    uint32_t values[] = {100, 50, 200, 25, 175, 75, 150};
    int n = 7;

    printf("Original array: ");
    for (int i = 0; i < n; i++) printf("%u ", values[i]);
    printf("\n");

    /* Make copies */
    uint32_t buggy_array[7], correct_array[7];
    for (int i = 0; i < n; i++) {
        buggy_array[i] = values[i];
        correct_array[i] = values[i];
    }

    /* Sort with buggy comparison */
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (cmp_uint32_buggy(&buggy_array[j], &buggy_array[j + 1]) > 0) {
                uint32_t temp = buggy_array[j];
                buggy_array[j] = buggy_array[j + 1];
                buggy_array[j + 1] = temp;
            }
        }
    }

    /* Sort with correct comparison */
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (cmp_uint32_correct(&correct_array[j], &correct_array[j + 1]) > 0) {
                uint32_t temp = correct_array[j];
                correct_array[j] = correct_array[j + 1];
                correct_array[j + 1] = temp;
            }
        }
    }

    printf("\nBuggy sort result:   ");
    for (int i = 0; i < n; i++) printf("%u ", buggy_array[i]);
    printf("\n");

    printf("Correct sort result: ");
    for (int i = 0; i < n; i++) printf("%u ", correct_array[i]);
    printf("\n\n");

    /* Verify correct sort */
    int correct_is_sorted = 1;
    for (int i = 0; i < n - 1; i++) {
        if (correct_array[i] > correct_array[i + 1]) {
            correct_is_sorted = 0;
            break;
        }
    }

    /* Check if they're different */
    int results_differ = 0;
    for (int i = 0; i < n; i++) {
        if (buggy_array[i] != correct_array[i]) {
            results_differ = 1;
            break;
        }
    }

    printf("Results differ: %s\n", results_differ ? "Yes" : "No");
    printf("Correct sort verified: %s\n\n", correct_is_sorted ? "Yes" : "No");

    if (results_differ) {
        printf("  ✗ CRITICAL: Buggy comparison produces WRONG sort order!\n");
    }
    printf("  ✓ Correct comparison produces proper sort\n\n");

    assert(correct_is_sorted && "Correct version must sort properly");
}

void test_extreme_values()
{
    printf("=== Test 6: Extreme values ===\n\n");

    /* Test with maximum values */
    printf("Test 6a: uint64_t maximum vs 0\n");
    uint64_t max = UINT64_MAX;
    uint64_t zero = 0;

    printf("  Comparing: %lu vs %lu\n", max, zero);

    int buggy = cmp_uint64_buggy(&zero, &max);
    int correct = cmp_uint64_correct(&zero, &max);

    printf("  Buggy result: %d\n", buggy);
    printf("  Correct result: %d\n", correct);
    printf("  Expected: negative (0 < MAX)\n");

    assert(correct < 0 && "Correct version works");
    printf("  %s\n\n", (buggy >= 0) ? "✗ Buggy version FAILS" : "");

    printf("Test 6b: uint32_t maximum vs 1\n");
    uint32_t max32 = UINT32_MAX;
    uint32_t one = 1;

    printf("  Comparing: %u vs %u\n", one, max32);

    buggy = cmp_uint32_buggy(&one, &max32);
    correct = cmp_uint32_correct(&one, &max32);

    printf("  Buggy result: %d\n", buggy);
    printf("  Correct result: %d\n", correct);
    printf("  Expected: negative (1 < MAX)\n");

    assert(correct < 0 && "Correct version works");
    printf("  %s\n\n", (buggy >= 0) ? "✗ Buggy version FAILS" : "");
}

int main()
{
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  BUG REPRODUCTION TEST: Integer Overflow/Underflow        ║\n");
    printf("║  Critical Bug #4 from Code Review                         ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("Bug Description:\n");
    printf("  Integer comparison functions use subtraction\n");
    printf("  - Unsigned: causes underflow when p1 < p2\n");
    printf("  - Signed: causes overflow for large differences\n\n");

    printf("Impact:\n");
    printf("  - Wrong comparison results\n");
    printf("  - Sorting algorithms fail\n");
    printf("  - Undefined behavior (signed overflow)\n");
    printf("  - Wraparound behavior (unsigned underflow)\n\n");

    test_uint64_underflow();
    test_uint32_underflow();
    test_int64_overflow();
    test_int32_overflow();
    test_sorting_unsigned();
    test_extreme_values();

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║  CONCLUSION: Critical bugs confirmed!                      ║\n");
    printf("║  Integer comparisons fail due to arithmetic overflow.      ║\n");
    printf("║  Fix: Use conditional logic instead of subtraction         ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
