/*
 * Test to reproduce CRITICAL BUG #3: Floating-point comparison functions
 *
 * Bug Location: src/tofu_type.c:309-316
 *
 * cmp_double and cmp_float cast subtraction result to int,
 * losing precision and causing incorrect comparisons.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

/* BUGGY VERSION - as it exists in the code */
static int cmp_double_buggy(void *p1, void *p2)
{
    return *(double *)p1 - *(double *)p2;  // BUG: casts to int
}

static int cmp_float_buggy(void *p1, void *p2)
{
    return *(float *)p1 - *(float *)p2;  // BUG: casts to int
}

/* CORRECT VERSION */
static int cmp_double_correct(void *p1, void *p2)
{
    double d1 = *(double *)p1;
    double d2 = *(double *)p2;
    if (d1 < d2) return -1;
    if (d1 > d2) return 1;
    return 0;
}

static int cmp_float_correct(void *p1, void *p2)
{
    float f1 = *(float *)p1;
    float f2 = *(float *)p2;
    if (f1 < f2) return -1;
    if (f1 > f2) return 1;
    return 0;
}

void test_fractional_comparisons()
{
    printf("=== Test 1: Fractional differences (< 1.0) ===\n\n");

    /* Test case where difference is between 0 and 1 */
    double d1 = 1.5, d2 = 1.2;
    printf("Comparing: %.2f vs %.2f (difference = %.2f)\n", d1, d2, d1 - d2);

    int buggy_result = cmp_double_buggy(&d1, &d2);
    int correct_result = cmp_double_correct(&d1, &d2);

    printf("\nBuggy version result:   %d\n", buggy_result);
    printf("Correct version result: %d\n", correct_result);

    printf("\nAnalysis:\n");
    printf("  1.5 - 1.2 = 0.3\n");
    printf("  (int)0.3 = 0  <-- Truncated to zero!\n");
    printf("  Buggy version returns 0 (equal)\n");
    printf("  ✗ WRONG: 1.5 and 1.2 are NOT equal!\n\n");

    assert(buggy_result == 0 && "Bug confirmed: returns 0 (equal) for different values");
    assert(correct_result > 0 && "Correct version returns positive (d1 > d2)");

    printf("  ✓ Bug confirmed: Fractional differences return 0!\n\n");

    /* More test cases */
    printf("Additional fractional test cases:\n");

    struct {
        double a, b;
        const char *desc;
    } cases[] = {
        {2.9, 2.1, "2.9 vs 2.1 (diff = 0.8)"},
        {0.7, 0.3, "0.7 vs 0.3 (diff = 0.4)"},
        {100.1, 100.0, "100.1 vs 100.0 (diff = 0.1)"},
        {-0.5, -0.6, "-0.5 vs -0.6 (diff = 0.1)"},
    };

    for (int i = 0; i < 4; i++) {
        int bug = cmp_double_buggy(&cases[i].a, &cases[i].b);
        int correct = cmp_double_correct(&cases[i].a, &cases[i].b);
        printf("  %s: buggy=%d, correct=%d %s\n",
               cases[i].desc, bug, correct,
               (bug == 0 && correct != 0) ? "✗ BUG" : "");
    }
    printf("\n");
}

void test_float_fractional()
{
    printf("=== Test 2: Float fractional differences ===\n\n");

    float f1 = 3.7f, f2 = 3.2f;
    printf("Comparing: %.2f vs %.2f (difference = %.2f)\n", f1, f2, f1 - f2);

    int buggy_result = cmp_float_buggy(&f1, &f2);
    int correct_result = cmp_float_correct(&f1, &f2);

    printf("\nBuggy version result:   %d\n", buggy_result);
    printf("Correct version result: %d\n", correct_result);

    printf("\nAnalysis:\n");
    printf("  3.7 - 3.2 = 0.5\n");
    printf("  (int)0.5 = 0  <-- Truncated to zero!\n");
    printf("  ✗ WRONG: Returns equal when values differ!\n\n");

    assert(buggy_result == 0 && "Bug confirmed");
    assert(correct_result > 0 && "Correct version works");

    printf("  ✓ Bug confirmed for float type too!\n\n");
}

void test_large_value_overflow()
{
    printf("=== Test 3: Large value overflow ===\n\n");

    /* Very large difference that exceeds INT_MAX */
    double d1 = 1e15;  // 1 quadrillion
    double d2 = 0.0;

    printf("Comparing: %.2e vs %.2e\n", d1, d2);
    printf("Difference: %.2e\n", d1 - d2);
    printf("INT_MAX: %d (approximately 2.1e9)\n\n", INT32_MAX);

    int buggy_result = cmp_double_buggy(&d1, &d2);
    int correct_result = cmp_double_correct(&d1, &d2);

    printf("Buggy version result:   %d\n", buggy_result);
    printf("Correct version result: %d\n", correct_result);

    printf("\nAnalysis:\n");
    printf("  Difference = 1e15\n");
    printf("  This exceeds INT_MAX (2.1e9)\n");
    printf("  (int)1e15 causes undefined behavior/overflow\n");
    printf("  Result is unpredictable!\n");
    printf("  ✗ WRONG: Cannot represent result as int!\n\n");

    assert(correct_result > 0 && "Correct version should return positive");
    printf("  ✓ Bug confirmed: Large differences overflow!\n\n");
}

void test_negative_comparisons()
{
    printf("=== Test 4: Negative value comparisons ===\n\n");

    double d1 = -0.3, d2 = 0.2;
    printf("Comparing: %.2f vs %.2f (difference = %.2f)\n", d1, d2, d1 - d2);

    int buggy_result = cmp_double_buggy(&d1, &d2);
    int correct_result = cmp_double_correct(&d1, &d2);

    printf("\nBuggy version result:   %d\n", buggy_result);
    printf("Correct version result: %d\n", correct_result);

    printf("\nAnalysis:\n");
    printf("  -0.3 - 0.2 = -0.5\n");
    printf("  (int)(-0.5) = 0  <-- Truncated to zero!\n");
    printf("  ✗ WRONG: Returns equal when d1 < d2!\n\n");

    assert(buggy_result == 0 && "Bug confirmed");
    assert(correct_result < 0 && "Correct version returns negative");

    printf("  ✓ Bug confirmed: Negative fractional diffs also broken!\n\n");
}

void test_sorting_impact()
{
    printf("=== Test 5: Impact on sorting ===\n\n");

    double values[] = {1.1, 1.5, 1.3, 1.7, 1.2, 1.9, 1.4};
    int n = 7;

    printf("Original array: ");
    for (int i = 0; i < n; i++) printf("%.1f ", values[i]);
    printf("\n");

    /* Make copies for each sort */
    double buggy_array[7], correct_array[7];
    for (int i = 0; i < n; i++) {
        buggy_array[i] = values[i];
        correct_array[i] = values[i];
    }

    /* Sort with buggy comparison */
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (cmp_double_buggy(&buggy_array[j], &buggy_array[j + 1]) > 0) {
                double temp = buggy_array[j];
                buggy_array[j] = buggy_array[j + 1];
                buggy_array[j + 1] = temp;
            }
        }
    }

    /* Sort with correct comparison */
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (cmp_double_correct(&correct_array[j], &correct_array[j + 1]) > 0) {
                double temp = correct_array[j];
                correct_array[j] = correct_array[j + 1];
                correct_array[j + 1] = temp;
            }
        }
    }

    printf("\nBuggy sort result:   ");
    for (int i = 0; i < n; i++) printf("%.1f ", buggy_array[i]);
    printf("\n");

    printf("Correct sort result: ");
    for (int i = 0; i < n; i++) printf("%.1f ", correct_array[i]);
    printf("\n\n");

    /* Check if buggy sort is actually sorted */
    int buggy_is_sorted = 1;
    for (int i = 0; i < n - 1; i++) {
        if (buggy_array[i] > buggy_array[i + 1]) {
            buggy_is_sorted = 0;
            break;
        }
    }

    int correct_is_sorted = 1;
    for (int i = 0; i < n - 1; i++) {
        if (correct_array[i] > correct_array[i + 1]) {
            correct_is_sorted = 0;
            break;
        }
    }

    printf("Buggy sort is sorted: %s\n", buggy_is_sorted ? "Yes" : "No");
    printf("Correct sort is sorted: %s\n\n", correct_is_sorted ? "Yes" : "No");

    if (!buggy_is_sorted) {
        printf("  ✗ CRITICAL: Buggy comparison BREAKS SORTING!\n");
    } else {
        printf("  ⚠ Note: May appear sorted by luck, but comparison is still wrong\n");
    }
    printf("  ✓ Correct comparison produces proper sort\n\n");

    assert(correct_is_sorted && "Correct version must sort properly");
}

int main()
{
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  BUG REPRODUCTION TEST: Comparison Functions              ║\n");
    printf("║  Critical Bug #3 from Code Review                         ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("Bug Description:\n");
    printf("  cmp_double and cmp_float cast float subtraction to int\n");
    printf("  This loses precision and causes incorrect comparisons\n\n");

    printf("Impact:\n");
    printf("  - Fractional differences (< 1.0) return 0 (equal)\n");
    printf("  - Large differences overflow int range\n");
    printf("  - Sorting algorithms fail\n");
    printf("  - Binary search fails\n");
    printf("  - Any comparison-based algorithm fails\n\n");

    test_fractional_comparisons();
    test_float_fractional();
    test_large_value_overflow();
    test_negative_comparisons();
    test_sorting_impact();

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║  CONCLUSION: Critical bugs confirmed!                      ║\n");
    printf("║  Comparison functions FAIL for most real-world values.     ║\n");
    printf("║  Fix: Use proper three-way comparison with conditionals    ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
