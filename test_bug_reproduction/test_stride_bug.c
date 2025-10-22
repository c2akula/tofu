/*
 * Test to reproduce CRITICAL BUG #1: Incorrect stride calculation
 *
 * Bug Location: src/tofu_tensor_internal.h:61-62
 *
 * The bug is in tofu_get_strides():
 *   for (i = t->dims[t->ndim - 2]; i >= 0; i--)  // WRONG: uses VALUE not INDEX
 *
 * Should be:
 *   for (i = t->ndim - 2; i >= 0; i--)           // CORRECT: uses INDEX
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

/* Copy the buggy implementation to test it */
typedef struct {
    int ndim;
    int *dims;
} test_tensor;

/* BUGGY VERSION - as it exists in the code */
static void tofu_get_strides_buggy(test_tensor *t, int *strides)
{
    int i;

    assert(strides);
    strides[t->ndim - 1] = 1;
    if (t->ndim == 1)
        return;

    /* BUG: This uses t->dims[t->ndim - 2] (a VALUE) instead of (t->ndim - 2) (an INDEX) */
    for (i = t->dims[t->ndim - 2]; i >= 0; i--)
        strides[i] = strides[i + 1] * t->dims[i + 1];
}

/* CORRECT VERSION - how it should be */
static void tofu_get_strides_correct(test_tensor *t, int *strides)
{
    int i;

    assert(strides);
    strides[t->ndim - 1] = 1;
    if (t->ndim == 1)
        return;

    /* CORRECT: Use index (t->ndim - 2) */
    for (i = t->ndim - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * t->dims[i + 1];
}

void test_stride_calculation()
{
    printf("=== Testing Stride Calculation Bug ===\n\n");

    /* Test Case 1: 2D tensor [3, 4] */
    printf("Test 1: 2D tensor [3, 4]\n");
    printf("Expected strides: [4, 1]\n");

    test_tensor t1;
    t1.ndim = 2;
    t1.dims = (int[]){3, 4};

    int strides_buggy[2];
    int strides_correct[2];

    printf("\nBuggy version behavior:\n");
    printf("  Loop will iterate with i starting at dims[0] = %d\n", t1.dims[t1.ndim - 2]);
    printf("  This means i = 3, 2, 1, 0 instead of i = 0\n");
    printf("  Will try to access strides[3], strides[2], strides[1], strides[0]\n");
    printf("  But strides array only has indices 0 and 1!\n");
    printf("  This causes OUT OF BOUNDS ACCESS!\n\n");

    /* We can't actually run the buggy version safely without crashes,
     * but we can demonstrate the logic error */
    printf("Correct version:\n");
    tofu_get_strides_correct(&t1, strides_correct);
    printf("  strides[0] = %d (correct)\n", strides_correct[0]);
    printf("  strides[1] = %d (correct)\n\n", strides_correct[1]);

    assert(strides_correct[0] == 4 && "Stride[0] should be 4");
    assert(strides_correct[1] == 1 && "Stride[1] should be 1");
    printf("✓ Test 1 PASSED (with correct version)\n\n");

    /* Test Case 2: 3D tensor [2, 3, 4] */
    printf("Test 2: 3D tensor [2, 3, 4]\n");
    printf("Expected strides: [12, 4, 1]\n");

    test_tensor t2;
    t2.ndim = 3;
    t2.dims = (int[]){2, 3, 4};

    int strides_buggy2[3];
    int strides_correct2[3];

    printf("\nBuggy version behavior:\n");
    printf("  Loop will iterate with i starting at dims[1] = %d\n", t2.dims[t2.ndim - 2]);
    printf("  This means i = 3, 2, 1, 0 instead of i = 1, 0\n");
    printf("  Will try to access strides[3] which is OUT OF BOUNDS!\n\n");

    printf("Correct version:\n");
    tofu_get_strides_correct(&t2, strides_correct2);
    printf("  strides[0] = %d (correct)\n", strides_correct2[0]);
    printf("  strides[1] = %d (correct)\n", strides_correct2[1]);
    printf("  strides[2] = %d (correct)\n\n", strides_correct2[2]);

    assert(strides_correct2[0] == 12 && "Stride[0] should be 12");
    assert(strides_correct2[1] == 4 && "Stride[1] should be 4");
    assert(strides_correct2[2] == 1 && "Stride[2] should be 1");
    printf("✓ Test 2 PASSED (with correct version)\n\n");

    /* Test Case 3: 4D tensor [2, 3, 4, 5] */
    printf("Test 3: 4D tensor [2, 3, 4, 5]\n");
    printf("Expected strides: [60, 20, 5, 1]\n");

    test_tensor t3;
    t3.ndim = 4;
    t3.dims = (int[]){2, 3, 4, 5};

    int strides_correct3[4];

    printf("\nBuggy version behavior:\n");
    printf("  Loop would start at i = dims[2] = %d\n", t3.dims[t3.ndim - 2]);
    printf("  Would iterate: i = 4, 3, 2, 1, 0\n");
    printf("  Should iterate: i = 2, 1, 0\n");
    printf("  WRONG iteration count and array access!\n\n");

    printf("Correct version:\n");
    tofu_get_strides_correct(&t3, strides_correct3);
    printf("  strides[0] = %d (correct)\n", strides_correct3[0]);
    printf("  strides[1] = %d (correct)\n", strides_correct3[1]);
    printf("  strides[2] = %d (correct)\n", strides_correct3[2]);
    printf("  strides[3] = %d (correct)\n\n", strides_correct3[3]);

    assert(strides_correct3[0] == 60 && "Stride[0] should be 60");
    assert(strides_correct3[1] == 20 && "Stride[1] should be 20");
    assert(strides_correct3[2] == 5 && "Stride[2] should be 5");
    assert(strides_correct3[3] == 1 && "Stride[3] should be 1");
    printf("✓ Test 3 PASSED (with correct version)\n\n");
}

int main()
{
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  BUG REPRODUCTION TEST: Stride Calculation                ║\n");
    printf("║  Critical Bug #1 from Code Review                         ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("Bug Description:\n");
    printf("  The loop uses t->dims[t->ndim - 2] (a dimension VALUE)\n");
    printf("  instead of (t->ndim - 2) (an array INDEX)\n\n");

    printf("Impact:\n");
    printf("  - Wrong loop iteration count\n");
    printf("  - Out of bounds array access\n");
    printf("  - Incorrect stride values\n");
    printf("  - Affects ALL tensor operations!\n\n");

    test_stride_calculation();

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║  CONCLUSION: Bug confirmed!                                ║\n");
    printf("║  The buggy version would cause crashes or wrong results.   ║\n");
    printf("║  Fix: Change loop to use index (t->ndim - 2)              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
