/**
 * Comprehensive tests for enhanced arange functionality
 * Tests all cases: positive/negative step, empty arrays, edge cases
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tofu_tensor.h"

#define TEST_PASS 0
#define TEST_FAIL 1
#define EPSILON 1e-6

/**
 * Helper function to verify tensor contents match expected array
 */
int verify_tensor_values(tofu_tensor* t, float* expected, int expected_len, const char* test_name) {
    if (!t) {
        printf("  ✗ %s: Tensor is NULL\n", test_name);
        return TEST_FAIL;
    }

    if (t->ndim != 1) {
        printf("  ✗ %s: Expected 1D tensor, got %d dimensions\n", test_name, t->ndim);
        return TEST_FAIL;
    }

    if (t->len != expected_len) {
        printf("  ✗ %s: Expected length %d, got %d\n", test_name, expected_len, t->len);
        return TEST_FAIL;
    }

    float* data = (float*)t->data;
    for (int i = 0; i < expected_len; i++) {
        if (fabs(data[i] - expected[i]) > EPSILON) {
            printf("  ✗ %s: At index %d: expected %.6f, got %.6f\n",
                   test_name, i, expected[i], data[i]);
            return TEST_FAIL;
        }
    }

    printf("  ✓ %s\n", test_name);
    return TEST_PASS;
}

/**
 * Test 1: Basic forward slicing (existing behavior)
 */
int test_forward_slicing() {
    printf("\nTest 1: Forward Slicing\n");
    printf("------------------------\n");
    int passed = 0, total = 0;

    /* Test 1.1: arange(0, 5, 1) -> [0, 1, 2, 3, 4] */
    tofu_tensor* t1 = tofu_tensor_arange(0, 5, 1, TOFU_FLOAT);
    float expected1[] = {0, 1, 2, 3, 4};
    passed += (verify_tensor_values(t1, expected1, 5, "arange(0, 5, 1)") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t1);

    /* Test 1.2: arange(1, 10, 2) -> [1, 3, 5, 7, 9] */
    tofu_tensor* t2 = tofu_tensor_arange(1, 10, 2, TOFU_FLOAT);
    float expected2[] = {1, 3, 5, 7, 9};
    passed += (verify_tensor_values(t2, expected2, 5, "arange(1, 10, 2)") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t2);

    /* Test 1.3: arange(0.5, 3.5, 0.5) -> [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] */
    tofu_tensor* t3 = tofu_tensor_arange(0.5, 3.5, 0.5, TOFU_FLOAT);
    float expected3[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    passed += (verify_tensor_values(t3, expected3, 6, "arange(0.5, 3.5, 0.5)") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t3);

    printf("\nForward Slicing: %d/%d tests passed\n", passed, total);
    return (passed == total) ? TEST_PASS : TEST_FAIL;
}

/**
 * Test 2: Reverse slicing with negative step
 */
int test_reverse_slicing() {
    printf("\nTest 2: Reverse Slicing (Negative Step)\n");
    printf("----------------------------------------\n");
    int passed = 0, total = 0;

    /* Test 2.1: arange(10, 0, -1) -> [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] */
    tofu_tensor* t1 = tofu_tensor_arange(10, 0, -1, TOFU_FLOAT);
    float expected1[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    passed += (verify_tensor_values(t1, expected1, 10, "arange(10, 0, -1)") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t1);

    /* Test 2.2: arange(5, -1, -1) -> [5, 4, 3, 2, 1, 0] */
    tofu_tensor* t2 = tofu_tensor_arange(5, -1, -1, TOFU_FLOAT);
    float expected2[] = {5, 4, 3, 2, 1, 0};
    passed += (verify_tensor_values(t2, expected2, 6, "arange(5, -1, -1)") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t2);

    /* Test 2.3: arange(10, 0, -2) -> [10, 8, 6, 4, 2] */
    tofu_tensor* t3 = tofu_tensor_arange(10, 0, -2, TOFU_FLOAT);
    float expected3[] = {10, 8, 6, 4, 2};
    passed += (verify_tensor_values(t3, expected3, 5, "arange(10, 0, -2)") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t3);

    /* Test 2.4: arange(3.0, 0.0, -0.5) -> [3.0, 2.5, 2.0, 1.5, 1.0, 0.5] */
    tofu_tensor* t4 = tofu_tensor_arange(3.0, 0.0, -0.5, TOFU_FLOAT);
    float expected4[] = {3.0, 2.5, 2.0, 1.5, 1.0, 0.5};
    passed += (verify_tensor_values(t4, expected4, 6, "arange(3.0, 0.0, -0.5)") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t4);

    printf("\nReverse Slicing: %d/%d tests passed\n", passed, total);
    return (passed == total) ? TEST_PASS : TEST_FAIL;
}

/**
 * Test 3: Empty arrays (returns NULL)
 */
int test_empty_arrays() {
    printf("\nTest 3: Empty Arrays (NULL return)\n");
    printf("-----------------------------------\n");
    int passed = 0, total = 0;

    /* Test 3.1: arange(5, 5, 1) -> NULL (start == stop) */
    tofu_tensor* t1 = tofu_tensor_arange(5, 5, 1, TOFU_FLOAT);
    if (t1 == NULL) {
        printf("  ✓ arange(5, 5, 1) [start==stop] returns NULL\n");
        passed++;
    } else {
        printf("  ✗ arange(5, 5, 1) [start==stop] should return NULL\n");
        tofu_tensor_free_data_too(t1);
    }
    total++;

    /* Test 3.2: arange(0, 0, 1) -> NULL (start == stop) */
    tofu_tensor* t2 = tofu_tensor_arange(0, 0, 1, TOFU_FLOAT);
    if (t2 == NULL) {
        printf("  ✓ arange(0, 0, 1) [start==stop] returns NULL\n");
        passed++;
    } else {
        printf("  ✗ arange(0, 0, 1) [start==stop] should return NULL\n");
        tofu_tensor_free_data_too(t2);
    }
    total++;

    /* Test 3.3: arange(5, 5, -1) -> NULL (start == stop with negative step) */
    tofu_tensor* t3 = tofu_tensor_arange(5, 5, -1, TOFU_FLOAT);
    if (t3 == NULL) {
        printf("  ✓ arange(5, 5, -1) [start==stop, neg step] returns NULL\n");
        passed++;
    } else {
        printf("  ✗ arange(5, 5, -1) [start==stop, neg step] should return NULL\n");
        tofu_tensor_free_data_too(t3);
    }
    total++;

    printf("\nEmpty Arrays: %d/%d tests passed\n", passed, total);
    return (passed == total) ? TEST_PASS : TEST_FAIL;
}

/**
 * Test 4: Incompatible step direction (returns NULL)
 */
int test_incompatible_step() {
    printf("\nTest 4: Incompatible Step Direction (NULL return)\n");
    printf("--------------------------------------------------\n");
    int passed = 0, total = 0;

    /* Test 4.1: arange(0, 10, -1) -> NULL (positive range, negative step) */
    tofu_tensor* t1 = tofu_tensor_arange(0, 10, -1, TOFU_FLOAT);
    if (t1 == NULL) {
        printf("  ✓ arange(0, 10, -1) [incompatible] returns NULL\n");
        passed++;
    } else {
        printf("  ✗ arange(0, 10, -1) [incompatible] should return NULL\n");
        tofu_tensor_free_data_too(t1);
    }
    total++;

    /* Test 4.2: arange(10, 0, 1) -> NULL (negative range, positive step) */
    tofu_tensor* t2 = tofu_tensor_arange(10, 0, 1, TOFU_FLOAT);
    if (t2 == NULL) {
        printf("  ✓ arange(10, 0, 1) [incompatible] returns NULL\n");
        passed++;
    } else {
        printf("  ✗ arange(10, 0, 1) [incompatible] should return NULL\n");
        tofu_tensor_free_data_too(t2);
    }
    total++;

    /* Test 4.3: arange(5, 2, 0.5) -> NULL (going backward with positive step) */
    tofu_tensor* t3 = tofu_tensor_arange(5, 2, 0.5, TOFU_FLOAT);
    if (t3 == NULL) {
        printf("  ✓ arange(5, 2, 0.5) [incompatible] returns NULL\n");
        passed++;
    } else {
        printf("  ✗ arange(5, 2, 0.5) [incompatible] should return NULL\n");
        tofu_tensor_free_data_too(t3);
    }
    total++;

    printf("\nIncompatible Step: %d/%d tests passed\n", passed, total);
    return (passed == total) ? TEST_PASS : TEST_FAIL;
}

/**
 * Test 5: In-place rearange function
 */
int test_rearange() {
    printf("\nTest 5: In-place Rearange\n");
    printf("--------------------------\n");
    int passed = 0, total = 0;

    /* Test 5.1: Forward rearange */
    int dims1[] = {5};
    tofu_tensor* t1 = tofu_tensor_zeros(1, dims1, TOFU_FLOAT);
    tofu_tensor_rearange(t1, 0, 5, 1);
    float expected1[] = {0, 1, 2, 3, 4};
    passed += (verify_tensor_values(t1, expected1, 5, "rearange forward") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t1);

    /* Test 5.2: Reverse rearange */
    int dims2[] = {6};
    tofu_tensor* t2 = tofu_tensor_zeros(1, dims2, TOFU_FLOAT);
    tofu_tensor_rearange(t2, 5, -1, -1);
    float expected2[] = {5, 4, 3, 2, 1, 0};
    passed += (verify_tensor_values(t2, expected2, 6, "rearange reverse") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t2);

    /* Test 5.3: Step size test */
    int dims3[] = {5};
    tofu_tensor* t3 = tofu_tensor_zeros(1, dims3, TOFU_FLOAT);
    tofu_tensor_rearange(t3, 1, 11, 2);
    float expected3[] = {1, 3, 5, 7, 9};
    passed += (verify_tensor_values(t3, expected3, 5, "rearange with step=2") == TEST_PASS);
    total++;
    tofu_tensor_free_data_too(t3);

    printf("\nRearange: %d/%d tests passed\n", passed, total);
    return (passed == total) ? TEST_PASS : TEST_FAIL;
}

/**
 * Test 6: Different data types
 */
int test_different_dtypes() {
    printf("\nTest 6: Different Data Types\n");
    printf("-----------------------------\n");
    int passed = 0, total = 0;

    /* Test 6.1: INT32 forward */
    tofu_tensor* t1 = tofu_tensor_arange(0, 5, 1, TOFU_INT32);
    if (t1 && t1->len == 5) {
        int32_t* data1 = (int32_t*)t1->data;
        int match = 1;
        for (int i = 0; i < 5; i++) {
            if (data1[i] != i) match = 0;
        }
        if (match) {
            printf("  ✓ arange(0, 5, 1) INT32\n");
            passed++;
        } else {
            printf("  ✗ arange(0, 5, 1) INT32: incorrect values\n");
        }
        total++;
        tofu_tensor_free_data_too(t1);
    } else {
        printf("  ✗ arange(0, 5, 1) INT32: creation failed\n");
        total++;
    }

    /* Test 6.2: INT32 reverse */
    tofu_tensor* t2 = tofu_tensor_arange(5, 0, -1, TOFU_INT32);
    if (t2 && t2->len == 5) {
        int32_t* data2 = (int32_t*)t2->data;
        int match = 1;
        int32_t expected[] = {5, 4, 3, 2, 1};
        for (int i = 0; i < 5; i++) {
            if (data2[i] != expected[i]) match = 0;
        }
        if (match) {
            printf("  ✓ arange(5, 0, -1) INT32\n");
            passed++;
        } else {
            printf("  ✗ arange(5, 0, -1) INT32: incorrect values\n");
        }
        total++;
        tofu_tensor_free_data_too(t2);
    } else {
        printf("  ✗ arange(5, 0, -1) INT32: creation failed\n");
        total++;
    }

    printf("\nData Types: %d/%d tests passed\n", passed, total);
    return (passed == total) ? TEST_PASS : TEST_FAIL;
}

int main() {
    int tests_passed = 0;
    int total_tests = 6;

    printf("============================================================\n");
    printf("Tofu Enhanced Arange Test Suite\n");
    printf("============================================================\n");
    printf("Testing comprehensive arange functionality including:\n");
    printf("- Forward slicing (positive step)\n");
    printf("- Reverse slicing (negative step)\n");
    printf("- Empty arrays (start == stop)\n");
    printf("- Incompatible step direction\n");
    printf("- In-place rearange\n");
    printf("- Multiple data types\n");
    printf("============================================================\n");

    tests_passed += (test_forward_slicing() == TEST_PASS);
    tests_passed += (test_reverse_slicing() == TEST_PASS);
    tests_passed += (test_empty_arrays() == TEST_PASS);
    tests_passed += (test_incompatible_step() == TEST_PASS);
    tests_passed += (test_rearange() == TEST_PASS);
    tests_passed += (test_different_dtypes() == TEST_PASS);

    printf("\n============================================================\n");
    printf("Test Summary\n");
    printf("============================================================\n");
    printf("Total:   %d test categories\n", total_tests);
    printf("Passed:  %d test categories\n", tests_passed);
    printf("Failed:  %d test categories\n", total_tests - tests_passed);
    printf("============================================================\n");

    return (tests_passed == total_tests) ? 0 : 1;
}
