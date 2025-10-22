/*
 * Test to reproduce CRITICAL BUG #2: Type cast errors in fprintf functions
 *
 * Bug Location: src/tofu_type.c:256, 264
 *
 * fprintf_uint64 and fprintf_uint32 cast to uint16_t instead of their correct types
 * when using custom format strings.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

/* BUGGY VERSION - as it exists in the code */
static int fprintf_uint64_buggy(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, "%.3f", *(uint64_t *)p);  // Default format (not the bug)
    else
        return fprintf(fp, fmt, *(uint16_t *)p);  // BUG: Should be uint64_t
}

static int fprintf_uint32_buggy(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, "%.3f", *(uint32_t *)p);  // Default format (not the bug)
    else
        return fprintf(fp, fmt, *(uint16_t *)p);  // BUG: Should be uint32_t
}

/* CORRECT VERSION */
static int fprintf_uint64_correct(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, "%lu", *(uint64_t *)p);
    else
        return fprintf(fp, fmt, *(uint64_t *)p);  // CORRECT
}

static int fprintf_uint32_correct(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, "%u", *(uint32_t *)p);
    else
        return fprintf(fp, fmt, *(uint32_t *)p);  // CORRECT
}

void test_fprintf_uint64_bug()
{
    printf("=== Test 1: fprintf_uint64 with custom format ===\n\n");

    uint64_t large_val = 0x123456789ABCDEF0ULL;  // Large 64-bit value
    printf("Test value: 0x%016lX (decimal: %lu)\n\n", large_val, large_val);

    FILE *fp_buggy = fmemopen(NULL, 256, "w+");
    FILE *fp_correct = fmemopen(NULL, 256, "w+");

    printf("Buggy version (casts to uint16_t):\n");
    fprintf_uint64_buggy(fp_buggy, "%lu", &large_val);
    fflush(fp_buggy);

    /* Read what was written */
    char buggy_output[256] = {0};
    fseek(fp_buggy, 0, SEEK_SET);
    fgets(buggy_output, sizeof(buggy_output), fp_buggy);
    printf("  Output: %s", buggy_output);

    /* Buggy version reads only 16 bits */
    uint16_t truncated = (uint16_t)(large_val & 0xFFFF);
    printf("  (Read only lower 16 bits: 0x%04X = %u)\n", truncated, truncated);
    printf("  ✗ WRONG: Lost upper 48 bits of data!\n\n");

    printf("Correct version:\n");
    fprintf_uint64_correct(fp_correct, "%lu", &large_val);
    fflush(fp_correct);

    char correct_output[256] = {0};
    fseek(fp_correct, 0, SEEK_SET);
    fgets(correct_output, sizeof(correct_output), fp_correct);
    printf("  Output: %s", correct_output);
    printf("  ✓ CORRECT: Full 64-bit value preserved!\n\n");

    fclose(fp_buggy);
    fclose(fp_correct);

    /* Verify the bug exists */
    uint64_t buggy_parsed = strtoull(buggy_output, NULL, 10);
    uint64_t correct_parsed = strtoull(correct_output, NULL, 10);

    printf("Comparison:\n");
    printf("  Buggy output value:   %lu\n", buggy_parsed);
    printf("  Correct output value: %lu\n", correct_parsed);
    printf("  Original value:       %lu\n", large_val);

    assert(correct_parsed == large_val && "Correct version should preserve full value");
    assert(buggy_parsed != large_val && "Buggy version should NOT match (demonstrates bug)");

    printf("\n  ✓ Bug confirmed: Buggy version loses data!\n\n");
}

void test_fprintf_uint32_bug()
{
    printf("=== Test 2: fprintf_uint32 with custom format ===\n\n");

    uint32_t large_val = 0x12345678U;  // Large 32-bit value
    printf("Test value: 0x%08X (decimal: %u)\n\n", large_val, large_val);

    FILE *fp_buggy = fmemopen(NULL, 256, "w+");
    FILE *fp_correct = fmemopen(NULL, 256, "w+");

    printf("Buggy version (casts to uint16_t):\n");
    fprintf_uint32_buggy(fp_buggy, "%u", &large_val);
    fflush(fp_buggy);

    char buggy_output[256] = {0};
    fseek(fp_buggy, 0, SEEK_SET);
    fgets(buggy_output, sizeof(buggy_output), fp_buggy);
    printf("  Output: %s", buggy_output);

    uint16_t truncated = (uint16_t)(large_val & 0xFFFF);
    printf("  (Read only lower 16 bits: 0x%04X = %u)\n", truncated, truncated);
    printf("  ✗ WRONG: Lost upper 16 bits of data!\n\n");

    printf("Correct version:\n");
    fprintf_uint32_correct(fp_correct, "%u", &large_val);
    fflush(fp_correct);

    char correct_output[256] = {0};
    fseek(fp_correct, 0, SEEK_SET);
    fgets(correct_output, sizeof(correct_output), fp_correct);
    printf("  Output: %s", correct_output);
    printf("  ✓ CORRECT: Full 32-bit value preserved!\n\n");

    fclose(fp_buggy);
    fclose(fp_correct);

    /* Verify the bug exists */
    uint32_t buggy_parsed = strtoul(buggy_output, NULL, 10);
    uint32_t correct_parsed = strtoul(correct_output, NULL, 10);

    printf("Comparison:\n");
    printf("  Buggy output value:   %u\n", buggy_parsed);
    printf("  Correct output value: %u\n", correct_parsed);
    printf("  Original value:       %u\n", large_val);

    assert(correct_parsed == large_val && "Correct version should preserve full value");
    assert(buggy_parsed != large_val && "Buggy version should NOT match (demonstrates bug)");

    printf("\n  ✓ Bug confirmed: Buggy version loses data!\n\n");
}

void test_edge_cases()
{
    printf("=== Test 3: Edge cases ===\n\n");

    /* Test with maximum values */
    printf("Test 3a: Maximum uint64_t value\n");
    uint64_t max64 = UINT64_MAX;
    printf("  Value: %lu (0xFFFFFFFFFFFFFFFF)\n", max64);

    FILE *fp = fmemopen(NULL, 256, "w+");
    fprintf_uint64_buggy(fp, "%lu", &max64);
    fflush(fp);

    char output[256] = {0};
    fseek(fp, 0, SEEK_SET);
    fgets(output, sizeof(output), fp);
    uint64_t parsed = strtoull(output, NULL, 10);

    printf("  Buggy output: %lu\n", parsed);
    printf("  Expected (from truncation): %u\n", (uint16_t)max64);
    printf("  ✗ Data corruption: %.2f%% of bits lost!\n\n",
           (1.0 - 16.0/64.0) * 100.0);

    fclose(fp);

    printf("Test 3b: Maximum uint32_t value\n");
    uint32_t max32 = UINT32_MAX;
    printf("  Value: %u (0xFFFFFFFF)\n", max32);

    fp = fmemopen(NULL, 256, "w+");
    fprintf_uint32_buggy(fp, "%u", &max32);
    fflush(fp);

    fseek(fp, 0, SEEK_SET);
    fgets(output, sizeof(output), fp);
    uint32_t parsed32 = strtoul(output, NULL, 10);

    printf("  Buggy output: %u\n", parsed32);
    printf("  Expected (from truncation): %u\n", (uint16_t)max32);
    printf("  ✗ Data corruption: %.2f%% of bits lost!\n\n",
           (1.0 - 16.0/32.0) * 100.0);

    fclose(fp);
}

int main()
{
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  BUG REPRODUCTION TEST: fprintf Type Cast Errors          ║\n");
    printf("║  Critical Bug #2 from Code Review                         ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("Bug Description:\n");
    printf("  fprintf_uint64 and fprintf_uint32 incorrectly cast to uint16_t\n");
    printf("  when using custom format strings\n\n");

    printf("Impact:\n");
    printf("  - Data loss: 48 bits lost for uint64, 16 bits lost for uint32\n");
    printf("  - Incorrect output in tensor printing\n");
    printf("  - Silent data corruption (no compile warning)\n\n");

    test_fprintf_uint64_bug();
    test_fprintf_uint32_bug();
    test_edge_cases();

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║  CONCLUSION: Bugs confirmed!                               ║\n");
    printf("║  Both functions lose significant data when casting.        ║\n");
    printf("║  Fix: Remove incorrect (uint16_t *) casts                 ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
