/*
 * Matrix multiplication benchmark
 * Compares performance for various matrix sizes
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tl_tensor.h"

double benchmark_matmul(int m, int n, int p, int iterations) {
    /* Create random matrices [m,n] @ [n,p] */
    float* a_data = (float*)malloc(m * n * sizeof(float));
    float* b_data = (float*)malloc(n * p * sizeof(float));

    for (int i = 0; i < m * n; i++) a_data[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < n * p; i++) b_data[i] = (float)rand() / RAND_MAX;

    tl_tensor* A = tl_tensor_create(a_data, 2, (int[]){m, n}, TL_FLOAT);
    tl_tensor* B = tl_tensor_create(b_data, 2, (int[]){n, p}, TL_FLOAT);

    /* Warmup */
    tl_tensor* warmup = tl_tensor_matmul(A, B, NULL);
    tl_tensor_free_data_too(warmup);

    /* Benchmark */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < iterations; i++) {
        tl_tensor* result = tl_tensor_matmul(A, B, NULL);
        tl_tensor_free_data_too(result);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    tl_tensor_free(A);
    tl_tensor_free(B);
    free(a_data);
    free(b_data);

    return elapsed / iterations;  /* Average time per operation */
}

int main() {
    printf("============================================================\n");
    printf("Matrix Multiplication Benchmark\n");
    printf("============================================================\n\n");

    srand(42);

    /* Test various sizes */
    struct {
        int m, n, p;
        int iterations;
    } tests[] = {
        {10, 10, 10, 10000},      /* Small */
        {50, 50, 50, 1000},       /* Medium-small */
        {100, 100, 100, 100},     /* Medium */
        {200, 200, 200, 10},      /* Large */
        {500, 500, 500, 1},       /* Very large */
    };

    printf("%-15s %-15s %-15s %-15s %-15s\n",
           "Size", "Iterations", "Avg Time (ms)", "GFLOP/s", "Throughput");
    printf("------------------------------------------------------------\n");

    for (int i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        int m = tests[i].m;
        int n = tests[i].n;
        int p = tests[i].p;
        int iters = tests[i].iterations;

        double avg_time = benchmark_matmul(m, n, p, iters);

        /* Calculate GFLOP/s: matmul does 2*m*n*p operations */
        double flops = 2.0 * m * n * p;
        double gflops = (flops / avg_time) / 1e9;

        /* Calculate throughput (matrices per second) */
        double throughput = 1.0 / avg_time;

        printf("[%3d,%3d,%3d] %-15d %-15.6f %-15.2f %-15.0f\n",
               m, n, p, iters, avg_time * 1000, gflops, throughput);
    }

    printf("\n============================================================\n");
    printf("Benchmark complete!\n");
    printf("============================================================\n");

    return 0;
}
