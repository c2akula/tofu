#!/usr/bin/env python3
"""
NumPy matrix multiplication benchmark for comparison
"""
import numpy as np
import time

def benchmark_matmul(m, n, p, iterations):
    """Benchmark matmul [m,n] @ [n,p]"""
    np.random.seed(42)
    A = np.random.rand(m, n).astype(np.float32)
    B = np.random.rand(n, p).astype(np.float32)

    # Warmup
    _ = A @ B

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = A @ B
    end = time.perf_counter()

    return (end - start) / iterations

def main():
    print("=" * 60)
    print("NumPy Matrix Multiplication Benchmark")
    print("=" * 60)
    print()

    tests = [
        (10, 10, 10, 10000),
        (50, 50, 50, 1000),
        (100, 100, 100, 100),
        (200, 200, 200, 10),
        (500, 500, 500, 1),
    ]

    print(f"{'Size':<15} {'Iterations':<15} {'Avg Time (ms)':<15} {'GFLOP/s':<15} {'Throughput':<15}")
    print("-" * 75)

    for m, n, p, iters in tests:
        avg_time = benchmark_matmul(m, n, p, iters)

        # Calculate GFLOP/s
        flops = 2.0 * m * n * p
        gflops = (flops / avg_time) / 1e9

        # Calculate throughput
        throughput = 1.0 / avg_time

        print(f"[{m:3},{n:3},{p:3}] {iters:<15} {avg_time*1000:<15.6f} {gflops:<15.2f} {throughput:<15.0f}")

    print()
    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
