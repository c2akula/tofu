# Benchmark Results

## Test Environment
- **Platform**: macOS (arm64)
- **Compiler**: clang with -O3 optimization
- **NumPy Version**: (system default)
- **Test Date**: 2025-10-18

## Matrix Multiplication Performance

### Tofu (Pure C)
```
Size            Avg Time (ms)   GFLOP/s
[ 10, 10, 10]   0.007520        0.27
[ 50, 50, 50]   0.617276        0.41
[100,100,100]   4.808720        0.42
[200,200,200]   38.271600       0.42
[500,500,500]   604.572000      0.41
```

### NumPy (with BLAS)
```
Size            Avg Time (ms)   GFLOP/s
[ 10, 10, 10]   0.001037        1.93
[ 50, 50, 50]   0.008652        28.89
[100,100,100]   0.041240        48.50
[200,200,200]   0.095596        167.37
[500,500,500]   1.895667        131.88
```

## Performance Comparison

| Size | Tofu Time (ms) | NumPy Time (ms) | Speedup (NumPy/Tofu) |
|------|----------------|-----------------|----------------------|
| 10×10×10 | 0.0075 | 0.0010 | **7.3× faster** |
| 50×50×50 | 0.617 | 0.009 | **71× faster** |
| 100×100×100 | 4.809 | 0.041 | **117× faster** |
| 200×200×200 | 38.27 | 0.096 | **400× faster** |
| 500×500×500 | 604.6 | 1.896 | **319× faster** |

## Analysis

### Correctness ✅
- **Neural Network Inference**: Tofu output matches NumPy **exactly** (max diff: 0.00000000)
- **Operations validated**: matmul, element-wise add, ReLU
- **Conclusion**: Tofu is **numerically correct**

### Performance
- **NumPy is 7-400× faster** due to optimized BLAS libraries (Apple Accelerate on macOS)
- Tofu uses **naive O(n³) algorithm** without SIMD or loop optimizations
- Performance is **consistent** (~0.4 GFLOP/s) across sizes, showing good scalability

### When to Use Tofu

**Tofu is ideal for:**
1. ✅ **Embedded systems** (ESP32, ARM Cortex-M) where NumPy isn't available
2. ✅ **Zero dependencies** - No Python, no BLAS, just pure C
3. ✅ **Small to medium matrices** - Performance penalty is acceptable for convenience
4. ✅ **Prototyping** - Quick tensor operations without Python overhead
5. ✅ **Learning** - Understanding how tensor operations work

**NumPy/BLAS is better for:**
- ❌ Large-scale numerical computation
- ❌ Performance-critical applications
- ❌ When BLAS libraries are available

### Optimization Opportunities

To improve Tofu performance:
1. **Add BLAS integration** (optional dependency) - would match NumPy performance
2. **SIMD optimizations** (ARM NEON, x86 AVX) - 4-8× speedup
3. **Loop tiling** - Better cache utilization
4. **Parallel processing** - Multi-threading for large matrices

## Conclusion

**Tofu successfully validates as:**
- ✅ **Correct**: Matches NumPy numerically
- ✅ **Functional**: Real neural network inference works
- ✅ **Practical**: Suitable for embedded/dependency-free environments
- ⚠️ **Performance**: 7-400× slower than optimized BLAS (expected for naive implementation)

The library achieves its goal: **NumPy-compatible tensor operations in pure C with zero dependencies**.
