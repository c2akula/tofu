#!/usr/bin/env python3
"""
Validate NumPy outer product behavior to establish ground truth for C implementation
"""
import numpy as np

print("="*80)
print("NumPy outer() Validation - Ground Truth for C Implementation")
print("="*80)

all_pass = True

# Test 1: 1-D @ 1-D (basic outer product)
print("\nTest 1: 1-D @ 1-D: [3] outer [4] -> [3,4]")
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([4, 5, 6, 7], dtype=np.int32)
result = np.outer(a, b)
expected = np.array([[4, 5, 6, 7],
                     [8, 10, 12, 14],
                     [12, 15, 18, 21]], dtype=np.int32)
print(f"  a: {a}")
print(f"  b: {b}")
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 2: Scalar-like inputs
print("\nTest 2: Scalar outer product: [1] outer [1] -> [1,1]")
a = np.array([5], dtype=np.int32)
b = np.array([3], dtype=np.int32)
result = np.outer(a, b)
expected = np.array([[15]], dtype=np.int32)
print(f"  Result shape: {result.shape}")
print(f"  Result: {result}")
print(f"  Expected: {expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 3: Different sizes
print("\nTest 3: Different sizes: [2] outer [5] -> [2,5]")
a = np.array([1, 2], dtype=np.float32)
b = np.array([1, 2, 3, 4, 5], dtype=np.float32)
result = np.outer(a, b)
expected = np.array([[1, 2, 3, 4, 5],
                     [2, 4, 6, 8, 10]], dtype=np.float32)
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 4: Multi-dimensional inputs (flattened behavior)
print("\nTest 4: Multi-dimensional inputs [2,2] outer [2,2] -> [4,4]")
a = np.array([[1, 2], [3, 4]], dtype=np.int32)
b = np.array([[1, 1], [2, 2]], dtype=np.int32)
result = np.outer(a, b)
print(f"  a.shape: {a.shape} (input is 2-D)")
print(f"  b.shape: {b.shape} (input is 2-D)")
print(f"  Result shape: {result.shape}")
print(f"  Note: np.outer() FLATTENS inputs first!")
print(f"  a flattened: {a.flatten()}")
print(f"  b flattened: {b.flatten()}")
# outer([1,2,3,4], [1,1,2,2])
expected = np.outer(a.flatten(), b.flatten())
print(f"  Result:\n{result}")
match = (result.shape == (4, 4))
print(f"  Shape correct: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 5: Negative numbers
print("\nTest 5: Negative numbers: [3] outer [3] -> [3,3]")
a = np.array([-1, 0, 1], dtype=np.int32)
b = np.array([1, 2, 3], dtype=np.int32)
result = np.outer(a, b)
expected = np.array([[-1, -2, -3],
                     [0, 0, 0],
                     [1, 2, 3]], dtype=np.int32)
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 6: Floating point
print("\nTest 6: Floating point: [3] outer [2] -> [3,2]")
a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
b = np.array([2.0, 4.0], dtype=np.float32)
result = np.outer(a, b)
expected = np.array([[3.0, 6.0],
                     [5.0, 10.0],
                     [7.0, 14.0]], dtype=np.float32)
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.allclose(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 7: Pre-allocated output (verify formula)
print("\nTest 7: Verify outer product formula: out[i,j] = a[i] * b[j]")
a = np.array([10, 20, 30], dtype=np.int32)
b = np.array([1, 2], dtype=np.int32)
result = np.outer(a, b)
print(f"  a: {a}")
print(f"  b: {b}")
print(f"  result[0,0] = a[0] * b[0] = {a[0]} * {b[0]} = {result[0,0]}")
print(f"  result[1,1] = a[1] * b[1] = {a[1]} * {b[1]} = {result[1,1]}")
print(f"  result[2,0] = a[2] * b[0] = {a[2]} * {b[0]} = {result[2,0]}")
match = (result[0,0] == 10 and result[1,1] == 40 and result[2,0] == 30)
print(f"  Formula verified: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

print("\n" + "="*80)
print("KEY OBSERVATIONS FOR C IMPLEMENTATION")
print("="*80)
print("""
1. Outer product formula: out[i,j] = a[i] * b[j]
   - Simple element-wise multiplication in cartesian product
   - No summation (unlike inner/matmul)

2. Always produces 2-D output: [m] outer [n] -> [m,n]
   - Even for scalar inputs: [1] outer [1] -> [1,1]

3. Multi-dimensional inputs are FLATTENED first:
   - [2,2] outer [3] -> flatten to [4] outer [3] -> [4,3]
   - This is different from inner/matmul which preserve structure

4. Works with all numeric types: int8, int32, float32, etc.

5. Output shape: (a.size, b.size) where size is total element count
   - a.size = product of all dimensions
   - b.size = product of all dimensions

6. No broadcasting involved (unlike matmul)
   - Simple nested loop: for i in a, for j in b: out[i,j] = a[i] * b[j]

7. Memory order: row-major (C-style)
   - out[i,j] is at index i*n + j where n = b.size
""")

print("\n" + "="*80)
print(f"OVERALL: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
print("="*80)

exit(0 if all_pass else 1)
