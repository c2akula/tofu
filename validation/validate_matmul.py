#!/usr/bin/env python3
"""
Validate NumPy matmul behavior to establish ground truth for C implementation
"""
import numpy as np

print("="*80)
print("NumPy matmul() Validation - Ground Truth for C Implementation")
print("="*80)

all_pass = True

# Test 1: 1-D arrays (vector dot product)
print("\nTest 1: 1-D arrays [3] @ [3] -> scalar")
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([4, 5, 6], dtype=np.int32)
result = np.matmul(a, b)
expected = 32
print(f"  a: {a}")
print(f"  b: {b}")
print(f"  np.matmul(a, b) = {result}")
print(f"  Expected: {expected}")
match = (result == expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 2: 2-D arrays (matrix multiplication)
print("\nTest 2: 2-D arrays [2,3] @ [3,2] -> [2,2]")
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
b = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32)
result = np.matmul(a, b)
expected = np.array([[14, 14], [32, 32]], dtype=np.int32)
print(f"  a.shape: {a.shape}")
print(f"  b.shape: {b.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 3: 2-D @ 1-D (matrix-vector multiplication)
print("\nTest 3: 2-D @ 1-D: [2,3] @ [3] -> [2]")
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
b = np.array([1, 2, 3], dtype=np.int32)
result = np.matmul(a, b)
expected = np.array([14, 32], dtype=np.int32)
print(f"  a.shape: {a.shape}")
print(f"  b.shape: {b.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Result: {result}")
print(f"  Expected: {expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 4: 1-D @ 2-D (vector-matrix multiplication)
print("\nTest 4: 1-D @ 2-D: [3] @ [3,2] -> [2]")
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32)
result = np.matmul(a, b)
expected = np.array([14, 14], dtype=np.int32)
print(f"  a.shape: {a.shape}")
print(f"  b.shape: {b.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Result: {result}")
print(f"  Expected: {expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 5: 3-D arrays (batch matrix multiplication with same batch dims)
print("\nTest 5: 3-D arrays [2,3,4] @ [2,4,5] -> [2,3,5]")
a = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
b = np.ones((2, 4, 5), dtype=np.float32)
result = np.matmul(a, b)
print(f"  a.shape: {a.shape}")
print(f"  b.shape: {b.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Expected shape: (2, 3, 5)")
shape_match = result.shape == (2, 3, 5)
# First batch: sum of a[0,0,:] = 1+2+3+4 = 10
expected_val = 10.0
val_match = result[0, 0, 0] == expected_val
print(f"  result[0,0,0] = {result[0,0,0]} (expected {expected_val})")
match = shape_match and val_match
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 6: 3-D with broadcasting [3,4] @ [2,4,5] -> [2,3,5]
print("\nTest 6: Broadcasting [3,4] @ [2,4,5] -> [2,3,5]")
a = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
b = np.ones((2, 4, 5), dtype=np.float32)
result = np.matmul(a, b)
print(f"  a.shape: {a.shape}")
print(f"  b.shape: {b.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Expected shape: (2, 3, 5)")
shape_match = result.shape == (2, 3, 5)
# a[0,:] = [1,2,3,4], sum = 10
expected_val = 10.0
val_match = result[0, 0, 0] == expected_val and result[1, 0, 0] == expected_val
print(f"  result[0,0,0] = {result[0,0,0]} (expected {expected_val})")
print(f"  result[1,0,0] = {result[1,0,0]} (expected {expected_val})")
match = shape_match and val_match
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 7: 4-D arrays (multiple batch dimensions)
print("\nTest 7: 4-D arrays [2,3,5,4] @ [2,3,4,6] -> [2,3,5,6]")
a = np.ones((2, 3, 5, 4), dtype=np.float32) * 2
b = np.ones((2, 3, 4, 6), dtype=np.float32) * 3
result = np.matmul(a, b)
print(f"  a.shape: {a.shape}")
print(f"  b.shape: {b.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Expected shape: (2, 3, 5, 6)")
shape_match = result.shape == (2, 3, 5, 6)
# Each element: sum of 4 products of 2*3 = 24
expected_val = 24.0
val_match = result[0, 0, 0, 0] == expected_val
print(f"  result[0,0,0,0] = {result[0,0,0,0]} (expected {expected_val})")
match = shape_match and val_match
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 8: Broadcasting with 1 in batch dimension [1,3,4] @ [2,4,5] -> [2,3,5]
print("\nTest 8: Broadcasting [1,3,4] @ [2,4,5] -> [2,3,5]")
a = np.arange(1, 13, dtype=np.float32).reshape(1, 3, 4)
b = np.ones((2, 4, 5), dtype=np.float32)
result = np.matmul(a, b)
print(f"  a.shape: {a.shape}")
print(f"  b.shape: {b.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Expected shape: (2, 3, 5)")
shape_match = result.shape == (2, 3, 5)
# Same calculation broadcast to both batches
expected_val = 10.0
val_match = result[0, 0, 0] == expected_val and result[1, 0, 0] == expected_val
print(f"  result[0,0,0] = {result[0,0,0]} (expected {expected_val})")
print(f"  result[1,0,0] = {result[1,0,0]} (expected {expected_val})")
match = shape_match and val_match
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test 9: Edge case - single element matrices [2,1,1] @ [2,1,1] -> [2,1,1]
print("\nTest 9: Edge case [2,1,1] @ [2,1,1] -> [2,1,1]")
a = np.array([[[5]], [[10]]], dtype=np.int32)
b = np.array([[[2]], [[3]]], dtype=np.int32)
result = np.matmul(a, b)
expected = np.array([[[10]], [[30]]], dtype=np.int32)
print(f"  a.shape: {a.shape}")
print(f"  b.shape: {b.shape}")
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

print("\n" + "="*80)
print("KEY OBSERVATIONS FOR C IMPLEMENTATION")
print("="*80)
print("""
1. 1-D @ 1-D: Simple dot product (sum of element-wise products)
   Output: scalar

2. 2-D @ 2-D: Standard matrix multiplication
   [m,n] @ [n,p] -> [m,p]
   result[i,j] = sum(a[i,:] * b[:,j])

3. N-D @ 1-D: Matrix-vector for last 2 dims, preserve batch dims
   [...,m,n] @ [n] -> [...,m]
   1-D vector is treated as column vector, result dimension is removed

4. 1-D @ N-D: Vector-matrix for last 2 dims, preserve batch dims
   [n] @ [...,n,p] -> [...,p]
   1-D vector is treated as row vector, result dimension is removed

5. N-D @ N-D: Batch matrix multiplication with broadcasting
   [...,m,n] @ [...,n,p] -> [...,m,p]
   - Batch dimensions [...] are broadcast using standard rules
   - Last 2 dimensions perform matrix multiplication
   - Batch indices are SHARED across a and b (not cartesian product!)

6. Broadcasting rules:
   - Dimensions are aligned from the right
   - Size 1 dimensions are stretched
   - Missing dimensions are treated as size 1
   - All batch dimensions (all except last 2) follow broadcasting

7. Contraction: Always last of a with second-to-last of b
   These dimensions MUST match exactly (no broadcasting on contracted dim)
""")

print("\n" + "="*80)
print(f"OVERALL: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
print("="*80)

exit(0 if all_pass else 1)
