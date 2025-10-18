#!/usr/bin/env python3
"""
Validate comprehensive inner product test cases against NumPy
"""
import numpy as np

print("="*80)
print("COMPREHENSIVE INNER PRODUCT VALIDATION")
print("="*80)

all_pass = True

# Test case 1: 1-D vectors
print("\nTest 1: 1-D vectors [3] x [3] -> scalar")
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([4, 5, 6], dtype=np.int32)
result = np.inner(a, b)
expected = 32
print(f"  np.inner({a}, {b}) = {result}")
print(f"  Expected: {expected}")
print(f"  Match: {result == expected} {'✓' if result == expected else '✗'}")
all_pass = all_pass and (result == expected)

# Test case 2: 2-D same shape
print("\nTest 2: 2-D same shape [2,3] x [2,3] -> [2,2]")
a = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int8)
b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
result = np.inner(a, b)
expected = np.array([[9, 21], [17, 41]], dtype=np.int8)
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test case 3: 2-D different shapes
print("\nTest 3: 2-D different shapes [2,4] x [3,4] -> [2,3]")
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int16)
b = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int16)
result = np.inner(a, b)
expected = np.array([[10, 20, 30], [26, 52, 78]], dtype=np.int16)
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test case 4: 3-D arrays
print("\nTest 4: 3-D arrays [2,2,3] x [2,2,3] -> [2,2,2,2]")
a = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.float32)
b = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]], dtype=np.float32)
result = np.inner(a, b)
print(f"  Result shape: {result.shape}")
print(f"  Expected shape: (2, 2, 2, 2)")
print(f"  result[0,0,0,0] = {result[0,0,0,0]} (expected 6.0)")
print(f"  result[0,0,1,0] = {result[0,0,1,0]} (expected 6.0)")
shape_match = result.shape == (2, 2, 2, 2)
val_match = (result[0,0,0,0] == 6.0 and result[0,0,1,0] == 6.0)
match = shape_match and val_match
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test case 5: Mixed dimensions
print("\nTest 5: Mixed dimensions [2,3,4] x [5,4] -> [2,3,5]")
a = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
b = np.ones((5, 4), dtype=np.float32)
result = np.inner(a, b)
print(f"  Result shape: {result.shape}")
print(f"  Expected shape: (2, 3, 5)")
print(f"  result[0,0,0] = {result[0,0,0]} (expected 10.0)")
shape_match = result.shape == (2, 3, 5)
val_match = result[0,0,0] == 10.0
match = shape_match and val_match
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

# Test case 6: Edge case - single element last dimension
print("\nTest 6: Single element last dimension [2,1] x [3,1] -> [2,3]")
a = np.array([[5], [10]], dtype=np.int32)
b = np.array([[2], [3], [4]], dtype=np.int32)
result = np.inner(a, b)
expected = np.array([[10, 15, 20], [20, 30, 40]], dtype=np.int32)
print(f"  Result shape: {result.shape}")
print(f"  Result:\n{result}")
print(f"  Expected:\n{expected}")
match = np.array_equal(result, expected)
print(f"  Match: {match} {'✓' if match else '✗'}")
all_pass = all_pass and match

print("\n" + "="*80)
print(f"OVERALL: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
print("="*80)

exit(0 if all_pass else 1)
