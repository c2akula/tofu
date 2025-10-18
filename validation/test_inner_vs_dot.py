#!/usr/bin/env python3
"""
Compare numpy.inner() vs numpy.dot() to understand the differences
"""
import numpy as np

print("="*80)
print("NumPy inner() vs dot() Comparison")
print("="*80)

# Case 1: 1-D vectors
print("\n1. 1-D vectors: [1,2,3] with [4,5,6]")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
inner_result = np.inner(a, b)
dot_result = np.dot(a, b)
print(f"   np.inner(a, b) = {inner_result}")
print(f"   np.dot(a, b)   = {dot_result}")
print(f"   Same result? {inner_result == dot_result} ✓")

# Case 2: 2-D arrays (THIS IS WHERE THEY DIFFER!)
print("\n2. 2-D arrays: [2,3] with [2,3]")
a = np.array([[1, 2, 3], [4, 5, 6]])  # shape [2, 3]
b = np.array([[1, 1, 1], [2, 2, 2]])  # shape [2, 3]
print(f"   a.shape: {a.shape}")
print(f"   b.shape: {b.shape}")
print(f"\n   a:\n{a}")
print(f"   b:\n{b}")

inner_result = np.inner(a, b)
print(f"\n   np.inner(a, b).shape = {inner_result.shape}")
print(f"   np.inner(a, b) =\n{inner_result}")
print(f"   Explanation: Sum over LAST axis of both")
print(f"   result[i,j] = sum(a[i,:] * b[j,:])")

try:
    dot_result = np.dot(a, b)
    print(f"\n   np.dot(a, b).shape = {dot_result.shape}")
    print(f"   np.dot(a, b) =\n{dot_result}")
except ValueError as e:
    print(f"\n   np.dot(a, b) = ERROR: {e}")
    print(f"   For 2-D: dot does MATRIX MULTIPLICATION")
    print(f"   Requires: a.shape[1] == b.shape[0]")
    print(f"   Here: a.shape[1]=3, b.shape[0]=2 ✗")

# Case 2b: Valid matrix multiplication
print("\n2b. Valid matrix multiplication: [2,3] × [3,2]")
a = np.array([[1, 2, 3], [4, 5, 6]])  # shape [2, 3]
b = np.array([[1, 1], [2, 2], [3, 3]])  # shape [3, 2]
print(f"   a.shape: {a.shape}")
print(f"   b.shape: {b.shape}")

inner_result = np.inner(a, b)
dot_result = np.dot(a, b)
print(f"\n   np.inner(a, b).shape = {inner_result.shape}")
print(f"   np.inner(a, b) =\n{inner_result}")

print(f"\n   np.dot(a, b).shape = {dot_result.shape}")
print(f"   np.dot(a, b) =\n{dot_result}")
print(f"\n   DIFFERENT results! inner ≠ dot for 2-D")

# Case 3: Higher dimensions
print("\n3. Higher dimensions: [2,3,4] with [5,4]")
a = np.ones((2, 3, 4))
b = np.ones((5, 4)) * 2
print(f"   a.shape: {a.shape}")
print(f"   b.shape: {b.shape}")

inner_result = np.inner(a, b)
print(f"\n   np.inner(a, b).shape = {inner_result.shape}")
print(f"   Formula: (*a.shape[:-1], *b.shape[:-1]) = (2,3) + (5,) = (2,3,5)")

try:
    dot_result = np.dot(a, b)
    print(f"\n   np.dot(a, b).shape = {dot_result.shape}")
    print(f"   For N-D: dot contracts LAST axis of a with SECOND-TO-LAST of b")
except ValueError as e:
    print(f"\n   np.dot(a, b) = ERROR: {e}")
    print(f"   dot requires: a.shape[-1] == b.shape[-2]")

# Case 4: Summary with clear example
print("\n" + "="*80)
print("KEY DIFFERENCES SUMMARY")
print("="*80)
print("""
1-D arrays:
  inner(a, b) = dot(a, b) = sum(a * b)  ← SAME

2-D arrays (matrices):
  inner(a, b)[i,j] = sum(a[i,:] * b[j,:])     ← sum over LAST axis of BOTH
  dot(a, b)[i,j]   = sum(a[i,:] * b[:,j])     ← matrix multiplication

  inner: [m,n] × [p,n] → [m,p]  (n must match)
  dot:   [m,n] × [n,p] → [m,p]  (n must match)

N-D arrays:
  inner: sum over LAST axis of both
         output.shape = (*a.shape[:-1], *b.shape[:-1])

  dot:   sum over LAST axis of a and SECOND-TO-LAST axis of b
         output.shape = (*a.shape[:-1], *b.shape[:-2], b.shape[-1])

RELATIONSHIP:
  - For 1-D: inner and dot are IDENTICAL (both compute inner product)
  - For 2-D: dot is matrix multiplication, inner is NOT
  - inner() is more consistent across dimensions (always contracts last axes)
  - dot() has special behavior for 2-D (matrix multiplication)
""")
