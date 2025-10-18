#!/usr/bin/env python3
"""
Compare numpy.dot() vs numpy.matmul() to understand the crucial differences
"""
import numpy as np

print("="*80)
print("NumPy dot() vs matmul() - The Key Differences")
print("="*80)

# Case 1: 1-D arrays (all three are identical)
print("\n1. 1-D arrays: [3] × [3]")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
inner_result = np.inner(a, b)
dot_result = np.dot(a, b)
matmul_result = np.matmul(a, b)
print(f"   np.inner(a, b)  = {inner_result}")
print(f"   np.dot(a, b)    = {dot_result}")
print(f"   np.matmul(a, b) = {matmul_result}")
print(f"   All identical? {inner_result == dot_result == matmul_result} ✓")

# Case 2: 2-D arrays - dot and matmul are identical (matrix multiplication)
print("\n2. 2-D arrays: [2,3] × [3,2] (matrix multiplication)")
a = np.array([[1, 2, 3], [4, 5, 6]])  # [2,3]
b = np.array([[1, 1], [2, 2], [3, 3]])  # [3,2]
# For inner, we need matching last dimension
b_inner = np.array([[1, 2, 3], [4, 5, 6]])  # [2,3]
inner_result = np.inner(a, b_inner)
dot_result = np.dot(a, b)
matmul_result = np.matmul(a, b)
print(f"   a.shape: {a.shape}, b.shape: {b.shape}")
print(f"   np.inner(a, b).shape  = {inner_result.shape}  (sum over LAST of both)")
print(f"   np.dot(a, b).shape    = {dot_result.shape}  (matrix multiply)")
print(f"   np.matmul(a, b).shape = {matmul_result.shape}  (matrix multiply)")
print(f"   dot == matmul? {np.array_equal(dot_result, matmul_result)} ✓")
print(f"   dot == inner?  {np.array_equal(dot_result, inner_result)} ✗")

# Case 3: N-D arrays - THIS IS WHERE dot() AND matmul() DIFFER!
print("\n3. N-D arrays: [2,3,4] × [2,4,5] - THE KEY DIFFERENCE!")
a = np.ones((2, 3, 4))
b = np.ones((2, 4, 5))
print(f"   a.shape: {a.shape}")
print(f"   b.shape: {b.shape}")

# For inner, both need matching last dimension
b_inner = np.ones((2, 4, 4))  # Changed last dim to 4
inner_result = np.inner(a, b_inner)
print(f"\n   np.inner(a, b).shape = {inner_result.shape}")
print(f"   Formula: (*a.shape[:-1], *b.shape[:-1]) = (2,3) + (2,4) = (2,3,2,4)")
print(f"   Contracts: LAST axis of both")

dot_result = np.dot(a, b)
print(f"\n   np.dot(a, b).shape = {dot_result.shape}")
print(f"   Formula: (*a.shape[:-1], *b.shape[:-2], b.shape[-1]) = (2,3) + (2,) + (5,)")
print(f"   Contracts: LAST of a with SECOND-TO-LAST of b")
print(f"   Result: dot(a,b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])")

matmul_result = np.matmul(a, b)
print(f"\n   np.matmul(a, b).shape = {matmul_result.shape}")
print(f"   Formula: Broadcast batch dims + matrix multiply last 2")
print(f"   Contracts: LAST of a with SECOND-TO-LAST of b (like dot)")
print(f"   BUT: Batch dimensions are BROADCAST, not cartesian product!")
print(f"   Result: matmul(a,b)[i,j,k] = sum(a[i,j,:] * b[i,:,k])")
print(f"              ^^ Note: i index is SHARED (broadcast)")

# Case 4: The smoking gun - shapes that show the difference clearly
print("\n4. The smoking gun: [9,5,7,4] × [9,5,4,3]")
a = np.ones((9, 5, 7, 4))
b = np.ones((9, 5, 4, 3))
print(f"   a.shape: {a.shape}")
print(f"   b.shape: {b.shape}")

try:
    inner_result = np.inner(a, b)
    print(f"\n   np.inner(a, b).shape = {inner_result.shape}")
    print(f"   Concatenates all non-last dims: (9,5,7) + (9,5) = (9,5,7,9,5)")
except Exception as e:
    print(f"   np.inner(a, b) error: {e}")

dot_result = np.dot(a, b)
print(f"\n   np.dot(a, b).shape = {dot_result.shape}")
print(f"   Cartesian product: (9,5,7) + (9,5) + (3,) = (9,5,7,9,5,3)")
print(f"   Every combination of batch indices!")

matmul_result = np.matmul(a, b)
print(f"\n   np.matmul(a, b).shape = {matmul_result.shape}")
print(f"   Broadcast batch dims: broadcast((9,5), (9,5)) + matmul = (9,5,7,3)")
print(f"   Batch indices are SHARED, not cartesian product!")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1-D arrays:
  inner(a,b) = dot(a,b) = matmul(a,b) = scalar inner product

2-D arrays:
  inner(a,b)[i,j]  = sum(a[i,:] * b[j,:])     ← DIFFERENT (sum over last of both)
  dot(a,b)[i,j]    = sum(a[i,:] * b[:,j])     ← matrix multiply
  matmul(a,b)[i,j] = sum(a[i,:] * b[:,j])     ← matrix multiply

  dot == matmul ✓, but both ≠ inner

N-D arrays (e.g., [2,3,4] × [2,4,5]):
  inner(a,b).shape  = (2,3,2,4)     ← cartesian product, sum over last of both
  dot(a,b).shape    = (2,3,2,5)     ← cartesian product, sum over last of a & 2nd-last of b
  matmul(a,b).shape = (2,3,5)       ← BROADCAST batch dims, matrix multiply last 2

  ALL THREE ARE DIFFERENT!

The CRITICAL difference between dot() and matmul():
  - dot():    Creates CARTESIAN PRODUCT of batch dimensions
  - matmul(): BROADCASTS batch dimensions (treats arrays as stacks of matrices)

For [9,5,7,4] × [9,5,4,3]:
  - dot():    (9,5,7,9,5,3) - 9*5*7*9*5*3 = 56,862,750 elements!
  - matmul(): (9,5,7,3)     - 9*5*7*3     = 945 elements (60,000× smaller!)

matmul() is more memory-efficient for batch operations!
""")

# Case 5: When does dot() == matmul()?
print("\n5. When are dot() and matmul() identical?")
print("\n   a) 2-D arrays (matrix multiplication):")
a = np.ones((3, 4))
b = np.ones((4, 5))
print(f"      [{a.shape}] × [{b.shape}] → {np.dot(a, b).shape} ✓")

print("\n   b) N-D × 1-D:")
a = np.ones((2, 3, 4))
b = np.ones(4)
print(f"      [{a.shape}] × [{b.shape}] → {np.dot(a, b).shape} ✓")

print("\n   c) When batch dimensions already match exactly:")
a = np.ones((5, 3, 4))
b = np.ones((5, 4, 2))
print(f"      [{a.shape}] × [{b.shape}]:")
print(f"      dot():    {np.dot(a, b).shape} (cartesian + shared first dim)")
print(f"      matmul(): {np.matmul(a, b).shape} (broadcast batch + matmul)")
print(f"      STILL DIFFERENT! ✗")

print("\n" + "="*80)
print("IMPLEMENTATION RECOMMENDATION")
print("="*80)
print("""
Given that:
1. inner() uses cartesian product + sum over last of both
2. dot() uses cartesian product + sum over last of a & 2nd-last of b
3. matmul() uses broadcasting + matrix multiply on last 2 dims

You should implement:
✅ tl_tensor_inner()  - DONE (cartesian product semantics)
⏳ tl_tensor_matmul() - RECOMMENDED NEXT (broadcast semantics, more useful)
⏳ tl_tensor_dot()    - LATER (less useful due to cartesian explosion)

matmul() is more practical for deep learning / batch operations!
""")
