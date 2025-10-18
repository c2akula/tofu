#!/usr/bin/env python3
"""
Broadcasting Mechanism Ground Truth Validator

This script validates NumPy's broadcasting behavior to establish ground truth
for testing the Tofu broadcasting implementation. It tests and documents:

1. Shape calculation (output shape from input shapes)
2. Stride calculation (how indices map through broadcasting)
3. Index mapping (which source elements map to which output elements)
4. Broadcasting properties (commutativity, associativity, etc.)
"""

import numpy as np
import json
from typing import Tuple, List, Dict, Any

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def compute_broadcast_shape(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute the output shape when broadcasting two shapes together.
    Follows NumPy's broadcasting rules.
    """
    # Pad shorter shape with 1s on the left
    max_ndim = max(len(shape1), len(shape2))
    s1 = (1,) * (max_ndim - len(shape1)) + shape1
    s2 = (1,) * (max_ndim - len(shape2)) + shape2

    # Compute output shape
    out_shape = []
    for d1, d2 in zip(s1, s2):
        if d1 == d2:
            out_shape.append(d1)
        elif d1 == 1:
            out_shape.append(d2)
        elif d2 == 1:
            out_shape.append(d1)
        else:
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")

    return tuple(out_shape)


def compute_broadcast_strides(src_shape: Tuple[int, ...],
                              out_shape: Tuple[int, ...]) -> List[int]:
    """
    Compute broadcasting strides for a tensor.

    Strides tell us how much to increment the source index when moving
    along each dimension of the output. A stride of 0 means that dimension
    is being broadcast (repeated).

    Returns: List of strides (one per output dimension)
    """
    # Align shapes from the right
    ndim_diff = len(out_shape) - len(src_shape)

    strides = []
    src_stride = 1

    # Process dimensions from right to left
    for i in range(len(out_shape) - 1, -1, -1):
        src_dim_idx = i - ndim_diff

        if src_dim_idx < 0:
            # This dimension doesn't exist in source - broadcast from implicit 1
            strides.append(0)
        elif src_shape[src_dim_idx] == 1:
            # This dimension is 1 in source - broadcast (stride = 0)
            strides.append(0)
        elif src_shape[src_dim_idx] == out_shape[i]:
            # This dimension matches - use actual stride
            strides.append(src_stride)
            src_stride *= src_shape[src_dim_idx]
        else:
            raise ValueError(f"Cannot broadcast shape {src_shape} to {out_shape}")

    # Reverse to match output dimension order
    strides.reverse()
    return strides


def compute_source_index(out_coords: Tuple[int, ...],
                         strides: List[int]) -> int:
    """
    Compute the flat source index from output coordinates and broadcast strides.

    Args:
        out_coords: Coordinates in the output tensor
        strides: Broadcasting strides

    Returns: Flat index into the source tensor
    """
    return sum(coord * stride for coord, stride in zip(out_coords, strides))


def validate_broadcasting_with_numpy(shape1: Tuple[int, ...],
                                     shape2: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Validate broadcasting behavior using NumPy and return detailed results.
    """
    # Create test arrays with distinct values for tracking
    size1 = np.prod(shape1)
    size2 = np.prod(shape2)
    arr1 = np.arange(size1, dtype=np.float32).reshape(shape1)
    arr2 = np.arange(100, 100 + size2, dtype=np.float32).reshape(shape2)

    # Perform broadcasting operation
    result = arr1 + arr2

    # Compute expected output shape
    expected_shape = compute_broadcast_shape(shape1, shape2)

    # Compute strides
    strides1 = compute_broadcast_strides(shape1, expected_shape)
    strides2 = compute_broadcast_strides(shape2, expected_shape)

    return {
        'shape1': shape1,
        'shape2': shape2,
        'output_shape': tuple(result.shape),
        'expected_shape': expected_shape,
        'shapes_match': tuple(result.shape) == expected_shape,
        'strides1': strides1,
        'strides2': strides2,
        'arr1_sample': arr1.flatten()[:5].tolist(),
        'arr2_sample': arr2.flatten()[:5].tolist(),
        'result_sample': result.flatten()[:5].tolist()
    }


def test_shape_calculation():
    """Test broadcasting shape calculation"""
    print_section("TEST 1: Broadcasting Shape Calculation")

    test_cases = [
        # (shape1, shape2, expected_output_shape)
        ((3, 3), (3, 3), (3, 3)),           # Same shape
        ((1,), (3, 3), (3, 3)),             # Scalar broadcast
        ((3,), (3, 3), (3, 3)),             # 1D to 2D broadcast
        ((3, 1), (3, 3), (3, 3)),           # Column broadcast
        ((1, 3), (3, 3), (3, 3)),           # Row broadcast
        ((2, 1, 3), (1, 4, 3), (2, 4, 3)),  # 3D broadcast
        ((5, 1, 4, 1), (3, 1, 1), (5, 3, 4, 1)),  # Complex 4D
        ((8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)),  # Another 4D
    ]

    results = []
    for shape1, shape2, expected in test_cases:
        try:
            computed = compute_broadcast_shape(shape1, shape2)
            match = computed == expected
            print(f"  {shape1} × {shape2} -> {computed}")
            print(f"    Expected: {expected}, Match: {match}")
            results.append({
                'shape1': shape1,
                'shape2': shape2,
                'expected': expected,
                'computed': computed,
                'match': match
            })
        except ValueError as e:
            print(f"  {shape1} × {shape2} -> ERROR: {e}")
            results.append({
                'shape1': shape1,
                'shape2': shape2,
                'expected': expected,
                'error': str(e)
            })

    print(f"\n  Passed: {sum(1 for r in results if r.get('match', False))}/{len(results)}")
    return results


def test_stride_calculation():
    """Test broadcasting stride calculation"""
    print_section("TEST 2: Broadcasting Stride Calculation")

    test_cases = [
        # (src_shape, out_shape, description)
        ((3, 3), (3, 3), "Same shape - no broadcasting"),
        ((1,), (3, 3), "Scalar to 2D"),
        ((3,), (3, 3), "Row vector to 2D"),
        ((3, 1), (3, 3), "Column vector to 2D"),
        ((1, 3), (3, 3), "Row broadcast"),
        ((2, 1, 4), (2, 3, 4), "Middle dimension broadcast"),
        ((1, 1, 4), (2, 3, 4), "Two dimension broadcast"),
    ]

    results = []
    for src_shape, out_shape, desc in test_cases:
        strides = compute_broadcast_strides(src_shape, out_shape)
        print(f"  {desc}")
        print(f"    Source: {src_shape}, Output: {out_shape}")
        print(f"    Strides: {strides}")

        # Verify with NumPy
        src = np.arange(np.prod(src_shape)).reshape(src_shape)
        out = np.broadcast_to(src, out_shape)

        # Check a few indices to validate strides
        valid = True
        for _ in range(min(5, np.prod(out_shape))):
            idx = tuple(np.random.randint(0, d) for d in out_shape)
            computed_src_idx = compute_source_index(idx, strides)

            # Get actual source index by checking which source element
            # maps to this output position
            expected_val = out[idx]
            src_flat = src.flatten()
            if computed_src_idx >= len(src_flat) or src_flat[computed_src_idx] != expected_val:
                valid = False
                break

        print(f"    Validation: {'PASS' if valid else 'FAIL'}\n")

        results.append({
            'src_shape': src_shape,
            'out_shape': out_shape,
            'strides': strides,
            'valid': valid
        })

    return results


def test_index_mapping():
    """Test index mapping through broadcasting"""
    print_section("TEST 3: Broadcasting Index Mapping")

    test_cases = [
        ((2, 1), (2, 3)),  # Column to matrix
        ((1, 3), (2, 3)),  # Row to matrix
        ((2, 1, 4), (2, 3, 4)),  # 3D with middle broadcast
    ]

    results = []
    for src_shape, out_shape in test_cases:
        print(f"  Mapping {src_shape} -> {out_shape}")

        src = np.arange(np.prod(src_shape)).reshape(src_shape)
        out = np.broadcast_to(src, out_shape)
        strides = compute_broadcast_strides(src_shape, out_shape)

        # Test all output positions
        all_correct = True
        sample_mappings = []

        for out_idx in range(min(10, np.prod(out_shape))):
            # Convert flat index to coordinates
            out_coords = np.unravel_index(out_idx, out_shape)

            # Compute source index using our stride calculation
            src_idx = compute_source_index(out_coords, strides)

            # Get expected value from NumPy
            expected_val = out[out_coords]
            actual_val = src.flatten()[src_idx]

            correct = (expected_val == actual_val)
            all_correct = all_correct and correct

            if len(sample_mappings) < 5:
                sample_mappings.append({
                    'out_coords': out_coords,
                    'src_idx': src_idx,
                    'value': float(expected_val),
                    'correct': correct
                })

        print(f"    Strides: {strides}")
        print(f"    Sample mappings:")
        for m in sample_mappings:
            print(f"      out{m['out_coords']} -> src[{m['src_idx']}] = {m['value']} ({'✓' if m['correct'] else '✗'})")
        print(f"    Overall: {'PASS' if all_correct else 'FAIL'}\n")

        results.append({
            'src_shape': src_shape,
            'out_shape': out_shape,
            'strides': strides,
            'all_correct': all_correct,
            'sample_mappings': sample_mappings
        })

    return results


def test_broadcasting_properties():
    """Test mathematical properties of broadcasting"""
    print_section("TEST 4: Broadcasting Properties")

    results = {}

    # Property 1: Commutativity of shape calculation
    print("  Property 1: Shape Commutativity")
    print("  broadcast_shape(A, B) == broadcast_shape(B, A)")
    test_cases = [
        ((3, 1), (1, 3)),
        ((2, 3, 1), (1, 4)),
        ((5, 1, 3), (1, 4, 3)),
    ]

    commutative = True
    for s1, s2 in test_cases:
        out1 = compute_broadcast_shape(s1, s2)
        out2 = compute_broadcast_shape(s2, s1)
        match = out1 == out2
        print(f"    {s1} × {s2} = {out1}")
        print(f"    {s2} × {s1} = {out2} {'✓' if match else '✗'}")
        commutative = commutative and match

    results['commutative'] = commutative
    print(f"  Result: {'PASS' if commutative else 'FAIL'}\n")

    # Property 2: Identity (broadcasting with same shape)
    print("  Property 2: Identity")
    print("  broadcast_shape(A, A) == A.shape")
    shapes = [(3, 3), (2, 4, 5), (1, 1, 1)]

    identity = True
    for shape in shapes:
        out = compute_broadcast_shape(shape, shape)
        match = out == shape
        print(f"    {shape} × {shape} = {out} {'✓' if match else '✗'}")
        identity = identity and match

    results['identity'] = identity
    print(f"  Result: {'PASS' if identity else 'FAIL'}\n")

    # Property 3: Broadcasting preserves data
    print("  Property 3: Data Preservation")
    print("  Broadcasting doesn't modify source values")

    preservation = True
    src = np.array([[1, 2, 3]])
    broadcasted = np.broadcast_to(src, (4, 3))

    # Check that each row is identical to source
    for i in range(broadcasted.shape[0]):
        if not np.array_equal(broadcasted[i], src[0]):
            preservation = False
            break

    print(f"    Source: {src.shape} -> Broadcast: {broadcasted.shape}")
    print(f"    All rows identical to source: {'✓' if preservation else '✗'}")
    results['preservation'] = preservation
    print(f"  Result: {'PASS' if preservation else 'FAIL'}\n")

    return results


def generate_c_test_data():
    """Generate test data for C unit tests"""
    print_section("TEST 5: Generate C Test Data")

    test_cases = [
        ((2, 3), (2, 3)),
        ((1,), (2, 3)),
        ((3,), (2, 3)),
        ((2, 1), (2, 3)),
        ((1, 3), (2, 3)),
        ((2, 1, 4), (2, 3, 4)),
    ]

    c_test_data = []

    for shape1, shape2 in test_cases:
        arr1 = np.arange(np.prod(shape1), dtype=np.int32).reshape(shape1) + 1
        arr2 = np.arange(np.prod(shape2), dtype=np.int32).reshape(shape2) + 100
        result = arr1 + arr2

        out_shape = compute_broadcast_shape(shape1, shape2)
        strides1 = compute_broadcast_strides(shape1, out_shape)
        strides2 = compute_broadcast_strides(shape2, out_shape)

        test_data = {
            'shape1': list(shape1),
            'shape2': list(shape2),
            'data1': arr1.flatten().tolist(),
            'data2': arr2.flatten().tolist(),
            'output_shape': list(out_shape),
            'expected_result': result.flatten().tolist(),
            'strides1': strides1,
            'strides2': strides2,
        }

        c_test_data.append(test_data)

        print(f"  Test case: {shape1} × {shape2} -> {out_shape}")
        print(f"    Strides1: {strides1}")
        print(f"    Strides2: {strides2}")
        print(f"    Result sample: {result.flatten()[:5].tolist()}\n")

    # Save to JSON for C test generation
    with open('broadcast_test_data.json', 'w') as f:
        json.dump(c_test_data, f, indent=2)

    print("  Generated broadcast_test_data.json for C tests")

    return c_test_data


def main():
    """Run all validation tests"""
    print("="*80)
    print("BROADCASTING MECHANISM - GROUND TRUTH VALIDATION")
    print("="*80)
    print("\nThis script establishes ground truth for the Tofu broadcasting")
    print("implementation by validating against NumPy's behavior.\n")

    # Run all tests
    shape_results = test_shape_calculation()
    stride_results = test_stride_calculation()
    mapping_results = test_index_mapping()
    property_results = test_broadcasting_properties()
    c_test_data = generate_c_test_data()

    # Summary
    print_section("SUMMARY")

    shape_pass = sum(1 for r in shape_results if r.get('match', False))
    print(f"  Shape Calculation: {shape_pass}/{len(shape_results)} passed")

    stride_pass = sum(1 for r in stride_results if r.get('valid', False))
    print(f"  Stride Calculation: {stride_pass}/{len(stride_results)} passed")

    mapping_pass = sum(1 for r in mapping_results if r.get('all_correct', False))
    print(f"  Index Mapping: {mapping_pass}/{len(mapping_results)} passed")

    property_pass = sum(1 for v in property_results.values() if v)
    print(f"  Properties: {property_pass}/{len(property_results)} passed")

    print(f"\n  C test data generated: {len(c_test_data)} test cases")

    all_pass = (shape_pass == len(shape_results) and
                stride_pass == len(stride_results) and
                mapping_pass == len(mapping_results) and
                property_pass == len(property_results))

    print(f"\n  Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
