# Validation Scripts

This directory contains Python scripts for validating the C implementation against NumPy.

## Core Validation Scripts

These scripts establish ground truth and validate the implementation:

- **`validate_inner_comprehensive.py`** - Validates `tl_tensor_inner()` against `np.inner()`
- **`validate_matmul.py`** - Validates `tl_tensor_matmul()` against `np.matmul()`
- **`test_broadcast_ground_truth.py`** - Validates broadcasting mechanism internals

## Documentation Scripts

These scripts document behavioral differences between NumPy operations:

- **`test_inner_vs_dot.py`** - Demonstrates differences between `np.inner()` and `np.dot()`
- **`test_dot_vs_matmul.py`** - Demonstrates differences between `np.dot()` and `np.matmul()`

## Test Data

- **`broadcast_test_data.json`** - Test data for broadcasting mechanism validation

## Usage

Run validation scripts to verify C implementation matches NumPy:

```bash
python3 validation/validate_inner_comprehensive.py
python3 validation/validate_matmul.py
python3 validation/test_broadcast_ground_truth.py
```

All scripts should output "ALL TESTS PASSED ✓" if the implementation is correct.
