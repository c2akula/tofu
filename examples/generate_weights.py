#!/usr/bin/env python3
"""
Generate test weights for a simple 2-layer neural network.
Network: input(4) -> hidden(8) -> output(3)
"""
import numpy as np

np.random.seed(42)

# Network architecture
INPUT_SIZE = 4
HIDDEN_SIZE = 8
OUTPUT_SIZE = 3

# Generate random weights
W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.5
b1 = np.random.randn(HIDDEN_SIZE).astype(np.float32) * 0.1

W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE).astype(np.float32) * 0.5
b2 = np.random.randn(OUTPUT_SIZE).astype(np.float32) * 0.1

print("Generated weights:")
print(f"W1 shape: {W1.shape}")
print(f"b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}")
print(f"b2 shape: {b2.shape}")

# Save as binary files
W1.tofile('examples/W1.bin')
b1.tofile('examples/b1.bin')
W2.tofile('examples/W2.bin')
b2.tofile('examples/b2.bin')

print("\nSaved weights to examples/*.bin")

# Test input
X = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
X.tofile('examples/test_input.bin')
print(f"Test input shape: {X.shape}")

# Run inference
def relu(x):
    return np.maximum(0, x)

# Forward pass
hidden = relu(X @ W1 + b1)
output = hidden @ W2 + b2

print(f"\nExpected output:")
print(output)
output.tofile('examples/expected_output.bin')

print("\n" + "="*60)
print("Network Summary:")
print("="*60)
print(f"Input:   [{INPUT_SIZE}]")
print(f"Hidden:  [{HIDDEN_SIZE}] with ReLU activation")
print(f"Output:  [{OUTPUT_SIZE}] (no activation)")
print("\nOperations:")
print("1. hidden = ReLU(X @ W1 + b1)")
print("2. output = hidden @ W2 + b2")
