#!/usr/bin/env python3
"""
Generate weights for a deeper neural network with batch processing
Network: input(10) -> 64 -> 32 -> 16 -> output(5)
Batch size: 8
"""
import numpy as np

np.random.seed(42)

# Network architecture
INPUT_SIZE = 10
HIDDEN1_SIZE = 64
HIDDEN2_SIZE = 32
HIDDEN3_SIZE = 16
OUTPUT_SIZE = 5
BATCH_SIZE = 8

print("Deep Neural Network Architecture:")
print(f"  Input:   [{BATCH_SIZE}, {INPUT_SIZE}]")
print(f"  Hidden1: [{BATCH_SIZE}, {HIDDEN1_SIZE}] with ReLU")
print(f"  Hidden2: [{BATCH_SIZE}, {HIDDEN2_SIZE}] with ReLU")
print(f"  Hidden3: [{BATCH_SIZE}, {HIDDEN3_SIZE}] with ReLU")
print(f"  Output:  [{BATCH_SIZE}, {OUTPUT_SIZE}]")
print()

# Generate random weights
W1 = np.random.randn(INPUT_SIZE, HIDDEN1_SIZE).astype(np.float32) * 0.1
b1 = np.random.randn(HIDDEN1_SIZE).astype(np.float32) * 0.01

W2 = np.random.randn(HIDDEN1_SIZE, HIDDEN2_SIZE).astype(np.float32) * 0.1
b2 = np.random.randn(HIDDEN2_SIZE).astype(np.float32) * 0.01

W3 = np.random.randn(HIDDEN2_SIZE, HIDDEN3_SIZE).astype(np.float32) * 0.1
b3 = np.random.randn(HIDDEN3_SIZE).astype(np.float32) * 0.01

W4 = np.random.randn(HIDDEN3_SIZE, OUTPUT_SIZE).astype(np.float32) * 0.1
b4 = np.random.randn(OUTPUT_SIZE).astype(np.float32) * 0.01

print("Saving weights...")
W1.tofile('examples/deep_W1.bin')
b1.tofile('examples/deep_b1.bin')
W2.tofile('examples/deep_W2.bin')
b2.tofile('examples/deep_b2.bin')
W3.tofile('examples/deep_W3.bin')
b3.tofile('examples/deep_b3.bin')
W4.tofile('examples/deep_W4.bin')
b4.tofile('examples/deep_b4.bin')

# Batch input
X = np.random.randn(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
X.tofile('examples/deep_input.bin')
print(f"Input shape: {X.shape}")

# Forward pass
def relu(x):
    return np.maximum(0, x)

h1 = relu(X @ W1 + b1)
h2 = relu(h1 @ W2 + b2)
h3 = relu(h2 @ W3 + b3)
output = h3 @ W4 + b4

print(f"\nOutput shape: {output.shape}")
print(f"Output sample (first 2 samples):")
print(output[:2])

output.tofile('examples/deep_expected.bin')

print("\nOperations:")
print("1. h1 = ReLU(X @ W1 + b1)      # [8,10] @ [10,64] = [8,64]")
print("2. h2 = ReLU(h1 @ W2 + b2)     # [8,64] @ [64,32] = [8,32]")
print("3. h3 = ReLU(h2 @ W3 + b3)     # [8,32] @ [32,16] = [8,16]")
print("4. output = h3 @ W4 + b4       # [8,16] @ [16,5]  = [8,5]")

print("\nTotal parameters:",
      INPUT_SIZE*HIDDEN1_SIZE + HIDDEN1_SIZE +
      HIDDEN1_SIZE*HIDDEN2_SIZE + HIDDEN2_SIZE +
      HIDDEN2_SIZE*HIDDEN3_SIZE + HIDDEN3_SIZE +
      HIDDEN3_SIZE*OUTPUT_SIZE + OUTPUT_SIZE)
