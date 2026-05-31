import numpy as np
from primerforge.transformer import MLMHeadLayer

np.random.seed(42)
B, T, D = 2, 3, 4
vocab_size = 8

layer = MLMHeadLayer(D, vocab_size)
x = np.random.normal(0, 1.0, (B, T, D)).astype(np.float32)

masked_positions = (np.array([0, 1]), np.array([1, 2]))
targets = np.array([2, 5], dtype=np.int32)

# Forward
probs = layer.forward(x, masked_positions)
print("probs:\n", probs)

# Backward
d_x_masked_analytic = layer.backward(targets).copy()
dW_analytic = layer.dW.copy()
print("dW_analytic:\n", dW_analytic)

# Numerical gradients
eps = 1e-4
num_masked = len(targets)
dW_numeric = np.zeros_like(layer.W)

for i in range(D):
    for j in range(vocab_size):
        old_val = layer.W[i, j]
        
        # Loss at +eps
        layer.W[i, j] = old_val + eps
        probs_plus = layer.forward(x, masked_positions)
        loss_plus = -np.mean(np.log(probs_plus[np.arange(num_masked), targets] + 1e-12))
        
        # Loss at -eps
        layer.W[i, j] = old_val - eps
        probs_minus = layer.forward(x, masked_positions)
        loss_minus = -np.mean(np.log(probs_minus[np.arange(num_masked), targets] + 1e-12))
        
        layer.W[i, j] = old_val
        
        dW_numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

print("dW_numeric:\n", dW_numeric)
print("diff:\n", dW_analytic - dW_numeric)
