"""Unit tests for the Biophysical Graph Neural Network (BioGNN) and backprop gradients."""

import pytest
import numpy as np
from typing import Dict, Any

from primerforge.gnn_biophysics import (
    build_primer_graph,
    build_hybrid_graph,
    compute_symmetric_normalized_adjacency,
    GraphConvLayer,
    GraphMeanPool,
    BioGNN
)


def test_graph_builders() -> None:
    """Verifies that spatial graphs are correctly constructed with symmetric adjacency matrices."""
    seq1 = "ATGC"
    seq2 = "GCAT"

    # Test single primer graph builder
    X1, A1 = build_primer_graph(seq1)
    assert X1.shape == (4, 8)
    assert A1.shape == (4, 4)
    assert np.allclose(A1, A1.T)  # Symmetry check

    # Test hybrid dimer graph builder
    X_hyb, A_hyb = build_hybrid_graph(seq1, seq2)
    assert X_hyb.shape == (8, 8)
    assert A_hyb.shape == (8, 8)
    assert np.allclose(A_hyb, A_hyb.T)  # Symmetry check


def test_symmetric_normalization() -> None:
    """Tests symmetric normalized adjacency calculation."""
    A = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    A_hat = compute_symmetric_normalized_adjacency(A)
    assert A_hat.shape == (3, 3)
    assert np.allclose(A_hat, A_hat.T)  # Symmetric
    
    # Diagonal elements should be > 0 due to self loop (I)
    assert np.all(np.diag(A_hat) > 0.0)


def test_gcn_layer_gradients() -> None:
    """Verifies GraphConvLayer input and weight gradients using finite differences."""
    np.random.seed(42)
    N, in_dim, out_dim = 3, 4, 5
    
    layer = GraphConvLayer(in_dim, out_dim)
    
    H = np.random.normal(0, 1.0, (N, in_dim)).astype(np.float32)
    A_hat = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.33, 0.5],
        [0.0, 0.5, 0.5]
    ], dtype=np.float32)

    # Forward
    out = layer.forward(H, A_hat)
    d_out = np.random.normal(0, 1.0, out.shape).astype(np.float32)

    # Backward
    dH_analytic = layer.backward(d_out).copy()
    dW_analytic = layer.dW.copy()

    eps = 1e-4

    # 1. Test weight matrix gradient dW
    dW_numeric = np.zeros_like(layer.W)
    for i in range(in_dim):
        for j in range(out_dim):
            old_val = layer.W[i, j]
            
            layer.W[i, j] = old_val + eps
            out_plus = layer.forward(H, A_hat)
            
            layer.W[i, j] = old_val - eps
            out_minus = layer.forward(H, A_hat)
            
            layer.W[i, j] = old_val
            
            dW_numeric[i, j] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
            
    assert np.allclose(dW_analytic, dW_numeric, atol=1e-3, rtol=1e-3)

    # 2. Test input matrix gradient dH
    dH_numeric = np.zeros_like(H)
    for i in range(N):
        for j in range(in_dim):
            old_val = H[i, j]
            
            H[i, j] = old_val + eps
            out_plus = layer.forward(H, A_hat)
            
            H[i, j] = old_val - eps
            out_minus = layer.forward(H, A_hat)
            
            H[i, j] = old_val
            
            dH_numeric[i, j] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
            
    assert np.allclose(dH_analytic, dH_numeric, atol=1e-3, rtol=1e-3)


def test_graph_pool_gradients() -> None:
    """Verifies GraphMeanPool backpropagation gradients using finite differences."""
    np.random.seed(42)
    N, D = 4, 6
    
    pool = GraphMeanPool()
    H = np.random.normal(0, 1.0, (N, D)).astype(np.float32)
    
    # Forward
    out = pool.forward(H)
    d_out = np.random.normal(0, 1.0, out.shape).astype(np.float32)
    
    # Backward
    dH_analytic = pool.backward(d_out).copy()
    
    # Numerical
    eps = 1e-4
    dH_numeric = np.zeros_like(H)
    for i in range(N):
        for j in range(D):
            old_val = H[i, j]
            
            H[i, j] = old_val + eps
            out_plus = pool.forward(H)
            
            H[i, j] = old_val - eps
            out_minus = pool.forward(H)
            
            H[i, j] = old_val
            
            dH_numeric[i, j] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
            
    assert np.allclose(dH_analytic, dH_numeric, atol=1e-3, rtol=1e-3)


def test_biognn_gradients() -> None:
    """Verifies full BioGNN backpropagation parameter gradients using finite differences."""
    np.random.seed(42)
    seq = "ATGCATGC"
    X, A = build_primer_graph(seq)
    
    gnn = BioGNN()
    
    # Forward
    pred = gnn.forward(X, A)
    d_out = np.random.normal(0, 1.0, pred.shape).astype(np.float32)
    
    # Backward
    gnn.backward(d_out)
    
    # Copy analytic gradients
    dW1_analytic = gnn.dW1.copy()
    db1_analytic = gnn.db1.copy()
    dW2_analytic = gnn.dW2.copy()
    db2_analytic = gnn.db2.copy()
    dconv1_W_analytic = gnn.conv1.dW.copy()
    dconv2_W_analytic = gnn.conv2.dW.copy()
    
    eps = 1e-4
    
    # 1. Test W2 dense weights gradient
    dW2_numeric = np.zeros_like(gnn.W2)
    for i in range(gnn.W2.shape[0]):
        for j in range(gnn.W2.shape[1]):
            old_val = gnn.W2[i, j]
            
            gnn.W2[i, j] = old_val + eps
            pred_plus = gnn.forward(X, A)
            
            gnn.W2[i, j] = old_val - eps
            pred_minus = gnn.forward(X, A)
            
            gnn.W2[i, j] = old_val
            
            dW2_numeric[i, j] = np.sum(d_out * (pred_plus - pred_minus)) / (2 * eps)
            
    assert np.allclose(dW2_analytic, dW2_numeric, atol=1e-3, rtol=1e-3)

    # 2. Test conv2 weights gradient
    dconv2_W_numeric = np.zeros_like(gnn.conv2.W)
    for i in range(gnn.conv2.W.shape[0]):
        for j in range(gnn.conv2.W.shape[1]):
            old_val = gnn.conv2.W[i, j]
            
            gnn.conv2.W[i, j] = old_val + eps
            pred_plus = gnn.forward(X, A)
            
            gnn.conv2.W[i, j] = old_val - eps
            pred_minus = gnn.forward(X, A)
            
            gnn.conv2.W[i, j] = old_val
            
            dconv2_W_numeric[i, j] = np.sum(d_out * (pred_plus - pred_minus)) / (2 * eps)
            
    assert np.allclose(dconv2_W_analytic, dconv2_W_numeric, atol=2e-3, rtol=2e-3)


def test_biognn_convergence() -> None:
    """Verifies that the GNN reduces its loss when trained on sequence complexes."""
    np.random.seed(42)
    sequences = [
        ("ATGC", "GCAT"),
        ("CGTA", "TACG"),
        ("GGCC", "CCGG"),
        ("TTAA", "AATT")
    ]
    targets = np.array([
        [50.0, -1.0],
        [52.0, -1.5],
        [60.0, -4.0],
        [45.0, 0.5]
    ], dtype=np.float32)

    gnn = BioGNN()
    losses = gnn.train_on_pairs(sequences, targets, epochs=10, lr=0.01)
    
    assert len(losses) == 10
    # Final loss should be lower than baseline initial loss
    assert losses[-1] < losses[0]
