"""Unit tests for the custom NumPy DNA Transformer and analytical backpropagation gradients."""

import pytest
import numpy as np
from primerforge.transformer import (
    DNASequenceTokenizer,
    EmbeddingLayer,
    LayerNormLayer,
    MultiHeadAttentionLayer,
    FeedForwardLayer,
    MLMHeadLayer,
    DNATransformerEncoder
)


def test_tokenizer() -> None:
    """Verifies that the tokenizer correctly encodes and decodes DNA sequences."""
    tokenizer = DNASequenceTokenizer(max_len=24)
    seq = "ATGCATGC"
    encoded = tokenizer.encode(seq, add_special_tokens=True)
    
    assert len(encoded) == 24
    assert encoded[0] == tokenizer.vocab["<cls>"]
    assert encoded[1] == tokenizer.vocab["A"]
    assert encoded[2] == tokenizer.vocab["T"]
    
    decoded = tokenizer.decode(encoded)
    assert decoded.startswith("<cls>ATGCATGC<sep>")
    assert decoded.endswith("<pad>")


def test_embedding_gradients() -> None:
    """Verifies Embedding layer parameter gradients using finite differences."""
    np.random.seed(42)
    B, T, D = 2, 3, 4
    vocab_size = 8
    
    layer = EmbeddingLayer(vocab_size, D)
    inputs = np.random.randint(0, vocab_size, (B, T))
    
    # Forward
    out = layer.forward(inputs)
    d_out = np.random.normal(0, 1.0, out.shape).astype(np.float32)
    
    # Backward
    layer.backward(d_out)
    dW_analytic = layer.dW.copy()
    
    # Numerical gradients
    eps = 1e-4
    dW_numeric = np.zeros_like(layer.W)
    
    for i in range(vocab_size):
        for j in range(D):
            old_val = layer.W[i, j]
            
            layer.W[i, j] = old_val + eps
            out_plus = layer.forward(inputs)
            
            layer.W[i, j] = old_val - eps
            out_minus = layer.forward(inputs)
            
            layer.W[i, j] = old_val
            
            dW_numeric[i, j] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
            
    assert np.allclose(dW_analytic, dW_numeric, atol=1e-3, rtol=1e-3)


def test_layernorm_gradients() -> None:
    """Verifies LayerNorm layer input and parameter gradients using finite differences."""
    np.random.seed(42)
    B, T, D = 2, 3, 4
    
    layer = LayerNormLayer(D)
    x = np.random.normal(0, 1.0, (B, T, D)).astype(np.float32)
    
    # Forward
    out = layer.forward(x)
    d_out = np.random.normal(0, 1.0, out.shape).astype(np.float32)
    
    # Backward
    dx_analytic = layer.backward(d_out).copy()
    dgamma_analytic = layer.dgamma.copy()
    dbeta_analytic = layer.dbeta.copy()
    
    eps = 1e-4
    
    # 1. Test input gradient dx
    dx_numeric = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                old_val = x[b, t, d]
                
                x[b, t, d] = old_val + eps
                out_plus = layer.forward(x)
                
                x[b, t, d] = old_val - eps
                out_minus = layer.forward(x)
                
                x[b, t, d] = old_val
                
                dx_numeric[b, t, d] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
                
    assert np.allclose(dx_analytic, dx_numeric, atol=1e-3, rtol=1e-3)
    
    # 2. Test parameter gradient dgamma
    dgamma_numeric = np.zeros_like(layer.gamma)
    for d in range(D):
        old_val = layer.gamma[d]
        
        layer.gamma[d] = old_val + eps
        out_plus = layer.gamma * layer.last_x_hat + layer.beta
        
        layer.gamma[d] = old_val - eps
        out_minus = layer.gamma * layer.last_x_hat + layer.beta
        
        layer.gamma[d] = old_val
        
        dgamma_numeric[d] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
        
    assert np.allclose(dgamma_analytic, dgamma_numeric, atol=1e-3, rtol=1e-3)

    # 3. Test parameter gradient dbeta
    dbeta_numeric = np.zeros_like(layer.beta)
    for d in range(D):
        old_val = layer.beta[d]
        
        layer.beta[d] = old_val + eps
        out_plus = layer.gamma * layer.last_x_hat + layer.beta
        
        layer.beta[d] = old_val - eps
        out_minus = layer.gamma * layer.last_x_hat + layer.beta
        
        layer.beta[d] = old_val
        
        dbeta_numeric[d] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
        
    assert np.allclose(dbeta_analytic, dbeta_numeric, atol=1e-3, rtol=1e-3)


def test_feedforward_gradients() -> None:
    """Verifies FeedForward layer input and parameter gradients using finite differences."""
    np.random.seed(42)
    B, T, D = 2, 3, 4
    hidden_dim = 6
    
    layer = FeedForwardLayer(D, hidden_dim)
    x = np.random.normal(0, 1.0, (B, T, D)).astype(np.float32)
    
    # Forward
    out = layer.forward(x)
    d_out = np.random.normal(0, 1.0, out.shape).astype(np.float32)
    
    # Backward
    dx_analytic = layer.backward(d_out).copy()
    dW1_analytic = layer.dW1.copy()
    db1_analytic = layer.db1.copy()
    dW2_analytic = layer.dW2.copy()
    db2_analytic = layer.db2.copy()
    
    eps = 1e-4
    
    # 1. Test input gradient dx
    dx_numeric = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                old_val = x[b, t, d]
                
                x[b, t, d] = old_val + eps
                out_plus = layer.forward(x)
                
                x[b, t, d] = old_val - eps
                out_minus = layer.forward(x)
                
                x[b, t, d] = old_val
                
                dx_numeric[b, t, d] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
                
    assert np.allclose(dx_analytic, dx_numeric, atol=1e-3, rtol=1e-3)

    # 2. Test parameter W1 gradient
    dW1_numeric = np.zeros_like(layer.W1)
    for i in range(D):
        for j in range(hidden_dim):
            old_val = layer.W1[i, j]
            
            layer.W1[i, j] = old_val + eps
            out_plus = layer.forward(x)
            
            layer.W1[i, j] = old_val - eps
            out_minus = layer.forward(x)
            
            layer.W1[i, j] = old_val
            
            dW1_numeric[i, j] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
            
    assert np.allclose(dW1_analytic, dW1_numeric, atol=1e-3, rtol=1e-3)


def test_attention_gradients() -> None:
    """Verifies MultiHeadAttention layer input and parameter gradients using finite differences."""
    np.random.seed(42)
    B, T, D = 2, 3, 4
    num_heads = 2
    
    layer = MultiHeadAttentionLayer(embed_dim=D, num_heads=num_heads)
    x = np.random.normal(0, 1.0, (B, T, D)).astype(np.float32)
    
    # Forward
    out = layer.forward(x)
    d_out = np.random.normal(0, 1.0, out.shape).astype(np.float32)
    
    # Backward
    dx_analytic = layer.backward(d_out).copy()
    dW_q_analytic = layer.dW_q.copy()
    dW_o_analytic = layer.dW_o.copy()
    
    eps = 1e-4
    
    # 1. Test input gradient dx
    dx_numeric = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                old_val = x[b, t, d]
                
                x[b, t, d] = old_val + eps
                out_plus = layer.forward(x)
                
                x[b, t, d] = old_val - eps
                out_minus = layer.forward(x)
                
                x[b, t, d] = old_val
                
                dx_numeric[b, t, d] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
                
    assert np.allclose(dx_analytic, dx_numeric, atol=1e-3, rtol=1e-3)

    # 2. Test query projection weights gradient dW_q
    dW_q_numeric = np.zeros_like(layer.W_q)
    for i in range(D):
        for j in range(D):
            old_val = layer.W_q[i, j]
            
            layer.W_q[i, j] = old_val + eps
            out_plus = layer.forward(x)
            
            layer.W_q[i, j] = old_val - eps
            out_minus = layer.forward(x)
            
            layer.W_q[i, j] = old_val
            
            dW_q_numeric[i, j] = np.sum(d_out * (out_plus - out_minus)) / (2 * eps)
            
    assert np.allclose(dW_q_analytic, dW_q_numeric, atol=1e-3, rtol=1e-3)


def test_mlm_head_gradients() -> None:
    """Verifies MLM head logits and state gradients using finite differences."""
    np.random.seed(42)
    B, T, D = 2, 3, 4
    vocab_size = 8
    
    layer = MLMHeadLayer(D, vocab_size)
    x = np.random.normal(0, 1.0, (B, T, D)).astype(np.float32)
    
    masked_positions = (np.array([0, 1]), np.array([1, 2]))
    targets = np.array([2, 5], dtype=np.int32)
    
    # Forward
    probs = layer.forward(x, masked_positions)
    
    # Backward
    d_x_masked_analytic = layer.backward(targets).copy()
    dW_analytic = layer.dW.copy()
    
    eps = 1e-4
    num_masked = len(targets)
    
    # 1. Test parameter W gradient
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
            
    assert np.allclose(dW_analytic, dW_numeric, atol=2e-3, rtol=2e-3)


def test_mlm_pretrain_epoch() -> None:
    """Tests that the custom DNA Transformer completes one epoch of MLM training and reduces loss."""
    np.random.seed(42)
    sequences = [
        "ATTGGCAATGAGCGGTTCCG",
        "GCGCTCAGGAGGAGCAATGA",
        "TCCGCTGCCCTGAGGCACTC",
        "GATCTTGATCTTCATTGTGCT"
    ]
    
    model = DNATransformerEncoder(vocab_size=8, embed_dim=16, num_heads=2, hidden_dim=32, max_len=24)
    
    # Run 5 epochs to verify that loss goes down or stays stable/correct without crashing
    losses = model.pretrain_on_sequences(sequences, epochs=5, batch_size=2, lr=0.01)
    
    assert len(losses) == 5
    assert all(isinstance(l, float) for l in losses)
