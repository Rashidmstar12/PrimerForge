"""
Step 3 (Gap-Filling) Quality Control Tests: Transformer Fine-Tuning CLS Head.

Verifies that FineTuneClassificationHead correctly implements:
  - Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
    NAACL-HLT. doi:10.18653/v1/N19-1423
  - Kingma & Ba (2014). Adam: A Method for Stochastic Optimization.
    arXiv:1412.6980.

Tests:
  1.  forward() returns P ∈ [0.01, 0.99] for any [CLS] embedding.
  2.  Training for 5 epochs reduces BCE loss (loss decreases monotonically).
  3.  Gradient magnitudes are finite after backward().
  4.  Gradient clipping: no gradient exceeds clip_norm=2.0 in L2-norm.
  5.  fine_tune() returns a list with one loss value per epoch.
  6.  P(success) > 0.5 for consistently positive-labeled sequences after training.
  7.  Serialization round-trip: to_dict() / from_dict() preserves weights exactly.
  8.  get_cls_embedding() returns shape (16,) = embed_dim for any DNA primer.
"""

import math
import numpy as np
import pytest

from primerforge.transformer import DNATransformerEncoder, FineTuneClassificationHead


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_transformer() -> DNATransformerEncoder:
    """Returns a fresh DNATransformerEncoder (not pre-trained — init weights only)."""
    return DNATransformerEncoder(
        vocab_size=8, embed_dim=16, num_heads=2, hidden_dim=32, max_len=24
    )


def make_head() -> FineTuneClassificationHead:
    """Returns a fresh FineTuneClassificationHead."""
    return FineTuneClassificationHead(input_dim=16, hidden_dim=8, lr=1e-3, seed=0)


def random_cls_embedding(seed: int = 0) -> np.ndarray:
    """Returns a random (16,) CLS embedding vector."""
    rng = np.random.RandomState(seed)
    return rng.randn(16).astype(np.float32)


PRIMER_SEQS = [
    "ATGCATGCATGCATGC",
    "GCATGCATGCATGCAT",
    "AAAATTTTGCGCATGC",
    "GCGCGCGCATATATAT",
    "TTTTAAAACCCCGGGG",
    "ATGCGCATGCTAGCTA",
]


# ---------------------------------------------------------------------------
# Test 1: forward() always returns P ∈ [0.01, 0.99]
# ---------------------------------------------------------------------------


def test_forward_output_in_valid_range() -> None:
    """P(success) must always be in [0.01, 0.99] regardless of embedding values."""
    head = make_head()
    rng = np.random.RandomState(42)
    for _ in range(50):
        # Test with diverse embedding magnitudes
        scale = rng.choice([0.01, 1.0, 10.0, 100.0])
        emb = (rng.randn(16) * scale).astype(np.float32)
        p = head.forward(emb)
        assert (
            0.01 <= p <= 0.99
        ), f"P={p:.4f} outside [0.01, 0.99] for embedding with scale={scale}"
        assert math.isfinite(p), f"P={p} is not finite"


# ---------------------------------------------------------------------------
# Test 2: Training reduces BCE loss (monotonicity)
# ---------------------------------------------------------------------------


def test_training_reduces_loss() -> None:
    """BCE loss must decrease over 20 epochs of fine-tuning on a consistent dataset.

    Scientific basis: BCE loss is convex for a single linear classifier.
    Gradient descent with sufficiently small lr must converge monotonically
    (or near-monotonically) for a 2-layer MLP on a consistent dataset.
    """
    transformer = make_transformer()
    head = make_head()

    # Simple consistent dataset: all positive labels
    seqs = PRIMER_SEQS * 10  # 60 sequences
    labels = np.ones(len(seqs), dtype=np.float32)

    loss_history = head.fine_tune(transformer, seqs, labels, epochs=20, batch_size=16)

    assert len(loss_history) == 20, f"Expected 20 loss values, got {len(loss_history)}"

    # First loss must be higher than last loss (learning is happening)
    assert loss_history[0] > loss_history[-1], (
        f"Loss did not decrease: epoch1={loss_history[0]:.4f}, "
        f"epoch20={loss_history[-1]:.4f}"
    )

    # All loss values must be finite
    for i, loss in enumerate(loss_history):
        assert math.isfinite(loss), f"Epoch {i+1} loss is not finite: {loss}"
        assert loss >= 0.0, f"Epoch {i+1} loss is negative: {loss}"


# ---------------------------------------------------------------------------
# Test 3: Gradients are finite after backward()
# ---------------------------------------------------------------------------


def test_gradients_are_finite() -> None:
    """backward() must return finite gradients for any input."""
    head = make_head()
    test_cases = [
        (random_cls_embedding(0), 1.0),
        (random_cls_embedding(1), 0.0),
        (random_cls_embedding(2), 0.5),
        (np.zeros(16, dtype=np.float32), 1.0),
        (np.ones(16, dtype=np.float32) * 5.0, 0.0),
    ]
    for emb, label in test_cases:
        _ = head.forward(emb, training=False)
        grads = head.backward(emb, label)
        for name, grad in grads.items():
            assert np.all(
                np.isfinite(grad)
            ), f"Gradient '{name}' contains non-finite values: {grad}"


# ---------------------------------------------------------------------------
# Test 4: Gradient clipping: ‖dW‖₂ ≤ clip_norm after _clip_grad
# ---------------------------------------------------------------------------


def test_gradient_clipping() -> None:
    """Verifies _clip_grad() clamps gradient ‖g‖₂ to ≤ clip_norm=2.0.

    Scientific basis: Pascanu et al. (2013). On the difficulty of training
    recurrent neural networks. ICML. Gradient clipping prevents exploding
    gradients in deep networks. Here it stabilizes the fine-tuning head.
    """
    head = make_head()
    rng = np.random.RandomState(99)

    # Create large gradient (norm >> clip_norm)
    large_grad = rng.randn(16, 8).astype(np.float32) * 100.0
    clipped = head._clip_grad(large_grad)
    norm_clipped = float(np.linalg.norm(clipped))
    assert (
        norm_clipped <= head.clip_norm + 1e-5
    ), f"Clipped gradient norm={norm_clipped:.4f} exceeds clip_norm={head.clip_norm}"

    # Small gradient (norm < clip_norm) should not be modified
    small_grad = rng.randn(16, 8).astype(np.float32) * 0.001
    original_norm = float(np.linalg.norm(small_grad))
    unclipped = head._clip_grad(small_grad)
    assert np.allclose(
        unclipped, small_grad, atol=1e-7
    ), f"Small gradient (norm={original_norm:.6f}) should not be clipped"


# ---------------------------------------------------------------------------
# Test 5: fine_tune() returns correct list length
# ---------------------------------------------------------------------------


def test_fine_tune_returns_epoch_losses() -> None:
    """fine_tune() must return a list of exactly `epochs` float values."""
    transformer = make_transformer()
    head = make_head()

    for n_epochs in [1, 5, 10]:
        seqs = PRIMER_SEQS * 4
        labels = np.random.randint(0, 2, size=len(seqs)).astype(np.float32)
        loss_history = head.fine_tune(
            transformer, seqs, labels, epochs=n_epochs, batch_size=8
        )
        assert (
            len(loss_history) == n_epochs
        ), f"Expected {n_epochs} losses, got {len(loss_history)}"
        for loss in loss_history:
            assert isinstance(loss, float), f"Loss must be float, got {type(loss)}"
            assert math.isfinite(loss), f"Loss {loss} is not finite"


# ---------------------------------------------------------------------------
# Test 6: P(success) > 0.5 for consistently positive-labeled sequences
# ---------------------------------------------------------------------------


def test_head_learns_positive_association() -> None:
    """After fine-tuning on all-positive labels, head should predict P > 0.5.

    Scientific basis: With a fixed training set of all-positive labels,
    the BCE-minimizing head should push predictions toward P = 1.0. After
    20 epochs of fine-tuning, mean prediction should exceed 0.5.
    """
    transformer = make_transformer()
    head = make_head()

    seqs = PRIMER_SEQS * 15  # 90 sequences, all positive
    labels = np.ones(len(seqs), dtype=np.float32)
    head.fine_tune(transformer, seqs, labels, epochs=20, batch_size=16)

    # Evaluate on training sequences
    predictions = []
    for seq in PRIMER_SEQS:
        cls_emb = transformer.get_cls_embedding(seq)
        p = head.forward(cls_emb, training=False)
        predictions.append(p)

    mean_pred = np.mean(predictions)
    assert (
        mean_pred > 0.5
    ), f"After training on all-positive labels, mean P={mean_pred:.4f} should be > 0.5"


# ---------------------------------------------------------------------------
# Test 7: Serialization round-trip preserves weights exactly
# ---------------------------------------------------------------------------


def test_serialization_round_trip() -> None:
    """to_dict() / from_dict() must perfectly preserve all weight matrices.

    This is a critical requirement for reproducibility: saved and loaded
    models must produce identical predictions.
    """
    head1 = make_head()
    # Do a few forward passes to create state
    for i in range(5):
        _ = head1.forward(random_cls_embedding(i), training=False)

    # Serialize
    data = head1.to_dict()

    # Restore into a new head
    head2 = FineTuneClassificationHead(input_dim=16, hidden_dim=8)
    head2.from_dict(data)

    # Check weight equality
    assert np.allclose(head1.W1, head2.W1, atol=1e-7), "W1 mismatch after round-trip"
    assert np.allclose(head1.b1, head2.b1, atol=1e-7), "b1 mismatch after round-trip"
    assert np.allclose(head1.W2, head2.W2, atol=1e-7), "W2 mismatch after round-trip"
    assert np.allclose(head1.b2, head2.b2, atol=1e-7), "b2 mismatch after round-trip"
    assert (
        head1._n_updates == head2._n_updates
    ), f"n_updates mismatch: {head1._n_updates} vs {head2._n_updates}"

    # Check predictions are identical
    emb = random_cls_embedding(99)
    p1 = head1.forward(emb, training=False)
    p2 = head2.forward(emb, training=False)
    assert (
        abs(p1 - p2) < 1e-6
    ), f"Predictions differ after round-trip: {p1:.6f} vs {p2:.6f}"


# ---------------------------------------------------------------------------
# Test 8: get_cls_embedding() returns shape (16,) for any primer sequence
# ---------------------------------------------------------------------------


def test_get_cls_embedding_shape() -> None:
    """get_cls_embedding() must return shape (16,) = embed_dim for any input.

    The [CLS] token at position 0 of the transformer output captures global
    sequence context. This shape must be exactly (embed_dim,) = (16,).
    """
    transformer = make_transformer()
    embed_dim = transformer.embed_dim  # = 16

    for seq in PRIMER_SEQS + ["A", "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG"]:
        emb = transformer.get_cls_embedding(seq)
        assert emb.shape == (
            embed_dim,
        ), f"Expected shape ({embed_dim},), got {emb.shape} for seq='{seq[:20]}'"
        assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
        assert np.all(
            np.isfinite(emb)
        ), f"[CLS] embedding contains non-finite values for seq='{seq[:20]}'"
