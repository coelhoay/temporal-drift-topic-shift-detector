"""Tests for drift computation."""
import pytest
import numpy as np
from src.drift import (
    rolling_window_centroids, compute_drift_signal, 
    apply_ewma, compute_drift_with_smoothing
)


def test_rolling_window_centroids():
    """Test windowing correctness."""
    # Create simple test embeddings
    embeddings = np.random.rand(10, 5)  # 10 messages, 5-dim embeddings
    window_size = 3
    stride = 1
    
    centroids, window_indices = rolling_window_centroids(embeddings, window_size, stride)
    
    # Should have (10 - 3 + 1) = 8 windows
    assert len(centroids) == 8
    assert len(window_indices) == 8
    assert centroids.shape[1] == 5  # Same embedding dimension
    
    # Check first window
    expected_first_centroid = embeddings[0:3].mean(axis=0)
    np.testing.assert_array_almost_equal(centroids[0], expected_first_centroid)
    
    # Check window indices
    assert window_indices[0, 0] == 0
    assert window_indices[0, 1] == 3


def test_rolling_window_centroids_large_window():
    """Test windowing when window is larger than data."""
    embeddings = np.random.rand(5, 3)
    window_size = 10
    
    centroids, window_indices = rolling_window_centroids(embeddings, window_size, stride=1)
    
    # Should return single centroid
    assert len(centroids) == 1
    assert len(window_indices) == 1
    expected_centroid = embeddings.mean(axis=0)
    np.testing.assert_array_almost_equal(centroids[0], expected_centroid)


def test_compute_drift_signal():
    """Test drift computation shape."""
    # Create centroids
    centroids = np.random.rand(5, 10)  # 5 windows, 10-dim
    
    drift = compute_drift_signal(centroids)
    
    # Should have (n_windows - 1) drift values
    assert len(drift) == 4
    assert drift.shape == (4,)
    
    # All drift values should be between 0 and 2 (cosine distance range)
    assert np.all(drift >= 0)
    assert np.all(drift <= 2)


def test_compute_drift_signal_single_centroid():
    """Test drift with single centroid."""
    centroids = np.random.rand(1, 10)
    drift = compute_drift_signal(centroids)
    assert len(drift) == 1
    assert drift[0] == 0.0


def test_apply_ewma():
    """Test EWMA smoothing."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.5
    
    smoothed = apply_ewma(signal, alpha)
    
    assert len(smoothed) == len(signal)
    assert smoothed[0] == signal[0]  # First value unchanged
    assert smoothed[-1] > signal[0]  # Should be smoothed upward


def test_apply_ewma_empty():
    """Test EWMA with empty signal."""
    signal = np.array([])
    smoothed = apply_ewma(signal, 0.3)
    assert len(smoothed) == 0


def test_compute_drift_with_smoothing():
    """Test full drift computation pipeline."""
    embeddings = np.random.rand(20, 8)
    window_size = 5
    stride = 2
    ewma_alpha = 0.3
    
    drift, centroids, window_indices = compute_drift_with_smoothing(
        embeddings, window_size, stride, ewma_alpha
    )
    
    # Check shapes
    n_windows = len(centroids)
    assert len(drift) == n_windows - 1
    assert len(window_indices) == n_windows
    assert centroids.shape[1] == embeddings.shape[1]
