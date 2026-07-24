"""Tests for change point detection."""
import pytest
import numpy as np
from src.cpd import (
    detect_change_points_ruptures_1d,
    detect_change_points_ruptures_multivariate,
    detect_change_points_threshold,
    map_window_change_points_to_messages,
    create_segments
)


def test_detect_change_points_ruptures_1d():
    """Test 1D change point detection returns sorted points within bounds."""
    # Create synthetic drift signal with clear change
    drift = np.concatenate([
        np.random.rand(10) * 0.1,  # Low drift
        np.random.rand(10) * 0.8 + 0.5,  # High drift
        np.random.rand(10) * 0.1  # Low drift again
    ])
    
    change_points = detect_change_points_ruptures_1d(
        drift, penalty=1.0, min_segment_length=2
    )
    
    # Should return sorted array
    assert len(change_points) == 0 or np.all(np.diff(change_points) >= 0)
    
    # All points should be within bounds
    assert np.all(change_points >= 0)
    assert np.all(change_points < len(drift))


def test_detect_change_points_threshold():
    """Test threshold-based change point detection."""
    drift = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.2, 0.1])
    threshold = 0.5
    min_segment_length = 1
    
    change_points = detect_change_points_threshold(drift, threshold, min_segment_length)
    
    # Should detect points above threshold
    assert len(change_points) > 0
    assert np.all(change_points >= 0)
    assert np.all(change_points < len(drift))


def test_map_window_change_points_to_messages():
    """Test mapping window change points to message indices."""
    change_points = np.array([2, 5])
    window_indices = np.array([
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8]
    ])
    
    message_cps = map_window_change_points_to_messages(change_points, window_indices)
    
    # Should map correctly
    assert len(message_cps) <= len(change_points)
    assert np.all(message_cps >= 0)


def test_create_segments():
    """Test segment creation respects min_segment_length."""
    n_messages = 20
    change_points = np.array([5, 10, 15])
    min_segment_length = 3
    
    segments = create_segments(n_messages, change_points, min_segment_length)
    
    # Should create segments
    assert len(segments) > 0
    
    # Check segment structure
    for seg in segments:
        assert 'start_idx' in seg
        assert 'end_idx' in seg
        assert seg['start_idx'] < seg['end_idx']
        assert seg['end_idx'] - seg['start_idx'] >= min_segment_length or seg['end_idx'] == n_messages


def test_create_segments_no_change_points():
    """Test segment creation with no change points."""
    n_messages = 10
    change_points = np.array([])
    
    segments = create_segments(n_messages, change_points, min_segment_length=2)
    
    # Should create single segment
    assert len(segments) == 1
    assert segments[0]['start_idx'] == 0
    assert segments[0]['end_idx'] == n_messages
