"""
cpd.py
------
Change-point detection module for the Temporal Drift Topic Shift Detector.

This module provides three distinct strategies for identifying locations in
the drift signal (produced by drift.py) at which a significant topic shift
has occurred. Each strategy makes different assumptions about the nature of
the change and exposes different tuning parameters.

Detection strategies
--------------------
1. ruptures_pelt_1d (detect_change_points_ruptures_1d)
   Applies the PELT (Pruned Exact Linear Time) algorithm from the
   ruptures library to the scalar drift signal using an L2 cost function.
   This is the recommended strategy for most datasets.

2. ruptures_pelt_multivariate (detect_change_points_ruptures_multivariate)
   Applies PELT directly to the full-dimensional centroid sequence using a
   Radial Basis Function (RBF) kernel cost. More computationally expensive
   but can detect directional changes not visible in the 1-D drift signal.

3. threshold_only (detect_change_points_threshold)
   Marks any point in the drift signal that exceeds a user-defined threshold
   as a change point. The simplest and most transparent approach but requires
   manual threshold selection.

After detection, the identified change points are in window-index space and
must be translated back to message-index space via
map_window_change_points_to_messages() before segments can be formed.
"""

import numpy as np
from typing import Optional
import ruptures as rpt


# ---------------------------------------------------------------------------
# Strategy 1: PELT on 1-D drift signal
# ---------------------------------------------------------------------------

def detect_change_points_ruptures_1d(
    drift_signal: np.ndarray,
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = None,
    min_segment_length: int = 2
) -> np.ndarray:
    """
    Detect change points in the 1-D drift signal using the PELT algorithm
    with an L2 cost function.

    PELT (Killick et al., 2012) finds the globally optimal segmentation of
    a signal by minimising a penalised cost function. The L2 cost penalises
    deviations from the segment mean. The penalty parameter controls the
    trade-off between fit quality and model complexity (number of change
    points).

    Parameters
    ----------
    drift_signal : np.ndarray
        1-D array of smoothed drift values produced by drift.py.
    penalty : float or None, optional
        Penalty applied per change point. A higher penalty yields fewer
        change points; a lower penalty yields more. If both penalty and
        n_bkps are None, a default penalty of 1.0 is used.
    n_bkps : int or None, optional
        Exact number of change points to detect. Overrides penalty when
        provided.
    min_segment_length : int, optional
        Minimum number of windows between two consecutive change points.
        Prevents spurious micro-shifts. Default is 2.

    Returns
    -------
    change_points : np.ndarray
        Sorted 1-D array of 0-indexed change-point positions within the
        drift signal. Empty if no change points are found.

    Known issue
    -----------
    When the penalty value is set too low (below approximately 0.3 on
    normalised drift signals), PELT tends to over-segment the signal,
    producing many spurious change points. This was observed during testing
    with the synthetic dataset (see io.py :: generate_sample_data).
    The user-facing penalty slider in app.py (line ~83) is bounded below
    at 0.1 to mitigate this.

    Implementation note: ruptures sentinel removal
    -----------------------------------------------
    The ruptures library always appends the length of the input signal as
    the final element of its prediction list (a sentinel marking the end of
    the sequence). This value must be removed before use; otherwise it would
    be interpreted as an out-of-bounds change-point index. The removal is
    performed by the change_points[:-1] slice below.
    """
    # Guard: PELT requires at least 2 * min_segment_length data points
    if len(drift_signal) < min_segment_length * 2:
        return np.array([])

    # ruptures requires a 2-D array (n_samples, n_features); reshape the
    # 1-D drift signal to (n, 1) to satisfy this requirement.
    signal_2d = drift_signal.reshape(-1, 1)

    # Instantiate PELT with L2 cost. L2 is appropriate here because the
    # drift values are continuous and approximately normally distributed
    # within a stable topic segment.
    algo = rpt.Pelt(model="l2", min_size=min_segment_length)
    algo.fit(signal_2d)

    if penalty is not None:
        change_points = algo.predict(pen=penalty)
    elif n_bkps is not None:
        change_points = algo.predict(n_bkps=n_bkps)
    else:
        change_points = algo.predict(pen=1.0)

    # Remove the trailing end-of-signal sentinel (ruptures convention)
    change_points = (
        np.array(change_points[:-1]) if len(change_points) > 0 else np.array([])
    )

    # Ensure all remaining change points are within valid array bounds
    change_points = change_points[
        (change_points >= 0) & (change_points < len(drift_signal))
    ]

    return np.sort(change_points)


# ---------------------------------------------------------------------------
# Strategy 2: PELT on multivariate centroid sequence
# ---------------------------------------------------------------------------

def detect_change_points_ruptures_multivariate(
    centroids: np.ndarray,
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = None,
    min_segment_length: int = 2
) -> np.ndarray:
    """
    Detect change points in the full-dimensional centroid sequence using
    PELT with an RBF kernel cost function.

    Unlike the 1-D approach, this function operates directly on the
    high-dimensional centroid embeddings. The RBF kernel measures
    distributional similarity between windows and can capture
    multi-directional changes that may be invisible in the scalar
    cosine-distance signal.

    Parameters
    ----------
    centroids : np.ndarray
        Shape (n_windows, embedding_dim). The window centroid sequence
        returned by drift.py :: compute_drift_with_smoothing().
    penalty : float or None, optional
        Same semantics as in detect_change_points_ruptures_1d.
    n_bkps : int or None, optional
        Same semantics as in detect_change_points_ruptures_1d.
    min_segment_length : int, optional
        Minimum number of consecutive windows in a valid segment.

    Returns
    -------
    change_points : np.ndarray
        Sorted 1-D array of 0-indexed change-point positions within the
        centroid sequence.

    Performance note
    ----------------
    The RBF kernel computation scales as O(n^2) in the number of windows,
    so this strategy can be noticeably slower than the 1-D approach when
    the dataset is large. For interactive use, the 1-D approach is
    recommended as the default.
    """
    if len(centroids) < min_segment_length * 2:
        return np.array([])

    algo = rpt.Pelt(model="rbf", min_size=min_segment_length)
    algo.fit(centroids)

    if penalty is not None:
        change_points = algo.predict(pen=penalty)
    elif n_bkps is not None:
        change_points = algo.predict(n_bkps=n_bkps)
    else:
        change_points = algo.predict(pen=1.0)

    # Remove ruptures end-of-sequence sentinel value
    change_points = (
        np.array(change_points[:-1]) if len(change_points) > 0 else np.array([])
    )

    # Bounds check
    change_points = change_points[
        (change_points >= 0) & (change_points < len(centroids))
    ]

    return np.sort(change_points)


# ---------------------------------------------------------------------------
# Strategy 3: Simple threshold-based detection
# ---------------------------------------------------------------------------

def detect_change_points_threshold(
    drift_signal: np.ndarray,
    threshold: float,
    min_segment_length: int = 2
) -> np.ndarray:
    """
    Detect change points using a simple amplitude threshold on the drift
    signal.

    Any window index at which the smoothed drift value exceeds threshold
    is treated as a potential change point. A minimum spacing constraint
    (min_segment_length) is then applied to prevent clusters of detections
    from being reported as multiple independent change points.

    Parameters
    ----------
    drift_signal : np.ndarray
        1-D array of smoothed drift values.
    threshold : float
        Drift value above which a window is classified as a change point.
        Typical values range from 0.1 to 0.4 depending on the embedding
        model and dataset characteristics.
    min_segment_length : int, optional
        Minimum number of windows between accepted change points. Default is 2.

    Returns
    -------
    change_points : np.ndarray
        Sorted 1-D array of change-point indices within the drift signal.

    Known issue / threshold sensitivity
    -------------------------------------
    This method is highly sensitive to the chosen threshold value. If the
    threshold is set below the noise floor of the drift signal, virtually
    every window will be flagged as a change point. Conversely, if the
    threshold exceeds the peak drift values, no change points will be
    detected. During development it was found that normalising the threshold
    relative to the 95th percentile of the drift signal would provide more
    robust defaults. This enhancement is identified as a candidate for
    future work.
    """
    if len(drift_signal) == 0:
        return np.array([])

    # Identify all window indices where the drift exceeds the threshold
    above_threshold = np.where(drift_signal > threshold)[0]

    if len(above_threshold) == 0:
        return np.array([])

    # Apply minimum spacing: greedily select change points that are at least
    # min_segment_length windows apart. The first exceedance is always kept.
    change_points = [above_threshold[0]]
    for idx in above_threshold[1:]:
        if idx - change_points[-1] >= min_segment_length:
            change_points.append(idx)

    return np.array(change_points)


# ---------------------------------------------------------------------------
# Coordinate mapping: window space -> message space
# ---------------------------------------------------------------------------

def map_window_change_points_to_messages(
    change_points: np.ndarray,
    window_indices: np.ndarray
) -> np.ndarray:
    """
    Translate change-point indices from window-index space to the original
    message-index space.

    The drift signal and change-point detectors operate on window indices
    (0, 1, 2, ..., n_windows-1). This function translates those indices
    to message positions using the window_indices array produced by
    rolling_window_centroids() in drift.py.

    Each change point cp in window space maps to the start index of window
    cp in the original message array (window_indices[cp, 0]).

    Parameters
    ----------
    change_points : np.ndarray
        1-D array of change-point indices in window space.
    window_indices : np.ndarray
        Shape (n_windows, 2). The [start_idx, end_idx] pairs for each
        window, as returned by rolling_window_centroids().

    Returns
    -------
    message_change_points : np.ndarray
        Sorted 1-D array of unique change-point indices in message space.
        np.unique is applied to remove duplicates that can arise when
        multiple window change points map to the same message start index
        (possible with a stride > 1).
    """
    if len(change_points) == 0:
        return np.array([])

    message_cps = []
    for cp in change_points:
        if cp < len(window_indices):
            message_cps.append(window_indices[cp, 0])

    return np.unique(np.array(message_cps))


# ---------------------------------------------------------------------------
# Segment creation with short-segment merging
# ---------------------------------------------------------------------------

def create_segments(
    n_messages: int,
    change_points: np.ndarray,
    min_segment_length: int = 2
) -> list:
    """
    Create a list of contiguous, non-overlapping topic segments from a set
    of message-space change points.

    After mapping change points back to message space, this function
    partitions the full message sequence into segments. Very short segments
    (shorter than min_segment_length) are merged with the preceding segment
    to avoid producing fragments that would be meaningless for topic
    labelling.

    Parameters
    ----------
    n_messages : int
        Total number of messages in the dataset.
    change_points : np.ndarray
        1-D array of change-point indices in message space.
    min_segment_length : int, optional
        Minimum number of messages a segment must contain. Default is 2.

    Returns
    -------
    segments : list of dict
        Each dict contains:
            'start_idx' : int  - inclusive start index in the message array
            'end_idx'   : int  - exclusive end index in the message array

    Design decision: forward-merging vs. backward-merging
    ------------------------------------------------------
    Short segments are merged with the *preceding* segment rather than the
    following one. This was chosen because a short fragment at the start of
    a new topic is semantically closer to what preceded it than to the full
    segment that follows it.
    """
    if len(change_points) == 0:
        return [{"start_idx": 0, "end_idx": n_messages}]

    all_points = np.concatenate([[0], np.sort(change_points), [n_messages]])

    segments = []
    for i in range(len(all_points) - 1):
        start = int(all_points[i])
        end = int(all_points[i + 1])
        segment_length = end - start

        if len(segments) > 0 and segment_length < min_segment_length:
            segments[-1]["end_idx"] = end
        else:
            segments.append({"start_idx": start, "end_idx": end})

    return segments
