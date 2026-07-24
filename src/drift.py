"""
drift.py
--------
Temporal embedding drift computation module.

This module implements the core mathematical operations required to model
how the semantic focus of an online conversation changes over time. The
primary technique is a rolling-window approach: the messages are grouped
into overlapping windows, a centroid (mean) embedding is computed for
each window, and the cosine distance between consecutive centroids
provides a scalar drift signal. Exponential Weighted Moving Average
(EWMA) smoothing is then applied to reduce noise before the signal is
passed to change-point detection.

Pipeline
--------
raw embeddings
    -> rolling_window_centroids()   # group into windows, compute means
    -> compute_drift_signal()       # cosine distance between consecutive centroids
    -> apply_ewma()                 # noise reduction via exponential smoothing
    -> [returned to cpd.py for change-point detection]

Mathematical background
-----------------------
Cosine distance between two vectors u and v is defined as:

    d(u, v) = 1 - (u . v) / (||u|| * ||v||)

A value of 0 indicates identical direction (same topic focus);
a value approaching 1 indicates a significant semantic departure.

EWMA smoothing at time t with factor alpha:

    s_t = alpha * x_t + (1 - alpha) * s_{t-1}

A smaller alpha retains more history (smoother signal); a larger alpha
reacts more quickly to recent changes but retains more noise.
"""

import numpy as np
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Low-level vector utilities
# ---------------------------------------------------------------------------

def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the centroid (arithmetic mean) of a set of embedding vectors.

    The centroid summarises the collective semantic focus of a group of
    messages. When all messages discuss a similar theme their embeddings
    cluster together and the centroid lies close to each of them. As
    topic diversity within the window increases, the centroid shifts
    towards the geometric centre of the spread, which still reflects the
    dominant direction of meaning.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_messages, embedding_dim). Each row is the
        embedding vector for one message. Must contain at least one row.

    Returns
    -------
    np.ndarray
        Shape (embedding_dim,). The element-wise mean across all rows.

    Notes
    -----
    No normalisation is applied here. The downstream cosine_distance
    function handles normalisation implicitly via cosine_similarity.
    """
    # np.mean over axis=0 collapses the message dimension, yielding one
    # vector whose i-th component is the mean of the i-th dimension across
    # all messages in the window.
    return np.mean(embeddings, axis=0)


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine distance between two embedding vectors.

    Cosine distance is defined as 1 minus the cosine similarity. It is
    preferred over Euclidean distance for high-dimensional embeddings
    because it is invariant to the magnitude of the vectors and measures
    purely directional difference, which corresponds well to semantic
    divergence.

    Parameters
    ----------
    vec1 : np.ndarray
        First vector. May be 1-D (embedding_dim,) or 2-D (1, embedding_dim).
    vec2 : np.ndarray
        Second vector. Same shape constraints as vec1.

    Returns
    -------
    float
        Cosine distance in the range [0, 2]. A value of 0 means the
        vectors are identical in direction; 1 means they are orthogonal
        (no semantic overlap); 2 means they are anti-parallel.

    Implementation note
    -------------------
    sklearn's cosine_similarity expects 2-D arrays. Single vectors are
    reshaped here to satisfy that requirement without modifying the
    caller's arrays (reshape returns a view, not a copy, for contiguous
    arrays).
    """
    # sklearn expects 2-D arrays; reshape 1-D inputs accordingly
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)

    # cosine_similarity returns a (1, 1) matrix; [0, 0] extracts the scalar
    similarity = cosine_similarity(vec1, vec2)[0, 0]
    return 1.0 - similarity


# ---------------------------------------------------------------------------
# Rolling window computation
# ---------------------------------------------------------------------------

def rolling_window_centroids(
    embeddings: np.ndarray,
    window_size: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide the message sequence into overlapping rolling windows and
    compute a centroid embedding for each window.

    The rolling window approach addresses the fact that individual messages
    are often too short and noisy to reveal the current topic reliably.
    By aggregating several consecutive messages into one window a more
    stable semantic representation of the local discussion context is
    obtained.

    For a sequence of N messages with window_size W and stride S, the
    number of complete windows is: floor((N - W) / S) + 1.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_messages, embedding_dim). Messages must already be sorted
        chronologically before this function is called; the sort is
        performed by validate_and_sort() in io.py.
    window_size : int
        The number of consecutive messages to include in each window.
        Larger values produce smoother, less reactive centroids; smaller
        values are more sensitive to local topic changes.
    stride : int, optional
        The step size between successive window start positions. A stride
        of 1 (default) produces maximally overlapping windows, giving the
        finest temporal resolution. A stride equal to window_size produces
        non-overlapping (tumbling) windows.

    Returns
    -------
    centroids : np.ndarray
        Shape (n_windows, embedding_dim). Each row is the centroid of the
        corresponding window.
    window_indices : np.ndarray
        Shape (n_windows, 2). Each row [start_idx, end_idx] records the
        half-open message index range [start_idx, end_idx) of the window.
        These indices are later used by map_window_change_points_to_messages
        in cpd.py to translate change points from window space back to the
        original message index space.

    Notes
    -----
    If window_size > n_messages, a single window spanning the entire
    sequence is returned. This prevents the function from raising an
    error on very small datasets, though the downstream analysis will
    produce a trivial (essentially zero) drift signal in that case.

    Known issue / parameter sensitivity
    ------------------------------------
    The window_size parameter has a significant effect on detection
    sensitivity. A value that is too small (e.g. 2-3 messages) causes the
    drift signal to be very noisy because individual message embeddings can
    vary substantially even within the same topic. A value that is too
    large (e.g. > 20 messages) smooths out genuine topic boundaries and
    may cause them to be missed entirely. Empirical testing on the
    synthetic dataset (see io.py :: generate_sample_data) showed that
    values in the range 6-12 provided the best balance.
    """
    n_messages = embeddings.shape[0]

    # Edge case: window larger than the available data. Return a single window
    # covering the entire sequence so the pipeline can continue gracefully
    # without raising an IndexError.
    if window_size > n_messages:
        single_centroid = embeddings.mean(axis=0, keepdims=True)  # shape (1, dim)
        single_indices = np.array([[0, n_messages]])               # shape (1, 2)
        return single_centroid, single_indices

    centroids = []
    window_indices = []

    # Iterate over all valid start positions for a window of the given size.
    # The upper bound ensures the last window is always complete (no padding).
    for start_idx in range(0, n_messages - window_size + 1, stride):
        end_idx = start_idx + window_size           # exclusive end index

        window_embeddings = embeddings[start_idx:end_idx]   # (window_size, dim)
        centroid = compute_centroid(window_embeddings)       # (dim,)

        centroids.append(centroid)
        window_indices.append([start_idx, end_idx])

    return np.array(centroids), np.array(window_indices)


# ---------------------------------------------------------------------------
# Drift signal computation
# ---------------------------------------------------------------------------

def compute_drift_signal(centroids: np.ndarray) -> np.ndarray:
    """
    Compute the temporal drift signal as the sequence of cosine distances
    between consecutive window centroids.

    Each element drift[i] = cosine_distance(centroids[i], centroids[i+1])
    quantifies how much the semantic focus of the conversation shifted
    between window i and window i+1. A sudden spike in this signal is a
    strong indicator of a topic shift.

    Parameters
    ----------
    centroids : np.ndarray
        Shape (n_windows, embedding_dim). Output of rolling_window_centroids.
        Must contain at least one row.

    Returns
    -------
    drift : np.ndarray
        Shape (n_windows - 1,). The i-th element is the cosine distance
        between centroid i and centroid i+1.

    Special case
    ------------
    If only one window is available (n_windows == 1), no comparison is
    possible. A single zero value [0.0] is returned to maintain a
    non-empty array contract expected by apply_ewma and the change-point
    detectors.
    """
    if len(centroids) < 2:
        # Cannot compute pairwise distance with fewer than two centroids.
        return np.array([0.0])

    drift = []
    for i in range(1, len(centroids)):
        # Compare each centroid to the one immediately preceding it.
        dist = cosine_distance(centroids[i - 1], centroids[i])
        drift.append(dist)

    return np.array(drift)


# ---------------------------------------------------------------------------
# Signal smoothing
# ---------------------------------------------------------------------------

def apply_ewma(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply Exponential Weighted Moving Average (EWMA) smoothing to a
    1-D signal.

    EWMA is a causal (one-pass) filter that assigns exponentially
    decreasing weights to older observations. It is preferred over a
    simple moving average because it does not introduce a fixed lag and
    adapts instantly when the signal level changes.

    The recurrence relation is:

        s_0 = x_0
        s_t = alpha * x_t + (1 - alpha) * s_{t-1},  for t >= 1

    Parameters
    ----------
    signal : np.ndarray
        1-D array of raw drift values. An empty array is returned unchanged.
    alpha : float, optional
        Smoothing factor in the range (0, 1]. Values close to 0 produce
        a heavily smoothed signal; values close to 1 approximate the raw
        signal with almost no smoothing. Default is 0.3.

    Returns
    -------
    smoothed : np.ndarray
        Shape identical to signal. The smoothed drift values.

    Known issue / parameter sensitivity
    ------------------------------------
    During development it was observed that alpha values below 0.1 caused
    genuine topic-shift spikes to be damped to the point where the
    change-point detector could no longer identify them. Values above 0.7
    preserved most of the noise from individual message-level embedding
    variation, leading to spurious change points. The interface exposes
    alpha as a user-adjustable slider (see app.py, line ~70) to allow
    dataset-specific tuning.
    """
    if len(signal) == 0:
        return signal   # return the empty array unchanged to avoid errors

    smoothed = np.zeros_like(signal, dtype=float)
    smoothed[0] = signal[0]    # initialise with the first raw observation

    for i in range(1, len(signal)):
        # Weighted blend of the current observation and the previous smoothed value
        smoothed[i] = alpha * signal[i] + (1.0 - alpha) * smoothed[i - 1]

    return smoothed


# ---------------------------------------------------------------------------
# Combined pipeline entry point
# ---------------------------------------------------------------------------

def compute_drift_with_smoothing(
    embeddings: np.ndarray,
    window_size: int,
    stride: int = 1,
    ewma_alpha: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute the full temporal embedding drift pipeline in a single call.

    This function chains rolling_window_centroids, compute_drift_signal,
    and apply_ewma and is the primary entry point called by the Streamlit
    application (app.py, lines ~155-160).

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n_messages, embedding_dim). Chronologically ordered message
        embeddings produced by src/embeddings.py :: embed_texts().
    window_size : int
        Number of messages per rolling window.
    stride : int, optional
        Step size between successive window start positions. Default is 1.
    ewma_alpha : float, optional
        EWMA smoothing factor. Default is 0.3.

    Returns
    -------
    drift_smoothed : np.ndarray
        Shape (n_windows - 1,). The smoothed drift signal ready for
        change-point detection.
    centroids : np.ndarray
        Shape (n_windows, embedding_dim). Window centroids. Returned so
        that the multivariate change-point detector in cpd.py can operate
        directly on the centroid sequence without recomputing it.
    window_indices : np.ndarray
        Shape (n_windows, 2). Window boundary indices in message space.
        Required by map_window_change_points_to_messages() in cpd.py.
    """
    # Step 1: Aggregate messages into overlapping windows and compute centroids
    centroids, window_indices = rolling_window_centroids(
        embeddings, window_size, stride
    )

    # Step 2: Compute raw drift as cosine distance between consecutive centroids
    drift_raw = compute_drift_signal(centroids)

    # Step 3: Apply EWMA smoothing to suppress short-lived noise spikes
    drift_smoothed = apply_ewma(drift_raw, alpha=ewma_alpha)

    return drift_smoothed, centroids, window_indices
