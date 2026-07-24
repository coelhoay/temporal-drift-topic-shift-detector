"""
io.py
-----
Data loading, validation, preprocessing, and synthetic data generation for
the Temporal Drift Topic Shift Detector.

This module forms the entry point of the data pipeline. It is responsible for:
1. Reading CSV files uploaded by the user (load_csv).
2. Parsing timestamps from a wide range of common formats (parse_timestamps).
3. Validating data integrity and sorting messages chronologically
   (validate_and_sort).
4. Computing basic descriptive statistics for the UI header metrics
   (get_data_stats).
5. Generating synthetic conversation data with known topic shifts for
   demonstration and testing purposes (generate_sample_data).
6. Computing a stable hash of the message list for cache-key purposes
   (hash_texts).

Expected CSV schema
-------------------
The system requires three columns in the input file:

    timestamp  : str / datetime  - when the message was posted
    author     : str             - name or identifier of the author
    text       : str             - the content of the message

Additional columns are accepted and passed through unchanged.

Design decisions
----------------
Aggressive text cleaning (lowercasing, stemming, punctuation removal) is
deliberately omitted. The SentenceTransformer models used in embeddings.py
are trained on natural text and produce better representations from unaltered
input. The only text filtering performed here is removal of rows with entirely
empty or whitespace-only text strings (validate_and_sort), which would
otherwise cause the embedding model to produce degenerate zero vectors.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import hashlib


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(file_path) -> pd.DataFrame:
    """
    Load a CSV file and verify that the required columns are present.

    The function accepts either a file path string or a file-like object
    (such as the UploadedFile returned by Streamlit's file_uploader widget)
    because pandas read_csv accepts both.

    Parameters
    ----------
    file_path : str or file-like object
        Path to the CSV file, or a file-like object opened in binary or text
        mode. In the Streamlit application (app.py, line ~43), this is an
        st.UploadedFile instance.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with at least the columns "timestamp", "author", and
        "text". Column data types are not yet converted at this stage;
        timestamp parsing is deferred to parse_timestamps().

    Raises
    ------
    ValueError
        If one or more of the required columns are absent from the CSV file.
        The error message lists all missing columns to help users correct
        their file format.
    """
    df = pd.read_csv(file_path)

    required_cols = ["timestamp", "author", "text"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"The CSV file must contain columns: {required_cols}."
        )

    return df


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def parse_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Parse the timestamp column from string to pandas Timestamp objects.

    Online discussion exports use a wide variety of timestamp formats
    depending on the platform. This function attempts parsing against a
    prioritised list of common formats and falls back to pandas' automatic
    parser if none match, with failed parses producing NaT values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the timestamp column.
    timestamp_col : str, optional
        Name of the column containing timestamps. Default is "timestamp".

    Returns
    -------
    pd.DataFrame
        Copy of df with the timestamp column converted to pandas Timestamp
        dtype (datetime64[ns] or datetime64[ns, tz]).

    Known issue
    -----------
    The loop-based format-trying approach raises and silently catches
    ValueError/TypeError exceptions for each non-matching format. For very
    large DataFrames this can be slow if most format attempts fail. A more
    efficient approach would be to inspect the first non-null value to infer
    the format. This is identified as a candidate for future optimisation.
    """
    df = df.copy()

    # Ordered list of timestamp format strings to attempt, from most to least
    # specific. ISO 8601 variants are listed first as they are most common in
    # modern discussion platforms.
    formats = [
        "%Y-%m-%d %H:%M:%S",       # e.g. 2024-01-15 09:30:00
        "%Y-%m-%dT%H:%M:%S",       # ISO 8601 without timezone
        "%Y-%m-%d %H:%M:%S.%f",    # with microseconds
        "%Y-%m-%dT%H:%M:%S.%f",    # ISO 8601 with microseconds
        "%Y-%m-%d",                 # date only
        "%m/%d/%Y %H:%M:%S",       # US date format
        "%d/%m/%Y %H:%M:%S",       # European date format
        "%Y-%m-%d %H:%M:%S%z",     # with UTC offset
        "%Y-%m-%dT%H:%M:%S%z",     # ISO 8601 with UTC offset
    ]

    if df[timestamp_col].dtype == "object":
        # Column is stored as strings; try each format in order
        parsed = None
        for fmt in formats:
            try:
                parsed = pd.to_datetime(df[timestamp_col], format=fmt, errors="raise")
                break   # stop at the first successful format
            except (ValueError, TypeError):
                continue

        if parsed is None:
            # None of the explicit formats matched; fall back to pandas'
            # general-purpose parser. errors="coerce" converts unparseable
            # values to NaT rather than raising an exception.
            parsed = pd.to_datetime(df[timestamp_col], errors="coerce")

        df[timestamp_col] = parsed
    else:
        # Column already has a numeric or datetime dtype; convert automatically
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Validation and sorting
# ---------------------------------------------------------------------------

def validate_and_sort(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Validate data integrity and sort messages into chronological order.

    The temporal drift analysis is strictly dependent on the messages being
    in chronological order. This function removes rows that cannot be used
    in the analysis (missing timestamps, empty text) and sorts the remaining
    rows by their timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the timestamp column already parsed by parse_timestamps().
    timestamp_col : str, optional
        Name of the timestamp column. Default is "timestamp".

    Returns
    -------
    pd.DataFrame
        Cleaned and chronologically sorted DataFrame with a contiguous integer
        index reset from 0.

    Notes
    -----
    The reset_index(drop=True) call ensures that the integer index of the
    returned DataFrame is contiguous from 0. This is required by the segment
    slicing operations in cpd.py :: create_segments() and
    labeling.py :: label_segments(). Without this reset, iloc-based slicing
    could produce incorrect results if the original DataFrame had a
    non-contiguous index after row removal.
    """
    df = df.copy()

    # Step 1: Remove rows where the timestamp could not be parsed (NaT values)
    df = df.dropna(subset=[timestamp_col])

    # Step 2: Remove rows where the text content is empty or whitespace-only.
    # These produce degenerate embeddings that create spurious drift spikes.
    df = df[df["text"].astype(str).str.strip() != ""]

    # Step 3: Sort by timestamp ascending (oldest message first).
    # Messages with identical timestamps retain their original relative order.
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Descriptive statistics for the UI
# ---------------------------------------------------------------------------

def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Compute basic descriptive statistics about the loaded dataset for
    display in the Streamlit metric widgets (app.py, lines ~118-127).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and sorted DataFrame produced by validate_and_sort().

    Returns
    -------
    dict with keys:
        "n_messages"    : int   - total number of messages
        "n_authors"     : int   - number of unique authors
        "time_range"    : dict  - "start" and "end" Timestamps
        "duration_days" : float - span of the discussion in days
    """
    return {
        "n_messages": len(df),
        "n_authors": df["author"].nunique() if "author" in df.columns else 0,
        "time_range": {
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max(),
        },
        "duration_days": (
            df["timestamp"].max() - df["timestamp"].min()
        ).total_seconds() / 86400,
    }


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_sample_data(n_messages: int = 100) -> pd.DataFrame:
    """
    Generate a synthetic conversation dataset with known topic shifts for
    demonstration and testing purposes.

    The generated data simulates a realistic multi-topic online discussion
    by cycling through five topic themes (Weather, Technology, Food, Sports,
    Travel) and assigning messages to each theme with added noise (30 % of
    messages in each segment are generic, topic-neutral messages). This noise
    approximates the real-world phenomenon of off-topic or transitional
    messages at segment boundaries.

    Parameters
    ----------
    n_messages : int, optional
        Total number of messages to generate. Default is 100. Must be at
        least 20 to accommodate the minimum shift margin.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns "timestamp", "author", and "text", sorted
        chronologically. A random seed (42) is set at the start to ensure
        reproducibility across runs.

    Notes
    -----
    The np.random.seed(42) call at the top of the function fixes the random
    state for the entire function, making the output deterministic for a
    given n_messages value. This is important for reproducibility of results
    described in the evaluation section of the report.

    The generated data does not reflect the full complexity of real-world
    discussions. Real discussions typically contain more nuanced topic
    overlaps, sarcasm, abbreviations, and cross-references. Evaluation on
    real datasets is therefore recommended to supplement conclusions drawn
    from the synthetic data.
    """
    np.random.seed(42)  # Fix random state for reproducibility

    topics = [
        {
            "name": "Weather",
            "keywords": ["weather", "rain", "sunny", "temperature", "clouds", "forecast", "storm"],
            "messages": ["The weather is nice today", "It might rain later", "Temperature is rising"]
        },
        {
            "name": "Technology",
            "keywords": ["computer", "software", "code", "algorithm", "programming", "debug", "API"],
            "messages": ["I love programming", "The new API is great", "Let me debug this code"]
        },
        {
            "name": "Food",
            "keywords": ["pizza", "restaurant", "cooking", "recipe", "delicious", "taste", "meal"],
            "messages": ["This pizza is amazing", "I tried a new recipe", "The restaurant was good"]
        },
        {
            "name": "Sports",
            "keywords": ["football", "game", "team", "score", "player", "match", "victory"],
            "messages": ["Great game yesterday", "Our team won", "The player scored a goal"]
        },
        {
            "name": "Travel",
            "keywords": ["vacation", "trip", "hotel", "flight", "beach", "sightseeing", "journey"],
            "messages": ["Planning a vacation", "The hotel was nice", "Beautiful beach views"]
        },
    ]

    # Randomly determine the number and positions of topic shifts.
    # The margin of 20 messages ensures no shift point falls too close to the
    # start or end of the sequence, giving each segment a minimum size.
    n_shifts = np.random.randint(3, 6)
    shift_points = sorted(
        np.random.choice(n_messages - 20, size=n_shifts, replace=False)
    )
    shift_points = [0] + shift_points + [n_messages]

    messages = []
    timestamps = []
    author_list = []
    author_names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

    base_time = datetime(2024, 1, 1, 10, 0, 0)

    for i in range(len(shift_points) - 1):
        start_idx = shift_points[i]
        end_idx = shift_points[i + 1]
        topic = topics[i % len(topics)]
        segment_length = end_idx - start_idx

        for j in range(segment_length):
            # 70 % topic-specific, 30 % generic noise
            if np.random.random() < 0.7:
                msg_template = np.random.choice(topic["messages"])
                msg = msg_template + " " + np.random.choice(["!", ".", "?"])
            else:
                msg = f"Message {start_idx + j} in conversation."

            messages.append(msg)
            # Each message is one minute after the previous
            timestamps.append(base_time + pd.Timedelta(minutes=start_idx + j))
            author_list.append(np.random.choice(author_names))

    df = pd.DataFrame({
        "timestamp": timestamps[:n_messages],
        "author": author_list[:n_messages],
        "text": messages[:n_messages]
    })

    return df


# ---------------------------------------------------------------------------
# Cache key utility
# ---------------------------------------------------------------------------

def hash_texts(texts: list) -> str:
    """
    Compute an MD5 hex digest of the message list for use as a Streamlit
    cache key in embeddings.py :: embed_texts().

    Streamlit's @st.cache_data decorator requires all function arguments to
    be hashable. Lists of strings are not reliably hashable by Streamlit's
    internal mechanism for large inputs, so this function converts the list
    to a single joined string and hashes it with MD5.

    Parameters
    ----------
    texts : list
        List of message strings to hash.

    Returns
    -------
    str
        32-character hexadecimal MD5 digest. Identical lists of messages
        will always produce the same digest, ensuring correct cache hits.

    Security note
    -------------
    MD5 is used here solely as a fast, non-cryptographic content hash for
    cache-key purposes, not for security. The known collision vulnerabilities
    of MD5 are not relevant in this context because the texts are trusted
    application data, not adversarial input.
    """
    text_str = "\n".join(str(t) for t in texts)
    return hashlib.md5(text_str.encode()).hexdigest()
