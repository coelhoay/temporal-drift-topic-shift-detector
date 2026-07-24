"""
embeddings.py
-------------
Sentence embedding generation module for the Temporal Drift Topic Shift Detector.

This module converts raw text messages into dense numerical vectors (embeddings)
that capture their semantic meaning. It bridges the raw text data loaded by io.py
and the mathematical drift analysis performed by drift.py.

Model selection
---------------
The default model, all-MiniLM-L6-v2, was selected because it provides a strong
balance between embedding quality and computational efficiency:

- Embedding dimension: 384 (compact relative to larger models).
- Inference speed: approximately 14,200 sentences per second on CPU
  (Reimers and Gurevych, 2019), making it practical for interactive use
  without a GPU.
- Quality: ranks in the top tier of models of its size class on the
  Sentence Transformers SBERT Benchmark.

Users may substitute any model from the sentence-transformers library by
changing the model name in the Streamlit sidebar (see app.py, line ~57).

Caching strategy
----------------
Two levels of Streamlit caching are applied:

1. @st.cache_resource on load_embedding_model: The SentenceTransformer object
   (including its weights) is cached as a shared resource. The model is loaded
   from disk only once per Streamlit session regardless of how many times
   embed_texts is called.

2. @st.cache_data on embed_texts: The computed embedding matrix is cached keyed
   by (texts_hash, model_name). Subsequent calls with the same texts and model
   return the cached result immediately, avoiding expensive re-inference when the
   user adjusts drift or detection parameters without changing the data or model.

Known issue
-----------
Streamlit's @st.cache_data decorator cannot hash numpy arrays or lists of strings
reliably across sessions. The texts_hash parameter (an MD5 digest computed by
io.py :: hash_texts()) is therefore passed explicitly as a stable, hashable cache
key. If texts_hash is omitted, Streamlit will attempt to hash the texts list
directly, which can fail or produce incorrect cache hits for large inputs.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import streamlit as st


@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a SentenceTransformer model and cache it as a Streamlit resource.

    The @st.cache_resource decorator instructs Streamlit to keep a single shared
    instance of the model object for the lifetime of the application process.
    This avoids the overhead of loading model weights from disk on every script
    re-run, which Streamlit triggers on each user interaction.

    Parameters
    ----------
    model_name : str, optional
        The identifier of the pre-trained SentenceTransformer model to load.
        Must be a valid model name from the Hugging Face Hub or a local path.
        Examples: "all-MiniLM-L6-v2", "all-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2". Default is "all-MiniLM-L6-v2".

    Returns
    -------
    SentenceTransformer
        A fully initialised SentenceTransformer model ready for inference.

    Notes
    -----
    On the first call, the model weights are downloaded from the Hugging Face
    Hub (approximately 80 MB for the default model) and cached locally by the
    sentence-transformers library in ~/.cache/torch/sentence_transformers.
    Subsequent calls use the local cache and do not require internet access.
    SentenceTransformer automatically selects a GPU if one is available;
    otherwise it falls back to CPU with no additional configuration required.
    """
    return SentenceTransformer(model_name)


@st.cache_data
def embed_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    texts_hash: Optional[str] = None
) -> np.ndarray:
    """
    Convert a list of text strings into a matrix of sentence embeddings.

    Each message in texts is encoded independently into a fixed-length dense
    vector. The resulting matrix has one row per message and one column per
    embedding dimension. This matrix is the primary input to the drift analysis
    pipeline (drift.py :: compute_drift_with_smoothing).

    Parameters
    ----------
    texts : List[str]
        The list of message strings to embed. Messages should be in their
        original form; aggressive preprocessing (stemming, stop-word removal,
        etc.) is deliberately avoided because the SentenceTransformer model was
        trained on natural text and produces better embeddings from unaltered
        input.
    model_name : str, optional
        The SentenceTransformer model identifier. Must match the value used for
        load_embedding_model to benefit from resource caching.
    texts_hash : str or None, optional
        An MD5 hex digest of the joined texts string, computed by
        io.py :: hash_texts(). Used as the Streamlit cache key. If None,
        Streamlit will attempt to hash texts directly (unreliable for large
        lists).

    Returns
    -------
    embeddings : np.ndarray
        Shape (n_texts, embedding_dim). Each row is an embedding vector for
        the corresponding message. The embedding_dim is determined by the chosen
        model (384 for "all-MiniLM-L6-v2").

    Implementation notes
    --------------------
    - show_progress_bar=False suppresses the tqdm progress bar; the Streamlit
      spinner in app.py (lines ~150-155) provides user feedback instead.
    - convert_to_numpy=True ensures the output is a numpy array rather than a
      PyTorch tensor, which is required by the downstream numpy-based drift and
      change-point detection code.
    """
    # Retrieve the cached model (or load it if not yet cached)
    model = load_embedding_model(model_name)

    # Encode all messages in a single batched call for efficiency
    embeddings = model.encode(
        texts,
        show_progress_bar=False,   # suppress terminal progress bar
        convert_to_numpy=True      # ensure numpy output for downstream code
    )
    return embeddings
