"""
labeling.py
-----------
Topic segment labelling and keyword extraction module.

Once the change-point detection pipeline (cpd.py) has partitioned the
conversation into segments, this module assigns human-readable labels to
each segment. Labelling is performed in two steps:

1. Keyword extraction: a small set of high-salience words is extracted
   from the segment's messages to characterise its topic.
2. Representative message selection: a single message from the segment
   that best exemplifies its dominant theme is selected as a summary.

Two keyword extraction back-ends are supported:

TF-IDF (default)
    Term Frequency - Inverse Document Frequency weighting scores words
    that are frequent within a segment but rare across other segments.
    This naturally surfaces topic-specific vocabulary. The implementation
    uses scikit-learn's TfidfVectorizer and requires no additional
    dependencies beyond the core requirements file.

BERTopic (optional)
    BERTopic (Grootendorst, 2022) uses the segment embeddings to identify
    coherent topic clusters and extracts representative keywords via a
    class-based TF-IDF variant (c-TF-IDF). This approach typically
    produces higher-quality keywords but requires the bertopic package,
    which adds a substantial dependency and is slower to run. BERTopic
    is therefore optional and disabled by default in the Streamlit sidebar
    (app.py, line ~96).

Design decisions
----------------
The two-step approach (keywords + representative message) was chosen over
a single-step summarisation approach because it provides complementary
information: keywords characterise the segment's vocabulary whilst the
representative message provides concrete context. Together they give the
analyst enough information to understand a segment's topic without reading
every message in it.

Module dependencies
-------------------
- scikit-learn  : TfidfVectorizer (mandatory)
- bertopic      : BERTopic.transform() (optional; imported lazily inside
                  label_segments to avoid ImportError when not installed)
- numpy         : array operations for centroid computation
- pandas        : DataFrame slicing for timestamp extraction
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from collections import Counter
import re


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def extract_keywords_tfidf(
    texts: List[str],
    n_keywords: int = 5,
    stop_words: Optional[List[str]] = None
) -> List[str]:
    """
    Extract the most salient keywords from a list of texts using
    Term Frequency - Inverse Document Frequency (TF-IDF) weighting.

    TF-IDF scores each word by how frequently it appears in the current
    segment (TF) relative to how commonly it appears across all segments
    (IDF). Words that appear often in one segment but rarely in others
    receive a high score and are therefore good topic descriptors.

    Parameters
    ----------
    texts : List[str]
        The list of message strings belonging to a single topic segment.
        Must not be empty.
    n_keywords : int, optional
        Maximum number of keywords to return. Default is 5.
    stop_words : List[str] or None, optional
        List of words to exclude from consideration. If None, a built-in
        list of common English function words is used. Providing a custom
        list allows domain-specific stop words to be excluded.

    Returns
    -------
    keywords : List[str]
        List of up to n_keywords high-salience words, ordered by
        decreasing TF-IDF score.

    Fallback behaviour
    ------------------
    If TfidfVectorizer raises an exception (e.g., because all messages
    in the segment contain only stop words, which can happen with very
    short or generic segments), the function falls back to simple word
    frequency counting. The fallback ensures that the labelling pipeline
    never crashes, even on degenerate input.

    Known issue
    -----------
    TF-IDF treats each segment independently and does not consider the
    semantic relationship between synonyms. Two segments that discuss the
    same topic using different vocabulary (e.g., 'automobile' vs. 'car')
    may receive entirely different keyword sets. BERTopic-based extraction
    (label_segments, use_bertopic=True) partially addresses this by
    operating in embedding space, but at the cost of additional complexity
    and computation time.

    Implementation note
    -------------------
    The token_pattern r'\b[a-z]+\b' restricts vocabulary to purely
    alphabetic lowercase words. Numbers, punctuation, hashtags, and
    mixed-case tokens (after lowercasing) are excluded because they rarely
    serve as useful topic descriptors in conversational data. max_features
    is set to n_keywords * 2 to give the vectorizer a slightly wider
    vocabulary before the final top-n selection; this reduces the risk of
    omitting a relevant term that ranks just outside the vectorizer's
    internal vocabulary limit.
    """
    if len(texts) == 0:
        return []

    # Default stop words: common English function words that carry no
    # topic-specific information. These are defined here rather than using
    # sklearn's built-in 'english' stop list to provide explicit control
    # over what is excluded and to allow easy extension for domain-specific
    # corpora.
    if stop_words is None:
        stop_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        ]

    try:
        # max_features is set to n_keywords * 2 to give the vectorizer a
        # wider vocabulary from which to select the final top-n_keywords
        # words after aggregating scores across all messages in the segment.
        vectorizer = TfidfVectorizer(
            max_features=n_keywords * 2,
            stop_words=stop_words,
            lowercase=True,
            # token_pattern restricts tokens to purely alphabetic words;
            # numbers, punctuation, and mixed tokens are excluded because
            # they rarely serve as useful topic descriptors.
            token_pattern=r'\b[a-z]+\b'
        )

        # tfidf_matrix shape: (n_messages, n_vocabulary_terms)
        tfidf_matrix = vectorizer.fit_transform(texts)

        feature_names = vectorizer.get_feature_names_out()

        # Aggregate TF-IDF scores across all messages in the segment by
        # summing columns. This gives each term a single score reflecting
        # its total salience within the segment.
        scores = tfidf_matrix.sum(axis=0).A1   # .A1 converts matrix to 1-D array

        # Select the top-n_keywords terms by descending score.
        # argsort returns ascending order; [::-1] reverses to descending.
        top_indices = scores.argsort()[-n_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices if scores[i] > 0]

        return keywords[:n_keywords]

    except Exception:
        # Fallback: plain word frequency if TfidfVectorizer fails.
        # This can occur when a segment is very short (1-2 messages) and
        # all tokens are in the stop list, causing the vectorizer to find
        # no valid vocabulary terms.
        all_words = []
        for text in texts:
            # Extract purely alphabetic tokens and apply stop-word filtering
            words = re.findall(r'\b[a-z]+\b', text.lower())
            words = [w for w in words if w not in stop_words and len(w) > 2]
            all_words.extend(words)

        word_counts = Counter(all_words)
        return [word for word, _ in word_counts.most_common(n_keywords)]


# ---------------------------------------------------------------------------
# Representative message selection
# ---------------------------------------------------------------------------

def find_representative_message(
    embeddings: np.ndarray,
    segment_embeddings: np.ndarray,
    segment_texts: List[str]
) -> str:
    """
    Identify the message within a segment whose embedding is closest to
    the segment's centroid embedding.

    The centroid of a segment's embeddings represents the average semantic
    meaning of that segment. The message that lies closest to this centroid
    (in terms of cosine similarity) is the one that most completely captures
    the segment's dominant topic, making it a natural summary representative.

    Parameters
    ----------
    embeddings : np.ndarray
        The full embedding matrix for all messages (n_all_messages, dim).
        This parameter is accepted for API compatibility but is not used
        in the current implementation, which operates only on
        segment_embeddings. It is retained so that future implementations
        can consider cross-segment context when selecting representatives.
    segment_embeddings : np.ndarray
        Shape (n_segment_messages, embedding_dim). The embeddings of the
        messages belonging to this segment only.
    segment_texts : List[str]
        The raw text strings of the messages in the segment, in the same
        order as segment_embeddings.

    Returns
    -------
    str
        The text of the message whose embedding is most similar (highest
        cosine similarity) to the segment centroid. Returns an empty string
        if the segment contains no messages.

    Notes
    -----
    Using the centroid-closest message as the representative is an
    approximation. In segments where messages are clustered into two or
    more distinct sub-groups (indicating a gradual topic transition), the
    centroid may lie between the clusters and the selected representative
    may not perfectly characterise either sub-group. A more sophisticated
    approach (e.g., selecting the medoid or the highest-degree node of a
    similarity graph) is left as future work.

    The reshape(1, -1) call on the centroid is required by sklearn's
    cosine_similarity function, which expects 2-D input arrays. Without
    the reshape, a 1-D centroid vector would cause a ValueError.
    """
    if len(segment_embeddings) == 0:
        return ""

    # Compute the centroid of the segment's embeddings.
    centroid = np.mean(segment_embeddings, axis=0)  # shape (embedding_dim,)

    # Compute cosine similarity between the centroid and every message
    # embedding in the segment. reshape(1, -1) satisfies the 2-D input
    # requirement of sklearn's cosine_similarity function.
    similarities = cosine_similarity(
        centroid.reshape(1, -1),   # (1, dim)
        segment_embeddings         # (n_segment, dim)
    )[0]                           # extract the (n_segment,) row

    # Select the index of the most similar message
    closest_idx = int(np.argmax(similarities))

    return segment_texts[closest_idx] if closest_idx < len(segment_texts) else ""


# ---------------------------------------------------------------------------
# Main labelling pipeline
# ---------------------------------------------------------------------------

def label_segments(
    df: pd.DataFrame,
    segments: List[Dict],
    embeddings: np.ndarray,
    use_bertopic: bool = False,
    bertopic_model=None
) -> List[Dict]:
    """
    Annotate each detected segment with keywords and a representative message.

    This function iterates over the segment list produced by
    cpd.py :: create_segments() and enriches each segment dictionary with:
    - 'keywords'               : comma-separated string of top keywords
    - 'representative_message' : most centroid-similar message in the segment
    - 'n_messages'             : message count
    - 'start_time'             : earliest timestamp in the segment
    - 'end_time'               : latest timestamp in the segment

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned, sorted message DataFrame (from io.py :: validate_and_sort).
        Must contain 'text' and 'timestamp' columns. The integer index must
        be contiguous from 0 (guaranteed by validate_and_sort's
        reset_index(drop=True) call) so that iloc-based slicing with
        start_idx / end_idx produces the correct rows.
    segments : List[Dict]
        List of segment dicts, each with 'start_idx' and 'end_idx' keys,
        as returned by cpd.py :: create_segments().
    embeddings : np.ndarray
        Shape (n_messages, embedding_dim). Message embeddings in the same
        row order as df.
    use_bertopic : bool, optional
        If True, use the pre-fitted BERTopic model for keyword extraction
        instead of TF-IDF. Default is False.
    bertopic_model : BERTopic or None, optional
        A fitted BERTopic model instance. Required when use_bertopic=True.
        Ignored if use_bertopic=False.

    Returns
    -------
    labeled_segments : List[Dict]
        Copies of the input segment dicts, each augmented with the keys
        described above. The original dicts are not modified (seg.copy()
        is used to avoid mutating the input list).

    Notes on BERTopic integration
    ------------------------------
    When use_bertopic=True, the function calls bertopic_model.transform()
    on each segment's texts to obtain topic assignments. The dominant topic
    (mode of the assigned topic IDs) is selected and its top-5 keywords
    are used as the segment label. If BERTopic assigns all messages to the
    outlier topic (-1), or if transform() raises an exception, the function
    silently falls back to TF-IDF to ensure the pipeline completes
    successfully. This fallback is intentional: BERTopic can struggle with
    very short segments (fewer than ~10 messages) and may produce
    uninformative topic assignments in those cases.

    Known issue
    -----------
    The BERTopic fallback is silent (uses a bare 'except Exception' clause).
    In a production system, it would be preferable to log the exception so
    that recurring failures can be diagnosed. For the current Streamlit
    prototype, silent fallback was chosen to avoid cluttering the UI with
    technical error messages that would confuse non-technical users.

    Implementation note: iloc vs. loc
    ----------------------------------
    Segment slicing uses df.iloc[start:end] (integer-position based) rather
    than df.loc[start:end] (label based). This is correct because
    start_idx and end_idx are integer positions (0-based indices), not
    index labels. Using loc would produce incorrect results if the DataFrame
    index were non-contiguous (e.g., after filtering without reset_index).
    """
    labeled_segments = []

    for seg in segments:
        start = seg['start_idx']
        end = seg['end_idx']

        # Slice the DataFrame and embeddings to the current segment's range.
        # df.iloc uses the integer position index, which is contiguous after
        # validate_and_sort's reset_index(drop=True) call in io.py.
        segment_df = df.iloc[start:end]
        segment_texts = segment_df['text'].tolist()
        segment_embeddings = embeddings[start:end]   # (n_seg_msgs, dim)

        # --- Keyword extraction ---
        if use_bertopic and bertopic_model is not None:
            try:
                # Transform segment texts using the pre-fitted BERTopic model.
                # transform() returns (topic_ids, probabilities) for each message.
                topics, probs = bertopic_model.transform(segment_texts)

                if len(topics) > 0:
                    # Select the most frequently assigned topic in this segment.
                    dominant_topic = Counter(topics).most_common(1)[0][0]

                    # Retrieve the top keyword-score pairs for the dominant topic.
                    # get_topic() returns None for the outlier topic (-1).
                    topic_info = bertopic_model.get_topic(dominant_topic)

                    if topic_info:
                        # topic_info is a list of (word, score) tuples; extract words
                        keywords = [word for word, _ in topic_info[:5]]
                    else:
                        # Dominant topic is the outlier cluster (-1); fall back to TF-IDF.
                        # The outlier topic arises when BERTopic cannot assign messages
                        # to any coherent cluster, typically due to segment too short
                        # or vocabulary too generic.
                        keywords = extract_keywords_tfidf(segment_texts)
                else:
                    keywords = extract_keywords_tfidf(segment_texts)

            except Exception:
                # BERTopic failed (e.g., segment too short, model not fitted
                # on these texts); fall back to TF-IDF silently.
                keywords = extract_keywords_tfidf(segment_texts)
        else:
            # Default path: TF-IDF keyword extraction
            keywords = extract_keywords_tfidf(segment_texts)

        # --- Representative message selection ---
        representative = find_representative_message(
            embeddings, segment_embeddings, segment_texts
        )

        # Build the annotated segment dict (do not modify the input dict).
        # seg.copy() performs a shallow copy; all values are scalars or
        # simple Python objects so a shallow copy is sufficient.
        labeled_seg = seg.copy()
        labeled_seg['keywords'] = ', '.join(keywords) if keywords else 'N/A'
        labeled_seg['representative_message'] = representative
        labeled_seg['n_messages'] = end - start
        labeled_seg['start_time'] = segment_df['timestamp'].min()
        labeled_seg['end_time'] = segment_df['timestamp'].max()

        labeled_segments.append(labeled_seg)

    return labeled_segments
