"""Temporal Drift Topic Shift Detector - Streamlit App."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import umap
import json
from datetime import datetime

from src.io import (
    load_csv, parse_timestamps, validate_and_sort, get_data_stats, 
    generate_sample_data, hash_texts
)
from src.embeddings import embed_texts
from src.drift import compute_drift_with_smoothing
from src.cpd import (
    detect_change_points_ruptures_1d, detect_change_points_ruptures_multivariate,
    detect_change_points_threshold, map_window_change_points_to_messages, create_segments
)
from src.labeling import label_segments

# Page config
st.set_page_config(
    page_title="Temporal Drift Topic Shift Detector",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Temporal Drift Topic Shift Detector")
st.markdown("Detect topic shifts in online discussions using temporal embedding drift")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'segments' not in st.session_state:
    st.session_state.segments = None
if 'drift_signal' not in st.session_state:
    st.session_state.drift_signal = None
if 'window_indices' not in st.session_state:
    st.session_state.window_indices = None
if 'change_points' not in st.session_state:
    st.session_state.change_points = None

# Sidebar - Parameters
st.sidebar.header("⚙️ Parameters")

# Data ingestion
st.sidebar.subheader("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'], help="CSV with columns: timestamp, author, text")

if st.sidebar.button("Load Sample Data"):
    with st.spinner("Generating sample data..."):
        sample_df = generate_sample_data(n_messages=100)
        st.session_state.df = sample_df
        st.session_state.embeddings = None  # Reset embeddings
        st.success("Sample data loaded!")

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        df = parse_timestamps(df)
        df = validate_and_sort(df)
        st.session_state.df = df
        st.session_state.embeddings = None  # Reset embeddings
        st.sidebar.success("Data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")

# Embedding parameters
st.sidebar.subheader("Embedding Settings")
embedding_model = st.sidebar.text_input(
    "Embedding Model",
    value="all-MiniLM-L6-v2",
    help="SentenceTransformer model name"
)

# Drift computation parameters
st.sidebar.subheader("Drift Computation")
window_size = st.sidebar.number_input(
    "Window Size",
    min_value=2,
    max_value=100,
    value=8,
    help="Number of messages per rolling window"
)
stride = st.sidebar.number_input(
    "Stride",
    min_value=1,
    max_value=20,
    value=1,
    help="Step size for rolling window"
)
ewma_alpha = st.sidebar.slider(
    "EWMA Alpha",
    min_value=0.01,
    max_value=1.0,
    value=0.3,
    step=0.01,
    help="Smoothing factor for exponential weighted moving average"
)

# Change point detection parameters
st.sidebar.subheader("Change Point Detection")
cpd_method = st.sidebar.selectbox(
    "CPD Method",
    ["ruptures_pelt_1d", "ruptures_pelt_multivariate", "threshold_only"],
    help="Method for detecting change points"
)

use_penalty = st.sidebar.checkbox("Use Penalty (instead of n_bkps)", value=True)
if use_penalty:
    penalty = st.sidebar.slider(
        "Penalty",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Penalty parameter for PELT"
    )
    n_bkps = None
else:
    n_bkps = st.sidebar.number_input(
        "Number of Breakpoints",
        min_value=1,
        max_value=50,
        value=5,
        help="Expected number of change points"
    )
    penalty = None

min_segment_length = st.sidebar.number_input(
    "Min Segment Length",
    min_value=1,
    max_value=50,
    value=2,
    help="Minimum number of messages per segment"
)

drift_threshold = st.sidebar.slider(
    "Drift Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01,
    help="Threshold for drift-based change point detection"
)

# Topic labeling
st.sidebar.subheader("Topic Labeling")
use_bertopic = st.sidebar.checkbox(
    "Use BERTopic Keywords",
    value=False,
    help="Use BERTopic for keyword extraction (slower but more accurate)"
)

# Main content
if st.session_state.df is None or len(st.session_state.df) == 0:
    st.info("👆 Please upload a CSV file or load sample data to get started.")
    st.markdown("""
    ### Expected CSV Format
    - **timestamp**: Date/time of the message (various formats supported)
    - **author**: Author name/ID
    - **text**: Message content
    
    ### Sample Data
    Click "Load Sample Data" to generate a synthetic conversation with topic shifts.
    """)
else:
    df = st.session_state.df
    
    # Show data stats
    stats = get_data_stats(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Messages", stats['n_messages'])
    with col2:
        st.metric("Authors", stats['n_authors'])
    with col3:
        st.metric("Start Time", stats['time_range']['start'].strftime('%Y-%m-%d %H:%M'))
    with col4:
        st.metric("Duration (days)", f"{stats['duration_days']:.1f}")
    
    # Check if data is too short
    if len(df) < window_size:
        st.warning(f"⚠️ Dataset has only {len(df)} messages, but window size is {window_size}. Please reduce window size or add more data.")
    else:
        # Compute embeddings
        if st.session_state.embeddings is None:
            with st.spinner("Computing embeddings..."):
                texts = df['text'].tolist()
                texts_hash = hash_texts(texts)
                embeddings = embed_texts(texts, model_name=embedding_model, texts_hash=texts_hash)
                st.session_state.embeddings = embeddings
                st.success("Embeddings computed!")
        
        embeddings = st.session_state.embeddings
        
        # Compute drift
        with st.spinner("Computing drift signal..."):
            drift_signal, centroids, window_indices = compute_drift_with_smoothing(
                embeddings, window_size=window_size, stride=stride, ewma_alpha=ewma_alpha
            )
            st.session_state.drift_signal = drift_signal
            st.session_state.window_indices = window_indices
        
        # Detect change points
        with st.spinner("Detecting change points..."):
            if cpd_method == "ruptures_pelt_1d":
                cp_window = detect_change_points_ruptures_1d(
                    drift_signal, penalty=penalty, n_bkps=n_bkps, min_segment_length=min_segment_length
                )
            elif cpd_method == "ruptures_pelt_multivariate":
                cp_window = detect_change_points_ruptures_multivariate(
                    centroids, penalty=penalty, n_bkps=n_bkps, min_segment_length=min_segment_length
                )
            else:  # threshold_only
                cp_window = detect_change_points_threshold(
                    drift_signal, threshold=drift_threshold, min_segment_length=min_segment_length
                )
            
            # Map to message space
            change_points = map_window_change_points_to_messages(cp_window, window_indices)
            st.session_state.change_points = change_points
            
            # Create segments
            segments = create_segments(len(df), change_points, min_segment_length=min_segment_length)
            
            # Label segments
            bertopic_model = None
            if use_bertopic:
                try:
                    from bertopic import BERTopic
                    with st.spinner("Fitting BERTopic model..."):
                        bertopic_model = BERTopic(verbose=False)
                        bertopic_model.fit_transform(texts)
                except Exception as e:
                    st.warning(f"BERTopic failed: {str(e)}. Using TF-IDF instead.")
                    use_bertopic = False
            
            labeled_segments = label_segments(df, segments, embeddings, use_bertopic=use_bertopic, bertopic_model=bertopic_model)
            st.session_state.segments = labeled_segments
        
        st.success(f"✅ Detected {len(change_points)} change points, creating {len(labeled_segments)} segments")
        
        # Visualizations
        st.header("📈 Visualizations")
        
        # Drift over time
        st.subheader("Drift Signal Over Time")
        
        # Create x-axis (message indices corresponding to windows)
        window_centers = window_indices[:, 0] + (window_indices[:, 1] - window_indices[:, 0]) / 2
        x_axis = window_centers[1:]  # Drift is between windows
        
        # Get timestamps for x-axis if available
        use_timestamps = df['timestamp'].notna().all()
        if use_timestamps:
            # Map window centers to approximate timestamps
            x_timestamps = []
            for center in x_axis:
                idx = int(np.clip(center, 0, len(df) - 1))
                x_timestamps.append(df.iloc[idx]['timestamp'])
            x_display = x_timestamps
            x_label = "Time"
        else:
            x_display = x_axis
            x_label = "Message Index"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_display,
            y=drift_signal,
            mode='lines',
            name='Drift Signal',
            line=dict(color='blue', width=2)
        ))
        
        # Add change points
        if len(change_points) > 0:
            for cp in change_points:
                if cp < len(df):
                    cp_time = df.iloc[cp]['timestamp'] if use_timestamps else cp
                    fig.add_vline(
                        x=cp_time,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"CP {cp}",
                        annotation_position="top"
                    )
        
        fig.update_layout(
            title="Temporal Drift Signal with Change Points",
            xaxis_title=x_label,
            yaxis_title="Drift (Cosine Distance)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment table
        st.subheader("📋 Segments")
        segment_df = pd.DataFrame([
            {
                'Segment ID': i + 1,
                'Start Index': seg['start_idx'],
                'End Index': seg['end_idx'],
                'Start Time': seg['start_time'],
                'End Time': seg['end_time'],
                '# Messages': seg['n_messages'],
                'Keywords': seg['keywords'],
                'Representative Message': seg['representative_message'][:100] + '...' if len(seg['representative_message']) > 100 else seg['representative_message']
            }
            for i, seg in enumerate(labeled_segments)
        ])
        st.dataframe(segment_df, use_container_width=True)
        
        # Conversation viewer
        st.subheader("💬 Conversation by Segment")
        for i, seg in enumerate(labeled_segments):
            with st.expander(f"Segment {i+1}: {seg['keywords']} ({seg['n_messages']} messages)"):
                segment_df_view = df.iloc[seg['start_idx']:seg['end_idx']][['timestamp', 'author', 'text']]
                st.dataframe(segment_df_view, use_container_width=True)
        
        # Embedding trajectory
        st.subheader("🗺️ Embedding Trajectory (UMAP 2D)")
        with st.spinner("Computing UMAP projection..."):
            try:
                if len(embeddings) < 2:
                    st.warning("Need at least 2 messages for UMAP visualization")
                else:
                    n_neighbors = min(15, max(2, len(embeddings) - 1))
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
                    embeddings_2d = reducer.fit_transform(embeddings)
                    
                    # Color by segment
                    segment_colors = []
                    for idx in range(len(df)):
                        for seg_idx, seg in enumerate(labeled_segments):
                            if seg['start_idx'] <= idx < seg['end_idx']:
                                segment_colors.append(seg_idx)
                                break
                        else:
                            segment_colors.append(-1)
                    
                    fig_umap = go.Figure()
                    
                    # Plot each segment separately for better visualization
                    unique_segments = sorted(set(segment_colors))
                    
                    for seg_idx in unique_segments:
                        mask = np.array(segment_colors) == seg_idx
                        if mask.sum() > 0:
                            seg_name = f"Segment {seg_idx + 1}" if seg_idx >= 0 else "Unknown"
                            fig_umap.add_trace(go.Scatter(
                                x=embeddings_2d[mask, 0],
                                y=embeddings_2d[mask, 1],
                                mode='markers',
                                name=seg_name,
                                text=[df.iloc[i]['text'][:50] + '...' if len(df.iloc[i]['text']) > 50 else df.iloc[i]['text'] 
                                      for i in np.where(mask)[0]],
                                hovertemplate='<b>%{text}</b><extra></extra>',
                                marker=dict(size=5, opacity=0.6)
                            ))
                    
                    fig_umap.update_layout(
                        title="Message Embeddings in 2D (UMAP) - Colored by Segment",
                        xaxis_title="UMAP Dimension 1",
                        yaxis_title="UMAP Dimension 2",
                        height=600,
                        hovermode='closest'
                    )
                    st.plotly_chart(fig_umap, use_container_width=True)
            except Exception as e:
                st.error(f"Error computing UMAP: {str(e)}")
        
        # Export
        st.header("💾 Export")
        
        # Add segment_id to original dataframe
        df_export = df.copy()
        segment_ids = np.zeros(len(df), dtype=int)
        for seg_idx, seg in enumerate(labeled_segments):
            segment_ids[seg['start_idx']:seg['end_idx']] = seg_idx + 1
        df_export['segment_id'] = segment_ids
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_export = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Segmented CSV",
                data=csv_export,
                file_name=f"segmented_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON summary
            summary = {
                'parameters': {
                    'embedding_model': embedding_model,
                    'window_size': window_size,
                    'stride': stride,
                    'ewma_alpha': ewma_alpha,
                    'cpd_method': cpd_method,
                    'penalty': penalty,
                    'n_bkps': n_bkps,
                    'min_segment_length': min_segment_length,
                    'drift_threshold': drift_threshold,
                    'use_bertopic': use_bertopic
                },
                'change_points': change_points.tolist(),
                'segments': [
                    {
                        'segment_id': i + 1,
                        'start_idx': seg['start_idx'],
                        'end_idx': seg['end_idx'],
                        'start_time': str(seg['start_time']),
                        'end_time': str(seg['end_time']),
                        'n_messages': seg['n_messages'],
                        'keywords': seg['keywords'],
                        'representative_message': seg['representative_message']
                    }
                    for i, seg in enumerate(labeled_segments)
                ]
            }
            json_export = json.dumps(summary, indent=2)
            st.download_button(
                label="📥 Download JSON Summary",
                data=json_export,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
