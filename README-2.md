# Temporal Drift Topic Shift Detector

This project detects when the topic of an online conversation changes, using the idea that if you track how the meaning of messages shifts over time, you can spot the moment the discussion moves on to something new.

It works by turning each message into a numerical vector (an embedding), grouping messages into rolling windows, and measuring how different each window is from the one before it using cosine distance. That produces a drift signal, which is then smoothed and analysed to find the points where a genuine topic shift occurred. The result is a segmented conversation with keywords and a representative message for each segment.

The interface is a Streamlit web app — you upload a CSV, adjust a few sliders, and see the drift plot and segments update in real time.

---

## Getting started

You need Python 3.9 or later.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## What the CSV needs to look like

The file must have these three columns:

| Column | What it is |
|---|---|
| `timestamp` | When the message was sent |
| `author` | Who sent it |
| `text` | The message content |

Most common timestamp formats are handled automatically. If you do not have a real dataset, click **Load Sample Data** in the sidebar — it generates a 100-message synthetic conversation with known topic shifts that you can use to try the system out.

A sample CSV is included in `CSV file sample tests/sample_discussion.csv`.

---

## Parameters

Everything is adjustable in the sidebar without restarting the app. The main ones to know about:

**Window size** (default 8) — how many messages get grouped together before comparing. Too small and the signal is noisy; too large and genuine shifts get smoothed over. Somewhere between 6 and 12 works well in practice.

**EWMA alpha** (default 0.3) — controls how much smoothing is applied to the drift signal. Lower values smooth more aggressively, which can hide real shifts. Higher values keep more noise. Stay roughly between 0.1 and 0.7.

**PELT penalty** (default 1.0) — controls how many change points get detected. Lower values find more (risking false positives); higher values find fewer (risking missed shifts). 1.0 worked correctly on the synthetic test data.

**CPD method** — three options: `ruptures_pelt_1d` is the recommended one and works on the scalar drift signal. `ruptures_pelt_multivariate` operates on the full embedding centroids and is slower. `threshold_only` just marks anything above a fixed threshold, which is the simplest to understand but needs manual tuning.

Embeddings are cached, so changing these parameters reruns only the drift and detection steps, not the embedding model.

---

## Project structure

```
Source code/
├── app.py                         # main Streamlit app
├── requirements.txt
├── src/
│   ├── io.py                      # loading, parsing, validation, sample data
│   ├── embeddings.py              # converts messages to vectors, handles caching
│   ├── drift.py                   # rolling windows, cosine distance, EWMA
│   ├── cpd.py                     # change-point detection and segmentation
│   └── labeling.py                # keyword extraction and representative messages
├── tests/
│   ├── test_drift.py              # 7 unit tests for the drift module
│   └── test_cpd.py                # 5 unit tests for change-point detection
└── CSV file sample tests/
    └── sample_discussion.csv      # example input file
```

---

## Running the tests

```bash
pytest tests/
```

12 tests in total. They cover the windowing logic, drift signal shape, EWMA behaviour, change-point bounds, and edge cases like empty inputs or windows larger than the dataset.

---

## Embedding model

The default model is `all-MiniLM-L6-v2` from the `sentence-transformers` library. It produces 384-dimensional vectors and runs at around 14,000 sentences per second on a CPU, so no GPU is needed. The model weights (~80 MB) download automatically on first run.

You can swap it for any other model from the `sentence-transformers` library by typing the model name into the **Embedding Model** field in the sidebar.
