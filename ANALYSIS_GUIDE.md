# Running the Extended Analysis — Step-by-Step Guide

## Prerequisites

- `conda activate cv_env` is already running ✅
- You are in the project folder: `c:\Users\Mayank\OneDrive\Desktop\CVProject`
- Trained models exist: `models/cnn_gesture.pt`, `models/lstm_gesture.pt` ✅

---

## How It Works

```
Your videos  +  test/labels.csv
        ↓
    benchmark.py          ← runs all 3 classifiers, logs per-frame results
        ↓
  benchmark/detailed_log.json
        ↓
  analysis_extended.py    ← generates 7 analysis plots
        ↓
  benchmark/analysis/*.png
```

---

## Step 1 — Find Your Test Videos

The `test/labels.csv` was generated from a **folder of videos** (frames numbered sequentially across all files). You need to find which folder those videos are in.

**Option A — Videos are already extracted:**
```bash
dir videos\
```

**Option B — Videos are still zipped (extract first):**
```bash
python -c "import zipfile; zipfile.ZipFile('videos.zip').extractall('.')"
```

---

## Step 2 — Regenerate Labels CSV (if needed)

> Skip this if `test/labels.csv` already matches your videos exactly.

If your videos are in a folder called `videos/`, regenerate the labels so frame numbers are aligned:

```bash
python label_gen.py --folder videos\
```

This creates `videos/labels.csv` with the correct sequential frame count for every `.mp4` in that folder.

---

## Step 3 — Run the Benchmark

This runs all 3 classifiers (Rule-based, CNN, LSTM) against GT labels and saves `benchmark/detailed_log.json`.

**If your videos are in a folder (multiple .mp4 files):**
```bash
python benchmark.py --folder videos\ --labels test\labels.csv --cnn --lstm
```

**If you have a single test video:**
```bash
python benchmark.py --video videos\YOUR_VIDEO_NAME.mp4 --labels test\labels.csv --cnn --lstm
```

> A preview window will open. **Press Q** to quit early or let it finish.
>
> Expected output at the end:
> ```
> [rule_based] Accuracy: 0.712
> [cnn]        Accuracy: 0.854
> [lstm]       Accuracy: 0.891
> Detailed log saved -> benchmark/detailed_log.json
> ```

---

## Step 4 — Run the Analysis

```bash
python analysis_extended.py --log benchmark\detailed_log.json --out benchmark\analysis
```

This prints F1 tables + classification reports to the console, and saves all 7 plots to `benchmark\analysis\`.

---

## Step 5 — View Results

```bash
start benchmark\analysis\
```

---

## Full Command Sequence (Copy-Paste)

```bash
# 1. Extract videos if still zipped
python -c "import zipfile; zipfile.ZipFile('videos.zip').extractall('.')"

# 2. Regenerate labels CSV
python label_gen.py --folder videos\

# 3. Run benchmark (generates detailed_log.json)
python benchmark.py --folder videos\ --labels test\labels.csv --cnn --lstm

# 4. Run analysis (generates 7 plots)
python analysis_extended.py --log benchmark\detailed_log.json --out benchmark\analysis

# 5. Open results in Explorer
start benchmark\analysis\
```

---

## Output Files

| File | What it shows |
|------|---------------|
| `benchmark/cm_rule_based.png` | Confusion matrix — Rule-based |
| `benchmark/cm_cnn.png` | Confusion matrix — CNN |
| `benchmark/cm_lstm.png` | Confusion matrix — LSTM/GRU |
| `benchmark/analysis/1_f1_heatmap.png` | **F1 scores: all models x all gestures** |
| `benchmark/analysis/2_confidence_cnn.png` | CNN: confident wrong vs correct predictions |
| `benchmark/analysis/2_confidence_lstm.png` | LSTM: confident wrong vs correct predictions |
| `benchmark/analysis/3_gru_attention_pass.png` | Which frames GRU attends to for PASS |
| `benchmark/analysis/4_rule_boundary.png` | SHOOT/PASS threshold boundary scatter |
| `benchmark/analysis/5_cross_model_errors.png` | Unique vs shared failures across models |
| `benchmark/analysis/6_visibility_error.png` | Error rate by lighting/visibility bin |
| `benchmark/analysis/7_per_gesture_error.png` | Per-gesture error rate per model |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No labelled frames found` | Frame count mismatch. Re-run `label_gen.py --folder videos\` first. |
| `FileNotFoundError: models/cnn_gesture.pt` | Model weights missing from `models/` folder. |
| `No module named 'mediapipe'` | Run `pip install mediapipe` in your `cv_env`. |
| All plots empty / 0 data | Benchmark ran without matching GT labels. Check frame alignment. |
