# GestureBall — CV Pipeline

Real-time foot gesture recognition for a multiplayer football game.  
Focus: computer vision only (gesture detection → game event).

---

## Project structure

```
gestureball_cv/
├── pipeline.py            # Pose extraction, feature engineering, benchmarker
├── benchmark.py           # Compare all methods; produce latency/accuracy plots
├── collect_data.py        # Interactive data labelling + SVM baseline
├── classifiers/
│   ├── rule_based.py      # Threshold classifier (baseline)
│   ├── cnn_classifier.py  # Single-frame CNN on pose landmarks
│   └── lstm_classifier.py # Temporal GRU with attention pooling
├── models/                # Saved .pt / .pkl weights (git-ignored)
├── data/
│   ├── cnn/               # Single-frame .npz samples
│   └── lstm/              # 30-frame sequence .npz samples
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Workflow

### 1. Run the live pipeline (rule-based, no training needed)
```bash
python pipeline.py
```
Press **B** to dump FPS/latency breakdown to `benchmark_results.json`.

### 2. Collect labelled data
```bash
python collect_data.py
```
Keys: `S` shoot · `P` pass · `M` move · `D` defend · `I` idle · `R` save LSTM sequence

Collect ≥200 samples per class for CNN; ≥100 sequences per class for LSTM.

### 3. Train models (run on Colab/Kaggle GPU)
```python
# CNN
from classifiers.cnn_classifier import train as train_cnn
train_cnn("data/cnn", epochs=50)

# LSTM
from classifiers.lstm_classifier import train as train_lstm
train_lstm("data/lstm", epochs=80)

# SVM (CPU, fast)
python collect_data.py --svm
```

### 4. Benchmark all methods
```bash
python benchmark.py --video data/test_video.mp4 --labels data/labels.csv --cnn --lstm
```
Outputs:
- `accuracy_vs_latency.png`
- `confusion_rule_based.png`, `confusion_cnn.png`, `confusion_lstm.png`
- `failure_cases/` — frames where classifiers fail
- `benchmark_summary.json`

---

## Reviewer questions answered

### Q: Pass vs Shoot distinction?
`classifiers/rule_based.py` — `_classify()`:
- **SHOOT**: high swing velocity + large ankle-to-hip amplitude + forward ankle motion (forward ratio > 0.65)
- **PASS**: moderate velocity + lateral motion dominant (lateral ratio > 0.40)
- CNN/LSTM learn this boundary from data — expected to outperform rules significantly here

### Q: Running direction?
`pipeline.py` — `FeatureEngineer.update()`:
- Hip centroid = (left_hip + right_hip) / 2
- Displacement vector between consecutive frames → `_direction_label()` maps to LEFT/RIGHT/UP/DOWN
- No threshold ambiguity: direction emerges from actual body movement, not a posed stance

### Q: FPS / latency analysis?
`pipeline.py` — `Benchmarker` class:
- Per-stage timing: pose extraction, feature engineering, classification
- `benchmark.py` produces `accuracy_vs_latency.png`
- Expected rough order: Rule-based ~1ms · CNN ~8ms · LSTM ~15ms · all within 40ms budget

### Q: Pose estimation not smooth?
`pipeline.py` — `LandmarkSmoother`:
- EMA smoother with tunable α (default 0.65)
- Applied before feature engineering
- LSTM naturally handles jitter via temporal context

### Q: Failure case analysis?
`benchmark.py` — `FailureAnalyser`:
- Records frames with: classifier disagreement, misclassification, occlusion, low confidence
- Saves annotated JPEG gallery to `failure_cases/`
- Known failure modes (see `rule_based.py::FAILURE_MODES`):
  - Low MediaPipe confidence in poor lighting
  - Shoot/pass confusion when kick is at ~45° (ambiguous axis)
  - Direction ambiguity when player pivots without translating

---

## Evaluation targets

| Metric              | Target       |
|---------------------|-------------|
| Gesture accuracy    | ≥ 85%        |
| End-to-end latency  | < 40ms       |
| Throughput          | ≥ 25 FPS     |
| Shoot/Pass F1       | ≥ 0.80 each  |
| Failure rate        | < 15% frames |
