# GestureBall CV Project — Antigravity Knowledge File

> This file is maintained by Antigravity (AI assistant) for personal context.
> Last updated: 2026-04-21

---

## Project Overview

**GestureBall** is a real-time foot-gesture recognition system that feeds into a multiplayer football game. The CV pipeline detects 5 gesture classes from webcam / video using MediaPipe Pose, then classifies via three competing classifiers.

**Repo path:** `c:\Users\Mayank\OneDrive\Desktop\CVProject`  
**GitHub:** VarunGolcha20/CVProject

---

## Gesture Classes

| Label   | Description                                 |
|---------|---------------------------------------------|
| IDLE    | Player standing still, no active gesture    |
| SHOOT   | High-velocity forward kick                  |
| PASS    | Moderate velocity, lateral foot direction   |
| MOVE    | Hip centroid displacement (running)         |
| DEFEND  | Wide bent-knee stance (rule-based only)     |

*(CNN/LSTM use only IDLE, SHOOT, PASS, MOVE — 4 classes)*  
*(aggregate_results.py also has MOVE_UP/DOWN/LEFT/RIGHT and BUFFERING in its class list)*

---

## Architecture

### Pipeline (`pipeline.py`)
- **PoseExtractor**: MediaPipe Pose (model_complexity=1), EMA-smoothed landmarks (α=0.65)
- **FeatureEngineer**: Converts pose → 32-dim FeatureVector. Needs history=35 frames
- **FeatureVector** (32 dims):
  - 0–15: Angles, velocities, swing amplitude/velocity, visibility
  - 16–21: SHOOT vs PASS discriminators (forward ratio, lateral offset, knee extension rate)
  - 22–25: IDLE detectors (kinetic energy, hip stability, moving joint count, upper body motion)
  - 26–31: MOVE/lean features (body lean, planted foot, ankle accel, stance width)
- **Benchmarker**: Per-stage timing (pose, features, classify)

### Classifiers

| Classifier     | File                          | Input                  | Notes                                         |
|----------------|-------------------------------|------------------------|-----------------------------------------------|
| Rule-based     | `classifiers/rule_based.py`   | FeatureVector          | Threshold-based, 8-frame cooldown, OCCLUDED state |
| CNN (PoseCNN)  | `classifiers/cnn_classifier.py` | (33,4) landmarks + 32-dim features | Dual-branch: Conv1D on landmarks + FC on features |
| GRU/LSTM       | `classifiers/lstm_classifier.py` | (35, 32) sequence     | Bidirectional=False, 4-layer GRU, attention pooling, WINDOW=35 |
| MobileNetV2    | `classifiers/mobilenet_classifier.py` | Raw image crop | Transfer learning, frozen backbone, fine-tuned head |

### Benchmarking (`benchmark.py`)
- Runs all classifiers on same video; logs per-frame predictions vs GT from CSV
- Outputs confusion matrices (`benchmark/cm_*.png`), accuracy-vs-latency scatter, failure gallery
- **FailureAnalyser** captures: `classifier_disagreement`, `misclassification`, `low_visibility`, `low_confidence`
- **`aggregate_results.py`**: Reads multiple `failure_report.json` files → global confusion matrix

---

## Known Failure Modes (documented in rule_based.py)

| Mode               | Root Cause                                               |
|--------------------|----------------------------------------------------------|
| OCCLUDED           | Landmark visibility < 0.5                               |
| SHOOT_FALSE_POS    | Brisk walking mimics shoot velocity; needs amplitude gate |
| PASS_SHOOT_CONF    | Lateral kick ≠ pass; needs foot orientation, not velocity |
| DIRECTION_AMBIG    | Player pivots without translating; hip displacement ≈ 0  |
| LIGHTING           | Poor lighting drops MediaPipe confidence                  |

---

## Performance Targets

| Metric              | Target   |
|---------------------|----------|
| Gesture accuracy    | ≥ 85%    |
| End-to-end latency  | < 40ms   |
| Throughput          | ≥ 25 FPS |
| Shoot/Pass F1       | ≥ 0.80   |
| Failure rate        | < 15%    |

## Measured Latency (benchmark_summary.json)

| Classifier  | Mean Latency (ms) | FPS     |
|-------------|-------------------|---------|
| Rule-based  | 0.009 ms          | ~103K   |
| CNN         | 4.02 ms           | ~248    |
| LSTM/GRU    | 28.0 ms           | ~36     |

*(Accuracy columns are all 0.0 — benchmark was run without GT labels CSV)*

---

## Reviewer Feedback (current issue)

**Feedback:** "Increase the analysis of results. Analyse why a particular model is misclassifying a specific gesture more than the others."

This means the report needs **deeper per-class, per-model error analysis** beyond confusion matrices:
- Feature-level explanation of why SHOOT↔PASS confusion happens in rule-based
- Why CNN misclassifies IDLE as MOVE (no temporal context)
- Why GRU misclassifies PASS (lateral kick incorrectly mapped)
- Per-class precision/recall/F1 breakdown
- Confidence distribution histograms per class
- Attention weight visualization for GRU (which frame drives prediction)
- Feature importance analysis for rule-based thresholds
- Error case clustering by lighting/visibility/kinetic_energy

---

## Key Files

| File                       | Purpose                              |
|----------------------------|--------------------------------------|
| `pipeline.py`              | Core pose + feature extraction       |
| `benchmark.py`             | Multi-classifier evaluation          |
| `aggregate_results.py`     | Global confusion from failure JSONs  |
| `classifiers/rule_based.py`| Threshold baseline                   |
| `classifiers/cnn_classifier.py` | CNN (PoseCNN, dual-branch)      |
| `classifiers/lstm_classifier.py` | GRU with attention              |
| `classifiers/mobilenet_classifier.py` | MobileNetV2 fine-tuned   |
| `benchmark_summary.json`   | Latency summary (no accuracy yet)    |
| `confusion_rule_based.png` | Existing confusion matrix image      |

---

## Data

- **CNN data**: `data/cnn/*.npz` — single-frame samples with `landmarks`, `features`, `label`
- **LSTM data**: `data/lstm/*.npz` — 35-frame sequences with `sequence`, `label`
- **CNN images**: `cnn_images/` — raw images for MobileNet
- **Videos**: `videos.zip` (535 MB), test videos labeled via `label_gen.py`

---

## Conventions & Gotchas

1. Landmarks are **normalized [0, 1]** (MediaPipe default) — not pixel coordinates
2. `global_frame_num` in `benchmark.py` must stay in sync with `label_gen.py` for GT lookup
3. LSTM returns `"BUFFERING"` until its 35-frame window is filled
4. CNN returns `"UNCERTAIN"` if max softmax prob < 0.6; LSTM threshold is 0.65
5. `benchmark.py` only counts frames that have a GT label toward accuracy
6. MobileNet uses raw image (BGR→RGB crop), not the pose feature pipeline
7. Rule-based has 8-frame cooldown — consecutive predictions are suppressed
