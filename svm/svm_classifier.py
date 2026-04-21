"""
classifiers/svm_classifier.py
==============================
RBF-kernel SVM classifier on engineered features + flattened landmarks.
Given the diagnostic showing 84.7% linear SVM accuracy, RBF SVM is the
most practical next step — handles the non-linear SHOOT/PASS boundary.

Usage:
    python classifiers/svm_classifier.py   # trains and saves model

Inference is ~0.1ms per sample — faster than CNN or LSTM.
"""

import numpy as np
import joblib
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from cnn_classifier import GESTURE_CLASSES, IDX2GESTURE, GESTURE2IDX


# ── Dataset loader with video-level grouping ──────────────────────────────────
def load_dataset(data_dir: str):
    """
    Returns X, y, groups where groups = video stem index.
    Video-level grouping ensures no video leaks across train/val.
    """
    files = list(Path(data_dir).glob("*.npz"))
    assert files, f"No .npz files in {data_dir}"

    X, y, groups = [], [], []
    stem_to_group = {}
    group_counter = 0

    for f in files:
        d    = np.load(f)
        # Combine flattened landmarks + engineered features
        feat = np.concatenate([d["landmarks"].flatten(), d["features"]])
        X.append(feat)
        y.append(int(d["label"]))

        # Video stem: drop trailing frame number
        parts = f.stem.rsplit("_", 1)
        stem  = parts[0] if len(parts) == 2 and parts[1].isdigit() else f.stem
        if stem not in stem_to_group:
            stem_to_group[stem] = group_counter
            group_counter += 1
        groups.append(stem_to_group[stem])

    return np.array(X), np.array(y), np.array(groups)


# ── Training ──────────────────────────────────────────────────────────────────
def train(data_dir: str = "data/cnn",
          save_path: str = "svm_gesture.pkl",
          tune: bool = True):
    """
    Train RBF SVM with video-level cross-validation.

    tune=True: grid search over C and gamma (takes ~2 min on 4k samples)
    tune=False: use sensible defaults, fast
    """
    print("Loading data...")
    X, y, groups = load_dataset(data_dir)
    print(f"  {len(X)} samples  |  {len(set(groups))} videos  |  "
          f"{len(set(y))} classes")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {IDX2GESTURE.get(u, u):<15} {c}")

    # Video-level CV — ensures no video in both train and val
    cv = StratifiedGroupKFold(n_splits=5)

    if tune:
        print("\nGrid search (C, gamma)...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", class_weight="balanced",
                          probability=True)),
        ])
        param_grid = {
            "svm__C":     [0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto", 0.001, 0.01],
        }
        search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro",
                      n_jobs=-1, verbose=1)
        search.fit(X, y, groups=groups)
        best = search.best_estimator_
        print(f"\n  Best params: {search.best_params_}")
        print(f"  Best CV F1 (macro): {search.best_score_:.3f}")
    else:
        print("\nTraining with default params (C=10, gamma=scale)...")
        best = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                          class_weight="balanced", probability=True)),
        ])
        # Manual video-level CV for reporting
        scores = []
        for train_idx, val_idx in cv.split(X, y, groups):
            best.fit(X[train_idx], y[train_idx])
            scores.append(best.score(X[val_idx], y[val_idx]))
        print(f"  5-fold CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        best.fit(X, y)   # final fit on all data

    # ── Evaluation on full dataset (train set check) ──────────────────────────
    y_pred = best.predict(X)
    target_names = [IDX2GESTURE.get(i, str(i)) for i in sorted(set(y))]
    print("\nClassification report (training set — shows if model can fit the data):")
    print(classification_report(y, y_pred,
                                target_names=target_names,
                                labels=sorted(set(y)),
                                zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred,
        display_labels=target_names,
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("SVM Confusion Matrix (training set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_svm.png", dpi=150)
    print("Confusion matrix saved → confusion_svm.png")

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, save_path)
    print(f"Model saved → {save_path}")
    return best


# ── Inference wrapper ──────────────────────────────────────────────────────────
class SVMClassifier:
    """
    Drop-in replacement for CNNClassifier / LSTMClassifier.
    ~0.1ms inference — fastest of all classifiers.
    """
    def __init__(self, model_path: str = "models/svm_gesture.pkl",
                 confidence_thresh: float = 0.55):
        self.model        = joblib.load(model_path)
        self.conf_thresh  = confidence_thresh
        from pipeline import KEYPOINTS
        self._kp_order    = list(KEYPOINTS.keys())

    def __call__(self, fv, raw_landmarks: dict = None) -> str:
        """
        fv: FeatureVector from FeatureEngineer
        raw_landmarks: dict of name→(x,y,z,vis) from PoseFrame
        """
        if raw_landmarks is not None:
            lm_arr = np.array(
                [raw_landmarks[k] for k in self._kp_order],
                dtype=np.float32).flatten()                  # (40,)
            feat = np.concatenate([lm_arr, fv.to_array()])  # (72,)
        else:
            feat = fv.to_array()                             # (32,) fallback

        probs = self.model.predict_proba([feat])[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])

        if conf < self.conf_thresh:
            return "UNCERTAIN"
            # return "IDLE"
        return IDX2GESTURE.get(idx, "IDLE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/cnn")
    parser.add_argument("--save",     default="models/svm_gesture.pkl")
    parser.add_argument("--no-tune",  action="store_true",
                        help="Skip grid search, use defaults (faster)")
    args = parser.parse_args()
    train(args.data_dir, args.save, tune=not args.no_tune)
