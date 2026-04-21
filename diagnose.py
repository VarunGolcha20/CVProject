"""
diagnose.py
===========
Before trying new architectures, check whether your features
actually separate the gesture classes at all.

Runs three checks:
  1. PCA visualisation — can you see clusters per class?
  2. Per-feature discriminability — which features differ most between classes?
  3. Linear SVM accuracy — what's the ceiling for a linear classifier?
     If SVM can't do it, no neural network will either with this data.

Usage:
    python diagnose.py --cnn_dir data/cnn
    python diagnose.py --cnn_dir data/cnn --lstm_dir data/lstm
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# ── Load data ─────────────────────────────────────────────────────────────────
def load_cnn_data(data_dir: str):
    files = list(Path(data_dir).glob("*.npz"))
    X, y, stems = [], [], []
    for f in files:
        d = np.load(f)
        # Combine landmarks (flattened) + features
        feat = np.concatenate([d["landmarks"].flatten(), d["features"]])
        X.append(feat)
        y.append(int(d["label"]))
        stems.append(f.stem)
    return np.array(X), np.array(y), stems


def load_lstm_data(data_dir: str):
    files = list(Path(data_dir).glob("*.npz"))
    X, y = [], []
    for f in files:
        d = np.load(f)
        # Flatten the sequence into a single vector (mean + std over time)
        seq = d["sequence"]                      # (T, 32)
        feat = np.concatenate([seq.mean(0), seq.std(0)])   # (64,)
        X.append(feat)
        y.append(int(d["label"]))
    return np.array(X), np.array(y)


# ── Label names ───────────────────────────────────────────────────────────────
def get_label_names():
    try:
        from classifiers.cnn_classifier import IDX2GESTURE
        return IDX2GESTURE
    except Exception:
        return {0:"IDLE",1:"SHOOT",2:"PASS",3:"DEFEND",
                4:"MOVE_LEFT",5:"MOVE_RIGHT",6:"MOVE_UP",7:"MOVE_DOWN"}


# ── Check 1: Linear SVM accuracy (video-level split) ─────────────────────────
def check_svm(X, y, label_names, title="CNN features"):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report

    print(f"\n{'='*55}")
    print(f"  Linear SVM — {title}")
    print(f"{'='*55}")
    print(f"  Samples: {len(X)}  |  Classes: {sorted(set(y))}")

    # Class counts
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {label_names.get(u, u):<15} {c} samples")

    if len(set(y)) < 2:
        print("  ⚠ Only one class in data — cannot evaluate.")
        return

    min_samples = min(counts)
    n_splits    = min(5, min_samples)   # can't have more folds than samples in a class

    if n_splits < 2:
        print(f"  ⚠ Too few samples per class ({min_samples}) for cross-validation.")
        print(f"  Need at least 2 samples per class.")
        return

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="linear", C=1.0, class_weight="balanced")),
    ])

    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    print(f"\n  {n_splits}-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Interpretation
    if scores.mean() > 0.85:
        print("  ✅ Features separate classes well. Model architecture is NOT the problem.")
        print("     Issue is likely: overfitting (too little data) or train/test mismatch.")
    elif scores.mean() > 0.65:
        print("  ⚠  Partial separation. Features have signal but overlap significantly.")
        print("     Try: more data, better features, or RBF kernel SVM.")
    else:
        print("  ❌ Features cannot separate classes even with a linear classifier.")
        print("     Root cause: features don't reflect the actual gesture difference,")
        print("     OR your videos are too similar / mislabelled.")

    # Per-class report
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    target_names = [label_names.get(i, str(i)) for i in sorted(set(y))]
    print("\n  Training set classification report (overfit check):")
    print(classification_report(y, y_pred, target_names=target_names,
                                labels=sorted(set(y)), zero_division=0))
    return pipe, scores.mean()


# ── Check 2: PCA visualisation ────────────────────────────────────────────────
def check_pca(X, y, label_names, title="CNN features", save_path=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    classes  = sorted(set(y))
    colors   = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, col in zip(classes, colors):
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=label_names.get(cls, str(cls)),
                   color=col, alpha=0.6, s=30)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title(f"PCA — {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = save_path or f"pca_{title.replace(' ','_').lower()}.png"
    plt.savefig(path, dpi=150)
    print(f"\n  PCA plot saved → {path}")

    # Interpretation
    total_var = sum(var)
    print(f"  PC1+PC2 explain {total_var*100:.1f}% of variance")
    if total_var < 0.3:
        print("  ⚠ Very low variance explained — data may be high-dimensional noise")
    else:
        print("  Check the plot: if class clusters overlap heavily → features need work")


# ── Check 3: Per-feature discriminability ─────────────────────────────────────
def check_feature_importance(X, y, feature_names, title="features"):
    """
    Use a Random Forest to rank which features matter most.
    Shows you which of the 32 engineered features are actually being used.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    if len(set(y)) < 2:
        return

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X[:, :len(feature_names)])   # only engineered features

    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                class_weight="balanced")
    rf.fit(X_s, y)
    importances = rf.feature_importances_

    # Top 15 features
    top_idx  = np.argsort(importances)[::-1][:15]
    top_imp  = importances[top_idx]
    top_names= [feature_names[i] if i < len(feature_names) else f"f{i}"
                for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(top_idx)), top_imp[::-1], color="steelblue")
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature importance (Random Forest)")
    ax.set_title(f"Top 15 discriminating {title}")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = f"feature_importance_{title.replace(' ','_')}.png"
    plt.savefig(path, dpi=150)
    print(f"  Feature importance plot saved → {path}")

    print(f"\n  Top 5 most discriminating features:")
    for name, imp in zip(top_names[:5], top_imp[:5]):
        print(f"    {name:<35} {imp:.4f}")

    # Check if new features are pulling weight
    new_feature_names = [
        "left_foot_forward_ratio", "right_foot_forward_ratio",
        "left_foot_lateral_offset", "right_foot_lateral_offset",
        "left_knee_extension_rate", "right_knee_extension_rate",
        "total_kinetic_energy", "hip_stability",
        "moving_joint_count", "upper_body_motion",
        "body_lean_forward", "body_lean_lateral",
        "planted_foot", "left_ankle_accel",
        "right_ankle_accel", "stance_width",
    ]
    new_in_top5 = [n for n in top_names[:5] if n in new_feature_names]
    if new_in_top5:
        print(f"\n  ✅ New features in top 5: {new_in_top5}")
    else:
        print(f"\n  ⚠ None of the new features appear in top 5 — original features dominate")


# ── Feature names for the 32-dim vector ──────────────────────────────────────
FEATURE_NAMES_32 = [
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_ankle_vel_x", "left_ankle_vel_y",
    "right_ankle_vel_x", "right_ankle_vel_y",
    "body_facing",
    "hip_disp_x", "hip_disp_y",
    "left_swing_amp", "right_swing_amp",
    "left_swing_vel", "right_swing_vel",
    "mean_visibility",
    # New
    "left_foot_forward_ratio", "right_foot_forward_ratio",
    "left_foot_lateral_offset", "right_foot_lateral_offset",
    "left_knee_extension_rate", "right_knee_extension_rate",
    "total_kinetic_energy", "hip_stability",
    "moving_joint_count", "upper_body_motion",
    "body_lean_forward", "body_lean_lateral",
    "planted_foot",
    "left_ankle_accel", "right_ankle_accel",
    "stance_width",
]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_dir",  default="data/cnn")
    parser.add_argument("--lstm_dir", default="data/lstm")
    args = parser.parse_args()

    label_names = get_label_names()

    print("\n" + "="*55)
    print("  GestureBall — Feature Diagnosis")
    print("="*55)

    # ── CNN features ──────────────────────────────────────────
    print(f"\nLoading CNN data from {args.cnn_dir}...")
    X_cnn, y_cnn, stems = load_cnn_data(args.cnn_dir)
    print(f"  Loaded {len(X_cnn)} samples, {X_cnn.shape[1]} features")

    check_svm(X_cnn, y_cnn, label_names, title="CNN features")

    # Only use the 32 engineered features (not raw landmarks) for importance
    n_lm = 10 * 4   # 10 keypoints × 4 values = 40
    X_eng = X_cnn[:, n_lm:]   # the engineered 32 features
    check_feature_importance(X_eng, y_cnn, FEATURE_NAMES_32, title="engineered features")
    check_pca(X_cnn, y_cnn, label_names, title="CNN features")

    # ── LSTM features ─────────────────────────────────────────
    if args.lstm_dir:
        print(f"\nLoading LSTM data from {args.lstm_dir}...")
        X_lstm, y_lstm = load_lstm_data(args.lstm_dir)
        print(f"  Loaded {len(X_lstm)} sequences")
        check_svm(X_lstm, y_lstm, label_names, title="LSTM sequence features (mean+std)")
        check_pca(X_lstm, y_lstm, label_names, title="LSTM sequence features")
