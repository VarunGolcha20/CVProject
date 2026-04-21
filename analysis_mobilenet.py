"""
analysis_mobilenet.py
====================
Deep error analysis for the MobileNet Classifier.

Reads  : benchmark/detailed_log.json
Outputs: benchmark/analysis_mobilenet/
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support

# ── Constants ────────────────────────────────────────────────────────────────
GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "MOVE"]
MODEL_NAMES     = ["mobilenet"]
MODEL_LABELS    = {"mobilenet": "MobileNet"}
MODEL_COLORS    = {"mobilenet": "#2CA02C"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

def load_log(log_path: str) -> list:
    with open(log_path) as f: raw = json.load(f)
    labelled = [d for d in raw if d.get("gt") and d["gt"] in GESTURE_CLASSES]
    print(f"\nLoaded {len(raw)} total frames | {len(labelled)} with GT labels.")
    return labelled

def plot_per_class_f1(data: list, out_dir: Path):
    models = [m for m in MODEL_NAMES if any(m in d.get("predictions", {}) for d in data)]
    if not models: return
    f1_mat = np.zeros((len(models), len(GESTURE_CLASSES)))

    print("\n" + "─"*55 + "\n1. PER-CLASS F1 SCORES\n" + "─"*55)
    for i, model in enumerate(models):
        y_true = [d["gt"] for d in data if model in d["predictions"]]
        y_pred = [d["predictions"][model] for d in data if model in d["predictions"]]
        if not y_true: continue
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=GESTURE_CLASSES, average=None, zero_division=0)
        f1_mat[i] = f1
        print(f"\n{MODEL_LABELS[model]}")
        print(classification_report(y_true, y_pred, labels=GESTURE_CLASSES, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 2.5))
    im = ax.imshow(f1_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(GESTURE_CLASSES))); ax.set_xticklabels(GESTURE_CLASSES, fontsize=12)
    ax.set_yticks(range(len(models)));          ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=11)
    for i in range(len(models)):
        for j in range(len(GESTURE_CLASSES)):
            v = f1_mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13, fontweight="bold", color="white" if v < 0.5 else "black")
    plt.colorbar(im, ax=ax, label="F1 Score"); ax.set_title("Per-Class F1 Score (MobileNet)", pad=12)
    plt.tight_layout(); plt.savefig(out_dir / "1_f1_heatmap_mobilenet.png"); plt.close()

def plot_visibility_vs_error(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n6. ERROR RATE VS VISIBILITY\n" + "─"*55)
    bins  = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    labels = ["0.0–0.3\n(poor)", "0.3–0.5\n(low)", "0.5–0.7\n(medium)", "0.7–1.0\n(good)"]
    models = [m for m in MODEL_NAMES if any(m in d.get("predictions", {}) for d in data)]
    if not models: return

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        error_rates = []
        for lo, hi in bins:
            bucket = [d for d in data if model in d["predictions"] and lo <= d.get("visibility", 1) < hi]
            if not bucket: error_rates.append(0); continue
            err = sum(1 for d in bucket if d["predictions"][model] != d["gt"]) / len(bucket)
            error_rates.append(err * 100)
        ax.bar(np.arange(len(bins)), error_rates, 0.4, label=MODEL_LABELS[model], color=MODEL_COLORS[model], alpha=0.85)
        print(f"  {MODEL_LABELS[model]}: {[f'{e:.1f}%' for e in error_rates]}")

    ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels(labels)
    ax.set_xlabel("Mean Visibility"); ax.set_ylabel("Error rate (%)")
    ax.set_title("Error Rate vs Landmark Visibility (MobileNet)"); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig(out_dir / "6_visibility_error_mobilenet.png"); plt.close()

def plot_per_gesture_error_rate(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n7. PER-GESTURE ERROR RATE\n" + "─"*55)
    models = [m for m in MODEL_NAMES if any(m in d.get("predictions", {}) for d in data)]
    if not models: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        err_rates = []
        for gesture in GESTURE_CLASSES:
            frames = [d for d in data if d["gt"] == gesture and model in d["predictions"]]
            if not frames: err_rates.append(0); continue
            err_rates.append((sum(1 for d in frames if d["predictions"][model] != gesture) / len(frames)) * 100)
        bars = ax.bar(np.arange(len(GESTURE_CLASSES)), err_rates, 0.4, color=MODEL_COLORS[model])
        for bar, val in zip(bars, err_rates):
            if val > 1: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.0f}%", ha="center")
        print(f"  {MODEL_LABELS[model]}: " + " | ".join(f"{g}={e:.1f}%" for g, e in zip(GESTURE_CLASSES, err_rates)))

    ax.set_xticks(np.arange(len(GESTURE_CLASSES))); ax.set_xticklabels(GESTURE_CLASSES)
    ax.set_title("Per-Gesture Error Rate (MobileNet)"); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig(out_dir / "7_per_gesture_error_mobilenet.png"); plt.close()

def run_analysis(log_path: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    data = load_log(log_path)
    if not data: return
    plot_per_class_f1(data, out)
    plot_visibility_vs_error(data, out)
    plot_per_gesture_error_rate(data, out)
    print(f"\n{'='*55}\nMobileNet Analysis complete. Saved to: {out}/\n{'='*55}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="benchmark/detailed_log.json")
    parser.add_argument("--out", default="benchmark/analysis_mobilenet")
    args = parser.parse_args()
    run_analysis(args.log, args.out)