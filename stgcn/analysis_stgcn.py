"""
analysis_stgcn.py
====================
Deep error analysis for the ST-GCN Classifier.

Reads  : benchmark/detailed_log_stgcn.json
Outputs: benchmark/analysis_stgcn/
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support

# ── Constants ────────────────────────────────────────────────────────────────
GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "MOVE"]
MODEL_NAMES     = ["stgcn"]
MODEL_LABELS    = {"stgcn": "ST-GCN (Temporal)"}
MODEL_COLORS    = {"stgcn": "#8c564b"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

def load_log(log_path: str) -> list:
    with open(log_path) as f: raw = json.load(f)
    labelled = [d for d in raw if d.get("gt") and d["gt"] in GESTURE_CLASSES]
    print(f"\nLoaded {len(raw)} total videos | {len(labelled)} with GT labels.")
    return labelled

def plot_per_class_f1(data: list, out_dir: Path):
    models = [m for m in MODEL_NAMES if any(m in d.get("prediction", {}) for d in data)]
    if not models: return
    f1_mat = np.zeros((len(models), len(GESTURE_CLASSES)))

    print("\n" + "─"*55 + "\n1. PER-CLASS F1 SCORES\n" + "─"*55)
    for i, model in enumerate(models):
        y_true = [d["gt"] for d in data if model in d["prediction"]]
        y_pred = [d["prediction"][model] for d in data if model in d["prediction"]]
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
    plt.colorbar(im, ax=ax, label="F1 Score"); ax.set_title("Per-Class F1 Score (ST-GCN)", pad=12)
    plt.tight_layout(); plt.savefig(out_dir / "1_f1_heatmap_stgcn.png"); plt.close()

def plot_confidence_distributions(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n2. CONFIDENCE DISTRIBUTIONS\n" + "─"*55)
    model = "stgcn"
    entries = [d for d in data if model in d.get("prediction", {}) and model in d.get("confidence", {})]
    if not entries: return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Confidence Distributions — ST-GCN\nHigh-confidence errors → systematic model failure", fontweight="bold")
    bins = np.linspace(0, 1, 21)

    for idx, gesture in enumerate(GESTURE_CLASSES):
        ax = axes[idx // 2][idx % 2]
        gframes = [d for d in entries if d["gt"] == gesture]
        correct = [d["confidence"][model] for d in gframes if d["prediction"][model] == gesture]
        wrong   = [d["confidence"][model] for d in gframes if d["prediction"][model] != gesture]

        if correct: ax.hist(correct, bins=bins, color="#27ae60", alpha=0.75, label=f"Correct (n={len(correct)})")
        if wrong: ax.hist(wrong, bins=bins, color="#e74c3c", alpha=0.75, label=f"Misclassified (n={len(wrong)})")
        
        ax.set_title(f"GT = {gesture}"); ax.set_xlim(0, 1); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(out_dir / f"2_confidence_{model}.png"); plt.close()

def plot_per_gesture_error_rate(data: list, out_dir: Path):
    print("\n" + "─"*55 + "\n7. PER-GESTURE ERROR RATE\n" + "─"*55)
    models = [m for m in MODEL_NAMES if any(m in d.get("prediction", {}) for d in data)]
    if not models: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        err_rates = []
        for gesture in GESTURE_CLASSES:
            frames = [d for d in data if d["gt"] == gesture and model in d["prediction"]]
            if not frames: err_rates.append(0); continue
            err_rates.append((sum(1 for d in frames if d["prediction"][model] != gesture) / len(frames)) * 100)
        bars = ax.bar(np.arange(len(GESTURE_CLASSES)), err_rates, 0.4, color=MODEL_COLORS[model])
        for bar, val in zip(bars, err_rates):
            if val > 1: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.0f}%", ha="center")
        print(f"  {MODEL_LABELS[model]}: " + " | ".join(f"{g}={e:.1f}%" for g, e in zip(GESTURE_CLASSES, err_rates)))

    ax.set_xticks(np.arange(len(GESTURE_CLASSES))); ax.set_xticklabels(GESTURE_CLASSES)
    ax.set_title("Per-Gesture Error Rate (ST-GCN)"); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig(out_dir / "7_per_gesture_error_stgcn.png"); plt.close()

def run_analysis(log_path: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    data = load_log(log_path)
    if not data: return
    plot_per_class_f1(data, out)
    plot_confidence_distributions(data, out)
    plot_per_gesture_error_rate(data, out)
    print(f"\n{'='*55}\nST-GCN Analysis complete. Saved to: {out}/\n{'='*55}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults explicitly to the ST-GCN log file
    parser.add_argument("--log", default="benchmark/detailed_log_stgcn.json")
    parser.add_argument("--out", default="benchmark/analysis_stgcn")
    args = parser.parse_args()
    run_analysis(args.log, args.out)