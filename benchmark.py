"""
benchmark.py
============
Runs all three classifiers on the same video and produces:
  1. FPS / latency table (addresses reviewer note)
  2. Confusion matrix for each method
  3. Failure case analysis (addresses reviewer note)
  4. accuracy_vs_latency.png plot

Usage:
  python benchmark.py --video data/test_video.mp4 --labels data/labels.csv
  python benchmark.py --webcam       (live mode, press keys to label)
"""

import argparse
import time
import csv
import numpy as np
import json
import cv2
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

from pipeline import PoseExtractor, FeatureEngineer, Benchmarker
from classifiers.rule_based import RuleBasedClassifier
from classifiers.cnn_classifier import CNNClassifier, GESTURE_CLASSES
from classifiers.lstm_classifier import LSTMClassifier


# ── Failure case detector ─────────────────────────────────────────────────────
class FailureAnalyser:
    """
    Captures frames where classifiers disagree or produce low-confidence results.
    Saves them as a labelled gallery for the report.
    """
    def __init__(self, save_dir: str = "failure_cases"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.cases = []

    def check(self, frame, predictions: dict, ground_truth: str = None,
              visibility: float = 1.0):
        """
        predictions: {"rule": str, "cnn": str, "lstm": str}
        """
        preds = list(predictions.values())
        agreement = len(set(preds)) == 1        # all agree?

        # Failure conditions
        reasons = []
        if not agreement:
            reasons.append("classifier_disagreement")
        if ground_truth and any(p != ground_truth for p in preds):
            reasons.append("misclassification")
        if visibility < 0.5:
            reasons.append("low_visibility")
        if any(p in ("UNCERTAIN", "OCCLUDED") for p in preds):
            reasons.append("low_confidence")

        if reasons:
            fname = self.save_dir / f"fail_{int(time.time()*1000)}.jpg"
            # Add annotation overlay
            display = frame.copy()
            y = 20
            for method, pred in predictions.items():
                color = (0,255,0) if (not ground_truth or pred == ground_truth) else (0,0,255)
                cv2.putText(display, f"{method}: {pred}", (5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 18
            if ground_truth:
                cv2.putText(display, f"GT: {ground_truth}", (5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(display, " | ".join(reasons), (5, frame.shape[0]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)
            cv2.imwrite(str(fname), display)
            self.cases.append({"reasons": reasons, "preds": predictions,
                                "gt": ground_truth, "file": str(fname)})

        return reasons

    def report(self):
        counts = defaultdict(int)
        for c in self.cases:
            for r in c["reasons"]:
                counts[r] += 1
        print("\n─── Failure Case Analysis ───────────────────────────────")
        for reason, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:<30} {count:>4} frames")
        print(f"  Total failure frames: {len(self.cases)}")
        with open(self.save_dir / "failure_report.json", "w") as f:
            json.dump({"counts": dict(counts), "cases": self.cases[:50]}, f, indent=2)
        return counts


# ── Accuracy vs Latency plot ───────────────────────────────────────────────────
def plot_accuracy_vs_latency(results: dict, save_path: str = "accuracy_vs_latency.png"):
    """
    results format:
    {
      "rule_based": {"accuracy": 0.72, "mean_latency_ms": 0.8,  "fps": 1250},
      "cnn":        {"accuracy": 0.85, "mean_latency_ms": 8.2,  "fps": 122},
      "lstm":       {"accuracy": 0.91, "mean_latency_ms": 14.5, "fps": 69},
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GestureBall: Accuracy vs Latency Trade-off", fontsize=14, fontweight="bold")

    colors = {"rule_based": "#EF9F27", "cnn": "#E85D30", "lstm": "#3B8BD4"}
    labels = {"rule_based": "Rule-based", "cnn": "CNN", "lstm": "LSTM/GRU"}

    # ── Plot 1: scatter ───────────────────────────────────────────────────────
    ax = axes[0]
    for method, data in results.items():
        ax.scatter(data["mean_latency_ms"], data["accuracy"] * 100,
                   color=colors[method], s=200, zorder=5, label=labels[method])
        ax.annotate(labels[method],
                    (data["mean_latency_ms"], data["accuracy"] * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.5, 90.5, "90% target", fontsize=8, color="gray")
    ax.axvline(x=40, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(40.5, 50, "40ms budget\n(25fps)", fontsize=8, color="gray")

    ax.set_xlabel("Mean inference latency (ms)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Latency vs Accuracy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: grouped bars ──────────────────────────────────────────────────
    ax2 = axes[1]
    methods = list(results.keys())
    x       = np.arange(len(methods))
    width   = 0.35

    bars1 = ax2.bar(x - width/2, [results[m]["accuracy"]*100 for m in methods],
                    width, label="Accuracy (%)", color=[colors[m] for m in methods],
                    alpha=0.85)
    ax3 = ax2.twinx()
    bars2 = ax3.bar(x + width/2, [results[m]["fps"] for m in methods],
                    width, label="FPS", color=[colors[m] for m in methods],
                    alpha=0.4, hatch="//")

    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax3.set_ylabel("Throughput (FPS)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[m] for m in methods])
    ax2.set_title("Accuracy vs FPS per Method")
    ax2.set_ylim(0, 105)

    # Legend
    patches = [
        mpatches.Patch(color="gray", alpha=0.85, label="Accuracy (left axis)"),
        mpatches.Patch(color="gray", alpha=0.4,  hatch="//", label="FPS (right axis)"),
    ]
    ax2.legend(handles=patches, fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    return fig


# ── Confusion matrix helper ───────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, method_name: str,
                          save_path: str = None):
    cm = confusion_matrix(y_true, y_pred, labels=GESTURE_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=GESTURE_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {method_name}", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")
    return fig


# ── Main benchmark runner ─────────────────────────────────────────────────────
def run_benchmark(video_source=0, labels_csv: str = None,
                  use_cnn: bool = False, use_lstm: bool = False):

    cap       = cv2.VideoCapture(video_source)
    extractor = PoseExtractor()
    engineer  = FeatureEngineer(history=30)
    failure_analyser = FailureAnalyser()

    # Classifiers
    classifiers = {"rule_based": RuleBasedClassifier()}
    if use_cnn:
        try:   classifiers["cnn"]  = CNNClassifier()
        except Exception as e: print(f"CNN not loaded: {e}")
    if use_lstm:
        try:   classifiers["lstm"] = LSTMClassifier()
        except Exception as e: print(f"LSTM not loaded: {e}")

    # Benchmarkers (one per classifier)
    benches = {name: Benchmarker() for name in classifiers}

    # Ground truth (optional)
    gt_map = {}
    if labels_csv:
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                gt_map[int(row["frame"])] = row["label"]

    # Predictions for metrics
    all_preds  = {name: [] for name in classifiers}
    all_truths = []
    frame_num  = 0

    print("Running benchmark — press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        t     = time.perf_counter()

        pose_frame = extractor.extract(frame, t)
        if not pose_frame:
            frame_num += 1
            continue

        fv = engineer.update(pose_frame)
        gt = gt_map.get(frame_num)

        predictions = {}
        for name, clf in classifiers.items():
            benches[name].start("total")
            if name == "cnn":
                pred = clf(fv, raw_landmarks=pose_frame.landmarks)
            else:
                pred = clf(fv)
            benches[name].stop("total")
            predictions[name] = pred
            all_preds[name].append(pred)

        if gt:
            all_truths.append(gt)

        # Failure analysis
        vis = fv.mean_visibility
        failure_analyser.check(frame, predictions, gt, vis)

        # Display
        y = 20
        for name, pred in predictions.items():
            color = (0,255,128) if (not gt or pred == gt) else (0,80,255)
            cv2.putText(frame, f"{name}: {pred}", (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += 22
        benches["rule_based"].overlay(frame)
        cv2.imshow("GestureBall Benchmark", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_num += 1

    cap.release(); extractor.close(); cv2.destroyAllWindows()

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n─── Benchmark Summary ─────────────────────────────────────")
    results = {}
    for name, bench in benches.items():
        s = bench.summary().get("total", {})
        ms  = s.get("mean_ms", 0)
        fps = s.get("fps", 0)
        print(f"\n  [{name}]  mean={ms:.1f}ms  fps={fps:.0f}")
        results[name] = {"mean_latency_ms": ms, "fps": fps, "accuracy": 0.0}

    if all_truths:
        for name, preds in all_preds.items():
            n = min(len(preds), len(all_truths))
            acc = np.mean([p == t for p, t in zip(preds[:n], all_truths[:n])])
            results[name]["accuracy"] = acc
            print(f"\n  [{name}] accuracy: {acc:.3f}")
            print(classification_report(all_truths[:n], preds[:n],
                                        labels=GESTURE_CLASSES, zero_division=0))
            plot_confusion_matrix(all_truths[:n], preds[:n], name,
                                  save_path=f"confusion_{name}.png")

    failure_analyser.report()

    if len(results) > 1:
        plot_accuracy_vs_latency(results)

    with open("benchmark_summary.json","w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmark_summary.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   default=None, help="Path to video file (default: webcam)")
    parser.add_argument("--labels",  default=None, help="CSV with frame,label columns")
    parser.add_argument("--cnn",     action="store_true", help="Include CNN classifier")
    parser.add_argument("--lstm",    action="store_true", help="Include LSTM classifier")
    parser.add_argument("--webcam",  action="store_true", help="Use webcam")
    args = parser.parse_args()

    source = 0 if (args.webcam or args.video is None) else args.video
    run_benchmark(source, args.labels, args.cnn, args.lstm)
