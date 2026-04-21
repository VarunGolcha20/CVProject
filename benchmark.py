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
import os
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


# ── Detailed per-frame logger ─────────────────────────────────────────────────
class DetailedLogger:
    """
    Captures per-frame data for deep analysis:
      - ground truth label
      - predictions from each model
      - confidence scores (CNN, LSTM)
      - key feature values (for rule-based boundary analysis)
      - LSTM attention weights (for attention heatmap)
      - mean visibility

    Call .log() each labelled frame, then .save() at the end.
    analysis_extended.py reads the resulting JSON.
    """

    def __init__(self):
        self._frames = []

    def log(self, frame_num: int, gt: str, fv,
            predictions: dict, confidences: dict, attention: list = None):
        """
        fv            : FeatureVector (used to extract rule-based boundary features)
        predictions   : {model_name: label_str}
        confidences   : {model_name: float}  — only CNN and LSTM populate this
        attention     : list[float] of length WINDOW, from LSTM (or None)
        """
        self._frames.append({
            "frame": frame_num,
            "gt": gt,
            "visibility": round(float(fv.mean_visibility), 4),
            "predictions": predictions,
            "confidences": {k: round(v, 4) for k, v in confidences.items()},
            # Key rule-based boundary features
            "features": {
                "left_forward_ratio":   round(float(fv.left_foot_forward_ratio), 4),
                "right_forward_ratio":  round(float(fv.right_foot_forward_ratio), 4),
                "left_lateral_offset":  round(float(fv.left_foot_lateral_offset), 4),
                "right_lateral_offset": round(float(fv.right_foot_lateral_offset), 4),
                "left_swing_vel":       round(float(fv.left_swing_velocity), 4),
                "right_swing_vel":      round(float(fv.right_swing_velocity), 4),
                "left_swing_amp":       round(float(fv.left_swing_amplitude), 4),
                "right_swing_amp":      round(float(fv.right_swing_amplitude), 4),
                "total_kinetic_energy": round(float(fv.total_kinetic_energy), 4),
                "hip_stability":        round(float(fv.hip_stability), 4),
                "moving_joint_count":   round(float(fv.moving_joint_count), 4),
            },
            "attention_weights": [round(w, 5) for w in attention] if attention else None,
        })

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self._frames, f)
        labelled = sum(1 for d in self._frames if d["gt"])
        print(f"\nDetailed log saved → {path}")
        print(f"  Total frames logged: {len(self._frames)} ({labelled} with GT labels)")
        print(f"  Run: python analysis_extended.py --log {path}")

    def __len__(self):
        return len(self._frames)


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


def run_benchmark(video_sources, labels_csv: str = None,
                  use_cnn: bool = False, use_lstm: bool = False):
    """
    video_sources: List of video paths or a single webcam index
    """
    output_dir = Path("benchmark")
    output_dir.mkdir(exist_ok=True)
    
    extractor = PoseExtractor()
    engineer  = FeatureEngineer(history=35)
    
    # We use a shared failure analyzer and detailed logger for the whole batch
    failure_analyser = FailureAnalyser(save_dir="benchmark/batch_failures")
    detailed_logger  = DetailedLogger()

    # Classifiers & Benchmarkers
    classifiers = {"rule_based": RuleBasedClassifier()}
    if use_cnn:
        try: classifiers["cnn"] = CNNClassifier()
        except Exception as e: print(f"CNN error: {e}")
    if use_lstm:
        try: classifiers["lstm"] = LSTMClassifier()
        except Exception as e: print(f"LSTM error: {e}")

    benches = {name: Benchmarker() for name in classifiers}

    # Load Ground Truth (Master CSV logic)
    gt_map = {}
    if labels_csv and os.path.exists(labels_csv):
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                gt_map[int(row["frame"])] = row["label"]

    all_preds  = {name: [] for name in classifiers}
    all_truths = []
    global_frame_num = 0  # <--- Crucial: This stays synced with label_gen.py

    # Process each video in the list
    for video_path in video_sources:
        cap = cv2.VideoCapture(video_path)
        print(f"\nProcessing: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            t = time.perf_counter()
            pose_frame = extractor.extract(frame, t)

            # Important: We increment frame count even if no pose is found 
            # to match how label_gen.py counts total video frames.
            if not pose_frame:
                global_frame_num += 1
                continue

            fv = engineer.update(pose_frame)
            gt = gt_map.get(global_frame_num)

            predictions = {}
            confidences = {}
            attention   = None

            for name, clf in classifiers.items():
                benches[name].start("total")

                if name == "cnn":
                    pred, conf = clf.predict_with_details(fv, raw_landmarks=pose_frame.landmarks)
                    confidences[name] = conf
                elif name == "lstm":
                    pred, conf, attn = clf.predict_with_details(fv)
                    confidences[name] = conf
                    attention = attn
                else:
                    pred = clf(fv)

                benches[name].stop("total")
                predictions[name] = pred

                # Only record metrics if we have a ground truth label for this frame
                if gt:
                    all_preds[name].append(pred)

            if gt:
                all_truths.append(gt)
                detailed_logger.log(global_frame_num, gt, fv,
                                    predictions, confidences, attention)

            # Failure analysis
            failure_analyser.check(frame, predictions, gt, fv.mean_visibility)

            # UI Overlay
            y = 20
            for name, pred in predictions.items():
                color = (0, 255, 128) if (not gt or pred == gt) else (0, 80, 255)
                cv2.putText(frame, f"{name}: {pred}", (5, y), 0, 0.55, color, 1)
                y += 22
            
            cv2.imshow("GestureBall Batch Benchmark", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return # Exit completely
            
            global_frame_num += 1
        
        cap.release()

    # ── Final Reporting ───────────────────────────────────────────────────
    cv2.destroyAllWindows()
    print("\n─── Batch Benchmark Summary ─────────────────────────────────")
    results = {}
    for name, bench in benches.items():
        s = bench.summary().get("total", {})
        results[name] = {"mean_latency_ms": s.get("mean_ms", 0), "fps": s.get("fps", 0), "accuracy": 0.0}

    if all_truths:
        for name, preds in all_preds.items():
            acc = np.mean([p == t for p, t in zip(preds, all_truths)])
            results[name]["accuracy"] = acc
            print(f"[{name}] Accuracy: {acc:.3f}")
            plot_confusion_matrix(all_truths, preds, name, f"benchmark/cm_{name}.png")

        plot_accuracy_vs_latency(results, "benchmark/batch_comparison.png")
        detailed_logger.save("benchmark/detailed_log.json")
    
    failure_analyser.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Single video file")
    parser.add_argument("--folder", help="Folder of videos")
    parser.add_argument("--labels", default="videos/labels.csv")
    parser.add_argument("--cnn", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    args = parser.parse_args()

    sources = []
    if args.folder:
        sources = sorted([str(p) for p in Path(args.folder).glob("*.mp4")])
    elif args.video:
        sources = [args.video]
    else:
        sources = [0] # Webcam

    run_benchmark(sources, args.labels, args.cnn, args.lstm)