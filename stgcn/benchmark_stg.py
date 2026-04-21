"""
benchmark_stgcn.py
==================
Runs the ST-GCN classifier on video files and produces:
  1. Confusion matrix plot
  2. Latency/Accuracy console output
  3. Failure case analysis & failure_report.json (per-video)
  4. detailed_log.json (per-video metrics)
"""

import argparse
import time
import json
import numpy as np
import cv2
import torch
import mediapipe as mp
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import the model architecture from the training script
from stgcn import GestureSTGCN, CORE_JOINTS, GESTURE_CLASSES

DISPLAY_CLASSES = GESTURE_CLASSES + ["UNCERTAIN"]


# ── Failure case detector (Video-level) ───────────────────────────────────────
class FailureAnalyser:
    def __init__(self, save_dir: str = "benchmark/batch_failures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cases = []

    def check(self, video_name: str, frame, pred: str, gt: str, conf: float):
        reasons = []
        if pred != gt:
            reasons.append("misclassification")
        if pred == "UNCERTAIN" or conf < 0.5:
            reasons.append("low_confidence")

        if reasons:
            fname = self.save_dir / f"fail_{video_name}.jpg"
            # We use the final extracted frame to display the failure
            display = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            cv2.putText(display, f"ST-GCN: {pred} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, f"GT: {gt}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, " | ".join(reasons), (10, display.shape[0]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imwrite(str(fname), display)
            self.cases.append({
                "video": video_name,
                "reasons": reasons,
                "pred": pred,
                "gt": gt,
                "confidence": round(conf, 4),
                "file": str(fname)
            })

        return reasons

    def report(self):
        counts = defaultdict(int)
        for c in self.cases:
            for r in c["reasons"]:
                counts[r] += 1
        print("\n─── Failure Case Analysis ───────────────────────────────")
        for reason, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:<30} {count:>4} videos")
        print(f"  Total failure videos: {len(self.cases)}")
        with open(self.save_dir / "failure_report.json", "w") as f:
            json.dump({"counts": dict(counts), "cases": self.cases[:50]}, f, indent=2)
        return counts


# ── Detailed per-video logger ─────────────────────────────────────────────────
class DetailedLogger:
    def __init__(self):
        self._videos = []

    def log(self, video_name: str, gt: str, pred: str, conf: float, latency: float):
        self._videos.append({
            "video": video_name,
            "gt": gt,
            "prediction": {"stgcn": pred},
            "confidence": {"stgcn": round(conf, 4)},
            "latency_ms": round(latency, 2)
        })

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self._videos, f, indent=2) 
        print(f"\nDetailed log saved → {path}")
        print(f"  Total videos logged: {len(self._videos)}")


# ── ST-GCN Classifier ─────────────────────────────────────────────────────────
class STGCN_Classifier:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GestureSTGCN(num_classes=len(GESTURE_CLASSES))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    @torch.no_grad()
    def predict_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.flip(frame, 1)) # Mirror
        cap.release()

        n = len(frames)
        if n == 0: return "UNCERTAIN", 0.0, 0, frames

        # Extract Skeletons (Strided)
        step = n / 15
        indices = [int(i * step) for i in range(15)]
        
        video_tensor = np.zeros((1, 3, 15, len(CORE_JOINTS)), dtype=np.float32)
        
        start_t = time.perf_counter()
        
        for t, idx in enumerate(indices):
            img_rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(img_rgb)
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                for v, joint_idx in enumerate(CORE_JOINTS):
                    video_tensor[0, 0, t, v] = lms[joint_idx].x
                    video_tensor[0, 1, t, v] = lms[joint_idx].y
                    video_tensor[0, 2, t, v] = lms[joint_idx].visibility

        # Inference
        input_tensor = torch.tensor(video_tensor).to(self.device)
        logits = self.model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        conf, pred_idx = probs.max(0)
        
        latency = (time.perf_counter() - start_t) * 1000
        label = GESTURE_CLASSES[pred_idx.item()] if conf.item() > 0.5 else "UNCERTAIN"
        
        # Returned confidence score alongside prediction, latency, and frames
        return label, conf.item(), latency, frames


def plot_results(all_truths, all_preds, mean_latency):
    Path("benchmark").mkdir(exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_truths, all_preds, labels=DISPLAY_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_CLASSES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("ST-GCN Confusion Matrix")
    plt.tight_layout()
    plt.savefig("benchmark/cm_stgcn.png")

    # 2. Latency/Accuracy text output
    acc = sum(1 for p, t in zip(all_preds, all_truths) if p == t) / len(all_truths)
    print("\n" + "="*50)
    print(f"✅ ST-GCN BENCHMARK COMPLETE")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Latency:  {mean_latency:.2f} ms per video (Includes MediaPipe)")
    print("="*50)
    print("Plots saved to benchmark/ directory.")


def run_benchmark(test_folder, model_path, headless=False):
    classifier = STGCN_Classifier(model_path)
    video_paths = list(Path(test_folder).glob("*.mp4"))
    
    # ── Initialize the Loggers ──
    detailed_logger = DetailedLogger()
    failure_analyser = FailureAnalyser()
    
    all_preds, all_truths, latencies = [], [], []

    print(f"--- Benchmarking {len(video_paths)} videos ---")
    for vid_path in video_paths:
        
        # Unpack the added confidence score
        pred, conf, lat, frames = classifier.predict_video(vid_path)
        
        gt = vid_path.name.split("_")[0].upper()
        if gt not in DISPLAY_CLASSES: gt = "UNCERTAIN"

        all_preds.append(pred)
        all_truths.append(gt)
        latencies.append(lat)
        
        # ── Log the video stats ──
        detailed_logger.log(vid_path.name, gt, pred, conf, lat)
        
        # ── Check for failures (using the last frame for the screenshot) ──
        last_frame = frames[-1] if frames else None
        failure_analyser.check(vid_path.name, last_frame, pred, gt, conf)

        color = "\033[92m" if pred == gt else "\033[91m"
        print(f"{vid_path.name:<25} | GT: {gt:<8} | PRED: {color}{pred:<8}\033[0m | CONF: {conf:.2f} | {lat:.1f}ms")

        if not headless and frames:
            display = frames[-1].copy()
            c_bgr = (0, 255, 0) if pred == gt else (0, 0, 255)
            cv2.putText(display, f"ST-GCN: {pred}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, c_bgr, 2)
            cv2.imshow("Benchmark", display)
            if cv2.waitKey(500) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    
    # ── Output Results & Save Logs ──
    if all_truths:
        plot_results(all_truths, all_preds, np.mean(latencies))
        detailed_logger.save("detailed_log_stgcn.json")
        failure_analyser.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="../videos", help="Folder of test .mp4s")
    parser.add_argument("--model", default="stgcn_best.pt", help="Weights file")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    
    # NOTE: Ensure you save the trainer as stgcn_trainer.py in the same directory
    run_benchmark(args.folder, args.model, args.headless)