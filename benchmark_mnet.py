import argparse
import time
import csv
import numpy as np
import os
import json
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- 1. GLOBAL CONFIGURATION ---
GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "MOVE"]
# We include UNCERTAIN in the display classes for the confusion matrix
DISPLAY_CLASSES = GESTURE_CLASSES + ["UNCERTAIN"]

# --- 2. PLOTTING & ANALYSIS ---
class FailureAnalyser:
    def __init__(self, save_dir: str = "benchmark/failures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cases = []

    def check(self, frame, prediction: str, ground_truth: str):
        if ground_truth and prediction != ground_truth:
            fname = self.save_dir / f"fail_{int(time.time()*1000)}.jpg"
            display = frame.copy()
            color = (0, 0, 255) # Red for failure
            cv2.putText(display, f"GT: {ground_truth} | PRED: {prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imwrite(str(fname), display)
            self.cases.append({"gt": ground_truth, "pred": prediction, "file": str(fname)})

    def report(self):
        print(f"\n--- Failure Case Report ---")
        print(f"Total misclassifications saved: {len(self.cases)}")

def plot_accuracy_vs_latency(results: dict, save_path: str = "benchmark/latency_report.png"):
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    accs = [results[m]["accuracy"] * 100 for m in methods]
    lats = [results[m]["mean_latency_ms"] for m in methods]
    
    plt.scatter(lats, accs, color='red', s=100)
    for i, m in enumerate(methods):
        plt.annotate(m, (lats[i], accs[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.title("GestureBall: Accuracy vs Latency")
    plt.xlabel("Mean Latency (ms)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"✅ Saved Latency Report to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path: str = "benchmark/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=DISPLAY_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("GestureBall Final Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved Confusion Matrix to {save_path}")

# --- 3. FINAL CLASSIFIER (TEMPORAL STRIDING) ---
class PretrainedImageClassifier:
    def __init__(self, model_path: str, num_frames: int = 15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        
        # Build Architecture matching our training
        self.model = models.mobilenet_v2()
        self.model.features[0][0] = nn.Conv2d(num_frames * 3, 32, 3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(GESTURE_CLASSES))
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"❌ Could not find weight file at {model_path}")

        # Load weights and clean prefixes
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("backbone.", "").replace("module.", "")
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_clip(self, frames):
        n = len(frames)
        if n == 0: return "UNCERTAIN"

        # TEMPORAL STRIDING: Cover the full video context
        step = n / self.num_frames
        indices = [int(i * step) for i in range(self.num_frames)]
        sequence = [frames[idx] for idx in indices]
        
        tensors = []
        for f in sequence:
            img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((224, 224))
            img_t = TF.to_tensor(img)
            img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensors.append(img_t)

        input_tensor = torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)
        logits = self.model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        
        return GESTURE_CLASSES[idx.item()] if conf.item() > 0 else "UNCERTAIN"

# --- 4. BENCHMARK ENGINE ---
def run_benchmark(video_folder, model_path, headless=False):
    classifier = PretrainedImageClassifier(model_path=model_path)
    video_dir = Path(video_folder)
    video_files = sorted(list(video_dir.glob("*.mp4")))
    
    output_dir = Path("benchmark")
    output_dir.mkdir(exist_ok=True)
    
    analyser = FailureAnalyser()
    all_preds = []
    all_truths = []
    latencies = []

    print(f"--- Starting Final Benchmark on {len(video_files)} Clips ---")

    for vid_path in video_files:
        cap = cv2.VideoCapture(str(vid_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Match Training Flip
            frame = cv2.flip(frame, 1)
            frames.append(frame)
        cap.release()

        if not frames: continue

        # Inference
        start_t = time.perf_counter()
        prediction = classifier.predict_clip(frames)
        latency = (time.perf_counter() - start_t) * 1000
        
        # Ground Truth from Filename (e.g., SHOOT_01.mp4)
        gt = vid_path.name.split("_")[0].upper()
        if gt not in DISPLAY_CLASSES: gt = "UNCERTAIN"

        all_preds.append(prediction)
        all_truths.append(gt)
        latencies.append(latency)

        analyser.check(frames[-1], prediction, gt)

        if not headless:
            display = frames[-1].copy()
            color = (0, 255, 0) if prediction == gt else (0, 0, 255)
            cv2.putText(display, f"PRED: {prediction} (GT: {gt})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Benchmark", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

    # --- 5. GENERATE FINAL REPORTS ---
    if all_truths:
        # Calculate Stats
        correct = sum(1 for p, t in zip(all_preds, all_truths) if p == t)
        accuracy = correct / len(all_truths)
        mean_latency = np.mean(latencies)
        
        print("\n" + "="*50)
        print(f"ACCURACY: {accuracy*100:.2f}%")
        print(f"LATENCY:  {mean_latency:.2f} ms/clip")
        print("="*50)

        # Plots
        plot_confusion_matrix(all_truths, all_preds)
        
        results_dict = {
            "mobilenet_stacked": {
                "accuracy": accuracy,
                "mean_latency_ms": mean_latency
            }
        }
        plot_accuracy_vs_latency(results_dict)
        analyser.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing test .mp4 clips")
    parser.add_argument("--model", default="models/gesture_ball_best.pt", help="Path to weights")
    parser.add_argument("--headless", action="store_true", help="Don't show video")
    args = parser.parse_args()

    run_benchmark(args.folder, args.model, args.headless)