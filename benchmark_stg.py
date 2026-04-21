import argparse
import time
import numpy as np
import cv2
import torch
import mediapipe as mp
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import the model architecture from the training script
from stgcn import GestureSTGCN, CORE_JOINTS, GESTURE_CLASSES

DISPLAY_CLASSES = GESTURE_CLASSES + ["UNCERTAIN"]

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
        if n == 0: return "UNCERTAIN", 0, frames

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
        
        return label, latency, frames

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
    
    all_preds, all_truths, latencies = [], [], []

    print(f"--- Benchmarking {len(video_paths)} videos ---")
    for vid_path in video_paths:
        pred, lat, frames = classifier.predict_video(vid_path)
        gt = vid_path.name.split("_")[0].upper()
        if gt not in DISPLAY_CLASSES: gt = "UNCERTAIN"

        all_preds.append(pred)
        all_truths.append(gt)
        latencies.append(lat)

        color = "\033[92m" if pred == gt else "\033[91m"
        print(f"{vid_path.name:<25} | GT: {gt:<8} | PRED: {color}{pred:<8}\033[0m | {lat:.1f}ms")

        if not headless and frames:
            display = frames[-1].copy()
            c_bgr = (0, 255, 0) if pred == gt else (0, 0, 255)
            cv2.putText(display, f"ST-GCN: {pred}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, c_bgr, 2)
            cv2.imshow("Benchmark", display)
            if cv2.waitKey(500) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    if all_truths:
        plot_results(all_truths, all_preds, np.mean(latencies))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="test", help="Folder of test .mp4s")
    parser.add_argument("--model", default="stgcn_best.pt", help="Weights file")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    
    # NOTE: Ensure you save the trainer as stgcn_trainer.py in the same directory
    run_benchmark(args.folder, args.model, args.headless)