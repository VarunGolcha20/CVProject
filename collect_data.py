"""
collect_data.py
===============
Interactive data collection for gesture labelling.
Also includes SVM baseline (another reviewer-suggested method).

Controls during collection:
  S = SHOOT  |  P = PASS  |  M = MOVE  |  D = DEFEND  |  I = IDLE
  R = record LSTM sequence  |  Q = quit
"""

import cv2
import time
import numpy as np
from collections import deque
from pipeline import PoseExtractor, FeatureEngineer
from classifiers.cnn_classifier import collect_samples, GESTURE_CLASSES
from classifiers.lstm_classifier import SequenceRecorder


KEY_MAP = {
    ord('s'): "SHOOT",
    ord('p'): "PASS",
    ord('m'): "MOVE_RIGHT",
    ord('d'): "DEFEND",
    ord('i'): "IDLE",
}


def collect(source: int = 0, out_cnn: str = "data/cnn",
            out_lstm: str = "data/lstm"):
    cap       = cv2.VideoCapture(source)
    extractor = PoseExtractor()
    engineer  = FeatureEngineer(history=30)
    recorder  = SequenceRecorder(out_dir=out_lstm)

    stats = {g: 0 for g in GESTURE_CLASSES}
    last_label = "IDLE"

    print("Data collection started.")
    print("Keys: S=SHOOT  P=PASS  M=MOVE  D=DEFEND  I=IDLE  R=save_sequence  Q=quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        t     = time.perf_counter()

        pose_frame = extractor.extract(frame, t)
        if pose_frame:
            fv = engineer.update(pose_frame)
            recorder.push(fv)

        # Display
        cv2.putText(frame, f"Current label: {last_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,128), 2)
        y = 60
        for g, count in stats.items():
            cv2.putText(frame, f"{g}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            y += 18
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        if key in KEY_MAP:
            last_label = KEY_MAP[key]
            if pose_frame:
                collect_samples(last_label, pose_frame, fv, out_cnn)
                stats[last_label] += 1
                print(f"  Saved CNN sample: {last_label}  (total {stats[last_label]})")

        if key == ord('r'):
            recorder.save(last_label)
            stats[last_label] += 1
            print(f"  Saved LSTM sequence: {last_label}")

    cap.release(); extractor.close(); cv2.destroyAllWindows()
    print("\nCollection complete:", stats)


# ── SVM baseline ──────────────────────────────────────────────────────────────
"""
Reviewer suggestion: SVM on landmark features.
Use this alongside CNN/LSTM to complete the comparison table.
"""
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path


def train_svm(data_dir: str = "data/cnn",
              save_path: str = "models/svm_gesture.pkl"):
    import glob
    X, y = [], []
    for f in glob.glob(f"{data_dir}/*.npz"):
        d = np.load(f)
        # Flatten landmarks + engineered features
        feat = np.concatenate([d["landmarks"].flatten(), d["features"]])
        X.append(feat);  y.append(int(d["label"]))

    X = np.array(X);  y = np.array(y)
    print(f"SVM training: {X.shape[0]} samples, {X.shape[1]} features")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, class_weight="balanced")),
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"SVM 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X, y)
    Path(save_path).parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"Saved {save_path}")
    return pipeline


class SVMClassifier:
    def __init__(self, model_path: str = "models/svm_gesture.pkl"):
        self.model = joblib.load(model_path)
        from classifiers.cnn_classifier import IDX2GESTURE
        self.idx2g = IDX2GESTURE

    def __call__(self, fv, raw_landmarks: dict = None) -> str:
        if raw_landmarks is None:
            feat = fv.to_array()
        else:
            lm_arr = np.array(list(raw_landmarks.values()), dtype=np.float32).flatten()
            feat   = np.concatenate([lm_arr, fv.to_array()])
        probs = self.model.predict_proba([feat])[0]
        idx   = int(np.argmax(probs))
        return self.idx2g.get(idx, "IDLE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--svm", action="store_true", help="Train SVM instead of collecting")
    args = parser.parse_args()
    if args.svm:
        train_svm()
    else:
        collect()
