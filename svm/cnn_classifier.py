"""
classifiers/cnn_classifier.py
==============================
CNN-based single-frame gesture classifier.
Input: (FEATURE_DIM,) feature vector OR (33, 4) raw landmark array
Output: gesture label string
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

# Import shared constants from config.py
# from config import FEATURE_DIM, GESTURE_CLASSES, KEYPOINT_NAMES
FEATURE_DIM = 32

GESTURE_CLASSES = [
    "IDLE", "SHOOT", "PASS", "MOVE"
]

# Defines the exact order of keypoints expected by the CNN
KEYPOINT_NAMES = [
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_shoulder", "right_shoulder",
    "left_foot", "right_foot"
]
# ── Gesture classes ────────────────────────────────────────────────────────────
NUM_CLASSES = len(GESTURE_CLASSES)
IDX2GESTURE = dict(enumerate(GESTURE_CLASSES))
GESTURE2IDX = {v: k for k, v in IDX2GESTURE.items()}


# ── Model ─────────────────────────────────────────────────────────────────────
class PoseCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)

        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.drop  = nn.Dropout(dropout)

        # FIX 1: Add a BatchNorm layer specifically for your raw features
        self.bn_feat = nn.BatchNorm1d(FEATURE_DIM)

        self.fc_feat = nn.Linear(FEATURE_DIM, 64)

        self.fc1 = nn.Linear(256 + 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, landmarks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # Conv branch
        x = landmarks.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)

        # FIX 2: Pass features through BatchNorm BEFORE the linear layer
        f = self.bn_feat(features)
        f = F.relu(self.fc_feat(f))

        # Fuse
        out = torch.cat([x, f], dim=1)
        out = self.drop(out)
        out = F.relu(self.fc1(out))
        return self.fc2(out)
# ── Dataset ───────────────────────────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = list(Path(data_dir).glob("*.npz"))
        assert self.files, f"No .npz files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])

        # Cast to float32 and normalize exactly like inference
        lm_arr = d["landmarks"].astype(np.float32)
        # lm_arr[:, :2] /= (lm_arr[:, :2].max() + 1e-8)

        return {
            "landmarks": torch.from_numpy(lm_arr),
            "features":  torch.from_numpy(d["features"]).float(),
            "label":     torch.tensor(int(d["label"]), dtype=torch.long),
        }


# ── Training ──────────────────────────────────────────────────────────────────
def _split_by_video(dataset, val_frac=0.2, seed=42):
    import random
    from collections import defaultdict
    random.seed(seed)
    video_groups = defaultdict(list)
    for i, f in enumerate(dataset.files):
        parts = f.stem.rsplit("_", 1)
        stem  = parts[0] if len(parts) == 2 and parts[1].isdigit() else f.stem
        video_groups[stem].append(i)
    stems = list(video_groups.keys())
    random.shuffle(stems)
    split       = max(1, int(val_frac * len(stems)))
    val_stems   = stems[:split]
    train_stems = stems[split:]
    train_idx = [i for s in train_stems for i in video_groups[s]]
    val_idx   = [i for s in val_stems   for i in video_groups[s]]
    print(f"  Video split: {len(train_stems)} train videos / {len(val_stems)} val videos")
    print(f"  Sample split: {len(train_idx)} train / {len(val_idx)} val")
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx))


def train(data_dir: str, epochs: int = 80, lr: float = 1e-3, save_path: str = "models/cnn_gesture.pt"):
    dataset          = GestureDataset(data_dir)
    train_ds, val_ds = _split_by_video(dataset)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = PoseCNN(NUM_CLASSES).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {device} | {len(train_ds)} train / {len(val_ds)} val")
    best_val = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in train_dl:
            lm = batch["landmarks"].to(device)
            ft = batch["features"].to(device)
            lb = batch["label"].to(device)
            logits = model(lm, ft)
            loss   = loss_fn(logits, lb)

            opt.zero_grad()
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()
            total_loss += loss.item()
        sched.step()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_dl:
                lm = batch["landmarks"].to(device)
                ft = batch["features"].to(device)
                lb = batch["label"].to(device)
                preds = model(lm, ft).argmax(1)
                correct += (preds == lb).sum().item()
        val_acc = correct / len(val_ds)

        if val_acc > best_val:
            best_val = val_acc
            Path(save_path).parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | loss={total_loss/len(train_dl):.4f} | val_acc={val_acc:.3f}")

    print(f"Best val acc: {best_val:.3f} → {save_path}")
    return model


# ── Inference wrapper ──────────────────────────────────────────────────────────
class CNNClassifier:
    def __init__(self, model_path: str = "models/cnn_gesture.pt", confidence_thresh: float = 0.6):
        self.device = "cpu"
        self.model  = PoseCNN(NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.conf_thresh = confidence_thresh
        self._kp_order = KEYPOINT_NAMES

    @torch.no_grad()
    # Note: Type hinting as a string ("FeatureVector") to avoid needing the import
    def __call__(self, fv: "FeatureVector", raw_landmarks: dict = None) -> str:
        label, _ = self.predict_with_details(fv, raw_landmarks)
        return label

    @torch.no_grad()
    def predict_with_details(self, fv: "FeatureVector", raw_landmarks: dict = None):
        """
        Returns (label: str, confidence: float).
        confidence is the max softmax probability (0-1).
        Used by DetailedLogger in benchmark.py for confidence distribution analysis.
        """
        if raw_landmarks is None:
            return "IDLE", 0.0

        lm_arr = np.array([raw_landmarks[k] for k in self._kp_order], dtype=np.float32)
        lm_arr[:, :2] /= (lm_arr[:, :2].max() + 1e-8)

        lm_t  = torch.from_numpy(lm_arr).unsqueeze(0).to(self.device)
        fv_t  = torch.from_numpy(fv.to_array()).float().unsqueeze(0).to(self.device)

        logits = self.model(lm_t, fv_t)
        probs  = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)

        if conf.item() < self.conf_thresh:
            return "UNCERTAIN", float(conf.item())

        return IDX2GESTURE[idx.item()], float(conf.item())


# ── Data collection helper ────────────────────────────────────────────────────
def collect_samples(label: str, pose_frame, fv: "FeatureVector", out_dir: str = "data/cnn"):
    import os; os.makedirs(out_dir, exist_ok=True)
    lm_arr = np.array([pose_frame.landmarks[k] for k in KEYPOINT_NAMES], dtype=np.float32)
    idx    = GESTURE2IDX.get(label, 0)
    fname  = f"{out_dir}/{label}_{int(time.time()*1000)}.npz"
    np.savez_compressed(fname, landmarks=lm_arr, features=fv.to_array(), label=np.array(idx))
if __name__ == "__main__":
    train(data_dir="data/cnn")