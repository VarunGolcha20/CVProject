# """
# classifiers/cnn_classifier.py
# ==============================
# CNN-based single-frame gesture classifier.
# Input: (17,) feature vector OR (33, 4) raw landmark array
# Output: gesture label string

# Reviewer note: rule-based is too easy — this is the CNN approach.
# Architecture: small 1D CNN over the landmark sequence (body as a sequence of joints).
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path
# from pipeline import FeatureVector

# # ── Gesture classes ────────────────────────────────────────────────────────────
# GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "DEFEND", "MOVE_LEFT",
#                    "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"]
# NUM_CLASSES = len(GESTURE_CLASSES)
# IDX2GESTURE = dict(enumerate(GESTURE_CLASSES))
# GESTURE2IDX = {v: k for k, v in IDX2GESTURE.items()}


# # ── Model ─────────────────────────────────────────────────────────────────────
# class PoseCNN(nn.Module):
#     """
#     Treats the 10 foot/leg/hip landmarks as a spatial sequence (each has 4 values).
#     1D convolutions capture spatial relationships between adjacent joints.
#     Very small footprint → fast inference on CPU.

#     Input shape:  (B, 10, 4)  — 10 keypoints × (x, y, z, visibility)
#     Output shape: (B, NUM_CLASSES)
#     """
#     def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
#         super().__init__()

#         self.conv1 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm1d(64)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.bn2   = nn.BatchNorm1d(128)
#         self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
#         self.bn3   = nn.BatchNorm1d(256)

#         self.pool  = nn.AdaptiveAvgPool1d(1)     # global average pool → (B, 256, 1)
#         self.drop  = nn.Dropout(dropout)

#         # Also accept the 17-dim feature vector as additional input
#         self.fc_feat = nn.Linear(16, 64)

#         self.fc1 = nn.Linear(256 + 64, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, landmarks: torch.Tensor,
#                 features: torch.Tensor) -> torch.Tensor:
#         """
#         landmarks: (B, 10, 4)  — raw joint coords
#         features:  (B, 17)     — engineered features from FeatureEngineer
#         """
#         # Conv branch
#         x = landmarks.permute(0, 2, 1)          # → (B, 4, 10)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x).squeeze(-1)            # (B, 256)

#         # Feature branch
#         f = F.relu(self.fc_feat(features))      # (B, 64)

#         # Fuse
#         out = torch.cat([x, f], dim=1)          # (B, 320)
#         out = self.drop(out)
#         out = F.relu(self.fc1(out))             # (B, 128)
#         return self.fc2(out)                    # (B, NUM_CLASSES) — logits


# # ── Dataset ───────────────────────────────────────────────────────────────────
# class GestureDataset(Dataset):
#     """
#     Expects a directory with .npz files, each containing:
#       landmarks: (10, 4)
#       features:  (17,)
#       label:     int in [0, NUM_CLASSES)
#     """
#     def __init__(self, data_dir: str):
#         self.files = list(Path(data_dir).glob("*.npz"))
#         assert self.files, f"No .npz files found in {data_dir}"

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         d = np.load(self.files[idx])
#         return {
#             "landmarks": torch.from_numpy(d["landmarks"]).float(),
#             "features":  torch.from_numpy(d["features"]).float(),
#             "label":     torch.tensor(int(d["label"]), dtype=torch.long),
#         }


# # ── Training ──────────────────────────────────────────────────────────────────
# def train(data_dir: str, epochs: int = 50, lr: float = 1e-3,
#           save_path: str = "models/cnn_gesture.pt"):
#     """
#     Quick training loop. Run on Colab/Kaggle GPU.
#     Expected dataset size: ~2000 samples per class minimum.
#     """
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import classification_report

#     dataset = GestureDataset(data_dir)
#     n_train = int(0.8 * len(dataset))
#     train_ds, val_ds = torch.utils.data.random_split(
#         dataset, [n_train, len(dataset) - n_train])

#     train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
#     val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model  = PoseCNN(NUM_CLASSES).to(device)
#     opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
#     loss_fn = nn.CrossEntropyLoss()

#     print(f"Training on {device} | {len(train_ds)} train / {len(val_ds)} val")
#     best_val = 0

#     for epoch in range(1, epochs+1):
#         model.train()
#         total_loss = 0
#         for batch in train_dl:
#             lm = batch["landmarks"].to(device)
#             ft = batch["features"].to(device)
#             lb = batch["label"].to(device)
#             logits = model(lm, ft)
#             loss   = loss_fn(logits, lb)
#             opt.zero_grad(); loss.backward(); opt.step()
#             total_loss += loss.item()
#         sched.step()

#         # Validation
#         model.eval()
#         correct = 0
#         with torch.no_grad():
#             for batch in val_dl:
#                 lm = batch["landmarks"].to(device)
#                 ft = batch["features"].to(device)
#                 lb = batch["label"].to(device)
#                 preds = model(lm, ft).argmax(1)
#                 correct += (preds == lb).sum().item()
#         val_acc = correct / len(val_ds)

#         if val_acc > best_val:
#             best_val = val_acc
#             Path(save_path).parent.mkdir(exist_ok=True)
#             torch.save(model.state_dict(), save_path)

#         if epoch % 5 == 0:
#             print(f"Epoch {epoch:03d} | loss={total_loss/len(train_dl):.4f} | val_acc={val_acc:.3f}")

#     print(f"Best val acc: {best_val:.3f} → {save_path}")
#     return model


# # ── Inference wrapper ──────────────────────────────────────────────────────────
# class CNNClassifier:
#     """Drop-in replacement for RuleBasedClassifier."""

#     def __init__(self, model_path: str = "models/cnn_gesture.pt",
#                  confidence_thresh: float = 0.6):
#         self.device = "cpu"   # deploy on CPU for low latency
#         self.model  = PoseCNN(NUM_CLASSES).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#         self.conf_thresh = confidence_thresh

#         # Keep reference to the KEYPOINT ordering from pipeline.py
#         from pipeline import KEYPOINTS
#         self._kp_order = list(KEYPOINTS.keys())

#     @torch.no_grad()
#     def __call__(self, fv: FeatureVector,
#                  raw_landmarks: dict = None) -> str:
#         """
#         fv: FeatureVector from FeatureEngineer
#         raw_landmarks: dict of name->(x,y,z,vis) from PoseFrame
#         """
#         if raw_landmarks is None:
#             # Fallback: can't run CNN without raw landmarks
#             return "IDLE"

#         # Build (10, 4) landmark tensor
#         lm_arr = np.array([raw_landmarks[k] for k in self._kp_order],
#                           dtype=np.float32)                # (10, 4)
#         # Normalise xy to [0,1]
#         lm_arr[:, :2] /= (lm_arr[:, :2].max() + 1e-8)

#         lm_t  = torch.from_numpy(lm_arr).unsqueeze(0)        # (1,10,4)
#         fv_t  = torch.from_numpy(fv.to_array()).unsqueeze(0)  # (1,17)

#         logits = self.model(lm_t.to(self.device), fv_t.to(self.device))
#         probs  = F.softmax(logits, dim=1)[0]
#         conf, idx = probs.max(0)

#         if conf.item() < self.conf_thresh:
#             return "UNCERTAIN"

#         return IDX2GESTURE[idx.item()]


# # ── Data collection helper ────────────────────────────────────────────────────
# def collect_samples(label: str, pose_frame, fv: FeatureVector,
#                     out_dir: str = "data/cnn"):
#     """
#     Call this during a labelled recording session.
#     Press a key per gesture to save a sample.
#     """
#     from pipeline import KEYPOINTS
#     import os; os.makedirs(out_dir, exist_ok=True)

#     lm_arr = np.array([pose_frame.landmarks[k] for k in KEYPOINTS], dtype=np.float32)
#     idx    = GESTURE2IDX.get(label, 0)
#     fname  = f"{out_dir}/{label}_{int(time.time()*1000)}.npz"

#     import time
#     np.savez_compressed(fname,
#         landmarks=lm_arr,
#         features=fv.to_array(),
#         label=np.array(idx))
# if __name__ == "__main__":
#     import numpy as np
#     import torch
#     from pathlib import Path
#     from classifiers.cnn_classifier import PoseCNN, NUM_CLASSES, IDX2GESTURE

#     # 1. Load one sample
#     data_dir = Path("data/cnn")
#     files = list(data_dir.glob("*.npz"))

#     assert files, "❌ No .npz files found in data/cnn"

#     sample_file = files[0]
#     data = np.load(sample_file)

#     landmarks = data["landmarks"]   # (10, 4)
#     features  = data["features"]    # (16,)
#     label     = int(data["label"])

#     print(f"Loaded file: {sample_file}")
#     print(f"Landmarks shape: {landmarks.shape}")
#     print(f"Features shape: {features.shape}")
#     print(f"Label: {label}")

#     # 2. Create model
#     model = PoseCNN(num_classes=NUM_CLASSES)

#     # (Optional) load trained weights
#     # model.load_state_dict(torch.load("models/cnn_gesture.pt"))
#     model.eval()

#     # 3. Convert to tensor
#     lm = torch.from_numpy(landmarks).float().unsqueeze(0)  # (1, 10, 4)
#     ft = torch.from_numpy(features).float().unsqueeze(0)   # (1, 16)

#     # 4. Forward pass
#     output = model(lm, ft)

#     print("\n--- CNN Real Data Check ---")
#     print(f"Output Shape: {output.shape} (Should be [1, {NUM_CLASSES}])")
#     print(f"Logits: {output.detach().numpy()}")

#     pred = output.argmax(dim=1).item()
#     print(f"Predicted class index: {pred}")
#     print(f"Ground truth index: {label}")
#     print(f"Predicted gesture: {IDX2GESTURE[pred]}")

#     print("✅ CNN model + data pipeline is working!")

#     train("data/cnn")
"""
classifiers/cnn_classifier.py
==============================
CNN-based single-frame gesture classifier.
Input: (17,) feature vector OR (33, 4) raw landmark array
Output: gesture label string

Reviewer note: rule-based is too easy — this is the CNN approach.
Architecture: small 1D CNN over the landmark sequence (body as a sequence of joints).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from pipeline import FeatureVector

# ── Gesture classes ────────────────────────────────────────────────────────────
GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "DEFEND", "MOVE_LEFT",
                   "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"]
NUM_CLASSES = len(GESTURE_CLASSES)
IDX2GESTURE = dict(enumerate(GESTURE_CLASSES))
GESTURE2IDX = {v: k for k, v in IDX2GESTURE.items()}


# ── Model ─────────────────────────────────────────────────────────────────────
class PoseCNN(nn.Module):
    """
    Treats the 10 foot/leg/hip landmarks as a spatial sequence (each has 4 values).
    1D convolutions capture spatial relationships between adjacent joints.
    Very small footprint → fast inference on CPU.

    Input shape:  (B, 10, 4)  — 10 keypoints × (x, y, z, visibility)
    Output shape: (B, NUM_CLASSES)
    """    
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)

        self.pool  = nn.AdaptiveAvgPool1d(1)     # global average pool → (B, 256, 1)
        self.drop  = nn.Dropout(dropout)

        # Also accept the 17-dim feature vector as additional input
        self.fc_feat = nn.Linear(32, 64)

        self.fc1 = nn.Linear(256 + 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, landmarks: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        """
        landmarks: (B, 10, 4)  — raw joint coords
        features:  (B, 32)     — engineered features from FeatureEngineer
        """
        # Conv branch
        x = landmarks.permute(0, 2, 1)          # → (B, 4, 10)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)            # (B, 256)

        # Feature branch
        f = F.relu(self.fc_feat(features))      # (B, 64)

        # Fuse
        out = torch.cat([x, f], dim=1)          # (B, 320)
        out = self.drop(out)
        out = F.relu(self.fc1(out))             # (B, 128)
        return self.fc2(out)                    # (B, NUM_CLASSES) — logits


# ── Dataset ───────────────────────────────────────────────────────────────────
class GestureDataset(Dataset):
    """
    Expects a directory with .npz files, each containing:
      landmarks: (10, 4)
      features:  (17,)
      label:     int in [0, NUM_CLASSES)
    """
    def __init__(self, data_dir: str):
        self.files = list(Path(data_dir).glob("*.npz"))
        assert self.files, f"No .npz files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return {
            "landmarks": torch.from_numpy(d["landmarks"]).float(),
            "features":  torch.from_numpy(d["features"]).float(),
            "label":     torch.tensor(int(d["label"]), dtype=torch.long),
        }


# ── Training ──────────────────────────────────────────────────────────────────
def _split_by_video(dataset, val_frac=0.2, seed=42):
    """
    Split by video stem so no video's frames appear in both train and val.
    Filename convention: LABEL_videostem_NNNNN.npz
    e.g. SHOOT_take3_00045.npz -> stem = "SHOOT_take3"
    """
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


def train(data_dir: str, epochs: int = 50, lr: float = 1e-3,
          save_path: str = "models/cnn_gesture.pt"):
    """
    Training loop with video-level train/val split.
    Run on Colab/Kaggle GPU.
    """
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
            opt.zero_grad(); loss.backward(); opt.step()
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
    """Drop-in replacement for RuleBasedClassifier."""

    def __init__(self, model_path: str = "models/cnn_gesture.pt",
                 confidence_thresh: float = 0.6):
        self.device = "cpu"   # deploy on CPU for low latency
        self.model  = PoseCNN(NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.conf_thresh = confidence_thresh

        # Keep reference to the KEYPOINT ordering from pipeline.py
        from pipeline import KEYPOINTS
        self._kp_order = list(KEYPOINTS.keys())

    @torch.no_grad()
    def __call__(self, fv: FeatureVector,
                 raw_landmarks: dict = None) -> str:
        """
        fv: FeatureVector from FeatureEngineer
        raw_landmarks: dict of name->(x,y,z,vis) from PoseFrame
        """
        if raw_landmarks is None:
            # Fallback: can't run CNN without raw landmarks
            return "IDLE"

        # Build (10, 4) landmark tensor
        lm_arr = np.array([raw_landmarks[k] for k in self._kp_order],
                          dtype=np.float32)                # (10, 4)
        # Normalise xy to [0,1]
        lm_arr[:, :2] /= (lm_arr[:, :2].max() + 1e-8)

        lm_t  = torch.from_numpy(lm_arr).unsqueeze(0)        # (1,10,4)
        fv_t  = torch.from_numpy(fv.to_array()).unsqueeze(0)  # (1,17)

        logits = self.model(lm_t.to(self.device), fv_t.to(self.device))
        probs  = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)

        if conf.item() < self.conf_thresh:
            return "UNCERTAIN"

        return IDX2GESTURE[idx.item()]


# ── Data collection helper ────────────────────────────────────────────────────
def collect_samples(label: str, pose_frame, fv: FeatureVector,
                    out_dir: str = "data/cnn"):
    """
    Call this during a labelled recording session.
    Press a key per gesture to save a sample.
    """
    from pipeline import KEYPOINTS
    import os; os.makedirs(out_dir, exist_ok=True)

    lm_arr = np.array([pose_frame.landmarks[k] for k in KEYPOINTS], dtype=np.float32)
    idx    = GESTURE2IDX.get(label, 0)
    fname  = f"{out_dir}/{label}_{int(time.time()*1000)}.npz"

    import time
    np.savez_compressed(fname,
        landmarks=lm_arr,
        features=fv.to_array(),
        label=np.array(idx))
if __name__ == "__main__":
    train(data_dir="data/cnn")