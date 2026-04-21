"""
classifiers/lstm_classifier.py
================================
LSTM-based temporal gesture classifier.
Operates on a sliding window of N frames — captures motion dynamics
that single-frame CNN cannot (e.g., the arc of a kick over time).

This directly addresses:
  - Shoot vs pass (temporal swing profile differs)
  - Running direction (consistent hip displacement over time)
  - Failure mode: jitter in pose (smoothed by sequence modelling)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
from pathlib import Path
from classifiers.cnn_classifier import GESTURE_CLASSES, NUM_CLASSES, IDX2GESTURE

# WINDOW = 30   # frames (~1 second at 30fps) — tune to gesture duration
WINDOW=35

# ── Model ─────────────────────────────────────────────────────────────────────
class GestureLSTM(nn.Module):
    """
    Input:  (B, T, 32) — T frames of FeatureVectors
    Output: (B, NUM_CLASSES)

    Architecture choices:
      - Bidirectional GRU (faster & fewer params than LSTM, similar accuracy)
      - Layer norm for training stability with small batches
      - Attention pooling: lets model focus on the key frame of the gesture
        (e.g., the moment of impact in a shoot)
    """
    def __init__(self, input_dim: int = 32, hidden: int = 128,
                 num_layers: int = 4, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.4):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.ln     = nn.LayerNorm(hidden)
        self.drop   = nn.Dropout(dropout)

        # Attention: learn which frame matters most
        self.attn   = nn.Linear(hidden, 1)

        self.fc1    = nn.Linear(hidden, 128)
        self.fc2    = nn.Linear(128, num_classes)
        self.input_norm = nn.LayerNorm(input_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 17)"""
        x = self.input_norm(x)
        out, _ = self.gru(x)                   # (B, T, 2H)
        out    = self.ln(out)

        # Attention pooling
        scores = self.attn(out).squeeze(-1)    # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (out * weights).sum(dim=1)   # (B, 2H) — weighted sum over time

        out = self.drop(context)
        out = F.relu(self.fc1(out))
        return self.fc2(out)                   # (B, NUM_CLASSES) logits


# ── Dataset ───────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """
    Each .npz file contains:
      sequence: (WINDOW, 17) — T frames of feature vectors
      label:    int
    """
    def __init__(self, data_dir: str):
        self.files = list(Path(data_dir).glob("*.npz"))
        assert self.files, f"No .npz files in {data_dir}"

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return {
            "sequence": torch.from_numpy(d["sequence"]).float(),
            "label":    torch.tensor(int(d["label"]), dtype=torch.long),
        }


# ── Training ──────────────────────────────────────────────────────────────────

def _split_by_video_lstm(dataset, val_frac=0.2, seed=42):
    """
    Split by video stem so no video appears in both train and val.
    LSTM filenames: LABEL_videostem.npz (no frame number suffix)
    so every file is already one independent clip — simple random split
    over files is correct here. But we still group by video stem in case
    you ever regenerate with sliding windows.
    """
    import random
    from collections import defaultdict
    random.seed(seed)
    video_groups = defaultdict(list)
    for i, f in enumerate(dataset.files):
        # LSTM files: SHOOT_take3.npz — no trailing frame number
        video_groups[f.stem].append(i)
    stems = list(video_groups.keys())
    random.shuffle(stems)
    split       = max(1, int(val_frac * len(stems)))
    val_stems   = stems[:split]
    train_stems = stems[split:]
    train_idx = [i for s in train_stems for i in video_groups[s]]
    val_idx   = [i for s in val_stems   for i in video_groups[s]]
    print(f"  Video split: {len(train_stems)} train / {len(val_stems)} val clips")
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, val_idx))


def train(data_dir: str = "data/lstm", epochs: int = 80,
          lr: float = 5e-4, save_path: str = "models/lstm_gesture.pt"):
    dataset          = SequenceDataset(data_dir)
    train_ds, val_ds = _split_by_video_lstm(dataset)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model    = GestureLSTM().to(device)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr*5, steps_per_epoch=len(train_dl), epochs=epochs)
    loss_fn  = nn.CrossEntropyLoss()

    best_val = 0
    print(f"LSTM training on {device} | window={WINDOW} frames")

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in train_dl:
            seq = batch["sequence"].to(device)
            lb  = batch["label"].to(device)
            logits = model(seq)
            loss   = loss_fn(logits, lb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_dl:
                preds = model(batch["sequence"].to(device)).argmax(1)
                correct += (preds == batch["label"].to(device)).sum().item()
        val_acc = correct / len(val_ds)

        if val_acc > best_val:
            best_val = val_acc
            Path(save_path).parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss={total_loss/len(train_dl):.4f} | val_acc={val_acc:.3f}")

    print(f"Best val acc: {best_val:.3f}")
    return model


# ── Inference wrapper ──────────────────────────────────────────────────────────
class LSTMClassifier:
    """
    Maintains an internal sliding window of FeatureVectors.
    Runs the model once the window is full.
    """
    def __init__(self, model_path: str = "models/lstm_gesture.pt",
                 window: int = WINDOW, confidence_thresh: float = 0.65):
        self.device = "cpu"
        self.model  = GestureLSTM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.window  = window
        self.buffer  = deque(maxlen=window)
        self.conf_thresh = confidence_thresh

    @torch.no_grad()
    def __call__(self, fv) -> str:
        label, _, _ = self.predict_with_details(fv)
        return label

    @torch.no_grad()
    def predict_with_details(self, fv):
        """
        Returns (label: str, confidence: float, attention_weights: list[float] | None).
        attention_weights is a list of length WINDOW (35) summing to ~1.0.
        High weight at frame i means the GRU used frame i most for its decision.
        Used by DetailedLogger in benchmark.py for attention analysis.
        """
        self.buffer.append(fv.to_array())

        if len(self.buffer) < self.window:
            return "BUFFERING", 0.0, None

        seq = np.stack(self.buffer, axis=0)
        t   = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)

        # Re-run forward pass manually to extract attention weights
        x       = self.model.input_norm(t)
        out, _  = self.model.gru(x)
        out     = self.model.ln(out)
        scores  = self.model.attn(out).squeeze(-1)          # (1, T)
        weights = F.softmax(scores, dim=1)                  # (1, T)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)  # (1, H)
        logits  = self.model.fc2(F.relu(self.model.fc1(self.model.drop(context))))

        probs     = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        attn_list = weights[0].cpu().numpy().tolist()

        if conf.item() < self.conf_thresh:
            return "UNCERTAIN", float(conf.item()), attn_list

        return IDX2GESTURE[idx.item()], float(conf.item()), attn_list


# ── Sequence data collection ───────────────────────────────────────────────────
class SequenceRecorder:
    """
    Use during labelled recording sessions.
    Fills a rolling buffer; call save() to write a labelled sequence.
    """
    def __init__(self, window: int = WINDOW, out_dir: str = "data/lstm"):
        self.window  = window
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._buf: deque = deque(maxlen=window)

    def push(self, fv):
        self._buf.append(fv.to_array())

    def save(self, label: str):
        from classifiers.cnn_classifier import GESTURE2IDX
        import time
        if len(self._buf) < self.window:
            print(f"Buffer not full yet ({len(self._buf)}/{self.window})")
            return
        seq  = np.stack(self._buf, axis=0)
        idx  = GESTURE2IDX.get(label, 0)
        fname = self.out_dir / f"{label}_{int(time.time()*1000)}.npz"
        np.savez_compressed(fname, sequence=seq, label=np.array(idx))
        print(f"Saved {fname}")
if __name__ == "__main__":
    train(data_dir="data/lstm")