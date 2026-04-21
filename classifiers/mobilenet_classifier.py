import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
from collections import defaultdict
import random
import os

# --- 1. CONFIGURATION ---
GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "MOVE"]
NUM_CLASSES = len(GESTURE_CLASSES)
FRAMES_PER_STACK = 15  # 0.5s context

# --- 2. RAM-CACHED & AUGMENTED DATASET ---
class GestureBallDataset(Dataset):
    def __init__(self, data_dir, video_stems, num_frames=15, is_training=False):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.is_training = is_training
        self.samples = []
        self.image_cache = {} 

        print(f"--- Loading {len(video_stems)} videos into RAM ---")
        grouped_frames = defaultdict(list)
        all_unique_files = set()
        
        files = sorted(list(self.data_dir.iterdir()))
        for file in files:
            if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]: continue
            parts = file.stem.split("_")
            if len(parts) < 3: continue
            vid_stem = "_".join(parts[1:-1])
            if vid_stem in video_stems:
                label_idx = GESTURE_CLASSES.index(parts[0].upper())
                grouped_frames[vid_stem].append((int(parts[-1]), file, label_idx))
                all_unique_files.add(file)

        # Decoding images once into RAM as tensors
        for file in tqdm(all_unique_files, desc="Caching JPEGs"):
            img = Image.open(file).convert("RGB").resize((256, 256))
            self.image_cache[file] = TF.to_tensor(img)

        for vid_stem, frames in grouped_frames.items():
            frames.sort(key=lambda x: x[0])
            for i in range(len(frames) - self.num_frames + 1):
                self.samples.append(frames[i : i + self.num_frames])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sequence = self.samples[idx]
        tensors = [self.image_cache[fpath] for _, fpath, _ in sequence]
        label = sequence[-1][2]
        
        if self.is_training:
            # Synchronized "Spatial Brutality"
            top, left = random.randint(0, 32), random.randint(0, 32)
            brightness = random.uniform(0.8, 1.2)
            tensors = [TF.adjust_brightness(t, brightness) for t in tensors]
            tensors = [TF.crop(t, top, left, 224, 224) for t in tensors]
        else:
            tensors = [TF.center_crop(t, [224, 224]) for t in tensors]
        
        tensors = [TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for t in tensors]
        return torch.cat(tensors, dim=0), label

# --- 3. STACKED MOBILENET ARCHITECTURE ---
class StackedMobileNet(nn.Module):
    def __init__(self, num_frames=15, num_classes=4):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Modify first layer for 15 frames (45 channels)
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            num_frames * 3, old_conv.out_channels, 
            kernel_size=3, stride=2, padding=1, bias=False
        )
        # Kaiming init for the new expanded layer
        nn.init.kaiming_normal_(self.backbone.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x): return self.backbone(x)

# --- 4. TRAINING FUNCTION ---
def train_gesture_ball(data_dir, epochs=15):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. SPLIT DATA
    all_files = list(Path(data_dir).glob("*.jpg"))
    video_stems = list(set("_".join(f.stem.split("_")[1:-1]) for f in all_files if len(f.stem.split("_")) >= 3))
    random.shuffle(video_stems)
    split = int(len(video_stems) * 0.8)

    train_ds = GestureBallDataset(data_dir, video_stems[:split], is_training=True)
    val_ds = GestureBallDataset(data_dir, video_stems[split:], is_training=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, pin_memory=True, num_workers=0)

    # 2. INITIALIZE MODEL
    model = StackedMobileNet(num_frames=15, num_classes=4)
    device_count = torch.cuda.device_count()
    
    if device_count > 1:
        print(f"🚀 Using {device_count} GPUs via DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Track the best accuracy to know when to save
    best_val_acc = 0.0

    print("\n--- Training Loop Starting ---")
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # --- SAVE IF BEST ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = "gesture_ball_best.pt"
            
            # If using DataParallel, save model.module.state_dict()
            # This makes the saved file usable on single-GPU systems later
            state_to_save = model.module.state_dict() if device_count > 1 else model.state_dict()
            
            torch.save(state_to_save, save_path)
            print(f"  ✨ New High Score! Saved to {save_path}")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}")