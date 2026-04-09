import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os

# -----------------------------
# CLASS DEFINITIONS
# -----------------------------
GESTURE_CLASSES = [
    "IDLE", "SHOOT", "PASS", "MOVE"
]

NUM_CLASSES = len(GESTURE_CLASSES)
IDX2GESTURE = dict(enumerate(GESTURE_CLASSES))


# -----------------------------
# CUSTOM DATASET (PREFIX BASED)
# -----------------------------
class PrefixDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.samples = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for file in self.root_dir.iterdir():
            if file.suffix.lower() in valid_exts:
                # Extract prefix BEFORE underscore
                prefix = file.stem.split("_")[0].upper()

                if prefix not in GESTURE_CLASSES:
                    continue

                label = GESTURE_CLASSES.index(prefix)
                self.samples.append((file, label))

        if not self.samples:
            raise ValueError("No valid images found!")

        print(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# -----------------------------
# MODEL CLASS
# -----------------------------
class PretrainedImageClassifier:
    def __init__(self, model_path: str = None, confidence_thresh: float = 0.6):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_thresh = confidence_thresh

        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # Replace classifier head
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )

        self.model.to(self.device)
        self.model.eval()

        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def __call__(self, frame_crop) -> str:
        if frame_crop is None:
            return "IDLE"

        if isinstance(frame_crop, np.ndarray):
            frame_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
            frame_crop = Image.fromarray(frame_crop)

        img_t = self.transforms(frame_crop).unsqueeze(0).to(self.device)

        logits = self.model(img_t)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)

        if conf.item() < self.conf_thresh:
            return "UNCERTAIN"

        return IDX2GESTURE[idx.item()]


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_image_classifier(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    save_path: str = "models/mobilenet_gesture.pt"
):
    classifier = PretrainedImageClassifier()
    model = classifier.model
    device = classifier.device

    # ✅ USE PREFIX DATASET HERE
    dataset = PrefixDataset(root_dir=data_dir, transform=classifier.transforms)

    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    loss_fn = nn.CrossEntropyLoss()

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[1].parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.classifier[1].parameters(), lr=1e-3)

    print(f"Training on {device} | {len(train_ds)} train | {len(val_ds)} val")
    best_val = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()

        val_acc = correct / len(val_ds)

        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            Path(save_path).parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"Done! Best Validation Accuracy: {best_val:.3f}")
    return model


# -----------------------------
# INFERENCE ON FOLDER
# -----------------------------
def process_image_folder(folder_path: str, model_path: str = None):
    classifier = PretrainedImageClassifier(model_path=model_path, confidence_thresh=0.6)

    folder = Path(folder_path)
    if not folder.exists():
        print(f"Directory not found: {folder_path}")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]

    print(f"Found {len(image_files)} images\n")
    print("-" * 50)
    print(f"{'FILENAME':<25} | {'PREDICTION'}")
    print("-" * 50)

    for img_path in sorted(image_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        prediction = classifier(frame)
        print(f"{img_path.name:<25} | {prediction}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    TARGET_FOLDER = "/kaggle/input/datasets/varungolcha/cnnimages/cnn_images"

    train_image_classifier(TARGET_FOLDER)

    process_image_folder(
        folder_path=TARGET_FOLDER,
        model_path="models/mobilenet_gesture.pt"
    )