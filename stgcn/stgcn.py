import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

GESTURE_CLASSES = ["IDLE", "SHOOT", "PASS", "MOVE"]
CORE_JOINTS = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30]
NUM_FRAMES = 15

# --- 1. THE GRAPH ---
class Graph:
    def __init__(self):
        self.num_node = 12
        self.edges = [
            (0, 1), (0, 2), (1, 3),      # Shoulders to elbows
            (0, 4), (1, 5), (4, 5),      # Torso
            (4, 6), (5, 7),              # Hips to knees
            (6, 8), (7, 9),              # Knees to ankles
            (8, 10), (9, 11)             # Ankles to heels
        ]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[i, j] = 1; A[j, i] = 1
        for i in range(self.num_node): A[i, i] = 1
        D = np.diag(np.sum(A, axis=1) ** -0.5)
        return torch.tensor(np.dot(np.dot(D, A), D), dtype=torch.float32)

# --- 2. ST-GCN MODEL ARCHITECTURE ---
class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1))
        self.register_buffer('A', A)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = torch.einsum('nctu,uv->nctv', x, self.A) 
        x = self.temporal_conv(x)
        return self.relu(x)

class GestureSTGCN(nn.Module):
    def __init__(self, num_classes=4, in_channels=3):
        super().__init__()
        graph = Graph()
        self.data_bn = nn.BatchNorm1d(in_channels * graph.num_node)
        self.l1 = STGCN_Block(in_channels, 64, graph.A)
        self.l2 = STGCN_Block(64, 64, graph.A)
        self.l3 = STGCN_Block(64, 128, graph.A, stride=2)
        self.l4 = STGCN_Block(128, 128, graph.A)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        x = self.l4(self.l3(self.l2(self.l1(x))))
        return self.fc(x.mean(dim=(2, 3))) # Global Average Pooling

# --- 3. TRAINING ENGINE ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training ST-GCN on {device}...")

    try:
        data = torch.load("skeleton_dataset.pt")
        dataset = TensorDataset(data['X'], data['y'])
    except FileNotFoundError:
        print("❌ Error: Run 1_extract_skeletons.py first to generate the dataset!")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = GestureSTGCN(num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epochs = 40 # Fast training, higher epoch count

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                preds = model(inputs.to(device)).argmax(1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        
        if epoch % 5 == 0 or val_acc > best_acc:
            print(f"Epoch {epoch:02d} | Val Acc: {val_acc*100:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "stgcn_best.pt")

    print(f"\n✅ Training Complete. Best Validation Accuracy: {best_acc*100:.1f}%")

if __name__ == "__main__":
    train()