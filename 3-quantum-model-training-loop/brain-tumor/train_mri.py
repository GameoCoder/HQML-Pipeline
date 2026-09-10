import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml

# 1. Setup Architecture Parameters
n_qubits = 14
n_layers = 4
num_classes = 4
batch_size = 32
epochs = 15
learning_rate = 0.01

data_dir = Path("mri_processed")

print("Loading processed numpy arrays...")
X_train = np.load(data_dir / 'X_train.npy')
y_train = np.load(data_dir / 'y_train.npy')
X_test = np.load(data_dir / 'X_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

# Convert arrays to PyTorch tensors
# Target labels are class indices (0, 1, 2, 3), requiring torch.long for CrossEntropyLoss
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=batch_size,
    shuffle=True
)

# 2. Define the Multi-Observable Quantum Circuit
dev = qml.device("lightning.gpu", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    # Angle encoding across all 14 wires using the (0 to pi) features
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    
    # Entangling variational layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Measure 4 individual qubits to represent the 4 target classes
    return [qml.expval(qml.PauliZ(i)) for i in range(num_classes)]

class HybridMRIVQC(nn.Module):
    def __init__(self):
        super(HybridMRIVQC, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        # Calibrates raw quantum expectation values [-1, 1] into classification logits
        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        # qlayer returns shape: (batch_size, 4)
        q_out = self.qlayer(x)
        logits = self.classifier(q_out)
        return logits

model = HybridMRIVQC()

# 3. Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"\nTraining 4-Class Hybrid VQC on {dev.name}...")
print(f"Total Training Samples: {len(X_train)} | Total Test Samples: {len(X_test)}")

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_train += (preds == batch_y).sum().item()
        total_train += batch_y.size(0)

    train_loss = running_loss / total_train
    train_acc = (correct_train / total_train) * 100.0

    # Evaluation on the Test Set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_loss = criterion(test_logits, y_test_t).item()
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item() * 100.0

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
          f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
          f"Test Loss: {test_loss:.4f} (Acc: {test_acc:.2f}%)")

# 4. Save Trained Model Weights
model_save_path = data_dir / 'hybrid_resnet_vqc_14q.pt'
torch.save(model.state_dict(), model_save_path)
print(f"\nModel successfully saved to {model_save_path}")