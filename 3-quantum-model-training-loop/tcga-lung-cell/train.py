import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
import numpy as np
import time

from process import process_data

## 1. Setup Data and Hyperparameters
csv_path = '../../1-datasets/tcga-lung-cell/tcga_dataset.csv'
(X_train, y_train), (X_val, y_val), _ = process_data(csv_path)

batch_size = 32
epochs = 15
learning_rate = 0.01

# Convert numpy arrays to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Create DataLoaders for efficient GPU batching
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

## 2. Define the Hybrid VQC Architecture
n_qubits = 14
n_layers = 4
dev = qml.device("lightning.gpu", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class HybridVQC(nn.Module):
    def __init__(self):
        super(HybridVQC, self).__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        q_out = self.qlayer(x).unsqueeze(-1) 
        return self.sigmoid(self.linear(q_out))

model = HybridVQC()

## 3. Training Loop
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"\nStarting training on {dev.name}...")
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        val_preds = (val_outputs >= 0.5).float()
        val_acc = (val_preds == y_val_t).float().mean().item()
        
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] - "
          f"Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

## 4. Save the Model
# Method A: Standard PyTorch state dictionary (Recommended)
torch.save(model.state_dict(), 'hybrid_vqc_25q.pt')

# Method B: Raw Quantum Weights as .npy (As requested)
quantum_weights = model.qlayer.weights.detach().cpu().numpy()
np.save('quantum_weights.npy', quantum_weights)

print("\nTraining complete. Model weights saved.")