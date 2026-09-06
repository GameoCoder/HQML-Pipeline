import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import pandas as pd
import joblib
import json
import time
from datetime import datetime, timezone

# 1. Define Model Architecture (Keep your 14-qubit HybridVQC class here)
n_qubits = 14
n_layers = 4
dev = qml.device("lightning.gpu", wires=n_qubits)

@qml.qnode(dev, interface="torch")
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


def run_inference():
    print("Loading saved pipeline artifacts...")
    scaler = joblib.load('../../2-data-processing/tcga-lung-cell/weights/standard_scaler.joblib')
    pca = joblib.load('../../2-data-processing/tcga-lung-cell/weights/pca_14.joblib')
    minmax = joblib.load('../../2-data-processing/tcga-lung-cell/weights/minmax_scaler.joblib')
    
    print("Extracting raw test patient...")
    # Simulating a raw data payload coming from your backend
    csv_path = '../../1-datasets/tcga-lung-cell/tcga_dataset.csv'
    df = pd.read_csv(csv_path, header=None, sep=None, engine='python')
    
    # Grab the raw FPKM values for one patient (e.g., column index 1)
    # Reshape to (1, n_features) because sklearn expects 2D arrays
    raw_features = df.iloc[2:, 1].values.astype(np.float32).reshape(1, -1)
    raw_features = np.nan_to_num(raw_features, nan=0.0)
    
    # Grab their actual label for verification
    actual_label = "Tumor" if df.iloc[1, 1].strip().lower() == 'tumor' else "Normal"
    
    print("Applying transformations (Inference Mode)...")
    # STRICT RULE: Use .transform() only! No .fit()!
    x_scaled = scaler.transform(raw_features)
    x_pca = pca.transform(x_scaled)
    x_final = minmax.transform(x_pca)
    
    print("Loading Quantum Model weights...")
    model = HybridVQC()
    model.load_state_dict(torch.load('../../3-quantum-model-training-loop/tcga-lung-cell/hybrid_vqc_25q.pt'))
    model.eval()
    
    # Prepare tensor
    x_tensor = torch.tensor(x_final, dtype=torch.float32)
    
    print("Running cuQuantum simulation...")
    start_time = time.time()
    with torch.no_grad():
        pauli_z_val = model.qlayer(x_tensor).item()
        tumor_prob = model(x_tensor).item()
        
    latency = (time.time() - start_time) * 1000
    
    # Determine classification
    prediction_label = "Tumor" if tumor_prob >= 0.5 else "Normal"
    confidence = (tumor_prob if tumor_prob >= 0.5 else 1 - tumor_prob) * 100
    
    # Build the Streamlined JSON Payload
    report = {
        "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "primary_diagnosis": {
            "label": prediction_label,
            "actual_ground_truth": actual_label,
            "binary_class": 1 if prediction_label == "Tumor" else 0,
            "confidence_percentage": round(confidence, 2),
            "quantum_expectation_value": round(pauli_z_val, 3)
        },
        "quantum_model_metrics": {
            "id": f"vqc_{n_qubits}q",
            "latency_ms": round(latency, 1),
            "qubits_used": n_qubits,
            "circuit_depth": n_layers
        }
    }
    
    json_output = json.dumps(report, indent=2)
    print("\n--- INFERENCE COMPLETE ---")
    print(json_output)
    
    with open("clinical_report-production.json", "w") as f:
        f.write(json_output)

if __name__ == "__main__":
    run_inference()