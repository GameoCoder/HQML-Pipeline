import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import json
import time
from datetime import datetime, timezone

# Import your preprocessing function to grab the untouched Test Set
from process import process_data

# 1. Define the exact same architecture used in Training
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
    print("Loading test data...")
    csv_path = '../../1-datasets/tcga-lung-cell/tcga_dataset.csv'
    _, _, (X_test, y_test) = process_data(csv_path)
    
    # Grab the first patient in the test set
    patient_features = X_test[0]
    actual_label = "Tumor" if y_test[0] == 1 else "Normal"
    
    print("Loading Quantum Model...")
    model = HybridVQC()
    # Note: Ensure this matches the filename you saved in Phase 2
    model.load_state_dict(torch.load('../../3-quantum-model-training-loop/tcga-lung-cell/hybrid_vqc_25q.pt')) 
    model.eval()
    
    # Prepare tensor (Batch size of 1)
    x_tensor = torch.tensor(patient_features, dtype=torch.float32).unsqueeze(0)
    
    print("Running cuQuantum simulation...")
    start_time = time.time()
    with torch.no_grad():
        # Get raw Pauli-Z expectation value directly from the quantum layer
        pauli_z_val = model.qlayer(x_tensor).item()
        
        # Get final probability from the full hybrid model
        tumor_prob = model(x_tensor).item()
        
    latency = (time.time() - start_time) * 1000 # Convert to ms
    
    # Determine classification
    prediction_label = "Tumor" if tumor_prob >= 0.5 else "Normal"
    confidence = (tumor_prob if tumor_prob >= 0.5 else 1 - tumor_prob) * 100

    # 2. Build the Streamlined JSON Payload
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
    
    # 3. Save and Print JSON
    json_output = json.dumps(report, indent=2)
    print("\n--- INFERENCE COMPLETE ---")
    print(json_output)
    
    with open("clinical_report_from_inference_controller.json", "w") as f:
        f.write(json_output)

if __name__ == "__main__":
    run_inference()