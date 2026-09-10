import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

def extract_features(dataloader, model, device):
    """Passes images through ResNet-50 to extract 2048D vectors."""
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            # Flatten the output from the pooling layer
            outputs = model(imgs).squeeze() 
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
            
    return np.vstack(features), np.concatenate(labels)

def process_mri_data(dataset_dir):
    print("Initializing ResNet-50 Feature Extractor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Pre-trained ResNet-50
    # We use weights=models.ResNet50_Weights.DEFAULT for best ImageNet weights
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Remove the final classification head (fc layer) to expose the 2048D bottleneck
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    
    # Standard ResNet image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # OS-agnostic path handling
    base_path = Path(dataset_dir)
    train_dir = base_path / 'Training'
    test_dir = base_path / 'Testing'
    
    print("Loading Image Datasets...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    
    # Batch size 32 is safe for the RTX 2050 during static inference
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 1. Extract 2048D Features
    print("Extracting deep spatial features from Training set (This may take a minute)...")
    X_train_raw, y_train = extract_features(train_loader, resnet, device)
    
    print("Extracting features from Testing set...")
    X_test_raw, y_test = extract_features(test_loader, resnet, device)
    
    # 2. PCA Compression (2048 -> 14 for 14 qubits)
    print("Fitting PCA to reduce to 14 dimensions...")
    pca = PCA(n_components=14, random_state=42)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)
    
    # 3. MinMax Scaling to Quantum Phase Angles (0 to Pi)
    print("Scaling values to [0, Pi]...")
    minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_train_final = minmax.fit_transform(X_train_pca)
    X_test_final = minmax.transform(X_test_pca)
    
    # 4. Save Artifacts
    os.makedirs("mri_processed", exist_ok=True)
    
    # Save the pipeline tools
    joblib.dump(pca, 'mri_processed/pca_14.joblib')
    joblib.dump(minmax, 'mri_processed/minmax_scaler.joblib')
    
    # Save the class mapping (e.g., 0: glioma, 1: meningioma...)
    class_mapping = {v: k for k, v in train_dataset.class_to_idx.items()}
    joblib.dump(class_mapping, 'mri_processed/class_mapping.joblib')
    
    # Save the processed numpy arrays for Phase 2 Training
    np.save('mri_processed/X_train.npy', X_train_final)
    np.save('mri_processed/y_train.npy', y_train)
    np.save('mri_processed/X_test.npy', X_test_final)
    np.save('mri_processed/y_test.npy', y_test)
    
    print("\n--- Preprocessing Complete ---")
    print(f"Train Shape: {X_train_final.shape}")
    print(f"Test Shape: {X_test_final.shape}")
    print("Saved all artifacts to /mri_processed/")

if __name__ == "__main__":
    # Point this to where you extracted the Kaggle dataset
    process_mri_data('../../1-datasets/brain-tumor')