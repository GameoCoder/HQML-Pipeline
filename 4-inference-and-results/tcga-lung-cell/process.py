import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def process_data(csv_path):
    print("Loading dataset...")
    # Load data without assuming a header row
    df = pd.read_csv(csv_path, header=None, sep=None, engine='python')
    
    # Based on image_c6e9ec.jpg:
    # Row index 0, Col index 1 to end: Patient IDs
    # Row index 1, Col index 1 to end: Labels ('normal', 'tumor', etc.)
    # Row index 2 to end, Col index 0: Gene names (e.g., A1BG)
    # Row index 2 to end, Col index 1 to end: FPKM Data
    
    # Safely extract and format labels using pandas string methods
    labels_series = df.iloc[1, 1:].astype(str).str.strip().str.lower()
    
    # Convert labels to binary (1 for 'tumor', 0 for anything else like 'normal')
    y = np.where(labels_series == 'tumor', 1, 0)
    
    # Extract feature data, ignoring the first column (Gene names)
    # Transpose (.T) so rows = patients, cols = genes
    X_raw = df.iloc[2:, 1:].values.T.astype(np.float32)
    
    # Fill any missing numerical values with 0.0
    X_raw = np.nan_to_num(X_raw, nan=0.0) 
    
    print(f"Successfully extracted {X_raw.shape[0]} patients and {X_raw.shape[1]} genes.")
    
    # Train-Validation-Test Split (80:10:10)
    # First split: 80% Train, 20% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    # Second split: 10% Val, 10% Test (which is 50% of the 20% Temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Pipeline Initialization
    scaler = StandardScaler()
    pca = PCA(n_components=14, random_state=42)
    # Scale between 0 and Pi for quantum angle embedding
    minmax = MinMaxScaler(feature_range=(0, np.pi))
    
    # Fit and Transform
    print("Fitting Scalers and PCA...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_train_final = minmax.fit_transform(X_train_pca)
    
    X_val_final = minmax.transform(pca.transform(scaler.transform(X_val)))
    X_test_final = minmax.transform(pca.transform(scaler.transform(X_test)))
    
    # Save the pipeline components
    joblib.dump(scaler, 'standard_scaler.joblib')
    joblib.dump(pca, 'pca_14.joblib')
    joblib.dump(minmax, 'minmax_scaler.joblib')
    print("Preprocessing complete. Scalers and PCA saved.")
    
    return (X_train_final, y_train), (X_val_final, y_val), (X_test_final, y_test)

process_data('../../1-datasets/tcga-lung-cell/tcga_dataset.csv')