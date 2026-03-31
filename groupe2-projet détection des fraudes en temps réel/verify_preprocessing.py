import numpy as np
import joblib
import os

# Load the preprocessed data
data = np.load('results/preprocessed_data.npz')
print("Preprocessed data loaded:")
print(f"X_train shape: {data['X_train'].shape}")
print(f"X_test shape: {data['X_test'].shape}")
print(f"y_train shape: {data['y_train'].shape}")
print(f"y_test shape: {data['y_test'].shape}")

# Check class distribution
unique_train, counts_train = np.unique(data['y_train'], return_counts=True)
unique_test, counts_test = np.unique(data['y_test'], return_counts=True)

print(f"\nTrain class distribution: {dict(zip(unique_train, counts_train))}")
print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

# Check if scaler exists
if os.path.exists('results/scaler.pkl'):
    scaler = joblib.load('results/scaler.pkl')
    print("Scaler loaded successfully")
else:
    print("Scaler not found")

print("Phase 1.2 verification completed!")