import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# Load data
df = pd.read_csv('dataset/creditcard.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Preprocess
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
features_to_scale = ['Amount', 'Time']
X_scaled = X.copy()
X_scaled[features_to_scale] = scaler.fit_transform(X[features_to_scale])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("Preprocessing completed:")
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Save
os.makedirs('results', exist_ok=True)
np.savez_compressed('results/preprocessed_data.npz',
                   X_train=X_train.values,
                   X_test=X_test.values,
                   y_train=y_train.values,
                   y_test=y_test.values)
joblib.dump(scaler, 'results/scaler.pkl')

print("Data saved to results/")
print("Phase 1.2 completed!")