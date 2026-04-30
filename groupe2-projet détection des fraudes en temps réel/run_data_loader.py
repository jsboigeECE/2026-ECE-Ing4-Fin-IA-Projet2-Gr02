import sys
import os
sys.path.append('src/preprocessing')
from data_loader import load_creditcard_data, preprocess_creditcard_data, save_preprocessed_data

# Load and preprocess data
df = load_creditcard_data()
X_train, X_test, y_train, y_test, scaler = preprocess_creditcard_data(df)

# Save results
save_preprocessed_data(X_train, X_test, y_train, y_test, scaler)