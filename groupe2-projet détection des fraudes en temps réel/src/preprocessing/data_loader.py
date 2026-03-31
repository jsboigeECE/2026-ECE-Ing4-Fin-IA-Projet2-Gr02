"""
Data Loader and Preprocessing for Credit Card Fraud Detection
Phase 1.2: Preprocessing and cleaning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

def load_creditcard_data(file_path='dataset/creditcard.csv'):
    """
    Load the credit card fraud dataset

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    pd.DataFrame: Loaded dataframe
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_creditcard_data(df, test_size=0.3, random_state=42):
    """
    Preprocess the credit card data: normalization and train/test split

    Parameters:
    df (pd.DataFrame): Raw dataframe
    test_size (float): Proportion of test set
    random_state (int): Random state for reproducibility

    Returns:
    tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Features to scale
    features_to_scale = ['Amount', 'Time']

    # Initialize scaler
    scaler = StandardScaler()

    # Fit and transform scaling features
    X_scaled = X.copy()
    X_scaled[features_to_scale] = scaler.fit_transform(X[features_to_scale])

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Important for imbalanced data
    )

    print("✅ Preprocessing completed:")
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Features scaled: {features_to_scale}")

    return X_train, X_test, y_train, y_test, scaler

def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, output_dir='results/'):
    """
    Save preprocessed data and scaler

    Parameters:
    X_train, X_test, y_train, y_test: Preprocessed data
    scaler: Fitted scaler
    output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save data
    np.savez_compressed(f'{output_dir}/preprocessed_data.npz',
                       X_train=X_train.values,
                       X_test=X_test.values,
                       y_train=y_train.values,
                       y_test=y_test.values)

    # Save scaler
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')

    print(f"✅ Preprocessed data saved to {output_dir}")

if __name__ == "__main__":
    # Load and preprocess data
    df = load_creditcard_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_creditcard_data(df)

    # Save results
    save_preprocessed_data(X_train, X_test, y_train, y_test, scaler)