#!/usr/bin/env python3
"""
Run preprocessing script
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.data_loader import load_creditcard_data, preprocess_creditcard_data, save_preprocessed_data

if __name__ == "__main__":
    try:
        # Load and preprocess data
        df = load_creditcard_data('dataset/creditcard.csv')
        X_train, X_test, y_train, y_test, scaler = preprocess_creditcard_data(df)

        # Save results
        save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, 'results/')

        print("✅ Phase 1.2 completed successfully!")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)