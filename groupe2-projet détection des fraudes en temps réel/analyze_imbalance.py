import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load original data for analysis
df = pd.read_csv('dataset/creditcard.csv')

# Basic statistics
total_transactions = len(df)
fraud_transactions = df['Class'].sum()
legitimate_transactions = total_transactions - fraud_transactions
fraud_ratio = (fraud_transactions / total_transactions) * 100

print("=== PHASE 1.3: IMBALANCE ANALYSIS ===")
print(f"Total transactions: {total_transactions:,}")
print(f"Legitimate transactions: {legitimate_transactions:,}")
print(f"Fraud transactions: {fraud_transactions:,}")
print(f"Fraud ratio: {fraud_ratio:.2f}%")
print(f"Imbalance ratio: 1:{legitimate_transactions/fraud_transactions:.0f}")

# Load preprocessed data
data = np.load('results/preprocessed_data.npz')
y_train = data['y_train']
y_test = data['y_test']

# Check stratification
train_fraud = np.sum(y_train)
train_legitimate = len(y_train) - train_fraud
test_fraud = np.sum(y_test)
test_legitimate = len(y_test) - test_fraud

print("\n=== TRAIN/TEST SPLIT VERIFICATION ===")
print(f"Train set: {len(y_train):,} total ({train_legitimate:,} legit, {train_fraud} fraud)")
print(f"Train fraud ratio: {train_fraud/len(y_train)*100:.3f}%")
print(f"Test set: {len(y_test):,} total ({test_legitimate:,} legit, {test_fraud} fraud)")
print(f"Test fraud ratio: {test_fraud/len(y_test)*100:.3f}%")

# Analysis of Amount feature for fraud vs legitimate
print("\n=== AMOUNT ANALYSIS ===")
fraud_amounts = df[df['Class'] == 1]['Amount']
legit_amounts = df[df['Class'] == 0]['Amount']

print(f"Fraud transactions - Mean: ${fraud_amounts.mean():.2f}, Median: ${fraud_amounts.median():.2f}, Max: ${fraud_amounts.max():.2f}")
print(f"Legitimate transactions - Mean: ${legit_amounts.mean():.2f}, Median: ${legit_amounts.median():.2f}, Max: ${legit_amounts.max():.2f}")

# Time analysis
print("\n=== TIME ANALYSIS ===")
fraud_times = df[df['Class'] == 1]['Time']
legit_times = df[df['Class'] == 0]['Time']

print(f"Fraud transactions - Time range: {fraud_times.min()/3600:.1f}h to {fraud_times.max()/3600:.1f}h")
print(f"Legitimate transactions - Time range: {legit_times.min()/3600:.1f}h to {legit_times.max()/3600:.1f}h")

print("\n✅ Phase 1.3 completed: Imbalance analysis done")
print(f"Key finding: Dataset is highly imbalanced with fraud ratio of {fraud_ratio:.2f}%")