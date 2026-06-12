import pandas as pd

# Load dataset
df = pd.read_csv("data/smart_logistics.csv")

# Show first rows
print("First 5 rows:")
print(df.head())

print("\n" + "=" * 50)

# Dataset size
print("Dataset shape:")
print(df.shape)

print("\n" + "=" * 50)

# Column names
print("Column names:")
print(df.columns.tolist())

print("\n" + "=" * 50)

# Data types
print("Data types:")
print(df.dtypes)

print("\n" + "=" * 50)

# Missing values
print("Missing values:")
print(df.isnull().sum())

print("\n" + "=" * 50)

# Basic statistics
print("Basic statistics:")
print(df.describe())

print("\n" + "=" * 50)

# Check target column
print("Target column distribution:")
print(df["Logistics_Delay"].value_counts())

print("\nTarget percentage:")
print(df["Logistics_Delay"].value_counts(normalize=True) * 100)