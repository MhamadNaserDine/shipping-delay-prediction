import pandas as pd

# Load the dataset
df = pd.read_csv("data/smart_logistics.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show dataset shape
print("\nDataset shape:")
print(df.shape)

# Show column names
print("\nColumns:")
print(df.columns)

# Show missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check target column
print("\nTarget column values:")
print(df["Logistics_Delay"].value_counts())