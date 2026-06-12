import pandas as pd

# Load dataset
df = pd.read_csv("data/smart_logistics.csv")

print("Original shape:", df.shape)

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Create useful time features
df["Hour"] = df["Timestamp"].dt.hour
df["Month"] = df["Timestamp"].dt.month
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

# Drop columns that we should not use for training
df = df.drop(columns=["Timestamp", "Logistics_Delay_Reason", "Shipment_Status"])

# Check result
print("New shape:", df.shape)
print("\nColumns after cleaning:")
print(df.columns.tolist())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned data
df.to_csv("data/cleaned_logistics.csv", index=False)

print("\nCleaned dataset saved successfully!")