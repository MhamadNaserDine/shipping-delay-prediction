import pandas as pd

df = pd.read_csv("data/cleaned_logistics.csv")

print("Shipment_Status values:")
print(df["Shipment_Status"].value_counts())

print("\nTraffic_Status values:")
print(df["Traffic_Status"].value_counts())

print("\nAsset_ID values:")
print(df["Asset_ID"].value_counts())

print("\nRelationship between Shipment_Status and Logistics_Delay:")
print(pd.crosstab(df["Shipment_Status"], df["Logistics_Delay"]))

print("\nRelationship between Traffic_Status and Logistics_Delay:")
print(pd.crosstab(df["Traffic_Status"], df["Logistics_Delay"]))