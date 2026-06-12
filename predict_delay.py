import pandas as pd
import joblib

# Load the saved best model
model = joblib.load("models/best_delay_model.pkl")

# Create one new shipment example
new_shipment = pd.DataFrame([{
    "Asset_ID": "Truck_7",
    "Latitude": 33.8938,
    "Longitude": 35.5018,
    "Inventory_Level": 120,
    "Temperature": 25.5,
    "Humidity": 60.0,
    "Traffic_Status": "Heavy",
    "Waiting_Time": 45,
    "User_Transaction_Amount": 500,
    "User_Purchase_Frequency": 3,
    "Asset_Utilization": 75.0,
    "Demand_Forecast": 220,
    "Hour": 14,
    "Month": 6,
    "DayOfWeek": 2
}])

# Make prediction
prediction = model.predict(new_shipment)[0]

# Get probability
probability = model.predict_proba(new_shipment)[0]

not_delayed_probability = probability[0]
delayed_probability = probability[1]

# Print result
print("Shipment Delay Prediction")
print("=" * 40)

if prediction == 1:
    print("Prediction: Delayed")
else:
    print("Prediction: Not Delayed")

print(f"Not delayed probability: {not_delayed_probability:.2f}")
print(f"Delayed probability: {delayed_probability:.2f}")