import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/best_delay_model.pkl")

# Page settings
st.set_page_config(
    page_title="Shipping Delay Prediction",
    page_icon="🚚",
    layout="centered"
)

# Title
st.title("🚚 Shipping Delay Prediction System")

st.write(
    "This dashboard predicts whether a shipment is likely to be delayed "
    "using a trained machine learning model."
)

st.header("Enter Shipment Information")

# Input fields
asset_id = st.selectbox(
    "Asset ID",
    ["Truck_1", "Truck_2", "Truck_3", "Truck_4", "Truck_5",
     "Truck_6", "Truck_7", "Truck_8", "Truck_9", "Truck_10"]
)

latitude = st.number_input("Latitude", value=33.8938)
longitude = st.number_input("Longitude", value=35.5018)

inventory_level = st.number_input(
    "Inventory Level",
    min_value=0,
    value=120
)

temperature = st.number_input(
    "Temperature",
    value=25.5
)

humidity = st.number_input(
    "Humidity",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

traffic_status = st.selectbox(
    "Traffic Status",
    ["Clear", "Detour", "Heavy"]
)

waiting_time = st.number_input(
    "Waiting Time",
    min_value=0,
    value=45
)

user_transaction_amount = st.number_input(
    "User Transaction Amount",
    min_value=0,
    value=500
)

user_purchase_frequency = st.number_input(
    "User Purchase Frequency",
    min_value=0,
    value=3
)

asset_utilization = st.number_input(
    "Asset Utilization",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

demand_forecast = st.number_input(
    "Demand Forecast",
    min_value=0,
    value=220
)

hour = st.slider("Hour", 0, 23, 14)
month = st.slider("Month", 1, 12, 6)
day_of_week = st.slider("Day of Week", 0, 6, 2)

st.caption("Day of Week: 0 = Monday, 1 = Tuesday, 2 = Wednesday, ..., 6 = Sunday")

# Prediction button
if st.button("Predict Delay"):

    # Create one-row dataframe
    new_shipment = pd.DataFrame([{
        "Asset_ID": asset_id,
        "Latitude": latitude,
        "Longitude": longitude,
        "Inventory_Level": inventory_level,
        "Temperature": temperature,
        "Humidity": humidity,
        "Traffic_Status": traffic_status,
        "Waiting_Time": waiting_time,
        "User_Transaction_Amount": user_transaction_amount,
        "User_Purchase_Frequency": user_purchase_frequency,
        "Asset_Utilization": asset_utilization,
        "Demand_Forecast": demand_forecast,
        "Hour": hour,
        "Month": month,
        "DayOfWeek": day_of_week
    }])

    # Predict
    prediction = model.predict(new_shipment)[0]
    probability = model.predict_proba(new_shipment)[0]

    not_delayed_probability = probability[0]
    delayed_probability = probability[1]

    # Show result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Delayed")
    else:
        st.success("Prediction: Not Delayed")

    st.write(f"Not Delayed Probability: {not_delayed_probability:.2f}")
    st.write(f"Delayed Probability: {delayed_probability:.2f}")

    st.progress(float(delayed_probability))