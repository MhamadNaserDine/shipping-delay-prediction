import pandas as pd
import joblib
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


# Get project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
model_path = BASE_DIR / "models" / "best_delay_model.pkl"
model = joblib.load(model_path)


# Create FastAPI app
app = FastAPI(
    title="Shipping Delay Prediction API",
    description="API for predicting shipment delay risk using a trained machine learning model.",
    version="1.0"
)


# Define the input structure
class ShipmentInput(BaseModel):
    Asset_ID: str
    Latitude: float
    Longitude: float
    Inventory_Level: int
    Temperature: float
    Humidity: float
    Traffic_Status: str
    Waiting_Time: int
    User_Transaction_Amount: int
    User_Purchase_Frequency: int
    Asset_Utilization: float
    Demand_Forecast: int
    Hour: int
    Month: int
    DayOfWeek: int


@app.get("/")
def home():
    return {
        "message": "Shipping Delay Prediction API is running."
    }


@app.post("/predict")
def predict_delay(data: ShipmentInput):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    result = "Delayed" if prediction == 1 else "Not Delayed"

    return {
        "prediction": result,
        "not_delayed_probability": round(float(probability[0]), 2),
        "delayed_probability": round(float(probability[1]), 2)
    }