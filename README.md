# Shipping Delay Prediction System

This project is a machine learning system that predicts whether a shipment is likely to be delayed using logistics-related data.

The goal of this project is to demonstrate how artificial intelligence can support logistics decision-making by predicting delay risk from shipment, traffic, demand, and asset-related features.

---

## Project Objective

The objective is to build an end-to-end AI project for shipment delay prediction.

The system takes shipment information as input and predicts:

- Delayed
- Not Delayed

It also returns the delay probability.

---

## Problem Type

This is a binary classification problem.

- 0 = Not Delayed
- 1 = Delayed

---

## Dataset

The dataset contains 1000 logistics records and 16 original columns.

Main features include:

- Asset ID
- Latitude
- Longitude
- Inventory Level
- Temperature
- Humidity
- Traffic Status
- Waiting Time
- User Transaction Amount
- User Purchase Frequency
- Asset Utilization
- Demand Forecast
- Timestamp
- Logistics Delay

The target column is:

```text
Logistics_Delay