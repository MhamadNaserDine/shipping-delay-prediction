import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_logistics.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# 2. Separate features X and target y
X = df.drop(columns=["Logistics_Delay"])
y = df["Logistics_Delay"]

# 3. Detect categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:")
print(categorical_cols)

print("\nNumerical columns:")
print(numerical_cols)

# 4. Preprocessing
# OneHotEncoder converts text columns into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# 5. Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 6. Create full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# 7. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train model
pipeline.fit(X_train, y_train)

# 9. Predict on test data
y_pred = pipeline.predict(X_test)

# 10. Evaluate model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Save model
joblib.dump(pipeline, "models/delay_model.pkl")

print("\nModel saved successfully in models/delay_model.pkl")