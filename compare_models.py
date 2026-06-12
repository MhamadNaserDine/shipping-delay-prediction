import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_logistics.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# 2. Separate features and target
X = df.drop(columns=["Logistics_Delay"])
y = df["Logistics_Delay"]

# 3. Detect categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

print("\nCategorical columns:")
print(categorical_cols)

print("\nNumerical columns:")
print(numerical_cols)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Preprocessing for Logistic Regression
preprocessor_scaled = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# 6. Preprocessing for tree models
preprocessor_tree = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# 7. Define models
models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor_scaled),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),

    "XGBoost": Pipeline(steps=[
        ("preprocessor", preprocessor_tree),
        ("model", XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss"
        ))
    ])
}

# 8. Train and evaluate each model
results = []

best_model = None
best_model_name = ""
best_f1 = 0

for name, pipeline in models.items():
    print("\n" + "=" * 60)
    print(f"Training model: {name}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

# 9. Show comparison table
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("Model Comparison:")
print(results_df)

# 10. Save best model
joblib.dump(best_model, "models/best_delay_model.pkl")

print("\nBest model:", best_model_name)
print("Best F1-score:", best_f1)
print("Best model saved in models/best_delay_model.pkl")