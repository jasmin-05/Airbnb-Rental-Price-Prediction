# Airbnb Rental Price Prediction – Full Working Code
# Author: jasmin

# ----------------------------------------------------
# IMPORT LIBRARIES
# ----------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# If XGBoost is installed
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False


# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------
df = pd.read_csv("data/airbnb.csv")     # ← your file here

print("Dataset loaded successfully!")
print(df.head())


# ----------------------------------------------------
# BASIC CLEANING
# ----------------------------------------------------
# Remove rows with missing target
df = df.dropna(subset=['price'])

# Fill missing values
df = df.fillna(df.median(numeric_only=True))

print("Shape after cleaning: ", df.shape)


# ----------------------------------------------------
# SELECT FEATURES
# ----------------------------------------------------
numeric_features = ["bedrooms", "bathrooms", "accommodates", "latitude", "longitude"]
categorical_features = ["room_type", "neighbourhood"]

X = df[numeric_features + categorical_features]
y = df["price"]


# ----------------------------------------------------
# PREPROCESS PIPELINE
# ----------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# ----------------------------------------------------
# MODELS
# ----------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

if xgb_available:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )


# ----------------------------------------------------
# TRAIN/TEST SPLIT
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------------------------------
# TRAIN & EVALUATE EACH MODEL
# ----------------------------------------------------
results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor),
                          ("model", model)])
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = (mae, rmse, r2)

    print(f"\n----- {name} -----")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R² Score:", r2)


# ----------------------------------------------------
# SAVE BEST MODEL
# ----------------------------------------------------
best_model_name = max(results, key=lambda x: results[x][2])  # Highest R2
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

final_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", best_model)
])
final_pipeline.fit(X, y)

import joblib
joblib.dump(final_pipeline, "models/airbnb_price_model.pkl")

print("Model saved successfully in /models folder!")
