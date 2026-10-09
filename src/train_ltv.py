import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_ltv_model(
    data_path: str, model_save_path: str
) -> tuple[Pipeline, float]:
    """Train RandomForestRegressor model for Customer Lifetime Value (LTV/TotalCharges) prediction."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Separate features and target (Target is TotalCharges representing LTV)
    X = df.drop(columns=["TotalCharges"])
    y = df["TotalCharges"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

    # Create full ML pipeline for regression
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100, random_state=42, max_depth=15
                ),
            ),
        ]
    )

    # Train test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train pipeline
    print("Training Customer Lifetime Value (LTV) Prediction Model...")
    pipeline.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R-squared Score (R2): {r2:.4f}")

    # Save model artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"\nModel pipeline successfully saved to {model_save_path}")

    return pipeline, r2


if __name__ == "__main__":
    processed_data_path = os.path.join(
        "data", "processed", "telco_churn_clean.csv"
    )
    model_output_path = os.path.join("models", "ltv_pipeline.joblib")

    train_ltv_model(processed_data_path, model_output_path)