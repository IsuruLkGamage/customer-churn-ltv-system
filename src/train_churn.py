import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_churn_model(
    data_path: str, model_save_path: str
) -> tuple[Pipeline, float]:
    """Train RandomForest model for Churn prediction and save pipeline artifact."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

    # Create full ML pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    # Train test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train pipeline
    print("Training Churn Prediction Model...")
    pipeline.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # Save model artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print(f"\nModel pipeline successfully saved to {model_save_path}")

    return pipeline, auc_score


if __name__ == "__main__":
    processed_data_path = os.path.join(
        "data", "processed", "telco_churn_clean.csv"
    )
    model_output_path = os.path.join("models", "churn_pipeline.joblib")

    train_churn_model(processed_data_path, model_output_path)