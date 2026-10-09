import os
import numpy as np
import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at {file_path}")
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco dataset and convert data types."""
    df_clean = df.copy()

    # Drop customerID as it is not a feature for modeling
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # Fix TotalCharges column: replace empty strings with NaN and convert to float
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"].replace(" ", np.nan), errors="coerce"
    )

    # Fill missing TotalCharges with 0 for new customers (tenure == 0)
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Convert binary target variable 'Churn' (Yes/No) to integer (1/0)
    if "Churn" in df_clean.columns:
        df_clean["Churn"] = (
            df_clean["Churn"].map({"Yes": 1, "No": 0}).astype(int)
        )

    return df_clean


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save processed dataset to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset successfully saved to {output_path}")


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "telco_churn.csv")
    processed_path = os.path.join("data", "processed", "telco_churn_clean.csv")

    raw_df = load_raw_data(raw_path)
    clean_df = preprocess_data(raw_df)
    save_processed_data(clean_df, processed_path)