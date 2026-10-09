import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SAVE_PATH = os.path.join("data", "raw", "telco_churn.csv")


def download_dataset():
    """Download the raw Telco Customer Churn dataset into data/raw/ directory."""
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print(f"Downloading dataset from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, SAVE_PATH)
    print(f"Dataset downloaded successfully and saved to {SAVE_PATH}")


if __name__ == "__main__":
    download_dataset()