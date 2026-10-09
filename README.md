# 📊 End-to-End Customer Churn & Lifetime Value (LTV) Prediction System

An end-to-end machine learning system designed to predict customer churn risk and estimate Customer Lifetime Value (LTV) for telecommunication service providers. Features modular code architecture, dual ML modeling pipelines, FastAPI backend REST endpoints, interactive Streamlit frontend dashboard, and Docker containerization.

---

## 🌟 Key Features

- **Dual Machine Learning Pipelines:**
  - **Churn Classification:** Random Forest Classifier with SMOTE/class-weighting for predicting customer attrition probability (`models/churn_pipeline.joblib`).
  - **LTV Regression:** Random Forest Regressor predicting future customer value (`models/ltv_pipeline.joblib`).
- **RESTful API Services:** High-performance API built with **FastAPI** and **Pydantic** schema validation (`/predict` endpoint).
- **Interactive UI Dashboard:** User-friendly **Streamlit** Web Interface for real-time inference and risk categorization.
- **Production Architecture:** Modular Python scripts (`src/`), clean virtual environment separation, and deployment readiness via **Docker & Docker Compose**.

---

## 📁 Project Directory Structure

```text
customer-churn-ltv-system/
├── app/
│   ├── main.py              # FastAPI REST API Backend
│   └── streamlit_app.py     # Streamlit Web UI Frontend
├── data/
│   ├── processed/           # Preprocessed dataset
│   └── raw/                 # Downloaded raw dataset
├── models/                  # Trained Model Artifacts (.joblib)
├── notebooks/               # Exploratory Data Analysis (01_eda.ipynb)
├── src/                     # Core Business Logic Modules
│   ├── download_data.py     # Data Ingestion Script
│   ├── feature_engineering.py # Data Cleaning & Type Conversion
│   ├── train_churn.py       # Churn Model Pipeline Training
│   └── train_ltv.py         # LTV Model Pipeline Training
├── .gitignore               # Git exclusions
├── Dockerfile               # Docker Build Config
├── docker-compose.yml       # Docker Services Definition
├── requirements.txt         # Dependencies
└── README.md                # Documentation
