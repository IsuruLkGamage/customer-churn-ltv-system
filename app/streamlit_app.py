import io
import os
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.analytics import get_dataset_analytics, get_feature_importance

# Page Config
st.set_page_config(
    page_title="Telco AI | Churn & LTV Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper function to load local models for Direct Inference (Fallback for Cloud)
@st.cache_resource
def load_local_models():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    churn_path = os.path.join(base_dir, "models", "churn_pipeline.joblib")
    ltv_path = os.path.join(base_dir, "models", "ltv_pipeline.joblib")
    
    churn_model = joblib.load(churn_path) if os.path.exists(churn_path) else None
    ltv_model = joblib.load(ltv_path) if os.path.exists(ltv_path) else None
    return churn_model, ltv_model

churn_model_obj, ltv_model_obj = load_local_models()

# Unified Prediction Handler (Tries Backend API first, falls back to Direct Model Inference)
def predict_single(payload):
    # Ensure TotalCharges exists in payload
    if "TotalCharges" not in payload or payload["TotalCharges"] is None:
        tenure_val = payload.get("tenure", 0)
        monthly_val = payload.get("MonthlyCharges", 0.0)
        payload["TotalCharges"] = round(float(tenure_val) * float(monthly_val), 2)

    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass  # Fallback to direct local inference if API is unreachable

    if churn_model_obj and ltv_model_obj:
        df = pd.DataFrame([payload])
        
        # 1. Calculate churn probability
        if hasattr(churn_model_obj, "predict_proba"):
            churn_prob = float(churn_model_obj.predict_proba(df)[0][1])
        else:
            churn_prob = float(churn_model_obj.predict(df)[0])
            
        churn_pred = 1 if churn_prob >= 0.5 else 0

        # 2. Add 'Churn' column to df for LTV model if missing
        if "Churn" not in df.columns:
            df["Churn"] = churn_pred
            
        # 3. Predict LTV
        predicted_ltv = float(ltv_model_obj.predict(df)[0])
        
        risk_level = "High Risk" if churn_prob >= 0.6 else ("Medium Risk" if churn_prob >= 0.3 else "Low Risk")
        
        return {
            "churn_prediction": churn_pred,
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk_level,
            "predicted_ltv": round(predicted_ltv, 2)
        }
    else:
        raise RuntimeError("FastAPI Backend is unreachable and local models could not be loaded.")

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e222d;
        border: 1px solid #2e364f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-title {
        color: #8b9bb4 !important;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: 700;
    }
    
    .high-risk-box {
        background-color: rgba(255, 75, 75, 0.15);
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 8px;
        color: #ff4b4b;
        font-weight: 600;
    }
    .med-risk-box {
        background-color: rgba(255, 171, 0, 0.15);
        border-left: 5px solid #ffab00;
        padding: 15px;
        border-radius: 8px;
        color: #ffab00;
        font-weight: 600;
    }
    .low-risk-box {
        background-color: rgba(0, 200, 83, 0.15);
        border-left: 5px solid #00c853;
        padding: 15px;
        border-radius: 8px;
        color: #00c853;
        font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image(
        "https://img.icons8.com/isometric-folders/100/combo-chart.png", width=70
    )
    st.title("Telco Analytics AI")
    st.caption("Enterprise Retention & LTV Suite")
    st.divider()
    st.success("🟢 **System Mode:** Cloud Hybrid Ready")

# Header Section
st.title("🔮 Enterprise Customer Retention & LTV Suite")
st.markdown(
    "Predict individual churn risk, execute bulk CSV batch inferences, and"
    " explore AI feature importance."
)
st.divider()

# Navigation Tabs
main_tab1, main_tab2, main_tab3 = st.tabs([
    "🎯 Single Customer Inference",
    "📂 Batch CSV Processing",
    "📊 Business Analytics & Explainability",
])

# -------------------------------------------------------------
# TAB 1: Single Customer Inference
# -------------------------------------------------------------
with main_tab1:
    tab1, tab2, tab3 = st.tabs(
        ["👤 Profile", "📡 Services", "💳 Billing & Contract"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
        with col2:
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
        with col2:
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
        with col3:
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=150.0,
                value=65.5,
            )

    if st.button(
        "🚀 Analyze Risk & Predict LTV",
        use_container_width=True,
        type="primary",
    ):
        total_charges = round(float(tenure) * float(monthly_charges), 2)
        payload = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            data = predict_single(payload)
            churn_prob = data["churn_probability"] * 100
            risk_level = data["risk_level"]
            ltv = data["predicted_ltv"]

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="metric-card"><div class="metric-title">Risk Level</div><div class="metric-value">{risk_level}</div></div>',
                unsafe_allow_html=True,
            )
            m2.markdown(
                f'<div class="metric-card"><div class="metric-title">Churn Probability</div><div class="metric-value">{churn_prob:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
            m3.markdown(
                f'<div class="metric-card"><div class="metric-title">Predicted LTV</div><div class="metric-value">${ltv:,.2f}</div></div>',
                unsafe_allow_html=True,
            )

            v1, v2 = st.columns([1.2, 1])
            with v1:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=churn_prob,
                        title={"text": "Churn Risk Gauge (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#6366f1"},
                            "steps": [
                                {
                                    "range": [0, 30],
                                    "color": "rgba(0,200,83,0.3)",
                                },
                                {
                                    "range": [30, 60],
                                    "color": "rgba(255,171,0,0.3)",
                                },
                                {
                                    "range": [60, 100],
                                    "color": "rgba(255,75,75,0.3)",
                                },
                            ],
                        },
                    )
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "gray"},
                    height=260,
                )
                st.plotly_chart(fig, use_container_width=True)

            with v2:
                st.markdown("### 💡 Recommended Action Plan")
                if risk_level == "High Risk":
                    st.markdown(
                        '<div class="high-risk-box">⚠️ HIGH RISK: Offer'
                        " long-term contract renewal discount and free tech"
                        " support package immediately.</div>",
                        unsafe_allow_html=True,
                    )
                elif risk_level == "Medium Risk":
                    st.markdown(
                        '<div class="med-risk-box">⚡ MODERATE RISK: Send'
                        " promotional bundle deals and service check-in"
                        " notification.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="low-risk-box">✅ LOW RISK: Highly'
                        " engaged customer. Target for upselling fiber"
                        " internet upgrades.</div>",
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Error executing inference: {e}")

# -------------------------------------------------------------
# TAB 2: Batch CSV Processing
# -------------------------------------------------------------
with main_tab2:
    st.subheader("📂 Batch Customer Prediction via CSV Upload")
    st.markdown(
        "Upload a raw CSV file containing customer records to generate risk"
        " scores and LTV estimates in bulk."
    )

    uploaded_file = st.file_uploader("Choose a CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Records:", batch_df.head(3))

        if st.button("🚀 Process Batch Inferences", type="primary"):
            results = []
            progress_bar = st.progress(0)

            for idx, row in batch_df.iterrows():
                payload = row.to_dict()
                try:
                    out = predict_single(payload)
                    payload["Churn_Prediction"] = out["churn_prediction"]
                    payload["Churn_Probability_%"] = round(
                        out["churn_probability"] * 100, 2
                    )
                    payload["Risk_Level"] = out["risk_level"]
                    payload["Predicted_LTV_$"] = out["predicted_ltv"]
                    results.append(payload)
                except Exception:
                    pass
                progress_bar.progress((idx + 1) / len(batch_df))

            res_df = pd.DataFrame(results)
            st.success("Batch Prediction Complete!")
            st.dataframe(res_df)

            csv_buffer = io.BytesIO()
            res_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Inference Report CSV",
                data=csv_buffer.getvalue(),
                file_name="churn_ltv_predictions.csv",
                mime="text/csv",
            )

# -------------------------------------------------------------
# TAB 3: Business Analytics & Feature Importance
# -------------------------------------------------------------
with main_tab3:
    st.subheader("📊 Dataset Insights & Explainable AI")

    col_a, col_b = st.columns(2)
    pie_chart, tenure_chart = get_dataset_analytics()

    if pie_chart and tenure_chart:
        with col_a:
            st.plotly_chart(pie_chart, use_container_width=True)
        with col_b:
            st.plotly_chart(tenure_chart, use_container_width=True)

    st.divider()
    st.subheader("🤖 Model Feature Importance (Explainability)")
    imp_chart = get_feature_importance()
    if imp_chart:
        st.plotly_chart(imp_chart, use_container_width=True)