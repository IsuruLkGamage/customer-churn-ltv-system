import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_feature_importance():
    """Extract feature importance from trained RandomForest Churn Model."""
    try:
        pipeline = joblib.load("models/churn_pipeline.joblib")
        rf_model = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        cat_cols = (
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out()
        )
        num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
        all_features = list(num_cols) + list(cat_cols)

        importances = rf_model.feature_importances_
        df_imp = (
            pd.DataFrame(
                {"Feature": all_features, "Importance": importances}
            )
            .sort_values(by="Importance", ascending=False)
            .head(10)
        )

        fig = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Risk Drivers (Model Feature Importance)",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            yaxis={"autorange": "reversed"},
            height=380,
        )
        return fig
    except Exception as e:
        return None


def get_dataset_analytics():
    """Generate overall dataset distribution charts."""
    try:
        df = pd.read_csv("data/processed/telco_churn_clean.csv")

        fig_pie = px.pie(
            df,
            names="Churn",
            title="Overall Customer Churn Distribution",
            color="Churn",
            color_discrete_map={0: "#00c853", 1: "#ff4b4b"},
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}
        )

        fig_tenure = px.histogram(
            df,
            x="tenure",
            color="Churn",
            barmode="overlay",
            title="Customer Tenure vs Churn Rate",
            color_discrete_map={0: "#00c853", 1: "#ff4b4b"},
        )
        fig_tenure.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
        )

        return fig_pie, fig_tenure
    except Exception as e:
        return None, None