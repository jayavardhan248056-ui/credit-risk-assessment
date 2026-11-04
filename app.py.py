
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px

st.set_page_config(
    page_title="CreditIQ | AI Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.joblib")

artifact = load_model()
model = artifact["model"]
imputer = artifact["imputer"]
features = artifact["feature_names"]
threshold = artifact["threshold"]

st.markdown("""
<style>
.stApp {background-color:#0b1220;color:white;}
[data-testid="stSidebar"] {background-color:#111d32;}
h1,h2,h3 {color:#f1f5f9;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏦 CREDITIQ")
st.sidebar.caption("EXPLAINABLE CREDIT RISK INTELLIGENCE")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Risk Prediction", "Batch Analysis",
     "Model Performance", "Explainable AI"]
)

def prepare_data(df):
    df = df.reindex(columns=features)
    df = df.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(
        imputer.transform(df),
        columns=features
    )

def predict_data(df):
    probability = model.predict_proba(df)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return probability, prediction

# OVERVIEW
if page == "Overview":
    st.title("Credit Risk Intelligence")
    st.write("AI-powered credit risk assessment with transparent predictions.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training Records", "246,008")
    c2.metric("Input Features", len(features))
    c3.metric("ML Algorithm", "XGBoost")

    st.subheader("Project Overview")
    st.write("""
    This platform predicts potential loan repayment difficulties
    using Machine Learning and explains predictions using SHAP.

    It helps users understand the factors influencing credit risk.
    """)
    st.info("Select Risk Prediction to assess an applicant.")

# RISK PREDICTION
elif page == "Risk Prediction":
    st.title("Credit Risk Prediction")
    st.write("Enter applicant details to predict repayment risk.")

    important_features = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_CREDIT",
        "AMT_GOODS_PRICE",
        "AMT_ANNUITY",
        "DAYS_EMPLOYED",
        "DAYS_BIRTH"
    ]

    defaults = dict(zip(features, imputer.statistics_))
    values = defaults.copy()

    available = [f for f in important_features if f in features]

    with st.form("applicant_form"):
        cols = st.columns(2)

        for i, feature in enumerate(available):
            with cols[i % 2]:
                values[feature] = st.number_input(
                    feature,
                    value=float(defaults[feature])
                )

        submitted = st.form_submit_button(
            "Predict Credit Risk",
            use_container_width=True
        )

    if submitted:
        row = pd.DataFrame([values])[features]
        probability, prediction = predict_data(row)

        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)
        c1.metric("Risk Probability", f"{probability[0]*100:.2f}%")
        c2.metric(
            "Risk Classification",
            "HIGH RISK" if prediction[0] else "LOW RISK"
        )

        if prediction[0]:
            st.warning("The model identifies potential repayment difficulty.")
        else:
            st.success("The model predicts a lower repayment risk.")

        st.caption("This is a model estimate for demonstration, not a lending decision.")

        st.subheader("Applicant Explanation")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(row)

            if isinstance(shap_values, list):
                vals = np.array(shap_values[-1])[0]
            else:
                vals = np.array(shap_values)
                if vals.ndim == 3:
                    vals = vals[0, :, -1]
                elif vals.ndim == 2:
                    vals = vals[0]

            explanation = pd.DataFrame({
                "Feature": features,
                "SHAP Impact": vals
            })
            explanation["Absolute Impact"] = explanation["SHAP Impact"].abs()
            explanation = explanation.sort_values(
                "Absolute Impact", ascending=False
            ).head(10)

            fig = px.bar(
                explanation.sort_values("SHAP Impact"),
                x="SHAP Impact",
                y="Feature",
                orientation="h",
                title="Top Factors Influencing This Prediction"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Plain-English explanation based on this applicant's SHAP values
            st.subheader("Simple Explanation")
            st.write(
                f"The model estimates a **{probability[0] * 100:.2f}%** "
                "probability of repayment difficulty for this applicant. "
                f"The model's classification threshold is **{threshold * 100:.0f}%**. "
                "This probability is a model estimate, not a guarantee of what will happen."
            )

            increasing = explanation[explanation["SHAP Impact"] > 0].nlargest(
                3, "SHAP Impact"
            )
            decreasing = explanation[explanation["SHAP Impact"] < 0].nsmallest(
                3, "SHAP Impact"
            )

            risk_col, lower_col = st.columns(2)

            with risk_col:
                st.markdown("**Factors pushing risk higher**")
                if increasing.empty:
                    st.write("No factors in the displayed top features are pushing risk higher.")
                else:
                    for _, item in increasing.iterrows():
                        feature_name = item["Feature"].replace("_", " ").title()
                        feature_value = row.iloc[0][item["Feature"]]
                        st.markdown(
                            f"- **{feature_name}** — applicant value: "
                            f"`{feature_value:,.2f}`; pushes the prediction toward higher risk."
                        )

            with lower_col:
                st.markdown("**Factors pushing risk lower**")
                if decreasing.empty:
                    st.write("No factors in the displayed top features are pushing risk lower.")
                else:
                    for _, item in decreasing.iterrows():
                        feature_name = item["Feature"].replace("_", " ").title()
                        feature_value = row.iloc[0][item["Feature"]]
                        st.markdown(
                            f"- **{feature_name}** — applicant value: "
                            f"`{feature_value:,.2f}`; pushes the prediction toward lower risk."
                        )

            st.caption(
                "SHAP shows how each feature influences this model's prediction. "
                "Positive SHAP values push toward the repayment-difficulty class; "
                "negative values push away from it. These are model associations, not causes."
            )
        except Exception as e:
            st.error(f"Explanation could not be generated: {e}")

# BATCH ANALYSIS
elif page == "Batch Analysis":
    st.title("Batch Credit Risk Analysis")
    st.write("Upload a CSV file containing applicant records.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        data = pd.read_csv(uploaded)
        st.write("Uploaded Dataset Preview")
        st.dataframe(data.head())

        if st.button("Analyze Applicants"):
            processed = prepare_data(data)
            probabilities, predictions = predict_data(processed)

            results = data.copy()
            results["Risk Probability (%)"] = probabilities * 100
            results["Risk Classification"] = np.where(
                predictions == 1, "HIGH RISK", "LOW RISK"
            )

            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Applicants", len(results))
            c2.metric("High Risk", int(sum(predictions)))
            c3.metric("Low Risk", int(len(predictions)-sum(predictions)))

            st.download_button(
                "Download Results",
                results.to_csv(index=False),
                "credit_risk_results.csv",
                "text/csv"
            )

# MODEL PERFORMANCE
elif page == "Model Performance":
    st.title("Model Information")

    st.metric("Algorithm", "XGBoost")
    st.metric("Input Features", len(features))
    st.metric("Classification Threshold", threshold)

    st.write("""
    The XGBoost model estimates the probability of repayment difficulty.
    A threshold of 0.20 is used in this demonstration to classify risk.
    """)

    st.info("The threshold is a prototype setting and requires further validation.")

# EXPLAINABLE AI
elif page == "Explainable AI":
    st.title("Explainable AI – SHAP")
    st.write("Understand which features influence model predictions.")

    if st.button("Generate Feature Importance"):
        with st.spinner("Calculating SHAP values..."):
            try:
                sample = pd.DataFrame(
                    [imputer.statistics_] * 100,
                    columns=features
                )

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(sample)

                if isinstance(shap_values, list):
                    vals = np.array(shap_values[-1])
                else:
                    vals = np.array(shap_values)
                    if vals.ndim == 3:
                        vals = vals[:, :, -1]

                importance = pd.DataFrame({
                    "Feature": features,
                    "Importance": np.abs(vals).mean(axis=0)
                }).sort_values(
                    "Importance", ascending=False
                ).head(10)

                st.subheader("Top 10 SHAP Features")

                fig = px.bar(
                    importance.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance using SHAP"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(importance, use_container_width=True)

            except Exception as e:
                st.error(f"SHAP explanation error: {e}")

st.markdown("---")
st.caption("CREDITIQ | Explainable AI Framework for Transparent Credit Risk Assessment")
