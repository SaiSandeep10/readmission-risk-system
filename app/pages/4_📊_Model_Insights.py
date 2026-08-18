import streamlit as st
import os

st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")
st.title("📊 Model Performance, Explainability & Fairness Insights")

fig_dir = "reports/eda_figures"

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Classification Performance",
    "🔍 Feature Importance (SHAP)",
    "🗂️ Cohort Clustering",
    "⚖️ Fairness & Subgroups"
])

with tab1:
    st.header("Classification & Probability Calibration")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("ROC, Precision-Recall & Confusion Matrix")
        if os.path.exists(f"{fig_dir}/model_evaluation.png"):
            st.image(f"{fig_dir}/model_evaluation.png", use_column_width=True)
        elif os.path.exists(f"{fig_dir}/model_evaluation_v2.png"):
            st.image(f"{fig_dir}/model_evaluation_v2.png", use_column_width=True)
    with col2:
        st.subheader("Probability Calibration Curve")
        if os.path.exists(f"{fig_dir}/calibration_curve.png"):
            st.image(f"{fig_dir}/calibration_curve.png", use_column_width=True)

    st.markdown("---")
    st.subheader("Summary Evaluation Metrics")
    if os.path.exists("reports/evaluation_report.md"):
        with open("reports/evaluation_report.md") as f:
            st.markdown(f.read())

with tab2:
    st.header("Global & Local Feature Explainability (SHAP)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance (Beeswarm)")
        if os.path.exists(f"{fig_dir}/shap_summary.png"):
            st.image(f"{fig_dir}/shap_summary.png", use_column_width=True)
    with col2:
        st.subheader("Individual Prediction Waterfall Example")
        if os.path.exists(f"{fig_dir}/shap_waterfall_example.png"):
            st.image(f"{fig_dir}/shap_waterfall_example.png", use_column_width=True)

with tab3:
    st.header("Patient Cohort Discovery (K-Means Clustering)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2D PCA Projection of Cohorts")
        if os.path.exists(f"{fig_dir}/cluster_scatter.png"):
            st.image(f"{fig_dir}/cluster_scatter.png", use_column_width=True)
    with col2:
        st.subheader("Optimal Cluster Selection (Elbow & Silhouette)")
        if os.path.exists(f"{fig_dir}/elbow_and_silhouette.png"):
            st.image(f"{fig_dir}/elbow_and_silhouette.png", use_column_width=True)

    if os.path.exists("reports/cohort_profiles.md"):
        st.markdown("---")
        with open("reports/cohort_profiles.md") as f:
            st.markdown(f.read())

with tab4:
    st.header("Demographic Fairness & Subgroup Analysis")
    st.subheader("Fairness Audit across Race & Gender Subgroups")
    if os.path.exists("reports/fairness_audit.md"):
        with open("reports/fairness_audit.md") as f:
            st.markdown(f.read())
    else:
        st.info("Fairness audit report not found.")

    st.markdown("---")
    st.subheader("Clinical & Demographic Readmission Rates")
    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists(f"{fig_dir}/readmit_by_prior_inpatient.png"):
            st.image(f"{fig_dir}/readmit_by_prior_inpatient.png", caption="By Prior Inpatient Visits")
    with col2:
        if os.path.exists(f"{fig_dir}/readmit_by_diagnosis.png"):
            st.image(f"{fig_dir}/readmit_by_diagnosis.png", caption="By Primary Diagnosis")
    with col3:
        if os.path.exists(f"{fig_dir}/readmit_by_age.png"):
            st.image(f"{fig_dir}/readmit_by_age.png", caption="By Age Group")