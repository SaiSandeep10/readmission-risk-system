import streamlit as st
import json
import os

st.set_page_config(
    page_title="Clinical Readmission Risk System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Clinical Readmission Risk & Cohort Clustering System")
st.markdown("### Decision Support & Patient Stratification for Diabetic Inpatient Care")

st.markdown("""
An end-to-end clinical machine learning system that predicts **30-day all-cause hospital readmission risk**
for diabetic patients and identifies clinically meaningful **patient cohorts** using unsupervised learning.
""")

# Key Performance & Dataset Metric Highlights
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏥 Patient Encounters", "69,973", "UCI Diabetes 130-US")
col2.metric("🎯 Model ROC-AUC", "0.650", "Calibrated XGBoost")
col3.metric("⚡ Clinical Recall", "50.4%", "@ 0.102 Decision Threshold")
col4.metric("🗂️ Discovered Cohorts", "4 Clusters", "K-Means + PCA")

st.markdown("---")

# Feature Cards
st.subheader("Explore the System Features")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    #### 🔮 [Predictor](Predictor)
    - Enter individual patient demographics, clinical history, diagnoses, and medication changes.
    - Obtain calibrated readmission probabilities, clinical risk tier (`Low` 🟢 / `Medium` 🟡 / `High` 🔴), and assigned health cohort.
    - View instant **TreeSHAP explainability force plots** detailing which factors drove the risk score up or down.

    #### 🗂️ [Cohort Explorer](Cohort_Explorer)
    - Explore 2D PCA projections of discovered patient cohorts.
    - Compare average length of stay, medication volume, diagnoses count, and baseline readmission rates across clusters.
    """)

with col_b:
    st.markdown("""
    #### 📤 [Batch Upload](Batch_Upload)
    - Upload CSV datasets containing multiple patient encounters.
    - Automatically execute feature vectorization, cohort clustering, and risk scoring in bulk.
    - Download enriched prediction results with probability scores and risk tiers.

    #### 📊 [Model Insights](Model_Insights)
    - Comprehensive model evaluation: ROC curves, Precision-Recall curves, and confusion matrices.
    - Probability calibration curves demonstrating clinical calibration error `< 0.005`.
    - Global feature importance (SHAP beeswarm) and demographic fairness audits across race and gender subgroups.
    """)

st.info("👈 **Get started:** Select a page from the sidebar navigation on the left.")