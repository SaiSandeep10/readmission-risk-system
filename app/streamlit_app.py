import streamlit as st

st.set_page_config(
    page_title="Clinical Readmission Risk System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Clinical Readmission Risk & Cohort Clustering System")

st.markdown("""
This system predicts 30-day hospital readmission risk for diabetic patients
and groups patients into clinically meaningful cohorts using unsupervised learning.

**Navigate using the sidebar:**
- **🔮 Predictor** — enter a patient's details and get a readmission risk score
- **🗂️ Cohort Explorer** — see how patients cluster into distinct health profiles
- **📤 Batch Upload** — score multiple patients at once via CSV
- **📊 Model Insights** — model performance, feature importance, and fairness audit

---
Built on the UCI *Diabetes 130-US Hospitals (1999–2008)* dataset, using K-Means
clustering for cohort discovery and XGBoost for readmission risk classification.
""")

st.info("👈 Select a page from the sidebar to get started.")