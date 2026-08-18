import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_artifacts, predict_single, risk_tier, engineer_features

st.set_page_config(page_title="Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Patient Readmission Risk Predictor")

pipeline, kmeans, classifier, base_model, config = load_artifacts()

AGE_MAP = {
    '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
    '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
}

# Realistic bounds — based on actual training data distribution
# (values beyond these are rare/unseen and produce unreliable predictions)
BOUNDS = {
    'number_inpatient': 8,
    'number_emergency': 10,
    'number_outpatient': 10,
    'num_medications': 80,
}

with st.sidebar:
    st.header("Patient Details")

    age = st.selectbox("Age Group", list(AGE_MAP.keys()), index=7)
    race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', 'Unknown'])
    gender = st.selectbox("Gender", ['Female', 'Male'])

    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
    num_lab_procedures = st.slider("Number of Lab Procedures", 0, 130, 40)
    num_procedures = st.slider("Number of Procedures", 0, 6, 1)
    num_medications = st.slider("Number of Medications", 1, BOUNDS['num_medications'], 15)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)

    number_outpatient = st.number_input("Prior Outpatient Visits", 0, BOUNDS['number_outpatient'], 0)
    number_emergency = st.number_input("Prior Emergency Visits", 0, BOUNDS['number_emergency'], 0)
    number_inpatient = st.number_input("Prior Inpatient Visits", 0, BOUNDS['number_inpatient'], 0)

    admission_type_options = {
        1: "1 - Emergency", 2: "2 - Urgent", 3: "3 - Elective",
        4: "4 - Newborn", 7: "7 - Trauma Center",
    }
    admission_type_label = st.selectbox("Admission Type", list(admission_type_options.values()))
    admission_type_id = [k for k, v in admission_type_options.items() if v == admission_type_label][0]

    discharge_options = {
        1: "1 - Discharged to home",
        2: "2 - Transferred to another short term hospital",
        3: "3 - Transferred to SNF (Skilled Nursing Facility)",
        6: "6 - Discharged to home with home health service",
        22: "22 - Transferred to rehab facility",
        23: "23 - Transferred to a long term care hospital",
    }
    discharge_label = st.selectbox("Discharge Disposition", list(discharge_options.values()))
    discharge_disposition_id = [k for k, v in discharge_options.items() if v == discharge_label][0]

    admission_source_options = {
        1: "1 - Physician Referral", 4: "4 - Transfer from a hospital",
        7: "7 - Emergency Room", 17: "17 - NULL/Not specified", 20: "20 - Not Mapped",
    }
    admission_source_label = st.selectbox("Admission Source", list(admission_source_options.values()))
    admission_source_id = [k for k, v in admission_source_options.items() if v == admission_source_label][0]

    diag_options = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury',
                    'Musculoskeletal', 'Genitourinary', 'Neoplasm', 'Other']
    diag_1_group = st.selectbox("Primary Diagnosis Group", diag_options, index=0)
    diag_2_group = st.selectbox("Secondary Diagnosis Group", diag_options, index=0)
    diag_3_group = st.selectbox("Tertiary Diagnosis Group", diag_options, index=0)

    max_glu_serum = st.selectbox("Max Glucose Serum", ['None', 'Norm', '>200', '>300'])
    A1Cresult = st.selectbox("A1C Result", ['None', 'Norm', '>7', '>8'])
    change = st.selectbox("Medication Change", ['No', 'Ch'])
    diabetesMed = st.selectbox("Diabetes Medication Prescribed", ['Yes', 'No'])

    submitted = st.button("Predict Risk", type="primary", use_container_width=True)

if submitted:
    # Flag if any input is in a sparsely-populated region of the training data
    ood_flags = []
    if number_inpatient >= 6:
        ood_flags.append(f"Prior Inpatient Visits = {number_inpatient} (rare in training data, <1% of patients)")
    if number_emergency >= 6:
        ood_flags.append(f"Prior Emergency Visits = {number_emergency} (rare in training data)")
    if num_medications >= 35:
        ood_flags.append(f"Number of Medications = {num_medications} (near the high end of training data)")

    patient_dict = {
        'race': race, 'gender': gender, 'age_ordinal': AGE_MAP[age],
        'admission_type_id': admission_type_id,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'time_in_hospital': time_in_hospital,
        'medical_specialty': 'Unknown',
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'diag_1_group': diag_1_group, 'diag_2_group': diag_2_group, 'diag_3_group': diag_3_group,
        'number_diagnoses': number_diagnoses,
        'max_glu_serum': max_glu_serum, 'A1Cresult': A1Cresult,
        'change': change, 'diabetesMed': diabetesMed,
        'total_prior_utilization': number_outpatient + number_emergency + number_inpatient,
        'num_meds_changed': 1 if change == 'Ch' else 0,
    }
    patient_df = pd.DataFrame([patient_dict])

    prob, cluster = predict_single(patient_df, pipeline, kmeans, classifier)
    tier, emoji = risk_tier(prob, config)

    if ood_flags:
        st.warning("⚠️ **Prediction may be less reliable** — one or more inputs are outside "
                   "the range commonly seen in training data:\n\n" +
                   "\n".join(f"- {f}" for f in ood_flags))

    col1, col2, col3 = st.columns(3)
    col1.metric("Readmission Probability", f"{prob:.1%}")
    col2.metric("Risk Tier", f"{emoji} {tier}")
    col3.metric("Assigned Cohort", f"Cluster {cluster}")

    st.progress(float(min(prob, 1.0)))

    st.subheader("Why this prediction? (SHAP)")
    try:
        X_proc = pipeline.transform(patient_df.drop(columns=[c for c in patient_df.columns if c == 'cluster'], errors='ignore'))
        X_final = np.hstack([X_proc, np.array([[cluster]])])

        feature_names = list(pipeline.named_steps['preprocessor'].get_feature_names_out()) + ['cohort_cluster']
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_final)

        plt.figure(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_final[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.close()
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
else:
    st.info("Fill in patient details in the sidebar and click **Predict Risk**.")