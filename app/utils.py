import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load('models/preprocessing_pipeline.joblib')
    kmeans = joblib.load('models/kmeans_model.joblib')
    classifier = joblib.load('models/classifier.joblib')          # calibrated — for predictions
    base_model = joblib.load('models/base_xgb_model.joblib')      # raw — for SHAP
    with open('models/model_config.json') as f:
        config = json.load(f)
    return pipeline, kmeans, classifier, base_model, config

@st.cache_data
def load_reference_data():
    df = pd.read_csv('data/processed/cleaned_data.csv', keep_default_na=False)
    return df

def predict_single(patient_df, pipeline, kmeans, classifier):
    X_proc = pipeline.transform(patient_df)
    cluster = kmeans.predict(X_proc)
    X_final = np.hstack([X_proc, cluster.reshape(-1, 1)])
    prob = classifier.predict_proba(X_final)[:, 1][0]
    return float(prob), int(cluster[0])

def risk_tier(prob, config):
    if prob < config['risk_tier_low_cutoff']:
        return "Low", "🟢"
    elif prob < config['risk_tier_high_cutoff']:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreates Phase 3 engineered features and ensures no NaNs remain
    in critical categorical columns (glucose, A1C, diagnosis codes).
    """
    df = df.copy()

    # --- Imputation for missing values ---
    for col in ['max_glu_serum', 'A1Cresult']:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    # --- Engineered features ---
    if set(['number_outpatient','number_emergency','number_inpatient']).issubset(df.columns):
        df['total_prior_utilization'] = (
            df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
        )

    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'examide',
                'citoglipton', 'insulin', 'glyburide-metformin',
                'glipizide-metformin', 'glimepiride-pioglitazone',
                'metformin-rosiglitazone', 'metformin-pioglitazone']
    med_cols = [c for c in med_cols if c in df.columns]

    df['num_meds_changed'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v in ['Up', 'Down']), axis=1
    )

    return df
