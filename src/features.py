"""
src/features.py
Feature engineering and ColumnTransformer preprocessing pipeline builder.
"""

import os
from typing import Tuple, List
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

NUMERIC_FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_diagnoses', 'total_prior_utilization',
    'num_meds_changed'
]

NOMINAL_FEATURES = [
    'race', 'gender', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'diag_1_group', 'diag_2_group', 'diag_3_group',
    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
]

ORDINAL_FEATURES = ['age_ordinal']
HIGH_CARD_FEATURES = ['medical_specialty']


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific engineered features:
      - total_prior_utilization: sum of outpatient, emergency, and inpatient encounters.
      - num_meds_changed: count of diabetic medications with dosage adjusted ('Up' or 'Down').
    Also ensures categorical missing values are imputed properly.
    """
    df = df.copy()

    # Impute missing categories if any remain
    for col in ['max_glu_serum', 'A1Cresult']:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    # Total prior utilization
    util_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
    if set(util_cols).issubset(df.columns):
        df['total_prior_utilization'] = (
            df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
        )

    # Count of active dosage adjustments
    active_meds = [c for c in MEDICATION_COLS if c in df.columns]
    if active_meds:
        df['num_meds_changed'] = df[active_meds].apply(
            lambda row: sum(1 for v in row if v in ['Up', 'Down']), axis=1
        )
    elif 'num_meds_changed' not in df.columns:
        df['num_meds_changed'] = 0

    return df


def build_preprocessing_pipeline() -> Pipeline:
    """
    Constructs the standard scikit-learn ColumnTransformer preprocessing pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), NOMINAL_FEATURES),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ORDINAL_FEATURES),
            ('spec', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=15), HIGH_CARD_FEATURES),
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    return pipeline


def prepare_features_and_splits(
    cleaned_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "data/processed",
    models_dir: str = "models"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Applies feature engineering, fits the preprocessing pipeline, creates stratified
    train/test splits, and saves artifacts to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = engineer_features(cleaned_df)

    drop_cols = [c for c in ['patient_nbr', 'readmitted', 'readmitted_binary'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['readmitted_binary']

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X)

    # Save preprocessing pipeline
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Save splits
    X_train.to_csv(os.path.join(save_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(save_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(save_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, "y_test.csv"), index=False)

    return X_train, X_test, y_train, y_test, pipeline
