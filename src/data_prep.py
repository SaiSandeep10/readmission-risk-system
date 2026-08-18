"""
src/data_prep.py
Module for loading, cleaning, and preprocessing the raw UCI Diabetes dataset.
"""

import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np


AGE_MAPPING = {
    '[0-10)': 0,
    '[10-20)': 1,
    '[20-30)': 2,
    '[30-40)': 3,
    '[40-50)': 4,
    '[50-60)': 5,
    '[60-70)': 6,
    '[70-80)': 7,
    '[80-90)': 8,
    '[90-100)': 9,
}

# Discharge dispositions corresponding to hospice or death (not eligible for readmission)
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


def map_icd9_to_group(code: str) -> str:
    """
    Maps an ICD-9 diagnosis code string to 1 of 9 clinically meaningful categories.
    """
    if pd.isna(code) or str(code).strip() in ['?', 'Missing', '']:
        return 'Other'
    
    code_str = str(code).strip()
    if code_str.startswith(('V', 'E')):
        return 'Other'
    
    try:
        val = float(code_str)
    except ValueError:
        return 'Other'

    if (390 <= val <= 459) or val == 785:
        return 'Circulatory'
    elif (460 <= val <= 519) or val == 786:
        return 'Respiratory'
    elif (520 <= val <= 579) or val == 787:
        return 'Digestive'
    elif int(val) == 250:
        return 'Diabetes'
    elif 800 <= val <= 999:
        return 'Injury'
    elif 710 <= val <= 739:
        return 'Musculoskeletal'
    elif (580 <= val <= 629) or val == 788:
        return 'Genitourinary'
    elif 140 <= val <= 239:
        return 'Neoplasm'
    else:
        return 'Other'


def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads raw encounters from diabetic_data.csv.
    """
    raw_path = os.path.join(data_dir, "diabetic_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")
    
    df = pd.read_csv(raw_path, na_values=['?'], keep_default_na=False)
    return df


def clean_encounter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full Phase 1 cleaning pipeline:
      1. Replaces '?' placeholders with NaN/defaults.
      2. Deduplicates to first encounter per patient_nbr to avoid data leakage.
      3. Removes patient encounters ending in hospice or death.
      4. Drops redundant or overwhelmingly sparse columns (e.g. weight, examide, citoglipton).
      5. Maps age intervals to ordinal integers (0-9).
      6. Maps raw ICD-9 codes (diag_1, diag_2, diag_3) to 9 clinical chapters.
      7. Creates binary target readmitted_binary (1 if '<30', else 0).
      8. Imputes missing categories ('None' for glucose/A1C, 'Unknown' for race/gender).
    """
    df = df.copy()

    # Deduplicate to first encounter per patient
    if 'patient_nbr' in df.columns:
        df = df.drop_duplicates(subset=['patient_nbr'], keep='first')

    # Remove hospice/death discharges
    if 'discharge_disposition_id' in df.columns:
        df = df[~df['discharge_disposition_id'].isin(EXCLUDED_DISCHARGE_IDS)]

    # Drop columns with extreme sparsity or constant values
    drop_cols = ['weight', 'payer_code', 'encounter_id']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Ordinal age mapping
    if 'age' in df.columns:
        df['age_ordinal'] = df['age'].map(AGE_MAPPING).fillna(-1).astype(int)

    # Impute categorical variables
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Unknown')
    if 'medical_specialty' in df.columns:
        df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')

    for col in ['max_glu_serum', 'A1Cresult']:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # Map diagnosis codes
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[f'{col}_group'] = df[col].apply(map_icd9_to_group)
            df[col] = df[col].fillna('Missing')

    # Create binary target (<30 days = 1, otherwise 0)
    if 'readmitted' in df.columns:
        df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)

    return df


def prepare_and_save_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    Loads raw data, cleans it, and persists data/processed/cleaned_data.csv.
    """
    os.makedirs(processed_dir, exist_ok=True)
    raw_df = load_raw_data(raw_dir)
    cleaned_df = clean_encounter_data(raw_df)
    
    out_path = os.path.join(processed_dir, "cleaned_data.csv")
    cleaned_df.to_csv(out_path, index=False)
    return cleaned_df
