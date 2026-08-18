# Data Cleaning & Preprocessing Report

## Dataset Summary
- **Source:** UCI Diabetes 130-US Hospitals Dataset (1999–2008)
- **Raw Encounters:** 101,766 encounters across 71,518 unique patients with 50 attributes.
- **Processed Encounters:** 69,973 encounters after filtering and deduplication.

---

## Cleaning Steps Executed

1. **Patient-Level Encounter Deduplication**:
   - The raw dataset contains repeat hospitalizations for the same patient.
   - Kept only the first chronological encounter per `patient_nbr` (reduced from 101,766 to 71,518 rows) to prevent data leakage between training and testing folds.

2. **Exclusion of Terminal & Hospice Discharges**:
   - Filtered out `discharge_disposition_id` entries corresponding to expiration or hospice care:
     - `11` (Expired)
     - `13` (Hospice / home)
     - `14` (Hospice / medical facility)
     - `19` (Expired at home, Medicaid/Medicare)
     - `20` (Expired in medical facility)
     - `21` (Expired, place unknown)
   - These patients are ineligible for readmission, leaving 69,973 valid encounters.

3. **High Sparsity & Redundant Column Removal**:
   - Dropped `weight` (>96% missing).
   - Dropped `payer_code` (not clinically relevant and ~40% missing).
   - Dropped `encounter_id` and raw `readmitted` string variants after target construction.

4. **ICD-9 Diagnosis Code Grouping**:
   - Mapped 700+ distinct ICD-9 codes in `diag_1`, `diag_2`, and `diag_3` into 9 clinical chapters:
     - `Circulatory` (390–459, 785)
     - `Respiratory` (460–519, 786)
     - `Digestive` (520–579, 787)
     - `Diabetes` (250.xx)
     - `Injury` (800–999)
     - `Musculoskeletal` (710–739)
     - `Genitourinary` (580–629, 788)
     - `Neoplasm` (140–239)
     - `Other` (all remaining codes and V/E codes)

5. **Missing Value Imputation**:
   - `max_glu_serum` and `A1Cresult`: Imputed `NaN` with `'None'` (indicating the diagnostic lab test was not ordered).
   - `race` and `medical_specialty`: Imputed missing values with `'Unknown'`.
   - `diag_1`, `diag_2`, `diag_3`: Imputed missing entries with `'Missing'`.

6. **Target Variable Formulation**:
   - Defined `readmitted_binary = 1` for `<30` days readmissions, and `0` for `>30` or `NO` readmissions.
   - Positive class base rate in the processed dataset: **~8.97%** (highly imbalanced).
