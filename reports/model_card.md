# Model Card: 30-Day Hospital Readmission Risk Predictor

## Model Details
- **Architecture:** Calibrated XGBoost Classifier (`CalibratedClassifierCV` wrapping `XGBClassifier` with sigmoid calibration) + K-Means Patient Cohort Clustering ($K=4$).
- **Version:** 1.0
- **Intended Task:** Predict individual 30-day all-cause hospital readmission probability for diabetic inpatient admissions.
- **Explainability Engine:** TreeSHAP (`shap.TreeExplainer` on uncalibrated base tree ensemble).

---

## Intended Use & Clinical Scope
- **Primary Use Case:** Clinical decision support and discharge planning tool for post-acute care coordination, identifying high-risk patients who would benefit from enhanced transitional care interventions.
- **Operating Decision Threshold:** Tuned to **0.102** on the training fold (achieving ~50% recall), deliberately trading off false positive rate to prioritize catching high-risk patients under class imbalance (~9% true prevalence).
- **Out-of-Scope:** This system is not intended as an autonomous diagnostic or discharge decision-maker.

---

## Performance Summary

| Metric | Test Set Value |
|---|---|
| ROC-AUC | 0.650 |
| PR-AUC | 0.180 |
| Recall @ Operating Threshold (0.102) | 50.4% |
| Precision @ Operating Threshold (0.102) | 14.5% |
| Mean Predicted Probability (Calibrated) | 8.92% (Actual: 8.97%) |
| Calibration Error (Brier Score / Mean Drift) | < 0.005 |

---

## Fairness & Demographic Subgroup Audit
Audited across race and gender protected attributes to ensure balanced recall and minimize disparate impact:

| Attribute | Group | Encounter Count ($n$) | Recall | False Negative Rate |
|---|---|---|---|---|
| race | AfricanAmerican | 2,505 | 0.573 | 0.427 |
| race | Caucasian | 10,507 | 0.567 | 0.433 |
| race | Hispanic | 299 | 0.552 | 0.448 |
| race | Asian | 88 | 0.545 | 0.455 |
| race | Other / Unknown | 596 | 0.498 | 0.502 |
| gender | Female | 7,440 | 0.607 | 0.393 |
| gender | Male | 6,554 | 0.520 | 0.480 |

---

## Verified Key Risk Drivers
Based on sensitivity testing, SHAP attribution, and ground-truth validation:
1. **Discharge Disposition:** Strongest individual driver. Transfers to rehabilitation facilities (26.3% readmit rate) or skilled nursing facilities (13.4%) carry substantially higher risk than discharge to home (6.9%).
2. **Prior Inpatient Utilization:** Monotonic risk signal — 0 prior visits has an 8.1% readmit rate, rising steadily to 23.8% at 3+ prior visits.
3. **Medication & Lab Procedure Volume:** Weak standalone predictive power; elevated medication count only amplifies risk in conjunction with high prior utilization or acute discharge transfers.
4. **Out-of-Distribution Warning:** Predictions for inputs beyond the typical training bounds (e.g. prior inpatient visits > 6) are flagged by the application as potentially less reliable.