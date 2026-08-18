# 🏥 Clinical Readmission Risk & Cohort Clustering System

An end-to-end machine learning system that predicts 30-day hospital readmission
risk for diabetic patients and discovers clinically meaningful patient cohorts
using unsupervised learning — built as a merit certificate project for the
IBM AI/ML Internship (Q2D).

**[🔗 GitHub Repository](https://github.com/SaiSandeep10/readmission-risk-system.git)**

---

## Overview

Hospital readmissions within 30 days are a key quality-of-care and cost metric
for healthcare systems. This project combines:

- **K-Means clustering** to group patients into distinct health cohorts based
  on utilization and clinical complexity
- **XGBoost classification** (calibrated) to predict individual readmission risk
- **A Streamlit web app** for interactive prediction, cohort exploration, and
  batch scoring

Built on the [UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
dataset — 101,766 raw encounters, cleaned to 69,973 unique patient encounters.

## Results

| Metric | Value |
|---|---|
| ROC-AUC (test set) | 0.650 |
| Recall @ operating threshold | 0.504 |
| Precision @ operating threshold | 0.145 |
| Silhouette Score (clustering, K=4) | 0.141 |
| Calibration error | <0.005 (mean predicted vs. actual positive rate) |

Performance is consistent with published benchmarks on this dataset
(typically 0.65–0.70 ROC-AUC). See [`reports/model_card.md`](reports/model_card.md)
for full methodology, limitations, and fairness audit.

## App Features

| Page | What it does |
|---|---|
| 🔮 Predictor | Enter a patient's details, get a risk score + SHAP explanation |
| 🗂️ Cohort Explorer | Visualize patient cohorts (PCA-projected) with profile summaries |
| 📤 Batch Upload | Score a CSV of multiple patients at once, download results |
| 📊 Model Insights | ROC curve, confusion matrix, feature importance, fairness audit |

## Project Structure
```text
readmission-risk-system/
├── data/           # raw + processed datasets
├── notebooks/      # Phase 1–4 development & exploration notebooks
├── src/            # reusable pipeline & training modules
├── models/         # trained artifacts (pipeline, kmeans, classifier, config)
├── app/            # multi-page Streamlit application
├── reports/        # cleaning report, model card, fairness audit, figures
├── tests/          # pytest unit test suite
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup & Local Run

```bash
# 1. Clone and enter the repo
git clone https://github.com/SaiSandeep10/readmission-risk-system.git
cd readmission-risk-system

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest tests/ -v

# 5. Run the web app
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`.

## Reproducing the Model from Scratch

You can reproduce all models and reports directly via the CLI:
```bash
python -m src.train
```

Or interactively through the development notebooks:
1. `notebooks/01_data_cleaning.ipynb`
2. `notebooks/02_eda.ipynb`
3. `notebooks/03_feature_engineering.ipynb`
4. `notebooks/04_clustering.ipynb`
5. `notebooks/05_classification.ipynb`
6. `notebooks/06_predictor_sensitivity_test.ipynb`

## Methodology Highlights

- **Data cleaning:** de-duplicated to first encounter per patient (avoids
  leakage from repeat visits), removed death/hospice discharges, bucketed
  raw ICD-9 codes into 9 clinical chapters.
- **Class imbalance:** true positive rate is ~9%. The model is calibrated
  (`CalibratedClassifierCV`) so predicted probabilities reflect real-world
  frequency, and the operating decision threshold (0.102, not the default 0.5)
  is tuned for ~50% recall — prioritizing catching true readmissions over
  minimizing false alarms, a deliberate clinical tradeoff.
- **Verified key risk drivers** (via feature importance + ground-truth
  cross-check): discharge disposition (transfer to institutional care vs.
  home) and prior inpatient visit count are the strongest predictors —
  stronger than raw medication or lab test counts.
- **Known limitation:** predictions for inputs far outside the training
  distribution (e.g. >8 prior inpatient visits) are less reliable — the
  app flags this explicitly.

## Tech Stack

Python · scikit-learn · XGBoost · SHAP · Streamlit · Plotly · pandas

## Author

Sai Sandeep — B.Tech CSE, IBM AI/ML Internship (Q2D) Merit Submission

## License

*(add if applicable — e.g. MIT, or "Academic project, not for production clinical use")*