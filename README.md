# 🏥 Clinical Readmission Risk & Cohort Clustering System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://readmission-risk-system.streamlit.app/)
[![Tests](https://img.shields.io/badge/pytest-25%20passed-brightgreen.svg)](tests/test_pipeline.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end clinical machine learning system that predicts **30-day hospital readmission risk** for diabetic patients and discovers clinically meaningful patient cohorts using unsupervised learning. Developed as a merit project for the **IBM AI/ML Internship (Q2D)**.

🔗 **Live Streamlit App:** [https://readmission-risk-system.streamlit.app](https://readmission-risk-system.streamlit.app)  
📦 **GitHub Repository:** [https://github.com/SaiSandeep10/readmission-risk-system.git](https://github.com/SaiSandeep10/readmission-risk-system.git)

---

## 📌 Executive Summary & Architecture

Hospital readmissions within 30 days are a key quality-of-care and cost-containment metric across healthcare systems. This project implements a dual-learning pipeline:

1. **Unsupervised Cohort Discovery:** K-Means clustering ($K=4$) to partition patients into distinct clinical utilization and complexity profiles.
2. **Supervised Risk Classification:** Tuned & calibrated XGBoost (`CalibratedClassifierCV`) with SHAP explainability to provide unbiased 30-day readmission probabilities.
3. **Interactive Multi-Page Web Application:** Streamlit application providing single-patient prediction, cohort exploration, batch scoring, and fairness audit dashboards.

```
                    ┌────────────────────────┐
                    │   UCI Diabetes Data    │
                    │   (101,766 Encounters) │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Phase 1: Data Prep   │
                    │  (Deduplicate/ICD-9)   │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Phase 2: Feature Eng.  │
                    │ (ColumnTransformer OHE)│
                    └─────┬────────────┬─────┘
                          │            │
         ┌────────────────▼──┐      ┌──▼─────────────────┐
         │ Phase 3: K-Means  │      │ Phase 4: Calibrated│
         │ Cohort Clustering │─────►│ XGBoost Classifier │
         └───────────────────┘      └──────────┬─────────┘
                                               │
                                    ┌──────────▼─────────┐
                                    │ Streamlit Web App  │
                                    │  (4 Custom Pages)  │
                                    └────────────────────┘
```

---

## 📊 Benchmark Results

| Metric | Test Set Value | Clinical Context / Benchmark |
|---|---|---|
| **ROC-AUC** | **0.650** | Consistent with published literature on this dataset (0.65–0.70) |
| **PR-AUC** | **0.180** | Baseline positive prevalence is ~8.97% |
| **Recall @ Operating Threshold** | **50.4%** | Captures majority of true readmissions |
| **Precision @ Operating Threshold** | **14.5%** | Deliberate clinical tradeoff prioritizing sensitivity |
| **Operating Decision Threshold** | **0.1015** | Tuned on train folds to achieve ~50% recall |
| **Silhouette Score (Clustering)** | **0.141** | $K=4$ clusters evaluated across Euclidean/PCA space |
| **Probability Calibration Drift** | **< 0.005** | Mean predicted probability (8.92%) $\approx$ True rate (8.97%) |

Detailed methodology, fairness audit, and limitations are documented in [`reports/model_card.md`](reports/model_card.md) and [`reports/cleaning_report.md`](reports/cleaning_report.md).

---

## 🖥️ Web Application Features

| Page | Functionality |
|---|---|
| 🔮 **Predictor** | Interactive patient admission entry $\to$ calibrated risk score, risk tier (`Low` 🟢 / `Medium` 🟡 / `High` 🔴), assigned cohort, and dynamic **SHAP force plot** explanation. |
| 🗂️ **Cohort Explorer** | Interactive 2D PCA projection of discovered patient cohorts with clinical profile metric summaries. |
| 📤 **Batch Upload** | Upload CSV batches of patient records, run automated vector transformations, score all patients simultaneously, and export prediction CSVs. |
| 📊 **Model Insights** | Full model card dashboard including ROC/PR curves, probability calibration plots, SHAP beeswarm/waterfall plots, and demographic fairness audits across race and gender subgroups. |

---

## 📁 Repository Structure

```text
readmission-risk-system/
├── app/                        # Streamlit Multi-Page Web Application
│   ├── pages/
│   │   ├── 1_🔮_Predictor.py       # Single-patient risk scoring & SHAP
│   │   ├── 2_🗂️_Cohort_Explorer.py  # Interactive PCA cluster visualizer
│   │   ├── 3_📤_Batch_Upload.py     # Bulk CSV batch scoring & export
│   │   └── 4_📊_Model_Insights.py   # Performance, SHAP & fairness audit
│   ├── streamlit_app.py        # Application landing page
│   └── utils.py                # Cached artifact loading & feature helpers
├── data/
│   ├── raw/                    # Raw diabetic_data.csv & IDS_mapping.csv
│   └── processed/              # Cleaned encounters, train/test splits
├── models/                     # Trained joblib artifacts & model config
│   ├── base_xgb_model.joblib
│   ├── classifier.joblib       # Calibrated XGBoost model
│   ├── kmeans_model.joblib     # K-Means cohort model
│   ├── model_config.json       # Thresholds and calibration cutoffs
│   └── preprocessing_pipeline.joblib
├── notebooks/                  # Phase 1–4 exploratory research notebooks
├── reports/                    # Reports, model cards & figures
│   ├── eda_figures/            # Generated high-resolution plots
│   ├── cleaning_report.md      # Data cleaning report
│   ├── cohort_profiles.md      # Summary profiles per cluster
│   ├── evaluation_report.md    # Test set metrics
│   ├── fairness_audit.md       # Demographic fairness subgroup audit
│   └── model_card.md           # Formal ML Model Card
├── src/                        # Modular, reusable Python package
│   ├── __init__.py
│   ├── classifier.py           # XGBoost training & probability calibration
│   ├── clustering.py           # K-Means clustering & silhouette scoring
│   ├── data_prep.py            # Ingestion, deduplication & ICD-9 mapping
│   ├── evaluate.py             # ROC/PR evaluation, fairness audit & SHAP
│   ├── features.py             # Feature engineering & ColumnTransformer
│   └── train.py                # End-to-end CLI training orchestrator
├── tests/
│   └── test_pipeline.py        # Automated pytest test suite (25 tests)
├── .dockerignore
├── .gitignore
├── .python-version             # Python 3.11 runtime pin
├── Dockerfile                  # Container deployment configuration
├── requirements.txt            # Project dependencies
├── runtime.txt
└── README.md
```

---

## ⚡ Quickstart & Local Setup

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/SaiSandeep10/readmission-risk-system.git
cd readmission-risk-system

# Create and activate Python 3.11 virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest tests/ -v
```

### 3. Launch Web Application

```bash
streamlit run app/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔄 Reproducing Pipeline from Scratch

To re-run the entire pipeline from raw dataset ingestion through feature engineering, clustering, model calibration, and report generation, execute:

```bash
python -m src.train
```

---

## 🐳 Docker Deployment

To build and run the application container locally:

```bash
# Build the Docker image
docker build -t readmission-risk-system .

# Run the container
docker run -p 8501:8501 readmission-risk-system
```
Access the application at `http://localhost:8501`.

---

## 🔬 Clinical Methodology Highlights

- **Data Cleaning & Deduplication:** Filtered repeat admissions to the first chronological visit per patient (`patient_nbr`), preventing data leakage between training and evaluation splits. Excluded expired/hospice discharges.
- **ICD-9 Chapter Mapping:** Mapped 700+ raw diagnosis codes across 3 fields (`diag_1`, `diag_2`, `diag_3`) into 9 clinical categories (`Circulatory`, `Respiratory`, `Digestive`, `Diabetes`, `Injury`, `Musculoskeletal`, `Genitourinary`, `Neoplasm`, `Other`).
- **Probability Calibration:** Addressed class imbalance (~9% prevalence) using `CalibratedClassifierCV` (sigmoid) so that predicted probabilities reflect real-world event frequencies rather than artificial decision boundaries.
- **Key Risk Drivers:** Prior inpatient encounters and discharge disposition (transfers to rehabilitation/skilled nursing facilities vs. home) are confirmed as the strongest risk drivers.

---

## 👤 Author

**Sai Sandeep**  
*B.Tech in Computer Science & Engineering*  
*IBM AI/ML Internship (Q2D) Merit Submission*  
- **GitHub:** [@SaiSandeep10](https://github.com/SaiSandeep10)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.