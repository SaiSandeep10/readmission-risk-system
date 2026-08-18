"""
src/train.py
End-to-end training pipeline CLI runner.
Usage:
    python -m src.train
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from src.data_prep import prepare_and_save_data
from src.features import engineer_features, prepare_features_and_splits
from src.clustering import train_kmeans_model, generate_cohort_summary_and_plot, find_optimal_clusters
from src.classifier import train_and_calibrate_classifier
from src.evaluate import evaluate_classifier_performance, run_fairness_audit, generate_shap_summary_plot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports"
):
    logger.info("==================================================")
    logger.info("Starting Readmission Risk System Training Pipeline")
    logger.info("==================================================")

    # 1. Data Preparation
    logger.info("Step 1/5: Loading and cleaning raw dataset...")
    cleaned_df = prepare_and_save_data(raw_dir=raw_dir, processed_dir=processed_dir)
    logger.info(f"Cleaned dataset saved: {cleaned_df.shape[0]} encounters, {cleaned_df.shape[1]} columns")

    # 2. Feature Engineering & Splits
    logger.info("Step 2/5: Feature engineering & ColumnTransformer preprocessing pipeline...")
    X_train, X_test, y_train, y_test, pipeline = prepare_features_and_splits(
        cleaned_df, test_size=0.2, random_state=42, save_dir=processed_dir, models_dir=models_dir
    )
    logger.info(f"Train split: {X_train.shape[0]} rows | Test split: {X_test.shape[0]} rows")

    # Transform feature matrices
    X_train_proc = pipeline.transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    # 3. K-Means Cohort Clustering
    logger.info("Step 3/5: Fitting K-Means cohort clustering (K=4)...")
    kmeans_model = train_kmeans_model(X_train_proc, n_clusters=4, random_state=42, models_dir=models_dir)
    
    # Generate cohort summary and PCA visualization
    generate_cohort_summary_and_plot(cleaned_df, pipeline.transform(engineer_features(cleaned_df)), kmeans_model, reports_dir=reports_dir)
    logger.info("Cohort profiles and PCA visualization generated.")

    # Attach cluster assignments as an explicit feature
    train_clusters = kmeans_model.predict(X_train_proc)
    test_clusters = kmeans_model.predict(X_test_proc)

    X_train_final = np.hstack([X_train_proc, train_clusters.reshape(-1, 1)])
    X_test_final = np.hstack([X_test_proc, test_clusters.reshape(-1, 1)])

    # 4. Classifier Training & Probability Calibration
    logger.info("Step 4/5: Training and calibrating XGBoost classifier...")
    base_model, calibrated_model, config = train_and_calibrate_classifier(
        X_train_final, y_train, X_test_final, y_test, models_dir=models_dir, reports_dir=reports_dir
    )
    logger.info(f"Trained model saved. Optimal Operating Threshold: {config['operating_threshold']:.4f}")
    logger.info(f"Test ROC-AUC: {config['test_roc_auc']:.4f} | Recall @ threshold: {config['test_recall_at_threshold']:.4f}")

    # 5. Evaluation, Fairness Audit & SHAP Explainability
    logger.info("Step 5/5: Generating evaluation metrics, fairness audit, and SHAP plots...")
    probs_test = calibrated_model.predict_proba(X_test_final)[:, 1]
    evaluate_classifier_performance(y_test, probs_test, config['operating_threshold'], reports_dir=reports_dir)
    
    # Fairness audit
    run_fairness_audit(X_test, y_test, probs_test, config['operating_threshold'], reports_dir=reports_dir)

    # SHAP explanations
    feature_names = list(pipeline.named_steps['preprocessor'].get_feature_names_out()) + ['cohort_cluster']
    generate_shap_summary_plot(base_model, X_test_final, feature_names=feature_names)

    logger.info("==================================================")
    logger.info("✅ Training pipeline completed successfully!")
    logger.info("Artifacts saved to models/ and reports/")
    logger.info("==================================================")


if __name__ == "__main__":
    run_full_pipeline()
