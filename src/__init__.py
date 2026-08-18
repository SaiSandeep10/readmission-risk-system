"""
Clinical Readmission Risk & Cohort Clustering System
Package initialization for pipeline modules.
"""

from src.data_prep import clean_encounter_data, load_raw_data, prepare_and_save_data
from src.features import engineer_features, build_preprocessing_pipeline, prepare_features_and_splits
from src.clustering import train_kmeans_model, generate_cohort_summary_and_plot
from src.classifier import train_and_calibrate_classifier
from src.evaluate import evaluate_classifier_performance, run_fairness_audit, generate_shap_summary_plot

__all__ = [
    "clean_encounter_data",
    "load_raw_data",
    "prepare_and_save_data",
    "engineer_features",
    "build_preprocessing_pipeline",
    "prepare_features_and_splits",
    "train_kmeans_model",
    "generate_cohort_summary_and_plot",
    "train_and_calibrate_classifier",
    "evaluate_classifier_performance",
    "run_fairness_audit",
    "generate_shap_summary_plot",
]
