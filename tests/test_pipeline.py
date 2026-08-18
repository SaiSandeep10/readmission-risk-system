# tests/test_pipeline.py
"""
Unit tests for the readmission risk system.
Run from project root: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from utils import engineer_features


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def cleaned_data():
    path = "data/processed/cleaned_data.csv"
    assert os.path.exists(path), f"Missing file: {path}"
    return pd.read_csv(path, keep_default_na=False)

@pytest.fixture(scope="module")
def pipeline():
    path = "models/preprocessing_pipeline.joblib"
    assert os.path.exists(path), f"Missing file: {path}"
    return joblib.load(path)

@pytest.fixture(scope="module")
def kmeans_model():
    path = "models/kmeans_model.joblib"
    assert os.path.exists(path), f"Missing file: {path}"
    return joblib.load(path)

@pytest.fixture(scope="module")
def classifier():
    path = "models/classifier.joblib"
    assert os.path.exists(path), f"Missing file: {path}"
    return joblib.load(path)

@pytest.fixture(scope="module")
def base_model():
    path = "models/base_xgb_model.joblib"
    assert os.path.exists(path), f"Missing file: {path}"
    return joblib.load(path)

@pytest.fixture(scope="module")
def model_config():
    path = "models/model_config.json"
    assert os.path.exists(path), f"Missing file: {path}"
    with open(path) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def sample_patient(cleaned_data):
    drop_cols = [c for c in ['patient_nbr', 'readmitted', 'readmitted_binary']
                 if c in cleaned_data.columns]
    df = cleaned_data.drop(columns=drop_cols).iloc[[0]].copy()
    df = engineer_features(df)
    return df


# ---------- Data integrity tests ----------

class TestCleanedData:

    def test_no_missing_values(self, cleaned_data):
        null_counts = cleaned_data.isnull().sum()
        assert null_counts.sum() == 0, f"Nulls found:\n{null_counts[null_counts > 0]}"

    def test_target_column_exists_and_binary(self, cleaned_data):
        assert 'readmitted_binary' in cleaned_data.columns
        assert set(cleaned_data['readmitted_binary'].unique()) <= {0, 1}

    def test_target_class_balance_reasonable(self, cleaned_data):
        rate = cleaned_data['readmitted_binary'].mean()
        assert 0.03 < rate < 0.25, f"Unexpected positive class rate: {rate:.3f}"

    def test_age_ordinal_no_nulls_and_in_range(self, cleaned_data):
        assert cleaned_data['age_ordinal'].isnull().sum() == 0
        assert cleaned_data['age_ordinal'].between(0, 9).all()

    def test_diag_groups_exist(self, cleaned_data):
        for col in ['diag_1_group', 'diag_2_group', 'diag_3_group']:
            assert col in cleaned_data.columns
            assert cleaned_data[col].isnull().sum() == 0

    def test_no_duplicate_patients(self, cleaned_data):
        if 'patient_nbr' in cleaned_data.columns:
            assert cleaned_data['patient_nbr'].duplicated().sum() == 0


# ---------- Feature engineering tests ----------

class TestFeatureEngineering:

    def test_engineer_features_adds_required_columns(self, sample_patient):
        assert 'total_prior_utilization' in sample_patient.columns
        assert 'num_meds_changed' in sample_patient.columns

    def test_no_nan_after_glucose_a1c_fill(self, sample_patient):
        for col in ['max_glu_serum', 'A1Cresult']:
            if col in sample_patient.columns:
                assert sample_patient[col].isnull().sum() == 0


# ---------- Preprocessing pipeline tests ----------

class TestPreprocessingPipeline:

    def test_pipeline_loads(self, pipeline):
        assert pipeline is not None

    def test_pipeline_transforms_without_error(self, pipeline, sample_patient):
        result = pipeline.transform(sample_patient)
        assert result is not None

    def test_pipeline_output_no_nans(self, pipeline, sample_patient):
        result = pipeline.transform(sample_patient)
        result = np.asarray(result, dtype=np.float64)
        assert not np.isnan(result).any(), "NaNs leaked through the preprocessing pipeline"

    def test_pipeline_output_shape_consistent(self, pipeline, sample_patient):
        result_1 = pipeline.transform(sample_patient)
        result_5 = pipeline.transform(pd.concat([sample_patient] * 5, ignore_index=True))
        result_1 = np.asarray(result_1, dtype=np.float64)
        result_5 = np.asarray(result_5, dtype=np.float64)
        assert result_1.shape[1] == result_5.shape[1]
        assert result_5.shape[0] == 5

    def test_pipeline_handles_unseen_category(self, pipeline, sample_patient):
        modified = sample_patient.copy()
        modified['race'] = 'SomeUnseenCategoryXYZ'
        result = pipeline.transform(modified)
        result = np.asarray(result, dtype=np.float64)
        assert not np.isnan(result).any()


# ---------- Clustering tests ----------

class TestClustering:

    def test_kmeans_loads(self, kmeans_model):
        assert kmeans_model is not None
        assert hasattr(kmeans_model, 'cluster_centers_')

    def test_kmeans_predicts_valid_cluster(self, pipeline, kmeans_model, sample_patient):
        X_proc = np.asarray(pipeline.transform(sample_patient), dtype=np.float64)
        cluster = kmeans_model.predict(X_proc)
        assert cluster[0] in range(kmeans_model.n_clusters)


# ---------- Classifier tests ----------

class TestClassifier:

    def test_classifier_loads(self, classifier):
        assert classifier is not None

    def test_base_model_loads(self, base_model):
        assert base_model is not None

    def test_classifier_predicts_valid_probability(self, pipeline, kmeans_model, classifier, sample_patient):
        X_proc = np.asarray(pipeline.transform(sample_patient), dtype=np.float64)
        cluster = kmeans_model.predict(X_proc)
        X_final = np.hstack([X_proc, cluster.reshape(-1, 1)])
        prob = classifier.predict_proba(X_final)[:, 1][0]
        assert 0.0 <= prob <= 1.0

    def test_classifier_is_calibrated(self, classifier):
        """Confirms we're using the calibrated wrapper, not the raw model."""
        assert hasattr(classifier, 'calibrated_classifiers_') or \
               'Calibrated' in type(classifier).__name__

    def test_classifier_deterministic(self, pipeline, kmeans_model, classifier, sample_patient):
        X_proc = np.asarray(pipeline.transform(sample_patient), dtype=np.float64)
        cluster = kmeans_model.predict(X_proc)
        X_final = np.hstack([X_proc, cluster.reshape(-1, 1)])
        prob_1 = classifier.predict_proba(X_final)[:, 1][0]
        prob_2 = classifier.predict_proba(X_final)[:, 1][0]
        assert prob_1 == prob_2


# ---------- Model config tests ----------

class TestModelConfig:

    def test_config_has_required_keys(self, model_config):
        required = ['operating_threshold', 'risk_tier_low_cutoff', 'risk_tier_high_cutoff']
        for key in required:
            assert key in model_config, f"Missing config key: {key}"

    def test_config_thresholds_ordered(self, model_config):
        assert model_config['risk_tier_low_cutoff'] < model_config['risk_tier_high_cutoff']

    def test_config_probability_is_calibrated(self, model_config):
        """Mean predicted probability should be close to the true positive rate."""
        diff = abs(model_config['mean_test_probability'] - model_config['test_positive_rate'])
        assert diff < 0.02, f"Calibration drift too high: {diff:.4f}"


# ---------- End-to-end integration test ----------

class TestEndToEnd:

    def test_full_prediction_pipeline(self, pipeline, kmeans_model, classifier, sample_patient):
        X_proc = np.asarray(pipeline.transform(sample_patient), dtype=np.float64)
        cluster = kmeans_model.predict(X_proc)
        X_final = np.hstack([X_proc, cluster.reshape(-1, 1)])
        prob = classifier.predict_proba(X_final)[:, 1][0]

        assert isinstance(prob, (float, np.floating))
        assert 0.0 <= prob <= 1.0
        assert cluster[0] >= 0

    def test_shap_explainer_works_on_base_model(self, base_model, pipeline, kmeans_model, sample_patient):
        import shap
        X_proc = np.asarray(pipeline.transform(sample_patient), dtype=np.float64)
        cluster = kmeans_model.predict(X_proc)
        X_final = np.hstack([X_proc, cluster.reshape(-1, 1)])

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_final)
        assert shap_values is not None