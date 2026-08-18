"""
src/classifier.py
Training, probability calibration, and threshold optimization for XGBoost readmission classifier.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import cross_val_predict


DEFAULT_XGB_PARAMS = {
    "subsample": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "n_estimators": 400,
    "min_child_weight": 3,
    "max_depth": 3,
    "learning_rate": 0.05,
    "colsample_bytree": 0.7
}


def compute_operating_threshold(
    base_model_class,
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_recall: float = 0.5,
    cv: int = 5,
    random_state: int = 42
) -> float:
    """
    Determines decision threshold using out-of-fold calibrated probabilities on training set
    to achieve the target clinical recall without test set leakage.
    """
    oof_probs = cross_val_predict(
        CalibratedClassifierCV(base_model_class, method='sigmoid', cv=3),
        X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1
    )[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)
    idx = np.argmin(np.abs(recalls - target_recall))
    operating_threshold = thresholds[min(idx, len(thresholds) - 1)]
    return float(operating_threshold)


def train_and_calibrate_classifier(
    X_train_final: np.ndarray,
    y_train: pd.Series,
    X_test_final: np.ndarray,
    y_test: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Tuple[XGBClassifier, CalibratedClassifierCV, Dict[str, Any]]:
    """
    Fits base XGBoost model with class weighting, applies CalibratedClassifierCV (sigmoid),
    determines operating threshold, computes risk tier percentiles, and persists artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if params is None:
        params_file = os.path.join(reports_dir, "best_params.json")
        if os.path.exists(params_file):
            with open(params_file, "r") as f:
                params = json.load(f)
        else:
            params = DEFAULT_XGB_PARAMS.copy()

    # Calculate class imbalance weighting
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    # 1. Base model for SHAP explanations
    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        **params
    )
    base_model.fit(X_train_final, y_train)

    base_model_path = os.path.join(models_dir, "base_xgb_model.joblib")
    joblib.dump(base_model, base_model_path)

    # 2. Calibrated model for unbiased clinical probabilities
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train_final, y_train)

    calibrated_model_path = os.path.join(models_dir, "classifier.joblib")
    joblib.dump(calibrated_model, calibrated_model_path)

    # 3. Decision threshold tuning on train fold
    operating_threshold = compute_operating_threshold(
        XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, n_jobs=1, **params),
        X_train_final, y_train.to_numpy(), target_recall=0.5
    )

    # 4. Evaluation metrics on test set
    probs_test = calibrated_model.predict_proba(X_test_final)[:, 1]
    y_pred_test = (probs_test >= operating_threshold).astype(int)

    config = {
        'operating_threshold': float(operating_threshold),
        'mean_test_probability': float(probs_test.mean()),
        'test_positive_rate': float(y_test.mean()),
        'test_roc_auc': float(roc_auc_score(y_test, probs_test)),
        'test_recall_at_threshold': float(recall_score(y_test, y_pred_test)),
        'test_precision_at_threshold': float(precision_score(y_test, y_pred_test)),
        'risk_tier_low_cutoff': float(np.percentile(probs_test, 50)),
        'risk_tier_high_cutoff': float(np.percentile(probs_test, 85)),
    }

    config_path = os.path.join(models_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return base_model, calibrated_model, config
