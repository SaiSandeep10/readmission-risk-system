"""
src/evaluate.py
Evaluation metrics, fairness audit, calibration analysis, and SHAP explainability.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, recall_score,
    precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.calibration import calibration_curve


def evaluate_classifier_performance(
    y_test: pd.Series,
    probs_test: np.ndarray,
    operating_threshold: float,
    reports_dir: str = "reports",
    fig_dir: str = "reports/eda_figures"
) -> Dict[str, float]:
    """
    Computes standard classification metrics, creates ROC/PR/Confusion Matrix plots,
    and writes reports/evaluation_report.md.
    """
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    y_pred = (probs_test >= operating_threshold).astype(int)

    precision_arr, recall_arr, _ = precision_recall_curve(y_test, probs_test)
    pr_auc = float(auc(recall_arr, precision_arr))
    roc_auc = float(roc_auc_score(y_test, probs_test))
    rec = float(recall_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    metrics = {
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'recall': round(rec, 4),
        'precision': round(prec, 4),
        'f1': round(f1, 4)
    }

    # Write evaluation report
    md_content = "# Model Evaluation Report\n\n"
    md_content += "**Best model:** XGBoost (tuned & calibrated)\n\n"
    md_content += "| Metric | Value |\n|---|---|\n"
    for k, v in metrics.items():
        md_content += f"| {k} | {v} |\n"

    with open(os.path.join(reports_dir, "evaluation_report.md"), "w") as f:
        f.write(md_content)

    # Generate multi-panel evaluation plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    fpr, tpr, _ = roc_curve(y_test, probs_test)
    axes[0].plot(fpr, tpr, color='#2E74B5', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_title('ROC Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

    axes[1].plot(recall_arr, precision_arr, color='#C0504D', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
    axes[1].axvline(rec, color='gray', linestyle=':', label=f'Operating point ({rec:.2f})')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Readmit', 'Readmit <30d'])
    disp.plot(ax=axes[2], cmap='Blues')
    axes[2].set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_evaluation.png"), dpi=150)
    plt.close()

    # Generate calibration curve plot
    prob_true, prob_pred = calibration_curve(y_test, probs_test, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', color='#2E74B5', label='Calibrated XGBoost')
    plt.plot([0, 0.4], [0, 0.4], 'k--', alpha=0.5, label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Observed Positive Fraction')
    plt.title('Probability Calibration Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "calibration_curve.png"), dpi=150)
    plt.close()

    return metrics


def run_fairness_audit(
    df_test_raw: pd.DataFrame,
    y_test: pd.Series,
    probs_test: np.ndarray,
    operating_threshold: float,
    reports_dir: str = "reports"
) -> pd.DataFrame:
    """
    Performs demographic subgroup fairness audit for race and gender groups,
    computing recall, false negative rate, and sample size.
    """
    os.makedirs(reports_dir, exist_ok=True)
    y_pred = (probs_test >= operating_threshold).astype(int)

    eval_df = df_test_raw.copy().reset_index(drop=True)
    eval_df['actual'] = y_test.values
    eval_df['predicted'] = y_pred

    audit_rows = []
    for attr in ['race', 'gender']:
        if attr in eval_df.columns:
            for group, grp_df in eval_df.groupby(attr):
                n = len(grp_df)
                actual_pos = (grp_df['actual'] == 1).sum()
                if actual_pos > 0:
                    tp = ((grp_df['actual'] == 1) & (grp_df['predicted'] == 1)).sum()
                    rec = tp / actual_pos
                    fnr = 1.0 - rec
                else:
                    rec, fnr = np.nan, np.nan
                
                audit_rows.append({
                    'attribute': attr,
                    'group': group,
                    'n': n,
                    'recall': round(rec, 3) if not np.isnan(rec) else None,
                    'false_negative_rate': round(fnr, 3) if not np.isnan(fnr) else None
                })

    audit_df = pd.DataFrame(audit_rows)
    md_content = audit_df.to_markdown(index=False)
    with open(os.path.join(reports_dir, "fairness_audit.md"), "w") as f:
        f.write(md_content + "\n")

    return audit_df


def generate_shap_summary_plot(
    base_model,
    X_test_final: np.ndarray,
    feature_names: Optional[List[str]] = None,
    fig_dir: str = "reports/eda_figures",
    sample_size: int = 2000
) -> None:
    """
    Generates SHAP summary beeswarm plot and saves to eda_figures/shap_summary.png.
    """
    os.makedirs(fig_dir, exist_ok=True)

    n_samples = min(sample_size, X_test_final.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X_test_final.shape[0], n_samples, replace=False)
    X_sample = X_test_final[sample_idx]

    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
