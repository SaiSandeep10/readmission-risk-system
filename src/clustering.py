"""
src/clustering.py
Patient cohort discovery using K-Means clustering and PCA analysis.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


def find_optimal_clusters(
    X_proc: np.ndarray,
    k_range: range = range(2, 11),
    sample_size: int = 10000,
    random_state: int = 42,
    fig_dir: str = "reports/eda_figures"
) -> Tuple[int, Dict[str, Any]]:
    """
    Computes inertia and silhouette scores across a range of K values and saves the elbow plot.
    """
    os.makedirs(fig_dir, exist_ok=True)
    inertias = []
    silhouette_scores = []

    n_samples = X_proc.shape[0]
    sample_n = min(sample_size, n_samples)
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(n_samples, sample_n, replace=False)
    X_sample = X_proc[sample_idx]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_proc)
        inertias.append(km.inertia_)

        sample_labels = labels[sample_idx]
        sil = silhouette_score(X_sample, sample_labels)
        silhouette_scores.append(sil)

    # Plot elbow & silhouette curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(list(k_range), inertias, marker='o', color='#2E74B5')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')

    axes[1].plot(list(k_range), silhouette_scores, marker='o', color='#C0504D')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score vs K')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'elbow_and_silhouette.png'), dpi=150)
    plt.close()

    best_k = list(k_range)[int(np.argmax(silhouette_scores))]
    results = {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'best_k': best_k,
        'best_silhouette': max(silhouette_scores)
    }
    return best_k, results


def train_kmeans_model(
    X_train_proc: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
    models_dir: str = "models"
) -> KMeans:
    """
    Fits K-Means clustering model and saves artifact.
    """
    os.makedirs(models_dir, exist_ok=True)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X_train_proc)

    model_path = os.path.join(models_dir, "kmeans_model.joblib")
    joblib.dump(km, model_path)
    return km


def generate_cohort_summary_and_plot(
    df: pd.DataFrame,
    X_proc: np.ndarray,
    kmeans_model: KMeans,
    reports_dir: str = "reports",
    fig_dir: str = "reports/eda_figures"
) -> pd.DataFrame:
    """
    Generates PCA scatter plot and cohort summary Markdown report.
    """
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    clusters = kmeans_model.predict(X_proc)
    df_eval = df.copy()
    df_eval['cluster'] = clusters

    # 2D PCA projection for visualization
    sample_size = min(5000, len(df_eval))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(df_eval), sample_size, replace=False)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_proc[sample_idx])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=clusters[sample_idx].astype(str),
        palette='tab10', alpha=0.5
    )
    plt.title("Patient Cohorts (PCA 2D Projection)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.legend(title="Cohort Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cluster_scatter.png"), dpi=150)
    plt.close()

    # Cohort profile statistics
    profile_cols = ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_diagnoses', 'total_prior_utilization']
    profile_cols = [c for c in profile_cols if c in df_eval.columns]

    profiles = df_eval.groupby('cluster')[profile_cols].mean().round(2)
    profiles['Patient Count'] = df_eval['cluster'].value_counts().sort_index()
    if 'readmitted_binary' in df_eval.columns:
        profiles['Readmission Rate'] = df_eval.groupby('cluster')['readmitted_binary'].mean().map(lambda x: f"{x:.1%}")

    # Write cohort_profiles.md
    md_content = "# Patient Cohort Profiles\n\n"
    md_content += f"Discovered {kmeans_model.n_clusters} clinical cohorts using K-Means clustering.\n\n"
    md_content += profiles.to_markdown() + "\n"

    with open(os.path.join(reports_dir, "cohort_profiles.md"), "w") as f:
        f.write(md_content)

    return profiles
