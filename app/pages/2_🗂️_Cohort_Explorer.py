import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os
from sklearn.decomposition import PCA

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_artifacts, load_reference_data, engineer_features

st.set_page_config(page_title="Cohort Explorer", page_icon="🗂️", layout="wide")
st.title("🗂️ Patient Cohort Explorer")

# ✅ Fix unpacking here
pipeline, kmeans, classifier, _, _ = load_artifacts()
df = load_reference_data()

st.markdown("Each point is a patient encounter, colored by which cohort (cluster) they belong to.")

# Sample data
sample_df = df.sample(min(5000, len(df)), random_state=42)

# Drop non-feature columns
drop_cols = [c for c in ['patient_nbr', 'readmitted', 'readmitted_binary', 'cluster'] if c in sample_df.columns]
X = sample_df.drop(columns=drop_cols)

# Apply feature engineering to X
X = engineer_features(X)

# Transform features and predict clusters
X_proc = pipeline.transform(X)
clusters = kmeans.predict(X_proc)

# PCA projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_proc)

plot_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': clusters.astype(str)
})

fig = px.scatter(
    plot_df, x='PC1', y='PC2', color='Cluster',
    title="Patient Cohorts (PCA Projection)", opacity=0.5
)
st.plotly_chart(fig, use_container_width=True)

# Cohort profiles
sample_df['cluster'] = clusters
st.subheader("Cohort Profiles")

profile_cols = ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_diagnoses']
profile_cols = [c for c in profile_cols if c in sample_df.columns]

profile = sample_df.groupby('cluster')[profile_cols].mean().round(1)
profile['size'] = sample_df['cluster'].value_counts().sort_index()

st.dataframe(profile, use_container_width=True)