import streamlit as st
import pandas as pd
import numpy as np
import sys, os

# Add parent directory to path for utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_artifacts, risk_tier, engineer_features

# Streamlit page setup
st.set_page_config(page_title="Batch Upload", page_icon="📤", layout="wide")
st.title("📤 Batch Patient Risk Scoring")

st.markdown("""
Upload a CSV of multiple patient encounters to score them all at once.
The file should contain the same columns as the training data (minus the target
columns `readmitted` / `readmitted_binary`, which will be ignored if present).
""")

# Load ML artifacts
pipeline, kmeans, classifier, base_model, config = load_artifacts()

# File uploader
uploaded_file = st.file_uploader("Upload a CSV of patient records", type=['csv'])

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"✅ Loaded {len(batch_df)} rows.")
        st.dataframe(batch_df.head())

        if st.button("Run Predictions", type="primary"):
            with st.spinner("Scoring patients..."):
                # Drop target/ID columns if present — not model inputs
                drop_cols = [c for c in ['patient_nbr', 'readmitted', 'readmitted_binary', 'cluster']
                             if c in batch_df.columns]
                X = batch_df.drop(columns=drop_cols)

                # Recreate Phase 3 engineered features
                # (total_prior_utilization, num_meds_changed) — not present in raw uploads
                X = engineer_features(X)

                # max_glu_serum / A1Cresult use 'None' to mean "test not taken" —
                # restore that convention for any real NaNs that slipped through
                for col in ['max_glu_serum', 'A1Cresult']:
                    if col in X.columns:
                        X[col] = X[col].fillna('None')

                # Safety net: fill any other unexpected NaNs in categorical columns
                cat_cols = X.select_dtypes(include='object').columns
                X[cat_cols] = X[cat_cols].fillna('Unknown')

                # Transform using the saved pipeline directly — it already handles
                # one-hot encoding, ordinal encoding, and scaling internally
                X_proc = pipeline.transform(X)

                # Force to a plain float64 numpy array regardless of what the
                # pipeline returned (DataFrame, sparse matrix, etc.)
                if hasattr(X_proc, "toarray"):
                    X_proc = X_proc.toarray()
                if isinstance(X_proc, pd.DataFrame):
                    X_proc = X_proc.to_numpy()
                X_proc = np.asarray(X_proc, dtype=np.float64)

                # Assign cohort cluster
                clusters = kmeans.predict(X_proc)

                # Build final feature matrix (matches training-time shape)
                X_final = np.hstack([X_proc, clusters.reshape(-1, 1)])

                # Predict readmission probability
                probs = classifier.predict_proba(X_final)[:, 1]

                # Attach results to the original dataframe
                batch_df['readmission_probability'] = probs.round(4)
                batch_df['risk_tier'] = [risk_tier(p, config)[0] for p in probs]
                batch_df['cohort_cluster'] = clusters

            st.success(f"✅ Predictions complete for {len(batch_df)} patients.")

            # Summary stats
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 High Risk", int((batch_df['risk_tier'] == 'High').sum()))
            col2.metric("🟡 Medium Risk", int((batch_df['risk_tier'] == 'Medium').sum()))
            col3.metric("🟢 Low Risk", int((batch_df['risk_tier'] == 'Low').sum()))

            st.dataframe(batch_df, use_container_width=True)

            # Download button
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "batch_predictions.csv",
                "text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.info("Make sure your CSV has the same columns as the training data "
                 "(see the format expected by the Predictor page).")
else:
    st.info("ℹ️ Upload a CSV with the same columns as the training data (minus the target).")