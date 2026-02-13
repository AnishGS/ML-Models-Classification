"""
ML Classification Models - Assignment 2
M.Tech (AIML/DSE) - Machine Learning
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import numpy as np

# Page config
st.set_page_config(page_title="ML Classification Models", layout="wide")

# Title
st.title("ML Classification Models")
st.write("Assignment 2 - Machine Learning")

# Get project paths
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "model"
MODELS_DIR.mkdir(exist_ok=True)

# Available models
MODELS = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "K-Nearest Neighbors": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# ============================================================================
# 1. Dataset Upload (CSV)
# ============================================================================
st.header("1. Dataset Upload")
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=['csv'])

# Option to specify if target column is included
has_target = st.checkbox("Dataset includes target column (last column)", value=True)

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(test_data.head())

    # Split features and target based on user selection
    if has_target and len(test_data.columns) > 1:
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        st.write(f"Features: {X_test.shape[1]}, Samples: {X_test.shape[0]}, Target: {test_data.columns[-1]}")
    else:
        X_test = test_data
        y_test = None
        st.write(f"Features: {X_test.shape[1]}, Samples: {X_test.shape[0]}")

    # ============================================================================
    # 2. Model Selection Dropdown
    # ============================================================================
    st.header("2. Model Selection")

    # Check which models are available
    available_models = {}
    for name, filename in MODELS.items():
        model_path = MODELS_DIR / filename
        if model_path.exists():
            available_models[name] = str(model_path)

    if not available_models:
        st.warning("No trained models found. Please train models first.")
    else:
        selected_model = st.selectbox("Select Model", list(available_models.keys()))

        if st.button("Evaluate Model" if has_target else "Make Predictions"):
            try:
                # Load model
                model = joblib.load(available_models[selected_model])

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                if has_target and y_test is not None:
                    # ============================================================================
                    # WITH TARGET: Show evaluation metrics
                    # ============================================================================

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    mcc = matthews_corrcoef(y_test, y_pred)

                    # AUC Score
                    try:
                        if len(np.unique(y_test)) == 2:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except:
                        auc = 0.0

                    # ============================================================================
                    # 3. Display Evaluation Metrics
                    # ============================================================================
                    st.header("3. Evaluation Metrics")

                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                        'Value': [accuracy, auc, precision, recall, f1, mcc]
                    })
                    st.table(metrics_df)

                    # ============================================================================
                    # 4. Confusion Matrix and Classification Report
                    # ============================================================================
                    st.header("4. Confusion Matrix")

                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.write(cm)

                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred)
                    st.text(report)

                else:
                    # ============================================================================
                    # WITHOUT TARGET: Just show predictions
                    # ============================================================================
                    st.header("3. Predictions")

                    results_df = X_test.copy()
                    results_df['Predicted_Class'] = y_pred

                    # Add probability columns
                    for i in range(y_pred_proba.shape[1]):
                        results_df[f'Probability_Class_{i}'] = y_pred_proba[:, i]

                    st.write("Prediction Results:")
                    st.dataframe(results_df)

                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a CSV file to begin evaluation.")
