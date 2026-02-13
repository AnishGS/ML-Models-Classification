"""
Loan Default Prediction - ML Classification Models
Assignment 2: M.Tech (AIML/DSE) - Machine Learning
Implements 6 classification algorithms for loan approval prediction
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction - ML Models",
    layout="wide"
)

# Header section
st.title("Loan Default Prediction System")
st.markdown("**Assignment 2 - Machine Learning Classification Models**")
st.markdown("**Submitted by:** Anish Sharma | **Email:** 2025aa05225@wilp.bits-pilani.ac.in")
st.markdown("---")

# Information box
st.info("""
**About this Application:**
- Compares 6 ML classification algorithms for loan default prediction
- Dataset: Loan Approval Dataset (45,000 instances, 13 features)
- Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost
- Evaluation: 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC) + Confusion Matrix
""")

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
st.header("Step 1: Upload Test Dataset")
st.markdown("Upload your test dataset in CSV format. The target column should be the **last column**.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload test data with features and target column (loan_status)"
)

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    st.success(f"Dataset uploaded successfully! ({len(test_data)} rows)")

    # Dataset preview in expandable section
    with st.expander("View Dataset Preview", expanded=True):
        st.dataframe(test_data.head(10), use_container_width=True)

    # Split features and target (target is last column)
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Display dataset information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Features", X_test.shape[1])
    with col2:
        st.metric("Number of Samples", X_test.shape[0])
    with col3:
        st.metric("Target Column", test_data.columns[-1])

    st.markdown("---")

    # ============================================================================
    # 2. Model Selection Dropdown
    # ============================================================================
    st.header("Step 2: Select Classification Model")

    # Check which models are available
    available_models = {}
    for name, filename in MODELS.items():
        model_path = MODELS_DIR / filename
        if model_path.exists():
            available_models[name] = str(model_path)

    if not available_models:
        st.warning("No trained models found. Please train models first using `python -m src.train_models`")
    else:
        st.markdown("Choose one of the 6 trained classification models:")

        selected_model = st.selectbox(
            "Select Model",
            list(available_models.keys()),
            help="All models are trained on the same training dataset"
        )

        st.markdown("---")

        # Evaluate button with custom styling
        if st.button("Evaluate Model", type="primary", use_container_width=True):
            try:
                with st.spinner(f'Loading {selected_model} and making predictions...'):
                    # Load model
                    model = joblib.load(available_models[selected_model])

                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    mcc = matthews_corrcoef(y_test, y_pred)

                    # AUC Score (handles binary classification)
                    try:
                        if len(np.unique(y_test)) == 2:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    except:
                        auc = 0.0

                st.success(f"Model evaluation completed for **{selected_model}**!")
                st.markdown("---")

                # ============================================================================
                # 3. Display Evaluation Metrics
                # ============================================================================
                st.header("Step 3: Evaluation Metrics")
                st.markdown(f"Performance metrics for **{selected_model}** on test dataset:")

                # Display metrics in a formatted table
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                    'Value': [f"{accuracy:.4f}", f"{auc:.4f}", f"{precision:.4f}",
                             f"{recall:.4f}", f"{f1:.4f}", f"{mcc:.4f}"]
                })

                st.table(metrics_df)

                # Highlight best metrics for imbalanced data
                st.info(f"**Key Metrics for Imbalanced Data:** F1 Score = {f1:.4f}, MCC = {mcc:.4f}")

                st.markdown("---")

                # ============================================================================
                # 4. Confusion Matrix
                # ============================================================================
                st.header("Step 4: Confusion Matrix")
                st.markdown("Confusion matrix shows the model's prediction accuracy for each class:")

                cm = confusion_matrix(y_test, y_pred)

                # Display confusion matrix as styled DataFrame
                cm_df = pd.DataFrame(
                    cm,
                    columns=[f'Predicted {i}' for i in range(cm.shape[1])],
                    index=[f'Actual {i}' for i in range(cm.shape[0])]
                )

                # Display with custom styling
                st.dataframe(cm_df, use_container_width=True)

                # Add interpretation
                st.markdown(f"""
                **Interpretation:**
                - True Negatives (TN): {cm[0][0]} - Correctly predicted as Class 0
                - False Positives (FP): {cm[0][1]} - Incorrectly predicted as Class 1
                - False Negatives (FN): {cm[1][0]} - Incorrectly predicted as Class 0
                - True Positives (TP): {cm[1][1]} - Correctly predicted as Class 1
                """)

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                st.info("Please ensure the uploaded dataset has the correct format and features.")
else:
    st.info("Please upload a CSV file to begin model evaluation.")

    # Instructions when no file is uploaded
    with st.expander("How to use this application"):
        st.markdown("""
        **Steps to evaluate models:**
        1. Upload your test dataset (CSV format) with target column as the last column
        2. Select one of the 6 trained classification models
        3. Click 'Evaluate Model' to see performance metrics and confusion matrix

        **Expected Dataset Format:**
        - CSV file with features and target column
        - Target column should be the **last column**
        - Example: `data/test_data.csv` (9,000 samples, 13 columns)

        **Available Models:**
        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbors (kNN)
        - Naive Bayes
        - Random Forest (Ensemble)
        - XGBoost (Ensemble)
        """)
