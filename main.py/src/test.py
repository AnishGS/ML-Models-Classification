from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src import data_reader
from src.data_definition import DataDefinition
from src.pipeline_buider import PipelineBuilder

import os
import numpy as np

def logistic_regression():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to main.py directory, then into data folder
    data_path = os.path.join(os.path.dirname(script_dir), 'data', 'loan_data.csv')
    reader = data_reader.DataReader(data_path)
    data = reader.load_data()
    pipeline_builder = PipelineBuilder()
    scale_encode_train_test_pipeline = pipeline_builder.build_pipeline()

    X = data.drop(columns=[DataDefinition.target_col])
    y = data[DataDefinition.target_col]

    # Handle outliers using IQR method or capping
    print("Original data shape:", X.shape)
    print("\nRemoving outliers and cleaning data...")

    # Remove rows with any NaN or infinite values
    mask = ~(X.isna().any(axis=1) | np.isinf(X.select_dtypes(include=[np.number])).any(axis=1))
    X = X[mask]
    y = y[mask]

    # Remove impossible/extreme values
    # Age should be reasonable (20-80)
    mask = (X['person_age'] >= 20) & (X['person_age'] <= 80)
    X = X[mask]
    y = y[mask]

    # Employment experience should be less than age - 18
    mask = X['person_emp_exp'] <= (X['person_age'] - 18)
    X = X[mask]
    y = y[mask]

    # Remove extreme income outliers (using 99th percentile)
    income_99th = X['person_income'].quantile(0.99)
    mask = X['person_income'] <= income_99th
    X = X[mask]
    y = y[mask]

    # Cap loan amount at 99th percentile
    loan_99th = X['loan_amnt'].quantile(0.99)
    X.loc[X['loan_amnt'] > loan_99th, 'loan_amnt'] = loan_99th

    print(f"Data shape after outlier removal: {X.shape}")
    print(f"Removed {data.shape[0] - X.shape[0]} rows ({100 * (data.shape[0] - X.shape[0]) / data.shape[0]:.2f}%)")

    X_train, X_test, y_train, y_test = (train_test_split
        (
      X, y, test_size=0.2, random_state=42, stratify=y
         )
    )
    # Check for NaN or infinite values before training
    print("\nChecking data quality after cleaning...")
    print(f"NaN in X_train: {X_train.isna().sum().sum()}")
    print(f"Inf in X_train: {np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"X_train shape: {X_train.shape}")
    print(f"\nX_train numeric columns stats:\n{X_train[DataDefinition.numeric_cols].describe()}")

    scale_encode_train_test_pipeline.fit(X_train, y_train)

    Xt = scale_encode_train_test_pipeline.named_steps['preprocessor'].transform(X_train)
    # Fix: Convert sparse matrix to array if needed, then get max/min
    if hasattr(Xt, 'toarray'):
        Xt_array = Xt.toarray()
    else:
        Xt_array = np.asarray(Xt)
    print("Max value:", Xt_array.max())
    print("Min value:", Xt_array.min())

    # Make predictions and evaluate
    y_pred = scale_encode_train_test_pipeline.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Uncomment to use custom threshold
    # THRESHOLD = 0.7
    # y_prob = scale_encode_train_test_pipeline.predict_proba(X_test)[:, 1]
    # y_pred = (y_prob >= THRESHOLD).astype(int)

    # Uncomment to plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    logistic_regression()