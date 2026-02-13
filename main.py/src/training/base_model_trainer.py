import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data_definition import DataDefinition


class BaseModelTrainer:
    """Base class for training classification models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.pipeline = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_dir = os.path.dirname(os.path.dirname(self.script_dir))
        
    def load_data(self):
        """Load the dataset"""
        data_path = os.path.join(self.project_dir, 'data', 'loan_data.csv')
        data = pd.read_csv(data_path)
        return data
    
    def clean_data(self, X, y):
        """Clean and preprocess the data"""
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
        print(f"Removed {mask.shape[0] - X.shape[0]} rows")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def build_pipeline(self):
        """Build the model pipeline - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement build_pipeline()")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("\nTraining model...")
        self.pipeline.fit(X_train, y_train)
        print("✓ Training complete!")
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model and print metrics"""
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Print all metrics
        print(f"\n{'='*50}")
        print(f"MODEL EVALUATION METRICS - {self.model_name}")
        print(f"{'='*50}")
        print(f"1. Accuracy:                    {accuracy:.4f}")
        print(f"2. AUC Score:                   {auc_score:.4f}")
        print(f"3. Precision:                   {precision:.4f}")
        print(f"4. Recall:                      {recall:.4f}")
        print(f"5. F1 Score:                    {f1:.4f}")
        print(f"6. Matthews Correlation Coeff:  {mcc:.4f}")
        print(f"{'='*50}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc
        }
    
    def save_model(self):
        """Save the trained model"""
        models_dir = os.path.join(self.project_dir, 'model')
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, f'{self.model_name}.pkl')
        joblib.dump(self.pipeline, model_path)
        print(f"\n{'='*50}")
        print(f"Model saved to: {model_path}")
        print(f"{'='*50}")
        
    def run(self):
        """Complete training pipeline"""
        # Load data
        data = self.load_data()
        X = data.drop(columns=[DataDefinition.target_col])
        y = data[DataDefinition.target_col]
        
        # Clean data
        X, y = self.clean_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Build pipeline
        self.pipeline = self.build_pipeline()
        
        # Train
        self.train(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test)
        
        # Save model
        self.save_model()
        
        return metrics

