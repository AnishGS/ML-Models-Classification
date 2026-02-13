from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data_definition import DataDefinition
from src.training.base_model_trainer import BaseModelTrainer


class LogisticRegressionTrainer(BaseModelTrainer):
    """Logistic Regression Classifier trainer"""
    
    def __init__(self, C=0.1, max_iter=2000):
        super().__init__(model_name='logistic_regression_model')
        self.C = C
        self.max_iter = max_iter
        
    def build_pipeline(self):
        """Build the Logistic Regression pipeline"""
        # Use RobustScaler for numeric features - more resistant to outliers
        numeric_transformer = Pipeline(
            steps=[('scaler', RobustScaler())]
        )
        
        # OneHotEncoder for categorical features
        categorical_transformer = Pipeline(
            steps=[('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, DataDefinition.numeric_cols),
                ('cat', categorical_transformer, DataDefinition.categorical_cols)
            ]
        )
        
        # Logistic Regression model with regularization
        model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,  # Regularization strength
            solver='saga',  # More stable solver
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Create pipeline
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )
        
        return pipeline


def train_logistic_regression():
    """Train Logistic Regression model"""
    trainer = LogisticRegressionTrainer(C=0.1, max_iter=2000)
    metrics = trainer.run()
    return trainer, metrics


if __name__ == '__main__':
    train_logistic_regression()

