from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data_definition import DataDefinition
from src.training.base_model_trainer import BaseModelTrainer


class NaiveBayesTrainer(BaseModelTrainer):
    """Naive Bayes Classifier trainer (Gaussian)"""
    
    def __init__(self, var_smoothing=1e-9):
        super().__init__(model_name='naive_bayes_model')
        self.var_smoothing = var_smoothing
        
    def build_pipeline(self):
        """Build the Naive Bayes pipeline"""
        # Use RobustScaler for numeric features
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
        
        # Gaussian Naive Bayes model
        # var_smoothing: Portion of the largest variance added to variances for stability
        model = GaussianNB(
            var_smoothing=self.var_smoothing
        )
        
        # Create pipeline
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )
        
        return pipeline


def train_naive_bayes():
    """Train Naive Bayes model"""
    trainer = NaiveBayesTrainer(var_smoothing=1e-9)
    metrics = trainer.run()
    return trainer, metrics


if __name__ == '__main__':
    train_naive_bayes()

