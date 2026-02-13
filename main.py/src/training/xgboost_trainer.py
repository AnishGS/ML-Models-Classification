from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data_definition import DataDefinition
from src.training.base_model_trainer import BaseModelTrainer


class XGBoostTrainer(BaseModelTrainer):
    """XGBoost Classifier trainer"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8):
        super().__init__(model_name='xgboost_model')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        
    def build_pipeline(self):
        """Build the XGBoost pipeline"""
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
        
        # XGBoost model
        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=42,
            eval_metric='logloss',  # Suppress warning
            use_label_encoder=False,  # Suppress warning
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


def train_xgboost():
    """Train XGBoost model"""
    trainer = XGBoostTrainer(n_estimators=100, max_depth=6, learning_rate=0.1)
    metrics = trainer.run()
    return trainer, metrics


if __name__ == '__main__':
    train_xgboost()

