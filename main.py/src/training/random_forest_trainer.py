from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data_definition import DataDefinition
from src.training.base_model_trainer import BaseModelTrainer


class RandomForestTrainer(BaseModelTrainer):
    """Random Forest Classifier trainer"""
    
    def __init__(self, n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5):
        super().__init__(model_name='random_forest_model')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def build_pipeline(self):
        """Build the Random Forest pipeline"""
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
        
        # Random Forest model
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
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


def train_random_forest():
    """Train Random Forest model"""
    trainer = RandomForestTrainer(n_estimators=100, max_depth=15, min_samples_split=10)
    metrics = trainer.run()
    return trainer, metrics


if __name__ == '__main__':
    train_random_forest()

