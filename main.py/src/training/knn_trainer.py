from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data_definition import DataDefinition
from src.models.base_model_trainer import BaseModelTrainer


class KNNTrainer(BaseModelTrainer):
    """K-Nearest Neighbors Classifier trainer"""
    
    def __init__(self, n_neighbors=5, weights='distance', metric='minkowski'):
        super().__init__(model_name='knn_model')
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        
    def build_pipeline(self):
        """Build the KNN pipeline"""
        # Use RobustScaler for numeric features - KNN is sensitive to feature scaling
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
        
        # KNN model
        # weights='distance' gives closer neighbors more weight
        # metric='minkowski' with p=2 is equivalent to Euclidean distance
        model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
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


def train_knn():
    """Train K-Nearest Neighbors model"""
    trainer = KNNTrainer(n_neighbors=5, weights='distance')
    metrics = trainer.run()
    return trainer, metrics


if __name__ == '__main__':
    train_knn()

