from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data_definition import DataDefinition
from src.training.base_model_trainer import BaseModelTrainer


class DecisionTreeTrainer(BaseModelTrainer):
    """Decision Tree Classifier trainer"""
    
    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10):
        super().__init__(model_name='decision_tree_model')
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def build_pipeline(self):
        """Build the Decision Tree pipeline"""
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
        
        # Decision Tree model with parameters to prevent overfitting
        model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            criterion='gini',  # or 'entropy'
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Create pipeline
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )
        
        return pipeline


def train_decision_tree():
    """Train Decision Tree model"""
    trainer = DecisionTreeTrainer(max_depth=10, min_samples_split=20, min_samples_leaf=10)
    metrics = trainer.run()
    return trainer, metrics


if __name__ == '__main__':
    train_decision_tree()

