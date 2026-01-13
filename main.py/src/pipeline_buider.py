from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.data_definition import DataDefinition


class PipelineBuilder:

    def build_pipeline(self):
        # Use RobustScaler instead of StandardScaler - it's more resistant to outliers
        numeric_transformer = Pipeline(
            steps=[('scaler', RobustScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, DataDefinition.numeric_cols),
                ('cat', categorical_transformer, DataDefinition.categorical_cols)
            ]
        )

        # Add stronger regularization and use a more stable solver
        # saga solver is more robust for large datasets
        model = LogisticRegression(
            max_iter=2000,
            C=0.1,  # Stronger regularization
            solver='saga',  # More stable solver
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

        clf = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )

        return clf
