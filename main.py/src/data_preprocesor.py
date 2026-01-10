from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src import data_definition
from src.data_definition import DataDefinition


class DataPreprocessor:

    def preprocess(self,data):
        numeric_transformer = Pipeline(
        steps = [('scaler',StandardScaler())])

        categorical_transformer = Pipeline(
        steps = [('encoder',OneHotEncoder())])


        preprocessor = ColumnTransformer(transformers=
                                         [
                                             ('num',numeric_transformer,DataDefinition.numeric_cols),
                                             ('cat',categorical_transformer,DataDefinition.categorical_cols)
                                         ]
                        )
        data_transformed = preprocessor.fit_transform(data)
        print(data_transformed)
        return data_transformed


    if __name__ == '__main__':
       preprocess()