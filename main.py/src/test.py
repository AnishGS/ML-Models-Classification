from src import data_reader
from src.data_preprocesor import DataPreprocessor


def test_import_data():
    reader = data_reader.DataReader('../data/loan_data.csv')
    data = reader.load_data()
    preprocessor = DataPreprocessor()
    preprocessor.preprocess(data)

if __name__ == '__main__':
    test_import_data()