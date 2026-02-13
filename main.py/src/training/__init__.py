from src.training.base_model_trainer import BaseModelTrainer
from src.training.logistic_regression_trainer import LogisticRegressionTrainer, train_logistic_regression
from src.training.decision_tree_trainer import DecisionTreeTrainer, train_decision_tree
from src.training.knn_trainer import KNNTrainer, train_knn
from src.training.naive_bayes_trainer import NaiveBayesTrainer, train_naive_bayes
from src.training.random_forest_trainer import RandomForestTrainer, train_random_forest

# Try to import XGBoost, but make it optional
try:
    from src.training.xgboost_trainer import XGBoostTrainer, train_xgboost
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"Warning: XGBoost not available: {e}")
    print("Install with: brew install libomp (Mac) or pip install xgboost")
    XGBoostTrainer = None
    train_xgboost = None
    XGBOOST_AVAILABLE = False

__all__ = [
    'BaseModelTrainer',
    'LogisticRegressionTrainer',
    'DecisionTreeTrainer',
    'KNNTrainer',
    'NaiveBayesTrainer',
    'RandomForestTrainer',
    'XGBoostTrainer',
    'train_logistic_regression',
    'train_decision_tree',
    'train_knn',
    'train_naive_bayes',
    'train_random_forest',
    'train_xgboost',
    'XGBOOST_AVAILABLE'
]

