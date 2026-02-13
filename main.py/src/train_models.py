"""
Script to train and compare different classification models
"""
import pandas as pd
from src.training import (
    train_logistic_regression,
    train_decision_tree,
    train_knn,
    train_naive_bayes,
    train_random_forest,
    train_xgboost,
    XGBOOST_AVAILABLE
)


def train_all_models():
    """Train all models and compare results"""
    results = {}

    print("="*70)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*70)
    lr_trainer, lr_metrics = train_logistic_regression()
    results['Logistic Regression'] = lr_metrics

    print("\n\n")
    print("="*70)
    print("TRAINING DECISION TREE MODEL")
    print("="*70)
    dt_trainer, dt_metrics = train_decision_tree()
    results['Decision Tree'] = dt_metrics

    print("\n\n")
    print("="*70)
    print("TRAINING K-NEAREST NEIGHBORS MODEL")
    print("="*70)
    knn_trainer, knn_metrics = train_knn()
    results['K-Nearest Neighbors'] = knn_metrics

    print("\n\n")
    print("="*70)
    print("TRAINING NAIVE BAYES MODEL")
    print("="*70)
    nb_trainer, nb_metrics = train_naive_bayes()
    results['Naive Bayes'] = nb_metrics

    print("\n\n")
    print("="*70)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*70)
    rf_trainer, rf_metrics = train_random_forest()
    results['Random Forest'] = rf_metrics

    if XGBOOST_AVAILABLE:
        print("\n\n")
        print("="*70)
        print("TRAINING XGBOOST MODEL")
        print("="*70)
        xgb_trainer, xgb_metrics = train_xgboost()
        results['XGBoost'] = xgb_metrics
    else:
        print("\n\n")
        print("="*70)
        print("SKIPPING XGBOOST MODEL (not available)")
        print("="*70)

    # Compare results
    print("\n\n")
    print("="*70)
    print("MODEL COMPARISON - ALL MODELS")
    print("="*70)

    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    print(comparison_df)

    # Find best model for each metric
    print("\n" + "="*70)
    print("BEST MODEL FOR EACH METRIC")
    print("="*70)
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        print(f"{metric:30s}: {best_model:25s} ({best_score:.4f})")

    # Overall best model (based on F1 score)
    print("\n" + "="*70)
    best_overall = comparison_df['f1_score'].idxmax()
    print(f"OVERALL BEST MODEL (by F1 Score): {best_overall}")
    print("="*70)

    return results


if __name__ == '__main__':
    train_all_models()

