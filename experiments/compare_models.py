"""
Experiment: Compare Models
Compares different models using various metrics and visualizations.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from preprocessing import load_breast_cancer_data, load_wine_data, filter_important_features_breast_cancer, filter_important_features_wine
from models import LinearRegression, LogisticRegression, MulticlassLogisticRegression
from utils import (compute_feature_importance, compute_feature_importance_multiclass,
                   compare_binary_models, compare_multiclass_models,
                   plot_roc_curve_comparison, plot_coefficient_comparison,
                   plot_multiclass_heatmap)


def compare_breast_cancer_models(data_path):
    """Compare models on breast cancer dataset."""
    print("=" * 70)
    print("BREAST CANCER MODEL COMPARISON")
    print("=" * 70)
    
    # Load and prepare data
    data = load_breast_cancer_data(data_path)
    feature_importance = compute_feature_importance(data['X_train'], data['y_train'])
    filtered_data = filter_important_features_breast_cancer(data, feature_importance)
    
    # Train models
    linear_model = LinearRegression(is_multiclass=False)
    linear_model.fit(filtered_data['X_train'], filtered_data['y_train'])
    
    logistic_model = LogisticRegression(lr=0.005, n_iter=1000)
    logistic_model.fit(filtered_data['X_train'], filtered_data['y_train'], verify_grad=False)
    
    # Compare models
    models_dict = {
        'Linear Regression': linear_model,
        'Logistic Regression': logistic_model
    }
    
    results = compare_binary_models(models_dict, filtered_data['X_test'], filtered_data['y_test'])
    
    print("\nTest Set Results:")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUROC: {result['auroc']:.4f}")
        print(f"  Confusion Matrix:\n{result['confusion_matrix']}\n")
    
    # Plot ROC curves
    plot_roc_curve_comparison(
        models_dict,
        filtered_data['X_test'],
        filtered_data['y_test'],
        labels=['Linear Regression', 'Logistic Regression']
    )
    
    # Plot coefficient comparison
    coefficients_dict = {
        'Linear Regression': linear_model.W.ravel(),
        'Logistic Regression': logistic_model.w
    }
    
    plot_coefficient_comparison(
        filtered_data['feature_names'],
        coefficients_dict,
        title="Coefficient Comparison - Breast Cancer Dataset"
    )
    
    return models_dict, filtered_data


def compare_wine_models(data_path):
    """Compare models on wine dataset."""
    print("\n" + "=" * 70)
    print("WINE MODEL COMPARISON")
    print("=" * 70)
    
    # Load and prepare data
    data = load_wine_data(data_path)
    n_classes = 3
    feature_importance = compute_feature_importance_multiclass(
        data['X_train'], data['y_train'], n_classes
    )
    filtered_data = filter_important_features_wine(data, feature_importance)
    
    # Train models
    linear_model = LinearRegression(is_multiclass=True)
    linear_model.fit(filtered_data['X_train'], filtered_data['y_train'])
    
    logistic_model = MulticlassLogisticRegression(lr=0.005, n_iter=1000)
    logistic_model.fit(filtered_data['X_train'], filtered_data['y_train'], verify_grad=False)
    
    # Compare models
    models_dict = {
        'Multiple Linear Regression': linear_model,
        'Multiclass Logistic Regression': logistic_model
    }
    
    results = compare_multiclass_models(models_dict, filtered_data['X_test'], filtered_data['y_test'])
    
    print("\nTest Set Results:")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Confusion Matrix:\n{result['confusion_matrix']}\n")
    
    # Compute feature importance for heatmap
    feature_importance_full = compute_feature_importance_multiclass(
        filtered_data['X_train'], filtered_data['y_train'], n_classes
    )
    
    # Plot coefficient heatmap
    class_labels = [f"Class {i+1}" for i in range(n_classes)]
    plot_multiclass_heatmap(
        feature_importance_full,
        logistic_model.W,
        filtered_data['feature_names'],
        class_labels
    )
    
    return models_dict, filtered_data


def main():
    """Main function to compare all models."""
    # Set data paths
    breast_cancer_path = "../data/wdbc.data"
    wine_path = "../data/wine.data"
    
    # Check if data files exist
    if not os.path.exists(breast_cancer_path):
        print(f"Warning: Breast cancer data file not found at {breast_cancer_path}")
        breast_cancer_path = None
    
    if not os.path.exists(wine_path):
        print(f"Warning: Wine data file not found at {wine_path}")
        wine_path = None
    
    # Compare models
    if breast_cancer_path:
        bc_models, bc_data = compare_breast_cancer_models(breast_cancer_path)
    
    if wine_path:
        wine_models, wine_data = compare_wine_models(wine_path)


if __name__ == "__main__":
    main()

