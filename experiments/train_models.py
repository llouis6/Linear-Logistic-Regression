"""
Experiment: Train and Evaluate Models
Trains linear and logistic regression models on both datasets.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from preprocessing import load_breast_cancer_data, load_wine_data, filter_important_features_breast_cancer, filter_important_features_wine
from models import LinearRegression, LogisticRegression, MulticlassLogisticRegression
from utils import (compute_feature_importance, compute_feature_importance_multiclass,
                   evaluate_binary_model, evaluate_multiclass_model, 
                   print_evaluation_results, plot_loss_history)
from sklearn.preprocessing import OneHotEncoder


def train_breast_cancer_models(data_path):
    """Train models on breast cancer dataset."""
    print("=" * 70)
    print("BREAST CANCER MODEL TRAINING")
    print("=" * 70)
    
    # Load and filter data
    data = load_breast_cancer_data(data_path)
    feature_importance = compute_feature_importance(data['X_train'], data['y_train'])
    filtered_data = filter_important_features_breast_cancer(data, feature_importance)
    
    # Train Linear Regression
    print("\n1. Training Linear Regression...")
    linear_model = LinearRegression(is_multiclass=False)
    linear_model.fit(filtered_data['X_train'], filtered_data['y_train'])
    
    # Train Logistic Regression
    print("\n2. Training Logistic Regression...")
    logistic_model = LogisticRegression(lr=0.005, n_iter=1000)
    logistic_model.fit(filtered_data['X_train'], filtered_data['y_train'], verify_grad=True)
    
    # Plot loss history
    plot_loss_history(
        logistic_model.loss_history,
        title="Cross Entropy Loss vs. Iterations - Breast Cancer Dataset"
    )
    
    # Evaluate models
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    for split_name in ['train', 'val', 'test']:
        X = filtered_data[f'X_{split_name}']
        y = filtered_data[f'y_{split_name}']
        
        print(f"\n{split_name.upper()} SET:")
        print("-" * 50)
        
        # Linear Regression
        linear_results = evaluate_binary_model(linear_model, X, y)
        print(f"Linear Regression - Accuracy: {linear_results['accuracy']:.4f}, "
              f"AUROC: {linear_results['auroc']:.4f}")
        
        # Logistic Regression
        logistic_results = evaluate_binary_model(logistic_model, X, y)
        print(f"Logistic Regression - Accuracy: {logistic_results['accuracy']:.4f}, "
              f"AUROC: {logistic_results['auroc']:.4f}")
    
    return linear_model, logistic_model, filtered_data


def train_wine_models(data_path):
    """Train models on wine dataset."""
    print("\n" + "=" * 70)
    print("WINE MODEL TRAINING")
    print("=" * 70)
    
    # Load and filter data
    data = load_wine_data(data_path)
    n_classes = 3
    feature_importance = compute_feature_importance_multiclass(
        data['X_train'], data['y_train'], n_classes
    )
    filtered_data = filter_important_features_wine(data, feature_importance)
    
    # Train Multiple Linear Regression
    print("\n1. Training Multiple Linear Regression...")
    linear_model = LinearRegression(is_multiclass=True)
    linear_model.fit(filtered_data['X_train'], filtered_data['y_train'])
    
    # Train Multivariate Linear Regression
    print("\n2. Training Multivariate Linear Regression...")
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = onehot_encoder.fit_transform(filtered_data['y_train'].reshape(-1, 1))
    
    multivariate_model = LinearRegression(is_multiclass=False, is_multivariate=True)
    multivariate_model.fit(filtered_data['X_train'], y_train_onehot)
    
    # Train Multiclass Logistic Regression
    print("\n3. Training Multiclass Logistic Regression...")
    logistic_model = MulticlassLogisticRegression(lr=0.005, n_iter=1000)
    logistic_model.fit(filtered_data['X_train'], filtered_data['y_train'], verify_grad=True)
    
    # Plot loss history
    plot_loss_history(
        logistic_model.loss_history,
        title="Cross Entropy Loss vs. Iterations - Wine Dataset"
    )
    
    # Evaluate models
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    for split_name in ['train', 'val', 'test']:
        X = filtered_data[f'X_{split_name}']
        y = filtered_data[f'y_{split_name}']
        
        print(f"\n{split_name.upper()} SET:")
        print("-" * 50)
        
        # Multiple Linear Regression
        linear_results = evaluate_multiclass_model(linear_model, X, y)
        print(f"Multiple Linear Regression - Accuracy: {linear_results['accuracy']:.4f}")
        
        # Multivariate Linear Regression
        y_pred_multi = multivariate_model.predict(X)
        y_pred_multi = np.argmax(y_pred_multi, axis=1)
        from sklearn.metrics import accuracy_score
        acc_multi = accuracy_score(y, y_pred_multi)
        print(f"Multivariate Linear Regression - Accuracy: {acc_multi:.4f}")
        
        # Multiclass Logistic Regression
        logistic_results = evaluate_multiclass_model(logistic_model, X, y)
        print(f"Multiclass Logistic Regression - Accuracy: {logistic_results['accuracy']:.4f}")
    
    return linear_model, multivariate_model, logistic_model, filtered_data


def main():
    """Main function to train and evaluate all models."""
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
    
    # Train models
    if breast_cancer_path:
        bc_linear, bc_logistic, bc_data = train_breast_cancer_models(breast_cancer_path)
    
    if wine_path:
        wine_linear, wine_multi, wine_logistic, wine_data = train_wine_models(wine_path)


if __name__ == "__main__":
    main()

