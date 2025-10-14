"""
Model Evaluation Utilities
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def evaluate_model(y_true, y_pred, is_multiclass=False):
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        is_multiclass (bool): Whether this is multiclass classification
        
    Returns:
        tuple: (accuracy, confusion_matrix)
    """
    acc = accuracy_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    return acc, conf_mat


def evaluate_binary_model(model, X_test, y_test):
    """
    Evaluate a binary classification model.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary with accuracy, AUROC, confusion matrix
    """
    y_pred = model.predict(X_test)
    
    # Handle probability prediction based on model type
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)
    else:
        # For linear regression, use sigmoid transformation
        y_pred_prob = 1 / (1 + np.exp(-(X_test @ model.W)))
        if y_pred_prob.ndim > 1:
            y_pred_prob = y_pred_prob.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_prob)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': acc,
        'auroc': auroc,
        'confusion_matrix': conf_mat
    }


def evaluate_multiclass_model(model, X_test, y_test):
    """
    Evaluate a multiclass classification model.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary with accuracy and confusion matrix
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': acc,
        'confusion_matrix': conf_mat
    }


def compare_binary_models(models_dict, X_test, y_test):
    """
    Compare multiple binary classification models.
    
    Args:
        models_dict (dict): Dictionary of model_name: model pairs
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary with results for each model
    """
    results = {}
    
    for name, model in models_dict.items():
        results[name] = evaluate_binary_model(model, X_test, y_test)
    
    return results


def compare_multiclass_models(models_dict, X_test, y_test):
    """
    Compare multiple multiclass classification models.
    
    Args:
        models_dict (dict): Dictionary of model_name: model pairs
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary with results for each model
    """
    results = {}
    
    for name, model in models_dict.items():
        results[name] = evaluate_multiclass_model(model, X_test, y_test)
    
    return results


def compute_roc_curve(model, X_test, y_test):
    """
    Compute ROC curve data for a binary classifier.
    
    Args:
        model: Trained binary classification model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (fpr, tpr, thresholds, auroc)
    """
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)
    else:
        # For linear regression, use sigmoid transformation
        y_pred_prob = 1 / (1 + np.exp(-(X_test @ model.W)))
        if y_pred_prob.ndim > 1:
            y_pred_prob = y_pred_prob.ravel()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auroc = roc_auc_score(y_test, y_pred_prob)
    
    return fpr, tpr, thresholds, auroc


def print_evaluation_results(results, dataset_name):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results (dict): Results dictionary from evaluate_*_model
        dataset_name (str): Name of the dataset
    """
    print(f"\n{dataset_name} Results:")
    print("-" * 50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    if 'auroc' in results:
        print(f"AUROC: {results['auroc']:.4f}")
    
    print("Confusion Matrix:")
    print(results['confusion_matrix'])

