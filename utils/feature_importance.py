"""
Feature Importance Computation
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def compute_feature_importance(X, y):
    """
    Compute feature importance using simple regression.
    Essentially computing the covariance between the feature
    and the target. If a feature has a large absolute weight,
    it means that it tends to change a lot when the target changes.

    Args:
        X: standardized features (N x D)
        y: standardized target (N x 1)
        
    Returns:
        feature importance coefficients (D x 1)
    """
    # Standardize y (even though it's binary)
    y_standardized = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    # Compute w = X^T y / N
    N = X.shape[0]
    w = (X.T @ y_standardized) / N  # Using the matrix computation method

    return w


def compute_feature_importance_multiclass(X, y, n_classes):
    """
    Compute feature importance using simple regression for multi-class classification.

    Args:
        X: standardized features (N x D)
        y: class labels (N,)
        n_classes: number of classes
        
    Returns:
        feature importance coefficients (D x C)
    """
    # One-hot encode the target
    onehot = OneHotEncoder(sparse_output=False)
    y_onehot = onehot.fit_transform(y.reshape(-1, 1))

    # Standardize one-hot encoded targets
    y_standardized = StandardScaler().fit_transform(y_onehot)

    # Compute W = X^T Y / N for all classes at once
    N = X.shape[0]
    W = (X.T @ y_standardized) / N

    return W


def rank_features_binary(feature_names, feature_importance, ascending=False):
    """
    Rank features by importance for binary classification.
    
    Args:
        feature_names (list): List of feature names
        feature_importance (np.ndarray): Feature importance scores
        ascending (bool): Sort order (False for descending)
        
    Returns:
        pd.DataFrame: Sorted dataframe with features and importance
    """
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    return feature_importance_df.sort_values('Importance', ascending=ascending)


def rank_features_multiclass(feature_names, feature_importance, class_idx=None):
    """
    Rank features by importance for multiclass classification.
    
    Args:
        feature_names (list): List of feature names
        feature_importance (np.ndarray): Feature importance scores (D x C)
        class_idx (int): Specific class to rank for, if None ranks by max across classes
        
    Returns:
        pd.DataFrame: Sorted dataframe with features and importance
    """
    if class_idx is not None:
        # Rank for specific class
        importance_scores = feature_importance[:, class_idx]
    else:
        # Rank by maximum absolute importance across all classes
        importance_scores = np.max(np.abs(feature_importance), axis=1)
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    
    return feature_importance_df.sort_values('Importance', ascending=False)


def select_important_features(feature_importance, threshold=None):
    """
    Select features above a threshold.
    
    Args:
        feature_importance (np.ndarray): Feature importance scores
        threshold (float): Threshold value, if None uses mean
        
    Returns:
        np.ndarray: Boolean mask of important features
    """
    if threshold is None:
        threshold = np.mean(np.abs(feature_importance))
    
    return np.abs(feature_importance) > threshold

