"""
Breast Cancer Dataset Preprocessing
Supports both file-based and UCI repository loading.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_breast_cancer_data(file_path=None, use_uci=True):
    """
    Load and preprocess the breast cancer dataset.
    
    Args:
        file_path (str): Path to the wdbc.data file (optional if use_uci=True)
        use_uci (bool): If True, fetch from UCI repository instead of file
        
    Returns:
        dict: Dictionary containing processed data splits
    """
    if use_uci:
        # Load from UCI repository
        from .uci_datasets import load_breast_cancer_uci
        X_df, y_series, metadata, variables = load_breast_cancer_uci()
        
        # Convert to numpy arrays
        X_breast_cancer = X_df.values
        
        # Map diagnosis to 0/1 (assumes 'M'/'B' or similar)
        if y_series.dtype == 'object':
            y_breast_cancer = y_series.map({"M": 1, "B": 0}).values
        else:
            y_breast_cancer = y_series.values
        
        feature_names = X_df.columns.tolist()
        
    else:
        # Load from file (original implementation)
        if file_path is None:
            raise ValueError("file_path must be provided when use_uci=False")
            
        # Column/feature names based on dataset documentation
        breast_cancer_columns = ["ID", "Diagnosis"] + [
            "Radius_mean", "Texture_mean", "Perimeter_mean",
            "Area_mean", "Smoothness_mean", "Compactness_mean", "Concavity_mean",
            "Concave points_mean", "Symmetry_mean", "Fractal_dimension_mean",
            "Radius_se", "Texture_se", "Perimeter_se", "Area_se", "Smoothness_se",
            "Compactness_se", "Concavity_se", "Concave points_se", "Symmetry_se",
            "Fractal_dimension_se", "Radius_worst", "Texture_worst", "Perimeter_worst",
            "Area_worst", "Smoothness_worst", "Compactness_worst", "Concavity_worst",
            "Concave points_worst", "Symmetry_worst", "Fractal_dimension_worst"
        ]

        # Load the dataset
        breast_cancer_df = pd.read_csv(file_path, header=None, names=breast_cancer_columns)

        # Drop the ID column
        breast_cancer_df.drop(columns=["ID"], inplace=True)

        # Convert Diagnosis: 'M' -> 1, 'B' -> 0
        breast_cancer_df["Diagnosis"] = breast_cancer_df["Diagnosis"].map({"M": 1, "B": 0})

        # Separate features and target
        X_breast_cancer = breast_cancer_df.drop(columns=["Diagnosis"]).values
        y_breast_cancer = breast_cancer_df["Diagnosis"].values
        feature_names = breast_cancer_df.drop(columns=["Diagnosis"]).columns.tolist()

    # Standardize the features
    scaler_breast_cancer = StandardScaler()
    X_breast_cancer_scaled = scaler_breast_cancer.fit_transform(X_breast_cancer)

    # Train test validation split
    X_train_bc, X_temp_bc, y_train_bc, y_temp_bc = train_test_split(
        X_breast_cancer_scaled, y_breast_cancer, 
        test_size=0.2, random_state=42, stratify=y_breast_cancer
    )
    X_val_bc, X_test_bc, y_val_bc, y_test_bc = train_test_split(
        X_temp_bc, y_temp_bc, 
        test_size=0.5, random_state=42, stratify=y_temp_bc
    )

    return {
        'X_train': X_train_bc,
        'X_val': X_val_bc,
        'X_test': X_test_bc,
        'y_train': y_train_bc,
        'y_val': y_val_bc,
        'y_test': y_test_bc,
        'feature_names': feature_names,
        'scaler': scaler_breast_cancer
    }


def filter_important_features_breast_cancer(data, feature_importance, mean_threshold=None):
    """
    Filter features based on importance scores.
    
    Args:
        data (dict): Data dictionary from load_breast_cancer_data
        feature_importance (np.ndarray): Feature importance scores
        mean_threshold (float): Threshold value, if None uses mean of importance
        
    Returns:
        dict: Filtered data dictionary with important features only
    """
    if mean_threshold is None:
        mean_threshold = np.mean(np.abs(feature_importance))

    # Select important features based on mean threshold
    important_features_mask = np.abs(feature_importance) > mean_threshold
    important_feature_names = [name for i, name in enumerate(data['feature_names']) 
                               if important_features_mask[i]]

    # Create filtered datasets with only important features
    X_train_filtered = data['X_train'][:, important_features_mask]
    X_val_filtered = data['X_val'][:, important_features_mask]
    X_test_filtered = data['X_test'][:, important_features_mask]

    return {
        'X_train': X_train_filtered,
        'X_val': X_val_filtered,
        'X_test': X_test_filtered,
        'y_train': data['y_train'],
        'y_val': data['y_val'],
        'y_test': data['y_test'],
        'feature_names': important_feature_names,
        'importance_mask': important_features_mask
    }
