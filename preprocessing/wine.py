"""
Wine Dataset Preprocessing
Supports both file-based and UCI repository loading.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_wine_data(file_path=None, use_uci=True):
    """
    Load and preprocess the wine dataset.
    
    Args:
        file_path (str): Path to the wine.data file (optional if use_uci=True)
        use_uci (bool): If True, fetch from UCI repository instead of file
        
    Returns:
        dict: Dictionary containing processed data splits
    """
    if use_uci:
        # Load from UCI repository
        from .uci_datasets import load_wine_uci
        X_df, y_series, metadata, variables = load_wine_uci()
        
        # Convert to numpy arrays
        X_wine = X_df.values
        y_wine = y_series.values
        
        # Convert class labels to 0-indexed (if needed)
        if y_wine.min() == 1:
            y_wine = y_wine - 1
        
        feature_names = X_df.columns.tolist()
        
    else:
        # Load from file (original implementation)
        if file_path is None:
            raise ValueError("file_path must be provided when use_uci=False")
            
        # Wine feature names based on dataset documentation
        wine_features = [
            "Class",
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline"
        ]

        # Load the dataset
        wine_df = pd.read_csv(file_path, header=None, names=wine_features)

        y_wine = wine_df.iloc[:, 0].values
        X_wine = wine_df.iloc[:, 1:].values

        # Convert class labels from (1,2,3) to (0,1,2)
        y_wine = y_wine - 1  # Useful for zero-based indexing
        
        feature_names = wine_features[1:]  # Exclude 'Class'

    # Standardize the features
    scaler_wine = StandardScaler()
    X_wine_scaled = scaler_wine.fit_transform(X_wine)

    # Train Test Validation split
    X_train_wine, X_temp_wine, y_train_wine, y_temp_wine = train_test_split(
        X_wine_scaled, y_wine, 
        test_size=0.2, random_state=42, stratify=y_wine
    )
    X_val_wine, X_test_wine, y_val_wine, y_test_wine = train_test_split(
        X_temp_wine, y_temp_wine, 
        test_size=0.5, random_state=42, stratify=y_temp_wine
    )

    return {
        'X_train': X_train_wine,
        'X_val': X_val_wine,
        'X_test': X_test_wine,
        'y_train': y_train_wine,
        'y_val': y_val_wine,
        'y_test': y_test_wine,
        'feature_names': feature_names,
        'scaler': scaler_wine
    }


def filter_important_features_wine(data, feature_importance, mean_threshold=None):
    """
    Filter features based on importance scores for multiclass classification.
    
    Args:
        data (dict): Data dictionary from load_wine_data
        feature_importance (np.ndarray): Feature importance scores (D x C)
        mean_threshold (float): Threshold value, if None uses mean of max importance
        
    Returns:
        dict: Filtered data dictionary with important features only
    """
    # For multi-class, take the maximum absolute importance across classes for each feature
    max_importance_per_feature = np.max(np.abs(feature_importance), axis=1)
    
    if mean_threshold is None:
        mean_threshold = np.mean(max_importance_per_feature)

    # Select important features based on mean threshold
    important_features_mask = max_importance_per_feature > mean_threshold
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
