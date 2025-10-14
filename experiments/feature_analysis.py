"""
Experiment: Feature Importance Analysis
Analyzes and visualizes feature importance for both datasets.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from preprocessing import load_breast_cancer_data, load_wine_data, filter_important_features_breast_cancer, filter_important_features_wine
from utils import compute_feature_importance, compute_feature_importance_multiclass, plot_feature_importance_binary, plot_feature_importance_multiclass


def analyze_breast_cancer_features(data_path):
    """Analyze feature importance for breast cancer dataset."""
    print("=" * 70)
    print("BREAST CANCER FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    # Load data
    data = load_breast_cancer_data(data_path)
    
    # Compute feature importance
    feature_importance = compute_feature_importance(data['X_train'], data['y_train'])
    
    # Print feature rankings
    print("\nFeature Rankings (by absolute importance):")
    print("-" * 50)
    from utils import rank_features_binary
    ranked_features = rank_features_binary(data['feature_names'], np.abs(feature_importance), ascending=False)
    print(ranked_features)
    
    # Plot feature importance
    plot_feature_importance_binary(
        data['feature_names'], 
        feature_importance,
        title="Feature Importance - Breast Cancer Dataset"
    )
    
    # Filter important features
    filtered_data = filter_important_features_breast_cancer(data, feature_importance)
    
    print(f"\nSelected {len(filtered_data['feature_names'])} important features:")
    print(filtered_data['feature_names'])
    
    return data, filtered_data, feature_importance


def analyze_wine_features(data_path):
    """Analyze feature importance for wine dataset."""
    print("\n" + "=" * 70)
    print("WINE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    # Load data
    data = load_wine_data(data_path)
    
    # Compute feature importance
    n_classes = 3
    feature_importance = compute_feature_importance_multiclass(
        data['X_train'], data['y_train'], n_classes
    )
    
    # Print feature rankings
    print("\nFeature Rankings (by maximum absolute importance across classes):")
    print("-" * 50)
    from utils import rank_features_multiclass
    ranked_features = rank_features_multiclass(data['feature_names'], feature_importance)
    print(ranked_features)
    
    # Plot feature importance by class
    plot_feature_importance_multiclass(
        data['feature_names'],
        feature_importance,
        n_classes=n_classes,
        title_prefix="Feature Importance - Wine Dataset"
    )
    
    # Filter important features
    filtered_data = filter_important_features_wine(data, feature_importance)
    
    print(f"\nSelected {len(filtered_data['feature_names'])} important features:")
    print(filtered_data['feature_names'])
    
    return data, filtered_data, feature_importance


def main():
    """Main function to run feature analysis experiments."""
    # Set data paths (update these to your actual data paths)
    breast_cancer_path = "../data/wdbc.data"
    wine_path = "../data/wine.data"
    
    # Check if data files exist
    if not os.path.exists(breast_cancer_path):
        print(f"Warning: Breast cancer data file not found at {breast_cancer_path}")
        print("Please update the path in this script or place the data file in the correct location.")
        breast_cancer_path = None
    
    if not os.path.exists(wine_path):
        print(f"Warning: Wine data file not found at {wine_path}")
        print("Please update the path in this script or place the data file in the correct location.")
        wine_path = None
    
    # Run analyses
    if breast_cancer_path:
        bc_data, bc_filtered, bc_importance = analyze_breast_cancer_features(breast_cancer_path)
    
    if wine_path:
        wine_data, wine_filtered, wine_importance = analyze_wine_features(wine_path)


if __name__ == "__main__":
    main()

