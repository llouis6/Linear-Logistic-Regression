"""
Main script to run all experiments.
"""
import os
import sys
from config import BREAST_CANCER_DATA_PATH, WINE_DATA_PATH

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def check_data_availability():
    """Check if ucimlrepo is available for auto-downloading datasets."""
    try:
        import ucimlrepo
        print("Using UCI repository for automatic data download")
        return True, True  # Both datasets available via UCI
    except ImportError:
        print("WARNING: ucimlrepo not installed. Install with: pip install ucimlrepo")
        print("   Or install all dependencies: pip install -r requirements.txt")
        return False, False


def run_feature_analysis(bc_exists, wine_exists):
    """Run feature importance analysis."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    from preprocessing import load_breast_cancer_data, load_wine_data, filter_important_features_breast_cancer, filter_important_features_wine
    from utils import compute_feature_importance, compute_feature_importance_multiclass, plot_feature_importance_binary, plot_feature_importance_multiclass
    import numpy as np
    
    if bc_exists:
        print("\n--- Breast Cancer Dataset ---")
        print("Loading data from UCI repository...")
        data = load_breast_cancer_data(use_uci=True)
        feature_importance = compute_feature_importance(data['X_train'], data['y_train'])
        
        from utils import rank_features_binary
        ranked_features = rank_features_binary(data['feature_names'], np.abs(feature_importance), ascending=False)
        print("\nTop 10 Features:")
        print(ranked_features.head(10))
        
        # Uncomment to show plots
        # plot_feature_importance_binary(data['feature_names'], feature_importance, 
        #                                title="Feature Importance - Breast Cancer")
    
    if wine_exists:
        print("\n--- Wine Dataset ---")
        print("Loading data from UCI repository...")
        data = load_wine_data(use_uci=True)
        n_classes = 3
        feature_importance = compute_feature_importance_multiclass(data['X_train'], data['y_train'], n_classes)
        
        from utils import rank_features_multiclass
        ranked_features = rank_features_multiclass(data['feature_names'], feature_importance)
        print("\nTop Features (by max importance across classes):")
        print(ranked_features.head(10))
        
        # Uncomment to show plots
        # plot_feature_importance_multiclass(data['feature_names'], feature_importance, n_classes=n_classes,
        #                                    title_prefix="Feature Importance - Wine")


def run_model_training(bc_exists, wine_exists):
    """Run model training experiments."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    from preprocessing import load_breast_cancer_data, load_wine_data, filter_important_features_breast_cancer, filter_important_features_wine
    from models import LinearRegression, LogisticRegression, MulticlassLogisticRegression
    from utils import compute_feature_importance, compute_feature_importance_multiclass, evaluate_binary_model, evaluate_multiclass_model
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    
    if bc_exists:
        print("\n--- Breast Cancer Dataset ---")
        print("Loading data from UCI repository...")
        data = load_breast_cancer_data(use_uci=True)
        feature_importance = compute_feature_importance(data['X_train'], data['y_train'])
        filtered_data = filter_important_features_breast_cancer(data, feature_importance)
        
        # Train models
        linear_model = LinearRegression(is_multiclass=False)
        linear_model.fit(filtered_data['X_train'], filtered_data['y_train'])
        
        logistic_model = LogisticRegression(lr=0.005, n_iter=1000)
        logistic_model.fit(filtered_data['X_train'], filtered_data['y_train'], verify_grad=False)
        
        # Evaluate
        print("\nTest Set Results:")
        linear_results = evaluate_binary_model(linear_model, filtered_data['X_test'], filtered_data['y_test'])
        print(f"Linear Regression - Accuracy: {linear_results['accuracy']:.4f}, AUROC: {linear_results['auroc']:.4f}")
        
        logistic_results = evaluate_binary_model(logistic_model, filtered_data['X_test'], filtered_data['y_test'])
        print(f"Logistic Regression - Accuracy: {logistic_results['accuracy']:.4f}, AUROC: {logistic_results['auroc']:.4f}")
    
    if wine_exists:
        print("\n--- Wine Dataset ---")
        print("Loading data from UCI repository...")
        data = load_wine_data(use_uci=True)
        n_classes = 3
        feature_importance = compute_feature_importance_multiclass(data['X_train'], data['y_train'], n_classes)
        filtered_data = filter_important_features_wine(data, feature_importance)
        
        # Train models
        linear_model = LinearRegression(is_multiclass=True)
        linear_model.fit(filtered_data['X_train'], filtered_data['y_train'])
        
        logistic_model = MulticlassLogisticRegression(lr=0.005, n_iter=1000)
        logistic_model.fit(filtered_data['X_train'], filtered_data['y_train'], verify_grad=False)
        
        # Evaluate
        print("\nTest Set Results:")
        linear_results = evaluate_multiclass_model(linear_model, filtered_data['X_test'], filtered_data['y_test'])
        print(f"Multiple Linear Regression - Accuracy: {linear_results['accuracy']:.4f}")
        
        logistic_results = evaluate_multiclass_model(logistic_model, filtered_data['X_test'], filtered_data['y_test'])
        print(f"Multiclass Logistic Regression - Accuracy: {logistic_results['accuracy']:.4f}")


def run_model_comparison(bc_exists, wine_exists):
    """Run model comparison experiments."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: MODEL COMPARISON")
    print("=" * 80)
    
    from preprocessing import load_breast_cancer_data, load_wine_data, filter_important_features_breast_cancer, filter_important_features_wine
    from models import LinearRegression, LogisticRegression, MulticlassLogisticRegression
    from utils import compute_feature_importance, compute_feature_importance_multiclass, compare_binary_models, compare_multiclass_models
    
    if bc_exists:
        print("\n--- Breast Cancer Dataset ---")
        print("Loading data from UCI repository...")
        data = load_breast_cancer_data(use_uci=True)
        feature_importance = compute_feature_importance(data['X_train'], data['y_train'])
        filtered_data = filter_important_features_breast_cancer(data, feature_importance)
        
        # Train models
        linear_model = LinearRegression(is_multiclass=False)
        linear_model.fit(filtered_data['X_train'], filtered_data['y_train'])
        
        logistic_model = LogisticRegression(lr=0.005, n_iter=1000)
        logistic_model.fit(filtered_data['X_train'], filtered_data['y_train'], verify_grad=False)
        
        # Compare
        models_dict = {
            'Linear Regression': linear_model,
            'Logistic Regression': logistic_model
        }
        
        results = compare_binary_models(models_dict, filtered_data['X_test'], filtered_data['y_test'])
        
        print("\nComparison Results:")
        for model_name, result in results.items():
            print(f"{model_name}: Accuracy={result['accuracy']:.4f}, AUROC={result['auroc']:.4f}")
        
        # Uncomment to show plots
        # from utils import plot_roc_curve_comparison
        # plot_roc_curve_comparison(models_dict, filtered_data['X_test'], filtered_data['y_test'])
    
    if wine_exists:
        print("\n--- Wine Dataset ---")
        print("Loading data from UCI repository...")
        data = load_wine_data(use_uci=True)
        n_classes = 3
        feature_importance = compute_feature_importance_multiclass(data['X_train'], data['y_train'], n_classes)
        filtered_data = filter_important_features_wine(data, feature_importance)
        
        # Train models
        linear_model = LinearRegression(is_multiclass=True)
        linear_model.fit(filtered_data['X_train'], filtered_data['y_train'])
        
        logistic_model = MulticlassLogisticRegression(lr=0.005, n_iter=1000)
        logistic_model.fit(filtered_data['X_train'], filtered_data['y_train'], verify_grad=False)
        
        # Compare
        models_dict = {
            'Multiple Linear Regression': linear_model,
            'Multiclass Logistic Regression': logistic_model
        }
        
        results = compare_multiclass_models(models_dict, filtered_data['X_test'], filtered_data['y_test'])
        
        print("\nComparison Results:")
        for model_name, result in results.items():
            print(f"{model_name}: Accuracy={result['accuracy']:.4f}")


def main():
    """Main function to run all experiments."""
    print("=" * 80)
    print("LINEAR AND LOGISTIC REGRESSION - FULL EXPERIMENT SUITE")
    print("=" * 80)
    
    # Check data availability
    bc_exists, wine_exists = check_data_availability()
    
    if not (bc_exists or wine_exists):
        print("\nERROR: Cannot load datasets. Please install ucimlrepo:")
        print("   pip install -r requirements.txt")
        return
    
    print("Ready to download datasets from UCI repository\n")
    
    # Run experiments
    try:
        run_feature_analysis(bc_exists, wine_exists)
        run_model_training(bc_exists, wine_exists)
        run_model_comparison(bc_exists, wine_exists)
        
        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNote: Some visualizations are commented out to prevent plot windows from blocking execution.")
        print("      Uncomment the plot function calls in main.py to see visualizations.")
        
    except Exception as e:
        print(f"\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

