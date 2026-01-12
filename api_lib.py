"""
API Library Layer - Wraps existing ML functionality for API consumption.
Pure function interface with no print statements, returns JSON-serializable data.
"""
import numpy as np
import time
from typing import Dict, Any, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve

from preprocessing import (
    load_breast_cancer_data, 
    load_wine_data,
    filter_important_features_breast_cancer,
    filter_important_features_wine
)
from models import LinearRegression, LogisticRegression, MulticlassLogisticRegression
from utils import (
    compute_feature_importance,
    compute_feature_importance_multiclass,
    evaluate_binary_model,
    evaluate_multiclass_model
)


class DatasetCache:
    """Pre-load and cache datasets for fast API responses."""
    
    def __init__(self):
        self._datasets = {}
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all datasets into memory."""
        try:
            self._datasets['breast_cancer'] = load_breast_cancer_data(use_uci=True)
        except Exception as e:
            # Fallback: datasets will be loaded on first request
            pass
        
        try:
            self._datasets['wine'] = load_wine_data(use_uci=True)
        except Exception as e:
            pass
    
    def get_dataset(self, name: str) -> Dict[str, Any]:
        """Get cached dataset or load it."""
        if name not in self._datasets:
            if name == 'breast_cancer':
                self._datasets[name] = load_breast_cancer_data(use_uci=True)
            elif name == 'wine':
                self._datasets[name] = load_wine_data(use_uci=True)
            else:
                raise ValueError(f"Unknown dataset: {name}")
        
        # Return a copy to avoid mutations
        return {k: v.copy() if isinstance(v, np.ndarray) else v 
                for k, v in self._datasets[name].items()}


# Global dataset cache (singleton)
_dataset_cache = None


def get_dataset_cache() -> DatasetCache:
    """Get or create dataset cache."""
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = DatasetCache()
    return _dataset_cache


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train and evaluate a model with given configuration.
    
    Args:
        config: {
            'dataset': 'breast_cancer' | 'wine',
            'model': 'linear' | 'logistic' | 'multiclass_logistic',
            'use_feature_selection': bool (default: True),
            'polynomial_degree': int 1-3 (default: 1)
        }
    
    Returns:
        {
            'success': bool,
            'metrics': {
                'accuracy': float,
                'auroc': float (binary only),
            },
            'coefficients': list or list[list] for multiclass,
            'feature_names': list[str],
            'selected_features_count': int,
            'total_features_count': int,
            'loss_history': list[float],
            'confusion_matrix': list[list[int]],
            'roc_curve': {'fpr': list, 'tpr': list} (binary only),
            'training_time_seconds': float,
            'convergence_iterations': int (for gradient descent models),
            'config': dict (echo back)
        }
    """
    start_time = time.time()
    
    try:
        # Validate configuration
        dataset_name = config.get('dataset')
        if dataset_name not in ['breast_cancer', 'wine']:
            return {
                'success': False,
                'error': f"Invalid dataset: {dataset_name}. Must be 'breast_cancer' or 'wine'."
            }
        
        model_type = config.get('model')
        valid_models = ['linear', 'logistic', 'multiclass_logistic']
        if model_type not in valid_models:
            return {
                'success': False,
                'error': f"Invalid model: {model_type}. Must be one of {valid_models}."
            }
        
        # Get dataset
        cache = get_dataset_cache()
        data = cache.get_dataset(dataset_name)
        is_binary = (dataset_name == 'breast_cancer')
        
        # Store original feature count
        original_feature_count = data['X_train'].shape[1]
        
        # Feature selection
        use_feature_selection = config.get('use_feature_selection', True)
        if use_feature_selection:
            if is_binary:
                importance = compute_feature_importance(data['X_train'], data['y_train'])
                data = filter_important_features_breast_cancer(data, importance)
            else:
                importance = compute_feature_importance_multiclass(
                    data['X_train'], data['y_train'], 3
                )
                data = filter_important_features_wine(data, importance)
        
        selected_feature_count = data['X_train'].shape[1]
        
        # Polynomial features
        poly_degree = config.get('polynomial_degree', 1)
        if poly_degree > 1:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            data['X_train'] = poly.fit_transform(data['X_train'])
            data['X_test'] = poly.transform(data['X_test'])
        
        # Train model
        model = None
        if model_type == 'linear':
            model = LinearRegression(is_multiclass=(not is_binary))
            model.fit(data['X_train'], data['y_train'])
        elif model_type == 'logistic':
            model = LogisticRegression(lr=0.005, n_iter=1000)
            model.fit(data['X_train'], data['y_train'], verify_grad=False)
        elif model_type == 'multiclass_logistic':
            model = MulticlassLogisticRegression(lr=0.005, n_iter=1000)
            model.fit(data['X_train'], data['y_train'], verify_grad=False)
        
        # Evaluate
        if is_binary:
            results = evaluate_binary_model(model, data['X_test'], data['y_test'])
            
            # Get ROC curve data
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(data['X_test'])
            else:
                # Linear regression - apply sigmoid
                y_pred_raw = data['X_test'] @ model.W
                if y_pred_raw.ndim > 1:
                    y_pred_raw = y_pred_raw.ravel()
                y_proba = 1 / (1 + np.exp(-y_pred_raw))
            
            fpr, tpr, _ = roc_curve(data['y_test'], y_proba.ravel())
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        else:
            results = evaluate_multiclass_model(model, data['X_test'], data['y_test'])
            roc_data = None
        
        # Extract coefficients
        coefficients = []
        if hasattr(model, 'w'):
            coefficients = model.w.tolist()
        elif hasattr(model, 'W'):
            coefficients = model.W.tolist()
        
        # Get loss history and convergence info
        loss_history = []
        convergence_iterations = None
        if hasattr(model, 'loss_history'):
            loss_history = [float(x) for x in model.loss_history]
            convergence_iterations = len(loss_history)
        
        training_time = time.time() - start_time
        
        return {
            'success': True,
            'metrics': {
                'accuracy': float(results['accuracy']),
                'auroc': float(results.get('auroc', 0.0)),
            },
            'coefficients': coefficients,
            'feature_names': data['feature_names'],
            'selected_features_count': selected_feature_count,
            'total_features_count': original_feature_count,
            'loss_history': loss_history,
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'roc_curve': roc_data,
            'training_time_seconds': round(training_time, 3),
            'convergence_iterations': convergence_iterations,
            'config': config
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'config': config
        }


def get_dataset_info() -> Dict[str, Any]:
    """
    Get metadata about available datasets.
    
    Returns:
        {
            'datasets': [
                {
                    'id': str,
                    'name': str,
                    'type': 'binary' | 'multiclass',
                    'samples': int,
                    'features': int,
                    'classes': int (multiclass only),
                    'description': str
                }
            ]
        }
    """
    return {
        'datasets': [
            {
                'id': 'breast_cancer',
                'name': 'Breast Cancer Wisconsin (Diagnostic)',
                'type': 'binary',
                'samples': 569,
                'features': 30,
                'description': 'Binary classification of breast tumors as malignant or benign based on cell nucleus measurements',
                'target_classes': ['Benign', 'Malignant']
            },
            {
                'id': 'wine',
                'name': 'Wine Recognition',
                'type': 'multiclass',
                'samples': 178,
                'features': 13,
                'classes': 3,
                'description': 'Multiclass classification of wine cultivars based on chemical analysis',
                'target_classes': ['Class 0', 'Class 1', 'Class 2']
            }
        ]
    }
