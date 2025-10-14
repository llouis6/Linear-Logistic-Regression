"""
Configuration file for the project.
Update these paths to match your data locations.
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
BREAST_CANCER_DATA_PATH = os.path.join(DATA_DIR, 'wdbc.data')
WINE_DATA_PATH = os.path.join(DATA_DIR, 'wine.data')

# Model hyperparameters
LINEAR_REGRESSION_CONFIG = {
    'is_multiclass': False,
    'is_multivariate': False
}

LOGISTIC_REGRESSION_CONFIG = {
    'lr': 0.005,
    'n_iter': 1000,
    'tol': 1e-5
}

MULTICLASS_LOGISTIC_REGRESSION_CONFIG = {
    'lr': 0.005,
    'n_iter': 1000,
    'tol': 1e-5
}

# Data split configuration
TRAIN_TEST_SPLIT_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

VAL_TEST_SPLIT_CONFIG = {
    'test_size': 0.5,
    'random_state': 42,
    'stratify': True
}

# Feature selection
FEATURE_SELECTION_CONFIG = {
    'use_mean_threshold': True,
    'custom_threshold': None  # Set to a value if you don't want to use mean
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'dpi': 100,
    'style': 'default',
    'figure_sizes': {
        'feature_importance': (14, 16),
        'loss_history': (10, 5),
        'confusion_matrix': (8, 6),
        'roc_curve': (10, 6),
        'heatmap': (16, 12)
    }
}

# Experiment settings
EXPERIMENT_CONFIG = {
    'verify_gradients': True,
    'verbose': True
}

