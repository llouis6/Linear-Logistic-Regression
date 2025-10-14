"""
Utilities module - Helper functions for feature importance, evaluation, and visualization.
"""
from .feature_importance import (
    compute_feature_importance,
    compute_feature_importance_multiclass,
    rank_features_binary,
    rank_features_multiclass,
    select_important_features
)
from .evaluation import (
    evaluate_model,
    evaluate_binary_model,
    evaluate_multiclass_model,
    compare_binary_models,
    compare_multiclass_models,
    compute_roc_curve,
    print_evaluation_results
)
from .visualization import (
    plot_feature_importance_binary,
    plot_feature_importance_multiclass,
    plot_loss_history,
    plot_confusion_matrix,
    plot_confusion_matrices_comparison,
    plot_roc_curve_comparison,
    plot_coefficient_comparison,
    plot_multiclass_heatmap
)

__all__ = [
    'compute_feature_importance',
    'compute_feature_importance_multiclass',
    'rank_features_binary',
    'rank_features_multiclass',
    'select_important_features',
    'evaluate_model',
    'evaluate_binary_model',
    'evaluate_multiclass_model',
    'compare_binary_models',
    'compare_multiclass_models',
    'compute_roc_curve',
    'print_evaluation_results',
    'plot_feature_importance_binary',
    'plot_feature_importance_multiclass',
    'plot_loss_history',
    'plot_confusion_matrix',
    'plot_confusion_matrices_comparison',
    'plot_roc_curve_comparison',
    'plot_coefficient_comparison',
    'plot_multiclass_heatmap'
]

