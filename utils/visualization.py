"""
Visualization Utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


def plot_feature_importance_binary(feature_names, feature_importance, title="Feature Importance", figsize=(14, 16)):
    """
    Plot feature importance for binary classification.
    
    Args:
        feature_names (list): List of feature names
        feature_importance (np.ndarray): Feature importance scores
        title (str): Plot title
        figsize (tuple): Figure size
    """
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance (not absolute)
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)
    
    # Create horizontal bar plot
    plt.figure(figsize=figsize)
    plt.barh(range(len(feature_importance)), feature_importance_df['Importance'])
    plt.yticks(range(len(feature_importance)), feature_importance_df['Feature'], fontsize=16)
    plt.xticks(fontsize=16)
    plt.title(title, fontsize=18)
    plt.xlabel('Regression Coefficient Weight', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_feature_importance_multiclass(feature_names, feature_importance, n_classes, 
                                       title_prefix="Feature Importance", figsize=(28, 14)):
    """
    Plot feature importance for multiclass classification (one subplot per class).
    
    Args:
        feature_names (list): List of feature names
        feature_importance (np.ndarray): Feature importance scores (D x C)
        n_classes (int): Number of classes
        title_prefix (str): Prefix for plot title
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    fig.suptitle(f'{title_prefix} by Class', fontsize=20)
    
    for class_idx in range(n_classes):
        # Create DataFrame for current class
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance[:, class_idx]
        })
        
        # Sort by importance in decreasing order
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create horizontal bar plot
        ax = axes[class_idx] if n_classes > 1 else axes
        ax.barh(range(len(feature_names)), importance_df['Importance'])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(importance_df['Feature'], fontsize=18)
        ax.set_title(f'Class {class_idx + 1}', fontsize=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Regression Coefficient Weight', fontsize=18)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(18)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_loss_history(loss_history, title="Loss vs. Iterations", figsize=(10, 5)):
    """
    Plot training loss history.
    
    Args:
        loss_history (list): List of loss values per iteration
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(conf_mat, class_names=None, title="Confusion Matrix", figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        conf_mat (np.ndarray): Confusion matrix
        class_names (list): List of class names
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_comparison(models_dict, X_test, y_test_dict, 
                                       titles=None, figsize=(12, 5)):
    """
    Plot confusion matrices for multiple models side by side.
    
    Args:
        models_dict (dict): Dictionary of model_name: model pairs
        X_test (dict): Dictionary of dataset_name: X_test pairs
        y_test_dict (dict): Dictionary of dataset_name: y_test pairs
        titles (list): List of titles for each subplot
        figsize (tuple): Figure size
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        dataset_name = list(X_test.keys())[idx]
        y_pred = model.predict(X_test[dataset_name])
        y_test = y_test_dict[dataset_name]
        
        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(y_test, y_pred)
        
        ax = axes[idx]
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        title = titles[idx] if titles else f'{name} Confusion Matrix'
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve_comparison(models_dict, X_test, y_test, labels=None, figsize=(10, 6)):
    """
    Plot ROC curves for multiple binary classification models.
    
    Args:
        models_dict (dict): Dictionary of model_name: model pairs
        X_test: Test features
        y_test: Test labels
        labels (list): List of labels for each model
        figsize (tuple): Figure size
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    plt.figure(figsize=figsize)
    
    for idx, (name, model) in enumerate(models_dict.items()):
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_prob = model.predict_proba(X_test)
        else:
            # For linear regression, use sigmoid
            y_pred_prob = 1 / (1 + np.exp(-(X_test @ model.W)))
            if y_pred_prob.ndim > 1:
                y_pred_prob = y_pred_prob.ravel()
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auroc = roc_auc_score(y_test, y_pred_prob)
        
        # Plot
        label = labels[idx] if labels else name
        linestyle = '--' if 'Linear' in name else '-'
        plt.plot(fpr, tpr, label=f'{label} (AUROC = {auroc:.4f})', linestyle=linestyle)
    
    plt.plot([0, 1], [0, 1], linestyle='dotted', color='black', label='Random Guess')
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve Comparison", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_coefficient_comparison(feature_names, coefficients_dict, 
                                title="Coefficient Comparison", figsize=(18, 8)):
    """
    Plot coefficient comparison for multiple models side by side.
    
    Args:
        feature_names (list): List of feature names
        coefficients_dict (dict): Dictionary of model_name: coefficients pairs
        title (str): Overall plot title
        figsize (tuple): Figure size
    """
    n_models = len(coefficients_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, coefficients) in enumerate(coefficients_dict.items()):
        # Create DataFrame
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values(by='Coefficient', key=np.abs, ascending=False)
        
        # Plot
        ax = axes[idx]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'])
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_yticks(np.arange(len(coef_df)))
        ax.set_yticklabels(coef_df['Feature'], fontsize=14)
        ax.set_xlabel('Regression Coefficient Weights', fontsize=14)
        ax.set_title(f'{name} Coefficients', fontsize=18)
        ax.invert_yaxis()
    
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def wrap_labels(labels, width):
    """Wrap long labels for better visualization."""
    return ["\n".join(textwrap.wrap(label.get_text(), width)) for label in labels]


def plot_multiclass_heatmap(linear_coefficients, logistic_coefficients, 
                            feature_names, class_labels, figsize=(16, 12)):
    """
    Plot heatmaps for multiclass coefficients.
    
    Args:
        linear_coefficients (np.ndarray): Linear regression coefficients (D x C)
        logistic_coefficients (np.ndarray): Logistic regression coefficients (D x C)
        feature_names (list): List of feature names
        class_labels (list): List of class labels
        figsize (tuple): Figure size
    """
    # Create DataFrames for visualization
    linear_df = pd.DataFrame(linear_coefficients, index=feature_names, columns=class_labels)
    logistic_df = pd.DataFrame(logistic_coefficients, index=feature_names, columns=class_labels)

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(linear_df, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=axes[0])
    axes[0].set_title("Multiple Linear Regression Coefficients", fontsize=18)
    axes[0].set_xlabel("Class", fontsize=16)
    axes[0].set_ylabel("Feature", fontsize=16)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=16)
    wrapped_labels0 = wrap_labels(axes[0].get_yticklabels(), 10)
    axes[0].set_yticklabels(wrapped_labels0, fontsize=16, rotation=0)

    sns.heatmap(logistic_df, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=axes[1])
    axes[1].set_title("Multiclass Logistic Regression Coefficients", fontsize=18)
    axes[1].set_xlabel("Class", fontsize=16)
    axes[1].set_ylabel("Feature", fontsize=16)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize=16)
    wrapped_labels1 = wrap_labels(axes[1].get_yticklabels(), 10)
    axes[1].set_yticklabels(wrapped_labels1, fontsize=16, rotation=0)

    plt.tight_layout()
    plt.show()

