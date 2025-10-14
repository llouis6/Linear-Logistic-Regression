"""
Linear Regression Model
Supports binary, multiclass, and multivariate regression.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class LinearRegression:
    def __init__(self, is_multiclass=False, is_multivariate=False):
        """
        Initialize Linear Regression model.
        
        Args:
            is_multiclass (bool): If True, uses one-hot encoding for multiclass classification
            is_multivariate (bool): If True, supports multiple continuous outputs
        """
        self.is_multiclass = is_multiclass
        self.W = None
        self.scaler_y = StandardScaler()
        self.is_multivariate = is_multivariate
        self.onehot = OneHotEncoder(sparse_output=False) if is_multiclass else None

    def fit(self, X, y):
        """
        Fit linear regression using closed-form solution.
        
        Args:
            X: input features (N x D)
            y: target values (N,) or (N x C) for multiclass/multivariate
        """
        if self.is_multiclass:
            # One-hot encode and standardize targets for multiclass
            y_onehot = self.onehot.fit_transform(y.reshape(-1, 1))
            y_standardized = self.scaler_y.fit_transform(y_onehot)
        elif self.is_multivariate:
            # Standardize multiple continuous outputs
            y_standardized = self.scaler_y.fit_transform(y)
        else:
            # Standardize binary targets
            y_standardized = self.scaler_y.fit_transform(y.reshape(-1, 1))

        # Compute closed-form solution
        # W = (X^T X)^(-1) X^T y
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX + 1e-8 * np.eye(X.shape[1]))  # Add small regularization for stability
        XTy = X.T @ y_standardized
        self.W = XTX_inv @ XTy

    def predict(self, X):
        """
        Predict using the linear model.
        
        Args:
            X: input features (N x D)
            
        Returns:
            predicted values (N,) or class labels for multiclass
        """
        y_pred = X @ self.W  # Compute raw predictions

        if self.is_multiclass:
            y_pred = self.scaler_y.inverse_transform(y_pred)
            return np.argmax(y_pred, axis=1)  # Return class with highest score
        elif self.is_multivariate:
            return self.scaler_y.inverse_transform(y_pred)  # Return multiple continuous outputs
        else:
            y_pred = self.scaler_y.inverse_transform(y_pred)
            return (y_pred > 0.5).astype(int).ravel()  # Binary threshold at 0.5

