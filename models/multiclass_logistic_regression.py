"""
Multiclass Logistic Regression Model with Gradient Descent
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class MulticlassLogisticRegression:
    def __init__(self, lr=0.005, n_iter=1000, tol=1e-5):
        """
        Initialize Multiclass Logistic Regression model.
        
        Args:
            lr (float): Learning rate
            n_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
        """
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.W = None
        self.loss_history = []
        self.onehot = OneHotEncoder(sparse_output=False)

    def softmax(self, Z):
        """
        Compute softmax activation function.
        
        Args:
            Z: input matrix (N x C)
            
        Returns:
            softmax probabilities (N x C)
        """
        # Subtract max for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def compute_loss(self, X, y):
        """
        Compute cross-entropy loss.
        
        Args:
            X: input features (N x D)
            y: class labels (N,)
            
        Returns:
            loss value
        """
        # One-hot encode targets if they're not already
        if len(y.shape) == 1:
            Y = self.onehot.fit_transform(y.reshape(-1, 1))
        else:
            Y = y

        Z = X @ self.W
        Y_hat = self.softmax(Z)
        loss = -np.sum(Y * np.log(Y_hat + 1e-15)) / len(y)
        return loss

    def compute_gradient(self, X, y):
        """
        Compute analytical gradient.
        
        Args:
            X: input features (N x D)
            y: class labels (N,) or one-hot encoded (N x C)
            
        Returns:
            gradient matrix (D x C)
        """
        # One-hot encode targets if they're not already
        if len(y.shape) == 1:
            Y = self.onehot.fit_transform(y.reshape(-1, 1))
        else:
            Y = y

        Z = X @ self.W
        Y_hat = self.softmax(Z)
        gradient = X.T @ (Y_hat - Y) / len(y)
        return gradient

    def compute_numerical_gradient(self, X, y, epsilon=1e-7):
        """
        Compute numerical gradient using finite differences for verification.
        
        Args:
            X: input features (N x D)
            y: class labels (N,)
            epsilon (float): Small perturbation value
            
        Returns:
            numerical gradient matrix (D x C)
        """
        numerical_gradient = np.zeros_like(self.W)

        # Iterate over each weight
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                # Add small perturbation
                self.W[i, j] += epsilon
                loss_plus = self.compute_loss(X, y)

                # Subtract small perturbation
                self.W[i, j] -= 2 * epsilon
                loss_minus = self.compute_loss(X, y)

                # Restore original weight
                self.W[i, j] += epsilon

                # Compute numerical gradient
                numerical_gradient[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        return numerical_gradient

    def verify_gradients(self, X, y):
        """
        Verify analytical gradients against numerical gradients.
        
        Args:
            X: input features (N x D)
            y: class labels (N,)
            
        Returns:
            tuple of (max_diff, mean_diff)
        """
        analytical_grad = self.compute_gradient(X, y)
        numerical_grad = self.compute_numerical_gradient(X, y)

        # Compute difference
        diff = np.abs(analytical_grad - numerical_grad)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"Maximum gradient difference: {max_diff:.10f}")
        print(f"Mean gradient difference: {mean_diff:.10f}")

        return max_diff, mean_diff

    def fit(self, X, y, verify_grad=True):
        """
        Fit multiclass logistic regression using gradient descent.
        
        Args:
            X: input features (N x D)
            y: class labels (N,)
            verify_grad (bool): Whether to verify gradients before training
        """
        D = X.shape[1]
        n_classes = len(np.unique(y))
        self.W = np.zeros((D, n_classes))  # Initialize weights to zero

        # Verify gradients before training
        if verify_grad:
            print("Verifying gradients before training:")
            self.verify_gradients(X, y)

        # One-hot encode targets
        if len(y.shape) == 1:
            Y = self.onehot.fit_transform(y.reshape(-1, 1))
        else:
            Y = y

        # Training loop
        for i in range(self.n_iter):
            # Compute gradient
            gradient = self.compute_gradient(X, Y)

            # Update weights
            self.W -= self.lr * gradient

            # Compute and store loss
            loss = self.compute_loss(X, Y)
            self.loss_history.append(loss)

            # Check convergence
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                print(f"Converged after {i} iterations")
                break

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: input features (N x D)
            
        Returns:
            predicted probabilities (N x C)
        """
        return self.softmax(X @ self.W)

    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: input features (N x D)
            
        Returns:
            predicted class labels (N,)
        """
        return np.argmax(self.predict_proba(X), axis=1)

