"""
Binary Logistic Regression Model with Gradient Descent
"""
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.005, n_iter=1000, tol=1e-5):
        """
        Initialize Logistic Regression model.
        
        Args:
            lr (float): Learning rate
            n_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
        """
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.w = None
        self.loss_history = []

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y):
        """
        Compute binary cross-entropy loss.
        
        Args:
            X: input features (N x D)
            y: binary targets (N,)
            
        Returns:
            loss value
        """
        z = X @ self.w
        y_hat = self.sigmoid(z)
        # Add small constant for numerical stability
        loss = -np.sum(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15)) / len(y)
        return loss

    def compute_gradient(self, X, y):
        """
        Compute analytical gradient.
        
        Args:
            X: input features (N x D)
            y: binary targets (N,)
            
        Returns:
            gradient vector
        """
        z = X @ self.w
        y_hat = self.sigmoid(z)
        gradient = X.T @ (y_hat - y) / len(y)
        return gradient

    def compute_numerical_gradient(self, X, y, epsilon=1e-7):
        """
        Compute numerical gradient using finite differences for verification.
        
        Args:
            X: input features (N x D)
            y: binary targets (N,)
            epsilon (float): Small perturbation value
            
        Returns:
            numerical gradient vector
        """
        numerical_gradient = np.zeros_like(self.w)
        for i in range(len(self.w)):
            # Add small perturbation
            self.w[i] += epsilon
            loss_plus = self.compute_loss(X, y)

            # Subtract small perturbation
            self.w[i] -= 2 * epsilon
            loss_minus = self.compute_loss(X, y)

            # Restore original weight
            self.w[i] += epsilon

            # Compute numerical gradient
            numerical_gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)

        return numerical_gradient

    def verify_gradients(self, X, y):
        """
        Verify analytical gradients against numerical gradients.
        
        Args:
            X: input features (N x D)
            y: binary targets (N,)
            
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
        Fit logistic regression using gradient descent.
        
        Args:
            X: input features (N x D)
            y: binary targets (N,)
            verify_grad (bool): Whether to verify gradients before training
        """
        D = X.shape[1]
        self.w = np.zeros(D)  # Initialize weights to zero

        # Verify gradients before training
        if verify_grad:
            print("Verifying gradients before training:")
            self.verify_gradients(X, y)

        # Training loop
        for i in range(self.n_iter):
            # Compute gradient
            gradient = self.compute_gradient(X, y)

            # Update weights
            self.w -= self.lr * gradient

            # Compute and store loss
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)

            # Check convergence
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                print(f"Converged after {i} iterations")
                break

        return self

    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: input features (N x D)
            
        Returns:
            predicted probabilities (N,)
        """
        return self.sigmoid(X @ self.w)

    def predict(self, X):
        """
        Predict binary class labels.
        
        Args:
            X: input features (N x D)
            
        Returns:
            predicted class labels (N,)
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

