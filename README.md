# Linear & Logistic Regression from Scratch

A from-scratch implementation of linear and logistic regression models using only NumPy. This project demonstrates the complete machine learning pipeline—from feature engineering to model evaluation—on real-world classification tasks.

## Report
[LR_Classifcation Report (PDF)](<Linear Regression Report.pdf>)

## What This Is

I built these classification algorithms from first principles without using high-level ML libraries like scikit-learn (except for metrics and preprocessing utilities). The implementation includes:

- **Binary & Multiclass Classification**: Linear regression adapted for classification, binary logistic regression with sigmoid activation, and softmax-based multiclass regression
- **Feature Importance Analysis**: Correlation-based feature selection that automatically identifies and filters the most predictive features
- **Gradient Verification**: Numerical gradient checking using finite differences to validate the analytical gradient implementations
- **Comprehensive Evaluation**: Full performance analysis with AUROC, confusion matrices, ROC curves, and coefficient comparisons

The models are tested on the [UCI Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) and [Wine](https://archive.ics.uci.edu/ml/datasets/Wine) datasets, achieving **~95% accuracy** on breast cancer diagnosis and **~94-100%** on wine classification.

## Getting Started

```bash
git clone <your-repo-url>
cd LRC

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the streamlined experiments (3 core experiments, ~30 seconds)
python main.py

# Or run the full research suite (9 comprehensive experiments, ~2 minutes)
python run_all_experiments.py
```

## Experiment Runners

The project includes two experiment scripts with different scopes:

### `main.py` — Core Experiments

A streamlined runner that executes the three fundamental experiments:

1. **Feature Importance Analysis** — Computes and ranks features by their correlation with target variables
2. **Model Training & Evaluation** — Trains linear and logistic regression models on filtered feature sets
3. **Model Comparison** — Evaluates and compares model performance

Completes in ~10-30 seconds and outputs clean metrics (accuracy, AUROC) for quick validation.

### `run_all_experiments.py` — Full Research Suite

The complete experimental pipeline with nine comprehensive experiments:

1. **Feature Importance Analysis & Selection** — Detailed feature ranking with mean-threshold filtering
2. **Gradient Verification** — Validates analytical gradients against numerical approximations
3. **Convergence Analysis** — Monitors training convergence and tracks loss curves
4. **Train/Val/Test Split Optimization** — Tests multiple split ratios to find optimal data partitioning
5. **Final Model Training** — Trains models on optimized splits with full evaluation metrics
6. **ROC Curve Comparison** — Generates ROC curves comparing model discrimination ability
7. **Coefficient Comparison** — Analyzes and visualizes learned weights across model types
8. **Polynomial Feature Transformation** — Tests performance improvements from non-linear feature engineering
9. **Comparison with Standard Models** — Benchmarks custom implementations against scikit-learn's KNN and Decision Trees

Runs in ~1-3 minutes and produces comprehensive results including split comparisons, gradient verification reports, and sklearn benchmarks.

## Project Structure

```
LRC/
├── models/
│   ├── linear_regression.py              # Closed-form linear regression
│   ├── logistic_regression.py            # Binary logistic with gradient descent
│   └── multiclass_logistic_regression.py # Softmax regression
├── preprocessing/
│   ├── uci_datasets.py                   # UCI repository auto-download
│   ├── breast_cancer.py                  # Breast cancer data pipeline
│   └── wine.py                           # Wine data pipeline
├── utils/
│   ├── feature_importance.py             # Feature selection algorithms
│   ├── evaluation.py                     # Metrics & model evaluation
│   └── visualization.py                  # Plotting functions
├── experiments/
│   ├── feature_analysis.py               # Feature importance analysis
│   ├── train_models.py                   # Model training pipeline
│   └── compare_models.py                 # Model comparison & benchmarking
├── data/                                  # Dataset cache (auto-created)
├── run_all_experiments.py                 # Full experiment suite
├── main.py                                # Streamlined experiment runner
└── config.py                              # Configuration settings
```

## Model Implementations

### Linear Regression

Uses the closed-form solution `W = (X^T X)^(-1) X^T y` for parameter estimation. The implementation supports binary classification (threshold at 0.5), multiclass classification (one-vs-rest), and multivariate outputs (one-hot encoded targets). Target standardization is applied to improve numerical stability.

### Binary Logistic Regression

Implements sigmoid activation with binary cross-entropy loss, optimized via batch gradient descent. The training loop includes early stopping based on loss convergence and supports numerical gradient verification to ensure implementation correctness.

### Multiclass Logistic Regression

Softmax-based classifier for multi-class problems. Uses cross-entropy loss with one-hot encoded targets and batch gradient descent. Analytical gradients are computed directly and verified numerically during development.

## Feature Engineering

The feature importance system ranks features based on their individual correlation with target variables. For binary classification, this uses simple linear regression coefficients. For multiclass problems, importance is computed per-class and then aggregated.

Features with absolute importance above the mean threshold are retained, effectively filtering out weak predictors. The system includes visualization tools that generate horizontal bar plots showing directional relationships between features and targets.

## Evaluation & Visualization

The evaluation suite computes standard metrics including accuracy, AUROC (for binary tasks), and confusion matrices for detailed error analysis. 

Visualization functions generate:
- **ROC curves** with AUROC scores for comparing binary classifiers
- **Loss history plots** showing convergence during training
- **Coefficient heatmaps** for analyzing learned weights across features and classes
- **Feature importance charts** with directional indicators

## Code Examples

### Binary Classification

```python
from preprocessing import load_breast_cancer_data
from models import LogisticRegression
from utils import evaluate_binary_model

data = load_breast_cancer_data(use_uci=True)

model = LogisticRegression(lr=0.005, n_iter=1000)
model.fit(data['X_train'], data['y_train'])

results = evaluate_binary_model(model, data['X_test'], data['y_test'])
print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"AUROC: {results['auroc']:.4f}")
```

### Feature Importance

```python
from utils import compute_feature_importance, plot_feature_importance_binary

importance = compute_feature_importance(data['X_train'], data['y_train'])

plot_feature_importance_binary(
    data['feature_names'], 
    importance,
    title="Feature Importance - Breast Cancer"
)
```

### Multi-Model Comparison

```python
from models import LinearRegression, LogisticRegression
from utils import plot_roc_curve_comparison

linear_model = LinearRegression(is_multiclass=False)
linear_model.fit(X_train, y_train)

logistic_model = LogisticRegression(lr=0.005, n_iter=1000)
logistic_model.fit(X_train, y_train)

models = {
    'Linear Regression': linear_model,
    'Logistic Regression': logistic_model
}
plot_roc_curve_comparison(models, X_test, y_test)
```

## Individual Experiments

The `experiments/` directory contains standalone scripts for specific analyses:

**`feature_analysis.py`** — Analyzes feature correlations and generates visualizations showing directional relationships between features and targets

**`train_models.py`** — Trains all model variants on both datasets with automatic gradient verification and reports accuracy, AUROC, and confusion matrices

**`compare_models.py`** — Benchmarks linear vs logistic regression with ROC curves and coefficient comparison plots

## Performance Results

| Dataset | Model | Accuracy | AUROC |
|---------|-------|----------|-------|
| Breast Cancer | Linear Regression | ~94-95% | ~0.999 |
| Breast Cancer | Logistic Regression | ~95-96% | ~0.997 |
| Wine | Linear Regression | ~100% | N/A |
| Wine | Logistic Regression | ~94-95% | N/A |

*Results on test sets with 80/10/10 train/val/test splits*

## Implementation Details

### Gradient Descent

The optimization loop uses batch gradient descent with configurable learning rates and iteration limits:

```python
for i in range(n_iter):
    gradient = self.compute_gradient(X, y)
    self.W -= self.lr * gradient
    loss = self.compute_loss(X, y)
    if abs(prev_loss - loss) < tol:
        break
```

Convergence is detected when the loss change falls below a tolerance threshold (default `1e-5`).

### Gradient Verification

During development, I validated the analytical gradients against numerical approximations using finite differences:

```python
numerical = (loss(θ + ε) - loss(θ - ε)) / (2ε)
analytical = ∂loss/∂θ
assert |numerical - analytical| < 1e-7
```

This ensures the gradient implementations are mathematically correct.

### Feature Selection

The feature ranking algorithm computes correlation coefficients via simple regression:

```python
w = (X^T y) / N  # Correlation coefficients
important_features = |w| > mean(|w|)
```

Only features with above-average absolute importance are retained.

## Configuration

Model hyperparameters and data processing settings are centralized in `config.py`:

```python
LOGISTIC_REGRESSION_CONFIG = {
    'lr': 0.005,
    'n_iter': 1000,
    'tol': 1e-5
}

TRAIN_TEST_SPLIT_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}
```

## Dependencies

- **NumPy** ≥1.21 — Core numerical operations and linear algebra
- **Pandas** ≥1.3 — Data manipulation and analysis
- **Matplotlib** ≥3.4 — Plotting and visualization
- **Seaborn** ≥0.11 — Statistical visualizations
- **Scikit-learn** ≥1.0 — Preprocessing utilities and evaluation metrics
- **ucimlrepo** — Automatic dataset downloads from UCI repository

## Datasets

### Breast Cancer Wisconsin (Diagnostic)
569 samples, 30 features. Binary classification task distinguishing malignant from benign tumors based on cell nucleus measurements (radius, texture, perimeter, area, smoothness, etc.).

### Wine Recognition
178 samples, 13 features. Multi-class classification task identifying wine cultivars based on chemical analysis (alcohol content, acidity, phenols, color intensity, etc.).

Both datasets download automatically from the UCI ML Repository on first run.

## Notes

This project was built to understand ML algorithms from the ground up. While production systems should use well-tested libraries like scikit-learn, implementing these algorithms manually provides insights into how they actually work under the hood.

The code emphasizes clarity and correctness over performance optimization—the goal is educational rather than production-ready.

## Acknowledgments

- UCI Machine Learning Repository for the datasets
- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995) — Breast Cancer Wisconsin dataset
- Forina, M. et al. (1991) — Wine Recognition dataset

---

*Built to demonstrate ML fundamentals and clean code practices*
