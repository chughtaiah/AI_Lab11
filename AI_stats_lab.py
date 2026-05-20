"""
AI_stats_lab.py

Lab: Bias-Variance Tradeoff

Topics:
- Nonlinear data generation
- Polynomial regression
- Train/dev error comparison
- Model complexity
- Bias-variance diagnosis
- Regularization comparison
- Model improvement recommendations

Instructions:
- Implement all functions.
- Do NOT change function names.
- Do NOT print inside functions.
- Return exactly the required formats.
"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# ============================================================
# Question 1: Model Complexity and Generalization
# ============================================================

def generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42):
    """
    Generate a nonlinear regression dataset.

    True function:
        y = sin(2*pi*x) + Gaussian noise

    Parameters:
        n_samples    : number of samples
        noise        : standard deviation of Gaussian noise
        random_state : random seed

    Returns:
        X, y

    X shape must be:
        (n_samples, 1)

    y shape must be:
        (n_samples,)
    """
    pass


def create_polynomial_model(degree):
    """
    Create a polynomial regression model using sklearn Pipeline.

    The pipeline must contain:
        PolynomialFeatures(degree=degree, include_bias=False)
        LinearRegression()

    Parameters:
        degree : polynomial degree

    Returns:
        sklearn Pipeline object
    """
    pass


def evaluate_polynomial_degrees(X, y, degrees, test_size=0.3, random_state=0):
    """
    Train polynomial models with different degrees and compute train/dev errors.

    Parameters:
        X            : feature matrix
        y            : target values
        degrees      : list of polynomial degrees
        test_size    : fraction of data used for dev set
        random_state : random seed

    Returns:
        {
            "degrees": list of degrees,
            "train_errors": list of training MSE values,
            "dev_errors": list of dev MSE values,
            "best_degree": degree with lowest dev error
        }

    Hints:
        - Split data into train/dev once using train_test_split.
        - For each degree:
            1. Create polynomial model.
            2. Fit on train set.
            3. Predict on train set and compute train MSE.
            4. Predict on dev set and compute dev MSE.
        - Select best_degree using the lowest dev error.
    """
    pass


def diagnose_from_errors(train_error, dev_error, high_error_threshold=0.15, gap_threshold=0.05):
    """
    Diagnose model behavior using train and dev error.

    Parameters:
        train_error          : training error
        dev_error            : dev error
        high_error_threshold : threshold for high train error
        gap_threshold        : threshold for high dev-train gap

    Returns:
        {
            "train_error": train_error,
            "dev_error": dev_error,
            "generalization_gap": dev_error - train_error,
            "diagnosis": diagnosis_string
        }

    Diagnosis rules:
        If train_error > high_error_threshold and gap <= gap_threshold:
            "high_bias"

        If train_error <= high_error_threshold and gap > gap_threshold:
            "high_variance"

        If train_error > high_error_threshold and gap > gap_threshold:
            "high_bias_and_high_variance"

        Otherwise:
            "good_fit"
    """
    pass


# ============================================================
# Question 2: Regularization and Model Improvement
# ============================================================

def regularization_comparison(X_train, y_train, X_dev, y_dev, alphas):
    """
    Compare Ridge regression models with different regularization strengths.

    Parameters:
        X_train : training features
        y_train : training targets
        X_dev   : dev features
        y_dev   : dev targets
        alphas  : list of Ridge alpha values

    Returns:
        {
            "alphas": list of alpha values,
            "train_errors": list of training MSE values,
            "dev_errors": list of dev MSE values,
            "best_alpha": alpha with lowest dev error
        }

    Hints:
        - Train Ridge(alpha=alpha) for each alpha.
        - Compute train and dev MSE.
        - Select best_alpha using the lowest dev error.
    """
    pass


def recommend_action(diagnosis):
    """
    Recommend an action based on bias/variance diagnosis.

    Required mapping:
        "high_bias" ->
            "increase_model_complexity"

        "high_variance" ->
            "add_regularization_or_more_data"

        "high_bias_and_high_variance" ->
            "increase_complexity_then_regularize"

        "good_fit" ->
            "keep_model_or_minor_tuning"

        anything else ->
            "unknown_diagnosis"
    """
    pass


if __name__ == "__main__":
    print("Implement all required functions.")
