import numpy as np
import AI_stats_lab as lab


# ============================================================
# Question 1 Tests
# ============================================================

def test_generate_nonlinear_data_shape():
    X, y = lab.generate_nonlinear_data(
        n_samples=50,
        noise=0.1,
        random_state=42
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (50, 1)
    assert y.shape == (50,)


def test_generate_nonlinear_data_reproducibility():
    X1, y1 = lab.generate_nonlinear_data(
        n_samples=30,
        noise=0.2,
        random_state=10
    )

    X2, y2 = lab.generate_nonlinear_data(
        n_samples=30,
        noise=0.2,
        random_state=10
    )

    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)


def test_create_polynomial_model_predicts():
    X = np.array([[0.0], [0.5], [1.0]])
    y = np.array([0.0, 1.0, 0.0])

    model = lab.create_polynomial_model(degree=2)
    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)


def test_evaluate_polynomial_degrees_structure():
    X, y = lab.generate_nonlinear_data(
        n_samples=80,
        noise=0.1,
        random_state=0
    )

    degrees = [1, 3, 5]

    results = lab.evaluate_polynomial_degrees(
        X,
        y,
        degrees,
        test_size=0.25,
        random_state=0
    )

    assert isinstance(results, dict)

    required_keys = {
        "degrees",
        "train_errors",
        "dev_errors",
        "best_degree"
    }

    assert required_keys.issubset(results.keys())
    assert results["degrees"] == degrees
    assert len(results["train_errors"]) == len(degrees)
    assert len(results["dev_errors"]) == len(degrees)

    for err in results["train_errors"]:
        assert err >= 0

    for err in results["dev_errors"]:
        assert err >= 0

    assert results["best_degree"] in degrees


def test_diagnose_from_errors_high_bias():
    result = lab.diagnose_from_errors(
        train_error=0.25,
        dev_error=0.27,
        high_error_threshold=0.15,
        gap_threshold=0.05
    )

    assert isinstance(result, dict)
    assert abs(result["generalization_gap"] - 0.02) < 1e-6
    assert result["diagnosis"] == "high_bias"


def test_diagnose_from_errors_high_variance():
    result = lab.diagnose_from_errors(
        train_error=0.05,
        dev_error=0.16,
        high_error_threshold=0.15,
        gap_threshold=0.05
    )

    assert isinstance(result, dict)
    assert abs(result["generalization_gap"] - 0.11) < 1e-6
    assert result["diagnosis"] == "high_variance"


def test_diagnose_from_errors_high_bias_and_high_variance():
    result = lab.diagnose_from_errors(
        train_error=0.25,
        dev_error=0.35,
        high_error_threshold=0.15,
        gap_threshold=0.05
    )

    assert isinstance(result, dict)
    assert abs(result["generalization_gap"] - 0.10) < 1e-6
    assert result["diagnosis"] == "high_bias_and_high_variance"


def test_diagnose_from_errors_good_fit():
    result = lab.diagnose_from_errors(
        train_error=0.08,
        dev_error=0.10,
        high_error_threshold=0.15,
        gap_threshold=0.05
    )

    assert isinstance(result, dict)
    assert abs(result["generalization_gap"] - 0.02) < 1e-6
    assert result["diagnosis"] == "good_fit"


# ============================================================
# Question 2 Tests
# ============================================================

def test_regularization_comparison_structure():
    X, y = lab.generate_nonlinear_data(
        n_samples=80,
        noise=0.1,
        random_state=5
    )

    X_train = X[:60]
    y_train = y[:60]
    X_dev = X[60:]
    y_dev = y[60:]

    alphas = [0.0, 0.1, 1.0]

    result = lab.regularization_comparison(
        X_train,
        y_train,
        X_dev,
        y_dev,
        alphas
    )

    assert isinstance(result, dict)

    required_keys = {
        "alphas",
        "train_errors",
        "dev_errors",
        "best_alpha"
    }

    assert required_keys.issubset(result.keys())

    assert result["alphas"] == alphas
    assert len(result["train_errors"]) == len(alphas)
    assert len(result["dev_errors"]) == len(alphas)
    assert result["best_alpha"] in alphas

    for err in result["train_errors"]:
        assert err >= 0

    for err in result["dev_errors"]:
        assert err >= 0


def test_recommend_action():
    assert lab.recommend_action("high_bias") == "increase_model_complexity"
    assert lab.recommend_action("high_variance") == "add_regularization_or_more_data"
    assert lab.recommend_action("high_bias_and_high_variance") == "increase_complexity_then_regularize"
    assert lab.recommend_action("good_fit") == "keep_model_or_minor_tuning"
    assert lab.recommend_action("other") == "unknown_diagnosis"
