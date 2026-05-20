import matplotlib
matplotlib.use("Agg")

import numpy as np
import AI_stats_lab as lab


# ============================================================
# Question 1 Tests
# ============================================================

def test_load_iris_unlabeled_structure():
    result = lab.load_iris_unlabeled(feature_indices=(0, 1))

    assert isinstance(result, dict)
    assert "X" in result
    assert "feature_names" in result

    X = result["X"]
    feature_names = result["feature_names"]

    assert isinstance(X, np.ndarray)
    assert X.shape == (150, 2)
    assert isinstance(feature_names, list)
    assert len(feature_names) == 2
    assert "sepal length" in feature_names[0].lower()
    assert "sepal width" in feature_names[1].lower()


def test_load_iris_unlabeled_no_labels():
    result = lab.load_iris_unlabeled(feature_indices=(0, 1))

    assert "y" not in result
    assert "target" not in result
    assert "labels" not in result


def test_standardize_features_mean_std():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    result = lab.standardize_features(X)

    assert isinstance(result, dict)
    assert "X_scaled" in result
    assert "mean" in result
    assert "std" in result

    X_scaled = result["X_scaled"]

    assert np.allclose(np.mean(X_scaled, axis=0), np.array([0.0, 0.0]))
    assert np.allclose(np.std(X_scaled, axis=0), np.array([1.0, 1.0]))


def test_standardize_features_zero_std():
    X = np.array([
        [1.0, 5.0],
        [1.0, 6.0],
        [1.0, 7.0]
    ])

    result = lab.standardize_features(X)

    X_scaled = result["X_scaled"]
    std = result["std"]

    assert std[0] == 1.0
    assert np.all(np.isfinite(X_scaled))


def test_fit_kmeans_structure():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 5.0],
        [5.1, 5.0],
        [10.0, 10.0],
        [10.1, 10.0]
    ])

    result = lab.fit_kmeans(X, K=3, random_state=0, n_init=10)

    assert isinstance(result, dict)

    required_keys = {
        "centroids",
        "labels",
        "objective",
        "n_iter"
    }

    assert required_keys.issubset(result.keys())

    assert result["centroids"].shape == (3, 2)
    assert len(result["labels"]) == len(X)
    assert result["objective"] >= 0
    assert result["n_iter"] > 0


def test_compute_kmeans_objective_simple():
    X = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [10.0, 0.0]
    ])

    centroids = np.array([
        [1.0, 0.0],
        [10.0, 0.0]
    ])

    labels = np.array([0, 0, 1])

    objective = lab.compute_kmeans_objective(X, centroids, labels)

    assert abs(objective - 2.0) < 1e-6


# ============================================================
# Question 2 Tests
# ============================================================

def test_evaluate_k_values_structure():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 5.0],
        [5.1, 5.0],
        [10.0, 10.0],
        [10.1, 10.0]
    ])

    k_values = [1, 2, 3]

    result = lab.evaluate_k_values(X, k_values, random_state=0, n_init=10)

    assert isinstance(result, dict)

    required_keys = {
        "k_values",
        "objectives",
        "relative_improvements"
    }

    assert required_keys.issubset(result.keys())

    assert result["k_values"] == k_values
    assert len(result["objectives"]) == len(k_values)
    assert len(result["relative_improvements"]) == len(k_values)

    assert result["relative_improvements"][0] == 0.0

    for obj in result["objectives"]:
        assert obj >= 0


def test_evaluate_k_values_objective_decreases():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 5.0],
        [5.1, 5.0],
        [10.0, 10.0],
        [10.1, 10.0]
    ])

    result = lab.evaluate_k_values(X, [1, 2, 3], random_state=0, n_init=10)

    objectives = result["objectives"]

    assert objectives[0] >= objectives[1]
    assert objectives[1] >= objectives[2]


def test_choose_elbow_k():
    k_values = [1, 2, 3, 4, 5]
    objectives = [100.0, 60.0, 30.0, 28.0, 27.0]

    best_k = lab.choose_elbow_k(k_values, objectives)

    assert best_k == 3


def test_cluster_size_summary():
    labels = np.array([0, 0, 1, 1, 1, 2])
    K = 3

    result = lab.cluster_size_summary(labels, K)

    assert isinstance(result, dict)
    assert result[0] == 2
    assert result[1] == 3
    assert result[2] == 1


def test_identify_outliers_by_distance():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 5.0],
        [20.0, 20.0]
    ])

    centroids = np.array([
        [0.0, 0.0],
        [5.0, 5.0]
    ])

    labels = np.array([0, 0, 1, 1])

    result = lab.identify_outliers_by_distance(
        X,
        centroids,
        labels,
        top_n=2
    )

    assert isinstance(result, dict)
    assert "indices" in result
    assert "distances" in result

    assert len(result["indices"]) == 2
    assert len(result["distances"]) == 2

    assert result["indices"][0] == 3
    assert result["distances"][0] >= result["distances"][1]


def test_diagnose_clustering_fit():
    assert lab.diagnose_clustering_fit(K=2, elbow_k=4) == "underfitting"
    assert lab.diagnose_clustering_fit(K=4, elbow_k=4) == "good_fit"
    assert lab.diagnose_clustering_fit(K=8, elbow_k=4) == "overfitting"


# ============================================================
# Question 3 Tests: Visualization
# ============================================================

def test_plot_unlabeled_data_returns_fig_ax():
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ])

    fig, ax = lab.plot_unlabeled_data(
        X,
        feature_names=["Feature 1", "Feature 2"],
        title="Test Plot"
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Test Plot"
    assert ax.get_xlabel() == "Feature 1"
    assert ax.get_ylabel() == "Feature 2"


def test_plot_kmeans_clusters_returns_fig_ax():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 5.0],
        [5.1, 5.0]
    ])

    labels = np.array([0, 0, 1, 1])

    centroids = np.array([
        [0.05, 0.0],
        [5.05, 5.0]
    ])

    fig, ax = lab.plot_kmeans_clusters(
        X,
        labels,
        centroids,
        feature_names=["x1", "x2"],
        title="Cluster Plot"
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Cluster Plot"
    assert ax.get_xlabel() == "x1"
    assert ax.get_ylabel() == "x2"


def test_plot_elbow_curve_returns_fig_ax():
    k_values = [1, 2, 3, 4]
    objectives = [100.0, 60.0, 30.0, 28.0]

    fig, ax = lab.plot_elbow_curve(
        k_values,
        objectives,
        title="Elbow Test"
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Elbow Test"
    assert ax.get_xlabel() == "Number of clusters K"
    assert ax.get_ylabel() == "Objective value"
