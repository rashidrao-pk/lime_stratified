import numpy as np
from scipy.special import binom

from lime.lime_base import LimeBase
from lime.lime_image import LimeImageExplainer


def test_standard_sampling_returns_no_weight_adjustments():
    explainer = LimeImageExplainer(random_state=42)

    image = np.zeros((4, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    data, labels, weight_adjustments = explainer.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=20,
        batch_size=5,
        use_stratification=False,
        progress_bar=False,
    )

    assert data.shape == (20, 4)
    assert labels.shape == (20, 2)
    assert weight_adjustments is None
    assert np.all(data[0] == 1)


def test_stratified_sampling_basic_properties():
    explainer = LimeImageExplainer(random_state=42)

    image = np.zeros((4, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    data, labels, weight_adjustments = explainer.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=20,
        batch_size=5,
        use_stratification=True,
        progress_bar=False,
    )

    assert data.shape == (20, 4)
    assert labels.shape == (20, 2)
    assert weight_adjustments.shape == (20,)

    assert np.all(data[0] == 1)
    assert np.all(np.isin(data, [0, 1]))

    assert np.all(np.isfinite(weight_adjustments))
    assert weight_adjustments[0] == 1


def test_stratified_sampling_is_reproducible():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    explainer_a = LimeImageExplainer(random_state=42)
    explainer_b = LimeImageExplainer(random_state=42)

    data_a, labels_a, weights_a = explainer_a.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=50,
        batch_size=10,
        use_stratification=True,
        progress_bar=False,
    )

    data_b, labels_b, weights_b = explainer_b.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=50,
        batch_size=10,
        use_stratification=True,
        progress_bar=False,
    )

    assert np.array_equal(data_a, data_b)
    assert np.array_equal(labels_a, labels_b)
    assert np.array_equal(weights_a, weights_b)


def test_stratified_weight_adjustments_match_formula():
    explainer = LimeImageExplainer(random_state=7)

    image = np.zeros((4, 4, 3), dtype=np.float32)
    fudged_image = np.zeros_like(image)

    segments = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ])

    def classifier_fn(images):
        return np.tile(np.array([[0.4, 0.6]]), (len(images), 1))

    data, _, weight_adjustments = explainer.data_labels(
        image=image,
        fudged_image=fudged_image,
        segments=segments,
        classifier_fn=classifier_fn,
        num_samples=30,
        batch_size=10,
        use_stratification=True,
        progress_bar=False,
    )

    n_features = data.shape[1]

    expected = np.ones(len(data), dtype=np.float64)

    for i in range(1, len(data)):
        k = np.sum(data[i])
        expected[i] = (
            binom(n_features, k)
            / (2 ** n_features)
        ) * (n_features + 1)

    assert np.allclose(weight_adjustments, expected)


def test_weight_adjustments_multiply_kernel_weights():
    recorded = {}

    def kernel_fn(distances):
        recorded["distances"] = np.array(distances, copy=True)
        return np.exp(-(distances ** 2))

    class RecordingRegressor:
        def fit(self, X, y, sample_weight=None):
            self.sample_weight = np.array(sample_weight, copy=True)
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0.0
            return self

        def score(self, X, y, sample_weight=None):
            return 1.0

        def predict(self, X):
            return np.zeros(X.shape[0])


    base = LimeBase(kernel_fn=kernel_fn, random_state=42)

    neighborhood_data = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
    ], dtype=float)

    neighborhood_labels = np.array([
        [0.2, 0.8],
        [0.4, 0.6],
        [0.7, 0.3],
    ])

    distances = np.array([0.0, 1.0, 2.0])
    weight_adjustments = np.array([1.0, 0.5, 0.25])

    original_distances = distances.copy()
    regressor = RecordingRegressor()

    base.explain_instance_with_data(
        neighborhood_data=neighborhood_data,
        neighborhood_labels=neighborhood_labels,
        distances=distances,
        label=1,
        num_features=2,
        feature_selection="none",
        weight_adjustments=weight_adjustments,
        model_regressor=regressor,
    )

    expected_kernel_weights = np.exp(-(original_distances ** 2))
    expected_weights = expected_kernel_weights * weight_adjustments

    assert np.allclose(recorded["distances"], original_distances)
    assert np.allclose(regressor.sample_weight, expected_weights)
