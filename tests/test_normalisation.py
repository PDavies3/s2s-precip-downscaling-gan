import numpy as np

from utils.normalisation import denormalize, normalize


def test_zscore_round_trip():
    x = np.array([-3.0, 0.0, 5.0, 12.5])
    config = {"type": "zscore", "stats": {"mean": 2.0, "std": 4.0}}
    np.testing.assert_allclose(denormalize(normalize(x, config), config), x)


def test_log1p_round_trip():
    x = np.array([0.0, 1.0, 10.0, 100.0])
    config = {"type": "log1p", "stats": {"log1p_mean": 0.5, "log1p_std": 1.2}}
    np.testing.assert_allclose(denormalize(normalize(x, config), config), x, atol=1e-6)


def test_minmax_round_trip():
    x = np.array([-10210.0, -5000.0, 0.0, 3000.0, 6276.0])
    config = {"type": "minmax", "stats": {"min": -10210.0, "max": 6276.0}}
    np.testing.assert_allclose(denormalize(normalize(x, config), config), x, atol=1e-6)


def test_minmax_maps_extremes_to_exactly_minus_one_and_one():
    config = {"type": "minmax", "stats": {"min": -10210.0, "max": 6276.0}}
    result = normalize(np.array([-10210.0, 6276.0]), config)
    np.testing.assert_allclose(result, [-1.0, 1.0])


def test_minmax_maps_midpoint_to_zero():
    config = {"type": "minmax", "stats": {"min": 0.0, "max": 3.0}}
    result = normalize(np.array([1.5]), config)
    np.testing.assert_allclose(result, [0.0])


def test_minmax_never_exceeds_bounds_for_values_within_the_fitted_range():
    # linear interpolation between grid points can't exceed the source
    # data's true range, so a value fit on the file's true global min/max
    # is guaranteed to normalize within [-1, 1].
    config = {"type": "minmax", "stats": {"min": 0.0, "max": 3.0}}
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 3.0, size=1000)
    result = normalize(values, config)
    assert result.min() >= -1.0
    assert result.max() <= 1.0
