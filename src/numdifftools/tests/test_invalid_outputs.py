"""Invalid output columns must not change other derivative estimates."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import numdifftools as nd
from numdifftools._typing import Array, EstimateResult
from numdifftools.limits import _Limit


def test_select_estimates_with_invalid_column() -> None:
    errors = np.array([[3.0, np.nan, 2.0], [1.0, np.nan, 3.0], [2.0, np.nan, 1.0]])
    original = errors.copy()
    with pytest.warns(UserWarning, match="All-NaN slice"):
        indices = _Limit._get_arg_min(errors)
    assert_array_equal(indices, [3, 1, 8])
    assert_array_equal(errors, original)


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
@pytest.mark.parametrize("n", [1, 2])
def test_valid_derivatives_unchanged_by_invalid_output(method: str, n: int) -> None:
    x = np.array([0.5, 1.0, 5.0])
    valid = np.array([True, False, True])

    def masked_exp(x: Array) -> Array:
        return np.where(valid, np.exp(x), np.nan)

    expected = nd.Derivative(np.exp, method=method, n=n, full_output=True)(x)
    with pytest.warns(Warning, match="All-NaN slice"):
        actual = nd.Derivative(masked_exp, method=method, n=n, full_output=True)(x)
    assert isinstance(actual, EstimateResult)
    assert isinstance(expected, EstimateResult)
    assert np.isnan(np.asarray(actual.estimate)[1])
    for actual_field, expected_field in zip(actual[:3], expected[:3], strict=True):
        assert_array_equal(np.asarray(actual_field)[valid], np.asarray(expected_field)[valid])
