import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from numdifftools._typing import Array
from numdifftools.fornberg import fd_derivative


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
)
@pytest.mark.parametrize("n", [1, 2])
def test_fractional_derivative_of_integer_samples(dtype: type[np.generic], n: int) -> None:
    samples = np.arange(11, dtype=dtype) ** n
    original = samples.copy()
    x = 2 * np.arange(11)

    result = fd_derivative(samples, x, n=n)

    assert result.dtype == np.dtype(float)
    assert_allclose(result, 0.5, atol=1e-12)
    assert_array_equal(samples, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_sampled_derivative_preserves_inexact_dtype(dtype: type[np.generic]) -> None:
    x = np.arange(11, dtype=float)
    samples = (x**2 / 4).astype(dtype)
    expected: Array = x / 2
    if np.issubdtype(dtype, np.complexfloating):
        samples += 1j * x
        expected = expected + 1j

    result = fd_derivative(samples, x)

    assert result.dtype == samples.dtype
    assert_allclose(result, expected, atol=1e-12)
