import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from numdifftools.fornberg import fd_weights, fd_weights_all


@pytest.mark.parametrize("center", [1j, 0.5 - 0.75j, np.complex64(0.25 + 0.5j)])
def test_weights_at_complex_center(center: complex) -> None:
    x = np.arange(-2, 3, dtype=float)
    original = x.copy()
    weights = fd_weights_all(x, x0=center, n=4)

    for degree in range(5):
        for order in range(5):
            expected = (
                math.factorial(degree) / math.factorial(degree - order) * complex(center) ** (degree - order)
                if degree >= order
                else 0
            )
            assert_allclose(weights[order] @ x**degree, expected, atol=1e-12)
    assert_array_equal(x, original)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_single_derivative_at_complex_center(order: int) -> None:
    x = np.arange(-2, 3, dtype=float)
    polynomial = np.polynomial.Polynomial([0, 2, 0, 1])

    result = fd_weights(x, x0=1j, n=order) @ polynomial(x)

    assert_allclose(result, polynomial.deriv(order)(1j), atol=1e-12)


def test_zero_imaginary_center_matches_real_center() -> None:
    x = np.arange(-2, 3, dtype=float)
    real = fd_weights_all(x, x0=0.5, n=4)
    complex_result = fd_weights_all(x, x0=0.5 + 0j, n=4)

    assert_allclose(complex_result.real, real, atol=1e-15)
    assert_array_equal(complex_result.imag, 0)
