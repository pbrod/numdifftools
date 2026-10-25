"""Complex-step restrictions apply to Jacobian and its Gradient subclass."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import numdifftools as nd
from numdifftools._typing import Array, ArrayOrScalar


@pytest.mark.parametrize("cls", [nd.Jacobian, nd.Gradient])
@pytest.mark.parametrize("method", ["complex", "multicomplex"])
@pytest.mark.parametrize("full_output", [False, True])
def test_complex_output_rejected_before_perturbation(
    cls: type[nd.Jacobian], method: str, full_output: bool
) -> None:
    calls: list[Array] = []

    def fun(x: Array) -> ArrayOrScalar:
        calls.append(x.copy())
        return np.sum((1 + 1j) * x**2)

    with pytest.raises(ValueError, match="function given is complex valued"):
        cls(fun, method=method, full_output=full_output)([1.0, 2.0])
    assert len(calls) == 1


@pytest.mark.parametrize("cls", [nd.Jacobian, nd.Gradient])
@pytest.mark.parametrize("method", ["complex", "multicomplex"])
def test_complex_variable_rejected(cls: type[nd.Jacobian], method: str) -> None:
    with pytest.raises(ValueError, match="complex variable was given"):
        cls(lambda x: np.sum(x**2), method=method)([1j, 2.0])


@pytest.mark.parametrize("cls", [nd.Jacobian, nd.Gradient])
@pytest.mark.parametrize("method", ["complex", "multicomplex"])
def test_real_valued_function_still_accepted(cls: type[nd.Jacobian], method: str) -> None:
    actual = cls(lambda x: np.sum(x**2), method=method)([1.0, 2.0])
    assert isinstance(actual, np.ndarray)
    assert_allclose(actual, [[2, 4]] if cls is nd.Jacobian else [2, 4])


def test_real_value_with_complex_dtype_accepted() -> None:
    actual = nd.Jacobian(lambda x: np.sum(x**2) + 0j, method="complex")([1.0, 2.0])
    assert isinstance(actual, np.ndarray)
    assert_allclose(actual, [[2, 4]])


def test_central_complex_output_unchanged() -> None:
    actual = nd.Jacobian(lambda x: (1 + 1j) * x**2, method="central")([1.0, 2.0])
    assert isinstance(actual, np.ndarray)
    assert_allclose(actual, np.diag([2 + 2j, 4 + 4j]))
