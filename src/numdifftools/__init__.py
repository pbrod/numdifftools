from __future__ import absolute_import

from . import extrapolation, limits, step_generators
from .core import (
    Derivative,
    Gradient,
    Hessdiag,
    Hessian,
    Jacobian,
    MaxStepGenerator,
    MinStepGenerator,
    Richardson,
    dea3,
    directionaldiff,
)
from .info import __doc__ as __doc__

__version__ = "0.9.41"

__all__ = (
    "Derivative",
    "Gradient",
    "Hessian",
    "Hessdiag",
    "Jacobian",
    "MaxStepGenerator",
    "MinStepGenerator",
    "Richardson",
    "dea3",
    "directionaldiff",
    "extrapolation",
    "limits",
    "step_generators",
)


def test(*options):
    """
    Run tests for module using pytest.

    Parameters
    ----------
    *options : optional
        options to pass to pytest. The most important ones include:
        '-v', '--verbose':
            increase verbosity.
        '-q', '--quiet':
            decrease verbosity.
        '--doctest-modules':
            run doctests in all .py modules
        '--cov':
            measure coverage for .py modules
        '-h', '--help':
            show full help message and display all possible options to use.

    Returns
    -------
    exit_code: scalar
        Exit code is 0 if all tests passed without failure.

    Examples
    --------
    import numdifftols as nd
    nd.test('-q', '--doctest-modules', '--cov', '--disable-warnings')
    """

    import pytest

    return pytest.main(["--pyargs", "numdifftools"] + list(options))
