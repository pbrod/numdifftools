from functools import wraps

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
from .testing import test as _test  # noqa

__version__ = "0.11.0"

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


@wraps(_test)
def test(*options: str) -> int:
    return _test(__name__, *options)
