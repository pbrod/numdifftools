"""
Created on 17. mai 2015

@author: pab
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import scipy.special as special
from numpy import (
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    cos,
    cosh,
    exp,
    exp2,
    expm1,
    log,
    log1p,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
)
from numpy.typing import ArrayLike

function_names: list[str] = [
    "cos",
    "sin",
    "tan",
    "cosh",
    "sinh",
    "tanh",
    "arcsinh",
    "exp",
    "expm1",
    "exp2",
    "square",
    "sqrt",
    "log",
    "log1p",
    "log10",
    "log2",
    "arccos",
    "arcsin",
    "arctan",
]


NumericFunc: TypeAlias = Callable[[ArrayLike], ArrayLike]
DerivativeFunc: TypeAlias = NumericFunc | None
FunctionPair: TypeAlias = tuple[
    DerivativeFunc,
    DerivativeFunc,
]

def darcsin(x: ArrayLike) -> ArrayLike:
    return 1.0 / sqrt(1 - x**2)


def ddarcsin(x: ArrayLike) -> ArrayLike:
    return x * darcsin(x) ** 3


def dddarcsin(x: ArrayLike) -> ArrayLike:
    y = darcsin(x)
    return y**3 * (1 + 3 * (x * y) ** 2)


def darccos(x: ArrayLike) -> ArrayLike:
    return -darcsin(x)


def ddarccos(x: ArrayLike) -> ArrayLike:
    return -ddarcsin(x)


def dddarccos(x: ArrayLike) -> ArrayLike:
    return -dddarcsin(x)


def derivative_arcsin(n: int) -> DerivativeFunc:
    return (arcsin, darcsin, ddarcsin, dddarcsin, None)[min(n, 4)]


def derivative_arccos(n: int) -> DerivativeFunc:
    return (arccos, darccos, ddarccos, dddarccos, None)[min(n, 4)]


def derivative_arctan(n: int) -> DerivativeFunc:
    def darctan(x: ArrayLike) -> ArrayLike:
        return 1.0 / (1 + x**2)

    def ddarctan(x: ArrayLike) -> ArrayLike:
        return -2 * x * darctan(x) ** 2

    def dddarctan(x: ArrayLike) -> ArrayLike:
        y = darctan(x)
        return 2 * (4.0 * x**2 * y - 1.0) * y**2

    def ddddarctan(x: ArrayLike) -> ArrayLike:
        y = darctan(x)
        return (1.0 - 2 * x**2 * y) * 24 * x * y**3

    return (arctan, darctan, ddarctan, dddarctan, ddddarctan, None)[min(n, 5)]


def derivative_sin(n: int):
    def dcos(x: ArrayLike) -> ArrayLike:
        return -sin(x)

    def ddcos(x: ArrayLike) -> ArrayLike:
        return -cos(x)

    return (sin, cos, dcos, ddcos)[n % 4]


def derivative_cos(n: int) -> tuple[NumericFunc, DerivativeFunc, DerivativeFunc, DerivativeFunc]:
    return derivative_sin(n + 1)


def derivative_tan(n: int):
    def dtan(x: ArrayLike) -> ArrayLike:
        return 1.0 / np.cos(x) ** 2

    def ddtan(x: ArrayLike) -> ArrayLike:
        return 2 * tan(x) / cos(x) ** 2

    def dddtan(x: ArrayLike) -> ArrayLike:
        y = tan(x)
        return 2 * (y**2 + 1) * (3 * y**2 + 1)

    def ddddtan(x: ArrayLike) -> ArrayLike:
        y = tan(x)
        return 8 * y * (y**2 + 1) * (3 * y**2 + 2)

    return (tan, dtan, ddtan, dddtan, ddddtan, None)[min(n, 5)]


def derivative_sinh(n: int) -> tuple[NumericFunc, DerivativeFunc, DerivativeFunc, DerivativeFunc]:
    return (sinh, cosh)[n % 2]


def derivative_cosh(n: int) -> tuple[NumericFunc, DerivativeFunc, DerivativeFunc, DerivativeFunc]:
    return derivative_sinh(n + 1)


def derivative_tanh(n: int):
    def dtanh(x: ArrayLike) -> ArrayLike:
        return 1.0 / cosh(x) ** 2

    def ddtanh(x: ArrayLike) -> ArrayLike:
        return -2 * sinh(x) / cosh(x) ** 3

    def dddtanh(x: ArrayLike) -> ArrayLike:
        y = cosh(x)
        return 4 * (tanh(x) / y) ** 2 - 2.0 / y**4

    def ddddtanh(x: ArrayLike) -> ArrayLike:
        y = tanh(x)
        return 8 * y * (y**2 - 1) * (3 * y**2 - 2)

    return (tanh, dtanh, ddtanh, dddtanh, ddddtanh, None)[min(n, 5)]


def _dddarc_h(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return (3 * (x * y) ** 2 - 1) * y**3


def derivative_arccosh(n: int) -> DerivativeFunc:
    def darccosh(x: ArrayLike) -> ArrayLike:
        return 1.0 / sqrt(x**2 - 1)

    def ddarccosh(x: ArrayLike) -> ArrayLike:
        return -x * darccosh(x) ** 3

    def dddarccosh(x: ArrayLike) -> ArrayLike:
        return _dddarc_h(x, darccosh(x))

    return (arccosh, darccosh, ddarccosh, dddarccosh, None)[min(n, 4)]


def derivative_arcsinh(n: int):
    def darcsinh(x: ArrayLike) -> ArrayLike:
        return 1.0 / sqrt(1 + x**2)

    def ddarcsinh(x: ArrayLike) -> ArrayLike:
        return -x * darcsinh(x) ** 3

    def dddarcsinh(x: ArrayLike) -> ArrayLike:
        return _dddarc_h(x, darcsinh(x))

    return (arcsinh, darcsinh, ddarcsinh, dddarcsinh, None)[min(n, 4)]


def derivative_arctanh(n: int):
    def darctanh(x: ArrayLike) -> ArrayLike:
        return 1.0 / (1 - x**2)

    def ddarctanh(x: ArrayLike) -> ArrayLike:
        return 2 * x * darctanh(x) ** 2

    def dddarctanh(x: ArrayLike) -> ArrayLike:
        y = darctanh(x)
        return 2 * y**2 * (1 + 4 * x**2 * y)

    return (arctanh, darctanh, ddarctanh, dddarctanh, None)[min(n, 4)]


def derivative_exp(n: int) -> DerivativeFunc:
    return exp


def derivative_expm1(n: int) -> DerivativeFunc:
    return (expm1, exp)[min(n, 1)]


def derivative_exp2(n: int) -> DerivativeFunc:
    def dexp2(x: ArrayLike) -> ArrayLike:
        return exp2(x) * log(2) ** n

    return dexp2


def derivative_square(n: int) -> DerivativeFunc:
    def dsquare(x: ArrayLike) -> ArrayLike:
        return 2 * x

    def ddsquare(x: ArrayLike) -> ArrayLike:
        return 2 * np.ones_like(x)

    def dddsquare(x: ArrayLike) -> ArrayLike:
        return np.zeros_like(x)

    return (square, dsquare, ddsquare, dddsquare)[min(n, 3)]


def derivative_log1p(n: int) -> DerivativeFunc:
    def dlog1p(x: ArrayLike) -> ArrayLike:
        return (-1) ** (n + 1) * special.gamma(n) / (1 + x) ** n

    if n > 5:
        return None
    return (log1p, dlog1p)[min(n, 1)]


def _derivative_loga(n: int, a: float = 10) -> DerivativeFunc:
    if n > 4:
        return None
    dlog = derivative_log(n)

    def dlog_a(x: ArrayLike) -> ArrayLike:
        return dlog(x) / log(a)

    return dlog_a


def derivative_log2(n: int) -> DerivativeFunc:
    return _derivative_loga(n, a=2)


def derivative_log10(n: int) -> DerivativeFunc:
    return _derivative_loga(n, a=10)


def derivative_log(n: int) -> DerivativeFunc:
    if n > 4:
        return None

    def dlog(x: ArrayLike) -> ArrayLike:
        return (-1) ** (n + 1) * special.gamma(n) / x**n

    return (log, dlog)[min(n, 1)]


def derivative_sqrt(n: int) -> DerivativeFunc:
    fact = 0.5 * (-1) ** (n + 1)
    for k in np.arange(0.5, n - 1):
        fact *= k

    def dsqrt(x):
        sx = sqrt(x)
        return fact / sx ** (2 * n - 1)

    if n > 5:
        return None
    return (sqrt, dsqrt)[min(n, 1)]


def derivative_inv(n: int) -> DerivativeFunc:
    def inv(x: ArrayLike) -> ArrayLike:
        return 1.0 / x

    def dinv(x: ArrayLike) -> ArrayLike:
        return (-1) ** n * special.gamma(n) / x ** (n + 1)

    return (inv, dinv)[min(n, 1)]


def get_function(fun_name: str, n: int = 1) -> FunctionPair | tuple[str, ...]:
    f_dic = {
        "cosh": derivative_cosh,
        "cos": derivative_cos,
        "sin": derivative_sin,
        "sinh": derivative_sinh,
        "tan": derivative_tan,
        "tanh": derivative_tanh,
        "arccosh": derivative_arccosh,
        "arcsinh": derivative_arcsinh,
        "arctanh": derivative_arctanh,
        "arccos": derivative_arccos,
        "arcsin": derivative_arcsin,
        "arctan": derivative_arctan,
        "exp": derivative_exp,
        "expm1": derivative_expm1,
        "exp2": derivative_exp2,
        "log1p": derivative_log1p,
        "log2": derivative_log2,
        "log10": derivative_log10,
        "log": derivative_log,
        "sqrt": derivative_sqrt,
        "square": derivative_square,
        "inv": derivative_inv,
    }
    if fun_name == "all":
        return tuple(f_dic)

    funs = f_dic.get(fun_name)
    return funs(0), funs(n)
