"""
Created on 17. mai 2015

@author: pab
"""

from __future__ import annotations

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

from numdifftools._typing import (
    ArrayOrScalar,
    DerivativeFactory,
    FuncOrNone,
    FunctionPair,
    MathFunc,
)

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


def darcsin(x: ArrayOrScalar) -> ArrayOrScalar:
    return 1.0 / sqrt(1 - x**2)


def ddarcsin(x: ArrayOrScalar) -> ArrayOrScalar:
    return x * darcsin(x) ** 3


def dddarcsin(x: ArrayOrScalar) -> ArrayOrScalar:
    y = darcsin(x)
    return y**3 * (1 + 3 * (x * y) ** 2)


def darccos(x: ArrayOrScalar) -> ArrayOrScalar:
    return -darcsin(x)


def ddarccos(x: ArrayOrScalar) -> ArrayOrScalar:
    return -ddarcsin(x)


def dddarccos(x: ArrayOrScalar) -> ArrayOrScalar:
    return -dddarcsin(x)


def derivative_arcsin(n: int) -> FuncOrNone:
    return (arcsin, darcsin, ddarcsin, dddarcsin, None)[min(n, 4)]


def derivative_arccos(n: int) -> FuncOrNone:
    return (arccos, darccos, ddarccos, dddarccos, None)[min(n, 4)]


def derivative_arctan(n: int) -> FuncOrNone:
    def darctan(x: ArrayOrScalar) -> ArrayOrScalar:
        return 1.0 / (1 + x**2)

    def ddarctan(x: ArrayOrScalar) -> ArrayOrScalar:
        return -2 * x * darctan(x) ** 2

    def dddarctan(x: ArrayOrScalar) -> ArrayOrScalar:
        y = darctan(x)
        return 2 * (4.0 * x**2 * y - 1.0) * y**2

    def ddddarctan(x: ArrayOrScalar) -> ArrayOrScalar:
        y = darctan(x)
        return (1.0 - 2 * x**2 * y) * 24 * x * y**3

    return (arctan, darctan, ddarctan, dddarctan, ddddarctan, None)[min(n, 5)]


def derivative_sin(n: int) -> MathFunc:

    def dcos(x: ArrayOrScalar) -> ArrayOrScalar:
        return -sin(x)

    def ddcos(x: ArrayOrScalar) -> ArrayOrScalar:
        return -cos(x)

    return (sin, cos, dcos, ddcos)[n % 4]


def derivative_cos(n: int) -> MathFunc:
    return derivative_sin(n + 1)


def derivative_tan(n: int) -> FuncOrNone:
    def dtan(x: ArrayOrScalar) -> ArrayOrScalar:
        return 1.0 / np.cos(x) ** 2

    def ddtan(x: ArrayOrScalar) -> ArrayOrScalar:
        return 2 * tan(x) / cos(x) ** 2

    def dddtan(x: ArrayOrScalar) -> ArrayOrScalar:
        y = tan(x)
        return 2 * (y**2 + 1) * (3 * y**2 + 1)

    def ddddtan(x: ArrayOrScalar) -> ArrayOrScalar:
        y = tan(x)
        return 8 * y * (y**2 + 1) * (3 * y**2 + 2)

    return (tan, dtan, ddtan, dddtan, ddddtan, None)[min(n, 5)]


def derivative_sinh(n: int) -> MathFunc:
    return (sinh, cosh)[n % 2]


def derivative_cosh(n: int) -> MathFunc:
    return derivative_sinh(n + 1)


def derivative_tanh(n: int) -> FuncOrNone:
    def dtanh(x: ArrayOrScalar) -> ArrayOrScalar:
        return 1.0 / cosh(x) ** 2

    def ddtanh(x: ArrayOrScalar) -> ArrayOrScalar:
        return -2 * sinh(x) / cosh(x) ** 3

    def dddtanh(x: ArrayOrScalar) -> ArrayOrScalar:
        y = cosh(x)
        return 4 * (tanh(x) / y) ** 2 - 2.0 / y**4

    def ddddtanh(x: ArrayOrScalar) -> ArrayOrScalar:
        y = tanh(x)
        return 8 * y * (y**2 - 1) * (3 * y**2 - 2)

    return (tanh, dtanh, ddtanh, dddtanh, ddddtanh, None)[min(n, 5)]


def _dddarc_h(x: ArrayOrScalar, y: ArrayOrScalar) -> ArrayOrScalar:
    return (3 * (x * y) ** 2 - 1) * y**3


def derivative_arccosh(n: int) -> FuncOrNone:
    def darccosh(x: ArrayOrScalar) -> ArrayOrScalar:
        return 1.0 / sqrt(x**2 - 1)

    def ddarccosh(x: ArrayOrScalar) -> ArrayOrScalar:
        return -x * darccosh(x) ** 3

    def dddarccosh(x: ArrayOrScalar) -> ArrayOrScalar:
        return _dddarc_h(x, darccosh(x))

    return (arccosh, darccosh, ddarccosh, dddarccosh, None)[min(n, 4)]


def derivative_arcsinh(n: int) -> FuncOrNone:
    def darcsinh(x: ArrayOrScalar) -> ArrayOrScalar:
        return 1.0 / sqrt(1 + x**2)

    def ddarcsinh(x: ArrayOrScalar) -> ArrayOrScalar:
        return -x * darcsinh(x) ** 3

    def dddarcsinh(x: ArrayOrScalar) -> ArrayOrScalar:
        return _dddarc_h(x, darcsinh(x))

    return (arcsinh, darcsinh, ddarcsinh, dddarcsinh, None)[min(n, 4)]


def derivative_arctanh(n: int) -> FuncOrNone:
    def darctanh(x: ArrayOrScalar) -> ArrayOrScalar:
        return 1.0 / (1 - x**2)

    def ddarctanh(x: ArrayOrScalar) -> ArrayOrScalar:
        return 2 * x * darctanh(x) ** 2

    def dddarctanh(x: ArrayOrScalar) -> ArrayOrScalar:
        y = darctanh(x)
        return 2 * y**2 * (1 + 4 * x**2 * y)

    return (arctanh, darctanh, ddarctanh, dddarctanh, None)[min(n, 4)]


def derivative_exp(n: int) -> MathFunc:
    return exp


def derivative_expm1(n: int) -> MathFunc:
    return (expm1, exp)[min(n, 1)]


def derivative_exp2(n: int) -> MathFunc:
    def dexp2(x: ArrayOrScalar) -> ArrayOrScalar:
        return exp2(x) * log(2) ** n

    return dexp2


def derivative_square(n: int) -> FuncOrNone:
    def dsquare(x: ArrayOrScalar) -> ArrayOrScalar:
        return 2 * x

    def ddsquare(x: ArrayOrScalar) -> ArrayOrScalar:
        return 2 * np.ones_like(x)

    def dddsquare(x: ArrayOrScalar) -> ArrayOrScalar:
        return np.zeros_like(x)

    return (square, dsquare, ddsquare, dddsquare)[min(n, 3)]


def derivative_log1p(n: int) -> FuncOrNone:
    def dlog1p(x: ArrayOrScalar) -> ArrayOrScalar:
        return (-1) ** (n + 1) * special.gamma(n) / (1 + x) ** n

    if n > 5:
        return None
    return (log1p, dlog1p)[min(n, 1)]


def _derivative_loga(n: int, a: float = 10) -> FuncOrNone:

    dlog = derivative_log(n)
    if dlog is None:
        return None

    def dlog_a(x: ArrayOrScalar) -> ArrayOrScalar:
        return dlog(x) / log(a)

    return dlog_a


def derivative_log2(n: int) -> FuncOrNone:
    return _derivative_loga(n, a=2)


def derivative_log10(n: int) -> FuncOrNone:
    return _derivative_loga(n, a=10)


def derivative_log(n: int) -> FuncOrNone:
    if n > 4:
        return None

    def dlog(x: ArrayOrScalar) -> ArrayOrScalar:
        return (-1) ** (n + 1) * special.gamma(n) / x**n

    return (log, dlog)[min(n, 1)]


def derivative_sqrt(n: int) -> FuncOrNone:
    fact = 0.5 * (-1) ** (n + 1)
    for k in np.arange(0.5, n - 1):
        fact *= k

    def dsqrt(x: ArrayOrScalar) -> ArrayOrScalar:
        sx = sqrt(x)
        return fact / sx ** (2 * n - 1)

    if n > 5:
        return None
    return (sqrt, dsqrt)[min(n, 1)]


def derivative_inv(n: int) -> FuncOrNone:
    def inv(x: ArrayOrScalar) -> ArrayOrScalar:
        return 1.0 / x

    def dinv(x: ArrayOrScalar) -> ArrayOrScalar:
        return (-1) ** n * special.gamma(n) / x ** (n + 1)

    return (inv, dinv)[min(n, 1)]


FUN_DICT: dict[str, DerivativeFactory] = {
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


def get_function(fun_name: str, n: int = 1) -> FunctionPair:
    funs = FUN_DICT.get(fun_name)
    if funs is None:
        raise KeyError(f"Unknown function {fun_name}")
    return funs(0), funs(n)
