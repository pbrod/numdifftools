"""
Created on 17. mai 2015

@author: pab
"""
from __future__ import division
import numpy as np
import scipy.special as special
from numpy import (cos, sin, tan, cosh, sinh, tanh,
                   arccosh, arcsinh, arctanh,
                   exp, expm1, exp2, square, sqrt,
                   log, log1p, log10, log2,
                   arccos, arcsin, arctan)
function_names = ['cos', 'sin', 'tan',
                  'cosh', 'sinh', 'tanh',
                  'arcsinh',
                  'exp', 'expm1', 'exp2', 'square',
                  'sqrt',
                  'log', 'log1p', 'log10', 'log2',
                  'arccos', 'arcsin', 'arctan', ]


def darcsin(x):
    return 1. / sqrt(1 - x**2)


def ddarcsin(x):
    return x * darcsin(x)**3


def dddarcsin(x):
    y = darcsin(x)
    return y**3 * (1 + 3 * (x * y) ** 2)


def darccos(x):
    return - darcsin(x)


def ddarccos(x):
    return - ddarcsin(x)


def dddarccos(x):
    return - dddarcsin(x)


def derivative_arcsin(n):
    return (arcsin, darcsin, ddarcsin, dddarcsin, None)[min(n, 4)]


def derivative_arccos(n):
    return (arccos, darccos, ddarccos, dddarccos, None)[min(n, 4)]


def derivative_arctan(n):
    def darctan(x):
        return 1. / (1 + x**2)

    def ddarctan(x):
        return -2 * x * darctan(x)**2

    def dddarctan(x):
        y = darctan(x)
        return 2 * (4.0 * x**2 * y - 1.0) * y**2

    def ddddarctan(x):
        y = darctan(x)
        return (1.0 - 2 * x**2 * y) * 24 * x * y**3
    return (arctan, darctan, ddarctan, dddarctan, ddddarctan, None)[min(n, 5)]


def derivative_sin(n):
    def dcos(x):
        return -sin(x)

    def ddcos(x):
        return -cos(x)
    return (sin, cos, dcos, ddcos)[n % 4]


def derivative_cos(n):
    return derivative_sin(n + 1)


def derivative_tan(n):
    def dtan(x):
        return 1. / np.cos(x)**2

    def ddtan(x):
        return 2 * tan(x) / cos(x)**2

    def dddtan(x):
        y = tan(x)
        return 2 * (y**2 + 1) * (3 * y**2 + 1)

    def ddddtan(x):
        y = tan(x)
        return 8 * y * (y**2 + 1) * (3 * y**2 + 2)
    return (tan, dtan, ddtan, dddtan, ddddtan, None)[min(n, 5)]


def derivative_sinh(n):
    return (sinh, cosh)[n % 2]


def derivative_cosh(n):
    return derivative_sinh(n + 1)


def derivative_tanh(n):
    def dtanh(x):
        return 1. / cosh(x) ** 2

    def ddtanh(x):
        return -2 * sinh(x) / cosh(x) ** 3

    def dddtanh(x):
        y = cosh(x)
        return 4 * (tanh(x) / y)**2 - 2. / y**4

    def ddddtanh(x):
        y = tanh(x)
        return 8 * y * (y**2 - 1) * (3 * y**2 - 2)
    return (tanh, dtanh, ddtanh, dddtanh, ddddtanh, None)[min(n, 5)]


def _dddarc_h(x, y):
    return (3 * (x * y) ** 2 - 1) * y ** 3


def derivative_arccosh(n):
    def darccosh(x):
        return 1.0 / sqrt(x**2 - 1)

    def ddarccosh(x):
        return -x * darccosh(x)**3

    def dddarccosh(x):
        return _dddarc_h(x, darccosh(x))
    return (arccosh, darccosh, ddarccosh, dddarccosh, None)[min(n, 4)]


def derivative_arcsinh(n):
    def darcsinh(x):
        return 1.0 / sqrt(1 + x**2)

    def ddarcsinh(x):
        return -x * darcsinh(x)**3

    def dddarcsinh(x):
        return _dddarc_h(x, darcsinh(x))
    return (arcsinh, darcsinh, ddarcsinh, dddarcsinh, None)[min(n, 4)]


def derivative_arctanh(n):
    def darctanh(x):
        return 1.0 / (1 - x**2)

    def ddarctanh(x):
        return 2 * x * darctanh(x)**2

    def dddarctanh(x):
        y = darctanh(x)
        return 2 * y**2 * (1 + 4 * x**2 * y)
    return (arctanh, darctanh, ddarctanh, dddarctanh, None)[min(n, 4)]


def derivative_exp(n):
    return exp


def derivative_expm1(n):
    return (expm1, exp)[min(n, 1)]


def derivative_exp2(n):
    def dexp2(x):
        return exp2(x) * log(2)**n
    return dexp2


def derivative_square(n):
    def dsquare(x):
        return 2 * x

    def ddsquare(x):
        return 2 * np.ones_like(x)

    def dddsquare(x):
        return np.zeros_like(x)
    return (square, dsquare, ddsquare, dddsquare)[min(n, 3)]


def derivative_log1p(n):
    def dlog1p(x):
        return (-1)**(n + 1) * special.gamma(n) / (1 + x)**n
    if n > 5:
        return None
    return (log1p, dlog1p)[min(n, 1)]


def _derivative_loga(n, a=10):

    if n > 4:
        return None
    dlog = derivative_log(n)

    def dlog_a(x):
        return dlog(x) / log(a)
    return dlog_a


def derivative_log2(n):
    return _derivative_loga(n, a=2)


def derivative_log10(n):
    return _derivative_loga(n, a=10)


def derivative_log(n):
    if n > 4:
        return None

    def dlog(x):
        return (-1)**(n + 1) * special.gamma(n) / x**n
    return (log, dlog)[min(n, 1)]


def derivative_sqrt(n):
    fact = 0.5 * (-1)**(n + 1)
    for k in np.arange(.5, n - 1):
        fact *= k

    def dsqrt(x):
        sx = sqrt(x)
        return fact / sx**(2 * n - 1)
    if n > 5:
        return None
    return (sqrt, dsqrt)[min(n, 1)]


def derivative_inv(n):
    def inv(x):
        return 1. / x

    def dinv(x):
        return (-1)**n * special.gamma(n) / x**(n + 1)
    return (inv, dinv)[min(n, 1)]


def get_function(fun_name, n=1):

    f_dic = dict(cosh=derivative_cosh,
                 cos=derivative_cos,
                 sin=derivative_sin,
                 sinh=derivative_sinh,
                 tan=derivative_tan,
                 tanh=derivative_tanh,
                 arccosh=derivative_arccosh,
                 arcsinh=derivative_arcsinh,
                 arctanh=derivative_arctanh,
                 arccos=derivative_arccos,
                 arcsin=derivative_arcsin,
                 arctan=derivative_arctan,
                 exp=derivative_exp,
                 expm1=derivative_expm1,
                 exp2=derivative_exp2,
                 log1p=derivative_log1p,
                 log2=derivative_log2,
                 log10=derivative_log10,
                 log=derivative_log,
                 sqrt=derivative_sqrt,
                 square=derivative_square,
                 inv=derivative_inv)
    if fun_name == 'all':
        return f_dic.keys()

    funs = f_dic.get(fun_name)
    return funs(0), funs(n)

