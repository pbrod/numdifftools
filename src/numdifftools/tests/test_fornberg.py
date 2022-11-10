from __future__ import absolute_import, print_function

import numpy as np
from numpy.testing import assert_allclose  # @UnresolvedImport

from hypothesis import given, note, settings, strategies as st  #, reproduce_failure

from numdifftools.example_functions import function_names, get_function
from numdifftools.fornberg import (fd_weights, fd_weights_all, derivative,
                                   fd_derivative,
                                   CENTRAL_WEIGHTS_AND_POINTS)

# @reproduce_failure('5.36.1', b'AAJRcYUpZHQ=')
# @reproduce_failure('4.32.2', b'AAJRcYUpZHQ=')
@settings(deadline=800.0)
@given(st.floats(min_value=1e-1, max_value=0.98))
def test_high_order_derivative(x):
    #     small_radius = ['sqrt', 'log', 'log2', 'log10', 'arccos', 'log1p',
    #                     'arcsin', 'arctan', 'arcsinh', 'tan', 'tanh',
    #                     'arctanh', 'arccosh']
    r = 0.0059
    n_max = 20
    y = x
    for name in function_names + ['arccosh', 'arctanh']:
        f, true_df = get_function(name, n=1)
        if name == 'arccosh':
            y = y + 1

        vals, info = derivative(f, y, r=r, n=n_max, full_output=True, step_ratio=1.6)
        for n in range(1, n_max):
            f, true_df = get_function(name, n=n)
            if true_df is None:
                continue

            tval = true_df(y)

            aerr0 = info.error_estimate[n] + 1e-15
            aerr = min(aerr0, max(np.abs(tval) * 1e-6, 1e-8))
            try:
                assert_allclose(np.real(vals[n]), tval, rtol=1e-6, atol=aerr)
            except AssertionError as error:
                print(n, name, y, vals[n], tval, info.iterations, aerr0, aerr)
                note("{}, {}, {}, {}, {}, {}, {}, {}".format(
                    n, name, y, vals[n], tval, info.iterations, aerr0, aerr))
                raise error


def test_all_weights():
    w = fd_weights_all(range(-2, 3), n=4)
    print(w)
    true_w = [[0., 0., 1., 0., 0.],
              [0.0833333333333, -0.6666666666667, 0., 0.6666666666667,
               -0.0833333333333],
              [-0.0833333333333, 1.333333333333, -2.5, 1.333333333333,
               -0.083333333333],
              [-0.5, 1., 0., -1., 0.5],
              [1., -4., 6., -4., 1.]]
    assert_allclose(w, true_w, atol=1e-12)


def test_weights():
    for name in CENTRAL_WEIGHTS_AND_POINTS:
        # print(name)
        n, m = name
        w, x = CENTRAL_WEIGHTS_AND_POINTS[name]
        assert len(w) == m
        weights = fd_weights(np.array(x, dtype=float), 0.0, n=n)
        assert_allclose(weights, w, atol=1e-15)


def test_fd_derivative():
    x = np.linspace(-1, 1, 25)
    fx = np.exp(x)
    for n in range(1, 7):
        df = fd_derivative(fx, x, n=n)
        m = n // 2 + 2
        assert_allclose(df[m:-m], fx[m:-m], atol=1e-5)
        assert_allclose(df[-m:], fx[-m:], atol=1e-4)
        assert_allclose(df[:m], fx[:m], atol=1e-4)


class ExampleFunctions(object):

    @staticmethod
    def fun0(z):
        return np.exp(z)

    @staticmethod
    def fun1(z):
        return np.exp(z) / (np.sin(z) ** 3 + np.cos(z) ** 3)

    @staticmethod
    def fun2(z):
        return np.exp(1.0j * z)

    @staticmethod
    def fun3(z):
        return z ** 6

    @staticmethod
    def fun4(z):
        return z * (0.5 + 1. / np.expm1(z))

    @staticmethod
    def fun5(z):
        return np.tan(z)

    @staticmethod
    def fun6(z):
        return 1.0j + z + 1.0j * z ** 2

    @staticmethod
    def fun7(z):
        return 1.0 / (1.0 - z)

    @staticmethod
    def fun8(z):
        return (1 + z) ** 10 * np.log1p(z)

    @staticmethod
    def fun9(z):
        return 10 * 5 + 1. / (1 - z)

    @staticmethod
    def fun10(z):
        return 1. / (1 - z)

    @staticmethod
    def fun11(z):
        return np.sqrt(z)

    @staticmethod
    def fun12(z):
        return np.arcsinh(z)

    @staticmethod
    def fun13(z):
        return np.cos(z)

    @staticmethod
    def fun14(z):
        return np.log1p(z)


def test_low_order_derivative_on_example_functions():
    for j in range(15):
        fun = getattr(ExampleFunctions, 'fun{}'.format(j))
        der, info = derivative(fun, z0=0., r=0.06, n=10, max_iter=30,
                               full_output=True, step_ratio=1.6)
        print(info)
        print('answer:')
        msg = '{0:3d}: {1:24.18f} + {2:24.18f}j ({3:g})'
        print(info.function_count)
        for i, der_i in enumerate(der):
            err = info.error_estimate[i]
            print(msg.format(i, der_i.real, der_i.imag, err))
