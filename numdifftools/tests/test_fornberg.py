from __future__ import print_function
import unittest
from numdifftools.fornberg import (fd_weights, fd_weights_all, derivative,
                                   fd_derivative,
                                   CENTRAL_WEIGHTS_AND_POINTS)
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from numdifftools.example_functions import function_names, get_function
from hypothesis import given, example, note, strategies as st

class TestExampleFunctions(unittest.TestCase):

    @staticmethod
    @given(st.floats(min_value=1e-1, max_value=0.98))
    def test_high_order_derivative(x):
        small_radius = ['sqrt', 'log', 'log2', 'log10', 'arccos', 'log1p',
                        'arcsin', 'arctan', 'arcsinh', 'tan', 'tanh',
                        'arctanh', 'arccosh']
        r = 0.0061
        n_max = 20
        y = x
        for name in function_names + ['arccosh', 'arctanh']:
            f, true_df = get_function(name, n=1)
            if name == 'arccosh':
                y = y + 1

            vals, info = derivative(f, y, r=r, n=n_max, full_output=True,
                                    step_ratio=1.6)
            for n in range(1, n_max):
                f, true_df = get_function(name, n=n)
                if true_df is None:
                    continue

                tval = true_df(y)

                aerr0 = info.error_estimate[n] + 1e-15
                aerr = min(aerr0, max(np.abs(tval)*1e-6, 1e-8))
                print(n, name, y, vals[n], tval, info.iterations, aerr0, aerr)
                note("{}, {}, {}, {}, {}, {}, {}, {}".format(
                    n, name, y, vals[n], tval, info.iterations,
                               aerr0, aerr))
                assert_allclose(np.real(vals[n]), tval, rtol=1e-6, atol=aerr)


class TestFornberg(unittest.TestCase):
    @staticmethod
    def test_all_weights():
        w = fd_weights_all(range(-2,3), n=4)
        print(w)
        true_w = [[0., 0., 1., 0., 0. ],
                  [ 0.0833333333333, -0.6666666666667, 0., 0.6666666666667,
                   -0.0833333333333],
                  [-0.0833333333333, 1.333333333333, -2.5,  1.333333333333,
                   -0.083333333333],
                  [-0.5, 1., 0., -1., 0.5 ],
                  [ 1., -4., 6., -4., 1.]]
        np.testing.assert_allclose(w, true_w, atol=1e-12)

    @staticmethod
    def test_weights():
        for name in CENTRAL_WEIGHTS_AND_POINTS:
            # print(name)
            n, m = name
            w, x = CENTRAL_WEIGHTS_AND_POINTS[name]

            weights = fd_weights(np.array(x, dtype=float), 0.0, n=n)
            np.testing.assert_allclose(weights, w, atol=1e-15)

    @staticmethod
    def test_fd_derivative():
        x = np.linspace(-1, 1, 25)
        h = np.diff(x).mean()
        fx = np.exp(x)
        for n in range(1, 7):
            df = fd_derivative(fx, x, n=n)
            m = n // 2 + 2
            np.testing.assert_allclose(df[m:-m], fx[m:-m], atol=1e-5)
            np.testing.assert_allclose(df[-m:], fx[-m:], atol=1e-4)
            np.testing.assert_allclose(df[:m], fx[:m], atol=1e-4)
