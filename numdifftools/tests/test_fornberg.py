from __future__ import print_function
import unittest
from numdifftools.fornberg import fornberg_weights, derivative
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
            #x = 0.5 if name != 'arccosh' else 1.5
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


class TestFornbergWeights(unittest.TestCase):

    @staticmethod
    def test_weights():
        x = np.r_[-1, 0, 1]
        xbar = 0
        k = 1
        weights = fornberg_weights(x, xbar, k)
        np.testing.assert_allclose(weights, [-.5, 0, .5])
