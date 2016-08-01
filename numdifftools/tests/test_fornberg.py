from __future__ import print_function
import unittest
from numdifftools.fornberg import fornberg_weights, derivative
import numpy as np
from numpy.testing import assert_array_almost_equal
from numdifftools.example_functions import function_names, get_function


class TestExampleFunctions(unittest.TestCase):

    @staticmethod
    def test_high_order_derivative():
        methods = ['complex', 'central', 'forward', 'backward']
        small_radius = ['sqrt', 'log', 'log2', 'log10', 'arccos', 'log1p',
                        'arcsin', 'arctan', 'arcsinh', 'tan', 'tanh',
                        'arctanh', 'arccosh']
        r = 0.0061
        n_max = 20
        for name in function_names + ['arctanh', 'arccosh']:
            f, true_df = get_function(name, n=1)
            x = 0.5 if name != 'arccosh' else 1.5
            # r = 0.0061 if name != 'log' else 0.061
            vals, info = derivative(f, x, r=r, n=n_max, full_output=True, step_ratio=1.6)
            for n in range(1, n_max):
                f, true_df = get_function(name, n=n)
                if true_df is None:
                    continue

                tval = true_df(x)
                dm = int(-np.log10(info.error_estimate[n] + 1e-16)) - 1
                print(n, name, info.iterations, dm)
                assert_array_almost_equal(vals[n], tval, decimal=max(dm, 6))
        # assert(False)

class TestFornbergWeights(unittest.TestCase):

    @staticmethod
    def test_weights():
        x = np.r_[-1, 0, 1]
        xbar = 0
        k = 1
        weights = fornberg_weights(x, xbar, k)
        np.testing.assert_allclose(weights, [-.5, 0, .5])
