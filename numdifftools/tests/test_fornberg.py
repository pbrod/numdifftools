from __future__ import print_function
import unittest
from numdifftools.fornberg import fornberg_weights, derivative
import numpy as np
from numpy.testing import assert_array_almost_equal
from numdifftools.example_functions import function_names, get_function


class TestExampleFunctions(unittest.TestCase):
    @staticmethod
    def test_high_order_derivative():
        x = 0.5
        methods = ['complex', 'central', 'forward', 'backward']
        n=20
        for name in function_names:
            f, true_df = get_function(name, n=1)
            r = 0.0061 if name in ['sqrt', 'log', 'log2', 'log10', 'arccos'] else 0.61
            vals, info = derivative(f, x, n=n, r=r, full_output=True)
            for n in range(1, n):
                f, true_df = get_function(name, n=n)
                if true_df is None:
                    continue

                tval = true_df(x)
                dm = int(-np.log10(info.error_estimate[n]))-1
                print(name, n, dm)
                assert_array_almost_equal(vals[n], tval, decimal=max(dm, 6))


class TestFornbergWeights(unittest.TestCase):

    @staticmethod
    def test_weights():
        x = np.r_[-1, 0, 1]
        xbar = 0
        k = 1
        weights = fornberg_weights(x, xbar, k)
        np.testing.assert_allclose(weights, [-.5, 0, .5])
