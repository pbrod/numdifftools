import unittest
import numpy as np
import numdifftools.core as nd
from numpy.testing import assert_array_almost_equal
from numdifftools.example_functions import function_names, get_function


class TestExampleFunctions(unittest.TestCase):
    @staticmethod
    def test_high_order_derivative():
        x = 0.5
        methods = ['complex', 'central', 'forward', 'backward']
        for name in function_names:
            for n in range(2, 11):
                f, true_df = get_function(name, n=n)
                if true_df is None:
                    continue
                for method in methods:
                    if n > 7 and method not in ['complex']:
                        continue
                    val = nd.Derivative(f, method=method, n=n)(x)
                    print(name, method, n)
                    tval = true_df(x)
                    assert_array_almost_equal(val, tval, decimal=2)


if __name__ == '__main__':
    unittest.main()
