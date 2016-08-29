import unittest
import numpy as np
import numdifftools.core as nd
import numdifftools.nd_algopy as nda
from numpy.testing import assert_array_almost_equal
from numdifftools.example_functions import function_names, get_function


class TestExampleFunctions(unittest.TestCase):
    @staticmethod
    def test_high_order_derivative():
        x = 0.5
        min_dm = dict(complex=2, forward=2, backward=2, central=4)
        methods = [ 'complex', 'central',  'backward', 'forward']

        for i, derivative in enumerate([nd.Derivative, nda.Derivative]):
            for name in function_names:
                if i>0 and name in ['arcsinh', 'exp2']:
                    continue
                for n in range(2, 11):
                    f, true_df = get_function(name, n=n)
                    if true_df is None:
                        continue
                    for method in methods[3*i:]:
                        if i==0 and n > 7 and method not in ['complex']:
                            continue
                        df = derivative(f, method=method, n=n, full_output=True)
                        val, info = df(x)
                        dm = max(int(-np.log10(info.error_estimate + 1e-16))-1,
                                 min_dm.get(method, 4))
                        print(i, name, method, n, dm)
                        tval = true_df(x)
                        assert_array_almost_equal(val, tval, decimal=dm)


if __name__ == '__main__':
    unittest.main()
