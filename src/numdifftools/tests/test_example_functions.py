import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal  # @UnresolvedImport

import numdifftools as nd
import numdifftools.nd_statsmodels as nds
from numdifftools.example_functions import function_names, get_function
try:
    import algopy
except ImportError:
    algopy = None
else:
    import numdifftools.nd_algopy as nda


pytestmark = pytest.mark.skipif(algopy is None, reason="algopy is not installed!")


class TestExampleFunctions(object):

    @staticmethod
    def test_high_order_derivative():
        x = 0.5
        min_dm = dict(complex=2, forward=2, backward=2, central=4)
        methods = ['complex', 'central', 'backward', 'forward']
        derivatives = [nd.Derivative]
        if nda is not None:
            derivatives.append(nda.Derivative)
        for i, derivative in enumerate(derivatives):
            for name in function_names:
                if i > 0 and name in ['arcsinh', 'exp2']:
                    continue
                for n in range(1, 11):
                    f, true_df = get_function(name, n=n)
                    if true_df is None:
                        continue
                    for method in methods[3 * i:]:
                        if i == 0 and n > 7 and method not in ['complex']:
                            continue
                        df = derivative(f, method=method, n=n, full_output=True)
                        val, info = df(x)
                        dm = max(int(-np.log10(info.error_estimate + 1e-16)) - 1,
                                 min_dm.get(method, 4))
                        print(i, name, method, n, dm)
                        tval = true_df(x)
                        assert_array_almost_equal(val, tval, decimal=dm)

    def test_first_order_derivative(self):
        x = 0.5
        methods = ['complex', 'central', 'backward', 'forward']
        derivatives = [nd.Derivative, nds.Gradient]

        if nda is not None:
            derivatives.append(nda.Derivative)

        for i, derivative in enumerate(derivatives):
            for name in function_names:
                if i > 1 and name in ['arcsinh', 'exp2']:
                    continue

                f, true_df = get_function(name, n=1)
                if true_df is None:
                    continue
                for method in methods[3 * (i > 1):]:

                    df = derivative(f, method=method)
                    val = df(x)
                    tval = true_df(x)
                    dm = 7
                    print(i, name, method, dm, np.abs(val - tval))
                    assert_array_almost_equal(val, tval, decimal=dm)
