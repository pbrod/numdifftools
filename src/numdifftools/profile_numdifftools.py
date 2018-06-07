"""
This script profile different parts of numdifftools.

"""
from __future__ import print_function
import numpy as np
import numdifftools as nd
import numdifftools.nd_statsmodels as nds
from numdifftools.profiletools import do_profile
from numdifftools.example_functions import function_names, get_function
from numdifftools.run_benchmark import BenchmarkFunction

def main0():
    for n in 4, 8, 16, 32, 64, 96:
        f = BenchmarkFunction(n)

        step = nd.step_generators.one_step
        #step = None
        # cls = nd.Jacobian(f, step=step, method='central')
        cls = nd.Hessian(f, step=step, method='central')
        follow = [cls._derivative_nonzero_order,
                  cls._apply_fd_rule,
                  cls._get_finite_difference_rule,
                  cls._vstack,
                  cls._central_even]
#         cls = nds.Hessian(f, step=None, method='central')
#         follow = [cls._derivative_nonzero_order, ]

        x = 3 * np.ones(n)

#        val = cls(x)
        val = do_profile(follow=follow)(cls)(x)


def main():
    x = 0.5
    min_dm = dict(complex=2, forward=2, backward=2, central=4)
    methods = ['complex', 'central', 'backward', 'forward']

    # for i, Derivative in enumerate([nd.Derivative, nds.Gradient,
    # nda.Derivative]):
    i = 0
    # Derivative = nd.Derivative
    for name in function_names:
        if i > 1 and name in ['arcsinh', 'exp2']:
            continue

        f, true_df = get_function(name, n=1)
        if true_df is None:
            continue
        for method in methods[3 * (i > 1):]:

            df = Derivative(f, method=method)
            val = df(x)
            tval = true_df(x)
            dm = 7
            print(i, name, method, dm, np.abs(val - tval))
            # assert_array_almost_equal(val, tval, decimal=dm)


def profile_main():
    import cProfile
    import pstats
    cProfile.run("main0()", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    # s.strip_dirs()
    s.sort_stats("time").print_stats(20)

if __name__ == '__main__':
    main0()
    # profile_main()
    # from numdifftools.testing import test_docstrings
    # test_docstrings()
