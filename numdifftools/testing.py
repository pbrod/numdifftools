'''
Created on Apr 4, 2016

@author: pab
'''
import inspect
import numpy as np


def rosen(x):
    """Rosenbrock function

    This is a non-convex function used as a performance test problem for
    optimization algorithms introduced by Howard H. Rosenbrock in 1960.[1]
    """
    x = np.atleast_1d(x)
    return (1 - x[0])**2 + 105. * (x[1] - x[0]**2)**2


def test_docstrings():
    # np.set_printoptions(precision=6)
    import doctest
    print('Testing docstrings in %s' % inspect.stack()[1][1])
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE|doctest.ELLIPSIS)
