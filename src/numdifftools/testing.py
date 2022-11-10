"""
Created on Apr 4, 2016

@author: pab
"""
from __future__ import absolute_import, print_function
import sys
import contextlib
import inspect
import numpy as np
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO


def rosen(x):
    """Rosenbrock function

    This is a non-convex function used as a performance test problem for
    optimization algorithms introduced by Howard H. Rosenbrock in 1960.[1]
    """
    x = np.atleast_1d(x)
    return (1 - x[0])**2 + 105. * (x[1] - x[0]**2)**2


def test_docstrings(name=''):
    # np.set_printoptions(precision=6)
    import doctest
    if not name:
        name = inspect.stack()[1][1]
    print('Testing docstrings in {}'.format(name))
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)


@contextlib.contextmanager
def capture_stdout_and_stderr():
    """
    Capture sys.stdout and sys.stderr

    Examples
    --------
    >>> from numdifftools.testing import capture_stdout_and_stderr
    >>> with capture_stdout_and_stderr() as out:
    ...    print('This is a test')
    >>> out[0].startswith('This is a test')
    True
    >>> out[1] == ''
    True
    """
    old_out = sys.stdout, sys.stderr
    out = [StringIO(), StringIO()]
    try:
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = old_out
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


if __name__ == '__main__':
    test_docstrings(__file__)
