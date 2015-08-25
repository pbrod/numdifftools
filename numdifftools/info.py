"""
Features
========

Numdifftools is a suite of tools written in Python to solve automatic numerical
differentiation problems in one or more variables. Finite differences are used
in an adaptive manner, coupled with a Richardson extrapolation methodology to
provide a maximally accurate result. The user can configure many options like;
changing the order of the method or the extrapolation, even allowing the user
to specify whether `complex`, `multicomplex`, `central`, `forward` or
`backward` differences are used. The methods provided are:

*Derivative:*
    Computates the derivative of order 1 through 10 on any scalar function.

*Gradient:*
    Computes the gradient vector of a scalar function of one or more variables.

*Jacobian:*
    Computes the Jacobian matrix of a vector valued function of one or more
    variables.

*Hessian:*
    Computes the Hessian matrix of all 2nd partial derivatives of a scalar
    function of one or more variables.

*Hessdiag:*
    Computes only the diagonal elements of the Hessian matrix

All of these methods also produce error estimates on the result.

Numdifftools also provide an easy to use interface to derivatives calculated
with AlgoPy. Algopy stands for Algorithmic Differentiation in Python.
The purpose of AlgoPy is the evaluation of higher-order derivatives in the
`forward` and `reverse` mode of Algorithmic Differentiation (AD) of functions
that are implemented as Python programs.

Documentation is at: http://numdifftools.readthedocs.org/

Code and issue tracker is at https://github.com/pbrod/numdifftools.

Latest stable release is at http://pypi.python.org/pypi/Numdifftools.

To test if the toolbox is working paste the following in an interactive
python session::

   import numdifftools as nd
   nd.test(coverage=True, doctests=True)

Examples
--------
Compute 1'st and 2'nd derivative of exp(x), at x == 1::

    >>> import numpy as np
    >>> import numdifftools as nd
    >>> fd = nd.Derivative(np.exp)        # 1'st derivative
    >>> fdd = nd.Derivative(np.exp, n=2)  # 2'nd derivative
    >>> np.allclose(fd(1), 2.7182818284590424)
    True
    >>> np.allclose(fdd(1), 2.7182818284590424)
    True

Nonlinear least squares::

    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> Jfun = nd.Jacobian(fun)
    >>> np.allclose(np.abs(Jfun([1,2,0.75])), 0) # should be numerically zero
    True

Compute gradient of sum(x**2)::

    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = nd.Gradient(fun)
    >>> dfun([1,2,3])
    array([ 2.,  4.,  6.])

Compute the same with AlgoPy

    >>> import numdifftools.nd_algopy as nda
    >>> fd = nda.Derivative(np.exp)        # 1'st derivative
    >>> fdd = nda.Derivative(np.exp, n=2)  # 2'nd derivative
    >>> np.allclose(fd(1), 2.7182818284590424)
    True
    >>> np.allclose(fdd(1), 2.7182818284590424)
    True

Nonlinear least squares::

    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> Jfun = nda.Jacobian(fun, method='reverse')
    >>> np.allclose(np.abs(Jfun([1,2,0.75])), 0) # should be numerically zero
    True

Compute gradient of sum(x**2)::

    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = nda.Gradient(fun)
    >>> dfun([1,2,3])
    array([ 2.,  4.,  6.])

See also
--------
scipy.misc.derivative

"""


def test_docstrings():
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()
