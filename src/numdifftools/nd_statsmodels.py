"""
Numdifftools.nd_statsmodels
===========================
This module provides an easy to use interface to derivatives calculated with
statsmodels.numdiff.
"""

from __future__ import absolute_import, division, print_function
from functools import partial
import warnings

import numpy as np

try:
    from statsmodels.tools.numdiff import (  # approx_fprime,
        # approx_fprime_cs,
        # approx_hess, # same as approx_hess3
        approx_hess1,
        approx_hess2,
        approx_hess3,
        approx_hess_cs,
        _get_epsilon)
except ImportError:
    approx_hess1=approx_hess2=approx_hess3=approx_hess_cs=_get_epsilon=None


_EPS = np.finfo(float).eps


def approx_fprime(x, f, epsilon=None, args=(), kwargs=None, centered=True):
    """
    Gradient of function, or Jacobian if function fun returns 1d array

    Parameters
    ----------
    x : array
        parameters at which the derivative is evaluated
    fun : function
        `fun(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. This is _EPS**(1/2)*x for
        `centered` == False and _EPS**(1/3)*x for `centered` == True.
    args : tuple
        Tuple of additional arguments for function `fun`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `fun`.
    centered : bool
        Whether central difference should be returned. If not, does forward
        differencing.

    Returns
    -------
    grad : array
        gradient or Jacobian

    Notes
    -----
    If fun returns a 1d array, it returns a Jacobian. If a 2d array is returned
    by fun (e.g., with a value for each observation), it returns a 3d array
    with the Jacobian of each observation with shape xk x nobs x xk. I.e.,
    the Jacobian of the first observation would be [:, 0, :]

    """
    kwargs = {} if kwargs is None else kwargs
    x = np.atleast_1d(x)  # .ravel()
    n = len(x)
    f0 = f(*(x,) + args, **kwargs)
    dim = np.atleast_1d(f0).shape  # it could be a scalar
    grad = np.zeros((n,) + dim, float)
    ei = np.zeros(np.shape(x), float)
    if not centered:
        epsilon = _get_epsilon(x, 2, epsilon, n)
        for k in range(n):
            ei[k] = epsilon[k]
            grad[k, :] = (f(*(x + ei,) + args, **kwargs) - f0) / epsilon[k]
            ei[k] = 0.0
    else:
        epsilon = _get_epsilon(x, 3, epsilon, n) / 2.
        for k in range(n):
            ei[k] = epsilon[k]
            grad[k, :] = (f(*(x + ei,) + args, **kwargs) -
                          f(*(x - ei,) + args, **kwargs)) / (2 * epsilon[k])
            ei[k] = 0.0
    axes = list(range(grad.ndim))
    axes[:2] = axes[1::-1]
    return np.transpose(grad, axes=axes)


def _approx_fprime_backward(x, f, epsilon=None, args=(), kwargs=None):
    x = np.atleast_1d(x)  # .ravel()
    n = len(x)
    epsilon = - np.abs(_get_epsilon(x, 2, epsilon, n))
    return approx_fprime(x, f, epsilon, args, kwargs, centered=False)


def approx_fprime_cs(x, f, epsilon=None, args=(), kwargs=None):
    '''
    Calculate gradient or Jacobian with complex step derivative approximation

    Parameters
    ----------
    x : array
        parameters at which the derivative is evaluated
    f : function
        `f(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. Optimal step-size is
        EPS*x. See note.
    args : tuple
        Tuple of additional arguments for function `f`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `f`.

    Returns
    -------
    partials : ndarray
       array of partial derivatives, Gradient or Jacobian

    Notes
    -----
    The complex-step derivative has truncation error O(epsilon**2), so
    truncation error can be eliminated by choosing epsilon to be very small.
    The complex-step derivative avoids the problem of round-off error with
    small epsilon because there is no subtraction.
    '''
    # From Guilherme P. de Freitas, numpy mailing list
    # May 04 2010 thread "Improvement of performance"
    # http://mail.scipy.org/pipermail/numpy-discussion/2010-May/050250.html
    kwargs = {} if kwargs is None else kwargs
    x = np.atleast_1d(x)  # .ravel()
    n = len(x)
    epsilon = _get_epsilon(x, 1, epsilon, n)
    increments = np.identity(n) * 1j * epsilon
    # TODO: see if this can be vectorized, but usually dim is small
    partials = [f(x+ih, *args, **kwargs).imag / epsilon[i]
                for i, ih in enumerate(increments)]
    axes = list(range(partials[0].ndim+1))
    axes[:2] = axes[1::-1]
    return np.transpose(partials, axes=axes)


def _approx_hess1_backward(x, f, epsilon=None, args=(), kwargs=None):
    n = len(x)
    kwargs = {} if kwargs is None else kwargs
    epsilon = - np.abs(_get_epsilon(x, 3, epsilon, n))
    return approx_hess1(x, f, epsilon, args, kwargs)


class _Common(object):
    def __init__(self, fun, step=None, method='central', order=None):
        self.fun = fun
        self.step = step
        self.method = method
        self.order = order

    _callables = {}
    n = property(fget=lambda cls: 1)

    @property
    def order(self):
        return dict(forward=1, backward=1).get(self.method, 2)

    @order.setter
    def order(self, order):
        if order is None:
            return
        valid_order = self.order
        if order != valid_order:
            msg = 'Can not change order to {}! The only valid order is {} for method={}.'
            warnings.warn(msg.format(order, valid_order, self.method))

    @property
    def method(self):
        return self._method  # pylint: disable=no-member

    @method.setter
    def method(self, method):
        self._metod = method
        callable_ = self._callables.get(method)
        if callable_:
            self._derivative_nonzero_order = callable_
        else:
            warnings.warn('{} is an illegal method! Setting method="central"'.format(method))
            self.method = 'central'

    def __call__(self, x, *args, **kwds):
        return self._derivative_nonzero_order(np.atleast_1d(x), self.fun, self.step, args, kwds)


class Hessian(_Common):
    """
    Calculate Hessian with finite difference approximation

    Parameters
    ----------
    fun : function
       function of one array fun(x, `*args`, `**kwds`)
    step : float, optional
        Stepsize, if None, optimal stepsize is used, i.e.,
        x * _EPS**(1/3) for method==`forward`, `complex`  or `central2`
        x * _EPS**(1/4) for method==`central`.
    method : {'central', 'complex', 'forward', 'backward'}
        defines the method used in the approximation.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_statsmodels as nd

    # Rosenbrock function, minimized at [1,1]

    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hfun = nd.Hessian(rosen)
    >>> h = Hfun([1, 1])
    >>> np.allclose(h, [[ 842., -420.], [-420.,  210.]])
    True

    # cos(x-y), at (0,0)

    >>> cos = np.cos
    >>> fun = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = nd.Hessian(fun)
    >>> h2 = Hfun2([0, 0])
    >>> np.allclose(h2, [[-1.,  1.], [ 1., -1.]])
    True

    See also
    --------
    Jacobian, Gradient
    """
    n = property(fget=lambda cls: 2)

    _callables = dict(complex=approx_hess_cs,
                      forward=approx_hess1,
                      backward=_approx_hess1_backward,
                      central=approx_hess3,
                      central2=approx_hess2)


class Jacobian(_Common):
    """
    Calculate Jacobian with finite difference approximation

    Parameters
    ----------
    fun : function
       function of one array fun(x, `*args`, `**kwds`)
    step : float, optional
        Stepsize, if None, optimal stepsize is used, i.e.,
        x * _EPS for method==`complex`
        x * _EPS**(1/2) for method==`forward`
        x * _EPS**(1/3) for method==`central`.
    method : {'central', 'complex', 'forward', 'backward'}
        defines the method used in the approximation.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_statsmodels as nd

    #(nonlinear least squares)

    >>> xdata = np.arange(0,1,0.1)
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> np.allclose(fun([1, 2, 0.75]).shape, (10,))
    True
    >>> dfun = nd.Jacobian(fun)
    >>> np.allclose(dfun([1, 2, 0.75]), np.zeros((10,3)))
    True

    >>> fun2 = lambda x : x[0]*x[1]*x[2]**2
    >>> dfun2 = nd.Jacobian(fun2)
    >>> np.allclose(dfun2([1.,2.,3.]), [[18., 9., 12.]])
    True

    >>> fun3 = lambda x : np.vstack((x[0]*x[1]*x[2]**2, x[0]*x[1]*x[2]))
    >>> np.allclose(nd.Jacobian(fun3)([1., 2., 3.]), [[[18.], [9.], [12.]], [[6.], [3.], [2.]]])
    True
    >>> np.allclose(nd.Jacobian(fun3)([4., 5., 6.]),
    ...            [[[180.], [144.], [240.]], [[30.], [24.], [20.]]])
    True

    >>> np.allclose(nd.Jacobian(fun3)(np.array([[1.,2.,3.], [4., 5., 6.]]).T),
    ...            [[[  18.,  180.],
    ...              [   9.,  144.],
    ...              [  12.,  240.]],
    ...             [[   6.,   30.],
    ...              [   3.,   24.],
    ...              [   2.,   20.]]])
    True
    """
    _callables = dict(complex=approx_fprime_cs,
                      central=partial(approx_fprime, centered=True),
                      forward=partial(approx_fprime, centered=False),
                      backward=_approx_fprime_backward)


class Gradient(Jacobian):
    """
    Calculate Gradient with finite difference approximation

    Parameters
    ----------
    fun : function
       function of one array fun(x, `*args`, `**kwds`)
    step : float, optional
        Stepsize, if None, optimal stepsize is used, i.e.,
        x * _EPS for method==`complex`
        x * _EPS**(1/2) for method==`forward`
        x * _EPS**(1/3) for method==`central`.
    method : {'central', 'complex', 'forward', 'backward'}
        defines the method used in the approximation.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_statsmodels as nd
    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = nd.Gradient(fun)
    >>> np.allclose(dfun([1,2,3]), [ 2.,  4.,  6.])
    True

    # At [x,y] = [1,1], compute the numerical gradient
    # of the function sin(x-y) + y*exp(x)

    >>> sin = np.sin; exp = np.exp
    >>> z = lambda xy: sin(xy[0]-xy[1]) + xy[1]*exp(xy[0])
    >>> dz = nd.Gradient(z)
    >>> grad2 = dz([1, 1])
    >>> np.allclose(grad2, [ 3.71828183,  1.71828183])
    True

    # At the global minimizer (1,1) of the Rosenbrock function,
    # compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> rd = nd.Gradient(rosen)
    >>> grad3 = rd([1,1])
    >>> np.allclose(grad3,[0, 0])
    True

    See also
    --------
    Hessian, Jacobian
    """

    def __call__(self, x, *args, **kwds):
        return super(Gradient, self).__call__(np.atleast_1d(x).ravel(),
                                              *args, **kwds).squeeze()


if __name__ == '__main__':
    from numdifftools.testing import test_docstrings
    test_docstrings(__file__)
#     print(np.log(_EPS)/np.log(1e-6))
#     print(_EPS**(1./2.5))
