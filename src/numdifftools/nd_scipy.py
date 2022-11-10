from __future__ import absolute_import, division, print_function
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import approx_fprime
import numpy as np


class _Common(object):
    def __init__(self, fun, step=None, method='central', order=2,
                 bounds=(-np.inf, np.inf), sparsity=None):
        self.fun = fun
        self.step = step
        self.method = method
        self.bounds = bounds
        self.sparsity = sparsity


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
    method : {'central', 'complex', 'forward'}
        defines the method used in the approximation.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_scipy as nd

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

    TODO: The following does not work:
    der3 = nd.Jacobian(fun3)([1., 2., 3.])
    np.allclose(der3,
    ...            [[18., 9., 12.], [6., 3., 2.]])
    True
    np.allclose(nd.Jacobian(fun3)([4., 5., 6.]),
    ...            [[180., 144., 240.], [30., 24., 20.]])
    True

    np.allclose(nd.Jacobian(fun3)(np.array([[1.,2.,3.], [4., 5., 6.]]).T),
    ...            [[[  18.,  180.],
    ...              [   9.,  144.],
    ...              [  12.,  240.]],
    ...             [[   6.,   30.],
    ...              [   3.,   24.],
    ...              [   2.,   20.]]])
    True
    """

    def __call__(self, x, *args, **kwds):
        x = np.atleast_1d(x)
        method = dict(complex='cs', central='3-point', forward='2-point',
                      backward='2-point')[self.method]
        options = dict(method=method, rel_step=self.step, args=args,
                       kwargs=kwds, bounds=self.bounds, sparsity=self.sparsity)

        grad = approx_derivative(self.fun, x, **options)

        return grad


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
    method : {'central', 'complex', 'forward'}
        defines the method used in the approximation.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_scipy as nd
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
    >>> np.allclose(grad3,[0, 0], atol=1e-7)
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
