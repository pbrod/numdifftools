# !/usr/bin/env python
"""numerical differentiation functions:

Derivative, Gradient, Jacobian, and Hessian

Author:      Per A. Brodtkorb
Created:     01.08.2008
Copyright:   (c) pab 2008
Licence:     New BSD
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
from numdifftools.extrapolation import Richardson, dea3  # @UnusedImport
from numdifftools.step_generators import MaxStepGenerator, MinStepGenerator
from numdifftools.limits import _Limit
from numdifftools.finite_difference import (LogRule,
                                            LogHessdiagRule,
                                            LogHessianRule,
                                            LogJacobianRule,
                                            )


__all__ = ('dea3', 'Derivative', 'Jacobian', 'Gradient', 'Hessian', 'Hessdiag',
           'MinStepGenerator', 'MaxStepGenerator', 'Richardson',
           'directionaldiff')
FD_RULES = {}
_SQRT_J = (1j + 1.0) / np.sqrt(2.0)  # = 1j**0.5


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


_CMN_DOC = """
    Calculate %(derivative)s with finite difference approximation

    Parameters
    ----------
    fun : function
       function of one array fun(x, `*args`, `**kwds`)
    step : float, array-like or StepGenerator object, optional
        Defines the spacing used in the approximation.
        Default is MinStepGenerator(**step_options) if method in in ['complex', 'multicomplex'],
        otherwise
            MaxStepGenerator(**step_options)
        The results are extrapolated if the StepGenerator generate more than 3
        steps.
    method : {'central', 'complex', 'multicomplex', 'forward', 'backward'}
        defines the method used in the approximation%(extra_parameter)s
    richardson_terms: scalar integer, default 2.
        number of terms used in the Richardson extrapolation.
    full_output : bool, optional
        If `full_output` is False, only the derivative is returned.
        If `full_output` is True, then (der, r) is returned `der` is the
        derivative, and `r` is a Results object.
    **step_options:
        options to pass on to the XXXStepGenerator used.

    Methods
    -------
    __call__ : callable with the following parameters:
        x : array_like
            value at which function derivative is evaluated
        args : tuple
            Arguments for function `fun`.
        kwds : dict
            Keyword arguments for function `fun`.
    %(returns)s
    Notes
    -----
    Complex methods are usually the most accurate provided the function to
    differentiate is analytic. The complex-step methods also requires fewer
    steps than the other methods and can work very close to the support of
    a function.
    The complex-step derivative has truncation error O(steps**2) for `n=1` and
    O(steps**4) for `n` larger, so truncation error can be eliminated by
    choosing steps to be very small.
    Especially the first order complex-step derivative avoids the problem of
    round-off error with small steps because there is no subtraction. However,
    this method fails if fun(x) does not support complex numbers or involves
    non-analytic functions such as e.g.: abs, max, min.
    Central difference methods are almost as accurate and has no restriction on
    type of function. For this reason the 'central' method is the default
    method, but sometimes one can only allow evaluation in forward or backward
    direction.

    For all methods one should be careful in decreasing the step size too much
    due to round-off errors.
    %(extra_note)s
    References
    ----------
    Ridout, M.S. (2009) Statistical applications of the complex-step method
        of numerical differentiation. The American Statistician, 63, 66-74

    K.-L. Lai, J.L. Crassidis, Y. Cheng, J. Kim (2005), New complex step
        derivative approximations with application to second-order
        kalman filtering, AIAA Guidance, Navigation and Control Conference,
        San Francisco, California, August 2005, AIAA-2005-5944.

    Lyness, J. M., Moler, C. B. (1966). Vandermonde Systems and Numerical
                     Differentiation. *Numerische Mathematik*.

    Lyness, J. M., Moler, C. B. (1969). Generalized Romberg Methods for
                     Integrals of Derivatives. *Numerische Mathematik*.
    %(example)s
    %(see_also)s
    """


class Derivative(_Limit):

    __doc__ = _CMN_DOC % dict(
        derivative='n-th derivative',
        extra_parameter="""
    order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.
    n : int, optional
        Order of the derivative.""",
        extra_note="""
    Higher order approximation methods will generally be more accurate, but may
    also suffer more from numerical problems. First order methods is usually
    not recommended.
    """, returns="""
    Returns
    -------
    der : ndarray
       array of derivatives
    """, example="""
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd

    # 1'st derivative of exp(x), at x == 1

    >>> fd = nd.Derivative(np.exp)
    >>> np.allclose(fd(1), 2.71828183)
    True

    >>> d2 = fd([1, 2])
    >>> np.allclose(d2, [ 2.71828183,  7.3890561 ])
    True

    >>> def f(x):
    ...     return x**3 + x**2

    >>> df = nd.Derivative(f)
    >>> np.allclose(df(1), 5)
    True
    >>> ddf = nd.Derivative(f, n=2)
    >>> np.allclose(ddf(1), 8)
    True
    """, see_also="""
    See also
    --------
    Gradient,
    Hessian
    """)

    _fd_rule = LogRule
    info = namedtuple('info', ['f_value', 'error_estimate', 'final_step', 'index'])

    def __init__(self, fun, step=None, method='central', order=2, n=1, **options):
        self.richardson_terms = options.pop('richardson_terms', 2)
        self.full_output = options.pop('full_output', False)

        self.fun = fun

        self.fd_rule = self._fd_rule(n=n, method=method, order=order)

        super(Derivative, self).__init__(step=step,  **options)
        self._set_derivative()

    @property
    def n(self):
        """Order of the derivative."""
        return self.fd_rule.n

    @n.setter
    def n(self, value):
        self.fd_rule.n = value
        self._set_derivative()

    @property
    def order(self):
        """Defines the order of the error term in the Taylor approximation used."""
        return self.fd_rule.order

    @order.setter
    def order(self, order):
        self.fd_rule.order = order

    @property
    def method(self):
        """Defines the method used in the finite difference approximation."""
        return self.fd_rule.method

    @method.setter
    def method(self, method):
        self.fd_rule.method = method

    @property
    def method_order(self):
        """Defines the leading order of the error term in the Richardson extrapolation method."""
        return self.fd_rule.method_order

    def _step_generator(self, step, options):
        if hasattr(step, '__call__'):
            return step

        if step is None and self.method not in ['complex', 'multicomplex']:
            return MaxStepGenerator(**options)
        if 'step_nom' not in options and step is not None:
            options['step_nom'] = 1.0
        return MinStepGenerator(base_step=step, **options)

    def _set_derivative(self):
        if self.n == 0:
            self._derivative = self._derivative_zero_order
        else:
            self._derivative = self._derivative_nonzero_order

    def _derivative_zero_order(self, x_i, args, kwds):
        steps = [np.zeros_like(x_i)]
        results = [self.fun(x_i, *args, **kwds)]
        self.set_richardson_rule(2, 0)
        return self._vstack(results, steps), results[0]

    def _derivative_nonzero_order(self, x_i, args, kwds):
        diff, f = self._get_functions(args, kwds)
        steps, step_ratio = self._get_steps(x_i)
        fxi = self._eval_first(f, x_i)
        results = [diff(f, fxi, x_i, h) for h in steps]

        self.set_richardson_rule(step_ratio, self.richardson_terms)

        return self.fd_rule.apply(results, steps, step_ratio), fxi

    def set_richardson_rule(self, step_ratio, num_terms=2):
        """Set Richardson exptrapolation options"""
        order = self.method_order
        step = self.fd_rule.richardson_step
        self.richardson = Richardson(step_ratio=step_ratio,
                                     step=step, order=order,
                                     num_terms=num_terms)

    def _get_functions(self, args, kwds):

        fun = self.fun

        def export_fun(x):
            return fun(x, *args, **kwds)

        return self.fd_rule.diff, export_fun

    def _get_steps(self, x_i):
        method, n, order = self.method, self.n, self.method_order
        # pylint: disable=no-member
        step_gen = self.step.step_generator_function(x_i, method, n, order)
        return list(step_gen()), step_gen.step_ratio

    def _raise_error_if_any_is_complex(self, x, f_x):
        msg = ('The {} step derivative method does only work on a real valued analytic '
               'function of a real variable!'.format(self.method))
        _assert(not np.any(np.iscomplex(x)),
                msg + ' But a complex variable was given!')

        _assert(not np.any(np.iscomplex(f_x)),
                msg + ' But the function given is complex valued!')

    def _eval_first(self, f, x):
        if self.method in ['complex', 'multicomplex']:
            f_x = f(x)
            self._raise_error_if_any_is_complex(x, f_x)
            return f_x
        if self.fd_rule.eval_first_condition or self.full_output:
            return f(x)
        return 0.0

    def __call__(self, x, *args, **kwds):
        x_i = np.asarray(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            results, f_xi = self._derivative(x_i, args, kwds)
            derivative, info = self._extrapolate(*results)
        if self.full_output:
            return derivative, self.info(f_xi, *info)
        return derivative


def directionaldiff(f, x0, vec, **options):
    """
    Return directional derivative of a function of n variables

    Parameters
    ----------
    f: function
        analytical function to differentiate.
    x0: array
        vector location at which to differentiate 'f'. If x0 is an nXm array,
        then 'f' is assumed to be a function of n*m variables.
    vec: array
        vector defining the line along which to take the derivative. It should
        be the same size as x0, but need not be a vector of unit length.
    **options:
        optional arguments to pass on to Derivative.

    Returns
    -------
    dder:  scalar
        estimate of the first derivative of 'f' in the specified direction.

    Examples
    --------
    At the global minimizer (1,1) of the Rosenbrock function,
    compute the directional derivative in the direction [1 2]

    >>> import numpy as np
    >>> import numdifftools as nd
    >>> vec = np.r_[1, 2]
    >>> rosen = lambda x: (1-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> dd, info = nd.directionaldiff(rosen, [1, 1], vec, full_output=True)
    >>> np.allclose(dd, 0)
    True
    >>> np.abs(info.error_estimate)<1e-14
    True

    See also
    --------
    Derivative,
    Gradient
    """
    x0 = np.asarray(x0)
    vec = np.asarray(vec)
    _assert(x0.size == vec.size, 'vec and x0 must be the same shapes')
    vec = np.reshape(vec / np.linalg.norm(vec.ravel()), x0.shape)
    return Derivative(lambda t: f(x0 + t * vec), **options)(0)


class Jacobian(Derivative):

    __doc__ = _CMN_DOC % dict(
        derivative='Jacobian',
        extra_parameter="""
    order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.""",
        returns="""
    Returns
    -------
    jacob : array
        Jacobian
    """, extra_note="""
    Higher order approximation methods will generally be more accurate, but may
    also suffer more from numerical problems. First order methods is usually
    not recommended.

    If fun returns a 1d array, it returns a Jacobian. If a 2d array is returned
    by fun (e.g., with a value for each observation), it returns a 3d array
    with the Jacobian of each observation with shape xk x nobs x xk. I.e.,
    the Jacobian of the first observation would be [:, 0, :]
    """, example="""
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd

    #(nonlinear least squares)

    >>> xdata = np.arange(0,1,0.1)
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> np.allclose(fun([1, 2, 0.75]).shape,  (10,))
    True

    >>> jfun = nd.Jacobian(fun)
    >>> val = jfun([1, 2, 0.75])
    >>> np.allclose(val, np.zeros((10,3)))
    True

    >>> fun2 = lambda x : x[0]*x[1]*x[2]**2
    >>> jfun2 = nd.Jacobian(fun2)
    >>> np.allclose(jfun2([1.,2.,3.]), [[18., 9., 12.]])
    True

    >>> fun3 = lambda x : np.vstack((x[0]*x[1]*x[2]**2, x[0]*x[1]*x[2]))
    >>> jfun3 = nd.Jacobian(fun3)

    >>> np.allclose(jfun3([1., 2., 3.]), [[[18.], [9.], [12.]], [[6.], [3.], [2.]]])
    True
    >>> np.allclose(jfun3([4., 5., 6.]), [[[180.], [144.], [240.]], [[30.], [24.], [20.]]])
    True
    >>> np.allclose(jfun3(np.array([[1.,2.,3.]]).T), [[[18.], [9.], [12.]], [[6.], [3.], [2.]]])
    True

    """, see_also="""
    See also
    --------
    Derivative, Hessian, Gradient
    """)

#     n = property(fget=lambda cls: 1,
#                  fset=lambda cls, val: cls._set_derivative())  # @UnusedVariable

    _fd_rule = LogJacobianRule

    @staticmethod
    def _expand_steps(steps, x_i, fxi):
        if np.size(fxi) == 1:
            return steps
        n = len(x_i)
        one = np.ones_like(fxi)
        return [np.array([one * h[i] for i in range(n)]) for h in steps]

    def _derivative_nonzero_order(self, x_i, args, kwds):
        diff, f = self._get_functions(args, kwds)
        steps, step_ratio = self._get_steps(x_i)
        fxi = f(x_i)
        results = [diff(f, fxi, x_i, h) for h in steps]

        steps2 = self._expand_steps(steps, x_i, fxi)

        self.set_richardson_rule(step_ratio, self.richardson_terms)
        return self.fd_rule.apply(results, steps2, step_ratio), fxi

    def __call__(self, x, *args, **kwds):
        return super(Jacobian, self).__call__(np.atleast_1d(x), *args, **kwds)


class Gradient(Jacobian):

    __doc__ = _CMN_DOC % dict(
        derivative='Gradient',
        extra_parameter="""
    order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.""",
        returns="""
    Returns
    -------
    grad : array
        gradient
    """, extra_note="""
    Higher order approximation methods will generally be more accurate, but may
    also suffer more from numerical problems. First order methods is usually
    not recommended.

    If x0 is an n x m array, then fun is assumed to be a function of n * m
    variables.
    """, example="""
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = nd.Gradient(fun)
    >>> np.allclose(dfun([1,2,3]), [ 2.,  4.,  6.])
    True

    # At [x,y] = [1,1], compute the numerical gradient
    # of the function sin(x-y) + y*exp(x)

    >>> sin = np.sin; exp = np.exp
    >>> x, y = 1, 1
    >>> z = lambda xy: sin(xy[0]-xy[1]) + xy[1]*exp(xy[0])
    >>> dz = nd.Gradient(z)
    >>> dz_dx, dz_dy = dz([x, y])
    >>> np.allclose([dz_dx, dz_dy],
    ...             [ 3.7182818284590686, 1.7182818284590162])
    True

    # At the global minimizer (1,1) of the Rosenbrock function,
    # compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> grad_rosen = nd.Gradient(rosen)
    >>> df_dx, df_dy = grad_rosen([x, y])
    >>> np.allclose([df_dx, df_dy], [0, 0])
    True""", see_also="""
    See also
    --------
    Derivative, Hessian, Jacobian
    """)

    def __call__(self, x, *args, **kwds):
        result = super(Gradient, self).__call__(np.atleast_1d(x).ravel(), *args, **kwds)
        if self.full_output:
            return result[0].squeeze(), result[1]
        return result.squeeze()


class Hessdiag(Derivative):

    __doc__ = _CMN_DOC % dict(
        derivative='Hessian diagonal',
        extra_parameter="""order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.""",
        returns="""
    Returns
    -------
    hessdiag : array
        hessian diagonal
    """, extra_note="""
    Higher order approximation methods will generally be more accurate, but may
    also suffer more from numerical problems. First order methods is usually
    not recommended.
    """, example="""
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> fun = lambda x : x[0] + x[1]**2 + x[2]**3
    >>> Hfun = nd.Hessdiag(fun, full_output=True)
    >>> hd, info = Hfun([1,2,3])
    >>> np.allclose(hd, [0.,   2.,  18.])
    True

    >>> np.all(info.error_estimate < 1e-11)
    True
    """, see_also="""
    See also
    --------
    Derivative, Hessian, Jacobian, Gradient
    """)

    _fd_rule = LogHessdiagRule

    def __init__(self, f, step=None, method='central', order=2, **options):
        options.pop('n', None)
        super(Hessdiag, self).__init__(f, step=step, method=method, n=2, order=order, **options)

    def __call__(self, x, *args, **kwds):
        return super(Hessdiag, self).__call__(np.atleast_1d(x), *args, **kwds)


class Hessian(Hessdiag):

    __doc__ = _CMN_DOC % dict(
        derivative='Hessian',
        extra_parameter="",
        returns=r"""
    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian
    """, extra_note=r"""
    Computes the Hessian according to method as:
    'forward' :eq:`7`, 'central' :eq:`9` and 'complex' :eq:`10`:

    .. math::
        \quad ((f(x + d_j e_j + d_k e_k) + f(x) - f(x + d_j e_j) - f(x + d_k e_k))) / (d_j d_k)
        :label: 7

    .. math::
        \quad  ((f(x + d_j e_j + d_k e_k) - f(x + d_j e_j - d_k e_k)) -
                (f(x - d_j e_j + d_k e_k) - f(x - d_j e_j - d_k e_k)) /
                (4 d_j d_k)
        :label: 9

    .. math::
        imag(f(x + i d_j e_j + d_k e_k) - f(x + i d_j e_j - d_k e_k)) /
            (2 d_j d_k)
        :label: 10

    where :math:`e_j` is a vector with element :math:`j` is one and the rest
    are zero and :math:`d_j` is a scalar spacing :math:`steps_j`.
    """, example="""
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd

    # Rosenbrock function, minimized at [1,1]

    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hfun = nd.Hessian(rosen)
    >>> h = Hfun([1, 1])
    >>> h
    array([[ 842., -420.],
           [-420.,  210.]])

    # cos(x-y), at (0,0)

    >>> cos = np.cos
    >>> fun = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = nd.Hessian(fun)
    >>> h2 = Hfun2([0, 0])
    >>> h2
    array([[-1.,  1.],
           [ 1., -1.]])""", see_also="""
    See also
    --------
    Derivative, Hessian
    """)

    _fd_rule = LogHessianRule

    def __init__(self, f, step=None, method='central', order=None, **options):
        if order is None:
            order = dict(backward=1, forward=1).get(method, 2)
        super(Hessian, self).__init__(f, step=step, method=method, order=order, **options)


if __name__ == "__main__":
    from numdifftools.testing import test_docstrings
    test_docstrings(__file__)
