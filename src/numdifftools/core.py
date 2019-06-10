# !/usr/bin/env python
"""numerical differentiation functions:

Derivative, Gradient, Jacobian, and Hessian

Author:      Per A. Brodtkorb
Created:     01.08.2008
Copyright:   (c) pab 2008
Licence:     New BSD
"""

from __future__ import division, print_function
import numpy as np
import warnings
from numpy import linalg
from numdifftools.multicomplex import Bicomplex
from numdifftools.extrapolation import Richardson, dea3, convolve
from numdifftools.step_generators import MaxStepGenerator, MinStepGenerator
from numdifftools.limits import _Limit
from numdifftools.finite_difference import LogRule
from scipy import special

__all__ = ('dea3', 'Derivative', 'Jacobian', 'Gradient', 'Hessian', 'Hessdiag',
           'MinStepGenerator', 'MaxStepGenerator', 'Richardson',
           'directionaldiff')
FD_RULES = {}
_SQRT_J = (1j + 1.0) / np.sqrt(2.0)  # = 1j**0.5


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


_cmn_doc = """
    Calculate %(derivative)s with finite difference approximation

    Parameters
    ----------
    fun : function
       function of one array fun(x, `*args`, `**kwds`)
    step : float, array-like or StepGenerator object, optional
        Defines the spacing used in the approximation.
        Default is MinStepGenerator(base_step=step, step_ratio=None,
                                   num_extrap=0, **step_options)
        if step or method in in ['complex', 'multicomplex'],
        otherwise
            MaxStepGenerator(step_ratio=None, num_extrap=14, **step_options)
        The results are extrapolated if the StepGenerator generate more than 3
        steps.
    method : {'central', 'complex', 'multicomplex', 'forward', 'backward'}
        defines the method used in the approximation%(extra_parameter)s
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


class _DerivativeDifferenceFunctions(object):

    @staticmethod
    def _central_even(f, f_x0i, x0i, h):
        return (f(x0i + h) + f(x0i - h)) / 2.0 - f_x0i

    @staticmethod
    def _central(f, f_x0i, x0i, h):
        return (f(x0i + h) - f(x0i - h)) / 2.0

    @staticmethod
    def _forward(f, f_x0i, x0i, h):
        return f(x0i + h) - f_x0i

    @staticmethod
    def _backward(f, f_x0i, x0i, h):
        return f_x0i - f(x0i - h)

    @staticmethod
    def _complex(f, fx, x, h):
        return f(x + 1j * h).imag

    @staticmethod
    def _complex_odd(f, fx, x, h):
        ih = h * _SQRT_J
        return ((_SQRT_J / 2.) * (f(x + ih) - f(x - ih))).imag

    @staticmethod
    def _complex_odd_higher(f, fx, x, h):
        ih = h * _SQRT_J
        return ((3 * _SQRT_J) * (f(x + ih) - f(x - ih))).real

    @staticmethod
    def _complex_even(f, fx, x, h):
        ih = h * _SQRT_J
        return (f(x + ih) + f(x - ih)).imag

    @staticmethod
    def _complex_even_higher(f, fx, x, h):
        ih = h * _SQRT_J
        return 12.0 * (f(x + ih) + f(x - ih) - 2 * fx).real

    @staticmethod
    def _multicomplex(f, fx, x, h):
        z = Bicomplex(x + 1j * h, 0)
        return Bicomplex.__array_wrap__(f(z)).imag

    @staticmethod
    def _multicomplex2(f, fx, x, h):
        z = Bicomplex(x + 1j * h, h)
        return Bicomplex.__array_wrap__(f(z)).imag12


class Derivative(_Limit):

    __doc__ = _cmn_doc % dict(
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

    def __init__(self, fun, step=None, method='central', order=2, n=1, full_output=False,
                 **step_options):
        self.n = n
        self.richardson_terms = 2
        self.log_rule = LogRule(n=n, method=method, order=order)

        super(Derivative,
              self).__init__(fun, step=step, method=method, order=order, full_output=full_output,
                             **step_options)

    n = property(fget=lambda cls: cls._n,
                 fset=lambda cls, n: (setattr(cls, '_n', n),
                                      cls._set_derivative()))

    _difference_functions = _DerivativeDifferenceFunctions()

    def _step_generator(self, step, options):
        if hasattr(step, '__call__'):
            return step
        step_options = dict(step_ratio=None, num_extrap=14)
        if step is None and self.method not in ['complex', 'multicomplex']:
            step_options.update(**options)
            return MaxStepGenerator(**step_options)

        step_options['num_extrap'] = 0
        step_options['step_nom'] = None if step is None else 1.0
        step_options.update(**options)
        return MinStepGenerator(base_step=step, **step_options)

    def _set_derivative(self):
        if self.n == 0:
            self._derivative = self._derivative_zero_order
        else:
            self._derivative = self._derivative_nonzero_order

    def _derivative_zero_order(self, xi, args, kwds):
        steps = [np.zeros_like(xi)]
        results = [self.fun(xi, *args, **kwds)]
        self.set_richardson_rule(2, 0)
        return self._vstack(results, steps)

    def _derivative_nonzero_order(self, xi, args, kwds):
        diff, f = self._get_functions(args, kwds)
        steps, step_ratio = self._get_steps(xi)
        fxi = self._eval_first(f, xi)
        results = [diff(f, fxi, xi, h) for h in steps]

        self.set_richardson_rule(step_ratio, self.richardson_terms)

        return self._apply_fd_rule(step_ratio, results, steps)

    @property
    def _method_order(self):
        step = self._richardson_step()
        # Make sure it is even and at least 2 or 4
        order = max((self.order // step) * step, step)
        return order

    @property
    def _complex_high_order(self):
        return self.method == 'complex' and (self.n > 1 or self.order >= 4)

    def _richardson_step(self):
        complex_step = 4 if self._complex_high_order else 2
        return dict(central=2,
                    central2=2,
                    complex=complex_step,
                    multicomplex=2).get(self.method, 1)

    def set_richardson_rule(self, step_ratio, num_terms=2):
        order = self._method_order
        step = self._richardson_step()
        self.richardson = Richardson(step_ratio=step_ratio,
                                     step=step, order=order,
                                     num_terms=num_terms)

    def _multicomplex_middle_name_or_empty(self):
        if self.method == 'multicomplex' and self.n > 1:
            _assert(self.n <= 2, 'Multicomplex method only support first '
                    'and second order derivatives.')
            return '2'
        return ''

    def _get_middle_name(self):
        if self._even_derivative and self.method in ('central', 'complex'):
            return '_even'
        if self._complex_high_order and self._odd_derivative:
            return '_odd'
        return self._multicomplex_middle_name_or_empty()

    def _get_last_name(self):
        last = ''
        if (self.method == 'complex' and self._derivative_mod_four_is_zero or
                self._complex_high_order and
                self._derivative_mod_four_is_three):
            last = '_higher'
        return last

    def _get_function_name(self):
        first = '_{0!s}'.format(self.method)
        middle = self._get_middle_name()
        last = self._get_last_name()
        name = first + middle + last
        return name

    def _get_functions(self, args, kwds):
        name = self._get_function_name()
        fun = self.fun

        def export_fun(x):
            return fun(x, *args, **kwds)

        return getattr(self._difference_functions, name), export_fun

    def _get_steps(self, xi):
        method, n, order = self.method, self.n, self._method_order
        step_gen = self.step.step_generator_function(xi, method, n, order)
        return [step for step in step_gen()], step_gen.step_ratio

    @property
    def _odd_derivative(self):
        return self.n % 2 == 1

    @property
    def _even_derivative(self):
        return self.n % 2 == 0

    @property
    def _derivative_mod_four_is_three(self):
        return self.n % 4 == 3

    @property
    def _derivative_mod_four_is_zero(self):
        return self.n % 4 == 0

    def _eval_first_condition(self):
        even_derivative = self._even_derivative
        return ((even_derivative and self.method in ('central', 'central2')) or
                self.method in ['forward', 'backward'] or
                self.method == 'complex' and self._derivative_mod_four_is_zero)

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
        if self._eval_first_condition():
            return f(x)
        return 0.0

    def __call__(self, x, *args, **kwds):
        xi = np.asarray(x)
        results = self._derivative(xi, args, kwds)
        derivative, info = self._extrapolate(*results)
        if self.full_output:
            return derivative, info
        return derivative

    @staticmethod
    def _fd_matrix(step_ratio, parity, nterms):
        """
        Return matrix for finite difference and complex step derivation.

        Parameters
        ----------
        step_ratio : real scalar
            ratio between steps in unequally spaced difference rule.
        parity : scalar, integer
            0 (one sided, all terms included but zeroth order)
            1 (only odd terms included)
            2 (only even terms included)
            3 (only every 4'th order terms included starting from order 2)
            4 (only every 4'th order terms included starting from order 4)
            5 (only every 4'th order terms included starting from order 1)
            6 (only every 4'th order terms included starting from order 3)
        nterms : scalar, integer
            number of terms
        """
        _assert(0 <= parity <= 6,
                'Parity must be 0, 1, 2, 3, 4, 5 or 6! ({0:d})'.format(parity))
        step = [1, 2, 2, 4, 4, 4, 4][parity]
        inv_sr = 1.0 / step_ratio
        offset = [1, 1, 2, 2, 4, 1, 3][parity]
        c0 = [1.0, 1.0, 1.0, 2.0, 24.0, 1.0, 6.0][parity]
        c = c0 / \
            special.factorial(np.arange(offset, step * nterms + offset, step))
        [i, j] = np.ogrid[0:nterms, 0:nterms]
        return np.atleast_2d(c[j] * inv_sr ** (i * (step * j + offset)))

    @property
    def _flip_fd_rule(self):
        return ((self._even_derivative and (self.method == 'backward')) or
                (self.method == 'complex' and (self.n % 8 in [3, 4, 5, 6])))

    def _parity_complex(self, order, method_order):
        if self.n == 1 and method_order < 4:
            return (order % 2) + 1
        return (3
                + 2 * int(self._odd_derivative)
                + int(self._derivative_mod_four_is_three)
                + int(self._derivative_mod_four_is_zero))

    def _parity(self, method, order, method_order):
        if method.startswith('central'):
            return (order % 2) + 1
        if method == 'complex':
            return self._parity_complex(order, method_order)
        return 0

    def _get_finite_difference_rule(self, step_ratio):
        """
        Generate finite differencing rule in advance.

        The rule is for a nominal unit step size, and will
        be scaled later to reflect the local step size.

        Member method used: _fd_matrix

        Member variables used:
        n
        order
        method
        """
        method = self.method
        if method in ('multicomplex',) or self.n == 0:
            return np.ones((1,))

        order, method_order = self.n - 1, self._method_order
        parity = self._parity(method, order, method_order)
        step = self._richardson_step()
        num_terms, ix = (order + method_order) // step, order // step
        fd_rules = FD_RULES.get((step_ratio, parity, num_terms))
        if fd_rules is None:
            fd_mat = self._fd_matrix(step_ratio, parity, num_terms)
            fd_rules = linalg.pinv(fd_mat)
            FD_RULES[(step_ratio, parity, num_terms)] = fd_rules

        if self._flip_fd_rule:
            return -fd_rules[ix]
        return fd_rules[ix]

    def _apply_fd_rule(self, step_ratio, sequence, steps):
        """
        Return derivative estimates of fun at x0 for a sequence of stepsizes h

        Member variables used
        ---------------------
        n
        """
        f_del, h, original_shape = self._vstack(sequence, steps)
        fd_rule = self._get_finite_difference_rule(step_ratio)
        ne = h.shape[0]
        nr = fd_rule.size - 1
        _assert(nr < ne, 'num_steps ({0:d}) must  be larger than '
                '({1:d}) n + order - 1 = {2:d} + {3:d} -1'
                ' ({4:s})'.format(ne, nr + 1, self.n, self.order, self.method))
        f_diff = convolve(f_del, fd_rule[::-1], axis=0, origin=nr // 2)

        der_init = f_diff / (h ** self.n)
        ne = max(ne - nr, 1)
        return der_init[:ne], h[:ne], original_shape


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


class _JacobianDifferenceFunctions(object):

    @staticmethod
    def _central(f, fx, x, h):
        n = len(x)
        return np.array([(f(x + hi) - f(x - hi)) / 2.0 for hi in Jacobian._increments(n, h)])

    @staticmethod
    def _backward(f, fx, x, h):
        n = len(x)
        return np.array([fx - f(x - hi) for hi in Jacobian._increments(n, h)])

    @staticmethod
    def _forward(f, fx, x, h):
        n = len(x)
        return np.array([f(x + hi) - fx for hi in Jacobian._increments(n, h)])

    @staticmethod
    def _complex(f, fx, x, h):
        n = len(x)
        return np.array([f(x + 1j * ih).imag for ih in Jacobian._increments(n, h)])

    @staticmethod
    def _complex_odd(f, fx, x, h):
        n = len(x)
        j1 = _SQRT_J
        return np.array([((j1 / 2.) * (f(x + j1 * ih) - f(x - j1 * ih))).imag
                         for ih in Jacobian._increments(n, h)])

    @staticmethod
    def _multicomplex(f, fx, x, h):
        n = len(x)
        cmplx_wrap = Bicomplex.__array_wrap__
        partials = [cmplx_wrap(f(Bicomplex(x + 1j * hi, 0))).imag
                    for hi in Jacobian._increments(n, h)]
        return np.array(partials)


class Jacobian(Derivative):

    __doc__ = _cmn_doc % dict(
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
    >>> np.allclose(jfun3([1., 2., 3.]), [[18., 9., 12.], [6., 3., 2.]])
    True
    >>> np.allclose(jfun3([4., 5., 6.]), [[180., 144., 240.], [30., 24., 20.]])
    True
    >>> np.allclose(jfun3(np.array([[1.,2.,3.]]).T), [[ 18.,   9.,  12.],
    ...                                               [ 6.,   3.,   2.]])
    True

    """, see_also="""
    See also
    --------
    Derivative, Hessian, Gradient
    """)
    n = property(fget=lambda cls: 1,
                 fset=lambda cls, val: cls._set_derivative())

    _difference_functions = _JacobianDifferenceFunctions()

    @staticmethod
    def _atleast_2d(original_shape, ndim):
        if ndim == 1:
            original_shape = (1,) + tuple(original_shape)
        return tuple(original_shape)

    def _vstack(self, sequence, steps):
        original_shape = list(np.shape(np.atleast_1d(sequence[0].squeeze())))
        ndim = len(original_shape)
        axes = [0, 1, 2][:ndim]
        axes[:2] = axes[1::-1]
        original_shape[:2] = original_shape[1::-1]

        f_del = np.vstack([np.atleast_1d(r.squeeze()).transpose(axes).ravel()
                           for r in sequence])
        h = np.vstack([np.atleast_1d(r.squeeze()).transpose(axes).ravel()
                       for r in steps])
        _assert(f_del.size == h.size, 'fun did not return data of correct '
                'size (it must be vectorized)')
        return f_del, h, self._atleast_2d(original_shape, ndim)

    @staticmethod
    def _increments(n, h):
        ei = np.zeros(np.shape(h), float)
        for k in range(n):
            ei[k] = h[k]
            yield ei
            ei[k] = 0

    @staticmethod
    def _expand_steps(steps, x_i, fxi):
        if np.size(fxi) == 1:
            return steps
        n = len(x_i)
        one = np.ones_like(fxi)
        return [np.array([one * h[i] for i in range(n)]) for h in steps]

    def _derivative_nonzero_order(self, xi, args, kwds):
        diff, f = self._get_functions(args, kwds)
        steps, step_ratio = self._get_steps(xi)
        fxi = f(xi)
        results = [diff(f, fxi, xi, h) for h in steps]

        steps2 = self._expand_steps(steps, xi, fxi)

        self.set_richardson_rule(step_ratio, self.richardson_terms)
        return self._apply_fd_rule(step_ratio, results, steps2)

    def __call__(self, x, *args, **kwds):
        return super(Jacobian, self).__call__(np.atleast_1d(x), *args, **kwds)


class Gradient(Jacobian):

    __doc__ = _cmn_doc % dict(
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
        return super(Gradient, self).__call__(np.atleast_1d(x).ravel(),
                                              *args, **kwds).squeeze()


class _HessdiagDifferenceFunctions(object):

    @staticmethod
    def _central2(f, fx, x, h):
        """Eq. 8"""
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + 2 * hi) + f(x - 2 * hi)
                     + 2 * fx - 2 * f(x + hi) - 2 * f(x - hi)) / 4.0
                    for hi in increments]
        return np.array(partials)

    @staticmethod
    def _central_even(f, fx, x, h):
        """Eq. 9"""
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + hi) + f(x - hi)) / 2.0 - fx for hi in increments]
        return np.array(partials)

    @staticmethod
    def _backward(f, fx, x, h):
        n = len(x)
        increments = np.identity(n) * h
        partials = [fx - f(x - hi) for hi in increments]
        return np.array(partials)

    @staticmethod
    def _forward(f, fx, x, h):
        n = len(x)
        increments = np.identity(n) * h
        partials = [f(x + hi) - fx for hi in increments]
        return np.array(partials)

    @staticmethod
    def _multicomplex2(f, fx, x, h):
        n = len(x)
        increments = np.identity(n) * h
        cmplx_wrap = Bicomplex.__array_wrap__
        partials = [cmplx_wrap(f(Bicomplex(x + 1j * hi, hi))).imag12
                    for hi in increments]
        return np.array(partials)

    @staticmethod
    def _complex_even(f, fx, x, h):
        n = len(x)
        increments = np.identity(n) * h * (1j + 1) / np.sqrt(2)
        partials = [(f(x + hi) + f(x - hi)).imag for hi in increments]
        return np.array(partials)


class Hessdiag(Derivative):

    __doc__ = _cmn_doc % dict(
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

    def __init__(self, f, step=None, method='central', order=2,
                 full_output=False, **step_options):
        super(Hessdiag, self).__init__(f, step=step, method=method, n=2,
                                       order=order, full_output=full_output,
                                       **step_options)

    _difference_functions = _HessdiagDifferenceFunctions()

    n = property(fget=lambda cls: 2,
                 fset=lambda cls, n: cls._set_derivative())

    def __call__(self, x, *args, **kwds):
        return super(Hessdiag, self).__call__(np.atleast_1d(x), *args, **kwds)


class _HessianDifferenceFunctions(object):

    @staticmethod
    def _complex_even(f, fx, x, h):
        """
        Calculate Hessian with complex-step derivative approximation

        The stepsize is the same for the complex and the finite difference part
        """
        n = len(x)
        ee = np.diag(h)
        hess = 2. * np.outer(h, h)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + 1j * ee[i] + ee[j])
                              - f(x + 1j * ee[i] - ee[j])).imag / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _multicomplex2(f, fx, x, h):
        """Calculate Hessian with Bicomplex-step derivative approximation"""
        n = len(x)
        ee = np.diag(h)
        hess = np.outer(h, h)
        cmplx_wrap = Bicomplex.__array_wrap__
        for i in range(n):
            for j in range(i, n):
                zph = Bicomplex(x + 1j * ee[i, :], ee[j, :])
                hess[i, j] = cmplx_wrap(f(zph)).imag12 / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _central_even(f, fx, x, h):
        """Eq 9."""
        n = len(x)
        ee = np.diag(h)
        dtype = np.result_type(fx, float)  # make sure it is at least float64
        hess = np.empty((n, n), dtype=dtype)
        np.outer(h, h, out=hess)
        for i in range(n):
            hess[i, i] = (f(x + 2 * ee[i, :]) - 2 * fx + f(x - 2 * ee[i, :])) / (4. * hess[i, i])
            for j in range(i + 1, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :])
                              - f(x + ee[i, :] - ee[j, :])
                              - f(x - ee[i, :] + ee[j, :])
                              + f(x - ee[i, :] - ee[j, :])) / (4. * hess[j, i])
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _central2(f, fx, x, h):
        """Eq. 8"""
        n = len(x)
        ee = np.diag(h)
        dtype = np.result_type(fx, float)
        g = np.empty(n, dtype=dtype)
        gg = np.empty(n, dtype=dtype)
        for i in range(n):
            g[i] = f(x + ee[i])
            gg[i] = f(x - ee[i])

        hess = np.empty((n, n), dtype=dtype)
        np.outer(h, h, out=hess)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :])
                              + f(x - ee[i, :] - ee[j, :])
                              - g[i] - g[j] + fx
                              - gg[i] - gg[j] + fx) / (2 * hess[j, i])
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _forward(f, fx, x, h):
        """Eq. 7"""
        n = len(x)
        ee = np.diag(h)
        dtype = np.result_type(fx, float)
        g = np.empty(n, dtype=dtype)
        for i in range(n):
            g[i] = f(x + ee[i, :])

        hess = np.empty((n, n), dtype=dtype)
        np.outer(h, h, out=hess)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :]) - g[i] - g[j] + fx) / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _backward(f, fx, x, h):
        return _HessianDifferenceFunctions._forward(f, fx, x, -h)


class Hessian(Hessdiag):

    __doc__ = _cmn_doc % dict(
        derivative='Hessian',
        extra_parameter="",
        returns="""
    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian
    """, extra_note="""
    Computes the Hessian according to method as:
    'forward' :eq:`7`, 'central' :eq:`9` and 'complex' :eq:`10`:

    .. math::
        \quad ((f(x + d_j e_j + d_k e_k) - f(x + d_j e_j))) / (d_j d_k)
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

    _difference_functions = _HessianDifferenceFunctions()

    @property
    def order(self):
        return dict(backward=1, forward=1).get(self.method, 2)

    @order.setter
    def order(self, order):
        valid_order = self.order
        if order != valid_order:
            msg = 'Can not change order to {}! The only valid order is {} for method={}.'
            warnings.warn(msg.format(order, valid_order, self.method))

    @staticmethod
    def _complex_high_order():
        return False

    def _apply_fd_rule(self, step_ratio, sequence, steps):
        """
        Return derivative estimates of fun at x0 for a sequence of stepsizes h

        Here the difference rule is already applied. Just return result.
        """
        return self._vstack(sequence, steps)
