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
from numpy import linalg
from numdifftools.multicomplex import Bicomplex
from numdifftools.extrapolation import Richardson, dea3, convolve
from numdifftools.step_generators import MaxStepGenerator, MinStepGenerator
from numdifftools.limits import _Limit
from scipy import misc

__all__ = ('dea3', 'Derivative', 'Jacobian', 'Gradient', 'Hessian', 'Hessdiag',
           'MinStepGenerator', 'MaxStepGenerator', 'Richardson',
           'directionaldiff')
_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps
EPS = np.MachAr().eps
_SQRT_J = (1j + 1.0) / np.sqrt(2.0)  # = 1j**0.5


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


_cmn_doc = """
    Calculate %(derivative)s with finite difference approximation

    Parameters
    ----------
    f : function
       function of one array f(x, `*args`, `**kwds`)
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

    Call Parameters
    ---------------
    x : array_like
       value at which function derivative is evaluated
    args : tuple
        Arguments for function `f`.
    kwds : dict
        Keyword arguments for function `f`.
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
    this method fails if f(x) does not support complex numbers or involves
    non-analytic functions such as e.g.: abs, max, min.
    Central difference methods are almost as accurate and has no restriction on
    type of function. For this reason the 'central' method is the default
    method, but sometimes one can only allow evaluation in forward or backward
    direction.

    For all methods one should be careful in decreasing the step size too much
    due to round-off errors.
    %(extra_note)s
    Reference
    ---------
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
    Example
    -------
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

    def __init__(self, fun, step=None, method='central', order=2, n=1,
                 full_output=False, **step_options):
        self.n = n
        self.richardson_terms = 2
        super(Derivative,
              self).__init__(fun, step=step, method=method, order=order,
                             full_output=full_output, **step_options)

    n = property(fget=lambda cls: cls._n,
                 fset=lambda cls, n: (setattr(cls, '_n', n),
                                      cls._set_derivative()))

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
        diff, f = self._get_functions()
        steps, step_ratio = self._get_steps(xi)
        fxi = self._eval_first(f, xi, *args, **kwds)
        results = [diff(f, fxi, xi, h, *args, **kwds) for h in steps]

        self.set_richardson_rule(step_ratio, self.richardson_terms)

        return self._apply_fd_rule(step_ratio, results, steps)

    def _make_generator(self, step, step_options):
        if hasattr(step, '__call__'):
            return step
        options = dict(step_ratio=None, num_extrap=14)
        if step is None and self.method not in ['complex', 'multicomplex']:
            options.update(**step_options)
            return MaxStepGenerator(**options)
        options['num_extrap'] = 0
        options.update(**step_options)
        return MinStepGenerator(base_step=step, **options)

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

        return dict(central=2, central2=2, complex=complex_step,
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

    def _get_functions(self):
        name = self._get_function_name()
        return getattr(self, name), self.fun

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

    def _eval_first(self, f, x, *args, **kwds):
        if self._eval_first_condition():
            return f(x, *args, **kwds)
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
            misc.factorial(np.arange(offset, step * nterms + offset, step))
        [i, j] = np.ogrid[0:nterms, 0:nterms]
        return np.atleast_2d(c[j] * inv_sr ** (i * (step * j + offset)))

    @property
    def _flip_fd_rule(self):
        return ((self._even_derivative and (self.method == 'backward')) or
                (self.method == 'complex' and (self.n % 8 in [3, 4, 5, 6])))

    def _parity_complex(self, order, method_order):
        if self.n == 1 and method_order < 4:
            return (order % 2) + 1
        return (3 + 2 * int(self._odd_derivative) +
                int(self._derivative_mod_four_is_three) +
                int(self._derivative_mod_four_is_zero))

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

        Member methods used
        -------------------
        _fd_matrix

        Member variables used
        ---------------------
        n
        order
        method
        """
        method = self.method
        if method in ('multicomplex', ) or self.n == 0:
            return np.ones((1,))

        order, method_order = self.n - 1, self._method_order
        parity = self._parity(method, order, method_order)
        step = self._richardson_step()
        num_terms, ix = (order + method_order) // step, order // step
        fd_mat = self._fd_matrix(step_ratio, parity, num_terms)
        fd_rule = linalg.pinv(fd_mat)[ix]

        if self._flip_fd_rule:
            fd_rule *= -1
        return fd_rule

    def _apply_fd_rule(self, step_ratio, sequence, steps):
        """
        Return derivative estimates of f at x0 for a sequence of stepsizes h

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
                ' ({4:s})'.format(ne, nr+1, self.n, self.order, self.method)
                             )
        f_diff = convolve(f_del, fd_rule[::-1], axis=0, origin=nr // 2)

        der_init = f_diff / (h ** self.n)
        ne = max(ne - nr, 1)
        return der_init[:ne], h[:ne], original_shape

    @staticmethod
    def _central_even(f, f_x0i, x0i, h, *args, **kwds):
        return (f(x0i + h, *args, **kwds) +
                f(x0i - h, *args, **kwds)) / 2.0 - f_x0i

    @staticmethod
    def _central(f, f_x0i, x0i, h, *args, **kwds):
        return (f(x0i + h, *args, **kwds) -
                f(x0i - h, *args, **kwds)) / 2.0

    @staticmethod
    def _forward(f, f_x0i, x0i, h, *args, **kwds):
        return f(x0i + h, *args, **kwds) - f_x0i

    @staticmethod
    def _backward(f, f_x0i, x0i, h, *args, **kwds):
        return f_x0i - f(x0i - h, *args, **kwds)

    @staticmethod
    def _complex(f, fx, x, h, *args, **kwds):
        return f(x + 1j * h, *args, **kwds).imag

    @staticmethod
    def _complex_odd(f, fx, x, h, *args, **kwds):
        ih = h * _SQRT_J
        return ((_SQRT_J / 2.) * (f(x + ih, *args, **kwds) -
                                  f(x - ih, *args, **kwds))).imag

    @staticmethod
    def _complex_odd_higher(f, fx, x, h, *args, **kwds):
        ih = h * _SQRT_J
        return ((3 * _SQRT_J) * (f(x + ih, *args, **kwds) -
                                 f(x - ih, *args, **kwds))).real

    @staticmethod
    def _complex_even(f, fx, x, h, *args, **kwds):
        ih = h * _SQRT_J
        return (f(x + ih, *args, **kwds) +
                f(x - ih, *args, **kwds)).imag

    @staticmethod
    def _complex_even_higher(f, fx, x, h, *args, **kwds):
        ih = h * _SQRT_J
        return 12.0 * (f(x + ih, *args, **kwds) +
                       f(x - ih, *args, **kwds) - 2 * fx).real

    @staticmethod
    def _multicomplex(f, fx, x, h, *args, **kwds):
        z = Bicomplex(x + 1j * h, 0)
        return Bicomplex.__array_wrap__(f(z, *args, **kwds)).imag

    @staticmethod
    def _multicomplex2(f, fx, x, h, *args, **kwds):
        z = Bicomplex(x + 1j * h, h)
        return Bicomplex.__array_wrap__(f(z, *args, **kwds)).imag12


def directionaldiff(f, x0, vec, **options):
    """
    Return directional derivative of a function of n variables

    Parameters
    ----------
    f: function
        analytical function to differentiate.
    x0: array
        vector location at which to differentiate f. If x0 is an nxm array,
        then f is assumed to be a function of n*m variables.
    vec: array
        vector defining the line along which to take the derivative. It should
        be the same size as x0, but need not be a vector of unit length.
    **options:
        optional arguments to pass on to Derivative.

    Returns
    -------
    dder:  scalar
        estimate of the first derivative of f in the specified direction.

    Example
    -------
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
    Example
    -------
    >>> import numdifftools as nd

    #(nonlinear least squares)

    >>> xdata = np.arange(0,1,0.1)
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> np.allclose(fun([1, 2, 0.75]).shape,  (10,))
    True

    >>> Jfun = nd.Jacobian(fun)
    >>> val = Jfun([1, 2, 0.75])
    >>> np.allclose(val, np.zeros((10,3)))
    True

    >>> fun2 = lambda x : x[0]*x[1]*x[2]**2
    >>> Jfun2 = nd.Jacobian(fun2)
    >>> np.allclose(Jfun2([1.,2.,3.]), [[18., 9., 12.]])
    True

    >>> fun3 = lambda x : np.vstack((x[0]*x[1]*x[2]**2, x[0]*x[1]*x[2]))
    >>> Jfun3 = nd.Jacobian(fun3)
    >>> np.allclose(Jfun3([1., 2., 3.]), [[18., 9., 12.], [6., 3., 2.]])
    True
    >>> np.allclose(Jfun3([4., 5., 6.]), [[180., 144., 240.], [30., 24., 20.]])
    True
    >>> np.allclose(Jfun3(np.array([[1.,2.,3.]]).T), [[ 18.,   9.,  12.],
    ...                                               [ 6.,   3.,   2.]])
    True

    """, see_also="""
    See also
    --------
    Derivative, Hessian, Gradient
    """)
    n = property(fget=lambda cls: 1,
                 fset=lambda cls, val: cls._set_derivative())

    @staticmethod
    def _check_equal_size(f_del, h):
        _assert(f_del.size == h.size, 'fun did not return data of correct '
                'size (it must be vectorized)')

    @staticmethod
    def _atleast_2d(original_shape, ndim):
        if ndim == 1:
            original_shape = (1, ) + tuple(original_shape)
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
        return f_del, h, self._atleast_2d(original_shape, ndim)

    @staticmethod
    def _identity(n):
        m = np.zeros((n, n, n))
        np.put(m, np.arange(0, n ** 3, n * (n + 1) + 1), 1)
        return m

    def _increments(self, n, h):
        return np.dot(self._identity(n), h)

    def _central(self, f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = self._increments(n, h)
        partials = [(f(x + hi, *args, **kwds) - f(x - hi, *args, **kwds)) / 2.0
                    for hi in increments]
        return np.array(partials)

    def _backward(self, f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = self._increments(n, h)
        partials = [fx - f(x - hi, *args, **kwds) for hi in increments]
        return np.array(partials)

    def _forward(self, f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = self._increments(n, h)
        partials = [f(x + hi, *args, **kwds) - fx for hi in increments]
        return np.array(partials)

    def _complex(self, f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = 1j * self._increments(n, h)
        partials = [f(x + ih, *args, **kwds).imag for ih in increments]
        return np.array(partials)

    def _complex_odd(self, f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = _SQRT_J * self._increments(n, h)
        partials = [((_SQRT_J / 2.) * (f(x + ih, *args, **kwds) -
                                       f(x - ih, *args, **kwds))).imag
                    for ih in increments]
        return np.array(partials)

    def _multicomplex(self, f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = 1j * self._increments(n, h)
        cmplx_wrap = Bicomplex.__array_wrap__
        partials = [cmplx_wrap(f(Bicomplex(x + hi, 0), *args, **kwds)).imag
                    for hi in increments]
        return np.array(partials)

    def _derivative_nonzero_order(self, xi, args, kwds):
        diff, f = self._get_functions()
        steps, step_ratio = self._get_steps(xi)
        fxi = f(xi, *args, **kwds)
        results = [diff(f, fxi, xi, h, *args, **kwds) for h in steps]

        n = len(xi)
        one = np.ones_like(fxi)
        steps2 = [np.array([one * h[i] for i in range(n)]) for h in steps]

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

    If x0 is an nxm array, then fun is assumed to be a function of n*m variables.
    """, example="""
    Example
    -------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = nd.Gradient(fun)
    >>> dfun([1,2,3])
    array([ 2.,  4.,  6.])

    # At [x,y] = [1,1], compute the numerical gradient
    # of the function sin(x-y) + y*exp(x)

    >>> sin = np.sin; exp = np.exp
    >>> z = lambda xy: sin(xy[0]-xy[1]) + xy[1]*exp(xy[0])
    >>> dz = nd.Gradient(z)
    >>> grad2 = dz([1, 1])
    >>> grad2
    array([ 3.71828183,  1.71828183])

    # At the global minimizer (1,1) of the Rosenbrock function,
    # compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> rd = nd.Gradient(rosen)
    >>> grad3 = rd([1,1])
    >>> np.allclose(grad3,[0, 0])
    True""", see_also="""
    See also
    --------
    Derivative, Hessian, Jacobian
    """)

    def __call__(self, x, *args, **kwds):
        return super(Gradient, self).__call__(np.atleast_1d(x).ravel(),
                                              *args, **kwds).squeeze()


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
    Example
    -------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> fun = lambda x : x[0] + x[1]**2 + x[2]**3
    >>> Hfun = nd.Hessdiag(fun, full_output=True)
    >>> hd, info = Hfun([1,2,3])
    >>> np.allclose(hd, [  0.,   2.,  18.])
    True

    >>> info.error_estimate < 1e-11
    array([ True,  True,  True], dtype=bool)
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

    n = property(fget=lambda cls: 2,
                 fset=lambda cls, n: cls._set_derivative())

    @staticmethod
    def _central2(f, fx, x, h, *args, **kwds):
        """Eq. 8"""
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + 2 * hi, *args, **kwds) +
                     f(x - 2 * hi, *args, **kwds) + 2 * fx -
                     2 * f(x + hi, *args, **kwds) -
                     2 * f(x - hi, *args, **kwds)) / 4.0
                    for hi in increments]
        return np.array(partials)

    @staticmethod
    def _central_even(f, fx, x, h, *args, **kwds):
        """Eq. 9"""
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + hi, *args, **kwds) +
                     f(x - hi, *args, **kwds)) / 2.0 - fx
                    for hi in increments]
        return np.array(partials)

    @staticmethod
    def _backward(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h
        partials = [fx - f(x - hi, *args, **kwds) for hi in increments]
        return np.array(partials)

    @staticmethod
    def _forward(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h
        partials = [f(x + hi, *args, **kwds) - fx for hi in increments]
        return np.array(partials)

    @staticmethod
    def _multicomplex2(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h
        cmplx_wrap = Bicomplex.__array_wrap__
        partials = [cmplx_wrap(f(Bicomplex(x + 1j * hi, hi), *args,
                                 **kwds)).imag12
                    for hi in increments]
        return np.array(partials)

    @staticmethod
    def _complex_even(f, fx, x, h, *args, **kwargs):
        n = len(x)
        increments = np.identity(n) * h * (1j + 1) / np.sqrt(2)
        partials = [(f(x + hi, *args, **kwargs) +
                     f(x - hi, *args, **kwargs)).imag
                    for hi in increments]
        return np.array(partials)

    def __call__(self, x, *args, **kwds):
        return super(Hessdiag, self).__call__(np.atleast_1d(x), *args, **kwds)


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
    Example
    -------
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

    order = property(fget=lambda cls: dict(backward=1, forward=1,
                                           complex=2).get(cls.method, 2),
                     fset=lambda cls, order: None)

    def _apply_fd_rule(self, step_ratio, sequence, steps):
        """
        Return derivative estimates of f at x0 for a sequence of stepsizes h

        Here the difference rule is already applied. Just return result.
        """
        return self._vstack(sequence, steps)

    @staticmethod
    def _complex_high_order():
        return False

    @staticmethod
    def _complex_even(f, fx, x, h, *args, **kwargs):
        """
        Calculate Hessian with complex-step derivative approximation

        The stepsize is the same for the complex and the finite difference part
        """
        n = len(x)
        ee = np.diag(h)
        hess = 2. * np.outer(h, h)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + 1j * ee[i] + ee[j], *args, **kwargs) -
                             f(x + 1j * ee[i] - ee[j], *args, **kwargs)
                             ).imag / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _multicomplex2(f, fx, x, h, *args, **kwargs):
        """Calculate Hessian with Bicomplex-step derivative approximation"""
        n = len(x)
        ee = np.diag(h)
        hess = np.outer(h, h)
        cmplx_wrap = Bicomplex.__array_wrap__
        for i in range(n):
            for j in range(i, n):
                zph = Bicomplex(x + 1j * ee[i, :], ee[j, :])
                hess[i, j] = cmplx_wrap(f(zph, *args,
                                          **kwargs)).imag12 / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _central_even(f, fx, x, h, *args, **kwargs):
        """Eq 9."""
        n = len(x)
        ee = np.diag(h)
        hess = np.outer(h, h)
        for i in range(n):
            hess[i, i] = (f(x + 2 * ee[i, :], *args, **kwargs) - 2 * fx +
                          f(x - 2 * ee[i, :], *args, **kwargs)
                          ) / (4. * hess[i, i])
            for j in range(i + 1, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :], *args, **kwargs) -
                              f(x + ee[i, :] - ee[j, :], *args, **kwargs) -
                              f(x - ee[i, :] + ee[j, :], *args, **kwargs) +
                              f(x - ee[i, :] - ee[j, :], *args, **kwargs)
                              ) / (4. * hess[j, i])
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _central2(f, fx, x, h, *args, **kwargs):
        """Eq. 8"""
        n = len(x)
        ee = np.diag(h)
        dtype = np.result_type(fx)
        g = np.empty(n, dtype=dtype)
        gg = np.empty(n, dtype=dtype)
        for i in range(n):
            g[i] = f(x + ee[i], *args, **kwargs)
            gg[i] = f(x - ee[i], *args, **kwargs)

        hess = np.empty((n, n), dtype=dtype)
        np.outer(h, h, out=hess)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :], *args, **kwargs) -
                              g[i] - g[j] + fx +
                              f(x - ee[i, :] - ee[j, :], *args, **kwargs) -
                              gg[i] - gg[j] + fx) / (2 * hess[j, i])
                hess[j, i] = hess[i, j]

        return hess

    @staticmethod
    def _forward(f, fx, x, h, *args, **kwargs):
        """Eq. 7"""
        n = len(x)
        ee = np.diag(h)
        dtype = np.result_type(fx)
        g = np.empty(n, dtype=dtype)
        for i in range(n):
            g[i] = f(x + ee[i, :], *args, **kwargs)

        hess = np.empty((n, n), dtype=dtype)
        np.outer(h, h, out=hess)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :], *args, **kwargs) -
                              g[i] - g[j] + fx) / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    def _backward(self, f, fx, x, h, *args, **kwargs):
        return self._forward(f, fx, x, -h, *args, **kwargs)


EPS = np.MachAr().eps


def _get_epsilon(x, s, epsilon, n):
    if epsilon is None:
        h = EPS**(1. / s) * np.maximum(np.abs(x), 0.1)
    else:
        if np.isscalar(epsilon):
            h = np.empty(n)
            h.fill(epsilon)
        else:  # pragma : no cover
            h = np.asarray(epsilon)
            if h.shape != x.shape:
                raise ValueError("If h is not a scalar it must have the same"
                                 " shape as x.")
    return h


def approx_fprime(x, f, epsilon=None, args=(), kwargs=None, centered=True):
    '''
    Gradient of function, or Jacobian if function fun returns 1d array

    Parameters
    ----------
    x : array
        parameters at which the derivative is evaluated
    fun : function
        `fun(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.
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

     Example
    -------
    >>> import numdifftools as nd

    #(nonlinear least squares)

    >>> xdata = np.arange(0,1,0.1)
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> np.allclose(fun([1, 2, 0.75]).shape, (10,))
    True
    >>> np.allclose(approx_fprime([1, 2, 0.75], fun), np.zeros((10,3)))
    True

    >>> fun2 = lambda x : x[0]*x[1]*x[2]**2
    >>> np.allclose(approx_fprime([1.,2.,3.], fun2), [[18., 9., 12.]])
    True

    >>> fun3 = lambda x : np.vstack((x[0]*x[1]*x[2]**2, x[0]*x[1]*x[2]))
    >>> np.allclose(approx_fprime([1., 2., 3.], fun3),
    ...            [[18., 9., 12.], [6., 3., 2.]])
    True
    >>> np.allclose(approx_fprime([4., 5., 6.], fun3),
    ...            [[180., 144., 240.], [30., 24., 20.]])
    True

    >>> np.allclose(approx_fprime(np.array([[1.,2.,3.], [4., 5., 6.]]).T, fun3),
    ...            [[[  18.,  180.],
    ...              [   9.,  144.],
    ...              [  12.,  240.]],
    ...             [[   6.,   30.],
    ...              [   3.,   24.],
    ...              [   2.,   20.]]])
    True
    '''
    kwargs = {} if kwargs is None else kwargs
    n = len(x)
    # TODO:  add scaled stepsize
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
    grad = grad.squeeze()
    axes = [0, 1, 2][:grad.ndim]
    axes[:2] = axes[1::-1]
    return np.transpose(grad, axes=axes).squeeze()

if __name__ == '__main__':
    from numdifftools.testing import test_docstrings
    test_docstrings()
