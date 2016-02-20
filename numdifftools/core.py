# !/usr/bin/env python
"""numerical differentiation functions:

Derivative, Gradient, Jacobian, and Hessian

Author:      Per A. Brodtkorb

Created:     01.08.2008
Copyright:   (c) pab 2008
Licence:     New BSD

Based on matlab functions derivest.m gradest.m hessdiag.m, hessian.m
and jacobianest.m version 1.0 released 12/27/2006 by  John D'Errico
(e-mail: woodchips@rochester.rr.com)

Also based on the python functions approx_fprime, approx_fprime_cs,
approx_hess_cs, approx_hess1, approx_hess2 and approx_hess3 in the
statsmodels.tools.numdiff module released in 2014 written by Josef Perktold.

"""

from __future__ import division, print_function
import numpy as np
from collections import namedtuple
from numdifftools.multicomplex import bicomplex
from numdifftools.extrapolation import Richardson, dea3, convolve
from numdifftools.test_functions import get_function  # , function_names
from numpy import linalg
from scipy import misc
from scipy.ndimage.filters import convolve1d
import warnings

__all__ = ('dea3', 'Derivative', 'Jacobian', 'Gradient', 'Hessian', 'Hessdiag',
           'MinStepGenerator', 'MaxStepGenerator', 'Richardson',
           'directionaldiff')
# NOTE: we only do double precision internally so far
_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps
EPS = np.MachAr().eps
_SQRT_J = (1j + 1.0) / np.sqrt(2.0)  # = 1j**0.5

_CENTRAL_WEIGHTS_AND_POINTS = {
    (1, 3): (np.array([-1, 0, 1]) / 2.0, np.arange(-1, 2)),
    (1, 5): (np.array([1, -8, 0, 8, -1]) / 12.0, np.arange(-2, 3)),
    (1, 7): (np.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0, np.arange(-3, 4)),
    (1, 9): (np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0,
             np.arange(-4, 5)),
    (2, 3): (np.array([1, -2.0, 1]), np.arange(-1, 2)),
    (2, 5): (np.array([-1, 16, -30, 16, -1]) / 12.0, np.arange(-2, 3)),
    (2, 7): (np.array([2, -27, 270, -490, 270, -27, 2]) / 180.0,
             np.arange(-3, 4)),
    (2, 9): (np.array([-9, 128, -1008, 8064, -14350,
                      8064, -1008, 128, -9]) / 5040.0,
             np.arange(-4, 5))}


def fornberg_weights_all(x, x0, M=1):
    """
    Return finite difference weights_and_points for derivatives of all orders.

    Parameters
    ----------
    x : vector, length n
        x-coordinates for grid points
    x0 : scalar
        location where approximations are to be accurate
    m : scalar integer
        highest derivative that we want to find weights_and_points for

    Returns
    -------
    C :  array, shape n x m+1
        contains coefficients for the j'th derivative in column j (0 <= j <= m)

    See also:
    ---------
    fornberg_weights

    Reference
    ---------
    B. Fornberg (1998)
    "Calculation of weights_and_points in finite difference formulas",
    SIAM Review 40, pp. 685-691.

    http://www.scholarpedia.org/article/Finite_difference_method
    """
    N = len(x)
    if M >= N:
        raise ValueError('length(x) must be larger than m')

    c1, c4 = 1, x[0] - x0
    C = np.zeros((N, M + 1))
    C[0, 0] = 1
    for n in range(1, N):
        m = np.arange(0, min(n, M) + 1)
        c2, c5, c4 = 1, c4, x[n] - x0
        for v in range(n):
            c3 = x[n] - x[v]
            c2, c6, c7 = c2 * c3, m * C[v, m-1], C[v, m]
            C[v, m] = (c4 * c7 - c6) / c3
        C[n, m] = c1 * (c6 - c5 * c7) / c2
        c1 = c2
    return C


def fornberg_weights(x, x0, m=1):
    """
    Return weights for finite difference approximation of the m'th derivative
    U^m(x0), evaluated at x0, based on n values of U at x[0], x[1],... x[n-1]:

        U^m(x0) = sum weights[i] * U(x[i])

    Parameters
    ----------
    x : vector
        abscissas used for the evaluation for the derivative at x0.
    x0 : scalar
        location where approximations are to be accurate
    m : integer
        order of derivative. Note for m=0 this can be used to evaluate the
        interpolating polynomial itself.

    Notes
    -----
    The x values can be arbitrarily spaced but must be distinct and len(x) > m.

    The Fornberg algorithm is much more stable numerically than regular
    vandermonde systems for large values of n.

    See also
    --------
    fornberg_weights_all
    """
    return fornberg_weights_all(x, x0, m)[:, -1]


def _make_exact(h):
    """Make sure h is an exact representable number

    This is important when calculating numerical derivatives and is
    accomplished by adding 1 and then subtracting 1..
    """
    return (h + 1.0) - 1.0


def default_scale(method='forward', n=1, order=2):
    # is_odd = (n % 2) == 1
    high_order = int(n > 1 or order >= 4)
    order2 = max(order // 2-1, 0)
    n4 = n // 4
    return (dict(multicomplex=1.35, complex=1.35).get(method, 2.5) +
            int((n - 1)) * dict(multicomplex=0, complex=0.0).get(method, 1.3) +
            order2 * dict(central=3, forward=2, backward=2).get(method, 0) +
            # is_odd * dict(complex=2.65*int(n//2)).get(method, 0) +
            (n % 4 == 1) * high_order * dict(complex=3.65 + n4 * (5 + 1.5**n4)
                                             ).get(method, 0) +
            (n % 4 == 3) * dict(complex=3.65*2 + n4 * (5 + 2.1**n4)
                                ).get(method, 0) +
            (n % 4 == 2) * dict(complex=3.65 + n4 * (5 + 1.7**n4)
                                ).get(method, 0) +
            (n % 4 == 0) * dict(complex=(n//4) * (10 + 1.5*int(n > 10))
                                ).get(method, 0))


def valarray(shape, value=np.NaN, typecode=None):
    """Return an array of all value."""
    if typecode is None:
        typecode = bool
    out = np.ones(shape, dtype=typecode) * value

    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


def nom_step(x=None):
    """Return nominal step."""
    if x is None:
        return 1.0
    return np.maximum(np.log1p(np.abs(x)), 1.0)


def _default_base_step(x, scale, epsilon=None):
    if epsilon is None:
        h = EPS ** (1. / scale) * nom_step(x)
    else:
        h = valarray(x.shape, value=epsilon)
    return h


class MinStepGenerator(object):

    """
    Generates a sequence of steps

    where steps = base_step * step_ratio ** (np.arange(num_steps) + offset)

    Parameters
    ----------
    base_step : float, array-like, optional
        Defines the base step, if None, then base_step is set to
        EPS**(1/scale)*max(log(1+|x|), 1) where x is supplied at runtime
        through the __call__ method.
    step_ratio : real scalar, optional, default 2
        Ratio between sequential steps generated.
        Note: Ratio > 1
        If None then step_ratio is 2 for n=1 otherwise step_ratio is 1.6
    num_steps : scalar integer, optional, default  n + order - 1 + num_extrap
        defines number of steps generated. It should be larger than
        n + order - 1
    offset : real scalar, optional, default 0
        offset to the base step
    scale : real scalar, optional
        scale used in base step. If not None it will override the default
        computed with the default_scale function.
    """

    def __init__(self, base_step=None, step_ratio=2, num_steps=None,
                 offset=0, scale=None, num_extrap=0, use_exact_steps=True,
                 check_num_steps=True):
        self.base_step = base_step
        self.num_steps = num_steps
        self.step_ratio = step_ratio
        self.offset = offset
        self.scale = scale
        self.check_num_steps = check_num_steps
        self.use_exact_steps = use_exact_steps
        self.num_extrap = num_extrap

    def __repr__(self):
        class_name = self.__class__.__name__
        kwds = ['{0!s}={1!s}'.format(name, str(getattr(self, name)))
                for name in self.__dict__.keys()]
        return """{0!s}({1!s})""".format(class_name, ','.join(kwds))

    def _default_scale(self, method, n, order):
        scale = self.scale
        if scale is None:
            scale = default_scale(method, n, order)
        return scale

    def _default_base_step(self, xi, method, n, order=2):
        scale = self._default_scale(method, n, order)
        base_step = _default_base_step(xi, scale, self.base_step)
        if self.use_exact_steps:
            base_step = _make_exact(base_step)
        return base_step

    @staticmethod
    def _min_num_steps(method, n, order):
        num_steps = n + order - 1

        if method in ['central', 'central2', 'complex', 'multicomplex']:
            step = 2
            if method == 'complex':
                step = 4 if n > 2 or order >= 4 else 2
            num_steps = (n + order-1) // step
        return max(int(num_steps), 1)

    def _default_num_steps(self, method, n, order):
        min_num_steps = self._min_num_steps(method, n, order)
        if self.num_steps is not None:
            num_steps = int(self.num_steps)
            if self.check_num_steps:
                num_steps = max(num_steps, min_num_steps)
            return num_steps
        return min_num_steps + int(self.num_extrap)

    def _default_step_ratio(self, n):
        if self.step_ratio is None:
            step_ratio = {1: 2.0}.get(n, 1.6)
        else:
            step_ratio = float(self.step_ratio)
        if self.use_exact_steps:
            step_ratio = _make_exact(step_ratio)
        return step_ratio

    def __call__(self, x, method='central', n=1, order=2):
        xi = np.asarray(x)
        base_step = self._default_base_step(xi, method, n, order)
        step_ratio = self._default_step_ratio(n)

        num_steps = self._default_num_steps(method, n, order)
        offset = self.offset
        for i in range(num_steps-1, -1, -1):
            h = (base_step * step_ratio**(i + offset))
            if (np.abs(h) > 0).all():
                yield h


class MinMaxStepGenerator(object):
    """
    Generates a sequence of steps

    where
        steps = logspace(log10(step_min), log10(step_max), num_steps)

    Parameters
    ----------
    step_min : float, array-like, optional
       Defines the minimim step. Default value is:
           EPS**(1/scale)*max(log(1+|x|), 1)
       where x and scale are supplied at runtime through the __call__ method.
    step_max : real scalar, optional
        maximum step generated. Default value is:
            exp(log(step_min) * scale / (scale + 1.5))
    num_steps : scalar integer, optional
        defines number of steps generated.
    scale : real scalar, optional
        scale used in base step. If set to a value it will override the scale
        supplied at runtime.
    """

    def __init__(self, step_min=None, step_max=None, num_steps=10, scale=None,
                 num_extrap=0):
        self.step_min = step_min
        self.num_steps = num_steps
        self.step_max = step_max
        self.scale = scale
        self.num_extrap = num_extrap

    def __repr__(self):
        class_name = self.__class__.__name__
        kwds = ['{0!s}={1!s}'.format(name, str(getattr(self, name)))
                for name in self.__dict__.keys()]
        return """{0!s}({1!s})""".format(class_name, ','.join(kwds))


    def _steps(self, x):
        if self.scale is not None:
            scale = self.scale
        xi = np.asarray(x)
        step_min, step_max = self.step_min, self.step_max
        delta = _default_base_step(xi, scale, step_min)
        if step_min is None:
            step_min = (10 * EPS) ** (1. / scale)
        if step_max is None:
            step_max = np.exp(np.log(step_min) * scale / (scale + 1.5))
        steps = np.logspace(0, np.log10(step_max) - np.log10(step_min), self.num_steps)[:-1]
        return steps, delta

    def __call__(self, x, method='forward', n=1, order=None):
        steps, delta = self._steps(x)

        for step in steps:
            h = _make_exact(delta * step)
            if (np.abs(h) > 0).all():
                yield h


class MaxStepGenerator(MinStepGenerator):
    """
    Generates a sequence of steps

    where
        steps = base_step * step_ratio ** (-np.arange(num_steps) + offset)
        base_step = step_max * step_nom

    Parameters
    ----------
    max_step : float, array-like, optional default 2
       Defines the maximum step
    step_ratio : real scalar, optional, default 2
        Ratio between sequential steps generated.
        Note: Ratio > 1
    num_steps : scalar integer, optional, default  n + order - 1 + num_extrap
        defines number of steps generated. It should be larger than
        n + order - 1
    step_nom :  default maximum(log1p(abs(x)), 1)
        Nominal step.
    offset : real scalar, optional, default 0
        offset to the base step: max_step * nom_step
    """

    def __init__(self, step_max=2.0, step_ratio=2.0, num_steps=15,
                 step_nom=None, offset=0, num_extrap=0,
                 use_exact_steps=False, check_num_steps=True):
        self.base_step = None
        self.step_max = step_max
        self.step_ratio = step_ratio
        self.num_steps = num_steps
        self.step_nom = step_nom
        self.offset = offset
        self.num_extrap = num_extrap
        self.check_num_steps = check_num_steps
        self.use_exact_steps = use_exact_steps

    def _default_step_nom(self, x):
        if self.step_nom is None:
            return nom_step(x)
        return valarray(x.shape, value=self.step_nom)

    def _default_base_step(self, xi, method, n, order=1):
        base_step = self.base_step
        if base_step is None:
            base_step = self.step_max * self._default_step_nom(xi)
        if self.use_exact_steps:
            base_step = _make_exact(base_step)
        return base_step

    def __call__(self, x, method='forward', n=1, order=None):
        xi = np.asarray(x)

        offset = self.offset

        base_step = self._default_base_step(xi, method, n)
        step_ratio = self._default_step_ratio(n)

        num_steps = self._default_num_steps(method, n, order)
        for i in range(num_steps):
            h = base_step * step_ratio**(-i + offset)
            if (np.abs(h) > 0).all():
                yield h




_cmn_doc = """
    Calculate %(derivative)s with finite difference approximation

    Parameters
    ----------
    f : function
       function of one array f(x, `*args`, `**kwds`)
    step : float, array-like or StepGenerator object, optional
       Defines the spacing used in the approximation.
       Default is  MinStepGenerator(base_step=step, step_ratio=None)
       if step or method in in ['complex', 'multicomplex'], otherwise
       MaxStepGenerator(step_ratio=None, num_extrap=14)
       The results are extrapolated if the StepGenerator generate more than 3
       steps.
    method : {'central', 'complex', 'multicomplex', 'forward', 'backward'}
        defines the method used in the approximation%(extra_parameter)s
    full_output : bool, optional
        If `full_output` is False, only the derivative is returned.
        If `full_output` is True, then (der, r) is returned `der` is the
        derivative, and `r` is a Results object.

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


class _Derivative(object):

    info = namedtuple('info', ['error_estimate', 'final_step', 'index'])

    def __init__(self, f, step=None, method='central',  order=2, n=1,
                 full_output=False):
        self.f = f
        self.n = n
        self.order = order
        self.method = method
        self.full_output = full_output
        self.richardson_terms = 2
        self.step = self._make_generator(step)

    def _make_generator(self, step):
        if hasattr(step, '__call__'):
            return step
        if step is None and self.method not in ['complex', 'multicomplex']:
            return MaxStepGenerator(step_ratio=None, num_extrap=14)
        return MinStepGenerator(base_step=step, step_ratio=None, num_extrap=0)

    @staticmethod
    def _get_arg_min(errors):
        shape = errors.shape
        try:
            arg_mins = np.nanargmin(errors, axis=0)
            min_errors = np.nanmin(errors, axis=0)
        except ValueError as msg:
            warnings.warn(str(msg))
            ix = np.arange(shape[1])
            return ix

        for i, min_error in enumerate(min_errors):
            idx = np.flatnonzero(errors[:, i] == min_error)
            arg_mins[i] = idx[idx.size // 2]
        ix = np.ravel_multi_index((arg_mins, np.arange(shape[1])), shape)
        return ix

    @staticmethod
    def _add_error_to_outliers(der, trim_fact=10):
        try:
            median = np.nanmedian(der, axis=0)
            p75 = np.nanpercentile(der, 75, axis=0)
            p25 = np.nanpercentile(der, 25, axis=0)
            iqr = np.abs(p75-p25)
        except ValueError as msg:
            warnings.warn(str(msg))
            return 0 * der

        a_median = np.abs(median)
        outliers = (((abs(der) < (a_median / trim_fact)) +
                    (abs(der) > (a_median * trim_fact))) * (a_median > 1e-8) +
                    ((der < p25-1.5*iqr) + (p75+1.5*iqr < der)))
        errors = outliers * np.abs(der - median)
        return errors

    def _get_best_estimate(self, der, errors, steps, shape):
        errors += self._add_error_to_outliers(der)
        ix = self._get_arg_min(errors)
        final_step = steps.flat[ix].reshape(shape)
        err = errors.flat[ix].reshape(shape)
        return der.flat[ix].reshape(shape), self.info(err, final_step, ix)

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
    @staticmethod
    def _wynn_extrapolate(der, steps):
        der, errors = dea3(der[0:-2], der[1:-1], der[2:], symmetric=False)
        return der, errors, steps[2:]

    def _extrapolate(self, results, steps, shape):
        der, errors, steps = self.richardson(results, steps)
        if len(der) > 2:
            # der, errors, steps = self.richardson(results, steps)
            der, errors, steps = self._wynn_extrapolate(der, steps)
        der, info = self._get_best_estimate(der, errors, steps, shape)
        return der, info

    def _get_middle_name(self):
        middle = ''
        if self._is_even_derivative and self.method in ('central', 'complex'):
            middle = '_even'
        elif self._complex_high_order and self._is_odd_derivative:
            middle = '_odd'
        elif self.method == 'multicomplex' and self.n > 1:
            middle = '2'
            if self.n > 2:
                raise ValueError('Multicomplex method only support first '
                                 'and second order derivatives.')
        return middle


    def _get_last_name(self):
        last = ''
        if (self.method in ('complex') and self._is_fourth_derivative or 
            self._complex_high_order and self._is_third_derivative):
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
        return getattr(self, name), self.f

    def _get_steps(self, xi):
        method, n, order = self.method, self.n, self._method_order
        return [step for step in self.step(xi, method, n, order)]

    @property
    def _is_odd_derivative(self):
        return self.n % 2 == 1

    @property
    def _is_even_derivative(self):
        return self.n % 2 == 0

    @property
    def _is_third_derivative(self):
        return self.n % 4 == 3

    @property
    def _is_fourth_derivative(self):
        return self.n % 4 == 0

    def _eval_first_condition(self):
        even_derivative = self._is_even_derivative
        return ((even_derivative and self.method in ('central', 'central2')) or
                self.method in ['forward', 'backward'] or
                self.method == 'complex' and self._is_fourth_derivative)

    def _eval_first(self, f, x, *args, **kwds):
        if self._eval_first_condition():
            return f(x, *args, **kwds)
        return 0.0

    @staticmethod
    def _vstack(sequence, steps):
        # sequence = np.atleast_2d(sequence)
        original_shape = np.shape(sequence[0])
        f_del = np.vstack(list(np.ravel(r)) for r in sequence)
        h = np.vstack(list(np.ravel(np.ones(original_shape)*step))
                      for step in steps)
        if f_del.size != h.size:
            raise ValueError('fun did not return data of correct size ' +
                             '(it must be vectorized)')
        return f_del, h, original_shape

    @staticmethod
    def _compute_step_ratio(steps):
        if len(steps) < 2:
            return 1
        return np.unique(steps[0]/steps[1]).mean()

    def __call__(self, x, *args, **kwds):
        xi = np.asarray(x)
        results = self._derivative(xi, args, kwds)
        derivative, info = self._extrapolate(*results)
        if self.full_output:
            return derivative, info
        return derivative


class Derivative(_Derivative):
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
    """
    Find the n-th derivative of a function at a point.

    Given a function, use a difference formula with spacing `dx` to
    compute the `n`-th derivative at `x0`.

    Parameters
    ----------
    f : function
        Input function.
    x0 : float
        The point at which `n`-th derivative is found.
    dx : float, optional
        Spacing.
    method : Method of estimation.  Valid options are:
        'central', 'forward' or 'backward'.          (Default 'central')
    n : int, optional (Default 1)
        Order of the derivative.
    order : int, optional       (Default 2)
        defining order of basic method used.
        For 'central' methods, it must be an even number eg. [2,4].

    Notes
    -----
    Decreasing the step size too small can result in round-off error.

    Note on order: higher order methods will generally be more accurate,
             but may also suffer more from numerical problems. First order
             methods would usually not be recommended.
    Complex methods are usually the most accurate provided the function to
        differentiate is analytic. The complex-step methods also requires fewer
        steps than the other methods and can work very close to the support of
        a function. Central difference methods are almost as accurate and has
        no restriction on type of function, but sometimes one can only allow
        evaluation in forward or backward direction.


    """
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

        if n == 0:
            self._derivative = self._derivative_zero_order
        else:
            self._derivative = self._derivative_nonzero_order

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
        nterms : scalar, integer
            number of terms
        """
        try:
            step = [1, 2, 2, 4, 4, 4, 4][parity]
        except Exception as e:
            msg = '{0!s}. Parity must be 0, 1, 2, 3, 4, 5 or 6! ({1:d})'.format(str(e),
                                                                      parity)
            raise ValueError(msg)
        inv_sr = 1.0 / step_ratio
        offset = [1, 1, 2, 2, 4, 1, 3][parity]
        c0 = [1.0, 1.0, 1.0, 2.0, 24.0, 1.0, 6.0][parity]
        c = c0/misc.factorial(np.arange(offset, step * nterms + offset, step))
        [i, j] = np.ogrid[0:nterms, 0:nterms]
        return np.atleast_2d(c[j] * inv_sr ** (i * (step * j + offset)))

    def _flip_fd_rule(self):
        n = self.n
        return ((self._is_even_derivative and (self.method == 'backward')) or
                (self.method == 'complex' and (n % 8 in [3, 4, 5, 6])))


    def _parity(self, method, order, method_order):
        parity = 0
        if (method.startswith('central') or
            (method.startswith('complex') and self.n == 1 and
                method_order < 4)):
            parity = (order % 2) + 1
        elif method == 'complex':
            if self._is_odd_derivative:
                parity = 6 if self._is_third_derivative else 5
            else:
                parity = 4 if self._is_fourth_derivative else 3
        return parity

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

        if self._flip_fd_rule():
            fd_rule *= -1
        return fd_rule

    def _apply_fd_rule(self, fd_rule, sequence, steps):
        """
        Return derivative estimates of f at x0 for a sequence of stepsizes h

        Member variables used
        ---------------------
        n
        """
        f_del, h, original_shape = self._vstack(sequence, steps)

        ne = h.shape[0]
        if ne < fd_rule.size:
            raise ValueError('num_steps (%d) must  be larger than '
                             '(%d) n + order - 1 = %d + %d -1'
                             ' (%s)' % (ne, fd_rule.size, self.n, self.order,
                                        self.method)
                             )
        nr = (fd_rule.size-1)
        f_diff = convolve(f_del, fd_rule[::-1], axis=0, origin=nr//2)

        der_init = f_diff / (h ** self.n)
        ne = max(ne - nr, 1)
        return der_init[:ne], h[:ne], original_shape

    def _derivative_zero_order(self, xi, args, kwds):
        steps = [np.zeros_like(xi)]
        results = [self.f(xi, *args, **kwds)]
        self.set_richardson_rule(2, 0)
        return self._vstack(results, steps)

    def _derivative_nonzero_order(self, xi, args, kwds):
        diff, f = self._get_functions()
        steps = self._get_steps(xi)
        fxi = self._eval_first(f, xi, *args, **kwds)
        results = [diff(f, fxi, xi, h, *args, **kwds) for h in steps]
        step_ratio = self._compute_step_ratio(steps)

        self.set_richardson_rule(step_ratio, self.richardson_terms)
        fd_rule = self._get_finite_difference_rule(step_ratio)
        return self._apply_fd_rule(fd_rule, results, steps)

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
        return (f(x0i + h, *args, **kwds) - f_x0i)

    @staticmethod
    def _backward(f, f_x0i, x0i, h, *args, **kwds):
        return (f_x0i - f(x0i - h, *args, **kwds))

    @staticmethod
    def _complex(f, fx, x, h, *args, **kwds):
        return f(x + 1j * h, *args, **kwds).imag

    @staticmethod
    def _complex_odd(f, fx, x, h, *args, **kwds):
        ih = h * _SQRT_J
        return ((_SQRT_J/2.) * (f(x + ih, *args, **kwds) -
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
        z = bicomplex(x + 1j * h, 0)
        return f(z, *args, **kwds).imag

    @staticmethod
    def _multicomplex2(f, fx, x, h, *args, **kwds):
        z = bicomplex(x + 1j * h, h)
        return f(z, *args, **kwds).imag12


def directionaldiff(f, x0, vec, **options):
    """
    Return directional derivative of a function of n variables

    Parameters
    ----------
    f: function
        analytical function to differentiate.
    x0: array
        vector location at which to differentiate f. If x0 is an nxm array,
        then fun is assumed to be a function of n*m variables.
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
    if x0.size != vec.size:
        raise ValueError('vec and x0 must be the same shapes')

    vec = np.reshape(vec/np.linalg.norm(vec.ravel()), x0.shape)
    return Derivative(lambda t: f(x0+t*vec), **options)(0)



class Gradient(Derivative):
    def __init__(self, f, step=None, method='central', order=2,
                 full_output=False):
        super(Gradient, self).__init__(f, step=step, method=method, n=1,
                                       order=order, full_output=full_output)
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

    @staticmethod
    def _central(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + hi, *args, **kwds) - f(x - hi, *args, **kwds)) / 2.0
                    for hi in increments]
        return np.array(partials).T

    @staticmethod
    def _backward(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h
        partials = [(fx - f(x - hi, *args, **kwds)) for hi in increments]
        return np.array(partials).T

    @staticmethod
    def _forward(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + hi, *args, **kwds) - fx) for hi in increments]
        return np.array(partials).T

    @staticmethod
    def _complex(f, fx, x, h, *args, **kwds):
        # From Guilherme P. de Freitas, numpy mailing list
        # http://mail.scipy.org/pipermail/numpy-discussion/2010-May/050250.html
        n = len(x)
        increments = np.identity(n) * 1j * h
        partials = [f(x + ih, *args, **kwds).imag for ih in increments]
        return np.array(partials).T

    @staticmethod
    def _complex_odd(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * _SQRT_J * h
        partials = [((_SQRT_J/2.) * (f(x + ih, *args, **kwds) -
                                     f(x - ih, *args, **kwds))).imag
                    for ih in increments]
        return np.array(partials).T

    @staticmethod
    def _multicomplex(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * 1j * h
        partials = [f(bicomplex(x + hi, 0), *args, **kwds).imag
                    for hi in increments]
        return np.array(partials).T

    def __call__(self, x, *args, **kwds):
        return super(Gradient, self).__call__(np.atleast_1d(x), *args, **kwds)


class Jacobian(Gradient):
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

    If f returns a 1d array, it returns a Jacobian. If a 2d array is returned
    by f (e.g., with a value for each observation), it returns a 3d array
    with the Jacobian of each observation with shape xk x nobs x xk. I.e.,
    the Jacobian of the first observation would be [:, 0, :]
    """, example="""
    Example
    -------
    >>> import numdifftools as nd

    #(nonlinear least squares)

    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2

    >>> Jfun = nd.Jacobian(fun)
    >>> val = Jfun([1,2,0.75])
    >>> np.allclose(val, np.zeros((10,3)))
    True

    >>> fun2 = lambda x : x[0]*x[1]*x[2] + np.exp(x[0])*x[1]
    >>> Jfun3 = nd.Jacobian(fun2)
    >>> Jfun3([3.,5.,7.])
    array([ 135.42768462,   41.08553692,   15.        ])
    """, see_also="""
    See also
    --------
    Derivative, Hessian, Gradient
    """)


class Hessdiag(Derivative):
    def __init__(self, f, step=None, method='central', order=2,
                 full_output=False):
        super(Hessdiag, self).__init__(f, step=step, method=method, n=2,
                                       order=order, full_output=full_output)
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

    @staticmethod
    def _central2(f, fx, x, h, *args, **kwds):
        """Eq. 8"""
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + 2*hi, *args, **kwds) +
                    f(x - 2*hi, *args, **kwds) + 2*fx -
                    2*f(x + hi, *args, **kwds) -
                    2*f(x - hi, *args, **kwds)) / 4.0
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
        partials = [(fx - f(x - hi, *args, **kwds)) for hi in increments]
        return np.array(partials)

    @staticmethod
    def _forward(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + hi, *args, **kwds) - fx) for hi in increments]
        return np.array(partials)

    @staticmethod
    def _multicomplex2(f, fx, x, h, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h
        partials = [f(bicomplex(x + 1j * hi, hi), *args, **kwds).imag12
                    for hi in increments]
        return np.array(partials)

    @staticmethod
    def _complex_even(f, fx, x, h, *args, **kwargs):
        n = len(x)
        increments = np.identity(n) * h * (1j+1) / np.sqrt(2)
        partials = [(f(x + hi, *args, **kwargs) +
                     f(x - hi, *args, **kwargs)).imag
                    for hi in increments]
        return np.array(partials)

    def __call__(self, x, *args, **kwds):
        return super(Hessdiag, self).__call__(np.atleast_1d(x), *args, **kwds)


class Hessian(_Derivative):
    def __init__(self, f, step=None, method='central', full_output=False):
        order = dict(backward=1, forward=1, complex=2).get(method, 2)
        super(Hessian, self).__init__(f, n=2, step=step, method=method,
                                      order=order, full_output=full_output)

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
        \quad  ((f(x + d_j e_j + d_k e_k) - f(x + d_j e_j - d_k e_k)) -  (f(x - d_j e_j + d_k e_k) - f(x - d_j e_j - d_k e_k)) / (4 d_j d_k)
        :label: 9

    .. math::
        imag(f(x + i d_j e_j + d_k e_k) - f(x + i d_j e_j - d_k e_k)) /(2 d_j d_k)
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

    @staticmethod
    def _complex_high_order():
        return False

    def _derivative(self, xi, args, kwds):
        xi = np.atleast_1d(xi)
        diff, f = self._get_functions()
        steps = self._get_steps(xi)

        fxi = self._eval_first(f, xi, *args, **kwds)
        results = [diff(f, fxi, xi, h, *args, **kwds) for h in steps]
        step_ratio = self._compute_step_ratio(steps)
        self.set_richardson_rule(step_ratio, self.richardson_terms)
        return self._vstack(results, steps)

    @staticmethod
    def _complex_even(f, fx, x, h, *args, **kwargs):
        """
        Calculate Hessian with complex-step derivative approximation

        The stepsize is the same for the complex and the finite difference part
        """
        n = len(x)
        # h = _default_base_step(x, 3, base_step, n)
        ee = np.diag(h)
        hes = 2. * np.outer(h, h)

        for i in range(n):
            for j in range(i, n):
                hes[i, j] = (f(x + 1j * ee[i] + ee[j], *args, **kwargs) -
                             f(x + 1j * ee[i] - ee[j], *args, **kwargs)
                             ).imag / hes[j, i]
                hes[j, i] = hes[i, j]
        return hes

    @staticmethod
    def _multicomplex2(f, fx, x, h, *args, **kwargs):
        """Calculate Hessian with bicomplex-step derivative approximation"""
        n = len(x)
        ee = np.diag(h)
        hess = np.outer(h, h)
        for i in range(n):
            for j in range(i, n):
                zph = bicomplex(x + 1j * ee[i, :], ee[j, :])
                hess[i, j] = (f(zph, *args, **kwargs)).imag12 / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _central_even(f, fx, x, h, *args, **kwargs):
        """Eq 9."""
        n = len(x)
        # h = _default_base_step(x, 4, base_step, n)
        ee = np.diag(h)
        hess = np.outer(h, h)

        for i in range(n):
            hess[i, i] = (f(x + 2*ee[i, :], *args, **kwargs) - 2*fx +
                          f(x - 2*ee[i, :], *args, **kwargs)
                          ) / (4. * hess[i, i])
            for j in range(i+1, n):
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
        # NOTE: ridout suggesting using eps**(1/4)*theta
        # h = _default_base_step(x, 3, base_step, n)
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


def test_docstrings():
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':  # pragma : no cover
    test_docstrings()
