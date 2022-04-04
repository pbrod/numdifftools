"""
Created on 27. aug. 2015

@author: pab
Author: John D'Errico
e-mail: woodchips@rochester.rr.com
Release: 1.0
Release date: 5/23/2008

"""
from __future__ import absolute_import, division, print_function
from collections import namedtuple
from functools import partial
import warnings
import numpy as np
from numdifftools.step_generators import MinStepGenerator
from numdifftools.extrapolation import Richardson, dea3


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


class CStepGenerator(MinStepGenerator):

    """
    Generates a sequence of steps

    where
        steps = base_step * step_nom * (exp(1j*dtheta) * step_ratio) ** (i + offset)
    for i = 0, 1, ..., num_steps-1

    Parameters
    ----------
    base_step : float, array-like, default None
        Defines the minimum step, if None, the value is set to EPS**(1/scale)
    step_ratio : real scalar, optional, default 4.0
        Ratio between sequential steps generated.
    num_steps : scalar integer, optional,
        defines number of steps generated.
        If None the value is 2 * int(round(16.0/log(abs(step_ratio)))) + 1
    step_nom :  default maximum(log(exp(1)+|x|), 1)
        Nominal step where x is supplied at runtime through the __call__ method.
    offset : real scalar, optional, default 0
        offset to the base step
    use_exact_steps : boolean, default True.
        If true make sure exact steps are generated.
    scale : real scalar, default 1.2
        scale used in base step.
    path : 'radial' or 'spiral'
        Specifies the type of path to take the limit along. Default 'radial'.
    dtheta: real scalar, default pi/8
        If the path is 'spiral' it will follow an exponential spiral into the
        limit, with angular steps at dtheta radians.

    """

    def __init__(self, base_step=None, step_ratio=4.0, num_steps=None, step_nom=None,
                 offset=0, scale=1.2, **options):
        self.path = options.pop('path', 'radial')
        self.dtheta = options.pop('dtheta', np.pi / 8)
        super(CStepGenerator,
              self).__init__(base_step=base_step, step_ratio=step_ratio,
                             num_steps=num_steps, step_nom=step_nom, offset=offset, scale=scale,
                             **options)
        self._check_path()

    def _check_path(self):
        _assert(self.path in ['spiral', 'radial'], 'Invalid Path: {}'.format(str(self.path)))

    @property
    def step_ratio(self):
        """Ratio between sequential steps generated."""
        dtheta = self.dtheta
        _step_ratio = float(self._step_ratio)  # radial path
        if dtheta != 0:
            _step_ratio = np.exp(1j * dtheta) * _step_ratio  # a spiral path
        return _step_ratio

    @step_ratio.setter
    def step_ratio(self, step_ratio):
        self._step_ratio = step_ratio

    @property
    def dtheta(self):
        """Angular steps in radians used for the exponential spiral path."""
        radial_path = self.path[0].lower() == 'r'
        return 0 if radial_path else self._dtheta

    @dtheta.setter
    def dtheta(self, dtheta):
        self._dtheta = dtheta

    @property
    def num_steps(self):
        """The number of steps generated"""
        if self._num_steps is None:
            return 2 * int(np.round(16.0 / np.log(np.abs(self.step_ratio)))) + 1
        return self._num_steps

    @num_steps.setter
    def num_steps(self, num_steps):
        self._num_steps = num_steps


class _Limit(object):

    """Common methods and member variables"""

    info = namedtuple('info', ['error_estimate', 'final_step', 'index'])

    def __init__(self, step=None, **options):

        self.step = step, options

        self.richardson = Richardson(step_ratio=1.6, step=1, order=1, num_terms=2)

    @staticmethod
    def _parse_step_options(step):
        options = {}
        if isinstance(step, tuple) and isinstance(step[-1], dict):
            step, options = step
        return step, options

    @staticmethod
    def _step_generator(step, options):
        if hasattr(step, '__call__'):
            return step
        step_nom = None if step is None else 1
        return CStepGenerator(base_step=step, step_nom=step_nom, **options)

    @property
    def step(self):
        """The step spacing(s) used in the approximation"""
        return self._step

    @step.setter
    def step(self, step_options):
        step, options = self._parse_step_options(step_options)
        self._step = self._step_generator(step, options)

    @staticmethod
    def _get_arg_min(errors):
        shape = errors.shape
        try:
            arg_mins = np.nanargmin(errors, axis=0)
            min_errors = np.nanmin(errors, axis=0)
        except ValueError as msg:
            warnings.warn(str(msg))
            return np.arange(shape[1])

        for i, min_error in enumerate(min_errors):
            idx = np.flatnonzero(errors[:, i] == min_error)
            arg_mins[i] = idx[idx.size // 2]
        return np.ravel_multi_index((arg_mins, np.arange(shape[1])), shape)

    @staticmethod
    def _add_error_to_outliers(der, trim_fact=10):
        """
        discard any estimate that differs wildly from the
        median of all estimates. A factor of 10 to 1 in either
        direction is probably wild enough here. The actual
        trimming factor is defined as a parameter.
        """
        try:
            if np.any(np.isnan(der)):
                p25, median, p75 = np.nanpercentile(der, [25,50, 75], axis=0) 
            else:
                p25, median, p75 = np.percentile(der, [25,50, 75], axis=0)

            iqr = np.abs(p75 - p25)
        except ValueError as msg:
            warnings.warn(str(msg))
            return 0 * der

        a_median = np.abs(median)
        outliers = (((abs(der) < (a_median / trim_fact)) +
                    (abs(der) > (a_median * trim_fact))) * (a_median > 1e-8) +
                    ((der < p25 - 1.5 * iqr) + (p75 + 1.5 * iqr < der)))
        errors = outliers * np.abs(der - median)
        return errors

    @staticmethod
    def _get_best_estimate(der, errors, steps, shape):
        errors += _Limit._add_error_to_outliers(der)
        idx = _Limit._get_arg_min(errors)
        final_step = steps.flat[idx].reshape(shape)
        err = errors.flat[idx].reshape(shape)
        return der.flat[idx].reshape(shape), _Limit.info(err, final_step, idx)

    @staticmethod
    def _wynn_extrapolate(der, steps):
        der, errors = dea3(der[0:-2], der[1:-1], der[2:], symmetric=False)
        return der, errors, steps[2:]

    def _extrapolate(self, results, steps, shape):
        # if len(results) > 2:
        #     der0, errors0, steps0 = self._wynn_extrapolate(results, steps)
        #     if len(der0) > 0:
        #         der2, errors2, steps2 = self._wynn_extrapolate(der0, steps0)
        # else:
        der1, errors1, steps = self.richardson(results, steps)
        if len(der1) > 2:
            der1, errors1, steps = self._wynn_extrapolate(der1, steps)
        der, info = self._get_best_estimate(der1, errors1, steps, shape)
        return der, info

    @staticmethod
    def _vstack(sequence, steps):
        original_shape = np.shape(sequence[0])
        f_del = np.vstack([np.ravel(r) for r in sequence])
        one = np.ones(original_shape)
        h = np.vstack([np.ravel(one * step) for step in steps])
        _assert(f_del.size == h.size, 'fun did not return data of correct '
                'size (it must be vectorized)')
        return f_del, h, original_shape


class Limit(_Limit):

    """
    Compute limit of a function at a given point

    Parameters
    ----------
    fun : callable
        function fun(z, `*args`, `**kwds`) to compute the limit for z->z0.
        The function, fun, is assumed to return a result of the same shape and
        size as its input, `z`.
    step: float, complex, array-like or StepGenerator object, optional
        Defines the spacing used in the approximation.
        Default is CStepGenerator(base_step=step, **options)
    method : {'above', 'below'}
        defines if the limit is taken from `above` or `below`
    order: positive scalar integer, optional.
        defines the order of approximation used to find the specified limit.
        The order must be member of [1 2 3 4 5 6 7 8]. 4 is a good compromise.
    full_output: bool
        If true return additional info.
    options:
        options to pass on to CStepGenerator

    Returns
    -------
    limit_fz: array like
        estimated limit of f(z) as z --> z0
    info:
        Only given if full_output is True and contains the following:

        error estimate: ndarray
            95 % uncertainty estimate around the limit, such that
            abs(limit_fz - lim z->z0 f(z)) < error_estimate
        final_step: ndarray
            final step used in approximation

    Notes
    -----
    `Limit` computes the limit of a given function at a specified
    point, z0. When the function is evaluable at the point in question,
    this is a simple task. But when the function cannot be evaluated
    at that location due to a singularity, you may need a tool to
    compute the limit. `Limit` does this, as well as produce an
    uncertainty estimate in the final result.

    The methods used by `Limit` are Richardson extrapolation in a combination
    with Wynn's epsilon algorithm which also yield an error estimate.
    The user can specify the method order, as well as the path into
    z0. z0 may be real or complex. `Limit` uses a proportionally cascaded
    series of function evaluations, moving away from your point of evaluation
    along a path along the real line (or in the complex plane for complex z0 or
    step.) The `step_ratio` is the ratio used between sequential steps. The
    sign of step allows you to specify a limit from above or below. Negative
    values of step will cause the limit to be taken approaching z0 from below.

    A smaller `step_ratio` means that `Limit` will take more function
    evaluations to evaluate the limit, but the result will potentially be less
    accurate. The `step_ratio` MUST be a scalar larger than 1. A value in the
    range [2,100] is recommended. 4 seems a good compromise.

    Examples
    --------
     Compute the limit of sin(x)./x, at x == 0. The limit is 1.

    >>> import numpy as np
    >>> from numdifftools.limits import Limit
    >>> def f(x): return np.sin(x)/x
    >>> lim_f0, err = Limit(f, full_output=True)(0)
    >>> np.allclose(lim_f0, 1)
    True
    >>> np.allclose(err.error_estimate, 1.77249444610966e-15)
    True

    Compute the derivative of cos(x) at x == pi/2. It should
    be -1. The limit will be taken as a function of the
    differential parameter, dx.

    >>> x0 = np.pi/2;
    >>> def g(x): return (np.cos(x0+x)-np.cos(x0))/x
    >>> lim_g0, err = Limit(g, full_output=True)(0)
    >>> np.allclose(lim_g0, -1)
    True
    >>> err.error_estimate < 1e-14
    True

    Compute the residue at a first order pole at z = 0
    The function 1./(1-exp(2*z)) has a pole at z == 0.
    The residue is given by the limit of z*fun(z) as z --> 0.
    Here, that residue should be -0.5.

    >>> def h(z): return -z/(np.expm1(2*z))
    >>> lim_h0, err = Limit(h, full_output=True)(0)
    >>> np.allclose(lim_h0, -0.5)
    True
    >>> err.error_estimate < 1e-14
    True

    Compute the residue of function 1./sin(z)**2 at z = 0.
    This pole is of second order thus the residue is given by the limit of
    z**2*fun(z) as z --> 0.

    >>> def g(z): return z**2/(np.sin(z)**2)
    >>> lim_gpi, err = Limit(g, full_output=True)(0)
    >>> np.allclose(lim_gpi, 1)
    True
    >>> err.error_estimate < 1e-14
    True

    A more difficult limit is one where there is significant
    subtractive cancellation at the limit point. In the following
    example, the cancellation is second order. The true limit
    should be 0.5.

    >>> def k(x): return (x*np.exp(x)-np.expm1(x))/x**2
    >>> lim_k0,err = Limit(k, full_output=True)(0)
    >>> np.allclose(lim_k0, 0.5)
    True
    >>> err.error_estimate < 1.0e-8
    True

    >>> def h(x): return  (x-np.sin(x))/x**3
    >>> lim_h0, err = Limit(h, full_output=True)(0)
    >>> np.allclose(lim_h0, 1./6)
    True
    >>> err.error_estimate < 1e-8
    True

    """

    def __init__(self, fun, step=None, method='above', order=4, full_output=False, **options):
        super(Limit, self).__init__(step=step,  **options)
        self.fun = fun
        self.method = method
        self.order = order
        self.full_output = full_output

    def _fun(self, z, d_z, args, kwds):
        return self.fun(z + d_z, *args, **kwds)

    def _get_steps(self, x_i):
        return list(self.step(x_i))  # pylint: disable=not-callable

    def _set_richardson_rule(self, step_ratio, num_terms=2):
        self.richardson = Richardson(step_ratio=step_ratio, step=1, order=1,
                                     num_terms=num_terms)

    def _lim(self, f, z):
        sign = dict(forward=1, above=1, backward=-1, below=-1)[self.method]
        steps = [sign * step for step in self.step(z)]  # pylint: disable=not-callable
        # pylint: disable=no-member
        self._set_richardson_rule(self.step.step_ratio, self.order + 1)
        sequence = [f(z, h) for h in steps]
        results = self._vstack(sequence, steps)
        lim_fz, info = self._extrapolate(*results)
        return lim_fz, info

    def limit(self, x, *args, **kwds):
        """Return lim f(z) as z-> x"""
        z = np.asarray(x)
        f = partial(self._fun, args=args, kwds=kwds)
        f_z, info = self._lim(f, z)
        if self.full_output:
            return f_z, info
        return f_z

    def _call_lim(self, f_z, z, f):
        err = np.zeros_like(f_z, dtype=float)
        final_step = np.zeros_like(f_z)
        index = np.zeros_like(f_z, dtype=int)
        k = np.flatnonzero(np.isnan(f_z))
        if k.size > 0:
            lim_fz, info1 = self._lim(f, z.flat[k])
            zero = np.zeros(1, dtype=np.result_type(lim_fz))
            f_z = np.where(np.isnan(f_z), zero, f_z)
            np.put(f_z, k, lim_fz)
            if self.full_output:
                final_step = np.where(np.isnan(f_z), zero, final_step)
                np.put(final_step, k, info1.final_step)
                np.put(index, k, info1.index)
                np.put(err, k, info1.error_estimate)
        return f_z, self.info(err, final_step, index)

    def __call__(self, x, *args, **kwds):
        z = np.asarray(x)
        f = partial(self._fun, args=args, kwds=kwds)
        with np.errstate(divide='ignore', invalid='ignore'):
            f_z = f(z, 0)
            f_z, info = self._call_lim(f_z, z, f)

        if self.full_output:
            return f_z, info
        return f_z


class Residue(Limit):

    """
    Compute residue of a function at a given point

    Parameters
    ----------
    fun : callable
        function fun(z, `*args`, `**kwds`) to compute the Residue at z=z0.
        The function, fun, is assumed to return a result of the same shape and
        size as its input, `z`.
    step: float, complex, array-like or StepGenerator object, optional
        Defines the spacing used in the approximation.
        Default is CStepGenerator(base_step=step, **options)
    method : {'above', 'below'}
        defines if the limit is taken from `above` or `below`
    order: positive scalar integer, optional.
        defines the order of approximation used to find the specified limit.
        The order must be member of [1 2 3 4 5 6 7 8]. 4 is a good compromise.
    pole_order : scalar integer
        specifies the order of the pole at z0.
    full_output: bool
        If true return additional info.
    options:
        options to pass on to CStepGenerator

    Returns
    -------
    res_fz: array like
        estimated residue, i.e., limit of f(z)*(z-z0)**pole_order as z --> z0
        When the residue is estimated as approximately zero,
          the wrong order pole may have been specified.
    info: namedtuple,
        Only given if full_output is True and contains the following:

        error estimate: ndarray
            95 % uncertainty estimate around the residue, such that
            abs(res_fz - lim z->z0 f(z)*(z-z0)**pole_order) < error_estimate
            Large uncertainties here suggest that the wrong order
            pole was specified for f(z0).
        final_step: ndarray
            final step used in approximation

    Notes
    -----
    Residue computes the residue of a given function at a simple first order
    pole, or at a second order pole.

    The methods used by residue are polynomial extrapolants, which also yield
    an error estimate. The user can specify the method order, as well as the
    order of the pole.

    z0  - scalar point at which to compute the residue. z0 may be
          real or complex.

    See the document DERIVEST.pdf for more explanation of the
    algorithms behind the parameters of Residue. In most cases,
    the user should never need to specify anything other than possibly
    the PoleOrder.


    Examples
    --------
    A first order pole at z = 0

    >>> import numpy as np
    >>> from numdifftools.limits import Residue
    >>> def f(z): return -1./(np.expm1(2*z))
    >>> res_f, info = Residue(f, full_output=True)(0)
    >>> np.allclose(res_f, -0.5)
    True
    >>> info.error_estimate < 1e-14
    True

    A second order pole around z = 0 and z = pi
    >>> def h(z): return 1.0/np.sin(z)**2
    >>> res_h, info = Residue(h, full_output=True, pole_order=2)([0, np.pi])
    >>> np.allclose(res_h, 1)
    True
    >>> (info.error_estimate < 1e-10).all()
    True

    """

    def __init__(self, f, step=None, method='above', order=None, pole_order=1,
                 full_output=False, **options):
        if order is None:
            # MethodOrder will always = pole_order + 2
            order = pole_order + 2

        _assert(pole_order < order, 'order must be at least pole_order+1.')
        self.pole_order = pole_order

        super(Residue, self).__init__(f, step=step, method=method, order=order,
                                      full_output=full_output, **options)

    def _fun(self, z, d_z, args, kwds):
        return self.fun(z + d_z, *args, **kwds) * (d_z ** self.pole_order)

    def __call__(self, x, *args, **kwds):
        return self.limit(x, *args, **kwds)


if __name__ == '__main__':
    from numdifftools.testing import test_docstrings
    test_docstrings(__file__)
