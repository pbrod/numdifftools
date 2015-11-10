"""
Created on 27. aug. 2015

@author: pab
Author: John D'Errico
e-mail: woodchips@rochester.rr.com
Release: 1.0
Release date: 5/23/2008

"""
from __future__ import division, print_function
import numpy as np
from collections import namedtuple
import warnings
from numdifftools.extrapolation import Richardson, dea3, EPS
_EPS = EPS


def _make_exact(h):
    """Make sure h is an exact representable number

    This is important when calculating numerical derivatives and is
    accomplished by adding 1 and then subtracting 1..
    """
    return (h + 1.0) - 1.0


def valarray(shape, value=np.NaN, typecode=None):
    """Return an array of all value."""
    if typecode is None:
        typecode = bool
    out = np.ones(shape, dtype=typecode) * value

    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


def nom_step(x=None):
    """Return nominal step"""
    if x is None:
        return 1.0
    return np.maximum(np.log1p(np.abs(x)), 1.0)


def _default_base_step(x, scale, epsilon=None):
    if epsilon is None:
        h = _EPS ** (1. / scale) * nom_step(x)
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
    step_ratio : real scalar, optional, default 4
        Ratio between sequential steps generated.
    num_steps : scalar integer, optional, default  n + order - 1 + num_extrap
        defines number of steps generated. It should be larger than
        n + order - 1
    offset : real scalar, optional, default 0
        offset to the base step
    scale : real scalar, optional
        scale used in base step. If not None it will override the default
        computed with the default_scale function.
    """

    def __init__(self, base_step=None, step_ratio=4.0, num_steps=None,
                 offset=0, scale=1.2, use_exact_steps=True):
        self.base_step = base_step
        self.num_steps = num_steps
        self.step_ratio = step_ratio
        self.offset = offset
        self.scale = scale
        self.use_exact_steps = use_exact_steps

    def __repr__(self):
        class_name = self.__class__.__name__
        kwds = ['%s=%s' % (name, str(getattr(self, name)))
                for name in self.__dict__.keys()]
        return """%s(%s)""" % (class_name, ','.join(kwds))

    def _default_base_step(self, xi):
        scale = self.scale
        base_step = _default_base_step(xi, scale, self.base_step)
        if self.use_exact_steps:
            base_step = _make_exact(base_step)
        return base_step

    def _default_num_steps(self):
        if self.num_steps is None:
            return 2 * int(np.round(np.log(2e7)/np.log(self.step_ratio))) + 1
        return self.num_steps

    def _default_step_ratio(self):
        step_ratio = float(self.step_ratio)
        if self.use_exact_steps:
            step_ratio = _make_exact(step_ratio)
        return step_ratio

    def __call__(self, x):
        xi = np.asarray(x)
        base_step = self._default_base_step(xi)
        step_ratio = self._default_step_ratio()
        num_steps = self._default_num_steps()
        offset = self.offset
        for i in range(num_steps-1, -1, -1):
            h = (base_step * step_ratio**(i + offset))
            if (np.abs(h) > 0).all():
                yield h


class Limit(object):
    """
    Compute limit of a function at a given point

    Parameters
    ----------
    f : callable
        function of one array f(z, `*args`, `**kwds`) to compute the limit for.
        The function, f, is assumed to return a result of the same shape and
        size as its input, `z`.
    step: float, complex, array-like or StepGenerator object, optional
        Defines the spacing used in the approximation.
        Default is  MinStepGenerator(base_step=step, step_ratio=4)
    method : {'above', 'below'}
        defines if the limit is taken from `above` or `below`
    order: positive scalar integer, optional.
        defines the order of approximation used to find the specified limit.
        The order must be member of [1 2 3 4 5 6 7 8]. 4 is a good compromise.

    Returns
    -------
    limit_fz: array like
        estimated limit of f(z) as z --> z0
    info:
        Only given if full_output is True and contains the following:
        error estimate: ndarray
            95 uncertainty estimate around the limit, such that
            abs(limit_fz - f(z0)*(z-z0)) < error_estimate
        final_step: ndarray
            final step used in approximation

    Description
    -----------
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

    Example
    -------
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

    A more difficult limit is one where there is significant
    subtractive cancellation at the limit point. In the following
    example, the cancellation is second order. The true limit
    should be 0.5.

    >>> def k(x): return (x*np.exp(x)-np.exp(x)+1)/x**2
    >>> lim_k0,err = Limit(k, full_output=True)(0)
    >>> np.allclose(lim_k0, 0.5)
    True
    >>> np.allclose(err.error_estimate, 7.4e-9)
    True

    >>> def h(x): return  (x-np.sin(x))/x**3
    >>> lim_h0, err = Limit(h, full_output=True)(0)
    >>> lim_h0, err
    """

    info = namedtuple('info', ['error_estimate', 'final_step', 'index'])

    def __init__(self, f, step=None, method='above', order=4,
                 full_output=False):
        self.f = f
        self.method = method
        self.order = order
        self.full_output = full_output
        self.step = self._make_generator(step)

    def _make_generator(self, step):
        if hasattr(step, '__call__'):
            return step
        return MinStepGenerator(base_step=step)

    def _get_arg_min(self, errors):
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

    def _add_error_to_outliers(self, der, trim_fact=10):
        # discard any estimate that differs wildly from the
        # median of all estimates. A factor of 10 to 1 in either
        # direction is probably wild enough here. The actual
        # trimming factor is defined as a parameter.
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

    def _set_richardson_rule(self, step_ratio, num_terms=2):
        self._richardson_extrapolate = Richardson(step_ratio=step_ratio,
                                                  step=1, order=1,
                                                  num_terms=num_terms)

    def _wynn_extrapolate(self, der, steps):
        der, errors = dea3(der[0:-2], der[1:-1], der[2:], symmetric=False)
        return der, errors, steps[2:]

    def _extrapolate(self, results, steps, shape):
        der1, errors1, steps = self._richardson_extrapolate(results, steps)
        if len(der1) > 2:
            # der, errors, steps = self._richardson_extrapolate(results, steps)
            der1, errors1, steps = self._wynn_extrapolate(der1, steps)
        der, info = self._get_best_estimate(der1, errors1, steps, shape)
        return der, info

    def _get_steps(self, xi):
        return [step for step in self.step(xi)]

    def _vstack(self, sequence, steps):
        # sequence = np.atleast_2d(sequence)
        original_shape = np.shape(sequence[0])
        f_del = np.vstack(list(np.ravel(r)) for r in sequence)
        h = np.vstack(list(np.ravel(np.ones(original_shape)*step))
                      for step in steps)
        if f_del.size != h.size:
            raise ValueError('fun did not return data of correct size ' +
                             '(it must be vectorized)')
        return f_del, h, original_shape

    def _lim(self, f, z, args, kwds):
        sign = dict(forward=1, above=1, backward=-1, below=-1)[self.method]
        steps = [sign * step for step in self.step(z)]

        self._set_richardson_rule(self.step.step_ratio, self.order + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sequence = [f(z + h, *args, **kwds) for h in steps]
        results = self._vstack(sequence, steps)
        lim_fz, info = self._extrapolate(*results)
        return lim_fz, info

    def limit(self, x, *args, **kwds):
        z = np.asarray(x)
        fz, info = self._lim(self.f, z, args, kwds)
        if self.full_output:
            return fz, info
        return fz

    def __call__(self, x, *args, **kwds):
        z = np.asarray(x)
        f = self.f
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fz = f(z, *args, **kwds)

        err = np.zeros_like(fz, )
        final_step = np.zeros_like(fz)
        index = np.zeros_like(fz, dtype=int)
        k = np.flatnonzero(np.isnan(fz))
        if k.size > 0:
            fz = np.where(np.isnan(fz), 0, fz)
            lim_fz, info1 = self._lim(f, z.flat[k], args, kwds)
            np.put(fz, k, lim_fz)
            if self.full_output:
                np.put(final_step, k, info1.final_step)
                np.put(index, k, info1.index)
                np.put(err, k, info1.error_estimate)

        if self.full_output:
            return fz, self.info(err, final_step, index)
        return fz


# class Residue(Limit):
#     """function [res,errest] = residueEst(fun,z0,varargin)
#     residueEst: residue of fun at z0 with an error estimate,
# 1st or 2nd order pole
#     usage: [res,errest] = residueEst(fun,z0)
#     usage: [res,errest] = residueEst(fun,z0,prop1,val1,prop2,val2,...)
#
#     ResidueEst computes the residue of a given function at a
#     simple first order pole, or at a second order pole.
#
#     The methods used by residueEst are polynomial extrapolants,
#     which also yield an error estimate. The user can specify the
#     method order, as well as the order of the pole. For more
#     information on the exact methods, see the pdf file for my
#     residueEst suite of codes.
#
#     Finally, While I have not written this function for the
#     absolute maximum speed, speed was a major consideration
#     in the algorithmic design. Maximum accuracy was my main goal.
#
#
#     Arguments (input)
#     fun - function to compute the residue for. May be an inline
#           function, anonymous, or an m-file. If there are additional
#           parameters to be passed into fun, then use of an anonymous
#           function is recommended.
#
#           fun should be vectorized to allow evaluation at multiple
#           locations at once. This will provide the best possible
#           speed. IF fun is not so vectorized, then you MUST set
#           'vectorized' property to 'no', so that residueEst will
#           then call your function sequentially instead.
#
#           Fun is assumed to return a result of the same
#           shape and size as its input.
#
#     z0  - scalar point at which to compute the residue. z0 may be
#           real or complex.
#
#     Additional inputs must be in the form of property/value pairs.
#     Properties are character strings. They may be shortened
#     to the extent that they are unambiguous. Properties are
#     not case sensitive. Valid property names are:
#
#     'PoleOrder', 'MethodOrder', 'Vectorized' 'StepRatio', 'MaxStep'
#     'Path', 'DZ',
#
#     All properties have default values, chosen as intelligently
#     as I could manage. Values that are character strings may
#     also be unambiguously shortened. The legal values for each
#     property are:
#
#     'PoleOrder' - specifies the order of the pole at z0.
#           Must be 1, 2 or 3.
#
#           DEFAULT: 1 (first order pole)
#
#     'Vectorized' - residueEst will normally assume that your
#           function can be safely evaluated at multiple locations
#           in a single call. This would minimize the overhead of
#           a loop and additional function call overhead. Some
#           functions are not easily vectorizable, but you may
#           (if your matlab release is new enough) be able to use
#           arrayfun to accomplish the vectorization.
#
#           When all else fails, set the 'vectorized' property
#           to 'no'. This will cause residueEst to loop over the
#           successive function calls.
#
#           DEFAULT: 'yes'
#
#     'Path' - Specifies the type of path to take the limit along.
#           Must be either 'spiral' or 'radial'. Spiral paths
#           will follow an exponential spiral into the pole, with
#           angular steps at pi/8 radians.
#
#           DEFAULT: 'radial'
#
#     'DZ' - Nominal step away from z0 taken in the estimation
#           All samples of fun will be taken at some path away
#           from zo, along the path z0 + dz. dz may be complex.
#
#           DEFAULT: 1e8*eps(z0)
#
#     'StepRatio' - ResidueEst uses a proportionally cascaded
#           series of function evaluations, moving away from your
#           point of evaluation along a path in the complex plane.
#           The StepRatio is the ratio used between sequential steps.
#
#           DEFAULT: 4
#
#
#     See the document DERIVEST.pdf for more explanation of the
#     algorithms behind the parameters of residueEst. In most cases,
#     I have chosen good values for these parameters, so the user
#     should never need to specify anything other than possibly
#     the PoleOrder. I've also tried to make my code robust enough
#     that it will not need much. But complete flexibility is in
#     there for your use.
#
#
#     Arguments: (output)
#     residue - residue estimate at z0.
#
#           When the residue is estimated as approximately zero,
#           the wrong order pole may have been specified.
#
#     errest - 95 uncertainty estimate around the residue, such that
#
#           abs(residue - fun(z0)*(z-z0)) < erest(j)
#
#           Large uncertainties here suggest that the wrong order
#           pole was specified for fun(z0).
#
#
#     Example:
#     A first order pole at z = 0
#
#     [r,e]=residueEst(@(z) 1./(1-exp(2*z)),0)
#
#     r =
#             -0.5
#
#     e =
#       4.5382e-12
#
#     Example:
#     A second order pole around z = pi
#
#     [r,e]=residueEst(@(z) 1./(sin(z).^2),pi,'poleorder',2)
#
#     r =
#                1
#
#     e =
#       2.6336e-11
#     """
#
#     def __init__(self, f, dz=None, order=None, pole_order=1, max_step=1000,
#                  step_ratio=2.0, path='radial', dtheta=np.pi/8):
#         if order is None:
#             order = pole_order + 1
#
#         if order <= pole_order:
#             raise ValueError('MethodOrder must be at least PoleOrder+1.')
#         if path not in ['spiral', 'radial']:
#             raise ValueError('Invalid Path: %s' % str(path))
#
#
#     # supply a default step?
#     if isempty(par.DZ)
#       if z0 == 0
#         # special case for zero
#         par.DZ = 1e8*eps(1);
#       else
#         par.DZ = 1e8*eps(z0);
#       end
#     elseif numel(par.DZ)>1
#       error('DZ must be scalar if supplied')
#     end
#
#     # MethodOrder will always = PoleOrder + 2
#     if isempty(par.MethodOrder)
#       par.MethodOrder = par.PoleOrder+2;
#     end
#
#     # if a radial path
#     if (lower(par.Path(1)) == 'r')
#       # a radial path. Just override any DTheta.
#       par.DTheta = 0;
#     else
#       # a spiral path
#       # par.DTheta has a default of pi/8 (radians)
#     end
#
#     # Define the samples to use along a linear path
#     k = (-15:15)';
#     theta = par.DTheta*k;
#     delta = par.DZ*exp(sqrt(-1)*theta).*(par.StepRatio.^k);
#     ndel = length(delta);
#     Z = z0 + delta;
#
#     # sample the function at these sample points
#     if strcmpi(par.Vectorized,'yes')
#       # fun is supposed to be vectorized.
#       fz = fun(Z);
#       fz = fz(:);
#       if numel(fz) ~= ndel
#         error('fun did not return a result of the proper size. Perhaps not properly vectorized?')
#       end
#     else
#       # evaluate in a loop
#       fz = zeros(size(Z));
#       for i = 1:ndel
#         fz(i) = fun(Z(i));
#       end
#     end
#
#     # multiply the sampled function by (Z - z0).^par.PoleOrder
#     fz = fz.*(delta.^par.PoleOrder);
#
#     # replicate the elements of fz into a sliding window
#     m = par.MethodOrder;
#     fz = fz(repmat((1:(ndel-m)),m+1,1) + repmat((0:m)',1,ndel-m));
#
#     # generate the general extrapolation rule
#     d = par.StepRatio.^((0:m)'-m/2);
#     A = repmat(d,1,m).^repmat(0:m-1,m+1,1);
#     [qA,rA] = qr(A,0);
#
#     # compute the various estimates of the prediction polynomials.
#     polycoef = rA\(qA'*fz);
#
#     # predictions for each model
#     pred = A*polycoef;
#     # and residual standard errors
#     ser = sqrt(sum((pred - fz).^2,1));
#
#     # the actual extrapolated estimates are just the first row of polycoef
#     # for a first order pole. For a second order pole, we need the first
#     # lim_fz, so we need the second row. Higher order poles are not
#     # estimable using this method due to numerical problems.
#     switch par.PoleOrder
#       case 1
#         residue_estimates = polycoef(par.PoleOrder,:);
#       case 2
#         # we need to scale the estimated parameters by delta, for each estimate
#         residue_estimates = polycoef(par.PoleOrder,:)./delta(1:(end - par.MethodOrder)).';
#         residue_estimates = residue_estimates*par.StepRatio.^(-par.MethodOrder/2);
#         # also the error estimate
#         ser = ser./delta(1:(end - par.MethodOrder)).' * par.StepRatio.^(-par.MethodOrder/2);
#       case 3
#         # we need to scale the estimated parameters by delta^(par.PoleOrder-1)
#         residue_estimates = polycoef(par.PoleOrder,:)./delta(1:(end - par.MethodOrder)).'.^2;
#         residue_estimates = residue_estimates*par.StepRatio.^(-2*par.MethodOrder/2);
#         ser = ser./delta(1:(end - par.MethodOrder)).'.^2 * par.StepRatio.^(-2*par.MethodOrder/2);
#     end
#
#     # uncertainty estimate of the limit
#     rAinv = rA\eye(m);
#     cov1 = sum(rAinv.^2,2);
#
#     # 1 spare dof, so we use a student's t with 1 dof
#     errest = 12.7062047361747*sqrt(cov1(1))*ser;
#
#     # drop any estimates that were inf or nan.
#     k = isnan(residue_estimates) | isinf(residue_estimates);
#     errest(k) = [];
#     residue_estimates(k) = [];
#     # delta(k) = [];
#
#     # if nothing remains, then there was a problem.
#     # possibly the wrong order pole, or a bad dz.
#     nres = numel(residue_estimates);
#     if nres < 1
#       error('Either the wrong order was specified for this pole, or dz was a very poor choice')
#     end
#
#     # sort the remaining estimates
#     [residue_estimates, tags] = sort(residue_estimates);
#     errest = errest(tags);
#     # delta = delta(tags);
#
#     # trim off the estimates at each end of the range
#     if nres > 4
#       residue_estimates([1,end]) = [];
#       errest([1,end]) = [];
#       # delta([1,end]) = [];
#     end
#
#     # and take the one that remains with the lowest error estimate
#     [errest,k] = min(errest);
#     res = residue_estimates(k);
#     # delta = delta(k);
#
#     # for higher order poles, we need to divide by factorial(PoleOrder-1)
#     if par.PoleOrder>2
#       res = res/factorial(par.PoleOrder-1);
#       errest = errest/factorial(par.PoleOrder-1);
#     end
#
#     end # mainline end
#
#


def test_docstrings():
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()
