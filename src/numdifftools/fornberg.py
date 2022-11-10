from __future__ import absolute_import, division
from collections import namedtuple
import numpy as np
from scipy.special import factorial
from numdifftools.extrapolation import EPS, dea3
from numdifftools.limits import _Limit

_INFO = namedtuple('info', ['error_estimate',
                            'degenerate',
                            'final_radius',
                            'function_count',
                            'iterations', 'failed'])
CENTRAL_WEIGHTS_AND_POINTS = {
    (1, 3): (np.array([-1., 0, 1]) / 2.0, np.arange(-1, 2)),
    (1, 5): (np.array([1., -8, 0, 8, -1]) / 12.0, np.arange(-2, 3)),
    (1, 7): (np.array([-1., 9, -45, 0, 45, -9, 1]) / 60.0, np.arange(-3, 4)),
    (1, 9): (np.array([3., -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0,
             np.arange(-4, 5)),
    (2, 3): (np.array([1., -2.0, 1]), np.arange(-1, 2)),
    (2, 5): (np.array([-1., 16, -30, 16, -1]) / 12.0, np.arange(-2, 3)),
    (2, 7): (np.array([2., -27, 270, -490, 270, -27, 2]) / 180.0,
             np.arange(-3, 4)),
    (2, 9): (np.array([-9., 128, -1008, 8064, -14350,
                       8064, -1008, 128, -9]) / 5040.0,
             np.arange(-4, 5))}


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def fd_weights_all(x, x0=0, n=1):
    """
    Return finite difference weights for derivatives of all orders up to n.

    Parameters
    ----------
    x : vector, length m
        x-coordinates for grid points
    x0 : scalar
        location where approximations are to be accurate
    n : scalar integer
        highest derivative that we want to find weights for

    Returns
    -------
    weights :  array, shape n+1 x m
        contains coefficients for the j'th derivative in row j (0 <= j <= n)

    Notes
    -----
    The x values can be arbitrarily spaced but must be distinct and len(x) > n.

    The Fornberg algorithm is much more stable numerically than regular
    vandermonde systems for large values of n.

    See also
    --------
    fd_weights

    References
    ----------
    B. Fornberg (1998)
    "Calculation of weights_and_points in finite difference formulas",
    SIAM Review 40, pp. 685-691.

    http://www.scholarpedia.org/article/Finite_difference_method
    """
    m = len(x)
    _assert(n < m, 'len(x) must be larger than n')

    weights = np.zeros((m, n + 1))
    _fd_weights_all(weights, x, x0, n)
    return weights.T


# from numba import jit, float64, int64, int32, int8, void
# @jit(void(float64[:,:], float64[:], float64, int64))
def _fd_weights_all(weights, x, x0, n):
    m = len(x)
    c_1, c_4 = 1, x[0] - x0
    weights[0, 0] = 1
    for i in range(1, m):
        j = np.arange(0, min(i, n) + 1)
        c_2, c_5, c_4 = 1, c_4, x[i] - x0
        for v in range(i):
            c_3 = x[i] - x[v]
            c_2, c_6, c_7 = c_2 * c_3, j * weights[v, j - 1], weights[v, j]
            weights[v, j] = (c_4 * c_7 - c_6) / c_3
        weights[i, j] = c_1 * (c_6 - c_5 * c_7) / c_2
        c_1 = c_2


def fd_weights(x, x0=0, n=1):
    """
    Return finite difference weights for the n'th derivative.

    Parameters
    ----------
    x : vector
        abscissas used for the evaluation for the derivative at x0.
    x0 : scalar
        location where approximations are to be accurate
    n : scalar integer
        order of derivative. Note for n=0 this can be used to evaluate the
        interpolating polynomial itself.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.fornberg as ndf
    >>> x = np.linspace(-1, 1, 5) * 1e-3
    >>> w = ndf.fd_weights(x, x0=0, n=1)
    >>> df = np.dot(w, np.exp(x))
    >>> np.allclose(df, 1)
    True

    See also
    --------
    fd_weights_all
    """
    return fd_weights_all(x, x0, n)[-1]


def fd_derivative(fx, x, n=1, m=2):
    """
    Return the n'th derivative for all points using Finite Difference method.

    Parameters
    ----------
    fx : vector
        function values which are evaluated on x i.e. fx[i] = f(x[i])
    x : vector
        abscissas on which fx is evaluated.  The x values can be arbitrarily
        spaced but must be distinct and len(x) > n.
    n : scalar integer
        order of derivative.
    m : scalar integer
        defines the stencil size. The stencil size is of 2 * mm + 1
        points in the interior, and 2 * mm + 2 points for each of the 2 * mm
        boundary points where mm = n // 2 + m.

    fd_derivative evaluates an approximation for the n'th derivative of the
    vector function f(x) using the Fornberg finite difference method.
    Restrictions: 0 < n < len(x) and 2*mm+2 <= len(x)

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.fornberg as ndf
    >>> x = np.linspace(-1, 1, 25)
    >>> fx = np.exp(x)
    >>> df = ndf.fd_derivative(fx, x, n=1)
    >>> np.allclose(df, fx)
    True

    See also
    --------
    fd_weights
    """
    num_x = len(x)
    _assert(n < num_x, 'len(x) must be larger than n')
    _assert(num_x == len(fx), 'len(x) must be equal len(fx)')

    du = np.zeros_like(fx)

    mm = n // 2 + m
    size = 2 * mm + 2  # stencil size at boundary
    # 2 * mm boundary points
    for i in range(mm):
        du[i] = np.dot(fd_weights(x[:size], x0=x[i], n=n), fx[:size])
        du[-i - 1] = np.dot(fd_weights(x[-size:], x0=x[-i - 1], n=n), fx[-size:])

    # interior points
    for i in range(mm, num_x - mm):
        du[i] = np.dot(fd_weights(x[i - mm:i + mm + 1], x0=x[i], n=n),
                       fx[i - mm:i + mm + 1])

    return du


def _circle(z, r, m):
    theta = np.linspace(0.0, 2.0 * np.pi, num=m, endpoint=False)
    return z + r * np.exp(theta * 1j)


def _poor_convergence(z, r, f, bn, mvec):
    """
    Test for poor convergence based on three function evaluations.

    This test evaluates the function at the three points and returns false if
    the relative error is greater than 1e-3.
    """
    check_points = (-0.4 + 0.3j, 0.7 + 0.2j, 0.02 - 0.06j)
    diffs = []
    ftests = []
    for check_point in check_points:
        rtest = r * check_point
        ztest = z + rtest
        ftest = f(ztest)
        # Evaluate powerseries:
        comp = np.sum(bn * np.power(check_point, mvec))
        ftests.append(ftest)
        diffs.append(comp - ftest)

    max_abs_error = np.max(np.abs(diffs))
    max_f_value = np.max(np.abs(ftests))
    return max_abs_error > 1e-3 * max_f_value


def _get_logn(n):
    if n == 1:
        return 0

    return np.int_(np.log2(n - 1) - 1.5849625007211561).clip(min=0)


def _num_taylor_coefficients(n):
    """
    Return number of taylor coefficients

    Parameters
    ----------
    n : scalar integer
        Wanted number of taylor coefficients

    Returns
    -------
    m : scalar integer
        Number of taylor coefficients calculated
           8 if       n <= 6
          16 if   6 < n <= 12
          32 if  12 < n <= 25
          64 if  25 < n <= 51
         128 if  51 < n <= 103
         256 if 103 < n <= 192
    """
    _assert(n < 193, 'Number of derivatives too large.  Must be less than 193')
    correction = np.array([0, 0, 1, 3, 4, 7])[_get_logn(n)]
    log2n = _get_logn(n - correction)
    m = 2 ** (log2n + 3)
    return m


def richardson_parameter(vals, k):
    c = np.real((vals[k - 1] - vals[k - 2]) / (vals[k] - vals[k - 1])) - 1.
    # The lower bound 0.07 admits the singularity x.^-0.9
    c = np.maximum(c, 0.07)
    return -c


def richardson(vals, k, c=None):
    """Richardson extrapolation with parameter estimation"""
    if c is None:
        c = richardson_parameter(vals, k)
    return vals[k] - (vals[k] - vals[k - 1]) / c


def _extrapolate(bs, rs, m):
    # Begin Richardson Extrapolation. Presumably we have bs[i]'s around three
    # successive circles and can now extrapolate those coefficients, zeroing
    # out higher order error terms.

    nk = len(rs)
    extrap0 = []
    extrap = []
    for k in range(1, nk):
        extrap0.append(richardson(bs, k=k, c=1.0 - (rs[k - 1] / rs[k]) ** m))

    for k in range(1, nk - 1):
        extrap.append(richardson(extrap0, k=k,
                                 c=1.0 - (rs[k - 1] / rs[k + 1]) ** m))
    return extrap


def _get_best_taylor_coefficients(bs, rs, m, max_m1m2):
    extrap = _extrapolate(bs, rs, m)
    mvec = np.arange(m)
    if len(extrap) > 2:
        all_coefs, all_errors = dea3(extrap[:-2], extrap[1:-1], extrap[2:])
        steps = np.atleast_1d(rs[4:])[:, None] * mvec
        # pylint: disable=protected-access
        coefs, info = _Limit._get_best_estimate(all_coefs, all_errors, steps, (m,))
        errors = info.error_estimate
    else:
        errors = EPS / np.power(rs[2], mvec) * max_m1m2()
        coefs = extrap[-1]
    return coefs, errors


def _check_fft(m1, m2, check_degenerate=True):
    # If not degenerate, check for geometric progression in the FFT by comparing m1 and m2:

    # If there's an extreme mismatch, then we can consider the
    # geometric progression degenerate, whether one way or the other,
    # and just alternate directions instead of trying to target a
    # specific error bound (not ideal, but not a good reason to fail
    # catastrophically):
    #
    # Note: only consider it degenerate if we've had a chance to steer
    # the radius in the direction at least `min_iter` times:
    degenerate = check_degenerate and (m1 < m2 * 1e-8 or m2 < m1 * 1e-8)
    needs_smaller = np.isnan(m1) or np.isnan(m2) or m1 < m2
    return degenerate, needs_smaller


class Taylor(object):

    """
    Return Taylor coefficients of complex analytic function using FFT

    Parameters
    ----------
    fun : callable
        function to differentiate
    z0 : real or complex scalar at which to evaluate the derivatives
    n : scalar integer, default 1
        Number of taylor coefficents to compute. Maximum number is 100.
    r : real scalar, default 0.0059
        Initial radius at which to evaluate. For well-behaved functions,
        the computation should be insensitive to the initial radius to within
        about four orders of magnitude.
    num_extrap : scalar integer, default 3
        number of extrapolation steps used in the calculation
    step_ratio : real scalar, default 1.6
        Initial grow/shrinking factor for finding the best radius.
    max_iter : scalar integer, default 30
        Maximum number of iterations
    min_iter : scalar integer, default max_iter // 2
        Minimum number of iterations before the solution may be deemed
        degenerate.  A larger number allows the algorithm to correct a bad
        initial radius.
    full_output : bool, optional
        If `full_output` is False, only the coefficents is returned (default).
        If `full_output` is True, then (coefs, status) is returned

    Returns
    -------
    coefs : ndarray
       array of taylor coefficents
    status: Optional object into which output information is written:
        degenerate: True if the algorithm was unable to bound the error
        iterations: Number of iterations executed
        function_count: Number of function calls
        final_radius: Ending radius of the algorithm
        failed: True if the maximum number of iterations was reached
        error_estimate: approximate bounds of the rounding error.

    Notes
    -----
    This module uses the method of Fornberg to compute the Taylor series
    coefficients of a complex analytic function along with error bounds. The
    method uses a Fast Fourier Transform to invert function evaluations around
    a circle into Taylor series coefficients and uses Richardson Extrapolation
    to improve and bound the estimate. Unlike real-valued finite differences,
    the method searches for a desirable radius and so is reasonably
    insensitive to the initial radius-to within a number of orders of
    magnitude at least. For most cases, the default configuration is likely to
    succeed.

    Restrictions:
    The method uses the coefficients themselves to control the truncation
    error, so the error will not be properly bounded for functions like
    low-order polynomials whose Taylor series coefficients are nearly zero.
    If the error cannot be bounded, degenerate flag will be set to true, and
    an answer will still be computed and returned but should be used with
    caution.

    Examples
    --------
    Compute the first 6 taylor coefficients 1 / (1 - z) expanded round  z0 = 0:

    >>> import numdifftools.fornberg as ndf
    >>> import numpy as np
    >>> c, info = ndf.Taylor(lambda x: 1./(1-x), n=6, full_output=True)(z0=0)
    >>> np.allclose(c, np.ones(8))
    True
    >>> np.all(info.error_estimate < 1e-9)
    True
    >>> (info.function_count, info.iterations, info.failed) == (136, 17, False)
    True

    References
    ----------
    [1] Fornberg, B. (1981).
        Numerical Differentiation of Analytic Functions.
        ACM Transactions on Mathematical Software (TOMS),
        7(4), 512-526. http://doi.org/10.1145/355972.355979
    """

    def __init__(self, fun, n=1, r=0.0059, num_extrap=3, step_ratio=1.6, **kwds):
        self.fun = fun
        self.max_iter = kwds.pop('max_iter', 30)
        self.min_iter = kwds.pop('min_iter', self.max_iter // 2)
        self.full_output = kwds.pop('full_output', False)
        self.n = n
        self.r = r
        self.num_extrap = num_extrap
        self.step_ratio = step_ratio

    def _initialize(self):
        m = _num_taylor_coefficients(self.n)
        self._step_ratio = self.step_ratio
        self._direction_changes = 0
        self._previous_direction = None
        self._degenerate = self._failed = False
        self._m = m
        self._mvec = np.arange(m)
        # A factor for testing against the targeted geometric progression of
        # FFT coefficients:
        self._crat = m * (np.exp(np.log(1e-4) / (m - 1))) ** self._mvec
        self._num_changes = 0
        return m, self._mvec

    def _get_max_m1m2(self, bn, m):
        m1, m2 = self._get_m1_m2(bn, m)
        return np.maximum(m1, m2)

    def _get_m1_m2(self, bn, m):
        # If not degenerate, check for geometric progression in the FFT:
        bnc = bn / self._crat
        m1 = np.max(np.abs(bnc[:m // 2]))
        m2 = np.max(np.abs(bnc[m // 2:]))
        return m1, m2

    def _check_convergence(self, i, z0, r, m, bn):
        if self._direction_changes > 1 or self._degenerate:
            self._num_changes += 1
            if self._num_changes >= 1 + self.num_extrap:
                return True, r

        if not self._degenerate:
            m1, m2 = self._get_m1_m2(bn, m)
            # Note: only consider it degenerate if we've had a chance to steer
            # the radius in the direction at least `min_iter` times:
            check_degenerate = i > self.min_iter
            self._degenerate, needs_smaller = _check_fft(m1, m2, check_degenerate)
            needs_smaller = needs_smaller or _poor_convergence(z0, r, self.fun, bn, self._mvec)
        if self._degenerate:
            needs_smaller = i % 2 == 0
        if self._previous_direction is not None and needs_smaller != self._previous_direction:
            self._direction_changes += 1
        if self._direction_changes > 0:
            # Once we've started changing directions, we've found our range so
            # start taking the square root of the growth factor so that
            # richardson extrapolation is well-behaved:
            self._step_ratio = np.sqrt(self._step_ratio)
        if needs_smaller:
            r /= self._step_ratio
        else:
            r *= self._step_ratio
        self._previous_direction = needs_smaller
        return False, r

    def __call__(self, z0=0):
        m, mvec = self._initialize()

        # Start iterating. The goal of this loops is to select a circle radius that
        # yields a nice geometric progression of the coefficients (which controls
        # the error), and then to accumulate *three* successive approximations as a
        # function of the circle radius r so that we can perform Richardson
        # Extrapolation and zero out error terms, *greatly* improving the quality
        # of the approximation.

        rs = []
        bs = []
        i = 0
        r = self.r
        fun = self.fun
        for i in range(self.max_iter):
            # print('r = %g' % (r))

            bn = np.fft.fft(fun(_circle(z0, r, m))) / m
            bs.append(bn * np.power(r, -mvec))
            rs.append(r)

            converged, r = self._check_convergence(i, z0, r, m, bn)
            if converged:
                break

        coefs, errors = _get_best_taylor_coefficients(bs, rs, m, lambda: self._get_max_m1m2(bn, m))
        if self.full_output:
            failed = not converged
            info = _INFO(errors, self._degenerate, final_radius=r,
                         function_count=i * m, iterations=i, failed=failed)
            return coefs, info
        return coefs


def taylor(fun, z0=0, n=1, r=0.0059, num_extrap=3, step_ratio=1.6, **kwds):
    """
    Return Taylor coefficients of complex analytic function using FFT

    Parameters
    ----------
    fun : callable
        function to differentiate
    z0 : real or complex scalar at which to evaluate the derivatives
    n : scalar integer, default 1
        Number of taylor coefficents to compute. Maximum number is 100.
    r : real scalar, default 0.0059
        Initial radius at which to evaluate. For well-behaved functions,
        the computation should be insensitive to the initial radius to within
        about four orders of magnitude.
    num_extrap : scalar integer, default 3
        number of extrapolation steps used in the calculation
    step_ratio : real scalar, default 1.6
        Initial grow/shrinking factor for finding the best radius.
    max_iter : scalar integer, default 30
        Maximum number of iterations
    min_iter : scalar integer, default max_iter // 2
        Minimum number of iterations before the solution may be deemed
        degenerate.  A larger number allows the algorithm to correct a bad
        initial radius.
    full_output : bool, optional
        If `full_output` is False, only the coefficents is returned (default).
        If `full_output` is True, then (coefs, status) is returned

    Returns
    -------
    coefs : ndarray
       array of taylor coefficents
    status: Optional object into which output information is written:
        degenerate: True if the algorithm was unable to bound the error
        iterations: Number of iterations executed
        function_count: Number of function calls
        final_radius: Ending radius of the algorithm
        failed: True if the maximum number of iterations was reached
        error_estimate: approximate bounds of the rounding error.

    Notes
    -----
    This module uses the method of Fornberg to compute the Taylor series
    coefficents of a complex analytic function along with error bounds. The
    method uses a Fast Fourier Transform to invert function evaluations around
    a circle into Taylor series coefficients and uses Richardson Extrapolation
    to improve and bound the estimate. Unlike real-valued finite differences,
    the method searches for a desirable radius and so is reasonably
    insensitive to the initial radius-to within a number of orders of
    magnitude at least. For most cases, the default configuration is likely to
    succeed.

    Restrictions:
    The method uses the coefficients themselves to control the truncation
    error, so the error will not be properly bounded for functions like
    low-order polynomials whose Taylor series coefficients are nearly zero.
    If the error cannot be bounded, degenerate flag will be set to true, and
    an answer will still be computed and returned but should be used with
    caution.

    Examples
    --------
    Compute the first 6 taylor coefficients 1 / (1 - z) expanded round  z0 = 0:

    >>> import numdifftools.fornberg as ndf
    >>> import numpy as np
    >>> c, info = ndf.taylor(lambda x: 1./(1-x), z0=0, n=6, full_output=True)
    >>> np.allclose(c, np.ones(8))
    True
    >>> np.all(info.error_estimate < 1e-9)
    True
    >>> (info.function_count, info.iterations, info.failed) == (136, 17, False)
    True


    References
    ----------
    [1] Fornberg, B. (1981).
        Numerical Differentiation of Analytic Functions.
        ACM Transactions on Mathematical Software (TOMS),
        7(4), 512-526. http://doi.org/10.1145/355972.355979
    """
    return Taylor(fun, n=n, r=r, num_extrap=num_extrap, step_ratio=step_ratio, **kwds)(z0)


def derivative(fun, z0, n=1, **kwds):
    """
    Calculate n-th derivative of complex analytic function using FFT

    Parameters
    ----------
    fun : callable
        function to differentiate
    z0 : real or complex scalar at which to evaluate the derivatives
    n : scalar integer, default 1
        Number of derivatives to compute where 0 represents the value of the
        function and n represents the nth derivative. Maximum number is 100.

    r : real scalar, default 0.0061
        Initial radius at which to evaluate. For well-behaved functions,
        the computation should be insensitive to the initial radius to within
        about four orders of magnitude.
    max_iter : scalar integer, default 30
        Maximum number of iterations
    min_iter : scalar integer, default max_iter // 2
        Minimum number of iterations before the solution may be deemed
        degenerate.  A larger number allows the algorithm to correct a bad
        initial radius.
    step_ratio : real scalar, default 1.6
        Initial grow/shrinking factor for finding the best radius.
    num_extrap : scalar integer, default 3
        number of extrapolation steps used in the calculation
    full_output : bool, optional
        If `full_output` is False, only the derivative is returned (default).
        If `full_output` is True, then (der, status) is returned `der` is the
        derivative, and `status` is a Results object.

    Returns
    -------
    der : ndarray
        array of derivatives
    status: Optional object into which output information is written. Fields:
        degenerate: True if the algorithm was unable to bound the error
        iterations: Number of iterations executed
        function_count: Number of function calls
        final_radius: Ending radius of the algorithm
        failed: True if the maximum number of iterations was reached
        error_estimate: approximate bounds of the rounding error.

    Notes
    -----
    This module uses the method of Fornberg to compute the derivatives of a
    complex analytic function along with error bounds. The method uses a
    Fast Fourier Transform to invert function evaluations around a circle into
    Taylor series coefficients, uses Richardson Extrapolation to improve
    and bound the estimate, then multiplies by a factorial to compute the
    derivatives. Unlike real-valued finite differences, the method searches for
    a desirable radius and so is reasonably insensitive to the initial
    radius-to within a number of orders of magnitude at least. For most cases,
    the default configuration is likely to succeed.

    Restrictions:
    The method uses the coefficients themselves to control the truncation
    error, so the error will not be properly bounded for functions like
    low-order polynomials whose Taylor series coefficients are nearly zero.
    If the error cannot be bounded, degenerate flag will be set to true, and
    an answer will still be computed and returned but should be used with
    caution.

    Examples
    --------
    To compute the first five derivatives of 1 / (1 - z) at z = 0:
    Compute the first 6 taylor derivatives of 1 / (1 - z) at z0 = 0:

    >>> import numdifftools.fornberg as ndf
    >>> import numpy as np
    >>> def fun(x):
    ...    return 1./(1-x)
    >>> c, info = ndf.derivative(fun, z0=0, n=6, full_output=True)
    >>> np.allclose(c, [1, 1, 2, 6, 24, 120, 720, 5040])
    True
    >>> np.all(info.error_estimate < 1e-9*c.real)
    True
    >>> (info.function_count, info.iterations, info.failed) == (136, 17, False)
    True


    References
    ----------
    [1] Fornberg, B. (1981).
        Numerical Differentiation of Analytic Functions.
        ACM Transactions on Mathematical Software (TOMS),
        7(4), 512-526. http://doi.org/10.1145/355972.355979
    """
    result = taylor(fun, z0, n=n, **kwds)
    # convert taylor series --> actual derivatives.
    m = _num_taylor_coefficients(n)
    fact = factorial(np.arange(m))
    if kwds.get('full_output'):
        coefs, info_ = result
        info = _INFO(info_.error_estimate * fact, *info_[1:])
        return coefs * fact, info
    return result * fact


if __name__ == '__main__':
    from numdifftools.testing import test_docstrings
    test_docstrings()
