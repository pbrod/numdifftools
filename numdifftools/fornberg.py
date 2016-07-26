
import numpy as np
from scipy.special import factorial
from numdifftools.extrapolation import EPS, dea3
from collections import namedtuple

EPSILON = EPS
EPSILON_3_14 = np.power(EPSILON, 3. / 14)
_INFO = namedtuple('info', ['error_estimate',
                                   'degenerate',
                                   'final_radius',
                                   'function_count',
                                   'iterations', 'failed'])

def fornberg_weights_all(x, x0=0, m=1):
    """
    Return finite difference weights_and_points for derivatives of all orders.

    Parameters
    ----------
    x : vector, length i
        x-coordinates for grid points
    x0 : scalar
        location where approximations are to be accurate
    j : scalar integer
        highest derivative that we want to find weights_and_points for

    Returns
    -------
    C :  array, shape i x j+1
        contains coefficients for the j'th derivative in column j (0 <= j <= j)

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
    n = len(x)
    if m >= n:
        raise ValueError('length(x) must be larger than j')

    c1, c4 = 1, x[0] - x0
    C = np.zeros((n, m + 1))
    C[0, 0] = 1
    for i in range(1, n):
        j = np.arange(0, min(i, m) + 1)
        c2, c5, c4 = 1, c4, x[i] - x0
        for v in range(i):
            c3 = x[i] - x[v]
            c2, c6, c7 = c2 * c3, j * C[v, j - 1], C[v, j]
            C[v, j] = (c4 * c7 - c6) / c3
        C[i, j] = c1 * (c6 - c5 * c7) / c2
        c1 = c2
    return C


def fornberg_weights(x, x0=0, m=1):
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


def _circle(z, r, m):
    theta = np.linspace(0.0, 2.0 * np.pi, num=m, endpoint=False)
    return z + r * np.exp(theta*1j)


def poor_convergence(z, r, f, m, bn):
    """
    Test for poor convergence based on three function evaluations.

    To avoid letting randomness enter the algorithm, three fixed
    points are used, as defined by check_points
    This test evaluates the function at the three points and returns false if
    the relative error is greater than 1e-3.
    """
    check_points = (-0.4 + 0.3j, 0.7 + 0.2j, 0.02 - 0.06j)
    rtest1 = r * check_points[0]
    rtest2 = r * check_points[1]
    rtest3 = r * check_points[2]
    ztest1 = z + rtest1
    ztest2 = z + rtest2
    ztest3 = z + rtest3

    ftest1 = f(ztest1)
    ftest2 = f(ztest2)
    ftest3 = f(ztest3)

    brn = bn / m
    comp1 = np.sum(brn * np.power(rtest1 / r, np.arange(m)))
    comp2 = np.sum(brn * np.power(rtest2 / r, np.arange(m)))
    comp3 = np.sum(brn * np.power(rtest3 / r, np.arange(m)))
    # print ftest1
    # print comp1

    diff1 = comp1 - ftest1
    diff2 = comp2 - ftest2
    diff3 = comp3 - ftest3

    max_abs_error = np.max(np.abs([diff1, diff2, diff3]))
    max_f_value = np.max(np.abs([ftest1, ftest2, ftest3]))
    # max_relative_error = max_abs_error / max_f_value
    # print(max_abs_error, max_f_value)
    return max_abs_error > 1e-3 * max_f_value


def _get_logn(n):
    return np.int_(np.log2(n-1)-1.5849625007211561).clip(min=0)

def _num_taylor_coefficients(n):
    """
    Return 8 if       n <= 6
          16 if   6 < n <= 12
          32 if  12 < n <= 25
          64 if  25 < n <= 51
         128 if  51 < n <= 103
         256 if 103 < n <= 206
    """
    if n>103:
        raise
    correction = np.array([0, 0, 1, 3, 4, 7])[_get_logn(n)]
    log2n = _get_logn(n - correction)
    m = 2 ** (log2n + 3)
    return m


def richardson_parameter(Q, k, c):
    c = np.real((Q[k - 1] - Q[k - 2]) / (Q[k] - Q[k - 1])) - 1.
    # The lower bound 0.07 admits the singularity x.^-0.9
    c = np.maximum(c, 0.07)
    return c

def richardson(Q, k, c=None):
    """Richardson extrapolation with parameter estimation"""
    if c is None:
        c = richardson_parameter(Q, k)
    R = Q[k] - (Q[k] - Q[k - 1]) / c
    return R


def _add_error_to_outliers(der, trim_fact=10):
    try:
        median = np.nanmedian(der, axis=0)
        p75 = np.nanpercentile(der, 75, axis=0)
        p25 = np.nanpercentile(der, 25, axis=0)
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


def _get_best_estimate(der, errors):
    errors += _add_error_to_outliers(der)
    ix = _get_arg_min(errors)
    return der.flat[ix], errors.flat[ix]


def taylor(f, z0, n=1, r=0.6, max_iter=30, min_iter=5, full_output=False):
    """
    Return Taylor coefficients of complex analytic function using FFT

    Parameters
    ----------
    f : callable
        function to differentiate
    z0 : real or complex scalar at which to evaluate the derivatives
    n : scalar integer, default 1
        Number of taylor coefficents to compute. Maximum number is 100.
    r : real scalar, default 0.6
        Initial radius at which to evaluate. For well-behaved functions,
        the computation should be insensitive to the initial radius to within
        about four orders of magnitude.
    max_iter : scalar integer, default 30
        Maximum number of iterations
    min_iter : scalar integer, default 5
        Minimum number of iterations before the solution may be deemed
        degenerate.  A larger number allows the algorithm to correct a bad
        initial radius.
    full_output : bool, optional
        If `full_output` is False, only the coefficents is returned.
        If `full_output` is True, then (coefs, status) is returned

    Returns
    -------
    coefs : ndarray
       array of taylor coefficents
    status: Optional object into which output information is written. Fields are:
        degenerate: True if the algorithm was unable to bound the error
        iterations: Number of iterations executed
        function_count: Number of function calls
        final_radius: Ending radius of the algorithm
        failed: True if the maximum number of iterations was reached
        error_estimate: approximate bounds of the rounding error.

    This module uses the method of Fornberg to compute the Taylor series
    coefficents of a complex analytic function along with error bounds. The
    method uses a Fast Fourier Transform to invert function evaluations around
    a circle into Taylor series coefficients and uses Richardson Extrapolation
    to improve and bound the estimate. Unlike real-valued finite differences,
    the method searches for a desirable radius and so is reasonably insensitive
    to the initial radius-to within a number of orders of magnitude at least.
    For most cases, the default configuration is likely to succeed.

    Restrictions

    The method uses the coefficients themselves to control the truncation error,
    so the error will not be properly bounded for functions like low-order
    polynomials whose Taylor series coefficients are nearly zero. If the error
    cannot be bounded, degenerate flag will be set to true, and an answer will
    still be computed and returned but should be used with caution.

    Example
    -------

    To compute the first five derivatives of 1 / (1 - z) at z = 0:

    References
    ----------
    [1] Fornberg, B. (1981).
        Numerical Differentiation of Analytic Functions.
        ACM Transactions on Mathematical Software (TOMS),
        7(4), 512-526. http://doi.org/10.1145/355972.355979
    """
    if min_iter is None:
        min_iter = max_iter // 2
    direction_changes = 0
    rs = []
    bs = []

    # Initial grow/shring factor for the circle:
    fac = 2
    pdirec = None
    degenerate = False
    m = _num_taylor_coefficients(n)

    # A factor for testing against the targeted geometric progression of
    # fourier coefficients:
    mvec = np.arange(m)
    crat = (np.exp(np.log(1e-4) / (m - 1))) ** mvec

    # Start iterating. The goal of this loops is to select a circle radius that
    # yields a nice geometric progression of the coefficients (which controls
    # the error), and then to accumulate *three* successive approximations as a
    # function of the circle radius r so that we can perform Richardson
    # Extrapolation and zero out error terms, *greatly* improving the quality
    # of the approximation.
    num_changes = 0
    for i in xrange(max_iter):
        # print 'r = %g' % (r)

        bn = np.fft.fft(f(_circle(z0, r, m)))
        bs.append(bn / m * np.power(r, -mvec))
        rs.append(r)
        if direction_changes > 1 or degenerate:

            num_changes += 1
            #if len(rs) >= 3:
            if num_changes >= 3:
                break

        if not degenerate:
            # If not degenerate, check for geometric progression in the fourier
            # transform:
            bnc = bn / crat
            m1 = np.max(np.abs(bnc[:m // 2]))
            m2 = np.max(np.abs(bnc[m // 2:]))
            # If there's an extreme mismatch, then we can consider the
            # geometric progression degenerate, whether one way or the other,
            # and just alternate directions instead of trying to target a
            # specific error bound (not ideal, but not a good reason to fail
            # catastrophically):
            #
            # Note: only consider it degenerate if we've had a chance to steer
            # the radius in the direction at least `min_iter` times:
            degenerate = i > min_iter and (m1 / m2 < 1e-8 or m2 / m1 < 1e-8)

        if degenerate:
            needs_smaller = i % 2 == 0
        else:
            needs_smaller = (m1 != m1 or m2 != m2 or m1 < m2 or
                             poor_convergence(z0, r, f, m, bn))

        if pdirec is not None and needs_smaller != pdirec:
            direction_changes += 1

        if direction_changes > 0:
            # Once we've started changing directions, we've found our range so start
            # taking the square root of the growth factor so that richardson
            # extrapolation is well-behaved:
            fac = np.sqrt(fac)

        if needs_smaller:
            r /= fac
        else:
            r *= fac

        pdirec = needs_smaller

    #     print np.real(bs[0])
    #     print np.real(bs[1])
    #     print np.real(bs[2])

    # Begin Richardson Extrapolation. Presumably we have bs[i]'s around three
    # successive circles and can now extrapolate those coefficients, zeroing out
    # higher order error terms.
#     extrap1 = bs[1] - (bs[1] - bs[0]) / (1.0 - (rs[0] / rs[1])**m)
#     extrap2 = bs[2] - (bs[2] - bs[1]) / (1.0 - (rs[1] / rs[2])**m)
#     extrap3 = extrap2 - (extrap2 - extrap1) / (1.0 - (rs[0] / rs[2])**m)

    nk = len(rs)
    extrap = []
    for k in range(1, nk):
        extrap.append(richardson(bs, k=k, c=(1.0 - (rs[k-1] / rs[k])**m)))
    if len(extrap)>2:
        all_coefs, all_errors = dea3(extrap[:-2], extrap[1:-1], extrap[2:])
        coefs, errors = _get_best_estimate(all_coefs, all_errors)
    else:
        errors = EPSILON / np.power(rs[2], np.arange(m)) * np.maximum(m1, m2)
        k = len(extrap)-1
        coefs = richardson(extrap, k=k, c=(1.0 - (rs[-3] / rs[-1])**m))


    if full_output:
        # compute the truncation error:

        truncation_error = EPSILON_3_14 * np.abs(coefs)
        rounding_error = EPSILON / np.power(rs[2], np.arange(m)) * np.maximum(m1, m2)
        info = _INFO(errors,
                     degenerate, final_radius=r,
                     function_count=i*m, iterations=i, failed=i==max_iter)
        # print(info)

#     print 'answer:'
#     for i in range(len(coefs)):
#         print '%3i: %24.18f + %24.18fj (%g, %g)' % (i,
#                                            np.real(coefs[i]),
#                                            np.imag(coefs[i]),
#                                            truncation_error[i],
#                                            rounding_error[i])

    if full_output:
        return coefs, info
    return coefs


def derivative(f, z0, n=1, r=0.6, max_iter=30, min_iter=5, full_output=False):
    """
    Calculate n-th derivative of complex analytic function using FFT

    Parameters
    ----------
    f : callable
        function to differentiate
    z0 : real or complex scalar at which to evaluate the derivatives
    n : scalar integer, default 1
        Number of derivatives to compute where 0 represents the value of the
        function and n represents the nth derivative. Maximum number is 100.

    r : real scalar, default 0.6
        Initial radius at which to evaluate. For well-behaved functions,
        the computation should be insensitive to the initial radius to within
        about four orders of magnitude.
    max_iter : scalar integer, default 30
        Maximum number of iterations
    min_iter : scalar integer, default 5
        Minimum number of iterations before the solution may be deemed
        degenerate.  A larger number allows the algorithm to correct a bad
        initial radius.
    full_output : bool, optional
        If `full_output` is False, only the derivative is returned.
        If `full_output` is True, then (der, status) is returned `der` is the
        derivative, and `status` is a Results object.

    Returns
    -------
    der : ndarray
       array of derivatives
    status: Optional object into which output information is written. Fields are:
        degenerate: True if the algorithm was unable to bound the error
        iterations: Number of iterations executed
        function_count: Number of function calls
        final_radius: Ending radius of the algorithm
        failed: True if the maximum number of iterations was reached
        error_estimate: approximate bounds of the rounding error.

    This module uses the method of Fornberg to compute the derivatives of a
    complex analytic function along with error bounds. The method uses a
    Fast Fourier Transform to invert function evaluations around a circle into
    Taylor series coefficients, uses Richardson Extrapolation to improve
    and bound the estimate, then multiplies by a factorial to compute the
    derivatives. Unlike real-valued finite differences, the method searches for
    a desirable radius and so is reasonably insensitive to the initial
    radius-to within a number of orders of magnitude at least. For most cases,
    the default configuration is likely to succeed.

    Restrictions

    The method uses the coefficients themselves to control the truncation error,
    so the error will not be properly bounded for functions like low-order
    polynomials whose Taylor series coefficients are nearly zero. If the error
    cannot be bounded, degenerate flag will be set to true, and an answer will
    still be computed and returned but should be used with caution.

    Example
    -------

    To compute the first five derivatives of 1 / (1 - z) at z = 0:

    References
    ----------
    [1] Fornberg, B. (1981).
        Numerical Differentiation of Analytic Functions.
        ACM Transactions on Mathematical Software (TOMS),
        7(4), 512-526. http://doi.org/10.1145/355972.355979
    """
    result = taylor(f, z0, n, r, max_iter, min_iter, full_output)
    # convert taylor series --> actual derivatives.
    m = _num_taylor_coefficients(n)
    fact = factorial(np.arange(m))
    if full_output:
        coefs, info_ = result
        info = _INFO(info_.error_estimate*fact, *info_[1:])
        return coefs * fact, info
    return result * fact


def main():
    def f(z):
        # return np.exp(1.0j * z)
        # return z**6
        # return z * (0.5 + 1./np.expm1(z))
        # return np.exp(z)
        # return np.tan(z)
        # return 1.0j + z + 1.0j * z**2
        # return 1.0 / (1.0 - z)
        return np.sqrt(z)
        return np.arcsinh(z)
        return np.cos(z)
        return np.log1p(z)

    der, info = derivative(f, z0=0.5, r=0.01, n=21, max_iter=30, min_iter=5,
                           full_output=True)
    print(info)
    print('answer:')
    for i in range(len(der)):
        print('%3i: %24.18f + %24.18fj (%g)' % (i,
                                           np.real(der[i]),
                                           np.imag(der[i]),
                                           info.error_estimate[i]))


if __name__ == '__main__':
    main()
