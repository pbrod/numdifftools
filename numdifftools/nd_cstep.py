"""numerical differentiation function, gradient, Jacobian, and Hessian

Author : pbrod, josef-pkt
License : BSD
Notes
-----
These are simple forward differentiation, so that we have them available
without dependencies.
* Jacobian should be faster than numdifftools.core because it doesn't use loop
  over observations.
* numerical precision will vary and depend on the choice of stepsizes
"""

# TODO:
# * some cleanup
# * check numerical accuracy (and bugs) with numdifftools and analytical
#   derivatives
#   - linear least squares case: (hess - 2*X'X) is 1e-8 or so
#   - gradient and Hessian agree with numdifftools when evaluated away from
#     minimum
#   - forward gradient, Jacobian evaluated at minimum is inaccurate, centered
#     (+/- epsilon) is ok
# * dot product of Jacobian is different from Hessian, either wrong example or
#   a bug (unlikely), or a real difference
#
#
# What are the conditions that Jacobian dotproduct and Hessian are the same?
#
# See also:
#
# BHHH: Greene p481 17.4.6,  MLE Jacobian = d loglike / d beta , where loglike
# is vector for each observation
#    see also example 17.4 when J'J is very different from Hessian
#    also does it hold only at the minimum, what's relationship to covariance
#    of Jacobian matrix
# http://projects.scipy.org/scipy/ticket/1157
# http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
#    objective: sum((y-f(beta,x)**2),   Jacobian = d f/d beta
#    and not d objective/d beta as in MLE Greene similar:
# http://crsouza.blogspot.com/2009/11/neural-network-learning-by-levenberg_18.html#hessian
#
# in example: if J = d x*beta / d beta then J'J == X'X
#    similar to
#    http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
from __future__ import print_function
import numpy as np
from numdifftools import dea3

# NOTE: we only do double precision internally so far
EPS = np.MachAr().eps


def _make_exact(h):
        '''Make sure h is an exact representable number
        This is important when calculating numerical derivatives and is
        accomplished by adding 1 and then subtracting 1..
        '''
        return (h + 1.0) - 1.0


def _get_epsilon(x, s, epsilon, n):
    if epsilon is None:
        h = (10 * EPS) ** (1. / s) * np.maximum(np.log1p(np.abs(x)), 0.1)
    else:
        if np.isscalar(epsilon):
            h = np.empty(n)
            h.fill(epsilon)
        else:  # pragma : no cover
            h = np.asarray(epsilon)
            if h.shape != x.shape:
                raise ValueError("If h is not a scalar it must have the same"
                                 " shape as x.")
    return _make_exact(h)


_cmn_doc = """
    Calculate %(derivative)s with finite difference approximation

    Parameters
    ----------
    f : function
       function of one array f(x, `*args`, `**kwargs`)
    epsilon : float, array-like or StepsGenerator object, optional
       Stepsize used, if None, then stepsize is automatically chosen
       according to 10*EPS**(1/scale)*max(log(1+|x|), 0.1) where scale is
       depending on method.
       A StepsGenerator can be used to extrapolate the results. However,
       the generator must generate minimum 3 steps in order to extrapolate
       the values.
    method : string, optional
        defines method used in the approximation
        'complex': complex-step derivative (scale=%(scale_complex)s)
        'central': central difference derivative (scale=%(scale_central)s)
        'forward': forward difference derivative (scale=%(scale_forward)s)
        %(extra_method)s
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
    The complex-step derivative has truncation error O(epsilon**2), so
    truncation error can be eliminated by choosing epsilon to be very small.
    The complex-step derivative avoids the problem of round-off error with
    small epsilon because there is no subtraction. However, the function
    needs to be analytic. This method does not work if f(x) involves non-
    analytic functions such as e.g.: abs, max, min
    %(extra_note)s
    References
    ----------
    Ridout, M.S. (2009) Statistical applications of the complex-step method
        of numerical differentiation. The American Statistician, 63, 66-74
    %(example)s
    %(see_also)s
    """


class StepsGenerator(object):
    '''
    Generates a sequence of steps

    where
        steps = epsilon if num_steps < 3
    otherwise
        steps = epsilon * step_ratio ** (np.arange(num_steps) + offset)

    Parameters
    ----------
    epsilon : float, array-like, optional
       Defines the base step, if None, then epsilon is set to
           (10*EPS)**(1/scale)*max(log(1+|x|), 0.1)
       where x and scale are supplied at runtime through the __call__ method.
    num_steps : scalar integer, optional
        defines number of steps generated.
        Note: num_steps < 3 will only return one step.
    step_ratio : real scalar, optional
        Ratio between sequential steps generated.
        Note: Ratio > 1
    offset : real scalar, optional
        offset
    '''
    def __init__(self, epsilon=None, num_steps=7, step_ratio=5, offset=-2):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_ratio = step_ratio
        self.offset = offset

    def __call__(self, x, scale=1):
        xi = np.asarray(x)
        delta = _get_epsilon(xi, scale, self.epsilon, n=xi.shape)

        num_steps = int(self.num_steps)
        if num_steps > 2:
            step_ratio = float(self.step_ratio)
            offset = self.offset
            for i in range(num_steps):
                h = _make_exact(delta * step_ratio**(i + offset))
                if (np.abs(h) > 0).all():
                    yield h
        else:
            yield delta


class _Derivative(object):
    def __init__(self, f, epsilon=None, method='complex', full_output=False):
        self.f = f
        self.epsilon = self._make_callable(epsilon)
        self.method = method
        self.full_output = full_output

    def _make_callable(self, epsilon):
        def _epsilon(xi, scale):
            yield _get_epsilon(xi, scale, epsilon, n=xi.shape)
        if hasattr(epsilon, '__call__'):
            return epsilon
        else:
            return _epsilon

    def _get_method(self):
        return getattr(self, '_' + self.method)

    def _get_scale(self):
        return dict(complex=1, central=3).get(self.method, 2)

    def _stepsizes(self, xi):
        scale = self._get_scale()
        for h in self.epsilon(xi, scale):
            yield h

    def __call__(self, x, *args, **kwds):
        xi = np.asarray(x)
        derivate = self._get_method()
        results = [derivate(self.f, xi, h, *args, **kwds)
                   for h in self._stepsizes(xi)]
        return self._extrapolate(results)

    def _extrapolate(self, sequence):
        dont_extrapolate = len(sequence) < 3
        if dont_extrapolate:
            if self.full_output:
                err = np.empty_like(sequence[0]).fill(np.NaN)
                return 0.5*(sequence[0] + sequence[-1]), {'error': err}
            else:
                return 0.5*(sequence[0] + sequence[-1])
        original_shape = sequence[0].shape
        res = np.vstack(r.ravel() for r in sequence)
        der, error_estimate = dea3(res[0:-2], res[1:-1], res[2:])
        i = np.nanargmin(error_estimate, axis=0)
        ix = np.ravel_multi_index((i, np.arange(der.shape[1])), der.shape)
        if self.full_output:
            err = error_estimate.flat[ix].reshape(original_shape)
            return der.flat[ix].reshape(original_shape), {'error': err}
        else:
            return der.flat[ix].reshape(original_shape)


class Derivative(_Derivative):
    __doc__ = _cmn_doc % dict(derivative='first order derivative',
                              scale_complex='1',
                              scale_forward='2',
                              scale_backward='2',
                              scale_central='3',
                              extra_method="'backward': backward difference "
                              "derivative (scale=2)",
                              extra_note='', returns="""
    Returns
    -------
    der : ndarray
       array of derivatives
    """, example="""
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_cstep as ndc

    # 1'st derivative of exp(x), at x == 1

    >>> fd = ndc.Derivative(np.exp)       # 1'st derivative
    >>> fd(1)
    array([ 2.71828183])

    >>> d2 = fd([1, 2])
    >>> d2
    array([ 2.71828183,  7.3890561 ])""", see_also="""
    See also
    --------
    Gradient,
    Hessian
    """)

    def _central(self, f, x, h2, *args, **kwds):
        h = h2 / 2.0
        return (f(x + h, *args, **kwds) - f(x - h, *args, **kwds)) / h2

    def _forward(self, f, x, h, *args, **kwds):
        return (f(x + h, *args, **kwds) - f(x, *args, **kwds)) / h

    def _backward(self, f, x, h, *args, **kwds):
        return (f(x, *args, **kwds) - f(x - h, *args, **kwds)) / h

    def _complex(self, f, x, h, *args, **kwds):
        return f(x + 1j * h, *args, **kwds).imag / h


class Gradient(_Derivative):
    __doc__ = _cmn_doc % dict(derivative='Gradient',
                              scale_complex='1',
                              scale_forward='2',
                              scale_backward='2',
                              scale_central='3',
                              extra_method="'backward' : backward difference "
                              "derivative (scale=2)", returns="""
    Returns
    -------
    grad : array
        gradient
    """, extra_note="""
    If f returns a 1d array, it returns a Jacobian. If a 2d array is returned
    by f (e.g., with a value for each observation), it returns a 3d array
    with the Jacobian of each observation with shape xk x nobs x xk. I.e.,
    the Jacobian of the first observation would be [:, 0, :]
    """, example="""
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_cstep as ndc
    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = ndc.Gradient(fun)
    >>> dfun([1,2,3])
    array([ 2.,  4.,  6.])

    # At [x,y] = [1,1], compute the numerical gradient
    # of the function sin(x-y) + y*exp(x)

    >>> sin = np.sin; exp = np.exp
    >>> z = lambda xy: sin(xy[0]-xy[1]) + xy[1]*exp(xy[0])
    >>> dz = ndc.Gradient(z)
    >>> grad2 = dz([1, 1])
    >>> grad2
    array([ 3.71828183,  1.71828183])

    # At the global minimizer (1,1) of the Rosenbrock function,
    # compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> rd = ndc.Gradient(rosen)
    >>> grad3 = rd([1,1])
    >>> grad3""", see_also="""
    See also
    --------
    Derivative, Hessian, Jacobian
    """)

    def _central(self, f, x, h2, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * h2 / 2.0
        partials = [(f(x + h, *args, **kwds) -
                     f(x - h, *args, **kwds)) / (h2[i])
                    for i, h in enumerate(increments)]
        return np.array(partials).T

    def _backward(self, f, x, epsilon, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * epsilon
        f0 = f(x, *args, **kwds)
        partials = [(f0 - f(x - h, *args, **kwds)) / epsilon[i]
                    for i, h in enumerate(increments)]
        return np.array(partials).T

    def _forward(self, f, x, epsilon, *args, **kwds):
        n = len(x)
        increments = np.identity(n) * epsilon
        f0 = f(x, *args, **kwds)
        partials = [(f(x + h, *args, **kwds) - f0) / epsilon[i]
                    for i, h in enumerate(increments)]
        return np.array(partials).T

    def _complex(self, f, x, epsilon, *args, **kwds):
        # From Guilherme P. de Freitas, numpy mailing list
        # http://mail.scipy.org/pipermail/numpy-discussion/2010-May/050250.html
        n = len(x)
        increments = np.identity(n) * 1j * epsilon
        partials = [f(x + ih, *args, **kwds).imag / epsilon[i]
                    for i, ih in enumerate(increments)]
        return np.array(partials).T


class Jacobian(Gradient):
    __doc__ = _cmn_doc % dict(derivative='Jacobian',
                              scale_complex='1',
                              scale_forward='2',
                              scale_backward='2',
                              scale_central='3',
                              extra_method="'backward' : backward difference "
                              "derivative (scale=2)", returns="""
    Returns
    -------
    jacob : array
        Jacobian
    """, extra_note="""
    If f returns a 1d array, it returns a Jacobian. If a 2d array is returned
    by f (e.g., with a value for each observation), it returns a 3d array
    with the Jacobian of each observation with shape xk x nobs x xk. I.e.,
    the Jacobian of the first observation would be [:, 0, :]
    """, example='''
     Examples
    --------
    >>> import numdifftools.nd_cstep as ndc

    #(nonlinear least squares)

    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2

    >>> Jfun = ndc.Jacobian(fun)
    >>> Jfun([1,2,0.75]).T # should be numerically zero
    >>> array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    ...       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    ...       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    >>> fun2 = lambda x : x[0]*x[1]*x[2] + np.exp(x[0])*x[1]
    >>> Jfun3 = ndc.Jacobian(fun2)
    >>> Jfun3([3.,5.,7.])
    array([ 135.42768462,   41.08553692,   15.        ])
    ''', see_also="""
    See also
    --------
    Derivative, Hessian, Gradient
    """)


class Hessian(_Derivative):
    __doc__ = _cmn_doc % dict(derivative='Hessian',
                              scale_complex='3',
                              scale_forward='3',
                              scale_central='4',
                              extra_method="'central2' : central difference "
                              "derivative (scale=3)", returns="""
    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian
    """, extra_note="""Computes the Hessian according to method as:
    'forward', Eq. (7):
        1/(d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j])))
    'central2', Eq. (8):
        1/(2*d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j])) -
                         (f(x + d[k]*e[k]) - f(x)) +
                         (f(x - d[j]*e[j] - d[k]*e[k]) - f(x + d[j]*e[j])) -
                         (f(x - d[k]*e[k]) - f(x)))
    'central', Eq. (9):
        1/(4*d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) -
                          f(x + d[j]*e[j] - d[k]*e[k])) -
                         (f(x - d[j]*e[j] + d[k]*e[k]) -
                          f(x - d[j]*e[j] - d[k]*e[k]))
    'complex', Eq. (10):
        1/(2*d_j*d_k) * imag(f(x + i*d[j]*e[j] + d[k]*e[k]) -
                            f(x + i*d[j]*e[j] - d[k]*e[k]))
    where e[j] is a vector with element j == 1 and the rest are zero and
    d[i] is epsilon[i].
    """, example="""
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_cstep as ndc

    # Rosenbrock function, minimized at [1,1]

    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hfun = ndc.Hessian(rosen)
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

    def _get_scale(self):
        return dict(central=5, central2=4, complex=6).get(self.method, 3)

    def _complex(self, f, x, h, *args, **kwargs):
        '''Calculate Hessian with complex-step derivative approximation
        The stepsize is the same for the complex and the finite difference part
        '''
        # TODO: might want to consider lowering the step for pure derivatives
        n = len(x)
        # h = _get_epsilon(x, 3, epsilon, n)
        ee = np.diag(h)
        hess = 2. * np.outer(h, h)

        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + 1j * ee[i, :] + ee[j, :], *args, **kwargs)
                              - f(*((x + 1j * ee[i, :] - ee[j, :],) + args),
                                  **kwargs)).imag / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    def _central(self, f, x, h, *args, **kwargs):
        '''Eq 9.'''
        n = len(x)
        # h = _get_epsilon(x, 4, epsilon, n)
        ee = np.diag(h)
        hess = np.outer(h, h)

        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :], *args, **kwargs)
                              - f(x + ee[i, :] - ee[j, :], *args, **kwargs)
                              - f(x - ee[i, :] + ee[j, :], *args, **kwargs)
                              + f(x - ee[i, :] - ee[j, :], *args, **kwargs)
                              ) / (4.*hess[j, i])
                hess[j, i] = hess[i, j]
        return hess

    def _central2(self, f, x, h, *args, **kwargs):
        '''Eq. 8'''
        n = len(x)
        # NOTE: ridout suggesting using eps**(1/4)*theta
        # h = _get_epsilon(x, 3, epsilon, n)
        ee = np.diag(h)
        f0 = f(x, *args, **kwargs)
        # Compute forward step
        g = np.zeros(n)
        gg = np.zeros(n)
        for i in range(n):
            g[i] = f(x + ee[i, :], *args, **kwargs)
            gg[i] = f(x - ee[i, :], *args, **kwargs)

        hess = np.outer(h, h)  # this is now epsilon**2
        # Compute "double" forward step
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :], *args, **kwargs) -
                              g[i] - g[j] + f0 +
                              f(x - ee[i, :] - ee[j, :], *args, **kwargs) -
                              gg[i] - gg[j] + f0) / (2 * hess[j, i])
                hess[j, i] = hess[i, j]

        return hess

    def _forward(self, f, x, h, *args, **kwargs):
        '''Eq. 7'''
        n = len(x)
        # h = _get_epsilon(x, 3, epsilon, n)
        ee = np.diag(h)

        f0 = f(x, *args, **kwargs)
        # Compute forward step
        g = np.zeros(n)
        for i in range(n):
            g[i] = f(x + ee[i, :], *args, **kwargs)

        hess = np.outer(h, h)  # this is now epsilon**2
        # Compute "double" forward step
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + ee[i, :] + ee[j, :], *args, **kwargs) -
                              g[i] - g[j] + f0) / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess


def main():
    import statsmodels.api as sm

    data = sm.datasets.spector.load()
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod = sm.Probit(data.endog, data.exog)
    _res = mod.fit(method="newton")
    _test_params = [1, 0.25, 1.4, -7]
    _llf = mod.loglike
    _score = mod.score
    _hess = mod.hessian

    def fun(beta, x):
        return np.dot(x, beta).sum(0)

    def fun1(beta, y, x):
        # print(beta.shape, x.shape)
        xb = np.dot(x, beta)
        return (y - xb) ** 2  # (xb-xb.mean(0))**2

    def fun2(beta, y, x):
        # print(beta.shape, x.shape)
        return fun1(beta, y, x).sum(0)

    nobs = 200
    x = np.random.randn(nobs, 3)

    # xk = np.array([1, 2, 3])
    xk = np.array([1., 1., 1.])
    # xk = np.zeros(3)
    beta = xk
    y = np.dot(x, beta) + 0.1 * np.random.randn(nobs)
    xk = np.dot(np.linalg.pinv(x), y)

    epsilon = 1e-6
    args = (y, x)
    from scipy import optimize
    _xfmin = optimize.fmin(fun2, (0, 0, 0), args)
    # print(approx_fprime((1, 2, 3), fun, epsilon, x))
    jac = Gradient(fun1, epsilon, method='forward')(xk, *args)
    jacmin = Gradient(fun1, -epsilon, method='forward')(xk, *args)
    # print(jac)
    print(jac.sum(0))
    print('\nnp.dot(jac.T, jac)')
    print(np.dot(jac.T, jac))
    print('\n2*np.dot(x.T, x)')
    print(2 * np.dot(x.T, x))
    jac2 = (jac + jacmin) / 2.
    print(np.dot(jac2.T, jac2))

    # he = approx_hess(xk,fun2,epsilon,*args)
    print(Hessian(fun2, 1e-3, method='central2')(xk, *args))
    he = Hessian(fun2, method='central2')(xk, *args)
    print('hessfd')
    print(he)
    print('epsilon =', None)
    print(he[0] - 2 * np.dot(x.T, x))

    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        print('eps =', eps)
        print(Hessian(fun2, eps, method='central2')(xk, *args) -
              2 * np.dot(x.T, x))

    hcs2 = Hessian(fun2, method='complex')(xk, *args)
    print('hcs2')
    print(hcs2 - 2 * np.dot(x.T, x))

    hfd3 = Hessian(fun2, method='central')(xk, *args)
    print('hfd3')
    print(hfd3 - 2 * np.dot(x.T, x))

    hfi = []
    epsi = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]) * 10.
    for eps in epsi:
        h = eps * np.maximum(np.log1p(np.abs(xk)), 0.1)
        hfi.append(Hessian(fun2, h, method='complex')(xk, *args))
        print('hfi, eps =', eps)
        print(hfi[-1] - 2 * np.dot(x.T, x))

    import numdifftools as nd
    print('Dea3')
    err = 1000 * np.ones(hfi[0].shape)
    val = np.zeros(err.shape)
    errt = []
    for i in range(len(hfi) - 2):
        tval, terr = nd.dea3(hfi[i], hfi[i + 1], hfi[i + 2])
        errt.append(terr)
        k = np.flatnonzero(terr < err)
        if k.size > 0:
            np.put(val, k, tval.flat[k])
            np.put(err, k, terr.flat[k])
    print(val - 2 * np.dot(x.T, x))
    print(err)
    erri = [v.max() for v in errt]
    import matplotlib.pyplot as plt
    plt.loglog(epsi[1:-1], erri)
    plt.show('hold')
    hnd = nd.Hessian(lambda a: fun2(a, y, x))
    hessnd = hnd(xk)
    print('numdiff')
    print(hessnd - 2 * np.dot(x.T, x))
    # assert_almost_equal(hessnd, he[0])
    gnd = nd.Gradient(lambda a: fun2(a, y, x))
    _gradnd = gnd(xk)

    print(Derivative(np.cosh)(0))
    print(nd.Derivative(np.cosh)(0))

if __name__ == '__main__':  # pragma : no cover
#    main()
#     import nxs
    epsilon = StepsGenerator(num_steps=7)
    d = Derivative(np.cos, method='central', epsilon=epsilon, full_output=True)
    print(d([0, 1e5*np.pi*2]))
#     print(d(1e10*np.pi*2))
