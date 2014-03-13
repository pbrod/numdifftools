"""
Numdifftools implementation

"""
#-------------------------------------------------------------------------
# Author:      Per A. Brodtkorb
#
# Created:     01.08.2008
# Copyright:   (c) pab 2008
# Licence:     New BSD
#
# Based on matlab functions derivest.m gradest.m hessdiag.m, hessian.m
# and jacobianest.m by:
#
# Author: John D'Errico
# e-mail: woodchips@rochester.rr.com
# Release: 1.0
# Release date: 12/27/2006
#-------------------------------------------------------------------------
#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.linalg as linalg
import scipy.misc as misc
import warnings
import matplotlib.pyplot as plt

__all__ = [
    'dea3', 'Derivative', 'Jacobian', 'Gradient', 'Hessian', 'Hessdiag'
]

_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps


def dea3(v0, v1, v2):
    '''
    Extrapolate a slowly convergent sequence

    Parameters
    ----------
    v0, v1, v2 : array-like
        3 values of a convergent sequence to extrapolate

    Returns
    -------
    result : array-like
        extrapolated value
    abserr : array-like
        absolute error estimate

    Description
    -----------
    DEA3 attempts to extrapolate nonlinearly to a better estimate
    of the sequence's limiting value, thus improving the rate of
    convergence. The routine is based on the epsilon algorithm of
    P. Wynn, see [1]_.

     Example
     -------
     # integrate sin(x) from 0 to pi/2

     >>> import numpy as np
     >>> import numdifftools as nd
     >>> Ei= np.zeros(3)
     >>> linfun = lambda k : np.linspace(0,np.pi/2.,2.**(k+5)+1)
     >>> for k in np.arange(3): 
     ...    x = linfun(k) 
     ...    Ei[k] = np.trapz(np.sin(x),x)
     >>> [En, err] = nd.dea3(Ei[0], Ei[1], Ei[2])
     >>> truErr = Ei-1.
     >>> (truErr, err, En)
     (array([ -2.00805680e-04,  -5.01999079e-05,  -1.25498825e-05]),
     array([ 0.00020081]), array([ 1.]))

     See also
     --------
     dea

     Reference
     ---------
     .. [1] C. Brezinski (1977)
            "Acceleration de la convergence en analyse numerique",
            "Lecture Notes in Math.", vol. 584,
            Springer-Verlag, New York, 1977.
    '''
    E0, E1, E2 = np.atleast_1d(v0, v1, v2)
    abs = np.abs  # @ReservedAssignment
    max = np.maximum  # @ReservedAssignment
    delta2, delta1 = E2 - E1, E1 - E0
    err2, err1 = abs(delta2), abs(delta1)
    tol2, tol1 = max(abs(E2), abs(E1)) * _EPS, max(abs(E1), abs(E0)) * _EPS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore division by zero and overflow
        ss = 1.0 / delta2 - 1.0 / delta1
        smallE2 = (abs(ss * E1) <= 1.0e-3).ravel()

    result = 1.0 * E2
    abserr = err1 + err2 + E2 * _EPS * 10.0
    converged = (err1 <= tol1) & (err2 <= tol2).ravel() | smallE2
    k4, = (1 - converged).nonzero()
    if k4.size > 0:
        result[k4] = E1[k4] + 1.0 / ss[k4]
        abserr[k4] = err1[k4] + err2[k4] + abs(result[k4] - E2[k4])
    return result, abserr


def vec2mat(vec, n, m):
    ''' forms the matrix M, such that M(i,j) = vec(i+j)
    '''
    [i, j] = np.ogrid[0:n, 0:m]
    return np.matrix(vec[i + j])


class _Derivative(object):

    ''' Object holding common variables and methods for the numdifftools

    Parameters
    ----------
    fun : callable
        function to differentiate.
    n : Integer from 1 to 4             (Default 1)
        defining derivative order.
    order : Integer from 1 to 4        (Default 2)
        defining order of basic method used.
        For 'central' methods, it must be from the set [2,4].
    method : Method of estimation.  Valid options are:
        'central', 'forward' or 'backward'.          (Default 'central')
    romberg_terms : integer from 0 to 3  (Default 2)
        Number of Romberg terms used in the extrapolation.
        Note: 0 disables the Romberg step completely.
    step_max : real scalar  (Default 2.0)
        Maximum allowed excursion from step_nom as a multiple of it. 
    step_nom : real scalar   default maximum(log1p(abs(x0)), 0.02)
        Nominal step.
    step_ratio: real scalar  (Default 2.0)
        Ratio used between sequential steps in the estimation of the derivative 
    step_num : integer, default 26
        The minimum step_num is
            7 + np.ceil(self.n/2.) + self.order + self.romberg_terms
        The steps: h_i = step_nom[i]*step_max*step_ratio**(-arange(steps_num))
    vectorized : Bool
        True  - if your function is vectorized.
        False - loop over the successive function calls (default).

    Uses a semi-adaptive scheme to provide the best estimate of the
    derivative by its automatic choice of a differencing interval. It uses
    finite difference approximations of various orders, coupled with a
    generalized (multiple term) Romberg extrapolation. This also yields the
    error estimate provided. See the document DERIVEST.pdf for more explanation
    of the algorithms behind the parameters.

     Note on order: higher order methods will generally be more accurate,
             but may also suffer more from numerical problems. First order
             methods would usually not be recommended.
     Note on method: Central difference methods are usually the most accurate,
            but sometimes one can only allow evaluation in forward or backward
            direction.
    '''

    def __init__(self, fun, n=1, order=2, method='central', romberg_terms=2,
                 step_max=2.0, step_nom=None, step_ratio=2.0, step_num=26,
                 vectorized=False, verbose=False):
        self.fun = fun
        self.n = n
        self.order = order
        self.method = method
        self.romberg_terms = romberg_terms
        self.step_max = step_max
        self.step_ratio = step_ratio
        self.step_nom = step_nom
        self.step_num = step_num
        self.vectorized = vectorized
        self.verbose = verbose

        self._check_params()

        self.error_estimate = None
        self.final_delta = None

        # The remaining member variables are set by _initialize
        self._fda_rule = None
        self._delta = None
        self._rmat = None
        self._qromb = None
        self._rromb = None
        self._fdiff = None

    finaldelta = property(lambda cls: cls.final_delta)

    def _check_params(self):
        ''' check the parameters for acceptability
        '''
        atleast_1d = np.atleast_1d
        kwds = self.__dict__
        for name in ['n', 'order']:
            val = np.atleast_1d(kwds[name])
            if ((len(val) != 1) or (not val in (1, 2, 3, 4))):
                raise ValueError('%s must be scalar, one of [1 2 3 4].' % name)
        name = 'romberg_terms'
        val = atleast_1d(kwds[name])
        if ((len(val) != 1) or (not val in (0, 1, 2, 3))):
            raise ValueError('%s must be scalar, one of [0 1 2 3].' % name)

        for name in ('step_max', 'step_num'):
            val = kwds[name]
            if (val != None and ((len(atleast_1d(val)) > 1) or (val <= 0))):
                raise ValueError('%s must be None or a scalar, >0.' % name)

        valid_methods = dict(c='central', f='forward', b='backward')
        method = valid_methods.get(kwds['method'][0])
        if method == None:
            t = 'Invalid method: Must start with one of c, f, b characters!'
            raise ValueError(t)
        if method[0] == 'c' and kwds['method'] in (1, 3):
            t = 'order 1 or 3 is not possible for central difference methods'
            raise ValueError(t)

    def _initialize(self):
        '''Set derivative parameters:
            stepsize, differention rule and romberg extrapolation matrices
        '''
        self._set_delta()
        self._set_fda_rule()
        self._set_romb_qr()
        self._set_difference_function()

    def _fder(self, fun, f_x0i, x0i, h):
        '''
        Return derivative estimates of f at x0 for a sequence of stepsizes h

        Member variables used
        ---------------------
        n
        _fda_rule
        romberg_terms
        '''
        fdarule = self._fda_rule
        nfda = fdarule.size
        ndel = h.size

        f_del = self._fdiff(fun, f_x0i, x0i, h)

        if f_del.size != h.size:
            raise ValueError('fun did not return data of correct size ' +
                             '(it must be vectorized)')

        ne = max(ndel + 1 - nfda - self.romberg_terms, 1)
        der_init = np.asarray(vec2mat(f_del, ne, nfda) * fdarule.T)
        der_init = der_init.ravel() / (h[0:ne]) ** self.n

        return der_init, h[0:ne]

    def _trim_estimates(self, der_romb, errors, h):
        '''
        trim off the estimates at each end of the scale
        '''
        trimdelta = h.copy()
        der_romb = np.atleast_1d(der_romb)
        num_vals = len(der_romb)
        nr_rem_min = int((num_vals - 1) / 2)
        nr_rem = min(2 * max((self.n - 1), 1), nr_rem_min)
        if nr_rem > 0:
            tags = der_romb.argsort()
            tags = tags[nr_rem:-nr_rem]
            der_romb = der_romb[tags]
            errors = errors[tags]
            trimdelta = trimdelta[tags]
        return der_romb, errors, trimdelta

    def _plot_errors(self, h2, errors, step_nom_i, der_romb):
        print('    Stepsize,        Value,           Errors:')
        print((np.vstack((h2, der_romb, errors)).T))

        plt.ioff()
        plt.subplot(2, 1, 1)
        try:
            plt.loglog(h2, der_romb - der_romb.min() + _EPS,
                       h2, der_romb - der_romb.min() + _EPS, '.')
            small = 2 * np.sqrt(_EPS) ** (1. / np.sqrt(self.n))
            plt.vlines(small, 1e-15, 1)
            plt.title('Absolute error as function of stepsize nom=%g' %
                      step_nom_i)
            plt.subplot(2, 1, 2)
            plt.loglog(h2, errors, 'r--', h2, errors, 'r.')
            plt.vlines(small, 1e-15, 1)

            plt.show()
        except:
            pass

    def _get_arg_min(self, errest):
        arg_mins = np.flatnonzero(errest==np.min(errest))
        n = arg_mins.size
        return arg_mins[n // 2]
        return errest.argmin()

    def _get_step_nom(self, step_nom, x0):
        if step_nom is None:
            step_nom = (np.maximum(np.log1p(np.abs(x0)), 0.02) + 1) - 1
        else:
            step_nom = (np.atleast_1d(step_nom) + 1) - 1
        return step_nom

    def _derivative(self, fun, x00, step_nom=None):
        x0 = np.atleast_1d(x00)
        step_nom = self._get_step_nom(step_nom, x0)

        # was a single point supplied?
        nx0 = x0.shape
        n = x0.size

        f_x0 = np.zeros(nx0)
        # will we need fun(x0)?
        evenOrder = (np.remainder(self.n, 2) == 0)
        if evenOrder or not self.method[0] == 'c':
            if self.vectorized:
                f_x0 = fun(x0)
            else:
                f_x0 = np.asfarray([fun(x0j) for x0j in x0])

        der = np.zeros(nx0)
        errest = np.zeros(nx0)
        final_delta = np.zeros(nx0)
        delta = self._delta
        for i in range(n):
            f_x0i = float(f_x0[i])
            x0i = float(x0[i])
            h = ((1.0 * step_nom[i]) * delta + 1 ) -1

            der_init, h1 = self._fder(fun, f_x0i, x0i, h)
            der_romb, errors, h2 = self._romb_extrap(der_init, h1)
            if self.verbose:
                self._plot_errors(h2, errors, step_nom[i], der_romb)
            der_romb, errors, h2 = self._trim_estimates(der_romb, errors, h2)

            ind = self._get_arg_min(errors)
            errest[i] = errors[ind]
            final_delta[i] = h2[ind]
            der[i] = der_romb[ind]

        self.error_estimate = errest
        self.final_delta = final_delta
        return der

    def _fda_mat(self, parity, nterms):
        ''' Return matrix for fda derivation.

        Parameters
        ----------
        parity : scalar, integer
            0 (one sided, all terms included but zeroth order)
            1 (only odd terms included)
            2 (only even terms included)
        nterms : scalar, integer
            number of terms

        Member variables used
        ---------------------
        step_ratio
        '''
        srinv = 1.0 / self.step_ratio
        factorial = misc.factorial
        arange = np.arange
        [i, j] = np.ogrid[0:nterms, 0:nterms]
        if parity == 0:
            c = 1.0 / factorial(arange(1, nterms + 1))
            mat = c[j] * srinv ** (i * (j + 1))
        elif parity == 1 or parity == 2:
            c = 1.0 / factorial(arange(parity, 2 * nterms + 1, 2))
            mat = c[j] * srinv ** (i * (2 * j + parity))
        return np.matrix(mat)

    def _set_fda_rule(self):
        '''
        Generate finite differencing rule in advance.

        The rule is for a nominal unit step size, and will
        be scaled later to reflect the local step size.

        Member methods used
        -------------------
        _fda_mat

        Member variables used
        ---------------------
        n
        order
        method
        '''
        der_order = self.n
        met_order = self.order
        method = self.method[0]

        matrix = np.matrix
        zeros = np.zeros
        fda_rule = matrix(der_order)

        pinv = linalg.pinv
        if method == 'c':  # 'central'
            if met_order == 2:
                if der_order == 3:
                    fda_rule = matrix([0, 1]) * pinv(self._fda_mat(1, 2))
                elif der_order == 4:
                    fda_rule = matrix([0, 1]) * pinv(self._fda_mat(2, 2))
            elif der_order == 1:
                fda_rule = matrix([1, 0]) * pinv(self._fda_mat(1, 2))
            elif der_order == 2:
                fda_rule = matrix([1, 0]) * pinv(self._fda_mat(2, 2))
            elif der_order == 3:
                fda_rule = matrix([0, 1, 0]) * pinv(self._fda_mat(1, 3))
            elif der_order == 4:
                fda_rule = matrix([0, 1, 0]) * pinv(self._fda_mat(2, 3))
        else:
            if met_order == 1:
                if der_order != 1:
                    v = zeros(der_order)
                    v[der_order - 1] = 1
                    fda_rule = matrix(v) * pinv(self._fda_mat(0, der_order))
            else:
                v = zeros(der_order + met_order - 1)
                v[der_order - 1] = 1
                dpm = der_order + met_order - 1
                fda_rule = matrix(v) * pinv(self._fda_mat(0, dpm))
            if method == 'b':  # 'backward' rule
                fda_rule = -fda_rule
        self._fda_rule = fda_rule.ravel()

    def _get_min_num_steps(self):
        n0 = 5 if self.method[0] == 'c' else 7
        return int(n0 + np.ceil(self.n / 2.) + self.order + self.romberg_terms)

    def _set_delta(self):
        ''' Set the steps to use in derivation.

            Member variables used:

            n
            order
            method
            romberg_terms
            step_max
        '''
        # Choose the step size h so that it is an exactly representable number.
        # This is important when calculating numerical derivatives and is
        #  accomplished by the following.
        step_ratio = float(self.step_ratio + 1.0) - 1.0
        if self.step_num is None:
            num_steps = self._get_min_num_steps()
        else:
            num_steps = max(self.step_num, 1)
        step1 = float(self.step_max + 1.0) - 1.0
        self._delta = step1 * step_ratio ** (-np.arange(num_steps))

    def _set_romb_qr(self):
        '''
        Member variables used
            order
            method
            romberg_terms
        '''
        nexpon = self.romberg_terms
        add1 = self.method[0] == 'c'
        rombexpon = (1 + add1) * np.arange(nexpon) + self.order

        srinv = 1.0 / self.step_ratio
        rmat = np.ones((nexpon + 2, nexpon + 1))
        if nexpon > 0:
            rmat[1, 1:] = srinv ** rombexpon
            for n in range(2, nexpon + 2):
                rmat[n, 1:] = srinv ** (n * rombexpon)
        rmat = np.matrix(rmat)
        self._qromb, self._rromb = linalg.qr(rmat)
        self._rmat = rmat

    def _set_difference_function(self):
        ''' Set _fdiff function according to method
        '''
        get_diff_fun = dict(c=self._central, b=self._backward,
                            f=self._forward)[self.method[0]]
        self._fdiff = get_diff_fun()

    def _central(self):
        ''' Return central difference function

        Member variables used
            n
            fun
            vectorized
        '''
        even_order = (np.remainder(self.n, 2) == 0)

        if self.vectorized:
            if even_order:
                f_del = lambda fun, f_x0i, x0i, h: (
                    fun(x0i + h) + fun(x0i - h)).ravel() / 2.0 - f_x0i
            else:
                f_del = lambda fun, f_x0i, x0i, h: (
                    fun(x0i + h) - fun(x0i - h)).ravel() / 2.0
        else:
            if even_order:
                f_del = lambda fun, f_x0i, x0i, h: np.asfarray(
                    [fun(x0i + h_j) + fun(x0i - h_j)
                                for h_j in h]).ravel() / 2.0 - f_x0i
            else:
                f_del = lambda fun, f_x0i, x0i, h: np.asfarray(
                    [fun(x0i + h_j) - fun(x0i - h_j)
                                        for h_j in h]).ravel() / 2.0
        return f_del

    def _forward(self):
        ''' Return forward difference function

        Member variables used
            fun
            vectorized
        '''
        if self.vectorized:
            f_del = lambda fun, f_x0i, x0i, h: (fun(x0i + h) - f_x0i).ravel()
        else:
            f_del = lambda fun, f_x0i, x0i, h: np.asfarray(
                [fun(x0i + h_j) - f_x0i for h_j in h]).ravel()
        return f_del

    def _backward(self):
        ''' Return backward difference function

        Member variables used
        ---------------------
        fun
        vectorized

        '''
        if self.vectorized:
            f_del = lambda fun, f_x0i, x0i, h: (fun(x0i - h) - f_x0i).ravel()
        else:
            f_del = lambda fun, f_x0i, x0i, h: np.asfarray(
                [fun(x0i - h_j) - f_x0i for h_j in h]).ravel()
        return f_del

    def _remove_non_finite(self, der_init, h1):
        isnonfinite = 1 - np.isfinite(der_init)
        i_nonfinite, = isnonfinite.ravel().nonzero()
        hout = h1
        if i_nonfinite.size > 0:
            allfinite_start = np.max(i_nonfinite) + 1
            der_init = der_init[allfinite_start:]
            hout = h1[allfinite_start:]
        return der_init, hout

    def _predict_uncertainty(self, rombcoefs, rhs):
        '''uncertainty estimate of derivative prediction'''
        sqrt = np.sqrt
        asarray = np.asarray

        s = sqrt(np.sum(asarray(rhs - self._rmat * rombcoefs[0]) ** 2, axis=0))
        rinv = asarray(linalg.pinv(self._rromb))
        cov1 = np.sum(rinv ** 2, axis=1)  # 1 spare dof
        errest = np.maximum(
            s * 12.7062047361747 * sqrt(cov1[0]), s * _EPS * 10.)
        errest = np.where(s == 0, _EPS, errest)
        return errest

    def _romb_extrap(self, der_init, h1):
        ''' Return Romberg extrapolated derivatives and error estimates
            based on the initial derivative estimates

        Parameter
        ---------
        der_init - initial derivative estimates
        h1 - stepsizes used in the derivative estimates

        Returns
        -------
        der_romb - derivative estimates returned
        errest - error estimates
        hout - stepsizes returned

        Member variables used
        ---------------------
        step_ratio - Ratio decrease in step
        rombexpon - higher order terms to cancel using the romberg step
        '''
        # amp = linalg.cond(self._rromb)
        # amp - noise amplification factor due to the romberg step
        # the noise amplification is further amplified by the Romberg step
        der_romb, hout = self._remove_non_finite(der_init, h1)
        # this does the extrapolation to a zero step size.
        nexpon = self.romberg_terms
        ne = der_romb.size
        if ne < nexpon + 2:
            errest = np.ones(der_init.shape) * hout
        else:
            rhs = vec2mat(der_romb, nexpon + 2, max(1, ne - (nexpon + 2)))

            rombcoefs = linalg.lstsq(self._rromb, (self._qromb.T * rhs))
            der_romb = rombcoefs[0][0, :]
            hout = hout[:der_romb.size]

            errest = self._predict_uncertainty(rombcoefs, rhs)
        if der_romb.size > 2:
            der_romb, err_dea = dea3(der_romb[0:-2], der_romb[1:-1],
                                     der_romb[2:])
            errest = np.maximum(errest[2:], err_dea)
            hout = hout[2:]
        return der_romb, errest, hout


class _PartialDerivative(_Derivative):

    def _partial_der(self, x00):
        ''' Return partial derivatives
        '''
        x0 = np.atleast_1d(x00)
        nx = len(x0)

        PD = np.zeros(nx)
        err = np.zeros(nx)
        final_delta = np.zeros(nx)

        step_nom = [None, ] * nx if self.step_nom is None else self.step_nom

        fun = self._fun
        self._x = np.asarray(x0, dtype=float)
        for ind in range(nx):
            self._ix = ind
            PD[ind] = self._derivative(fun, x0[ind], step_nom[ind])
            err[ind] = self.error_estimate
            final_delta[ind] = self.final_delta
        self.error_estimate = err
        self.final_delta = final_delta
        return PD

    def _fun(self, xi):
        x = self._x.copy()
        x[self._ix] = xi
        return self.fun(x)


class Derivative(_Derivative):
    __doc__ = (  # @ReservedAssignment
'''Estimate n'th derivative of fun at x0, with error estimate 
    ''' + _Derivative.__doc__.partition('\n')[2] + '''
    Examples
    --------
     >>> import numpy as np
     >>> import numdifftools as nd
     
     # 1'st and 2'nd derivative of exp(x), at x == 1
     >>> fd = nd.Derivative(np.exp)       # 1'st derivative
     >>> fdd = nd.Derivative(np.exp,n=2)  # 2'nd derivative
     >>> fd(1)
     array([ 2.71828183])

     >>> d2 = fdd([1, 2])
     >>> d2
     array([ 2.71828183,  7.3890561 ])
     
     >>> np.abs(d2-np.exp([1,2]))< fdd.error_estimate # Check error estimate
     array([ True,  True], dtype=bool)

     # 3'rd derivative of x.^3+x.^4, at x = [0,1]
     >>> fun = lambda x: x**3 + x**4
     >>> dfun = lambda x: 6 + 4*3*2*np.asarray(x)
     >>> fd3 = nd.Derivative(fun,n=3)
     >>> fd3([0,1])          #  True derivatives: [6,30]
     array([  6.,  30.])

     >>> np.abs(fd3([0,1])-dfun([0,1])) <= fd3.error_estimate
     array([ True,  True], dtype=bool)

     See also
     --------
     Gradient,
     Hessdiag,
     Hessian,
     Jacobian
    ''')

    def __call__(self, x00):
        return self.derivative(x00)

    def derivative(self, x0):
        ''' Return estimate of n'th derivative of fun at x0
            using romberg extrapolation
        '''
        self._initialize()
        x00 = np.atleast_1d(x0)
        shape = x00.shape
        tmp = self._derivative(self.fun, x00.ravel(), self.step_nom)
        return tmp.reshape(shape)


class Jacobian(_Derivative):
    _jacob_txt = _Derivative.__doc__.partition('\n')[2].replace(
        'Integer from 1 to 4 defining derivative order. (Default 1)',
        'Derivative order is always 1')
    __doc__ = (  # @ReservedAssignment
'''Estimate Jacobian matrix, with error estimate
    ''' + _jacob_txt + '''

    The Jacobian matrix is the matrix of all first-order partial derivatives
    of a vector-valued function.

    Assumptions
    -----------
    fun : (vector valued)
        analytical function to differentiate.
        fun must be a function of the vector or array x0.

    x0 : vector location at which to differentiate fun
        If x0 is an N x M array, then fun is assumed to be
        a function of N*M variables.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    
    #(nonlinear least squares)
    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> Jfun = nd.Jacobian(fun)
    >>> h = Jfun([1., 2., 0.75]) # should be numerically zero
    >>> np.abs(h) < 1e-14
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    
    >>> np.abs(h) <= 100 * Jfun.error_estimate
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
     
    See also
    --------
    Gradient,
    Derivative,
    Hessdiag,
    Hessian
    ''')

    def __call__(self, x00):
        return self.jacobian(x00)

    def jacobian(self, x00):
        '''
        Return Jacobian matrix of a vector valued function of n variables


        Parameter
        ---------
        x0 : vector
            location at which to differentiate fun.
            If x0 is an nxm array, then fun is assumed to be
            a function of n*m variables.

        Member variable used
        --------------------
        fun : (vector valued) analytical function to differentiate.
                fun must be a function of the vector or array x0.

        Returns
        -------
        jac : array-like
           first partial derivatives of fun. Assuming that x0
           is a vector of length p and fun returns a vector
           of length n, then jac will be an array of size (n,p)

        err - vector
            of error estimates corresponding to each partial
            derivative in jac.

        See also
        --------
        Derivative,
        Gradient,
        Hessian,
        Hessdiag
        '''
        self.n = 1
        fun = self.fun
        self._initialize()

        zeros = np.zeros
        newaxis = np.newaxis
        x0 = np.atleast_1d(x00)
        nx = x0.size

        f0 = fun(x0)
        f0 = f0.ravel()
        n = f0.size

        jac = zeros((n, nx))
        if n == 0:
            self.error_estimate = jac
            return jac

        delta = self._delta
        nsteps = delta.size
        
        step_nom = self._get_step_nom(self.step_nom, x0)

        err = jac.copy()
        final_delta = jac.copy()
        for i in range(nx):
            x0_i = x0[i]
            h = ((1.0 * step_nom[i]) * delta + 1) - 1

            # evaluate at each step, centered around x0_i
            # difference to give a second order estimate
            fdel = zeros((n, nsteps))
            xp = x0.copy()
            xm = x0.copy()
            for j in range(nsteps):
                xp[i] = x0_i + h[j]
                xm[i] = x0_i - h[j]
                fdif = fun(xp) - fun(xm)
                fdel[:, j] = 0.5 * fdif.ravel()
            derest = fdel / h[newaxis, :]

            for j in range(n):
                der_romb, errest, h1 = self._romb_extrap(derest[j, :], h)
                der_romb, errest, h1 = self._trim_estimates(
                    der_romb, errest, h)
                ind = self._get_arg_min(errest)
                err[j, i] = errest[ind]
                final_delta[j, i] = h1[ind]
                jac[j, i] = der_romb[ind]

        self.final_delta = final_delta
        self.error_estimate = err
        return jac


class Gradient(_PartialDerivative):
    _grad_txt = _Derivative.__doc__.partition('\n')[2].replace(
        'Integer from 1 to 4 defining derivative order. (Default 1)',
        'Derivative order is always 1')
    __doc__ = (  # @ReservedAssignment
    '''Estimate gradient of fun at x0, with error estimate
    ''' + _grad_txt + '''

    Assumptions
    -----------
    fun : SCALAR analytical function to differentiate.
        fun must be a function of the vector or array x0,
        but it needs not to be vectorized.

    x0 : vector location at which to differentiate fun
        If x0 is an N x M array, then fun is assumed to be
        a function of N*M variables.


    Examples
    --------
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
    >>> np.abs(grad3 - 0)<=rd.error_estimate
    array([ True,  True], dtype=bool)


    See also
    --------
    Derivative, Hessdiag, Hessian, Jacobian
    ''')

    def __call__(self, x00):
        return self.gradient(x00)

    def gradient(self, x00):
        ''' Gradient vector of an analytical function of n variables

         CALL: [grad,err,finaldelta] = fun.gradient(x0)

          grad = first partial derivatives of fun evaluated at x0.    Size 1 x N
          err  = error estimates corresponding to each value in grad. Size 1 x N
          finaldelta = vector of final step sizes chosen for each partial derivative.
          fun  = analytical function to differentiate. fun must
                be a function of the vector or array x0.
          x0   = vector location at which to differentiate fun
                If x0 is an nxm array, then fun is assumed to be
                a function of N = n*m variables.

         GRADEST estimate first partial derivatives of fun evaluated at x0.
         GRADEST uses derivest to provide both derivative estimates
         and error estimates. fun needs not be vectorized.

         Examples

          #[grad,err] = gradest(@(x) sum(x.^2),[1 2 3]) #  grad = [ 2,4, 6]

        '''
        self.n = 1
        self.vectorized = False

        self._initialize()
        return self._partial_der(x00)


class Hessdiag(_PartialDerivative):
    _hessdiag_txt = _Derivative.__doc__.partition('\n')[2].replace(
        'Integer from 1 to 4 defining derivative order. (Default 1)',
        'Derivative order is always 2')
    __doc__ = (  # @ReservedAssignment
    '''Estimate diagonal elements of Hessian of fun at x0,
    with error estimate
    ''' + _hessdiag_txt + '''

    HESSDIAG return a vector of second order partial derivatives of fun.
    These are the diagonal elements of the Hessian matrix, evaluated
    at x0.  When all that you want are the diagonal elements of the hessian
    matrix, it will be more efficient to call HESSDIAG than HESSIAN.
    HESSDIAG uses DERIVATIVE to provide both second derivative estimates
    and error estimates.

    Assumptions
    ------------
    fun : SCALAR analytical function to differentiate.
        fun must be a function of the vector or array x0,
        but it needs not to be vectorized.

    x0 : vector location at which to differentiate fun
        If x0 is an N x M array, then fun is assumed to be
        a function of N*M variables.

    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools as nd
    >>> fun = lambda x : x[0] + x[1]**2 + x[2]**3
    >>> ddfun = lambda x : np.asarray((0, 2, 6*x[2]))
    >>> Hfun = nd.Hessdiag(fun)
    >>> hd = Hfun([1,2,3]) # HD = [ 0,2,18]
    >>> hd
    array([  0.,   2.,  18.])
    >>> np.abs(ddfun([1,2,3])-hd) <= Hfun.error_estimate
    array([ True,  True,  True], dtype=bool)


    See also
    --------
    Gradient, Derivative, Hessian, Jacobian
    ''')

    def __call__(self, x00):
        return self.hessdiag(x00)

    def hessdiag(self, x00):
        ''' Diagonal elements of Hessian matrix

         See also derivative, gradient, hessian, jacobian
        '''
        self.n = 2
        self.vectorized = False
        self._initialize()
        return self._partial_der(x00)


class Hessian(Hessdiag):
    _hessian_txt = _Derivative.__doc__.partition('\n')[2].replace(
        'Integer from 1 to 4 defining derivative order. (Default 1)',
        'Derivative order is always 2')
    __doc__ = (  # @ReservedAssignment
    ''' Estimate Hessian matrix, with error estimate
    ''' + _hessian_txt + '''

    HESSIAN estimate the matrix of 2nd order partial derivatives of a real
    valued function FUN evaluated at X0. HESSIAN is NOT a tool for frequent
    use on an expensive to evaluate objective function, especially in a large
    number of dimensions. Its computation will use roughly  O(6*n^2) function
    evaluations for n parameters.

    Assumptions
    -----------
    fun : SCALAR analytical function
        to differentiate. fun must be a function of the vector or array x0,
        but it needs not to be vectorized.

    x0 : vector location
        at which to differentiate fun
        If x0 is an N x M array, then fun is assumed to be a function
        of N*M variables.

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
    >>> Hfun.error_estimate < 1.e-11
    array([[ True,  True],
           [ True,  True]], dtype=bool)

    # cos(x-y), at (0,0)
    >>> cos = np.cos
    >>> fun = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = nd.Hessian(fun)
    >>> h2 = Hfun2([0, 0]) 
    >>> h2
    array([[-1.,  1.],
           [ 1., -1.]])
    >>> np.abs(h2-np.array([[-1,  1],[ 1, -1]])) < Hfun2.error_estimate
    array([[ True,  True],
           [ True,  True]], dtype=bool)

    >>> Hfun2.romberg_terms = 3
    >>> h3 = Hfun2([0,0])
    >>> h3
    array([[-1.,  1.],
           [ 1., -1.]])
    >>> np.abs(h3-np.array([[-1,  1],[ 1, -1]])) < Hfun2.error_estimate
    array([[ True,  True],
           [ True,  True]], dtype=bool)


    See also
    --------
    Gradient,
    Derivative,
    Hessdiag,
    Jacobian
    ''')

    def __call__(self, x00):
        return self.hessian(x00)

    def hessian(self, x00):
        '''Hessian matrix i.e., array of 2nd order partial derivatives

         See also derivative, gradient, hessdiag, jacobian
        '''
        x0 = np.atleast_1d(x00)
        nx = len(x0)
        self.method = 'central'

        hess = self.hessdiag(x0)
        err = self.error_estimate

        hess, err = np.diag(hess), np.diag(err)
        if nx < 2:
            return hess  # the hessian matrix is 1x1. all done

        # Decide on intelligent step sizes for the mixed partials
        stepsize = self.final_delta
        ndel = np.maximum(self._get_min_num_steps(), self.romberg_terms + 2)

        dfac = ((1.0 * self.step_ratio) ** (-np.arange(ndel)) + 1) - 1
        stepmax = stepsize / dfac[ndel // 2]
        fun = self.fun
        zeros = np.zeros
        for i in range(1, nx):
            for j in range(i):
                dij, step = zeros(ndel), zeros(nx)
                step[[i, j]] = stepmax[[i, j]]
                for k in range(int(ndel)):
                    x1 = x0 + step * dfac[k]
                    x2 = x0 - step * dfac[k]
                    step[j] = -step[j]
                    x3 = x0 + step * dfac[k]
                    step = -step
                    x4 = x0 + step * dfac[k]
                    step[i] = -step[i]
                    dij[k] = fun(x1) + fun(x2) - fun(x3) - fun(x4)
                dij = dij / 4 / stepmax[[i, j]].prod() / (dfac ** 2)

                hess_romb, errors, _dfac1 = self._romb_extrap(dij, dfac)
                ind = self._get_arg_min(errors)
                hess[j, i] = hess[i, j] = hess_romb[ind]
                err[j, i] = err[i, j] = errors[ind]

        self.error_estimate = err
        return hess


def test_docstrings():
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    # test_docstrings()
    fun = np.cos
    dfun = lambda x: -np.sin(x)
    #print((2 * np.sqrt(1e-16)))
    fun = np.tanh
    dfun = lambda x : 1./np.cosh(x)**2
    fun  = np.log
    dfun = lambda x : 1./x
#    fun  = lambda x : 1./x
#    dfun = lambda x : -1./x**2

    h = 1e-4
    fd = Derivative(fun, method='forward', step_max=1, step_ratio=2, step_num=32,
                    verbose=True, vectorized=True, romberg_terms=2)
    x = 0.01
    t = fd(x)
    print(('(f(x+h)-f(x))/h = %g\n true df(x) = %g\n estimated df(x) = %g\n' +
          ' true err = %g\n err estimate = %g\n relative err = %g\n' +
          ' final_delta = %g\n') %  ((fun(x + h) - fun(x)) / (h),
                                     dfun(x), t, dfun(x) - t,
                                     fd.error_estimate,
                                     fd.error_estimate / t,
                                     fd.final_delta))
