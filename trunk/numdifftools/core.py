"""
Numdifftools implementation

"""
#-------------------------------------------------------------------------
#Author:      Per A. Brodtkorb
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
#import numpy.linalg as linalg
import scipy.misc as misc

__all__ = [
    'dea3', 'Derivative', 'Jacobian', 'Gradient', 'Hessian', 'Hessdiag'
    ]

def dea3(v0, v1, v2):
    '''
    Extrapolate a slowly convergent sequence

    Parameters
    ----------
    v0,v1,v2 : array-like
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
     >>> Ei= np.zeros(3)
     >>> linfun = lambda k : np.linspace(0,np.pi/2.,2.**(k+5)+1)
     >>> for k in np.arange(3): x = linfun(k); Ei[k] = np.trapz(np.sin(x),x)
     >>> [En, err] = dea3(Ei[0],Ei[1],Ei[2])
     >>> truErr = Ei-1.
     >>> (truErr, err, En)
     (array([ -2.00805680e-04,  -5.01999079e-05,  -1.25498825e-05]),
      array([ 0.00020081]),
      array([ 1.]))

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
    abs = np.abs
    max = np.maximum
    zeros = np.zeros
    ten = 10.0
    one = np.ones(1)
    small = np.finfo(float).eps  #1.0e-16 #spacing(one)
    delta2 = E2 - E1
    delta1 = E1 - E0
    err2 = abs(delta2)
    err1 = abs(delta1)
    tol2 = max(abs(E2), abs(E1)) * small
    tol1 = max(abs(E1), abs(E0)) * small

    result = zeros(E0.shape)
    abserr = result.copy()
    converged = (err1 <= tol1) & (err2 <= tol2).ravel()
    k0, = converged.nonzero()
    if k0.size > 0 :
        #%C           IF E0, E1 AND E2 ARE EQUAL TO WITHIN MACHINE
        #%C           ACCURACY, CONVERGENCE IS ASSUMED.
        result[k0] = E2[k0]
        abserr[k0] = err1[k0] + err2[k0] + E2[k0] * small * ten

    k1, = (1 - converged).nonzero()

    if k1.size > 0 :
        ss = one / delta2[k1] - one / delta1[k1]
        smallE2 = (abs(ss * E1[k1]) <= 1.0e-3).ravel()
        k2 = k1[smallE2.nonzero()]
        if k2.size > 0 :
            result[k2] = E2[k2]
            abserr[k2] = err1[k2] + err2[k2] + E2[k2] * small * ten

        k4, = (1 - smallE2).nonzero()
        if k4.size > 0 :
            k3 = k1[k4]
            result[k3] = E1[k3] + one / ss[k4]
            abserr[k3] = err1[k3] + err2[k3] + abs(result[k3] - E2[k3])

    return result, abserr
#max(abserr,small)

def vec2mat(vec, n, m):
    ''' forms the matrix M, such that M(i,j) = vec(i+j)
    '''
    [i, j] = np.ogrid[0:n, 0:m]
    ind = i + j
    return np.matrix(vec[ind])



class _CommonDiffPar(object):
    ''' Object holding common variables and methods for the numdifftools


    Input arguments
    ===============
    fun = function to differentiate.

    **kwds
    ------
    derOrder : Integer from 1 to 4 defining derivative order.     (Default 1)
    metOrder : Integer from 1 to 4 defining order of basic method used.
            For 'central' methods, it must be from the set [2,4]. (Default 2)
    method : Method of estimation.  Valid options are:
            'central', 'forward' or 'backwards'.          (Default 'central')
    numTerms : Number of Romberg terms used in the extrapolation.
            Must be an integer from 0 to 3.                       (Default 2)
            Note: 0 disables the Romberg step completely.
    stepFix : If not None, it will define the maximum excursion from stepNom
                 that is used and prevent the adaptive logic from working.
                 This will be considerably faster, but not necessarily
                 as accurate as allowing the adaptive logic to run.
                (Default: None)
    stepMax  : Maximum allowed excursion from stepNom as a multiple of it. (Default 100)
    stepNom  : Nominal step.                          default maximum(x0, 0.02) 
    stepRatio: Ratio used between sequential steps in the estimation
                 of the derivative (Default 2)
            The steps used h_i = stepNom[i]*stepMax*stepRatio**arange(ndel)
            where ndel = 26 if stepFix is None
                  ndel = 3.+ np.ceil(self.derOrder/2.) + self.metOrder + self.numTerms
    vectorized : True  - if your function is vectorized.
                   False - loop over the successive function calls (default).

    Uses a semi-adaptive scheme to provide the best estimate of the
    derivative by its automatic choice of a differencing interval. It uses
    finite difference approximations of various orders, coupled with a
    generalized (multiple term) Romberg extrapolation. This also yields the
    error estimate provided. See the document DERIVEST.pdf for more explanation
    of the algorithms behind the parameters.

     Note on metOrder: higher order methods will generally be more accurate,
             but may also suffer more from numerical problems. First order
             methods would usually not be recommended.
     Note on method: Central difference methods are usually the most accurate,
            but sometimes one can only allow evaluation in forward or backward
            direction.

    '''
    def __init__(self, fun, **kwds):


        self.fun = fun
        self.derOrder = 1
        self.metOrder = 2
        self.method = 'central'
        self.numTerms = 2
        self.stepFix = None
        self.stepMax = (100.0 + 1.0) - 1.0
        self.stepRatio = (2.0 + 1.0) - 1.0
        self.stepNom = None
        self.vectorized = False


        valid_keys = self.__dict__.keys()
        dict2update = dict((k, kwds[k]) for k in valid_keys if k in kwds)

        if any(dict2update):
            self.__dict__.update(dict2update)

        self._check_params()

        self.error_estimate = None
        self.finaldelta = None

        # The remaining member variables are set by _set_all_der_par
        self.fdarule = None
        self.delta = None
        self.rombexpon = None
        self._fdiff = None


    def _check_params(self):
        ''' check the parameters for acceptability
        '''
        atleast_1d = np.atleast_1d
        kwds = self.__dict__
        for name in ['derOrder', 'metOrder']:
            val = np.atleast_1d(kwds[name])
            if ((len(val) != 1) or (not val in (1, 2, 3, 4))):
                raise ValueError('%s must be scalar, one of [1 2 3 4].' % name)
        name = 'numTerms'
        val = atleast_1d(kwds[name])
        if ((len(val) != 1) or (not val in (0, 1, 2, 3))):
            raise ValueError('%s must be scalar, one of [0 1 2 3].' % name)

        for name in ('stepFix', 'stepMax'):
            val = kwds[name]
            if (val != None and ((len(atleast_1d(val)) > 1) or (val <= 0))):
                raise ValueError('%s must be None or a scalar, >0.' % name)

        validMethods = dict(c='central', f='forward', b='backward')
        method = validMethods.get(kwds['method'][0])
        if method == None :
            t = 'Invalid method: Must start with one of c,f,b characters!'
            raise ValueError(t)
        if method[0] == 'c' and kwds['method'] in (1, 3):
            t = 'metOrder 1 or 3 is not possible for central difference methods'
            raise ValueError(t)

    def _set_all_der_par(self):
        '''Set derivative parameters:
            stepsize, differention rule and romberg extrapolation
        '''
        self._set_delta()
        self._set_fdarule()
        self._set_rombexpon()
        self._set_fdiff()


    def _fdamat(self, parity, nterms):
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
        stepRatio
        '''
        # sr is the ratio between successive steps
        srinv = 1 / self.stepRatio
        factorial = misc.factorial
        arange = np.arange
        [i, j] = np.ogrid[0:nterms, 0:nterms]
        if parity == 0:
            #% single sided rule
            c = 1.0 / factorial(arange(1, nterms + 1))
            mat = c[j] * srinv ** (i * (j + 1))
        elif parity == 1:
            #% odd order derivative
            c = 1.0 / factorial(arange(1, 2 * nterms + 1, 2))
            mat = c[j] * srinv ** (i * (2 * j + 1))
        elif parity == 2:
            #% even order derivative
            c = 1.0 / factorial(arange(2, 2 * nterms + 1, 2))
            mat = c[j] * srinv ** (i * (2 * j + 2))

        return np.matrix(mat)


    def _set_fdarule(self):
        '''
        Generate finite differencing rule in advance.

        The rule is for a nominal unit step size, and will
        be scaled later to reflect the local step size.

        Member methods used
        -------------------
        _fdamat

        Member variables used
        ---------------------
        derOrder
        metOrder
        method
        '''
        derOrder = self.derOrder
        metOrder = self.metOrder
        method = self.method[0]
        array = np.atleast_1d
        matrix = np.matrix
        zeros = np.zeros
        self.fdarule = matrix(derOrder)
        #lstsq = linalg.lstsq
        pinv = linalg.pinv
        if method == 'c' :
            #'central'
            #% for central rules, we will reduce the load by an
            #% even or odd transformation as appropriate.
            if metOrder == 2:
                if derOrder == 1:
                    #% the odd transformation did all the work
                    #self.fdarule[0] = 1
                    pass
                elif derOrder == 2:
                    #% the even transformation did all the work
                    #self.fdarule[0] = 2
                    pass
                elif derOrder == 3:
                    #% the odd transformation did most of the work, but
                    #% we need to kill off the linear term
                    self.fdarule = matrix([0, 1]) * pinv(self._fdamat(1, 2))
                elif derOrder == 4:
                    #% the even transformation did most of the work, but
                    #% we need to kill off the quadratic term
                    self.fdarule = matrix([0, 1]) * pinv(self._fdamat(2, 2))

            #% a 4th order method. We've already ruled out the 1st
            #% order methods since these are central rules.
            elif derOrder == 1:
                # the odd transformation did most of the work, but
                # we need to kill off the cubic term
                self.fdarule = matrix([1, 0]) * pinv(self._fdamat(1, 2))
                #self.fdarule = lstsq(self._fdamat(1,2).T,matrix([1, 0]).T)[0]
            elif derOrder == 2:
                # the even transformation did most of the work, but
                # we need to kill off the quartic term
                self.fdarule = matrix([1, 0]) * pinv(self._fdamat(2, 2))
            elif derOrder == 3:
                # the odd transformation did much of the work, but
                # we need to kill off the linear & quintic terms
                self.fdarule = matrix([0, 1, 0]) * pinv(self._fdamat(1, 3))
            elif derOrder == 4:
                # the even transformation did much of the work, but
                # we need to kill off the quadratic and 6th order terms
                self.fdarule = matrix([0, 1, 0]) * pinv(self._fdamat(2, 3))

        else:
            # Cases {'forward' 'backward'}
            # These two cases are identical, except at the very end,
            # where a sign will be introduced.

            # No odd/even trans, but we already dropped
            # off the constant term
            if metOrder == 1:
                if derOrder == 1:
                    #self.fdarule[0] = 1
                    pass
                else:
                    #% 2:4
                    v = zeros(derOrder)
                    v[derOrder - 1] = 1
                    self.fdarule = matrix(v) * pinv(self._fdamat(0, derOrder))

            else:
                # par.MethodOrder methods drop off the lower order terms,
                # plus terms directly above DerivativeOrder
                v = zeros(derOrder + metOrder - 1)
                v[derOrder - 1] = 1
                dpm = derOrder + metOrder - 1
                self.fdarule = matrix(v) * pinv(self._fdamat(0, dpm))


            #% correct sign for the 'backward' rule
            if method == 'b':
                self.fdarule = -self.fdarule

        self.fdarule = self.fdarule.ravel()
    def _set_delta(self):
        ''' Set the steps to use in derivation.

            Member variables used:

            derOrder
            metOrder
            method
            numTerms
            stepFix
            stepMax
        '''
        # Always choose the stepsize h so that
        # it is an exactly representable number.
        # This is important when calculating numerical derivatives and is
        #  accomplished by the following.
        stepRatio = float(self.stepRatio + 2.0) - 2.0
        
        if self.stepFix == None:
            step1 = float(self.stepMax + 2.0) - 2.0 
            ndel = 26
        else:
            #% Fixed, user supplied absolute sequence of steps.
            step1 = (self.stepFix + 2.0) - 2.0
            ndel = 3. + np.ceil(self.derOrder / 2.) + self.metOrder + self.numTerms
            if self.method[0] == 'c':
                ndel = ndel - 2

        self.delta = step1 * stepRatio ** (-np.arange(ndel))

        #hn^N<sqrt(eps)

    def _set_rombexpon(self):
        '''
          Member variables used
            metOrder
            method
            numTerms

        '''
        add1 = self.method[0] == 'c'
        self.rombexpon = (1 + add1) * np.arange(self.numTerms) + self.metOrder

    def _set_fdiff(self):
        ''' Set _fdiff fun according to method
        '''
        diff_fun = dict(c=self._fdiff_c, b=self._fdiff_b, f=self._fdiff_f)
        self._fdiff = diff_fun[self.method[0]]

    def  _fdiff_c(self, f_x0i, x0i, h):
        ''' Return central differences

        Member variables used
            derOrder
            fun
            vectorized
        '''
        #% A central rule, so we will need to evaluate
        #% symmetrically around x0i.
        fun = self.fun
        if self.vectorized:
            f_plusdel = fun(x0i + h)
            f_minusdel = fun(x0i - h)
        else:
            #% not vectorized, so loop
            f_minusdel = np.zeros_like(h)
            f_plusdel = f_minusdel.copy()
            for j, h_j in enumerate(h):
                f_plusdel[j] = fun(x0i + h_j)
                f_minusdel[j] = fun(x0i - h_j)
        oddOrder = self.derOrder in (1, 3)
        if oddOrder:
            # odd transformation
            f_del = (f_plusdel - f_minusdel) / 2.0
        else:
            f_del = (f_plusdel + f_minusdel) / 2.0 - f_x0i
        return f_del.ravel()

    def _fdiff_f(self, f_x0i, x0i, h):
        ''' Return forward differences

        Member variables used
            fun
            vectorized

        '''
        fun = self.fun
        #% drop off the constant only
        if self.vectorized:
            f_del = fun(x0i + h) - f_x0i
        else:
            f_del = np.zeros_like(h)
            for j , h_j in enumerate(h):
                f_del[j] = fun(x0i + h_j) - f_x0i
        return f_del.ravel()

    def _fdiff_b(self, f_x0i, x0i, h):
        ''' Return backward differences

        Member variables used
        ---------------------
        fun
        vectorized

        '''
        fun = self.fun
        #% drop off the constant only
        if self.vectorized:
            f_del = fun(x0i - h) - f_x0i
        else:
            f_del = np.zeros_like(h)
            for j , h_j in enumerate(h):
                f_del[j] = fun(x0i - h_j) - f_x0i
        return f_del.ravel()



    def _rombextrap(self, der_init):
        ''' Return Romberg extrapolated derivatives and error estimates
            based on the initial derivative estimates

        Parameter
        ---------
        der_init - initial derivative estimates

        Returns
        -------
        der_romb - derivative estimates returned
        errest - error estimates

        Member variables used
        ---------------------
        stepRatio - Ratio decrease in step
        rombexpon - higher order terms to cancel using the romberg step
        '''
        srinv = 1.0 / self.stepRatio
        rombexpon = self.rombexpon

        # do nothing if no romberg terms
        nexpon = len(rombexpon)
        rmat = np.ones((nexpon + 2, nexpon + 1))
        #if nexpon== 0
        # rmat is simple: ones(2,1)
        if nexpon > 0:
            rmat[1, 1:] = srinv ** rombexpon
            for n in range(2, nexpon + 2):
                rmat[n, 1:] = srinv ** (n * rombexpon)


        rmat = np.matrix(rmat)
        #% qr factorization used for the extrapolation as well
        #% as the uncertainty estimates
        [qromb, rromb] = linalg.qr(rmat) #,econ=True)

        # amp - noise amplification factor due to the romberg step
        # the noise amplification is further amplified by the Romberg step.
        #% amp = cond(rromb)

        isnonfinite = 1 - np.isfinite(der_init)
        i_nonfinite, = isnonfinite.ravel().nonzero()

        #% this does the extrapolation to a zero step size.
        ne = der_init.size
        rhs = vec2mat(der_init, nexpon + 2, max(1, ne - (nexpon + 2)))


        if i_nonfinite.size > 0:
            rhsmax = np.nanmax(np.abs(der_init))
            ix_nans = np.isfinite(rhs) == 0
            rhs[ix_nans] = np.random.normal(size=ix_nans.sum())*rhsmax

        rombcoefs = linalg.lstsq(rromb, (qromb.T * rhs))
        der_romb = rombcoefs[0][0, :]

        sqrt = np.sqrt
        sum = np.sum
        asarray = np.asarray
        #% uncertainty estimate of derivative prediction
        s = sqrt(sum(asarray(rhs - rmat * rombcoefs[0]) ** 2, axis=0))
        rinv = asarray(linalg.pinv(rromb))
        cov1 = sum(rinv ** 2, axis=1) # 1 spare dof
        eps = np.finfo(float).eps
        errest = np.maximum(s * 12.7062047361747 * sqrt(cov1[0]), eps * 10.)

        if der_romb.size > 2:
            der_romb, err_dea = dea3(der_romb[0:-2], der_romb[1:-1],
                                        der_romb[2:])
            errest = np.maximum(errest[2:], err_dea)
        #der_dea, err_dea = dea3(der_init[0:-2],der_init[1:-1],der_init[2:])

        return der_romb, errest
        #end % _rombextrap


##'''
##     Arguments: (output)
##      der     = derivative estimate for each element of x0
##      errest  = 95% uncertainty estimate of the derivative, such that
##                 abs(der(j) - f'(x0(j))) < erest(j)
##        fstep = The final overall stepsize chosen by DERIVEST
##
##     Arguments (input)
##      fun = function to differentiate.
##            IF fun is not vectorized, you MUST set 'vectorized' to False.
##      x0  = scalar, vector, or array of points at which to differentiate fun.
##
##    '''
class Derivative(_CommonDiffPar):
    __doc__ = '''Estimate n'th derivative of fun at x0, with error estimate
    ''' + _CommonDiffPar.__doc__.partition('\n')[2] + '''
    Examples
    --------
     # 1'st and 2'nd derivative of exp(x), at x == 1
     >>> import numpy as np
     >>> fd = Derivative(np.exp)              # 1'st derivative
     >>> fdd = Derivative(np.exp,derOrder=2)  # 2'nd derivative
     >>> fd(1)
     array([ 2.71828183])

     >>> d2 = fdd(1)
     >>> fdd.error_estimate # Get error estimate
     array([  5.95741234e-11])

     # 3'rd derivative of x.^3+x.^4, at x = [0,1]
     >>> fun = lambda x: x**3 + x**4
     >>> fd3 = Derivative(fun,derOrder=3)
     >>> fd3([0,1])          #  True derivatives: [6,30]
     array([  6.,  30.])

     >>> fd3.error_estimate
     array([  2.65282677e-14,   1.06113071e-13])

     See also
     --------
     Gradient,
     Hessdiag,
     Hessian,
     Jacobian
    '''
    def __init__(self, fun, **kwds):
        super(Derivative, self).__init__(fun, **kwds)



    def _fder(self, f_x0i, x0i, h):
        ''' Return derivative estimates of f at x0 for a sequence of stepsizes h

        Member variables used
        ---------------------
        derOrder
        fdarule
        numTerms

        '''

        fdarule = self.fdarule
        nfda = fdarule.size
        ndel = h.size


        f_del = self._fdiff(f_x0i, x0i, h)

         #% check the size of f_del to ensure it was properly vectorized.
        if f_del.size != h.size:
            t = 'fun did not return data of correct size (it must be vectorized)'
            raise ValueError(t)

        #% Apply the finite difference rule at each delta, scaling
        #% as appropriate for delta and the requested DerivativeOrder.
        #% First, decide how many of these estimates we will end up with.
        ne = ndel + 1 - nfda - self.numTerms

        # Form the initial derivative estimates from the chosen
        # finite difference method.

        der_init = np.asarray(vec2mat(f_del, ne, nfda) * fdarule.T)

        # scale to reflect the local delta
        der_init = der_init.ravel() / (h[0:ne]) ** self.derOrder

        return der_init

    def __call__(self, x00):
        return self.derivative(x00)

    def derivative(self, x00):
        ''' Return estimate of n'th derivative of fun at x0
            using romberg extrapolation
        '''
        self._set_all_der_par()
        return self._derivative(x00, self.stepNom)

    def _derivative(self, x00, stepNom=None):
        x0 = np.atleast_1d(x00)

        if stepNom is None:
            stepNom = np.maximum(np.abs(x0), 0.02)
        else:
            stepNom = np.atleast_1d(stepNom)
        

        #% was a single point supplied?
        nx0 = x0.shape
        n = x0.size

        f_x0 = np.zeros(nx0)
        #% will we need fun(x0)?
        evenOrder = (np.remainder(self.derOrder, 2) == 0)
        if  evenOrder or not self.method[0] == 'c':
            if self.vectorized:
                f_x0 = self.fun(x0)
            else:
                for j, x0j in enumerate(x0):
                    f_x0[j] = self.fun(x0j)



        #% Loop over the elements of x0, reducing it to
        #% a scalar problem. Sorry, vectorization is not
        #% complete here, but this IS only a single loop.
        der = np.zeros(nx0)
        errest = der.copy()
        finaldelta = der.copy()
        delta = self.delta
        for i in range(n):
            f_x0i = float(f_x0[i])
            x0i = float(x0[i])
            h = (1.0 * stepNom[i]) * delta

            der_init = self._fder(f_x0i, x0i, h)


            #% Each approximation that results is an approximation
            #% of order derOrder to the desired derivative.
            #% Additional (higher order, even or odd) terms in the
            #% Taylor series also remain. Use a generalized (multi-term)
            #% Romberg extrapolation to improve these estimates.


            [der_romb, errors] = self._rombextrap(der_init)

            #% Choose which result to return

            #% first, trim off the
            if self.stepFix == None:
                # trim off the estimates at each end of the scale
                nr_rem = 2 * max((self.derOrder - 1), 1)

                der_romb = np.atleast_1d(der_romb)
                tags = der_romb.argsort()
                tags = tags[nr_rem:-nr_rem]
                der_romb = der_romb[tags]

                errors = errors[tags]
                trimdelta = h[tags]
            else:
                trimdelta = h

            ind = errors.argmin()
            errest[i] = errors[ind]

            finaldelta[i] = trimdelta[ind]
            der[i] = der_romb[ind]

        # Save errorEstimate and final step
        self.error_estimate = errest
        self.finaldelta = finaldelta
        return der


    def _partial_der(self, x00):
        ''' Return partial derivatives
        '''
        x0 = np.atleast_1d(x00)
        nx = len(x0)

        PD = np.zeros(nx)
        err = PD.copy()
        finaldelta = PD.copy()
        
        stepNom = [None,]*nx if self.stepNom is None else self.stepNom 
        self.fun_org = self.fun
        self.fun = self._fun
        self._x = np.asarray(x0, dtype=float)
        for ind in range(nx):
            self._ix = ind
            PD[ind] = self._derivative(x0[ind], stepNom[ind])
            err[ind] = self.error_estimate
            finaldelta[ind] = self.finaldelta

        self.fun = self.fun_org
        self.error_estimate = err
        self.finaldelta = finaldelta
        return PD

    def _fun(self, xi):
        x = self._x.copy()
        x[self._ix] = xi
        return  self.fun_org(x)

    def _gradient(self, x00):

        self.derOrder = 1
        self.vectorized = False

        self._set_all_der_par()
        return self._partial_der(x00)

    def _hessdiag(self, x00):
        self.derOrder = 2
        self.vectorized = False
        self._set_all_der_par()

        return self._partial_der(x00)

    def _hessian(self, x00):

        zeros = np.zeros
        x0 = np.atleast_1d(x00)
        nx = len(x0)
        self.method = 'central'
        #sx = nx #size(x0)

        #% get the diagonal elements of the hessian (2nd partial
        #% derivatives wrt each variable.)
        hess = self._hessdiag(x00)
        err = self.error_estimate

        #% form the eventual hessian matrix, stuffing only
        #% the diagonals for now.
        hess = np.diag(hess)
        err = np.diag(err)
        if nx < 2 :
            #% the hessian matrix is 1x1. all done
            return hess

        #% get the gradient vector. This is done only to decide
        #% on intelligent step sizes for the mixed partials
        #grad = self._gradient(x00)
        stepsize = self.finaldelta #np.maximum(, 1e-4)


        #self.numTerms = 4

        #% Get params.RombergTerms+1 estimates of the upper
        #% triangle of the hessian matrix

        ndel = 3. + np.ceil(self.derOrder / 2.) + self.metOrder + self.numTerms
        if self.method[0] == 'c':
            ndel = ndel - 2
        ndelMin = len(self.rombexpon) + 2
        ndel = np.maximum(ndelMin, ndel)
        #ndel = ndelMin
        dfac = (1.0 * self.stepRatio) ** (-np.arange(ndel))
        fun = self.fun
        for i in range(1, nx):
            for j in range(i):
                dij = zeros(ndel)
                step = zeros(nx)
                step[[i, j]] = stepsize[[i, j]]

                for k in range(int(ndel)):
                    x1 = x0 + step * dfac[k]
                    x2 = x0 - step * dfac[k]
                    step[j] = -step[j]
                    x3 = x0 + step * dfac[k]; step = -step
                    x4 = x0 + step * dfac[k]
                    step[i] = -step[i]
                    dij[k] = fun(x1) + fun(x2) - fun(x3) - fun(x4)

                dij = dij / 4 / stepsize[[i, j]].prod()
                dij = dij / (dfac ** 2)

                #% Romberg extrapolation step
                [hess_romb, errors] = self._rombextrap(dij)
                ind = errors.argmin()

                hess[j, i] = hess[i, j] = hess_romb[ind]
                err[j, i] = err[i, j] = errors[ind]

        self.error_estimate = err
        return hess
class Jacobian(_CommonDiffPar):
    _jacob_txt = _CommonDiffPar.__doc__.partition('\n')[2].replace(
    'Integer from 1 to 4 defining derivative order. (Default 1)',
    'Derivative order is always 1')
    __doc__ = '''Estimate Jacobian matrix, with error estimate
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

    #(nonlinear least squares)
    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> Jfun = Jacobian(fun)
    >>> Jfun([1,2,0.75]) # should be numerically zero
    array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,  -1.30229526e-17],
           [  0.00000000e+00,  -2.12916532e-17,   6.35877095e-17],
           [  0.00000000e+00,   0.00000000e+00,   6.95367972e-19],
           [  0.00000000e+00,   0.00000000e+00,  -2.13524915e-17],
           [  0.00000000e+00,  -3.08563327e-16,   7.43577440e-16],
           [  0.00000000e+00,   1.16128292e-15,   1.71041646e-15],
           [  0.00000000e+00,   0.00000000e+00,  -5.51592310e-16],
           [  0.00000000e+00,  -4.51138245e-19,   1.90866225e-15],
           [ -2.40861944e-19,  -1.82530534e-15,  -4.02819694e-15]])

    >>> Jfun.error_estimate
    array([[  2.22044605e-15,   2.22044605e-15,   2.22044605e-15],
           [  2.22044605e-15,   2.22044605e-15,   2.22044605e-15],
           [  2.22044605e-15,   2.22044605e-15,   2.22044605e-15],
           [  2.22044605e-15,   2.22044605e-15,   2.22044605e-15],
           [  2.22044605e-15,   2.22044605e-15,   2.22044605e-15],
           [  2.22044605e-15,   2.22044605e-15,   2.22044605e-15],
           [  2.22044605e-15,   2.22044605e-15,   2.22044605e-15],
           [  2.22044605e-15,   2.22044605e-15,   6.17089138e-15],
           [  2.22044605e-15,   2.22044605e-15,   2.74306254e-15],
           [  2.22044605e-15,   2.22044605e-15,   2.22044605e-15]])


    See also
    --------
    Gradient,
    Derivative,
    Hessdiag,
    Hessian
    '''
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
        self.derOrder = 1
        fun = self.fun
        self._set_all_der_par()

        zeros = np.zeros
        newaxis = np.newaxis
        x0 = np.atleast_1d(x00)
        nx = x0.size


        #% get fun at the center point
        f0 = fun(x0)
        f0 = f0.ravel()
        n = f0.size

        jac = zeros((n, nx))
        if n == 0 :
            #% empty begets empty
            self.error_estimate = jac
            return jac

        delta = self.delta
        nsteps = delta.size

        if self.stepNom == None :
            stepNom = np.maximum(np.abs(x0), 0.02)
        else:
            stepNom = self.stepNom

        err = jac.copy()
        finaldelta = jac.copy()
        for i in range(nx):
            x0_i = x0[i]
            h = (1.0 * stepNom[i]) * delta

            #% evaluate at each step, centered around x0_i
            #% difference to give a second order estimate
            fdel = zeros((n, nsteps))
            xp = x0.copy()
            xm = x0.copy()
            for j in range(nsteps):
                xp[i] = x0_i + h[j]
                xm[i] = x0_i - h[j]
                fdif = fun(xp) - fun(xm)
                fdel[:, j] = fdif.ravel()


            #% these are pure second order estimates of the
            #% first derivative, for each trial delta.
            derest = fdel * 0.5 / h[newaxis, :]

            #% The error term on these estimates has a second order
            #% component, but also some 4th and 6th order terms in it.
            #% Use Romberg exrapolation to improve the estimates to
            #% 6th order, as well as to provide the error estimate.

            #% loop here, as rombextrap coupled with the trimming
            #% will get complicated otherwise.
            for j in range(n):
                [der_romb, errest] = self._rombextrap(derest[j, :])

                #% trim off 3 estimates at each end of the scale

                tags = der_romb.argsort()
                tags = tags[3:-3]
                der_romb = der_romb[tags]

                errest = errest[tags]
                trimdelta = h[tags]

                #% now pick the estimate with the lowest predicted error
                ind = errest.argmin()
                err[j, i] = errest[ind]
                finaldelta[j, i] = trimdelta[ind]
                jac[j, i] = der_romb[ind]

        self.finaldelta = finaldelta
        self.error_estimate = err
        return jac


class Gradient(Derivative):
    _grad_txt = _CommonDiffPar.__doc__.partition('\n')[2].replace(
        'Integer from 1 to 4 defining derivative order. (Default 1)',
        'Derivative order is always 1')
    __doc__ = '''Estimate gradient of fun at x0, with error estimate
    ''' + _grad_txt + '''

    Assumptions
    -----------
      fun - SCALAR analytical function to differentiate.
            fun must be a function of the vector or array x0,
            but it needs not to be vectorized.

      x0  - vector location at which to differentiate fun
            If x0 is an N x M array, then fun is assumed to be
            a function of N*M variables.


    Examples
    --------
    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = Gradient(fun)
    >>> dfun([1,2,3])
    array([ 2.,  4.,  6.])

    #At [x,y] = [1,1], compute the numerical gradient
    #of the function sin(x-y) + y*exp(x)

    >>> sin = np.sin; exp = np.exp
    >>> z = lambda xy: sin(xy[0]-xy[1]) + xy[1]*exp(xy[0])
    >>> dz = Gradient(z)
    >>> grad2 = dz([1, 1])
    >>> grad2
    array([ 3.71828183,  1.71828183])
    >>> dz.error_estimate
    array([  1.05697754e-12,   6.39224510e-13])

    #At the global minimizer (1,1) of the Rosenbrock function,
    #compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> rd = Gradient(rosen)
    >>> grad3 = rd([1,1])
    >>> grad3
    array([ 0.,  0.])
    >>> rd.error_estimate
    array([  2.22044605e-15,   2.22044605e-15])


    See also
    --------
    Derivative, Hessdiag, Hessian, Jacobian
    '''

    def __call__(self, x00):
        return self._gradient(x00)
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
        return self._gradient(x00)

class Hessian(Derivative):
    _hessian_txt = _CommonDiffPar.__doc__.partition('\n')[2].replace(
        'Integer from 1 to 4 defining derivative order. (Default 1)',
        'Derivative order is always 2')
    __doc__ = ''' Estimate Hessian matrix, with error estimate
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

    #Rosenbrock function, minimized at [1,1]
    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hfun = Hessian(rosen)
    >>> h = Hfun([1, 1]) #  h =[ 842 -420; -420, 210];
    >>> Hfun.error_estimate
    array([[  2.86982123e-12,   1.92513461e-12],
           [  1.92513461e-12,   9.62567303e-13]])

    #cos(x-y), at (0,0)
    >>> cos = np.cos
    >>> fun = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = Hessian(fun)
    >>> h2 = Hfun2([0, 0]) # h2 = [-1 1; 1 -1];
    >>> h2
    array([[-1.,  1.],
           [ 1., -1.]])
    >>> Hfun2.error_estimate
    array([[  4.34170696e-15,   4.34170696e-15],
           [  4.34170696e-15,   4.34170696e-15]])

    >>> Hfun2.numTerms = 3
    >>> h3 = Hfun2([0,0])
    >>> Hfun2.error_estimate
    array([[  1.76511286e-14,   1.29285752e-12],
           [  1.29285752e-12,   1.76511286e-14]])


    See also
    --------
    Gradient,
    Derivative,
    Hessdiag,
    Jacobian
    '''
    def __call__(self, x00):
        return self._hessian(x00)

    def hessian(self, x00):
        '''Hessian matrix i.e., array of 2nd order partial derivatives

         See also derivative, gradient, hessdiag, jacobian

        '''

        return self._hessian(x00)

class Hessdiag(Derivative):
    _hessdiag_txt = _CommonDiffPar.__doc__.partition('\n')[2].replace(
        'Integer from 1 to 4 defining derivative order. (Default 1)',
        'Derivative order is always 2')
    __doc__ = '''Estimate diagonal elements of Hessian of fun at x0,
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
      fun - SCALAR analytical function to differentiate.
            fun must be a function of the vector or array x0,
            but it needs not to be vectorized.

      x0  - vector location at which to differentiate fun
            If x0 is an N x M array, then fun is assumed to be
            a function of N*M variables.

    Examples
    --------

     >>> fun = lambda x : x[0] + x[1]**2 + x[2]**3
     >>> Hfun = Hessdiag(fun)
     >>> hd = Hfun([1,2,3]) # HD = [ 0,2,18]
     >>> Hfun.error_estimate
     array([  2.22044605e-15,   8.68341393e-15,   1.44607740e-13])


     See also
     --------
     Gradient, Derivative, Hessian, Jacobian
    '''
##    def __init__(self,fun,**kwds):
##        super(Hessdiag,self).__init__(fun,**kwds)
##        self.vectorized = False
##        self.derOrder = 2

    def __call__(self, x00):
        return self._hessdiag(x00)
    def hessdiag(self, x00):
        ''' Diagonal elements of Hessian matrix

         See also derivative, gradient, hessian, jacobian
        '''

        return self._hessdiag(x00)

def _test_fun():

    dea3(1.1, 1.05, 1.01)
    xdata = np.reshape(np.arange(0, 1, 0.1), (-1, 1))
    ydata = 1 + 2 * np.exp(0.75 * xdata)
    fun = lambda c: (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2
    rd = Jacobian(fun)
    rd([1, 2, 0.75])
    rd.error_estimate

##    fun = lambda x: x**3 + x**4
##    fd3 = Derivative(fun,derOrder=3)
##    fd3([0,1])          #  True derivatives: [6,30]
##
##    fd = Derivative(np.exp)              # 1'st derivative
##    fdd = Derivative(np.exp,derOrder=2)  # 2'nd derivative
##    d = fd(1)
##    d2 = fdd(1)
##
##    fun = lambda x : x[0] + x[1]**2 + x[2]**3
##    fd = Hessdiag(fun)
##    hd = fd([1,2,3]) # HD = [ 0,2,18]
##    fd.error_estimate
##
##    rosen = lambda x : (1-x[0])**2 + 105*(x[1]-x[0]**2)**2
##    rd = Hessian(rosen)
##    h = rd([1, 1])  #%  h =[ 842 -420; -420, 210];
##    rd.error_estimate




if __name__ == '__main__':
    if  True: # False: #
        import doctest
        doctest.testmod()
    else:
        _test_fun()


