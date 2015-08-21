'''
Easy to use interface to derivatives in algopy
'''
from __future__ import division
import numpy as np
from scipy import misc
try:
    import algopy
    from algopy import UTPM
except ImportError:
    algopy = None


class _Common(object):
    def __init__(self, f, method='forward'):
        self.f = f
        self.method = method
        self.initialize()

    def _initialize_reverse(self, x):
        # x = np.asarray(x, dtype=float)
        # self.x = x.copy()
        # STEP 1: trace the function evaluation
        cg = algopy.CGraph()
        if True:
            x = algopy.Function(x)
            # x = [x]
        else:
            x = np.array([algopy.Function(x[i]) for i in range(len(x))])

        y = self.f(x)
        cg.trace_off()
        cg.independentFunctionList = [x]
        cg.dependentFunctionList = [y]
        self._cg = cg

    def initialize(self):
        if self.method.startswith('reverse'):
            # reverse mode using a computational graph
            # self._initialize_reverse(x)
            self._gradient = self._gradient_reverse
            self._hessian = self._hessian_reverse
            self._jacobian = self._jacobian_reverse
        else:  # forward mode without building the computational graph
            self._gradient = self._gradient_forward
            self._hessian = self._hessian_forward
            self._jacobian = self._gradient_forward

    def _derivative(self, x):
        xi = np.asarray(x, dtype=float)
        shape0 = xi.shape
        y = np.array([self._gradient(xj) for xj in xi.ravel()])
        return y.reshape(shape0)

    def _jacobian_reverse(self, x):
        self._initialize_reverse(x)
        return self._cg.jacobian([np.asarray(x)])

    def _gradient_reverse(self, x):
        self._initialize_reverse(x)
        return self._cg.gradient([np.asarray(x)])

    def _hessian_reverse(self, x):
        self._initialize_reverse(x)
        return self._cg.hessian([np.asarray(x)])
        # return self._cg.hessian([x])

    def _jacobian_forward(self, x):
        x = np.asarray(x)
        # shape = x.shape
        D, Nm = 2, x.size
        P = Nm
        y = UTPM(np.zeros((D, P, Nm)))

        y.data[0, :] = x.ravel()
        y.data[1, :] = np.eye(Nm)
        z0 = self.f(y)
        z = UTPM.as_utpm(z0)
        J = z.data[1, :, :, 0]
        return J

    def _gradient_forward(self, x):
        # forward mode without building the computational graph

        tmp = algopy.UTPM.init_jacobian(np.asarray(x, dtype=float))
        tmp2 = self.f(tmp)
        return algopy.UTPM.extract_jacobian(tmp2)

    def _hessian_forward(self, x):
        tmp = algopy.UTPM.init_hessian(np.asarray(x, dtype=float))
        tmp2 = self.f(tmp)
        return algopy.UTPM.extract_hessian(len(x), tmp2)


class Derivative(object):
    '''
    Estimate n'th derivative of f at x0

     Parameters
    ----------
    f : function
       function of one array f(x, `*args`, `**kwargs`)
    n : int, optional
        Order of the derivative.

    Call Parameters
    ---------------
    x : array_like
       value at which function derivative is evaluated
    args : tuple
        Arguments for function `f`.
    kwds : dict
        Keyword arguments for function `f`.


    Examples
    --------
    # 1'st and 2'nd derivative of exp(x), at x == 1

    >>> import numpy as np
    >>> import numdifftools.nd_algopy as nda
    >>> fd = nda.Derivative(np.exp)              # 1'st derivative
    >>> np.allclose(fd(1), 2.718281828459045)
    True
    >>> fd5 = nda.Derivative(np.exp, n=5)         # 5'th derivative
    >>> np.allclose(fd5(1), 2.718281828459045)
    True

    # 1'st derivative of x.^3+x.^4, at x = [0,1]

    >>> f = lambda x: x**3 + x**4
    >>> fd3 = nda.Derivative(f)
    >>> np.allclose(fd3([0,1]), [ 0.,  7.])
    True

    See also
    --------
    Gradient,
    Hessdiag,
    Hessian,
    Jacobian
    '''

    def __init__(self, f, n=1, method='forward'):
        self.f = f
        self.n = n
        self.method = method

    def derivative(self, x0, *args, **kwds):
        return self(x0, *args, **kwds)

    def __call__(self, x0, *args, **kwds):
        x0 = np.asarray(x0)
        shape = x0.shape
        P = 1
        x = UTPM(np.zeros((self.n + 1, P) + shape))
        x.data[0, 0] = x0
        x.data[1, 0] = 1

        y = UTPM.as_utpm(self.f(x, *args, **kwds))

        return y.data[self.n, 0] * misc.factorial(self.n)


class Jacobian(_Common):
    '''Estimate Jacobian matrix

    The Jacobian matrix is the matrix of all first-order partial derivatives
    of a vector-valued function.

    Assumptions
    -----------
    f : (vector valued)
        analytical function to differentiate.
        f must be a function of the vector or array x0.

    x0 : vector location at which to differentiate f
        If x0 is an N x M array, then f is assumed to be
        a function of N*M variables.

    Examples
    --------
    >>> import numdifftools.nd_algopy as nda

    #(nonlinear least squares)

    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> f = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2

    Jfun = nda.Jacobian(f) # Todo: This does not work
    Jfun([1,2,0.75]) # should be numerically zero
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    Jfun2 = Jacobian(f, method='reverse')
    Jfun2([1,2,0.75]) # should be numerically zero
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    >>> f2 = lambda x : x[0]*x[1]*x[2] + np.exp(x[0])*x[1]
    >>> Jfun3 = nda.Jacobian(f2)
    >>> Jfun3([3.,5.,7.])
    array([ 135.42768462,   41.08553692,   15.        ])

    Jfun4 = nda.Jacobian(f2, method='reverse')
    Jfun4([3,5,7])

    See also
    --------
    Gradient,
    Derivative,
    Hessdiag,
    Hessian
    '''
    def __call__(self, x0):
        return self.jacobian(x0)

    def jacobian(self, x0):
        '''
        Return Jacobian matrix of a vector valued function of n variables


        Parameter
        ---------
        x0 : vector
            location at which to differentiate f.
            If x0 is an nxm array, then f is assumed to be
            a function of n*m variables.

        Member variable used
        --------------------
        f : (vector valued) analytical function to differentiate.
                f must be a function of the vector or array x0.

        Returns
        -------
        jac : array-like
           first partial derivatives of f. Assuming that x0
           is a vector of length p and f returns a vector
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
        return self._jacobian(x0)


class Gradient(_Common):
    '''Estimate gradient of f at x0

    Assumptions
    -----------
      f - SCALAR analytical function to differentiate.
            fun must be a function of the vector or array x0,
            but it needs not to be vectorized.

      x0  - vector location at which to differentiate f
            If x0 is an N x M array, then f is assumed to be
            a function of N*M variables.


    Examples
    --------
    >>> import numdifftools.nd_algopy as nda
    >>> f = lambda x: np.sum(x**2)
    >>> df = nda.Gradient(f)
    >>> df([1,2,3])
    array([ 2.,  4.,  6.])

    #At [x,y] = [1,1], compute the numerical gradient
    #of the function sin(x-y) + y*exp(x)

    >>> sin = np.sin; exp = np.exp
    >>> z = lambda xy: sin(xy[0]-xy[1]) + xy[1]*exp(xy[0])
    >>> dz = nda.Gradient(z)
    >>> grad2 = dz([1, 1])
    >>> grad2
    array([ 3.71828183,  1.71828183])

    #At the global minimizer (1,1) of the Rosenbrock function,
    #compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> rd = nda.Gradient(rosen)
    >>> grad3 = rd([1,1])
    >>> grad3==np.array([ 0.,  0.])
    array([ True,  True], dtype=bool)

    See also
    --------
    Derivative, Hessdiag, Hessian, Jacobian
    '''

    def gradient(self, x0):
        ''' Gradient vector of an analytical function of n variables
        '''
        return self._gradient(x0)

    def __call__(self, x):
        return self._gradient(x)


class Hessian(_Common):
    ''' Estimate Hessian matrix

    HESSIAN estimate the matrix of 2nd order partial derivatives of a real
    valued function FUN evaluated at X0.

    Assumptions
    -----------
    f : SCALAR analytical function
        to differentiate. f must be a function of the vector or array x0,
        but it needs not to be vectorized.

    x0 : vector location
        at which to differentiate f
        If x0 is an N x M array, then f is assumed to be a function
        of N*M variables.

    Examples
    --------
    >>> import numdifftools.nd_algopy as nda

    # Rosenbrock function, minimized at [1,1]

    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hf = nda.Hessian(rosen)
    >>> h = Hf([1, 1]) #  h =[ 842 -420; -420, 210];
    >>> h
    array([[ 842., -420.],
           [-420.,  210.]])

    # cos(x-y), at (0,0)

    >>> cos = np.cos
    >>> f = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = nda.Hessian(f)
    >>> h2 = Hfun2([0, 0]) # h2 = [-1 1; 1 -1]
    >>> h2
    array([[-1.,  1.],
           [ 1., -1.]])

    Hfun3 = Hessian(f, method='reverse') # TODO: Hfun3 fails in this case
    h3 = Hfun3([0, 0]) # h2 = [-1, 1; 1, -1];
    h3
    array([[[-1.,  1.],
            [ 1., -1.]]])

    See also
    --------
    Gradient,
    Derivative,
    Hessdiag,
    Jacobian
    '''

    def hessian(self, x0):
        '''Hessian matrix i.e., array of 2nd order partial derivatives

        See also
        --------
        derivative, gradient, hessdiag, jacobian
        '''
        return self._hessian(x0)

    def __call__(self, x):
        return self._hessian(x)


class Hessdiag(Hessian):
    ''' Estimate Hessian matrix

    HESSIAN estimate the matrix of 2nd order partial derivatives of a real
    valued function FUN evaluated at X0.

    Assumptions
    -----------
    f : SCALAR analytical function
        to differentiate. f must be a function of the vector or array x0,
        but it needs not to be vectorized.

    x0 : vector location
        at which to differentiate f
        If x0 is an N x M array, then f is assumed to be a function
        of N*M variables.

    Examples
    --------
    >>> import numdifftools.nd_algopy as nda

    # Rosenbrock function, minimized at [1,1]

    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hfun = nda.Hessdiag(rosen)
    >>> h = Hfun([1, 1]) #  h =[ 842, 210]
    >>> h
    array([ 842.,  210.])

    # cos(x-y), at (0,0)

    >>> cos = np.cos
    >>> f = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = nda.Hessdiag(f)
    >>> h2 = Hfun2([0, 0]) # h2 = [-1, -1]
    >>> h2
    array([-1., -1.])

    Hfun3 = Hessdiag(f, method='reverse') # TODO: Hfun3 fails in this case
    h3 = Hfun3([0, 0]) # h2 = [-1, -1];
    h3
    array([-1., -1.])

    See also
    --------
    Gradient,
    Derivative,
    Hessdiag,
    Jacobian
    '''

    def __call__(self, x):
        return np.diag(self._hessian(x))




def _example_taylor():
    def f(x):
        return x*x*x*x  # np.sin(np.cos(x) + np.sin(x))
    D = 5
    P = 1
    x = UTPM(np.zeros((D, P)))
    x.data[0, 0] = 1.0
    x.data[1, 0] = 1

    y = f(x)
    print('coefficients of y =', y.data[:, 0])


def test_docstrings():
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    test_docstrings()
    #_example_taylor()
