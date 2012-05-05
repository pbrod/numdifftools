'''
Easy to use interface to derivatives in ScientificPython
http://dirac.cnrs-orleans.fr/ScientificPython/ScientificPythonManual/

'''
import numpy as np

try:
    import  Scientific.Functions.Derivatives as SFD
except ImportError:
    SFD = None
try:
    import  Scientific.Functions.FirstDerivatives as SFF
except ImportError:
    SFF = None
class _Common(object):
    def __init__(self, fun, derOrder=1):
        self.fun = fun
        self.derOrder = derOrder
    def _derivative(self, x0):
        x = np.atleast_1d(x0)
        DerivVar = SFD.DerivVar if self.derOrder > 1 else SFF.DerivVar
        der = self.fun(DerivVar(x, 0, self.derOrder))[self.derOrder][0]
        for _i in xrange(self.derOrder - 1):
            der = der[0]
        return der
    def _jacobian(self, x):
        return np.squeeze(self._gradient(x)).T
    
    def _gradient(self, x):
        N = len(x)
        sx = np.array([SFF.DerivVar(x[i], i, 1) for i in range(N)])
        return np.array(self.fun(sx)[1])

    def _hessian(self, x):
        N = len(x)
        sx = np.array([SFD.DerivVar(x[i], i, 2) for i in range(N)])
        return np.array(self.fun(sx)[2])

class Derivative(_Common):
    '''Estimate n'th derivative of fun at x0
    
    Examples
    --------
     # 1'st and 2'nd derivative of exp(x), at x == 1
     
     >>> import numpy as np
     >>> import numdifftools.nd_scientific as nds
     >>> fd = nds.Derivative(np.exp)              # 1'st derivative
     >>> fdd = nds.Derivative(np.exp, derOrder=2)  # 2'nd derivative
     >>> fd(1)
     array([ 2.71828183])

     >>> d2 = fdd([1,2])
     >>> d2
     array([ 2.71828183,  7.3890561 ])

     # 3'rd derivative of x.^3+x.^4, at x = [0,1]
     >>> fun = lambda x: x**3 + x**4
     >>> fd3 = nds.Derivative(fun,derOrder=3)
     >>> fd3([0,1])          #  True derivatives: [6,30]
     array([ 6, 30])
 

     See also
     --------
     Gradient,
     Hessdiag,
     Hessian,
     Jacobian
    '''
    def derivative(self, x0):
        return self._derivative(x0)
    def __call__(self, x0):
        return self._derivative(x0)
    
class Jacobian(_Common):
    '''Estimate Jacobian matrix
    
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
    >>> import numdifftools.nd_scientific as nds
    
    #(nonlinear least squares)
    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> Jfun = nds.Jacobian(fun)
    >>> Jfun([1,2,0.75]) # should be numerically zero
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])

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
        return self._jacobian(x0)
class Gradient(_Common):
    '''Estimate gradient of fun at x0

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
    >>> import numdifftools.nd_scientific as nds
    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = nds.Gradient(fun)
    >>> dfun([1,2,3])
    array([2, 4, 6])

    #At [x,y] = [1,1], compute the numerical gradient
    #of the function sin(x-y) + y*exp(x)

    >>> sin = np.sin; exp = np.exp
    >>> z = lambda xy: sin(xy[0]-xy[1]) + xy[1]*exp(xy[0])
    >>> dz = nds.Gradient(z)
    >>> grad2 = dz([1, 1])
    >>> grad2
    array([ 3.71828183,  1.71828183])
     

    #At the global minimizer (1,1) of the Rosenbrock function,
    #compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> rd = nds.Gradient(rosen)
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
    fun : SCALAR analytical function
        to differentiate. fun must be a function of the vector or array x0,
        but it needs not to be vectorized.

    x0 : vector location
        at which to differentiate fun
        If x0 is an N x M array, then fun is assumed to be a function
        of N*M variables.

    Examples
    --------
    >>> import numdifftools.nd_scientific as nds
    
    #Rosenbrock function, minimized at [1,1]
    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hfun = nds.Hessian(rosen)
    >>> h = Hfun([1, 1]) #  h =[ 842 -420; -420, 210];
    >>> h
    array([[ 842., -420.],
           [-420.,  210.]])
     
    #cos(x-y), at (0,0)
    >>> cos = np.cos
    >>> fun = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = nds.Hessian(fun)
    >>> h2 = Hfun2([0, 0]) # h2 = [-1 1; 1 -1];
    >>> h2
    array([[-1.,  1.],
           [ 1., -1.]])
    
   
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
        derivative, gradient, hessdiag, jacobian
        '''
        return self._hessian(x0)
     
    def __call__(self, x):
        return self._hessian(x) 
    
 
if __name__ == '__main__':
    import doctest
    doctest.testmod()
     



