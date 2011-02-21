from __future__ import division
import numpy as np
try:
    import algopy
except ImportError:
    algopy = None

class _Common(object):
    def __init__(self, fun, x=0, method='forward'):
        self.fun = fun
        self.method = method
        self.initialize(x)
        
    def initialize(self, x):
        if self.method.startswith('reverse'):
            # reverse mode using a computational graph
            x = np.asarray(x, dtype=float)
            self.x = x.copy()
            # STEP 1: trace the function evaluation
            cg = algopy.CGraph()
            x = algopy.Function(x)
            y = self.fun(x)
            cg.trace_off()
            cg.independentFunctionList = [x]
            cg.dependentFunctionList = [y]
            self._cg = cg
            self._gradient = self._gradient_reverse
            self._hessian = self._hessian_reverse 
        else: # forward mode without building the computational graph
            self._gradient = self._gradient_forward
            self._hessian = self._hessian_forward
            
    def _gradient_reverse(self, x):
        return self._cg.gradient([x])
    def _hessian_reverse(self, x):
        return self._cg.hessian([np.asarray(x)])
    def _gradient_forward(self, x):
        tmp = algopy.UTPM.init_jacobian(np.asarray(x, dtype=float))
        return algopy.UTPM.extract_jacobian(self.fun(tmp))
    def _hessian_forward(self, x):
        tmp = algopy.UTPM.init_hessian(np.asarray(x, dtype=float))
        tmp2 = self.fun(tmp)
        return algopy.UTPM.extract_hessian(len(x), tmp2)
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
     

    #At the global minimizer (1,1) of the Rosenbrock function,
    #compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> rd = Gradient(rosen)
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

    #Rosenbrock function, minimized at [1,1]
    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hfun = Hessian(rosen)
    >>> h = Hfun([1, 1]) #  h =[ 842 -420; -420, 210];
    >>> h
    array([[ 842., -420.],
           [-420.,  210.]])
     
    #cos(x-y), at (0,0)
    >>> cos = np.cos
    >>> fun = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = Hessian(fun)
    >>> h2 = Hfun2([0, 0]) # h2 = [-1 1; 1 -1] # TODO: Hfun2 fails in this case
    >>> h2
    array([[-1.,  1.],
           [ 1., -1.]])
    
    >>> Hfun3 = Hessian(fun,x=[0,0], method='reverse')
    >>> h3 = Hfun3([0, 0]) # h2 = [-1, 1; 1, -1];
    >>> h3
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
        derivative, gradient, hessdiag, jacobian
        '''
        return self._hessian(x0)
     
    def __call__(self, x):
        return self._hessian(x) 
    
 
if __name__ == '__main__':
    import doctest
    doctest.testmod()
     
