'''
Numdifftools.nd_algopy
======================
This module provide an easy to use interface to derivatives calculated with
AlgoPy. Algopy stands for Algorithmic Differentiation in Python.

The purpose of AlgoPy is the evaluation of higher-order derivatives in the
forward and reverse mode of Algorithmic Differentiation (AD) of functions that
are implemented as Python programs. Particular focus are functions that contain
numerical linear algebra functions as they often appear in statistically
motivated functions. The intended use of AlgoPy is for easy prototyping at
reasonable execution speeds. More precisely, for a typical program a
directional derivative takes order 10 times as much time as time as the
function evaluation. This is approximately also true for the gradient.


Algoritmic differentiation
==========================

Algorithmic differentiation (AD) is a set of techniques to numerically
evaluate the derivative of a function specified by a computer program. AD
exploits the fact that every computer program, no matter how complicated,
executes a sequence of elementary arithmetic operations (addition,
subtraction, multiplication, division, etc.) and elementary functions
(exp, log, sin, cos, etc.). By applying the chain rule repeatedly to these
operations, derivatives of arbitrary order can be computed automatically,
accurately to working precision, and using at most a small constant factor
more arithmetic operations than the original program.

Algorithmic differentiation is not:

Symbolic differentiation, nor Numerical differentiation (the method of
finite differences). These classical methods run into problems:
symbolic differentiation leads to inefficient code (unless carefully done)
and faces the difficulty of converting a computer program into a single
expression, while numerical differentiation can introduce round-off errors
in the discretization process and cancellation. Both classical methods have
problems with calculating higher derivatives, where the complexity and
errors increase. Finally, both classical methods are slow at computing the
partial derivatives of a function with respect to many inputs, as is needed
for gradient-based optimization algorithms. Algoritmic differentiation
solves all of these problems.

References
----------
Sebastian F. Walter and Lutz Lehmann 2013,
"Algorithmic differentiation in Python with AlgoPy",
in Journal of Computational Science, vol 4, no 5, pp 334 - 344,
http://www.sciencedirect.com/science/article/pii/S1877750311001013

https://en.wikipedia.org/wiki/Automatic_differentiation

https://pythonhosted.org/algopy/index.html
'''
from __future__ import division
import numpy as np
from scipy import misc
try:
    import algopy
    from algopy import UTPM
except ImportError:
    algopy = None


_cmn_doc = """
    Calculate %(derivative)s with Algorithmic Differentiation method

    Parameters
    ----------
    f : function
       function of one array f(x, `*args`, `**kwds`)%(extra_parameter)s
    method : string, optional {'forward', 'reverse'}
        defines method used in the approximation
    %(returns)s
    Notes
    -----
    Algorithmic differentiation is a set of techniques to numerically
    evaluate the derivative of a function specified by a computer program. AD
    exploits the fact that every computer program, no matter how complicated,
    executes a sequence of elementary arithmetic operations (addition,
    subtraction, multiplication, division, etc.) and elementary functions
    (exp, log, sin, cos, etc.). By applying the chain rule repeatedly to these
    operations, derivatives of arbitrary order can be computed automatically,
    accurately to working precision, and using at most a small constant factor
    more arithmetic operations than the original program.
    %(extra_note)s
    References
    ----------
    Sebastian F. Walter and Lutz Lehmann 2013,
    "Algorithmic differentiation in Python with AlgoPy",
    in Journal of Computational Science, vol 4, no 5, pp 334 - 344,
    http://www.sciencedirect.com/science/article/pii/S1877750311001013

    https://en.wikipedia.org/wiki/Automatic_differentiation

    %(example)s
    %(see_also)s
    """


class _Common(object):
    def __init__(self, f, method='forward'):
        self.f = f
        self.method = method

    def _initialize_reverse(self, x, *args, **kwds):
        # STEP 1: trace the function evaluation
        cg = algopy.CGraph()
        x = algopy.Function(x)

        y = self.f(x, *args, **kwds)
        # y = UTPM.as_utpm(z)
        cg.trace_off()
        cg.independentFunctionList = [x]
        cg.dependentFunctionList = [y]
        self._cg = cg

    def _get_function(self):
        name = '_' + self.method
        return getattr(self, name)

    def __call__(self, x0, *args, **kwds):
        fun = self._get_function()
        return fun(x0, *args, **kwds)


class Derivative(_Common):
    __doc__ = _cmn_doc % dict(
        derivative='n-th derivative',
        extra_parameter="""
    n : int, optional
        Order of the derivative.""",
        extra_note="", returns="""
    Returns
    -------
    der : ndarray
       array of derivatives
    """, example='''
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
    ''', see_also='''
    See also
    --------
    Gradient,
    Hessdiag,
    Hessian,
    Jacobian
    ''')

    def __init__(self, f, n=1, method='forward'):
        self.f = f
        self.n = n
        self.method = method

    def _derivative(self, x):
        xi = np.asarray(x, dtype=float)
        shape0 = xi.shape
        y = np.array([self._gradient(xj) for xj in xi.ravel()])
        return y.reshape(shape0)

    def _forward(self, x, *args, **kwds):
        x0 = np.asarray(x)
        shape = x0.shape
        P = 1
        x = UTPM(np.zeros((self.n + 1, P) + shape))
        x.data[0, 0] = x0
        x.data[1, 0] = 1

        y = UTPM.as_utpm(self.f(x, *args, **kwds))

        return y.data[self.n, 0] * misc.factorial(self.n)


class Jacobian(_Common):
    __doc__ = _cmn_doc % dict(
        derivative='Jacobian',
        extra_parameter="",
        extra_note="", returns="""
    Returns
    -------
    jacob : array
        Jacobian
    """, example='''
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

    >>> Jfun2 = Jacobian(f, method='reverse')
    >>> Jfun2([1,2,0.75]).T # should be numerically zero
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    >>> f2 = lambda x : x[0]*x[1]*x[2] + np.exp(x[0])*x[1]
    >>> Jfun3 = nda.Jacobian(f2)
    >>> Jfun3([3.,5.,7.])
    array([[ 135.42768462,   41.08553692,   15.        ]])

    >>> Jfun4 = nda.Jacobian(f2, method='reverse')
    >>> Jfun4([3,5,7])
    array([[ 135.42768462,   41.08553692,   15.        ]])
    ''', see_also='''
    See also
    --------
    Derivative
    Gradient,
    Hessdiag,
    Hessian,
    ''')

    def _jacobian_forward(self, x, *args, **kwds):
        x = np.asarray(x, dtype=float)
        # shape = x.shape
        D, Nm = 2, x.size
        P = Nm
        y = UTPM(np.zeros((D, P, Nm)))

        y.data[0, :] = x.ravel()
        y.data[1, :] = np.eye(Nm)
        z0 = self.f(y, *args, **kwds)
        z = UTPM.as_utpm(z0)
        J = z.data[1, :, :, 0]
        return J

    def _forward(self, x, *args, **kwds):
        # forward mode without building the computational graph
        x0 = np.asarray(x, dtype=float)
        tmp = algopy.UTPM.init_jacobian(x0)
        y = self.f(tmp, *args, **kwds)
        return np.atleast_2d(algopy.UTPM.extract_jacobian(y))

    def _reverse(self, x, *args, **kwds):
        x = np.asarray(x, dtype=float)
        self._initialize_reverse(x, *args, **kwds)
        return self._cg.jacobian(x)


class Gradient(_Common):
    _doc__ = _cmn_doc % dict(
        derivative='Gradient',
        extra_parameter="",
        extra_note="", returns="""
    Returns
    -------
    grad : array
        gradient
    """, example='''
    Examples
    --------
    >>> import numdifftools.nd_algopy as nda
    >>> f = lambda x: np.sum(x**2)
    >>> df = nda.Gradient(f, method='reverse')
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
    ''', see_also='''
    See also
    --------
    Derivative
    Jacobian,
    Hessdiag,
    Hessian,
    ''')

    def _reverse(self, x, *args, **kwds):
        x = np.asarray(x, dtype=float)
        self._initialize_reverse(x, *args, **kwds)
        return self._cg.gradient(x)

    def _forward(self, x, *args, **kwds):
        # forward mode without building the computational graph
        x0 = np.asarray(x, dtype=float)
        tmp = algopy.UTPM.init_jacobian(x0)
        y = self.f(tmp, *args, **kwds)
        return algopy.UTPM.extract_jacobian(y)


class Hessian(_Common):
    __doc__ = _cmn_doc % dict(
        derivative='Hessian',
        extra_parameter="",
        returns="""
    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian
    """, extra_note='', example='''
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

    >>> Hfun3 = Hessian(f, method='reverse')
    >>> h3 = Hfun3([0, 0]) # h2 = [-1, 1; 1, -1];
    >>> h3
    array([[-1.,  1.],
           [ 1., -1.]])
    ''', see_also='''
    See also
    --------
    Derivative
    Gradient,
    Jacobian,
    Hessdiag,
    ''')

    def _forward(self, x, *args, **kwds):
        x0 = np.asarray(x, dtype=float)
        tmp = algopy.UTPM.init_hessian(x0)
        y = self.f(tmp, *args, **kwds)
        return algopy.UTPM.extract_hessian(len(x0), y)

    def _reverse(self, x, *args, **kwds):
        x = np.asarray(x, dtype=float)
        self._initialize_reverse(x, *args, **kwds)
        return self._cg.hessian(x)


class Hessdiag(Hessian):
    __doc__ = _cmn_doc % dict(
        derivative='Hessian diagonal',
        extra_parameter="",
        returns="""
    Returns
    -------
    hessdiag : ndarray
       Hessian diagonal array of partial second order derivatives.
    """, extra_note='', example='''
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

    >>> Hfun3 = Hessdiag(f, method='reverse')
    >>> h3 = Hfun3([0, 0]) # h2 = [-1, -1];
    >>> h3
    array([-1., -1.])

    ''', see_also='''
    See also
    --------
    Derivative
    Gradient,
    Jacobian,
    Hessian,
    ''')

    def __call__(self, x, *args, **kwds):
        return np.diag(super(Hessdiag, self).__call__(x, *args, **kwds))


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
    # _example_taylor()
