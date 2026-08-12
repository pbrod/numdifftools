"""
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
"""

# mypy: disable-error-code=no-redef
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import special

from numdifftools._typing import Array, ArrayOrScalar, EstimateResult, FunctionLike

algopy: Any
UTPM: Any
try:
    import algopy
    from algopy import UTPM
except ImportError:
    UTPM = algopy = None


EPS = np.finfo(float).eps

_cmn_doc = """
    Calculate %(derivative)s with Algorithmic Differentiation method

    Parameters
    ----------
    fun: function
        function of one array fun(x, `*args`, `**kwds`)%(extra_parameter)s
    method: string, optional {'forward', 'reverse'}
        defines method used in the approximation

    Methods
    -------
    __call__: callable with the following parameters:
        x: array_like
           value at which function derivative is evaluated
        args: tuple
            Arguments for function `fun`.
        kwds: dict
            Keyword arguments for function `fun`.
    %(returns)s
    Notes
    -----
    Algorithmic differentiation is a set of techniques to numerically
    evaluate the derivative of a function specified by a computer program.
    AD exploits the fact that every computer program, no matter how
    complicated, executes a sequence of elementary arithmetic operations
    (addition, subtraction, multiplication, division, etc.) and elementary
    functions (exp, log, sin, cos, etc.). By applying the chain rule
    repeatedly to these operations, derivatives of arbitrary order can be
    computed automatically, accurately to working precision, and using at
    most a small constant factor more arithmetic operations than the original
    program.
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


class _Derivative:
    """Base class"""

    _fun: FunctionLike | None
    _computational_graph: Any | None

    def __init__(
        self,
        fun: FunctionLike | None = None,
        n: int = 1,
        method: str = "forward",
        full_output: bool = False,
    ) -> None:

        self.fun = fun
        self.method = method
        self.n = n
        self.full_output = full_output

    @property
    def fun(self) -> FunctionLike | None:
        return self._fun

    @fun.setter
    def fun(self, fun: FunctionLike | None) -> None:
        self._fun = fun
        self._computational_graph = None

    def computational_graph(
        self,
        x: ArrayLike,
        *args: Any,
        **kwds: Any,
    ) -> Any:
        if self._computational_graph is None:
            # STEP 1: trace the function evaluation
            cg: Any = algopy.CGraph()
            tmp: Any = algopy.Function(x)
            assert self.fun is not None
            y = self.fun(tmp, *args, **kwds)

            cg.trace_off()
            cg.independentFunctionList = [tmp]
            cg.dependentFunctionList = [y]
            self._computational_graph = cg
        return self._computational_graph

    def _get_function(self) -> FunctionLike | None:
        if self.n == 0:
            return self.fun
        name = "_" + {"backward": "reverse"}.get(self.method, self.method)
        return getattr(self, name)

    def __call__(
        self,
        x: ArrayLike,
        *args: Any,
        **kwds: Any,
    ) -> ArrayOrScalar | EstimateResult:
        fun = self._get_function()
        assert fun is not None
        x0 = np.asarray(x, dtype=float)
        df = np.asarray(fun(x0, *args, **kwds))
        if self.full_output:
            error = np.maximum(10 * EPS * np.abs(df), EPS)
            return EstimateResult(df, error_estimate=error, final_step=EPS, best_index=0)
        return df


class Derivative(_Derivative):
    __doc__ = _cmn_doc % {
        "derivative": "n-th derivative",
        "extra_parameter": """
    n: int, optional
        Order of the derivative.""",
        "extra_note": "",
        "returns": """
    Returns
    -------
    der: ndarray
       array of derivatives
    """,
        "example": """
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

    # 1'st derivative of x^3+x^4, at x = [0,1]

    >>> fun = lambda x: x**3 + x**4
    >>> fd3 = nda.Derivative(fun)
    >>> np.allclose(fd3([0,1]), [ 0.,  7.])
    True
    """,
        "see_also": """
    See also
    --------
    Gradient,
    Hessdiag,
    Hessian,
    Jacobian
    """,
    }

    def _forward(
        self,
        x: ArrayLike,
        *args: Any,
        **kwds: Any,
    ) -> Array:
        assert self.fun is not None
        x0 = np.array(x)
        shape = x0.shape
        P = 1
        x_a = UTPM(np.zeros((self.n + 1, P) + shape))
        x_a.data[0, 0] = x0
        x_a.data[1, 0] = 1
        z = self.fun(x_a, *args, **kwds)
        y = UTPM.as_utpm(z)

        return np.asarray(y.data[self.n, 0] * special.factorial(self.n))

    def _reverse(
        self,
        x: Array,
        *args: Any,
        **kwds: Any,
    ) -> Array:
        if self.n != 1:
            raise NotImplementedError("Derivative reverse not implemented for n>1")

        c_graph = self.computational_graph(np.asarray(1), *args, **kwds)
        shape0 = x.shape
        y = np.asarray([c_graph.gradient(xi) for xi in x.ravel()])
        return y.reshape(shape0)


class Gradient(_Derivative):
    __doc__ = _cmn_doc % {
        "derivative": "Gradient",
        "extra_parameter": "",
        "extra_note": "",
        "returns": """
    Returns
    -------
    grad: array
        gradient
    """,
        "example": """
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_algopy as nda
    >>> fun = lambda x: np.sum(x**2)
    >>> df = nda.Gradient(fun, method='reverse')
    >>> np.allclose(df([1,2,3]), [ 2.,  4.,  6.])
    True

    #At [x,y] = [1,1], compute the numerical gradient
    #of the function sin(x-y) + y*exp(x)

    >>> sin = np.sin; exp = np.exp
    >>> z = lambda xy: sin(xy[0]-xy[1]) + xy[1]*exp(xy[0])
    >>> dz = nda.Gradient(z)
    >>> grad2 = dz([1, 1])
    >>> np.allclose(grad2, [ 3.71828183,  1.71828183])
    True

    #At the global minimizer (1,1) of the Rosenbrock function,
    #compute the gradient. It should be essentially zero.

    >>> rosen = lambda x : (1-x[0])**2 + 105.*(x[1]-x[0]**2)**2
    >>> rd = nda.Gradient(rosen)
    >>> grad3 = rd([1,1])
    >>> np.allclose(grad3, [ 0.,  0.])
    True

    """,
        "see_also": """
    See also
    --------
    Derivative
    Jacobian,
    Hessdiag,
    Hessian,
    """,
    }

    def _forward(
        self,
        x: ArrayLike,
        *args: Any,
        **kwds: Any,
    ) -> Array:
        # forward mode without building the computational graph

        assert self.fun is not None
        tmp = UTPM.init_jacobian(x)
        y = self.fun(tmp, *args, **kwds)
        return np.asarray(UTPM.extract_jacobian(y))

    def _reverse(
        self,
        x: Array,
        *args: Any,
        **kwds: Any,
    ) -> Array:
        c_graph = self.computational_graph(x, *args, **kwds)
        return np.asarray(c_graph.gradient(x))


class Jacobian(Gradient):
    __doc__ = _cmn_doc % {
        "derivative": "Jacobian",
        "extra_parameter": "",
        "extra_note": "",
        "returns": """
    Returns
    -------
    jacob: array
        Jacobian
    """,
        "example": """
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_algopy as nda

    #(nonlinear least squares)

    >>> xdata = np.arange(0,1,0.1)
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2

    Jfun = nda.Jacobian(fun) # Todo: This does not work
    Jfun([1,2,0.75]).T # should be numerically zero
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    >>> Jfun2 = nda.Jacobian(fun, method='reverse')
    >>> res = Jfun2([1,2,0.75]).T # should be numerically zero
    >>> np.allclose(res,
    ...                [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    ...                 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    ...                 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    True

    >>> f2 = lambda x : x[0]*x[1]*x[2]**2
    >>> Jfun2 = nda.Jacobian(f2)
    >>> np.allclose(Jfun2([1., 2., 3.]), [[ 18., 9., 12.]])
    True

    >>> Jfun21 = nda.Jacobian(f2, method='reverse')
    >>> np.allclose(Jfun21([1., 2., 3.]), [[ 18., 9., 12.]])
    True

    >>> def fun3(x):
    ...     n = int(np.prod(np.shape(x[0])))
    ...     out = nda.algopy.zeros((2, n), dtype=x)
    ...     out[0] = x[0]*x[1]*x[2]**2
    ...     out[1] = x[0]*x[1]*x[2]
    ...     return out
    >>> Jfun3 = nda.Jacobian(fun3)

    >>> np.allclose(Jfun3([1., 2., 3.]), [[[18., 9., 12.]], [[6., 3., 2.]]])
    True
    >>> np.allclose(Jfun3([4., 5., 6.]), [[[180., 144., 240.]],
    ...                                   [[30., 24., 20.]]])
    True
    >>> np.allclose(Jfun3(np.array([[1.,2.,3.], [4., 5., 6.]]).T),
    ...             [[[18.,    0.,    9.,    0.,   12.,    0.],
    ...               [0.,  180.,    0.,  144.,    0.,  240.]],
    ...              [[6.,    0.,    3.,    0.,    2.,    0.],
    ...               [0.,   30.,    0.,   24.,    0.,   20.]]])
    True
    """,
        "see_also": """
    See also
    --------
    Derivative
    Gradient,
    Hessdiag,
    Hessian,
    """,
    }

    def _forward(self, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        return np.atleast_2d(super()._forward(x, *args, **kwds))

    def _reverse(self, x: Array, *args: Any, **kwds: Any) -> Array:
        x = np.atleast_1d(x)
        c_graph = self.computational_graph(x, *args, **kwds)
        return np.asarray(c_graph.jacobian(x))


class Hessian(_Derivative):
    __doc__ = _cmn_doc % {
        "derivative": "Hessian",
        "extra_parameter": "",
        "returns": """
    Returns
    -------
    hess : ndarray
       array of partial second derivatives, Hessian
    """,
        "extra_note": "",
        "example": """
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_algopy as nda

    # Rosenbrock function, minimized at [1,1]

    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hf = nda.Hessian(rosen)
    >>> h = Hf([1, 1]) #  h =[ 842 -420; -420, 210];
    >>> np.allclose(h, [[ 842., -420.],
    ...                 [-420.,  210.]])
    True

    # cos(x-y), at (0,0)

    >>> cos = np.cos
    >>> fun = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = nda.Hessian(fun)
    >>> h2 = Hfun2([0, 0]) # h2 = [-1 1; 1 -1]
    >>> np.allclose(h2, [[-1.,  1.],
    ...                  [ 1., -1.]])
    True

    >>> Hfun3 = nda.Hessian(fun, method='reverse')
    >>> h3 = Hfun3([0, 0]) # h2 = [-1, 1; 1, -1];
    >>> np.allclose(h3, [[-1.,  1.],
    ...                  [ 1., -1.]])
    True
    """,
        "see_also": """
    See also
    --------
    Derivative
    Gradient,
    Jacobian,
    Hessdiag,
    """,
    }

    def __init__(
        self,
        f: FunctionLike | None = None,
        method: str = "forward",
        full_output: bool = False,
    ) -> None:
        super().__init__(f, n=2, method=method, full_output=full_output)

    def _forward(self, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        assert self.fun is not None
        x = np.atleast_1d(x)
        tmp = UTPM.init_hessian(x)
        y = self.fun(tmp, *args, **kwds)
        return np.asarray(UTPM.extract_hessian(len(x), y))

    def _reverse(self, x: Array, *args: Any, **kwds: Any) -> Array:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        c_graph = self.computational_graph(x, *args, **kwds)
        return np.array(c_graph.hessian(x))


class Hessdiag(Hessian):
    __doc__ = _cmn_doc % {
        "derivative": "Hessian diagonal",
        "extra_parameter": "",
        "returns": """
    Returns
    -------
    hessdiag : ndarray
       Hessian diagonal array of partial second order derivatives.
    """,
        "extra_note": "",
        "example": """
    Examples
    --------
    >>> import numpy as np
    >>> import numdifftools.nd_algopy as nda

    # Rosenbrock function, minimized at [1,1]

    >>> rosen = lambda x : (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> Hfun = nda.Hessdiag(rosen)
    >>> h = Hfun([1, 1]) #  h =[ 842, 210]
    >>> np.allclose(h, [ 842.,  210.])
    True

    # cos(x-y), at (0,0)

    >>> cos = np.cos
    >>> fun = lambda xy : cos(xy[0]-xy[1])
    >>> Hfun2 = nda.Hessdiag(fun)
    >>> h2 = Hfun2([0, 0]) # h2 = [-1, -1]
    >>> np.allclose(h2, [-1., -1.])
    True

    >>> Hfun3 = nda.Hessdiag(fun, method='reverse')
    >>> h3 = Hfun3([0, 0]) # h2 = [-1, -1];
    >>> np.allclose(h3, [-1., -1.])
    True

    """,
        "see_also": """
    See also
    --------
    Derivative
    Gradient,
    Jacobian,
    Hessian,
    """,
    }

    def _forward(self, x: ArrayLike, *args: Any, **kwds: Any) -> Array:
        x_arr = np.asarray(x)
        d, n = 2 + 1, x_arr.size
        p = n
        y = UTPM(np.zeros((d, p, n)))

        y.data[0, :] = x_arr.ravel()
        y.data[1, :] = np.eye(n)
        assert self.fun is not None
        z0 = self.fun(y, *args, **kwds)
        z = UTPM.as_utpm(z0)
        H = z.data[2, ...] * 2
        return np.asarray(H)

    def _reverse(self, x: Array, *args: Any, **kwds: Any) -> Array:
        return np.diag(super()._reverse(x, *args, **kwds))


def directionaldiff(
    f: FunctionLike,
    x0: ArrayLike,
    vec: ArrayLike,
    **options: Any,
) -> ArrayOrScalar | EstimateResult:
    """
    Return directional derivative of a function of n variables

    Parameters
    ----------
    fun: callable
        analytical function to differentiate.
    x0: array
        vector location at which to differentiate fun. If x0 is an nxm array,
        then fun is assumed to be a function of n*m variables.
    vec: array
        vector defining the line along which to take the derivative. It should
        be the same size as x0, but need not be a vector of unit length.
    **options:
        optional arguments to pass on to Derivative.

    Returns
    -------
    dder:  scalar
        estimate of the first derivative of fun in the specified direction.

    Examples
    --------
    At the global minimizer (1,1) of the Rosenbrock function,
    compute the directional derivative in the direction [1 2]

    >>> import numpy as np
    >>> import numdifftools.nd_algopy as nda
    >>> vec = np.r_[1, 2]
    >>> rosen = lambda x: (1-x[0])**2 + 105*(x[1]-x[0]**2)**2
    >>> dd = nda.directionaldiff(rosen, [1, 1], vec)
    >>> np.allclose(dd, 0)
    True

    See also
    --------
    Derivative,
    Gradient
    """
    x_arr = np.asarray(x0)
    vec_arr = np.asarray(vec)
    if x_arr.size != vec_arr.size:
        raise ValueError("vec and x0 must be the same shapes")

    vec_arr = np.reshape(vec_arr / np.linalg.norm(vec_arr.ravel()), x_arr.shape)
    dfun = Derivative(lambda t: f(x_arr + t * vec_arr), **options)
    return dfun(0)


class Taylor:
    """
    Return Taylor coefficients of function using algorithmic differentiation

    Parameters
    ----------
    fun: callable
        function to differentiate
    z0: real or complex array
        at which to evaluate the derivatives
    n: scalar integer, default 1
        Number of taylor coefficents to compute.

    Returns
    -------
    coefs: ndarray
       array of taylor coefficents

    Examples
    --------
    Compute the first 6 taylor coefficients 1 + 2*z + 3*z**2 expanded round  z0 = 0:

    >>> import numpy as np
    >>> import numdifftools.nd_algopy as nda
    >>> c = nda.Taylor(lambda x: 1+2*x+3*x**2, n=6)(z0=0)
    >>> np.allclose(c, [1, 2, 3, 0, 0, 0])
    True

    """

    def __init__(
        self,
        fun: FunctionLike | None = None,
        n: int = 1,
    ) -> None:
        self.fun = fun
        self.n = n

    def __call__(
        self,
        z0: ArrayLike | float | complex = 0,
    ) -> Array:
        z = np.atleast_1d(z0).ravel()
        x = UTPM(np.zeros((self.n, 1, z.size), dtype=z.dtype))

        x.data[0, 0, :] = z
        x.data[1, 0, :] = 1

        assert self.fun is not None
        y = self.fun(x)
        coefs = np.squeeze(y.data)
        return coefs


if __name__ == "__main__":
    from numdifftools.testing import test_docstrings

    test_docstrings()
