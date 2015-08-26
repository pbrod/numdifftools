# -*- coding:utf-8 -*-
""""""

from __future__ import division
import unittest
import numdifftools.nd_algopy as nd
import numpy as np
from numpy import pi, r_, sqrt, array
from numpy.testing import assert_array_almost_equal
from scipy import linalg, optimize, constants

_TINY = np.finfo(float).machar.tiny


#  Hamiltonian
#     H = sum_i(p_i2/(2m)+ 1/2 * m * w2 x_i2) + sum_(i!=j)(a/|x_i-x_j|)
class classicalHamiltonian:
    '''
    Parameters
    ----------
    N : scalar
        number of ions in the chain
    w : scalar
        angular trap frequency
    C : scalar
        Coulomb constant times the electronic charge in SI units.
    m : scalar
        the mass of a single trapped ion in the chain
    '''

    def __init__(self):
        self.N = 2
        f = 1000000      # f is a scalar, it's the trap frequency
        self.w = 2 * pi * f
        self.C = (4 * pi * constants.epsilon_0) ** (-1) * constants.e ** 2
        # C is a scalar, it's the I
        self.m = 39.96 * 1.66e-27

    def potential(self, positionvector):
        '''
        positionvector is an 1-d array (vector) of length N that contains the
        positions of the N ions
        '''
        x = positionvector
        w = self.w
        C = self.C
        m = self.m

        # First we consider the potential of the harmonic oscillator
        Vx = 0.5 * m * (w ** 2) * sum(x ** 2)
        # then we add the coulomb interaction:
        for i, xi in enumerate(x):
            for xj in x[i + 1:]:
                Vx += C / (abs(xi - xj))
        return Vx

    def initialposition(self):
        """Defines initial position as an estimate for the minimize process."""
        N = self.N
        x_0 = r_[-(N - 1) / 2:(N - 1) / 2:N * 1j]
        return x_0

    def normal_modes(self, eigenvalues):
        '''the computed eigenvalues of the matrix Vx are of the form
        (normal_modes)2*m.'''
        m = self.m
        normal_modes = sqrt(eigenvalues / m)
        return normal_modes


def _run_hamiltonian(verbose=True):
    c = classicalHamiltonian()
    if verbose:
        print(c.potential(array([-0.5, 0.5])))
        print(c.potential(array([-0.5, 0.0])))
        print(c.potential(array([0.0, 0.0])))

    xopt = optimize.fmin(c.potential, c.initialposition(), xtol=1e-10)

    hessian = nd.Hessian(c.potential)

    H = hessian(xopt)
    true_H = np.array([[5.23748385e-12, -2.61873829e-12],
                       [-2.61873829e-12, 5.23748385e-12]])
    error_estimate = np.NAN
    if verbose:
        print(xopt)
        print('H', H)
        print('H-true_H', np.abs(H - true_H))
        # print('error_estimate', info.error_estimate)

        eigenvalues = linalg.eigvals(H)
        normal_modes = c.normal_modes(eigenvalues)

        print('eigenvalues', eigenvalues)
        print('normal_modes', normal_modes)
    return H, error_estimate, true_H


class TestHessian(unittest.TestCase):

    def test_run_hamiltonian(self):
        H, _error_estimate, true_H = _run_hamiltonian(verbose=False)
        self.assertTrue((np.abs(H - true_H) < 1e-18).all())

    def test_hessian_cosIx_yI_at_I0_0I(self):
        # cos(x-y), at (0,0)

        def fun(xy):
            return np.cos(xy[0] - xy[1])
        htrue = [[-1., 1.], [1., -1.]]
        methods = ['forward', ]  # 'reverse']

        for method in methods:
            Hfun2 = nd.Hessian(fun, method=method)
            h2 = Hfun2([0, 0])
            # print(method, (h2-np.array(htrue)))
            assert_array_almost_equal(h2, htrue)


class TestDerivative(unittest.TestCase):

    # TODO: Derivative does not tackle non-finite values.
    #     def test_infinite_functions(self):
    #         def finf(x):
    #             return np.inf * np.ones_like(x)
    #         df = nd.Derivative(finf, method='forward')
    #         val = df(0)
    #         self.assert_(np.isnan(val))

    def test_high_order_derivative_cos(self):
        true_vals = (-1.0, 0.0, 1.0, 0.0) * 5

        x = np.pi / 2  # np.linspace(0, np.pi/2, 15)
        for method in ['forward', 'reverse']:
            nmax = 15 if method in ['forward'] else 2
            for n in range(1, nmax):
                d3cos = nd.Derivative(np.cos, n=n, method=method)
                y = d3cos(x)
                assert_array_almost_equal(y, true_vals[n - 1])

    def test_derivative_cube(self):
        '''Test for Issue 7'''
        def cube(x):
            return x * x * x

        shape = (3, 2)
        x = np.ones(shape) * 2
        for method in ['forward', 'reverse']:
            dcube = nd.Derivative(cube, method=method)
            dx = dcube(x)
            assert_array_almost_equal(list(dx.shape), list(shape),
                                      decimal=13,
                                      err_msg='Shape mismatch')
            txt = 'First differing element %d\n value = %g,\n true value = %g'
            for i, (val, tval) in enumerate(zip(dx.ravel(),
                                                (3 * x**2).ravel())):
                assert_array_almost_equal(val, tval, decimal=8,
                                          err_msg=txt % (i, val, tval))

    def test_derivative_exp(self):
        # derivative of exp(x), at x == 0
        for method in ['forward', 'reverse']:
            dexp = nd.Derivative(np.exp, method=method)
            assert_array_almost_equal(dexp(0), np.exp(0), decimal=8)

    def test_derivative_sin(self):
        # Evaluate the indicated (default = first)
        # derivative at multiple points
        for method in ['forward', 'reverse']:
            dsin = nd.Derivative(np.sin, method=method)
            x = np.linspace(0, 2. * np.pi, 13)
            y = dsin(x)
            np.testing.assert_almost_equal(y, np.cos(x), decimal=8)

    def test_derivative_on_sinh(self):
        for method in ['forward', ]:  # 'reverse']: # TODO: reverse fails
            dsinh = nd.Derivative(np.sinh, method=method)
            self.assertAlmostEqual(dsinh(0.0), np.cosh(0.0))

    def test_derivative_on_log(self):

        x = np.r_[0.01, 0.1]
        for method in ['forward', 'reverse']:
            dlog = nd.Derivative(np.log, method=method)

            assert_array_almost_equal(dlog(x), 1.0 / x)


class TestJacobian(unittest.TestCase):

    def test_on_scalar_function(self):
        def f2(x):
            return x[0] * x[1] * x[2] + np.exp(x[0]) * x[1]
        for method in ['forward', 'reverse']:
            Jfun3 = nd.Jacobian(f2, method=method)
            x = Jfun3([3., 5., 7.])
            assert_array_almost_equal(x, [[135.42768462, 41.08553692, 15.]])

    def test_on_vector_valued_function(self):
        xdata = np.reshape(np.arange(0, 1, 0.1), (-1, 1))
        ydata = 1 + 2 * np.exp(0.75 * xdata)

        def fun(c):
            return (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2

        for method in ['reverse']:  # TODO: 'forward' fails

            Jfun = nd.Jacobian(fun, method=method)
            J = Jfun([1, 2, 0.75])  # should be numerically zero
            assert_array_almost_equal(J, np.zeros(J.shape))


class TestGradient(unittest.TestCase):

    def test_on_scalar_function(self):
        def fun(x):
            return np.sum(x ** 2)

        dtrue = [2., 4., 6.]

        for method in ['forward', 'reverse']:  #

            dfun = nd.Gradient(fun, method=method)
            d = dfun([1, 2, 3])
            assert_array_almost_equal(d, dtrue)


class TestHessdiag(unittest.TestCase):

    def test_forward(self):
        def fun(x):
            return x[0] + x[1] ** 2 + x[2] ** 3
        htrue = np.array([0., 2., 18.])
        Hfun = nd.Hessdiag(fun)
        hd = Hfun([1, 2, 3])
        _error = hd - htrue
        assert_array_almost_equal(hd, htrue)

    def test_reverse(self):
        def fun(x):
            return x[0] + x[1] ** 2 + x[2] ** 3
        htrue = np.array([0., 2., 18.])
        Hfun = nd.Hessdiag(fun, method='reverse')
        hd = Hfun([1, 2, 3])
        _error = hd - htrue
        assert_array_almost_equal(hd, htrue)

if __name__ == '__main__':
    # _run_hamiltonian()
    unittest.main()
