""" Test functions for numdifftools module

"""
import unittest
import numdifftools as nd
import numpy as np
from numpy.testing import assert_array_almost_equal


class TestDerivative(unittest.TestCase):
    def test_derivative_cube(self):
        '''Test for Issue 7'''
        cube = lambda x: x * x * x
        dcube = nd.Derivative(cube)
        shape = (3, 2)
        x = np.ones(shape)*2
        dx = dcube(x)
        assert_array_almost_equal(list(dx.shape), list(shape),
                                  decimal=12,
                                  err_msg='Shape mismatch')
        txt = 'First differing element %d\n value = %g,\n true value = %g'
        for i, (val, tval) in enumerate(zip(dx.ravel(), (3*x**2).ravel())):
            assert_array_almost_equal(val, tval, decimal=12,
                                      err_msg=txt % (i, val, tval))

    def test_derivative_exp(self):
        # derivative of exp(x), at x == 0
        dexp = nd.Derivative(np.exp)
        assert_array_almost_equal(dexp(0), np.exp(0), decimal=8)
        dexp.n = 2
        t = dexp(0)
        assert_array_almost_equal(t, np.exp(0))

    def test_derivative_sin(self):
        # Evaluate the indicated (default = first)
        # derivative at multiple points
        dsin = nd.Derivative(np.sin)
        x = np.linspace(0, 2. * np.pi, 13)
        y = dsin(x)
        small = np.abs(y - np.cos(x)) <= dsin.error_estimate * 100
        self.assertTrue(np.all(small))

    def test_second_and_fourth_derivative_of_sin(self):
        # Higher order derivatives (second derivative)
        # Truth: 0
        d2sin = nd.Derivative(np.sin, n=2, step_max=0.5)
        assert_array_almost_equal(d2sin(np.pi), 0.0, decimal=8)

        # Higher order derivatives (up to the fourth derivative)
        # Truth: sqrt(2)/2 = 0.707106781186548
        d2sin.n = 4
        y = d2sin(np.pi / 4)
        small = np.abs(y - np.sqrt(2.) / 2.) < d2sin.error_estimate
        self.assertTrue(small)

    def test_high_order_derivative_cos(self):
        # Higher order derivatives (third derivative)
        # Truth: 1
        for n, true_val in zip([1, 2, 3, 4], (-1.0, 0.0, 1.0, 0.0)):
            for order in [1, 2, 3, 4]:
                d3cos = nd.Derivative(np.cos, n=n, order=order, method='forward')
                y = d3cos(np.pi / 2.0)
                small = np.abs(y - true_val) < 10 * d3cos.error_estimate
                self.assertTrue(small)

    def test_backward_derivative_on_sinh(self):
        # Compute the derivative of a function using a backward difference
        # scheme.  A backward scheme will only look below x0.
        dsinh = nd.Derivative(np.sinh, method='backward')
        small = np.abs(dsinh(0.0) - np.cosh(0.0)) < dsinh.error_estimate
        self.assertTrue(small)

    def test_central_and_forward_derivative_on_log(self):
        # Although a central rule may put some samples in the wrong places, it
        # may still succeed
        dlog = nd.Derivative(np.log, method='central')
        x = 0.001
        small = np.abs(dlog(x) - 1.0 / x) < dlog.error_estimate
        self.assertTrue(small)

        # But forcing the use of a one-sided rule may be smart anyway
        dlog.method = 'forward'
        small = np.abs(dlog(x) - 1 / x) < dlog.error_estimate
        self.assertTrue(small)

    def test_forward_derivative_on_tan(self):
        # Control the behavior of Derivative - forward 2nd order method, with
        # only 1 Romberg term.
        dtan = nd.Derivative(np.tan, n=1, order=2, method='forward',
                             romberg_terms=1)
        y = dtan(np.pi)
        abserr = dtan.error_estimate
        self.assertTrue(np.abs(y - 1.0) < abserr)
        assert_array_almost_equal(y, 1.0, decimal=8)

    def test_derivative_poly1d(self):
        p0 = np.poly1d(range(1, 6))
        fd = nd.Derivative(p0, n=4, romberg_terms=0)
        p4 = p0.deriv(4)
        assert_array_almost_equal(fd(1), p4(1), decimal=4)

    def test_vectorized_derivative_of_x2(self):
        # Functions should be vectorized for speed, but its not
        # always easy to do.
        fun = lambda x: x**2
        df = nd.Derivative(fun, vectorized=True)
        x = np.linspace(0, 5, 6)
        assert_array_almost_equal(df(x), 2*x)


class TestJacobian(unittest.TestCase):

    def testjacobian(self):
        xdata = np.reshape(np.arange(0, 1, 0.1), (-1, 1))
        ydata = 1 + 2 * np.exp(0.75 * xdata)
        fun = lambda c: (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2
        Jfun = nd.Jacobian(fun)
        J = Jfun([1, 2, 0.75])  # should be numerically zero
        for ji in J.ravel():
            assert_array_almost_equal(ji, 0.0)


class TestGradient(unittest.TestCase):
    def testgradient(self):
        fun = lambda x: np.sum(x ** 2)
        dfun = nd.Gradient(fun)
        d = dfun([1, 2, 3])
        dtrue = [2., 4., 6.]
        for (di, dit) in zip(d, dtrue):
            assert_array_almost_equal(di, dit)


class TestHessian(unittest.TestCase):

    def testhessian(self):
        # cos(x-y), at (0,0)
        cos = np.cos
        fun = lambda xy: cos(xy[0] - xy[1])
        Hfun2 = nd.Hessian(fun)
        h2 = Hfun2([0, 0])  # h2 = [-1 1; 1 -1];
        htrue = [-1., 1., 1., -1.]
        for (hi, hit) in zip(h2.ravel(), htrue):
            assert_array_almost_equal(hi, hit)


class TestHessdiag(unittest.TestCase):

    def testhessdiag(self):
        fun = lambda x: x[0] + x[1] ** 2 + x[2] ** 3
        Hfun = nd.Hessdiag(fun)
        hd = Hfun([1, 2, 3])
        htrue = [0., 2., 18.]
        for (hi, hit) in zip(hd, htrue):
            assert_array_almost_equal(hi, hit)


class TestGlobalFunctions(unittest.TestCase):
    def test_vec2mat(self):
        mat = nd.core.vec2mat(np.arange(6), n=2, m=3)
        assert_array_almost_equal(mat.tolist(), [[0, 1, 2], [1, 2, 3]],
                                  decimal=12)

        mat = nd.core.vec2mat(np.arange(12), 3, 4)
        assert_array_almost_equal(mat.tolist(), [[0, 1, 2, 3],
                                                 [1, 2, 3, 4],
                                                 [2, 3, 4, 5]],
                                  decimal=12)

    def testdea3(self):
        Ei = np.zeros(3)
        linfun = lambda k: np.linspace(0, np.pi / 2., 2. ** (k + 5) + 1)
        for k in np.arange(3):
            x = linfun(k)
            Ei[k] = np.trapz(np.sin(x), x)
        [En, err] = nd.dea3(Ei[0], Ei[1], Ei[2])
        self.assertTrue(np.abs(En - 1) < err)
        assert_array_almost_equal(En, 1.0, decimal=8)

if __name__ == '__main__':
    unittest.main()
