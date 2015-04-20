""" Test functions for numdifftools module

"""
import unittest
import numdifftools.nd_cstep as nd
import numpy as np
from numpy.testing import assert_array_almost_equal


class TestStepGenerator(unittest.TestCase):

    def test_default_generator(self):
        step_gen = nd.StepsGenerator(base_step=None, num_steps=10,
                                     step_ratio=4, offset=-1)
        h = np.array([h for h in step_gen(0, scale=2)])
        desired = np.array([1.235265e-03, 3.088162e-04, 7.720404e-05,
                            1.930101e-05, 4.825253e-06, 1.206313e-06,
                            3.015783e-07, 7.539457e-08,
                            1.884864e-08, 4.712161e-09, 1.178040e-09])
        assert_array_almost_equal((h - desired) / desired, 0)

    def test_default_base_step(self):
        step_gen = nd.StepsGenerator(num_steps=1)
        h = [h for h in step_gen(0, scale=2)]
        desired = (10 * nd.EPS) ** (1. / 2) * 0.1
        assert_array_almost_equal((h[0] - desired) / desired, 0)

    def test_fixed_base_step(self):
        desired = 0.1
        step_gen = nd.StepsGenerator(base_step=desired, num_steps=1)
        h = [h for h in step_gen(0, scale=2)]
        assert_array_almost_equal((h[0] - desired) / desired, 0)


class TestDerivative(unittest.TestCase):

    def test_set_scale(self):
        def cube(x):
            return x * x * x
        dcube = nd.Derivative(cube)
        for method in ['complex', 'central', 'forward', 'backward']:
            dcube.method = method
            np.testing.assert_allclose(dcube.scale,
                                       dcube.default_scale(dcube.method))
        dcube.scale = 10
        np.testing.assert_allclose(dcube.scale, 10)

    def test_derivative_cube(self):
        '''Test for Issue 7'''
        def cube(x):
            return x * x * x
        dcube = nd.Derivative(cube)
        shape = (3, 2)
        x = np.ones(shape) * 2
        dx = dcube(x)
        assert_array_almost_equal(list(dx.shape), list(shape),
                                  decimal=12,
                                  err_msg='Shape mismatch')
        txt = 'First differing element %d\n value = %g,\n true value = %g'
        for i, (val, tval) in enumerate(zip(dx.ravel(), (3 * x**2).ravel())):
            assert_array_almost_equal(val, tval, decimal=12,
                                      err_msg=txt % (i, val, tval))

    def test_derivative_exp(self):
        # derivative of exp(x), at x == 0
        dexp = nd.Derivative(np.exp)
        assert_array_almost_equal(dexp(0), np.exp(0), decimal=8)

    def test_derivative_sin(self):
        # Evaluate the indicated (default = first)
        # derivative at multiple points
        dsin = nd.Derivative(np.sin)
        x = np.linspace(0, 2. * np.pi, 13)
        y = dsin(x)
        np.testing.assert_allclose(y, np.cos(x))

    def test_backward_derivative_on_sinh(self):
        # Compute the derivative of a function using a backward difference
        # scheme.  A backward scheme will only look below x0.
        dsinh = nd.Derivative(np.sinh, method='backward')
        self.assertAlmostEqual(dsinh(0.0), np.cosh(0.0))

    def test_central_and_forward_derivative_on_log(self):
        # Although a central rule may put some samples in the wrong places, it
        # may still succeed
        epsilon = nd.StepsGenerator(num_steps=10)
        dlog = nd.Derivative(np.log, method='central', steps=epsilon)
        x = 0.001
        self.assertAlmostEqual(dlog(x), 1.0 / x)

        # But forcing the use of a one-sided rule may be smart anyway
        dlog = nd.Derivative(np.log, method='forward', steps=epsilon)
        self.assertAlmostEqual(dlog(x), 1 / x)


class TestJacobian(unittest.TestCase):

    def testjacobian(self):
        xdata = np.reshape(np.arange(0, 1, 0.1), (-1, 1))
        ydata = 1 + 2 * np.exp(0.75 * xdata)

        def fun(c):
            return (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2
        Jfun = nd.Jacobian(fun)
        J = Jfun([1, 2, 0.75])  # should be numerically zero
        for ji in J.ravel():
            assert_array_almost_equal(ji, 0.0)


class TestGradient(unittest.TestCase):

    def testgradient(self):
        def fun(x):
            return np.sum(x ** 2)
        dtrue = [2., 4., 6.]
        epsilon = nd.StepsGenerator(num_steps=10)
        for method in ['complex', 'central', 'backward', 'forward']:
            dfun = nd.Gradient(fun, method=method, steps=epsilon)
            d = dfun([1, 2, 3])

            for (di, dit) in zip(d, dtrue):
                assert_array_almost_equal(di, dit)


class TestHessian(unittest.TestCase):

    def test_hessian_cosIx_yI_at_I0_0I(self):
        # cos(x-y), at (0,0)
        epsilon = nd.StepsGenerator(num_steps=10)
        cos = np.cos

        def fun(xy):
            return cos(xy[0] - xy[1])
        htrue = [-1., 1., 1., -1.]
        methods = ['complex', 'central', 'central2', 'forward', 'backward']
        for method in methods:
            Hfun2 = nd.Hessian(fun, method=method, steps=epsilon)
            h2 = Hfun2([0, 0])  # h2 = [-1 1; 1 -1];
            for (hi, hit) in zip(h2.ravel(), htrue):
                assert_array_almost_equal(hi, hit)


if __name__ == '__main__':
    unittest.main()
