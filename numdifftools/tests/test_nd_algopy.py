# -*- coding:utf-8 -*-
""""""

from __future__ import division
import unittest
import numdifftools.nd_algopy as nd
import numpy as np
from numpy.testing import assert_allclose
import algopy
from numdifftools.testing import rosen
from numdifftools.tests.hamiltonian import run_hamiltonian
from hypothesis import given, example, note, strategies as st

class TestHessian(unittest.TestCase):

    def test_run_hamiltonian(self):
        h, _error_estimate, true_h = run_hamiltonian(nd.Hessian(None),
                                                     verbose=False)
        self.assertTrue((np.abs((h - true_h)/true_h) < 1e-4).all())

    @staticmethod
    def test_hessian_cos_x_y__at_0_0():
        # cos(x-y), at (0,0)

        def fun(xy):
            return np.cos(xy[0] - xy[1])
        htrue = [[-1., 1.], [1., -1.]]
        methods = ['forward', ]  # 'reverse']

        for method in methods:
            h_fun = nd.Hessian(fun, method=method)
            h2 = h_fun([0, 0])
            # print(method, (h2-np.array(htrue)))
            assert_allclose(h2, htrue)


class TestDerivative(unittest.TestCase):

    # TODO: Derivative does not tackle non-finite values.
    #     def test_infinite_functions(self):
    #         def finf(x):
    #             return np.inf * np.ones_like(x)
    #         df = nd.Derivative(finf, method='forward')
    #         val = df(0)
    #         self.assert_(np.isnan(val))
    @staticmethod
    def test_directional_diff():
        v = [1, -1]
        x0 = [2, 3]

        directional_diff = nd.directionaldiff(rosen, x0, v)
        assert_allclose(directional_diff, 743.87633380824832)

    @staticmethod
    def test_high_order_derivative_cos():
        true_vals = (-1.0, 0.0, 1.0, 0.0) * 5

        x = np.pi / 2  # np.linspace(0, np.pi/2, 15)
        for method in ['forward', 'reverse']:
            nmax = 15 if method in ['forward'] else 2
            for n in range(1, nmax):
                d3cos = nd.Derivative(np.cos, n=n, method=method)
                y = d3cos(x)
                assert_allclose(y, true_vals[n - 1], atol=1e-15)

    @staticmethod
    def test_fun_with_additional_parameters():
        """Test for issue #9"""
        def func(x, a, b=1):
            return b * a * x * x * x
        methods = ['reverse', 'forward']
        dfuns = [nd.Jacobian, nd.Derivative, nd.Gradient, nd.Hessdiag,
                 nd.Hessian]
        for dfun in dfuns:
            for method in methods:
                df = dfun(func, method=method)
                val = df(0.0, 1.0, b=2)
                assert_allclose(val, 0)

    @staticmethod
    def test_derivative_cube():
        """Test for Issue 7"""
        def cube(x):
            return x * x * x

        shape = (3, 2)
        x = np.ones(shape) * 2
        for method in ['forward', 'reverse']:
            dcube = nd.Derivative(cube, method=method)
            dx = dcube(x)
            assert_allclose(list(dx.shape), list(shape),
                                      err_msg='Shape mismatch')
            txt = 'First differing element %d\n value = %g,\n true value = %g'
            for i, (val, tval) in enumerate(zip(dx.ravel(),
                                                (3 * x**2).ravel())):
                assert_allclose(val, tval, err_msg=txt % (i, val, tval))

    @staticmethod
    @given(st.floats())
    @example(0)
    def test_derivative_exp(x):
        for method in ['forward', 'reverse']:
            dexp = nd.Derivative(np.exp, method=method)
            assert_allclose(dexp(x), np.exp(x))

    @staticmethod
    def test_derivative_sin():
        # Evaluate the indicated (default = first)
        # derivative at multiple points
        for method in ['forward', 'reverse']:
            dsin = nd.Derivative(np.sin, method=method)
            x = np.linspace(0, 2. * np.pi, 13)
            y = dsin(x)
            assert_allclose(y, np.cos(x))

    @given(st.floats())
    def test_derivative_on_sinh(self, x):
        true_val = np.cosh(x)
        for method in ['forward', ]:  # 'reverse']: # TODO: reverse fails
            dsinh = nd.Derivative(np.sinh, method=method)
            val = dsinh(x)
            if np.isnan(true_val):
                self.assertEqual(np.isnan(val), np.isnan(true_val))
            else:
                self.assertAlmostEqual(val, true_val)

    @staticmethod
    @given(st.floats(min_value=1e-5))
    def test_derivative_on_log(x):
        for method in ['forward', 'reverse']:
            dlog = nd.Derivative(np.log, method=method)

            assert_allclose(dlog(x), 1.0 / x)


class TestJacobian(unittest.TestCase):
    @staticmethod
    @given(st.floats(min_value=-1e153, max_value=1e153))
    def test_scalar_to_vector(val):
        def fun(x):
            out = algopy.zeros((3, ), dtype=x)
            out[0] = x
            out[1] = x**2
            out[2] = x**3
            return out

        for method in ['reverse', 'forward']:
            j0 = nd.Jacobian(fun, method=method)(val).T
            assert_allclose(j0, [[1., 2*val, 3*val**2]], atol=1e-14)

    @staticmethod
    def test_on_scalar_function():
        def fun(x):
            return x[0] * x[1] * x[2] + np.exp(x[0]) * x[1]
        for method in ['forward', 'reverse']:
            j_fun = nd.Jacobian(fun, method=method)
            x = j_fun([3., 5., 7.])
            assert_allclose(x, [[135.42768462, 41.08553692, 15.]])

    def test_on_vector_valued_function(self):
        xdata = np.arange(0, 1, 0.1)
        ydata = 1 + 2 * np.exp(0.75 * xdata)

        def fun(c):
            return (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2

        for method in ['reverse', ]:

            j_fun = nd.Jacobian(fun, method=method)
            J = j_fun([1, 2, 0.75])  # should be numerically zero
            assert_allclose(J, np.zeros((ydata.size, 3)))

        # TODO: 'forward' not implemented
        j_fun = nd.Jacobian(fun, method='forward')
        self.assertRaises(NotImplementedError, j_fun, [1, 2, 0.75])

    @staticmethod
    def test_on_matrix_valued_function():
        def fun(x):

            f0 = x[0] ** 2 + x[1] ** 2
            f1 = x[0] ** 3 + x[1] ** 3

            s0 = f0.size
            s1 = f1.size
            out = algopy.zeros((2, (s0 + s1) / 2), dtype=x)
            out[0, :] = f0
            out[1, :] = f1
            return out

        x = np.array([(1, 2, 3, 4),
                      (5, 6, 7, 8)], dtype=float)

        y = fun(x)
        assert_allclose(y, [[26., 40., 58., 80.], [126., 224., 370., 576.]])

        for method in ['forward', ]:  # TODO: 'reverse' fails
            jaca = nd.Jacobian(fun, method=method)

            assert_allclose(jaca([1, 2]), [[[2., 4.]],
                                           [[3., 12.]]])
            assert_allclose(jaca([3, 4]), [[[6., 8.]],
                                           [[27., 48.]]])

            assert_allclose(jaca([[1, 2],
                                  [3, 4]]), [[[2., 0., 6., 0.],
                                              [0., 4., 0., 8.]],
                                             [[3., 0., 27., 0.],
                                              [0., 12., 0., 48.]]])

            val = jaca(x)
            assert_allclose(val, [[[2., 0., 0., 0., 10., 0., 0., 0.],
                                   [0., 4., 0., 0., 0., 12., 0., 0.],
                                   [0., 0., 6., 0., 0., 0., 14., 0.],
                                   [0., 0., 0., 8., 0., 0., 0., 16.]],
                                  [[3., 0., 0., 0., 75., 0., 0., 0.],
                                   [0., 12., 0., 0., 0., 108., 0., 0.],
                                   [0., 0., 27., 0., 0., 0., 147., 0.],
                                   [0., 0., 0., 48., 0., 0., 0., 192.]]])

    @staticmethod
    def test_issue_25():
        def g_fun(x):
            out = algopy.zeros((2, 2), dtype=x)
            out[0, 0] = x[0]
            out[0, 1] = x[1]
            out[1, 0] = x[0]
            out[1, 1] = x[1]
            return out

        dg_dx = nd.Jacobian(g_fun) # TODO:  method='reverse' fails
        x = [1, 2]
        dg = dg_dx(x)
        assert_allclose(dg, [[[1., 0.],
                             [0., 1.]],
                            [[1., 0.],
                             [0., 1.]]])


class TestGradient(unittest.TestCase):
    @staticmethod
    def test_on_scalar_function():
        def fun(x):
            return np.sum(x ** 2)

        dtrue = [2., 4., 6.]

        for method in ['forward', 'reverse']:  #

            dfun = nd.Gradient(fun, method=method)
            d = dfun([1, 2, 3])
            assert_allclose(d, dtrue)


class TestHessdiag(unittest.TestCase):
    @staticmethod
    def test_forward():
        def fun(x):
            return x[0] + x[1] ** 2 + x[2] ** 3
        htrue = np.array([0., 2., 18.])
        h_fun = nd.Hessdiag(fun)
        hd = h_fun([1, 2, 3])
        _error = hd - htrue
        assert_allclose(hd, htrue)

    @staticmethod
    def test_reverse():
        def fun(x):
            return x[0] + x[1] ** 2 + x[2] ** 3
        htrue = np.array([0., 2., 18.])
        h_fun = nd.Hessdiag(fun, method='reverse')
        hd = h_fun([1, 2, 3])
        _error = hd - htrue
        assert_allclose(hd, htrue)

if __name__ == '__main__':
    # _run_hamiltonian()
    unittest.main()
