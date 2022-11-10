# -*- coding:utf-8 -*-
""""""
from __future__ import absolute_import, division

import pytest

import numpy as np
from numpy.testing import assert_allclose  # @UnresolvedImport
from hypothesis import given, strategies as st

try:
    import scipy
except ImportError:
    scipy = None
else:
    import numdifftools.nd_scipy as nd

pytestmark = pytest.mark.skipif(scipy is None, reason="scipy is not installed!")


class TestJacobian(object):

    @staticmethod
    @given(st.floats(min_value=-1e53, max_value=1e53))
    def test_scalar_to_vector(val):

        def fun(x):
            return np.array([x, x ** 2, x ** 3]).ravel()

        for method in ['backward', 'forward', "central", "complex"]:
            j0 = nd.Jacobian(fun, method=method)(val).T
            assert_allclose(j0, [[1., 2 * val, 3 * val ** 2]], atol=1e-6)

    @staticmethod
    def test_on_scalar_function():

        def fun(x):
            return x[0] * x[1] * x[2] + np.exp(x[0]) * x[1]

        for method in ['forward', 'backward', "central", "complex"]:
            j_fun = nd.Jacobian(fun, method=method)
            x = j_fun([3., 5., 7.])
            assert_allclose(x, [135.42768462, 41.08553692, 15.])

    def test_on_vector_valued_function(self):
        xdata = np.arange(0, 1, 0.1)
        ydata = 1 + 2 * np.exp(0.75 * xdata)

        def fun(c):
            return (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2

        for method in ['forward', 'backward', "central", "complex"]:

            j_fun = nd.Jacobian(fun, method=method)
            J = j_fun([1, 2, 0.75])  # should be numerically zero
            assert_allclose(J, np.zeros((ydata.size, 3)), atol=1e-6)

    @pytest.mark.skip("Not implemented for matrix valued functions")
    def test_on_matrix_valued_function(self):

        def fun(x):

            f0 = x[0] ** 2 + x[1] ** 2
            f1 = x[0] ** 3 + x[1] ** 3

            s0 = f0.size
            s1 = f1.size
            out = np.zeros((2, (s0 + s1) // 2), dtype=float)
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

    @pytest.mark.skip("Does not work on matrix valued functions.")
    def test_issue_25(self):

        def g_fun(x):
            out = np.zeros((2, 2), dtype=float)
            out[0, 0] = x[0]
            out[0, 1] = x[1]
            out[1, 0] = x[0]
            out[1, 1] = x[1]
            return out

        dg_dx = nd.Jacobian(g_fun)
        x = np.array([1, 2])

        tv = [[[1., 0.],
               [0., 1.]],
              [[1., 0.],
               [0., 1.]]]
        # _EPS = np.MachAr().eps
        # epsilon = _EPS**(1./4)
        # assert_allclose(nd.approx_fprime(x, g_fun, epsilon), tv)
        dg = dg_dx(x)
        assert_allclose(dg, tv)


class TestGradient(object):

    @staticmethod
    def test_on_scalar_function():

        def fun(x):
            return np.sum(x ** 2)

        dtrue = [2., 4., 6.]

        for method in ['forward', 'backward', "central", "complex"]:

            dfun = nd.Gradient(fun, method=method)
            d = dfun([1, 2, 3])
            assert_allclose(d, dtrue)

