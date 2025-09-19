# -*- coding:utf-8 -*-
""""""
from __future__ import absolute_import, division

from hypothesis import given, example, strategies as st
from numpy.testing import assert_allclose  # @UnresolvedImport
import pytest

import numpy as np

from .hamiltonian import run_hamiltonian
import numdifftools.nd_statsmodels as nd

pytestmark = pytest.mark.skipif(nd.approx_hess1 is None, reason="statsmodels is not installed!")


class TestHessian(object):

    def test_run_hamiltonian(self):
        # Important to restrict the step in order to avoid the
        # discontinutiy at x=[0,0] of the hamiltonian
        for method in ['central', 'complex']:

            hessian = nd.Hessian(None, step=1e-8, method=method)
            h, _error_estimate, true_h = run_hamiltonian(hessian,
                                                         verbose=False,
                                                         full_output=False)
            assert (np.abs((h - true_h) / true_h) < 1e-4).all()

    @staticmethod
    def test_hessian_cos_x_y__at_0_0():
        # cos(x-y), at (0,0)

        def fun(xy):
            return np.cos(xy[0] - xy[1])

        htrue = [[-1., 1.], [1., -1.]]
        methods = ['forward', 'backward', 'complex', 'central', 'central2']

        for method in methods:
            h_fun = nd.Hessian(fun, method=method)
            h2 = h_fun([0, 0])
            # print(method, (h2-np.array(htrue)))
            assert_allclose(h2, htrue, rtol=1e-3)


class TestJacobian(object):

    @staticmethod
    @given(st.floats(min_value=-1000, max_value=1000))
    def test_scalar_to_vector(val):

        def fun(x):
            return np.hstack((x, x ** 2, x ** 3)).ravel()

        with np.errstate(all='ignore'):
            for method in ['backward', 'forward',  'central', 'complex']:
                dfun = nd.Jacobian(fun, method=method)
                j0 = dfun(val).T
                assert_allclose(j0, [[1., 2 * val, 3 * val ** 2]], atol=1e-7)

    @staticmethod
    def test_on_scalar_function():

        def fun(x):
            return x[0] * x[1] * x[2] + np.exp(x[0]) * x[1]

        for method in ['forward', 'backward', 'central', 'complex']:
            j_fun = nd.Jacobian(fun, method=method)
            x = j_fun([3., 5., 7.])
            assert_allclose(x, [[135.42768462, 41.08553692, 15.]])

    def test_on_vector_valued_function(self):
        xdata = np.arange(0, 1, 0.1)
        ydata = 1 + 2 * np.exp(0.75 * xdata)

        def fun(c):
            return (c[0] + c[1] * np.exp(c[2] * xdata) - ydata) ** 2

        for method in ['backward', 'forward', 'central', 'complex']:

            j_fun = nd.Jacobian(fun, method=method)
            J = j_fun([1, 2, 0.75])  # should be numerically zero
            assert_allclose(J, np.zeros((ydata.size, 3)), atol=1e-5)

    @staticmethod
    def test_on_matrix_valued_function():

        def fun(x):
            x = np.atleast_1d(x)
            f0 = x[0] ** 2 + x[1] ** 2
            f1 = x[0] ** 3 + x[1] ** 3
            return np.array([f0, f1])

        def dfun(x):
            x = np.atleast_1d(x)
            f0_d0 = np.atleast_1d(x[0] * 2)
            f0_d1 = np.atleast_1d(x[1] * 2)
            f1_d0 = np.atleast_1d(3 * x[0] ** 2)
            f1_d1 = np.atleast_1d(3 * x[1] ** 2)
            # algopy way:
            # df0 = np.hstack([np.diag(f0_d0), np.diag(f0_d1)])
            # df1 = np.hstack([np.diag(f1_d0), np.diag(f1_d1)])
            # numdifftools way:
            df0 = np.vstack([f0_d0, f0_d1])
            df1 = np.vstack([f1_d0, f1_d1])

            return np.array([df0, df1]).squeeze()

        x = np.array([(1, 2, 3, 4),
                      (5, 6, 7, 8)], dtype=float)

        y = fun(x)
        assert_allclose(y, [[26., 40., 58., 80.], [126., 224., 370., 576.]])

        for method in ['forward','backward', 'central', 'complex']:
            jaca = nd.Jacobian(fun, method=method)

            val0 = jaca([1, 2])
            assert_allclose(val0, [[2., 4.],
                                   [3., 12.]])
            val1 = jaca([3, 4])
            assert_allclose(val1, [[6., 8.],
                                   [27., 48.]])

            val2 = jaca([[1, 2], [3, 4]])
            assert_allclose(val2, [[[2., 4.],
                                    [6., 8.]],
                                   [[3., 12.],
                                    [27., 48.]]])
# algopy format:
#                                   [[[2., 0., 6., 0.],
#                                               [0., 4., 0., 8.]],
#                                              [[3., 0., 27., 0.],
#                                               [0., 12., 0., 48.]]])

            val = jaca(x)
            assert_allclose(val, [[[2., 4., 6., 8.],
                                   [10., 12., 14., 16.]],
                                  [[3., 12., 27., 48.],
                                   [75., 108., 147., 192.]]])
# algopy format:
#                             [[[2., 0., 0., 0., 10., 0., 0., 0.],
#                                    [0., 4., 0., 0., 0., 12., 0., 0.],
#                                    [0., 0., 6., 0., 0., 0., 14., 0.],
#                                    [0., 0., 0., 8., 0., 0., 0., 16.]],
#                                   [[3., 0., 0., 0., 75., 0., 0., 0.],
#                                    [0., 12., 0., 0., 0., 108., 0., 0.],
#                                    [0., 0., 27., 0., 0., 0., 147., 0.],
#                                    [0., 0., 0., 48., 0., 0., 0., 192.]]])

    @staticmethod
    def test_issue_25():

        def g_fun(x):
            out = np.zeros((2, 2), dtype=float)
            out[0, 0] = x[0]
            out[0, 1] = x[1]
            out[1, 0] = x[0]
            out[1, 1] = x[1]
            return out

        dg_dx = nd.Jacobian(g_fun)
        x = [1, 2]
        dg = dg_dx(x)
        assert_allclose(dg, [[[1., 0.],
                              [0., 1.]],
                             [[1., 0.],
                              [0., 1.]]])


class TestGradient(object):

    @staticmethod
    def test_on_scalar_function():

        def fun(x):
            return np.sum(x ** 2)

        dtrue = [2., 4., 6.]

        for method in ['forward', 'backward', 'central', 'complex']:  #

            dfun = nd.Gradient(fun, method=method)
            d = dfun([1, 2, 3])
            assert_allclose(d, dtrue, atol=1e-8)
