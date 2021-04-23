"""
Created on 28. aug. 2015

@author: pab
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose  # @UnresolvedImport
from numdifftools.limits import Limit, Residue, CStepGenerator
from numdifftools.step_generators import make_exact
from numdifftools.extrapolation import EPS


class TestCStepGenerator(object):

    @staticmethod
    def test_default_generator():
        step_gen = CStepGenerator(num_steps=8)
        h = np.array([h for h in step_gen(0)])
        print(h)
        desired = np.array([[1.47701940e-09, 3.69254849e-10, 9.23137122e-11,
                            2.30784281e-11, 5.76960701e-12, 1.44240175e-12,
                            3.60600438e-13, 9.01501096e-14]])

        assert_array_almost_equal((h - desired) / desired, 0)

    @staticmethod
    def test_default_base_step():
        step_gen = CStepGenerator(num_steps=1, offset=0)
        h = [h for h in step_gen(0)]
        desired = make_exact(EPS ** (1. / 1.2))
        assert_array_almost_equal((h[0] - desired) / desired, 0)

    @staticmethod
    def test_fixed_base_step():
        desired = 0.1
        step_gen = CStepGenerator(base_step=desired, num_steps=1, scale=2, offset=0)
        h = [h for h in step_gen(0)]
        assert_array_almost_equal((h[0] - desired) / desired, 0)


class TestLimit(object):

    def test_sinx_div_x(self):

        def fun(x):
            return np.sin(x) / x

        for path in ['radial', 'spiral']:
            lim_f = Limit(fun, path=path, full_output=True)

            x = np.arange(-10, 10) / np.pi
            lim_f0, err = lim_f(x * np.pi)
            assert_array_almost_equal(lim_f0, np.sinc(x))
            assert np.all(err.error_estimate < 1.0e-14)

    def test_derivative_of_cos(self):
        x0 = np.pi / 2

        def fun(x):
            return (np.cos(x0 + x) - np.cos(x0)) / x

        lim, err = Limit(fun, step=CStepGenerator(), full_output=True)(0)
        assert_allclose(lim, -1)
        assert err.error_estimate < 1e-14

    def test_residue_1_div_1_minus_exp_x(self):

        def fun(z):
            return -z / (np.expm1(2 * z))

        lim, err = Limit(fun, full_output=True)(0)
        assert_allclose(lim, -0.5)

        assert err.error_estimate < 1e-14

    def test_difficult_limit(self):

        def fun(x):
            return (x * np.exp(x) - np.expm1(x)) / x ** 2

        for path in ['radial', ]:
            lim, err = Limit(fun, path=path, full_output=True)(0)
            assert_allclose(lim, 0.5)

            assert err.error_estimate < 1e-8


class TestResidue(object):

    def test_residue_1_div_1_minus_exp_x(self):

        def fun(z):
            return -1.0 / (np.expm1(2 * z))

        res_h, err = Residue(fun, full_output=True)(0)
        assert_allclose(res_h, -0.5)

        assert err.error_estimate < 1e-14

    def test_residue_1_div_sin_x2(self):

        def fun(z):
            return 1.0 / np.sin(z) ** 2

        res_h, info = Residue(fun, full_output=True, pole_order=2)(np.pi)
        assert_allclose(res_h, 1)

        assert info.error_estimate < 1e-10
