"""
Created on 28. aug. 2015

@author: pab
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal  # @UnresolvedImport

from numdifftools.extrapolation import EPS
from numdifftools.limits import CStepGenerator, Limit, Residue
from numdifftools.step_generators import make_exact


class TestCStepGenerator:
    @staticmethod
    def test_default_generator():
        step_gen = CStepGenerator(num_steps=8)
        h = np.array(list(step_gen(0)))
        print(h)
        desired = np.array(
            [
                [
                    1.47701940e-09,
                    3.69254849e-10,
                    9.23137122e-11,
                    2.30784281e-11,
                    5.76960701e-12,
                    1.44240175e-12,
                    3.60600438e-13,
                    9.01501096e-14,
                ]
            ]
        )

        assert_array_almost_equal((h - desired) / desired, 0)

    @staticmethod
    def test_default_base_step():
        step_gen = CStepGenerator(num_steps=1, offset=0)
        h = list(step_gen(0))
        desired = make_exact(EPS ** (1.0 / 1.2))
        assert_array_almost_equal((h[0] - desired) / desired, 0)

    @staticmethod
    def test_fixed_base_step():
        desired = 0.1
        step_gen = CStepGenerator(base_step=desired, num_steps=1, scale=2, offset=0)
        h = list(step_gen(0))
        assert_array_almost_equal((h[0] - desired) / desired, 0)


class TestLimit:
    def test_sinx_div_x(self):
        def fun(x):
            return np.sin(x) / x

        for path in ["radial", "spiral"]:
            lim_f = Limit(fun, path=path, full_output=True)

            x = np.arange(-10, 10) / np.pi
            lim_f0 = lim_f(x * np.pi)
            assert_array_almost_equal(lim_f0.estimate, np.sinc(x))
            assert np.all(lim_f0.error_estimate < 1.0e-14)

    def test_derivative_of_cos(self):
        x0 = np.pi / 2

        def fun(x):
            return (np.cos(x0 + x) - np.cos(x0)) / x

        lim_f = Limit(fun, step=CStepGenerator(), full_output=True)(0)
        assert_allclose(lim_f.estimate, -1)
        assert lim_f.error_estimate < 1e-14

    def test_residue_1_div_1_minus_exp_x(self):
        def fun(z):
            return -z / (np.expm1(2 * z))

        lim_f = Limit(fun, full_output=True)(0)
        assert_allclose(lim_f.estimate, -0.5)

        assert lim_f.error_estimate < 1e-14

    def test_difficult_limit(self):
        def fun(x):
            return (x * np.exp(x) - np.expm1(x)) / x**2

        for path in [
            "radial",
        ]:
            lim_f = Limit(fun, path=path, full_output=True)(0)
            assert_allclose(lim_f.estimate, 0.5)

            assert lim_f.error_estimate < 1e-8


class TestResidue:
    def test_residue_1_div_1_minus_exp_x(self):
        def fun(z):
            return -1.0 / (np.expm1(2 * z))

        res_f = Residue(fun, full_output=True)(0)
        assert_allclose(res_f.estimate, -0.5)

        assert res_f.error_estimate < 1e-14

    def test_residue_1_div_sin_x2(self):
        def fun(z):
            return 1.0 / np.sin(z) ** 2

        res_f = Residue(fun, full_output=True, pole_order=2)(np.pi)
        assert_allclose(res_f.estimate, 1)

        assert res_f.error_estimate < 1e-10
