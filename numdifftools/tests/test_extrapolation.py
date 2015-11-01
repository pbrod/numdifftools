import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from numdifftools.extrapolation import Dea, dea3, Richardson


class TestExtrapolation(unittest.TestCase):

    def setUp(self):
        n = 7
        Ei = np.zeros(n)
        h = np.zeros(n)
        linfun = lambda i : np.linspace(0, np.pi/2., 2**(i+5)+1)
        for k in np.arange(n):
            x = linfun(k)
            Ei[k] = np.trapz(np.sin(x),x)
            h[k] = x[1]
        self.Ei = Ei
        self.h = h


    def test_dea3_on_trapz_sin(self):
        Ei = self.Ei
        [En, err] = dea3(Ei[0], Ei[1], Ei[2])
        truErr = Ei[:3]-1.
        assert_allclose(truErr,
                        [ -2.00805680e-04, -5.01999079e-05, -1.25498825e-05])
        assert_allclose(En,  1.)
        self.assertLessEqual(err, 0.00021)


    def test_dea_on_trapz_sin(self):
        Ei = self.Ei
        dea_3 = Dea(3)
        for E in Ei:
            En, err = dea_3(E)

        truErr = Ei-1.
        err_bound = 10 * np.array([-2.00805680e-04,  -5.01999079e-05,
                                   -1.25498825e-05 -3.13746471e-06,
                                   -7.84365809e-07, -1.96091429e-07,
                                   -4.90228558e-08])
        self.assertTrue(np.all(truErr< err_bound))
        assert_allclose(En,  1.)
        self.assertLessEqual(err, 1e-10)

    def test_richardson(self):
        Ei, h = self.Ei[:, np.newaxis], self.h[:, np.newaxis]
        En, err, step = Richardson(step=1, order=1)(Ei, h)
        assert_allclose(En,  1.)
        self.assertTrue(np.all(err<0.0022))

#     def test_epsal():
#         HUGE = 1.E+60
#         TINY = 1.E-60
#         ZERO = 0.E0
#         ONE = 1.E0
#         true_vals = [0.78539816, 0.94805945, 0.99945672]
#         E = []
#         for N, SOFN in enumerate([0.78539816, 0.94805945, 0.98711580]):
#             E.append(SOFN)
#             if N == 0:
#                 ESTLIM = SOFN
#             else:
#                 AUX2 = ZERO
#                 for J in range(N, 0, -1):
#                     AUX1 = AUX2
#                     AUX2 = E[J-1]
#                     DIFF = E[J] - AUX2
#                     if (abs(DIFF) <= TINY):
#                         E[J-1] = HUGE
#                     else:
#                         E[J-1] = AUX1 + ONE/DIFF
#
#                 if (N % 2) == 0:
#                     ESTLIM = E[0]
#                 else:
#                     ESTLIM = E[1]
#             print(ESTLIM, true_vals[N])


if __name__ == "__main__":
    unittest.main()
