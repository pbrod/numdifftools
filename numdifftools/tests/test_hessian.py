# -*- coding:utf-8 -*-
from __future__ import division
import unittest
import numdifftools as nd
import numpy as np
from numpy import pi, r_, sqrt, array
from scipy import linalg, optimize, constants
from numdifftools.multicomplex import c_abs as abs
_TINY = np.finfo(float).machar.tiny

#  Hamiltonian
#     H = sum_i(p_i2/(2m)+ 1/2 * m * w2 x_i2) + sum_(i!=j)(a/|x_i-x_j|)
class ClassicalHamiltonian(object):
    """
    Hamiltonian

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
    """

    def __init__(self):
        self.N = 2
        f = 1000000      # f is a scalar, it's the trap frequency
        self.w = 2 * pi * f
        self.C = (4 * pi * constants.epsilon_0) ** (-1) * constants.e ** 2
        # C is a scalar, it's the I
        self.m = 39.96 * 1.66e-27

    def potential(self, positionvector):
        """
        Return potential

        positionvector is an 1-d array (vector) of length N that contains the
        positions of the N ions
        """
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
        """
        Return normal modes

        the computed eigenvalues of the matrix Vx are of the form
        (normal_modes)2*m.
        """
        m = self.m
        normal_modes = sqrt(eigenvalues / m)
        return normal_modes


def _run_hamiltonian(verbose=True):
    c = ClassicalHamiltonian()
    if verbose:
        print(c.potential(array([-0.5, 0.5])))
        print(c.potential(array([-0.5, 0.0])))
        print(c.potential(array([0.0, 0.0])))

    xopt = optimize.fmin(c.potential, c.initialposition(), xtol=1e-10)
    # Important to restrict the step in order to avoid the discontinutiy at
    # x=[0,0]
    # hessian = nd.Hessian(c.potential, step_max=1.0, step_nom=np.abs(xopt))
    step = nd.MaxStepGenerator(step_max=2, step_ratio=4, num_steps=16)
    hessian = nd.Hessian(c.potential, step=step, method='central',
                         full_output=True)
    # hessian = algopy.Hessian(c.potential) # Does not work
    # hessian = scientific.Hessian(c.potential) # does not work
    H, info = hessian(xopt)
    true_H = np.array([[5.23748385e-12, -2.61873829e-12],
                       [-2.61873829e-12, 5.23748385e-12]])
    if verbose:
        print(xopt)
        print('H', H)
        print('H-true_H', np.abs(H-true_H))
        print('error_estimate', info.error_estimate)

        eigenvalues = linalg.eigvals(H)
        normal_modes = c.normal_modes(eigenvalues)

        print('eigenvalues', eigenvalues)
        print('normal_modes', normal_modes)
    return H, info.error_estimate, true_H


class TestHessian(unittest.TestCase):
    def test_hessian(self):
        H, _error_estimate, true_H = _run_hamiltonian(verbose=False)
        self.assertTrue((np.abs(H-true_H) < 1e-18).all())


if __name__ == '__main__':
    # _run_hamiltonian()
    unittest.main()
