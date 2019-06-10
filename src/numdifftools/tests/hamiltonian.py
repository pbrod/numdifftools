'''
Created on Jun 25, 2016

@author: pab
'''
import numpy as np
from numpy import pi, r_, sqrt
from scipy import constants, linalg, optimize
from numdifftools.multicomplex import c_abs


class ClassicalHamiltonian(object):
    """
    Hamiltonian

    Parameters
    ----------
    n : scalar
        number of ions in the chain
    w : scalar
        angular trap frequency
    C : scalar
        Coulomb constant times the electronic charge in SI units.
    m : scalar
        the mass of a single trapped ion in the chain
    """

    def __init__(self):
        self.n = 2
        f = 1000000      # f is a scalar, it's the trap frequency
        self.w = 2 * pi * f
        self.C = (4 * pi * constants.epsilon_0) ** (-1) * constants.e ** 2
        # C is a scalar, it's the I
        self.m = 39.96 * 1.66e-27

    def potential(self, positionvector):
        """
        Return potential

        Parameters
        ----------
        positionvector:  1-d array (vector) of length n
            positions of the n ions
        """
        x = np.asarray(positionvector)
        w = self.w
        C = self.C
        m = self.m

        # First we consider the potential of the harmonic oscillator
        v_x = 0.5 * m * (w ** 2) * sum(x ** 2)
        # then we add the coulomb interaction:
        for i, xi in enumerate(x):
            for xj in x[i + 1:]:
                v_x += C / (c_abs(xi - xj))
        return v_x

    def initialposition(self):
        """Defines initial position as an estimate for the minimize process."""
        n = self.n
        x_0 = r_[-(n - 1) / 2:(n - 1) / 2:n * 1j]
        return x_0

    def normal_modes(self, eigenvalues):
        """Return normal modes

        Computed eigenvalues of the matrix Vx are of the form
            (normal_modes)**2*m.
        """
        m = self.m
        normal_modes = sqrt(eigenvalues / m)
        return normal_modes


def run_hamiltonian(hessian, verbose=True):
    c = ClassicalHamiltonian()

    xopt = optimize.fmin(c.potential, c.initialposition(), xtol=1e-10)

    hessian.fun = c.potential
    hessian.full_output = True

    h, info = hessian(xopt)
    true_h = np.array([[5.23748385e-12, -2.61873829e-12],
                       [-2.61873829e-12, 5.23748385e-12]])
    eigenvalues = linalg.eigvals(h)
    normal_modes = c.normal_modes(eigenvalues)

    if verbose:
        print(c.potential([-0.5, 0.5]))
        print(c.potential([-0.5, 0.0]))
        print(c.potential([0.0, 0.0]))
        print(xopt)
        print('h', h)
        print('h-true_h', np.abs(h - true_h))
        print('error_estimate', info.error_estimate)

        print('eigenvalues', eigenvalues)
        print('normal_modes', normal_modes)
    return h, info.error_estimate, true_h
