"""
Created on 22. apr. 2015

@author: pab
"""
import unittest

from numdifftools.multicomplex import Bicomplex
from numdifftools.example_functions import get_function
import numpy as np

EPS = np.MachAr().eps


def _default_base_step(x, scale, epsilon=None):
    h = (10 * EPS) ** (1. / scale) * np.maximum(np.log1p(np.abs(x)), 0.1)
    return h


class BicomplexTester(unittest.TestCase):

    def test_init(self):
        z = Bicomplex(1, 2)
        self.assertEqual(z.z1, 1)
        self.assertEqual(z.z2, 2)

    def test_neg(self):
        z = Bicomplex(1, 2)
        z2 = -z
        self.assertEqual(z2.z1, -z.z1)
        self.assertEqual(z2.z2, -z.z2)

    def test_shape(self):
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)
        self.assertEqual(z.shape, shape)

        z = Bicomplex(1, 2)
        self.assertEqual(z.shape, ())

    def test_norm(self):
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)
        np.testing.assert_array_equal(z.norm(), np.sqrt(5*t**2))

        z = Bicomplex(1, 2)
        self.assertEqual(z.norm(), np.sqrt(5))

    @staticmethod
    def test_lt():
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)
        z2 = Bicomplex(1, 2)
        val = z < z2
        truth = np.array([[True, False, False],
                          [False, False, False],
                          [False, False, False]], dtype=bool)
        np.testing.assert_array_equal(val, truth)

    @staticmethod
    def test_le():
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)
        z2 = Bicomplex(1, 2)
        val = z <= z2
        truth = np.array([[True, True, False],
                          [False, False, False],
                          [False, False, False]], dtype=bool)
        np.testing.assert_array_equal(val, truth)

    @staticmethod
    def test_ge():
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)
        z2 = Bicomplex(1, 2)
        val = z >= z2
        truth = np.array([[False, True, True],
                          [True, True, True],
                          [True, True, True]], dtype=bool)
        np.testing.assert_array_equal(val, truth)

    @staticmethod
    def test_gt():
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)
        z2 = Bicomplex(1, 2)
        val = z > z2
        truth = np.array([[False, False, True],
                          [True, True, True],
                          [True, True, True]], dtype=bool)
        np.testing.assert_array_equal(val, truth)

    @staticmethod
    def test_eq():
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)
        z2 = Bicomplex(1, 2)
        val = z == z2
        truth = np.array([[False, True, False],
                          [False, False, False],
                          [False, False, False]], dtype=bool)
        np.testing.assert_array_equal(val, truth)

    def test_conjugate(self):
        z = Bicomplex(1, 2)
        z2 = Bicomplex(1, -2)
        self.assertTrue(z.conjugate() == z2)

    def test_flat(self):
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)
        t = z.flat(1)
        self.assertTrue(t == Bicomplex(1, 2))

    @staticmethod
    def test_subsref():
        shape = (3, 3)
        t = np.arange(9).reshape(shape)
        z = Bicomplex(t, 2 * t)

        z0 = z[0]
        np.testing.assert_array_equal(z0.z1, z.z1[0])
        np.testing.assert_array_equal(z0.z2, z.z2[0])
        z1 = z[:]
        np.testing.assert_array_equal(z1.z1, z.z1[:])
        np.testing.assert_array_equal(z1.z2, z.z2[:])
        z1 = z[1:3, 1:3]
        np.testing.assert_array_equal(z1.z1, z.z1[1:3, 1:3])
        np.testing.assert_array_equal(z1.z2, z.z2[1:3, 1:3])

    @staticmethod
    def test_assign():
        shape = (3, 3)
        z = Bicomplex(np.ones(shape), 2 * np.ones(shape))
        z0 = z[0]
        np.testing.assert_array_equal(z0.z1, z.z1[0])
        np.testing.assert_array_equal(z0.z2, z.z2[0])
        z1 = z[:]
        np.testing.assert_array_equal(z1.z1, z.z1[:])
        np.testing.assert_array_equal(z1.z2, z.z2[:])

    @staticmethod
    def test_add():
        shape = (3, 3)
        z0 = Bicomplex(np.ones(shape), 2 * np.ones(shape))
        z1 = Bicomplex(3 * np.ones(shape), 4 * np.ones(shape))
        z2 = z0 + z1

        np.testing.assert_array_equal(z2.z1, z0.z1 + z1.z1)
        np.testing.assert_array_equal(z2.z2, z0.z2 + z1.z2)
        z3 = z0 + 1
        np.testing.assert_array_equal(z3.z1, z0.z1 + 1)
        np.testing.assert_array_equal(z3.z2, z0.z2)

    @staticmethod
    def test_sub():
        shape = (3, 3)
        z0 = Bicomplex(np.ones(shape), 2 * np.ones(shape))
        z1 = Bicomplex(3 * np.ones(shape), 4 * np.ones(shape))
        z2 = z0 - z1

        np.testing.assert_array_equal(z2.z1, z0.z1 - z1.z1)
        np.testing.assert_array_equal(z2.z2, z0.z2 - z1.z2)

    @staticmethod
    def test_rsub():
        z1 = Bicomplex(2, 1)
        a = 1 + 1j
        z2 = a - z1
        np.testing.assert_array_equal(z2.z1, a - z1.z1)
        np.testing.assert_array_equal(z2.z2, -z1.z2)

    def test_repr(self):
        z = Bicomplex(1, 2)
        txt = repr(z)
        self.assertEqual(txt, "Bicomplex(z1=(1+0j), z2=(2+0j))")

    @staticmethod
    def test_multiplication():
        z1 = Bicomplex(1, 2)
        z2 = Bicomplex(3, 4)
        z3 = z1 * z2
        np.testing.assert_array_equal(z3.z1, z1.z1 * z2.z1 - z1.z2 * z2.z2)
        np.testing.assert_array_equal(z3.z2, z1.z1 * z2.z2 + z1.z2 * z2.z1)

    @staticmethod
    def test_pow():
        z1 = Bicomplex(1, 2)
        z2 = z1 ** 2
        z3 = z1 * z1
        np.testing.assert_allclose(z2.z1, z1.z1 * z1.z1 - z1.z2 * z1.z2)
        np.testing.assert_allclose(z2.z2, z1.z1 * z1.z2 + z1.z2 * z1.z1)
        np.testing.assert_allclose(z3.z1, z1.z1 * z1.z1 - z1.z2 * z1.z2)
        np.testing.assert_allclose(z3.z2, z1.z1 * z1.z2 + z1.z2 * z1.z1)

        z1 = Bicomplex(z1=-1j, z2=-1-0j)

        z2 = z1 * z1
        z3 = z1 ** 2
        np.testing.assert_allclose(z2.z1, z1.z1 * z1.z1 - z1.z2 * z1.z2)
        np.testing.assert_allclose(z2.z2, z1.z1 * z1.z2 + z1.z2 * z1.z1)
        np.testing.assert_allclose(z3.z1, z1.z1 * z1.z1 - z1.z2 * z1.z2)
        np.testing.assert_allclose(z3.z2, z1.z1 * z1.z2 + z1.z2 * z1.z1)

    @staticmethod
    def test_division():
        z1 = Bicomplex(1, 2)
        z2 = Bicomplex(3, 4)
        z3 = z1 / z2
        z4 = z1 * (z2**-1)
        np.testing.assert_allclose(z3.z1, z4.z1)
        np.testing.assert_allclose(z3.z2, z4.z2)

# TODO: test_rdivision crashes on python3.4 on travis
#     def test_rdivision(self):
#
#         z2 = Bicomplex(3, 4)
#         z3 = 1 / z2
#         z4 = (z2**-1)
#         np.testing.assert_array_equal(z3.z1, z4.z1)
#         np.testing.assert_array_equal(z3.z2, z4.z2)

    @staticmethod
    def test_rpow():
        z2 = Bicomplex(3, 4)
        z3 = 2. ** z2
        z4 = np.exp(z2*np.log(2))
        np.testing.assert_allclose(z3.z1, z4.z1)
        np.testing.assert_allclose(z3.z2, z4.z2)

    @staticmethod
    def test_dot():
        z1 = Bicomplex(1, 2)
        z2 = Bicomplex(3, 4)
        z3 = z1.dot(z2)
        z4 = z1 * z2
        np.testing.assert_array_equal(z3.z1, z4.z1)
        np.testing.assert_array_equal(z3.z2, z4.z2)

    @staticmethod
    def test_cos():
        z1 = Bicomplex(np.linspace(0, np.pi, 5), 0)
        z2 = z1.cos()  # np.cos(z1)
        np.testing.assert_array_equal(z2.z1, np.cos(z1.z1))

    @staticmethod
    def test_arg_c():
        z1 = Bicomplex(np.linspace(0, np.pi, 5), 0)
        z2 = z1.arg_c()
        np.testing.assert_array_equal(z2, np.arctan2(z1.z2.real, z1.z1.real))

        z3 = Bicomplex(0.1, np.linspace(0, np.pi, 5))
        z4 = z3.arg_c()
        np.testing.assert_allclose(z4.real, np.arctan2(z3.z2.real, z3.z1.real))

    @staticmethod
    def test_arcsin():
        z1 = Bicomplex(np.linspace(-0.98, 0.98, 5), 0)
        z2 = z1.arcsin()
        np.testing.assert_allclose(z2.real, np.arcsin(z1.z1).real, atol=1e-15)
        np.testing.assert_allclose(z2.imag1, np.arcsin(z1.z1).imag, atol=1e-15)

    @staticmethod
    def test_arccos():
        z1 = Bicomplex(np.linspace(-0.98, 0.98, 5), 0)
        z2 = z1.arccos()
        np.testing.assert_allclose(z2.real, np.arccos(z1.z1).real, atol=1e-15)
        np.testing.assert_allclose(z2.imag1, np.arccos(z1.z1).imag, atol=1e-15)

    @staticmethod
    def test_der_cos():
        x = np.linspace(-0.99, 0.99, 5)
        h = 1e-9
        der1 = np.cos(Bicomplex(x + h * 1j, 0)).imag1 / h
        np.testing.assert_allclose(der1, - np.sin(x))
        h *= 100
        der2 = np.cos(Bicomplex(x + h * 1j, h)).imag12 / h**2
        np.testing.assert_allclose(der2, -np.cos(x))

    @staticmethod
    def test_der_log():
        x = np.linspace(0.001, 5, 6)
        h = 1e-15
        der1 = np.log(Bicomplex(x + h * 1j, 0)).imag1 / h
        np.testing.assert_allclose(der1, 1./x)
        der2 = np.log(Bicomplex(x + h * 1j, h)).imag12 / h**2
        np.testing.assert_allclose(der2, -1./x**2)

    @staticmethod
    def test_der_arccos():
        x = np.linspace(-0.98, 0.98, 5)
        h = 1e-8
        der1 = np.arccos(Bicomplex(x + h * 1j, 0)).imag1 / h
        np.testing.assert_allclose(der1, -1. / np.sqrt(1 - x**2))

        h = (_default_base_step(x, scale=2.5) + 1) - 1
        der2 = np.arccos(Bicomplex(x + h * 1j, h)).imag12 / h**2
        true_der2 = -x / (1 - x**2)**(3. / 2)
        np.testing.assert_allclose(der2, true_der2, atol=1e-5)

    @staticmethod
    def test_der_arccosh():
        x = np.linspace(1.2, 5, 5)
        h = 1e-8
        der1 = np.arccosh(Bicomplex(x + h * 1j, 0)).imag1 / h
        np.testing.assert_allclose(der1, 1. / np.sqrt(x**2 - 1))

        h = (_default_base_step(x, scale=2.5) + 1) - 1
        der2 = np.arccosh(Bicomplex(x + h * 1j, h)).imag12 / h**2
        true_der2 = -x / (x**2-1)**(3. / 2)
        np.testing.assert_allclose(der2, true_der2, atol=1e-5)

    @staticmethod
    def test_der_abs():
        x = np.linspace(-0.98, 0.98, 5)
        h = 1e-8
        der1 = abs(Bicomplex(x + h * 1j, 0)).imag1 / h
        np.testing.assert_allclose(der1, np.where(x < 0, -1, 1))
        der2 = abs(Bicomplex(x + h * 1j, h)).imag12 / h**2
        np.testing.assert_allclose(der2, 0, atol=1e-6)

    @staticmethod
    def test_der_arctan():
        x = np.linspace(0, 2, 5)
        h = 1e-8
        der1 = np.arctan(Bicomplex(x + h * 1j, 0)).imag1 / h
        np.testing.assert_allclose(der1, 1. / (1 + x**2))

        der2 = Bicomplex(x+h*1j, h).arctan().imag12/h**2
        np.testing.assert_allclose(der2, -2*x/(1+x**2)**2)


def _test_first_derivative(name):
    x = np.linspace(0.0001, 0.98, 5)
    h = _default_base_step(x, scale=2)
    f, df = get_function(name, n=1)

    der = f(Bicomplex(x + h * 1j, 0)).imag1 / h
    der_true = df(x)
    np.testing.assert_allclose(der, der_true, err_msg='{0!s}'.format(name))


def _test_second_derivative(name):
    x = np.linspace(0.01, 0.98, 5)
    h = _default_base_step(x, scale=2.5)

    f, df = get_function(name, n=2)

    der = f(Bicomplex(x + h * 1j, h)).imag12 / h**2
    der_true = df(x)
    np.testing.assert_allclose(der, der_true, err_msg='{0!s}'.format(name))

_function_names = ['cos', 'sin', 'tan', 'arccos', 'arcsin', 'arctan', 'cosh',
                   'sinh', 'tanh', 'exp', 'log', 'exp2', 'square', 'sqrt',
                   'log1p', 'expm1', 'log10', 'log2', 'arcsinh',
                   'arctanh']


class DerivativeTester(unittest.TestCase):
    @staticmethod
    def test_all_first_derivatives():
        for name in _function_names:
            _test_first_derivative(name)

    @staticmethod
    def test_all_second_derivatives():
        for name in _function_names:
            _test_second_derivative(name)


if __name__ == "__main__":
    unittest.main()
