"""
Created on 22. apr. 2015

@author: pab

References
----------
A methodology for robust optimization of
low-thrust trajectories in multi-body
environments
Gregory Lantoine (2010)
Phd thesis, Georgia Institute of Technology

Using multicomplex variables for automatic
computation of high-order derivatives
Gregory Lantoine, Ryan P. Russell , and Thierry Dargent
ACM Transactions on Mathematical Software, Vol. 38, No. 3, Article 16,
April 2012, 21 pages,
http://doi.acm.org/10.1145/2168773.2168774

Bicomplex Numbers and Their Elementary Functions
M.E. Luna-Elizarraras, M. Shapiro, D.C. Struppa1, A. Vajiac (2012)
CUBO A Mathematical Journal
Vol. 14, No 2, (61-80). June 2012.

Computation of higher-order derivatives using the multi-complex
step method
Adriaen Verheyleweghen, (2014)
Project report, NTNU

"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from numdifftools._typing import Array

_tiny_name = "tiny" if np.__version__ < "1.22" else "smallest_normal"
_TINY = getattr(np.finfo(float), _tiny_name)


def c_atan2(
    x: ArrayLike,
    y: ArrayLike,
) -> Array:
    a, b = np.real(x), np.imag(x)
    c, d = np.real(y), np.imag(y)
    return np.arctan2(a, c) + 1j * (c * b - a * d) / (a**2 + c**2)


def c_max(
    x: ArrayLike,
    y: ArrayLike,
) -> Array:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return np.where(x_arr.real < y_arr.real, y, x)


def c_min(
    x: ArrayLike,
    y: ArrayLike,
) -> Array:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return np.where(x_arr.real > y_arr.real, y, x)


def c_abs(z: ArrayLike) -> Array:
    z_arr = np.asarray(z)

    if np.all(np.iscomplex(z_arr)):
        return np.where(np.real(z_arr) >= 0, z_arr, -z_arr)
    return np.abs(z_arr)


class Bicomplex:
    """
    BICOMPLEX(z1, z2)

    Creates an instance of a Bicomplex object.
    zeta = z1 + j*z2, where z1 and z2 are complex numbers.
    """

    __slots__ = ("z1", "z2")

    z1: Array
    z2: Array

    def __init__(
        self,
        z1: ArrayLike,
        z2: ArrayLike,
        dtype: DTypeLike = np.complex128,
    ) -> None:
        z1, z2 = np.broadcast_arrays(z1, z2)
        self.z1 = np.asanyarray(z1, dtype=dtype)
        self.z2 = np.asanyarray(z2, dtype=dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.z1.shape

    @property
    def size(self) -> int:
        return self.z1.size

    def mod_c(self) -> Array:
        """Complex modulus"""
        r11, r22 = self.z1 * self.z1, self.z2 * self.z2
        r = np.sqrt(r11 + r22)
        return r

    def norm(self) -> Array:
        z1, z2 = self.z1, self.z2
        return np.sqrt(z1.real**2 + z2.real**2 + z1.imag**2 + z2.imag**2)

    @property
    def real(self) -> Array:
        return self.z1.real

    @property
    def imag(self) -> Array:
        return self.z1.imag

    @property
    def imag1(self) -> Array:
        return self.z1.imag

    @property
    def imag2(self) -> Array:
        return self.z2.real

    @property
    def imag12(self) -> Array:
        return self.z2.imag

    @staticmethod
    def asarray(other: Bicomplex) -> Array:
        z1, z2 = other.z1, other.z2
        return np.vstack((np.hstack((z1, -z2)), np.hstack((z2, z1))))

    @staticmethod
    def _coerce(other: Any) -> Bicomplex:
        if not isinstance(other, Bicomplex):
            return Bicomplex(other, np.zeros(np.shape(other)))
        return other

    @staticmethod
    def mat2bicomp(arr: Array) -> Bicomplex:
        shape = np.array(arr.shape)
        shape[:2] = shape[:2] // 2
        z1 = arr[: shape[0]]
        z2 = arr[shape[0] :]
        slices = tuple([slice(None, None, 1)] + [slice(n) for n in shape[1:]])
        return Bicomplex(z1[slices], z2[slices])

    @staticmethod
    def __array_wrap__(result: Any, context: Any = None, return_scalar: bool = False) -> Bicomplex:
        if isinstance(result, Bicomplex):
            return result
        shape = result.shape
        result = np.atleast_1d(result)
        z1 = np.array([cls.z1 for cls in result.ravel()])
        z2 = np.array([cls.z2 for cls in result.ravel()])
        return Bicomplex(z1.reshape(shape), z2.reshape(shape))

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"""{name!s}(z1={str(self.z1)!s}, z2={str(self.z2)!s})"""

    def __lt__(self, other: Any) -> Array:
        other = self._coerce(other)
        return self.z1.real < other.z1.real

    def __le__(self, other: Any) -> Array:
        other = self._coerce(other)
        return self.z1.real <= other.z1.real

    def __gt__(self, other: Any) -> Array:
        other = self._coerce(other)
        return self.z1.real > other.z1.real

    def __ge__(self, other: Any) -> Array:
        other = self._coerce(other)
        return self.z1.real >= other.z1.real

    def __eq__(self, other: Any) -> Array:  # type: ignore[override]
        other = self._coerce(other)
        return (self.z1 == other.z1) * (self.z2 == other.z2)

    def __getitem__(self, index: Any) -> Bicomplex:
        return Bicomplex(self.z1[index], self.z2[index])

    def __setitem__(self, index: Any, value: Any) -> None:
        value = self._coerce(value)
        if isinstance(index, str) and index in {"z1", "z2"}:
            setattr(self, index, value)
        else:
            self.z1[index] = value.z1
            self.z2[index] = value.z2

    def __abs__(self) -> Bicomplex:
        z1, z2 = self.z1, self.z2
        mask = self >= 0
        return Bicomplex(np.where(mask, z1, -z1), np.where(mask, z2, -z2))

    def __neg__(self) -> Bicomplex:
        return Bicomplex(-self.z1, -self.z2)

    def __add__(self, other: Any) -> Bicomplex:
        other = self._coerce(other)
        return Bicomplex(self.z1 + other.z1, self.z2 + other.z2)

    def __sub__(self, other: Any) -> Bicomplex:
        other = self._coerce(other)
        return Bicomplex(self.z1 - other.z1, self.z2 - other.z2)

    def __rsub__(self, other: Any) -> Bicomplex:
        return -self.__sub__(other)

    def __div__(self, other: Any) -> Bicomplex:  # python 2
        """elementwise division"""
        return self * other**-1  # np.exp(-np.log(other))

    __truediv__ = __div__  # python 3

    def __rdiv__(self, other: Any) -> Bicomplex:  # python 2
        """elementwise division"""
        return other * self**-1

    __rtruediv__ = __rdiv__  # python3

    def __mul__(self, other: Any) -> Bicomplex:
        """elementwise multiplication"""
        other = self._coerce(other)
        return Bicomplex(self.z1 * other.z1 - self.z2 * other.z2, self.z1 * other.z2 + self.z2 * other.z1)

    def _pow_singular(self, other: Any) -> Bicomplex:
        z1, z2 = self.z1, self.z2
        z01 = 0.5 * (z1 - 1j * z2) ** other
        z02 = 0.5 * (z1 + 1j * z2) ** other
        return Bicomplex(z01 + z02, (z01 - z02) * 1j)

    def __pow__(self, other: Any) -> Bicomplex:
        # TODO: Check correctness
        out = (self.log() * other).exp()
        non_invertible = np.abs(self.mod_c()) < 1e-15
        if non_invertible.any():
            out[non_invertible] = self[non_invertible]._pow_singular(other)
        return out

    def __rpow__(self, other: Any) -> Bicomplex:
        return (np.log(other) * self).exp()

    __radd__ = __add__
    __rmul__ = __mul__

    def __len__(self) -> int:
        return len(self.z1)

    def conjugate(self) -> Bicomplex:
        return Bicomplex(self.z1, -self.z2)

    def flat(self, index: Any) -> Bicomplex:
        return Bicomplex(self.z1.flat[index], self.z2.flat[index])

    def dot(self, other: Any) -> Bicomplex:
        other = self._coerce(other)
        if self.size == 1 or other.size == 1:
            return self * other
        return self.mat2bicomp(self.asarray(self).dot(self.asarray(other).T))

    def logaddexp(self, other: Any) -> Bicomplex:
        other = self._coerce(other)
        return self + np.log1p(np.exp(other - self))

    def logaddexp2(self, other: Any) -> Bicomplex:
        other = self._coerce(other)
        return self + np.log2(1 + np.exp2(other - self))

    def sin(self) -> Bicomplex:
        z1 = np.cosh(self.z2) * np.sin(self.z1)
        z2 = np.sinh(self.z2) * np.cos(self.z1)
        return Bicomplex(z1, z2)

    def cos(self) -> Bicomplex:
        z1 = np.cosh(self.z2) * np.cos(self.z1)
        z2 = -np.sinh(self.z2) * np.sin(self.z1)
        return Bicomplex(z1, z2)

    def tan(self) -> Bicomplex:
        return self.sin() / self.cos()

    def cot(self) -> Bicomplex:
        return self.cos() / self.sin()

    def sec(self) -> Bicomplex:
        return 1.0 / self.cos()

    def csc(self) -> Bicomplex:
        return 1.0 / self.sin()

    def cosh(self) -> Bicomplex:
        z1 = np.cosh(self.z1) * np.cos(self.z2)
        z2 = np.sinh(self.z1) * np.sin(self.z2)
        return Bicomplex(z1, z2)

    def sinh(self) -> Bicomplex:
        z1 = np.sinh(self.z1) * np.cos(self.z2)
        z2 = np.cosh(self.z1) * np.sin(self.z2)
        return Bicomplex(z1, z2)

    def tanh(self) -> Bicomplex:
        return self.sinh() / self.cosh()

    def coth(self) -> Bicomplex:
        return self.cosh() / self.sinh()

    def sech(self) -> Bicomplex:
        return 1.0 / self.cosh()

    def csch(self) -> Bicomplex:
        return 1.0 / self.sinh()

    def exp2(self) -> Bicomplex:
        return np.exp(self * np.log(2))

    def sqrt(self) -> Bicomplex:
        return self.__pow__(0.5)

    def log10(self) -> Bicomplex:
        return self.log() / np.log(10)

    def log2(self) -> Bicomplex:
        return self.log() / np.log(2)

    def log1p(self) -> Bicomplex:
        return Bicomplex(np.log1p(self.mod_c()), self.arg_c1p())

    def expm1(self) -> Bicomplex:
        expz1 = np.expm1(self.z1)
        return Bicomplex(expz1 * np.cos(self.z2), expz1 * np.sin(self.z2))

    def exp(self) -> Bicomplex:
        expz1 = np.exp(self.z1)
        return Bicomplex(expz1 * np.cos(self.z2), expz1 * np.sin(self.z2))

    def log(self) -> Bicomplex:
        mod_c = self.mod_c()
        #         if (mod_c == 0).any():
        #             raise ValueError('mod_c is zero -> number not invertable!')
        return Bicomplex(np.log(mod_c + _TINY), self.arg_c())

    #     def _log_m(self, m=0):
    #         return np.log(self.mod_c() + _TINY) + 1j * \
    #             (self.arg_c() + 2 * m * np.pi)
    #
    #     def _log_mn(self, m=0, n=0):
    #         arg_c = self.arg_c()
    #         log_m = np.log(self.mod_c() + _TINY) + 1j * (2 * m * np.pi)
    #         return Bicomplex(log_m, arg_c + 2 * n * np.pi)

    def arcsin(self) -> Bicomplex:
        J = Bicomplex(0, 1)
        return -J * ((J * self + (1 - self**2) ** 0.5).log())

    def arccos(self) -> Bicomplex:
        return np.pi / 2 - self.arcsin()

    def arctan(self) -> Bicomplex:
        J = Bicomplex(0, 1)
        arg1, arg2 = 1 - J * self, 1 + J * self
        tmp = J * (arg1.log() - arg2.log()) * 0.5
        return Bicomplex(tmp.z1, tmp.z2)

    def arccosh(self) -> Bicomplex:
        return (self + (self**2 - 1) ** 0.5).log()

    def arcsinh(self) -> Bicomplex:
        return (self + (self**2 + 1) ** 0.5).log()

    def arctanh(self) -> Bicomplex:
        return 0.5 * (((1 + self) / (1 - self)).log())

    @staticmethod
    def _arg_c(z1: Array, z2: Array) -> Array:
        sign = np.where((z1.real == 0) * (z2.real == 0), 0, np.where(0 <= z2.real, 1, -1))
        # clip to avoid nans for complex args
        arg = z2 / (z1 + _TINY).clip(min=-1e150, max=1e150)
        arg_c = np.arctan(arg) + sign * np.pi * (z1.real <= 0)
        return arg_c

    def arg_c1p(self) -> Array:
        z1, z2 = 1 + self.z1, self.z2
        return self._arg_c(z1, z2)

    def arg_c(self) -> Array:
        return self._arg_c(self.z1, self.z2)
