from __future__ import division, print_function
import numpy as np
import warnings
from numpy import linalg
from scipy import special
from numdifftools.extrapolation import convolve

_SQRT_J = (1j + 1.0) / np.sqrt(2.0)  # = 1j**0.5

# step_ratio, parity, nterms
FD_RULES = {}
#     (2.0, 1, 1): array([[1.]]),
#     (2.0, 1, 2): array([[-0.333333333333333333333, 2.666666666666666666666666666],
#                         [8., -16.]]),
#     (2.0, 1, 3): array([[2.22222222222222222e-02, -8.8888888888889e-01, 5.6888888888888889e+00],
#                         [-2.666666666666667e+00, 9.0666666666666667e+01, -1.7066666666666667e+02],
#                         [1.7066666666666667e+02, -1.7066666666666667e+03, 2.7306666666666667e+03]]),
#     (2.0, 0, 2): array([[-1., 4.],
#                         [4., -8.]]),
#     (2.0, 0, 4): array([[-4.76190476e-02, 1.33333333e+00, -1.06666667e+01, 2.43809524e+01],
#                         [1.33333333e+00, -3.46666667e+01, 2.34666667e+02, -3.41333333e+02],
#                         [-1.60000000e+01, 3.52000000e+02, -1.66400000e+03, 2.04800000e+03],
#                         [7.31428571e+01, -1.02400000e+03, 4.09600000e+03, -4.68114286e+03]])}


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


class LogRule(object):
    """ Log spaced finite difference rule class

    Parameters
    ----------
    n : int, optional
        Order of the derivative.
    method : {'central', 'complex', 'multicomplex', 'forward', 'backward'}
        defines the method used in the approximation
    order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.
    step_ratio : real scalar, optional, default 2.0
        Ratio between sequential steps generated.


    Examples
    --------
    >>> np.allclose(LogRule(n=1, method='central', order=2, step_ratio=2.).rule, 1)
    True
    >>> np.allclose(LogRule(n=1, method='central', order=4, step_ratio=2.).rule,
    ...             [-0.33333333,  2.66666667])
    True
    >>> np.allclose(LogRule(n=1, method='central', order=6, step_ratio=2.).rule,
    ...             [ 0.02222222, -0.88888889,  5.68888889])
    True

    >>> np.allclose(LogRule(n=1, method='forward', order=2, step_ratio=2.).rule, [-1.,  4.])
    True

    >>> np.allclose(LogRule(n=1, method='forward', order=4, step_ratio=2.).rule,
    ...             [ -0.04761905,   1.33333333, -10.66666667,  24.38095238])
    True
    >>> np.allclose(LogRule(n=1, method='forward', order=6, step_ratio=2.).rule,
    ...    [ -1.02406554e-04,   1.26984127e-02,  -5.07936508e-01,
    ...       8.12698413e+00,  -5.20126984e+01,   1.07381055e+02])
    True
    >>> fd_rule = LogRule(n=2, method='forward', order=4, step_ratio=2.0)
    >>> h = 0.002*(1./fd_rule.step_ratio)**np.arange(6)
    >>> x0 = 1.
    >>> f = np.exp
    >>> f_x0 = f(x0)
    >>> f_del = f(x0+h) - f_x0  # forward difference
    >>> fder, h = fd_rule.apply(f_del, h)
    >>> np.allclose(fder, f(x0))
    True

    """

    def __init__(self, n, method, order=2, step_ratio=2.0):
        self.n = n
        self.method = method
        self.order = order
        self.step_ratio = step_ratio

# --- properties ---

    @property
    def _odd_derivative(self):
        return self.n % 2 == 1

    @property
    def _even_derivative(self):
        return self.n % 2 == 0

    @property
    def _derivative_mod_four_is_three(self):
        return self.n % 4 == 3

    @property
    def _derivative_mod_four_is_zero(self):
        return self.n % 4 == 0

    @property
    def _complex_high_order(self):
        return self.method == 'complex' and (self.n > 1 or self.order >= 4)

    def _richardson_step(self):
        complex_step = 4 if self._complex_high_order else 2
        return dict(central=2,
                    central2=2,
                    complex=complex_step,
                    multicomplex=2).get(self.method, 1)

    @property
    def _method_order(self):
        step = self._richardson_step()
        # Make sure it is even and at least 2 or 4
        order = max((self.order // step) * step, step)
        return order


    def _parity_complex(self, order, method_order):
        if self.n == 1 and method_order < 4:
            return (order % 2) + 1
        return (3
                + 2 * int(self._odd_derivative)
                + int(self._derivative_mod_four_is_three)
                + int(self._derivative_mod_four_is_zero))

    def _parity(self, method, order, method_order):
        if method.startswith('central'):
            return (order % 2) + 1
        if method == 'complex':
            return self._parity_complex(order, method_order)
        return 0

    @staticmethod
    def _fd_matrix(step_ratio, parity, nterms):
        """
        Return matrix for finite difference and complex step derivation.

        Parameters
        ----------
        step_ratio : real scalar
            ratio between steps in unequally spaced difference rule.
        parity : scalar, integer
            0 (one sided, all terms included but zeroth order)
            1 (only odd terms included)
            2 (only even terms included)
            3 (only every 4'th order terms included starting from order 2)
            4 (only every 4'th order terms included starting from order 4)
            5 (only every 4'th order terms included starting from order 1)
            6 (only every 4'th order terms included starting from order 3)
        nterms : scalar, integer
            number of terms
        """
        _assert(0 <= parity <= 6,
                'Parity must be 0, 1, 2, 3, 4, 5 or 6! ({0:d})'.format(parity))
        step = [1, 2, 2, 4, 4, 4, 4][parity]
        inv_sr = 1.0 / step_ratio
        offset = [1, 1, 2, 2, 4, 1, 3][parity]
        c0 = [1.0, 1.0, 1.0, 2.0, 24.0, 1.0, 6.0][parity]
        c = c0 / \
            special.factorial(np.arange(offset, step * nterms + offset, step))
        [i, j] = np.ogrid[0:nterms, 0:nterms]
        return np.atleast_2d(c[j] * inv_sr ** (i * (step * j + offset)))

    @property
    def _flip_fd_rule(self):
        return ((self._even_derivative and (self.method == 'backward'))
                or (self.method == 'complex' and (self.n % 8 in [3, 4, 5, 6])))

    @property
    def _multicomplex_middle_name_or_empty(self):
        if self.method == 'multicomplex' and self.n > 1:
            _assert(self.n <= 2, 'Multicomplex method only support first '
                    'and second order derivatives.')
            return '2'
        return ''

    def _get_middle_name(self):
        if self._even_derivative and self.method in ('central', 'complex'):
            return '_even'
        if self._complex_high_order and self._odd_derivative:
            return '_odd'
        return self._multicomplex_middle_name_or_empty

    def _get_last_name(self):
        last = ''
        if (self.method == 'complex' and self._derivative_mod_four_is_zero or
                self._complex_high_order and
                self._derivative_mod_four_is_three):
            last = '_higher'
        return last

    @property
    def function(self):
        first = '_{0!s}'.format(self.method)
        middle = self._get_middle_name()
        last = self._get_last_name()
        name = first + middle + last
        return getattr(self, name)

    @property
    def rule(self):
        """
        Return finite differencing rule.

        The rule is for a nominal unit step size, and must be scaled later
        to reflect the local step size.

        Member methods used
        -------------------
        _fd_matrix

        Member variables used
        ---------------------
        n
        order
        method
        """
        step_ratio = self.step_ratio
        method = self.method
        if method in ('multicomplex', ) or self.n == 0:
            return np.ones((1,))

        order, method_order = self.n - 1, self._method_order
        parity = self._parity(method, order, method_order)
        step = self._richardson_step()
        num_terms, ix = (order + method_order) // step, order // step
        fd_rules = FD_RULES.get((step_ratio, parity, num_terms))
        if fd_rules is None:
            fd_mat = self._fd_matrix(step_ratio, parity, num_terms)
            fd_rules = linalg.pinv(fd_mat)
            FD_RULES[(step_ratio, parity, num_terms)] = fd_rules

        if self._flip_fd_rule:
            return -fd_rules[ix]
        return fd_rules[ix]

    def apply(self, f_del, h):
        """
        Apply finite difference rule along the first axis.
        """

        fd_rule = self.rule

        ne = h.shape[0]
        nr = fd_rule.size - 1
        _assert(nr < ne, 'num_steps ({0:d}) must  be larger than '
                '({1:d}) n + order - 1 = {2:d} + {3:d} -1'
                ' ({4:s})'.format(ne, nr+1, self.n, self.order, self.method))
        f_diff = convolve(f_del, fd_rule[::-1], axis=0, origin=nr // 2)

        der_init = f_diff / (h ** self.n)
        ne = max(ne - nr, 1)
        return der_init[:ne], h[:ne]


class DifferenceFunction(LogRule):
    """

    Derivative(fun, step, diff)
    Examples
    --------
    >>> fd_rule = DifferenceFunction(n=2, method='forward', order=4, step_ratio=2.0)
    >>> h = 0.002*(1./fd_rule.step_ratio)**np.arange(6)
    >>> x0 = 1.
    >>> f = np.exp
    >>> f_x0 = f(x0)

    >>> f_del = fd_rule.function(f, f_x0, x0, h)
    >>> fder, h = fd_rule.apply(f_del, h)
    >>> np.allclose(fder, f(x0))
    True
    """

    @staticmethod
    def _central_even(f, f_x0i, x0i, h):
        return (f(x0i + h) + f(x0i - h)) / 2.0 - f_x0i

    @staticmethod
    def _central(f, f_x0i, x0i, h):
        return (f(x0i + h) - f(x0i - h)) / 2.0

    @staticmethod
    def _forward(f, f_x0i, x0i, h):
        return f(x0i + h) - f_x0i

    @staticmethod
    def _backward(f, f_x0i, x0i, h):
        return f_x0i - f(x0i - h)

    @staticmethod
    def _complex(f, fx, x, h):
        return f(x + 1j * h).imag

    @staticmethod
    def _complex_odd(f, fx, x, h):
        ih = h * _SQRT_J
        return ((_SQRT_J / 2.) * (f(x + ih) - f(x - ih))).imag

    @staticmethod
    def _complex_odd_higher(f, fx, x, h):
        ih = h * _SQRT_J
        return ((3 * _SQRT_J) * (f(x + ih) - f(x - ih))).real

    @staticmethod
    def _complex_even(f, fx, x, h):
        ih = h * _SQRT_J
        return (f(x + ih) + f(x - ih)).imag

    @staticmethod
    def _complex_even_higher(f, fx, x, h):
        ih = h * _SQRT_J
        return 12.0 * (f(x + ih) + f(x - ih) - 2 * fx).real

    @staticmethod
    def _multicomplex(f, fx, x, h):
        z = Bicomplex(x + 1j * h, 0)
        return Bicomplex.__array_wrap__(f(z)).imag

    @staticmethod
    def _multicomplex2(f, fx, x, h):
        z = Bicomplex(x + 1j * h, h)
        return Bicomplex.__array_wrap__(f(z)).imag12


if __name__ == '__main__':
    from numdifftools.testing import test_docstrings
    test_docstrings()
