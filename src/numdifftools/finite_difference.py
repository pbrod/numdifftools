"""
Finite difference methods module.
"""
from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from numpy import linalg
from scipy import special
from numdifftools.extrapolation import convolve
from numdifftools.multicomplex import Bicomplex

_SQRT_J = (1j + 1.0) / np.sqrt(2.0)  # = 1j**0.5

# step_ratio, parity, nterms
FD_RULES = {}
# (2.0, 1, 1): array([[1.]]),
# (2.0, 1, 2): array([[-0.333333333333333333333, 2.666666666666666666666666666],
#                     [8., -16.]]),
# (2.0, 1, 3): array([[2.22222222222222222e-02, -8.8888888888889e-01, 5.6888888888888889e+00],
#                     [-2.666666666666667e+00, 9.0666666666666667e+01, -1.7066666666666667e+02],
#                     [1.7066666666666667e+02, -1.7066666666666667e+03, 2.7306666666666667e+03]]),
# (2.0, 0, 2): array([[-1., 4.],
#                     [4., -8.]]),
# (2.0, 0, 4): array([[-4.76190476e-02, 1.33333333e+00, -1.06666667e+01, 2.43809524e+01],
#                     [1.33333333e+00, -3.46666667e+01, 2.34666667e+02, -3.41333333e+02],
#                     [-1.60000000e+01, 3.52000000e+02, -1.66400000e+03, 2.04800000e+03],
#                     [7.31428571e+01, -1.02400000e+03, 4.09600000e+03, -4.68114286e+03]])}

# @PydevIgnore


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def make_exact(h):
    """Make sure h is an exact representable number

    This is important when calculating numerical derivatives and is
    accomplished by adding 1.0 and then subtracting 1.0.
    """
    return (h + 1.0) - 1.0


class DifferenceFunctions(object):
    """
    Class defining difference functions

    Notes
    -----
    The d
    """
    # pylint: disable=unused-argument
    @staticmethod
    def _central_even(f, f_x0i, x0i, h):
        return (f(x0i + h) + f(x0i - h)) / 2.0 - f_x0i

    @staticmethod
    def _central(f, f_x0i, x0i, h):  # @UnusedVariable
        return (f(x0i + h) - f(x0i - h)) / 2.0

    @staticmethod
    def _forward(f, f_x0i, x0i, h):
        return f(x0i + h) - f_x0i

    @staticmethod
    def _backward(f, f_x0i, x0i, h):
        return f_x0i - f(x0i - h)

    @staticmethod
    def _complex(f, f_x, x, h):  # @UnusedVariable
        return f(x + 1j * h).imag

    @staticmethod
    def _complex_odd(f, f_x, x, h):
        i_h = h * _SQRT_J
        return ((_SQRT_J / 2.) * (f(x + i_h) - f(x - i_h))).imag

    @staticmethod
    def _complex_odd_higher(f, f_x, x, h):
        i_h = h * _SQRT_J
        return ((3 * _SQRT_J) * (f(x + i_h) - f(x - i_h))).real

    @staticmethod
    def _complex_even(f, f_x, x, h):
        i_h = h * _SQRT_J
        return (f(x + i_h) + f(x - i_h)).imag

    @staticmethod
    def _complex_even_higher(f, f_x, x, h):
        i_h = h * _SQRT_J
        return 12.0 * (f(x + i_h) + f(x - i_h) - 2 * f_x).real

    @staticmethod
    def _multicomplex(f, f_x, x, h):
        z = Bicomplex(x + 1j * h, 0)
        return Bicomplex.__array_wrap__(f(z)).imag

    @staticmethod
    def _multicomplex2(f, f_x, x, h):
        z = Bicomplex(x + 1j * h, h)
        return Bicomplex.__array_wrap__(f(z)).imag12


class JacobianDifferenceFunctions(object):
    """Class defining Jacobian difference functions"""
    # pylint: disable=unused-argument
    @staticmethod
    def increments(n, h):
        """Returns Jacobian steps"""
        e_i = np.zeros(np.shape(h), float)
        for k in range(n):
            e_i[k] = h[k]
            yield e_i
            e_i[k] = 0

    @staticmethod
    def _central(f, f_x, x, h):
        n = len(x)
        steps = JacobianDifferenceFunctions.increments(n, h)
        return np.array([(f(x + hi) - f(x - hi)) / 2.0 for hi in steps])

    @staticmethod
    def _central_even(f, f_x, x, h):
        n = len(x)
        steps = JacobianDifferenceFunctions.increments(n, h)
        return np.array([(f(x + hi) + f(x - hi)) / 2.0 - f_x for hi in steps])

    @staticmethod
    def _backward(f, f_x, x, h):
        n = len(x)
        steps = JacobianDifferenceFunctions.increments(n, h)
        return np.array([f_x - f(x - hi) for hi in steps])

    @staticmethod
    def _forward(f, f_x, x, h):
        n = len(x)
        steps = JacobianDifferenceFunctions.increments(n, h)
        return np.array([f(x + hi) - f_x for hi in steps])

    @staticmethod
    def _complex(f, f_x, x, h):
        n = len(x)
        steps = JacobianDifferenceFunctions.increments(n, h)
        return np.array([f(x + 1j * ih).imag for ih in steps])

    @staticmethod
    def _complex_even(f, f_x, x, h):
        n = len(x)
        j_1 = _SQRT_J
        steps = JacobianDifferenceFunctions.increments(n, h)
        return np.array([(f(x + j_1*ih) + f(x - j_1*ih)).imag for ih in steps])

    @staticmethod
    def _complex_odd(f, f_x, x, h):
        n = len(x)
        j_1 = _SQRT_J
        steps = JacobianDifferenceFunctions.increments(n, h)
        return np.array([((j_1 / 2.) * (f(x + j_1 * ih) - f(x - j_1 * ih))).imag for ih in steps])

    @staticmethod
    def _multicomplex(f, f_x, x, h):
        n = len(x)
        steps = JacobianDifferenceFunctions.increments(n, h)
        cmplx_wrap = Bicomplex.__array_wrap__
        partials = [cmplx_wrap(f(Bicomplex(x + 1j * hi, 0))).imag for hi in steps]
        return np.array(partials)


class HessdiagDifferenceFunctions(object):
    """Class defining Hessdiag difference functions

    References
    ----------
    Ridout, M.S. (2009) Statistical applications of the complex-step method
        of numerical differentiation. The American Statistician, 63, 66-74
    """
    # pylint: disable=unused-argument
    @staticmethod
    def _central2(f, f_x, x, h):
        """Eq. 8 in Ridout (2009)."""
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + 2 * hi) + f(x - 2 * hi)
                     + 2 * f_x - 2 * f(x + hi) - 2 * f(x - hi)) / 4.0
                    for hi in increments]
        return np.array(partials)

    @staticmethod
    def _central_even(f, f_x, x, h):
        """Eq. 9 in Ridout (2009)."""
        n = len(x)
        increments = np.identity(n) * h
        partials = [(f(x + hi) + f(x - hi)) / 2.0 - f_x for hi in increments]
        return np.array(partials)

    @staticmethod
    def _backward(f, f_x, x, h):
        n = len(x)
        increments = np.identity(n) * h
        partials = [f_x - f(x - hi) for hi in increments]
        return np.array(partials)

    @staticmethod
    def _forward(f, f_x, x, h):
        n = len(x)
        increments = np.identity(n) * h
        partials = [f(x + hi) - f_x for hi in increments]
        return np.array(partials)

    @staticmethod
    def _multicomplex2(f, f_x, x, h):
        n = len(x)
        increments = np.identity(n) * h
        cmplx_wrap = Bicomplex.__array_wrap__
        partials = [cmplx_wrap(f(Bicomplex(x + 1j * hi, hi))).imag12
                    for hi in increments]
        return np.array(partials)

    @staticmethod
    def _complex_even(f, f_x, x, h):
        n = len(x)
        increments = np.identity(n) * h * (1j + 1) / np.sqrt(2)
        partials = [(f(x + hi) + f(x - hi)).imag for hi in increments]
        return np.array(partials)


class HessianDifferenceFunctions(object):
    """Class defining Hessian difference functions

    References
    ----------
    Ridout, M.S. (2009)
    "Statistical applications of the complex-step method of numerical differentiation",
    The American Statistician, 63, 66-74
    """
    # pylint: disable=unused-argument
    @staticmethod
    def _complex_even(f, f_x, x, h):
        """
        Calculate Hessian with complex-step derivative approximation

        The stepsize is the same for the complex and the finite difference part

        Eq 10 in Ridout (2009).
        """
        n = len(x)
        eee = np.diag(h)
        hess = 2. * np.outer(h, h)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + 1j * eee[i] + eee[j])
                              - f(x + 1j * eee[i] - eee[j])).imag / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _multicomplex2(f, f_x, x, h):
        """Calculate Hessian with Bicomplex-step derivative approximation"""
        n = len(x)
        eee = np.diag(h)
        hess = np.outer(h, h)
        cmplx_wrap = Bicomplex.__array_wrap__
        for i in range(n):
            for j in range(i, n):
                zph = Bicomplex(x + 1j * eee[i, :], eee[j, :])
                hess[i, j] = cmplx_wrap(f(zph)).imag12 / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _central_even(f, f_x, x, h):
        """Eq 9 in Ridout (2009)."""
        n = len(x)
        eee = np.diag(h)
        dtype = np.result_type(f_x, float)  # make sure it is at least float64
        hess = np.empty((n, n), dtype=dtype)
        np.outer(h, h, out=hess)
        for i in range(n):
            e_i = eee[i, :]
            hess[i, i] = (f(x + 2 * e_i) - 2 * f_x + f(x - 2 * e_i)) / (4. * hess[i, i])
            for j in range(i + 1, n):
                e_j = eee[j, :]
                hess[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j)
                              - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4. * hess[j, i])
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _central2(f, f_x, x, h):
        """Eq. 8 in Ridout (2009)"""
        n = len(x)
        eee = np.diag(h)
        dtype = np.result_type(f_x, float)
        f_xpe = np.empty(n, dtype=dtype)
        f_xme = np.empty(n, dtype=dtype)
        for i in range(n):
            f_xpe[i] = f(x + eee[i])
            f_xme[i] = f(x - eee[i])

        hess = np.empty((n, n), dtype=dtype)
        np.outer(h, h, out=hess)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + eee[i, :] + eee[j, :])
                              + f(x - eee[i, :] - eee[j, :])
                              - f_xpe[i] - f_xpe[j] + f_x
                              - f_xme[i] - f_xme[j] + f_x) / (2 * hess[j, i])
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _forward(f, f_x, x, h):
        """Eq. 7 in Ridout (2009)"""
        n = len(x)
        eee = np.diag(h)
        dtype = np.result_type(f_x, float)
        g = np.empty(n, dtype=dtype)
        for i in range(n):
            g[i] = f(x + eee[i, :])

        hess = np.empty((n, n), dtype=dtype)
        np.outer(h, h, out=hess)
        for i in range(n):
            for j in range(i, n):
                hess[i, j] = (f(x + eee[i, :] + eee[j, :]) - g[i] - g[j] + f_x) / hess[j, i]
                hess[j, i] = hess[i, j]
        return hess

    @staticmethod
    def _backward(f, f_x, x, h):
        return HessianDifferenceFunctions._forward(f, f_x, x, -h)


class LogRule(object):
    """Log spaced finite difference rule class

    Parameters
    ----------
    n : int, optional
        Order of the derivative.
    method : {'central', 'complex', 'multicomplex', 'forward', 'backward'}
        defines the method used in the approximation
    order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.


    Examples
    --------
    >>> from numdifftools.finite_difference import LogRule
    >>> np.allclose(LogRule(n=1, method='central', order=2).rule(step_ratio=2.0), 1)
    True
    >>> np.allclose(LogRule(n=1, method='central', order=4).rule(step_ratio=2.),
    ...             [-0.33333333,  2.66666667])
    True
    >>> np.allclose(LogRule(n=1, method='central', order=6).rule(step_ratio=2.),
    ...             [ 0.02222222, -0.88888889,  5.68888889])
    True

    >>> np.allclose(LogRule(n=1, method='forward', order=2).rule(step_ratio=2.), [-1.,  4.])
    True

    >>> np.allclose(LogRule(n=1, method='forward', order=4).rule(step_ratio=2.),
    ...             [ -0.04761905,   1.33333333, -10.66666667,  24.38095238])
    True
    >>> np.allclose(LogRule(n=1, method='forward', order=6).rule(step_ratio=2.),
    ...    [ -1.02406554e-04,   1.26984127e-02,  -5.07936508e-01,
    ...       8.12698413e+00,  -5.20126984e+01,   1.07381055e+02])
    True
    >>> step_ratio=2.0
    >>> fd_rule = LogRule(n=2, method='forward', order=4)
    >>> h = 0.002*(1./step_ratio)**np.arange(6)
    >>> x0 = 1.
    >>> f = np.exp
    >>> f_x0 = f(x0)
    >>> f_del = f(x0+h) - f_x0  # forward difference
    >>> f_del = fd_rule.diff(f, f_x0, x0, h)  # or alternatively
    >>> fder, h, shape = fd_rule.apply(f_del, h, step_ratio)
    >>> np.allclose(fder, f(x0))
    True

    """
    _difference_functions = DifferenceFunctions()

    def __init__(self, n=1, method='central', order=2):
        self.n = n
        self.method = method
        self.order = order

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
    def eval_first_condition(self):
        """True if f(x0) needs to be evaluated given the differentiation method."""
        even_derivative = self._even_derivative
        return ((even_derivative and self.method in ('central', 'central2')) or
                self.method in ['forward', 'backward'] or
                self.method == 'complex' and self._derivative_mod_four_is_zero)

    @property
    def _complex_high_order(self):
        return self.method == 'complex' and (self.n > 1 or self.order >= 4)

    @property
    def richardson_step(self):
        """The step between exponents in the error polynomial of the Richardson extrapolation."""
        complex_step = 4 if self._complex_high_order else 2
        return dict(central=2,
                    central2=2,
                    complex=complex_step,
                    multicomplex=2).get(self.method, 1)

    @property
    def method_order(self):
        """The leading order of the truncation error of the Richardson extrapolation."""
        step = self.richardson_step
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
        c_0 = [1.0, 1.0, 1.0, 2.0, 24.0, 1.0, 6.0][parity]
        c = c_0 / special.factorial(np.arange(offset, step * nterms + offset, step))
        [i, j] = np.ogrid[0:nterms, 0:nterms]
        return np.atleast_2d(c[j] * inv_sr ** (i * (step * j + offset)))

    @property
    def _flip_fd_rule(self):
        return ((self._even_derivative and (self.method == 'backward'))
                or (self.method == 'complex' and (self.n % 8 in [3, 4, 5, 6])))

    @property
    def _multicomplex_middle_name(self):
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
        return self._multicomplex_middle_name

    def _get_last_name(self):
        last = ''
        if (self.method == 'complex' and self._derivative_mod_four_is_zero or
                self._complex_high_order and
                self._derivative_mod_four_is_three):
            last = '_higher'
        return last

    @property
    def diff(self):
        "The difference function"
        first = '_{0!s}'.format(self.method)
        middle = self._get_middle_name()
        last = self._get_last_name()
        name = first + middle + last
        return getattr(self._difference_functions, name)

    def rule(self, step_ratio=2.0):
        """
        Return finite differencing rule.

        Parameters
        ----------
        step_ratio : real scalar, optional, default 2.0
            Ratio between sequential steps generated.

        Notes
        -----
        The rule is for a nominal unit step size, and must be scaled later
        to reflect the local step size.

        Member method used:  _fd_matrix

        Member variables used:
        n
        order
        method
        """
        method = self.method
        if method in ('multicomplex',) or self.n == 0:
            return np.ones((1,))
        step_ratio = make_exact(step_ratio)
        order, method_order = self.n - 1, self.method_order
        parity = self._parity(method, order, method_order)
        step = self.richardson_step
        num_terms = (order + method_order) // step
        fd_rules = FD_RULES.get((step_ratio, parity, num_terms))
        if fd_rules is None:
            fd_mat = self._fd_matrix(step_ratio, parity, num_terms)
            fd_rules = linalg.pinv(fd_mat)
            FD_RULES[(step_ratio, parity, num_terms)] = fd_rules

        rule_index = order // step
        if self._flip_fd_rule:
            return -fd_rules[rule_index]
        return fd_rules[rule_index]

    @staticmethod
    def _vstack(sequence, steps):
        original_shape = np.shape(sequence[0])
        f_del = np.vstack([np.ravel(r) for r in sequence])
        one = np.ones(original_shape)
        h = np.vstack([np.ravel(one * step) for step in steps])
        _assert(f_del.size == h.size, 'fun did not return data of correct '
                'size (it must be vectorized)')
        return f_del, h, original_shape

    def apply(self, sequence, steps, step_ratio=2.0):
        """
        Apply finite difference rule along the first axis.

        Return derivative estimates of fun at x0 for a sequence of stepsizes h

        Parameters
        ----------
        sequence: finite differences
        steps: steps

        """
        f_del, h, original_shape = self._vstack(sequence, steps)
        der_init, h = self._apply(f_del, h, step_ratio)
        return der_init, h, original_shape

    def _apply(self, f_del, h, step_ratio=2.0):
        """
        Apply finite difference rule along the first axis.

        Return derivative estimates of fun at x0 for a sequence of stepsizes h

        Parameters
        ----------
        f_del: finite differences
        h: steps

        """
        fd_rule = self.rule(step_ratio)

        num_steps = h.shape[0]
        n_r = fd_rule.size - 1
        _assert(n_r < num_steps, 'num_steps ({0:d}) must  be larger than '
                '({1:d}) n + order - 1 = {2:d} + {3:d} -1'
                ' ({4:s})'.format(num_steps, n_r + 1, self.n, self.order, self.method))
        f_diff = convolve(f_del, fd_rule[::-1], axis=0, origin=n_r // 2)

        der_init = f_diff / (h ** self.n)
        num_steps = max(num_steps - n_r, 1)
        return der_init[:num_steps], h[:num_steps]


class LogJacobianRule(LogRule):
    """Log spaced finite difference Jacobian rule class

    Parameters
    ----------
    n : 1
        Order of the derivative.
    method : {'central', 'complex', 'multicomplex', 'forward', 'backward'}
        defines the method used in the approximation
    order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.

    Examples
    --------
    >>> from numdifftools.finite_difference import LogJacobianRule as Rule
    >>> np.allclose(Rule(n=1, method='central', order=2).rule(step_ratio=2.0), 1)
    True
    >>> np.allclose(Rule(n=1, method='central', order=4).rule(step_ratio=2.),
    ...             [-0.33333333,  2.66666667])
    True
    >>> np.allclose(Rule(n=1, method='central', order=6).rule(step_ratio=2.),
    ...             [ 0.02222222, -0.88888889,  5.68888889])
    True

    >>> np.allclose(Rule(n=1, method='forward', order=2).rule(step_ratio=2.), [-1.,  4.])
    True

    >>> np.allclose(Rule(n=1, method='forward', order=4).rule(step_ratio=2.),
    ...             [ -0.04761905,   1.33333333, -10.66666667,  24.38095238])
    True
    >>> np.allclose(Rule(n=1, method='forward', order=6).rule(step_ratio=2.),
    ...    [ -1.02406554e-04,   1.26984127e-02,  -5.07936508e-01,
    ...       8.12698413e+00,  -5.20126984e+01,   1.07381055e+02])
    True
    >>> step_ratio=2.0
    >>> fd_rule = Rule(n=1, method='forward', order=4)
    >>> steps = 0.002*(1./step_ratio)**np.arange(6)

    >>> x0 = np.atleast_1d(1.)
    >>> f = np.exp
    >>> f_x0 = f(x0)
    >>> f_del = [f(x0+h) - f_x0  for h in steps] # forward difference
    >>> f_del = [fd_rule.diff(f, f_x0, x0, h) for h in steps[:, None]]  # or alternatively
    >>> fder, h, shape = fd_rule.apply(f_del, steps, step_ratio)
    >>> np.allclose(fder, f(x0))
    True

    """
    _difference_functions = JacobianDifferenceFunctions()

    # n = property(fget=lambda cls: 1, fset=lambda cls, n: None)

    @staticmethod
    def _atleast_2d(original_shape, ndim):
        if ndim == 1:
            original_shape = (1,) + tuple(original_shape)
        return tuple(original_shape)

    def _vstack(self, sequence, steps):
        original_shape = list(np.shape(np.atleast_1d(sequence[0])))
        ndim = len(original_shape)
        if sum(original_shape) == ndim:
            f_del = np.vstack(sequence)
            h = np.vstack(steps)
        else:
            axes = [0, 1, 2][:ndim]
            axes[:2] = axes[1::-1]
            original_shape[:2] = original_shape[1::-1]

            f_del = np.vstack([np.atleast_1d(r).transpose(axes).ravel() for r in sequence])
            h = np.vstack([np.atleast_1d(r).transpose(axes).ravel() for r in steps])
        _assert(f_del.size == h.size, 'fun did not return data of correct '
                'size (it must be vectorized)')
        return f_del, h, self._atleast_2d(original_shape, ndim)


class LogHessdiagRule(LogRule):
    """Log spaced finite difference Hessdiag rule class

    Parameters
    ----------
    n : 2
        Order of the derivative.
    method : {'central', 'complex', 'multicomplex', 'forward', 'backward'}
        defines the method used in the approximation
    order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.

    Examples
    --------
    >>> import numpy as np
    >>> from numdifftools.finite_difference import LogHessdiagRule as Rule

    >>> np.allclose(Rule(method='central', order=2).rule(step_ratio=2.0), 2)
    True
    >>> np.allclose(Rule(method='central', order=4).rule(step_ratio=2.),
    ...             [-0.66666667, 10.66666667])
    True
    >>> np.allclose(Rule(method='central', order=6).rule(step_ratio=2.),
    ...             [ 4.44444444e-02, -3.55555556e+00,  4.55111111e+01])
    True
    >>> np.allclose(Rule(method='forward', order=2).rule(step_ratio=2.), [ -4.,  40., -64.])
    True
    >>> np.allclose(Rule(method='forward', order=4).rule(step_ratio=2.),
    ...          [-1.90476190e-01,  1.10476190e+01, -1.92000000e+02,
    ...          1.12152381e+03, -1.56038095e+03])
    True
    >>> np.allclose(Rule(method='forward', order=6).rule(step_ratio=2.),
    ...    [-4.09626216e-04,  1.02406554e-01, -8.33015873e+00,  2.76317460e+02,
    ...     -3.84893968e+03,  2.04024004e+04, -2.74895500e+04])
    True
    >>> step_ratio=2.0
    >>> fd_rule = Rule(method='forward', order=4)
    >>> steps = 0.002*(1./step_ratio)**np.arange(6)
    >>> x0 = np.array([0., 0.])
    >>> f = lambda xy : np.cos(xy[0]-xy[1])
    >>> f_x0 = f(x0)
    >>> f_del = [f(x0+h) - f_x0  for h in steps] # forward difference
    >>> f_del = [fd_rule.diff(f, f_x0, x0, h) for h in steps]  # or alternatively
    >>> fder, h, shape = fd_rule.apply(f_del, steps, step_ratio)

    >>> np.allclose(fder, [[-1., -1.], [-1., -1.]])
    True

    """
    _difference_functions = HessdiagDifferenceFunctions()

    n = property(fget=lambda cls: 2, fset=lambda cls, n: None)


class LogHessianRule(LogRule):
    """Log spaced finite difference Hessian rule class

    Parameters
    ----------
    n : 2
        Order of the derivative.
    method : {'central', 'complex', 'multicomplex', 'forward', 'backward'}
        defines the method used in the approximation
    order : int, optional
        defines the order of the error term in the Taylor approximation used.
        For 'central' and 'complex' methods, it must be an even number.
    """
    _difference_functions = HessianDifferenceFunctions()

    n = property(fget=lambda cls: 2, fset=lambda cls, unused_n: None)

    @property
    def order(self):
        """The order of the error term in the Taylor approximation used"""
        return dict(backward=1, forward=1).get(self.method, 2)

    @order.setter
    def order(self, order):
        valid_order = self.order

        if order is not None and order != valid_order:
            msg = 'Can not change order to {}! The only valid order is {} for method={}.'
            warnings.warn(msg.format(order, valid_order, self.method))

    @property
    def _complex_high_order(self):
        return False

    def apply(self, sequence, steps, step_ratio=2.0):
        """
        Apply finite difference rule along the first axis.

        Return derivative estimates of fun at x0 for a sequence of stepsizes h

        Parameters
        ----------
        sequence: finite differences
        steps: steps

        """
        f_del, h, original_shape = self._vstack(sequence, steps)
        return f_del, h, original_shape


if __name__ == '__main__':
    from numdifftools.testing import test_docstrings
    test_docstrings()
