"""
Step generators module
"""
from __future__ import absolute_import, division
from collections import namedtuple
import numpy as np
from numdifftools.extrapolation import EPS

_STATE = namedtuple('State', ['x', 'method', 'n', 'order'])
__all__ = ('one_step', 'make_exact', 'get_nominal_step', 'get_base_step',
           'default_scale', 'MinStepGenerator', 'MaxStepGenerator',
           'BasicMaxStepGenerator', 'BasicMinStepGenerator')


def make_exact(h):
    """Make sure h is an exact representable number

    This is important when calculating numerical derivatives and is
    accomplished by adding 1.0 and then subtracting 1.0.
    """
    return (h + 1.0) - 1.0


def get_nominal_step(x=None):
    """Return nominal step"""
    if x is None:
        return 1.0
    return np.log(1.718281828459045 + np.abs(x)).clip(min=1)


def get_base_step(scale):
    """Return base_step = EPS ** (1. / scale)"""
    return EPS ** (1. / scale)


def default_scale(method='forward', n=1, order=2):
    """Returns good scale for MinStepGenerator"""
    high_order = int(n > 1 or order >= 4)
    order2 = max(order // 2 - 1, 0)
    n_4 = n // 4
    n_mod_4 = n % 4
    c = ([n_4 * (10 + 1.5 * int(n > 10)),
          3.65 + n_4 * (5 + 1.5 ** n_4),
          3.65 + n_4 * (5 + 1.7 ** n_4),
          7.30 + n_4 * (5 + 2.1 ** n_4)][n_mod_4]) if high_order else 0

    return (dict(multicomplex=1.06, complex=1.06 + c).get(method, 2.5) +
            int(n - 1) * dict(multicomplex=0, complex=0.0).get(method, 1.3) +
            order2 * dict(central=3, forward=2, backward=2).get(method, 0))


class BasicMaxStepGenerator(object):

    """
    Generates a sequence of steps of decreasing magnitude

    where
        steps = base_step * step_ratio ** (-i + offset)

    for i=0, 1,.., num_steps-1.


    Parameters
    ----------
    base_step : float, array-like.
       Defines the start step, i.e., maximum step
    step_ratio : real scalar.
        Ratio between sequential steps generated.  Note: Ratio > 1
    num_steps : scalar integer.
        defines number of steps generated.
    offset : real scalar, optional, default 0
        offset to the base step

    Examples
    --------
    >>> from numdifftools.step_generators import BasicMaxStepGenerator
    >>> step_gen = BasicMaxStepGenerator(base_step=2.0, step_ratio=2,
    ...                                  num_steps=4)
    >>> [s for s in step_gen()]
    [2.0, 1.0, 0.5, 0.25]

    """

    _sign = -1

    def __init__(self, base_step, step_ratio, num_steps, offset=0):
        self.base_step = base_step
        self.step_ratio = step_ratio
        self.num_steps = num_steps
        self.offset = offset

    def _range(self):
        return range(self.num_steps)

    def __call__(self):
        base_step, step_ratio = self.base_step, self.step_ratio
        sgn, offset = self._sign, self.offset
        for i in self._range():
            step = base_step * step_ratio ** (sgn * i + offset)
            if (np.abs(step) > 0).all():
                yield step


class BasicMinStepGenerator(BasicMaxStepGenerator):

    """
    Generates a sequence of steps of decreasing magnitude

    where
        steps = base_step * step_ratio ** (i + offset), i=num_steps-1,... 1, 0.


    Parameters
    ----------
    base_step : float, array-like.
       Defines the end step, i.e., minimum step
    step_ratio : real scalar.
        Ratio between sequential steps generated.  Note: Ratio > 1
    num_steps : scalar integer.
        defines number of steps generated.
    offset : real scalar, optional, default 0
        offset to the base step

    Examples
    --------
    >>> from numdifftools.step_generators import BasicMinStepGenerator
    >>> step_gen = BasicMinStepGenerator(base_step=0.25, step_ratio=2,
    ...                                  num_steps=4)
    >>> [s for s in step_gen()]
    [2.0, 1.0, 0.5, 0.25]

    """

    _sign = 1

    def _range(self):
        return range(self.num_steps - 1, -1, -1)


class MinStepGenerator(object):

    """
    Generates a sequence of steps

    where
        steps = step_nom * base_step * step_ratio ** (i + offset)
    for  i = num_steps-1,... 1, 0.

    Parameters
    ----------
    base_step : float, array-like, optional
        Defines the minimum step, if None, the value is set to EPS**(1/scale)
    step_ratio : real scalar, optional, default 2
        Ratio between sequential steps generated.
        Note: Ratio > 1
        If None then step_ratio is 2 for n=1 otherwise step_ratio is 1.6
    num_steps : scalar integer, optional, default  min_num_steps + num_extrap
        defines number of steps generated. It should be larger than
        min_num_steps = (n + order - 1) / fact where fact is 1, 2 or 4
        depending on differentiation method used.
    step_nom :  default maximum(log(exp(1)+|x|), 1)
        Nominal step where x is supplied at runtime through the __call__ method.
    offset : real scalar, optional, default 0
        offset to the base step
    num_extrap : scalar integer, default 0
        number of points used for extrapolation
    check_num_steps : boolean, default True
        If True make sure num_steps is larger than the minimum required steps.
    use_exact_steps : boolean, default True
        If true make sure exact steps are generated
    scale : real scalar, optional
        scale used in base step. If not None it will override the default
        computed with the default_scale function.
    """

    _step_generator = BasicMinStepGenerator

    def __init__(self, base_step=None, step_ratio=None, num_steps=None, step_nom=None,
                 offset=0, num_extrap=0, use_exact_steps=True, check_num_steps=True, scale=None):
        self.base_step = base_step
        self.step_nom = step_nom
        self.num_steps = num_steps
        self.step_ratio = step_ratio
        self.offset = offset
        self.num_extrap = num_extrap
        self.check_num_steps = check_num_steps
        self.use_exact_steps = use_exact_steps
        self.scale = scale

        self._state = _STATE(x=np.asarray(1), method='forward', n=1, order=2)

    def __repr__(self):
        class_name = self.__class__.__name__
        kwds = ['{0!s}={1!s}'.format(name, str(getattr(self, name))) for name in self.__dict__]
        return """{0!s}({1!s})""".format(class_name, ','.join(kwds))

    @property
    def scale(self):
        """Scale used in base step."""
        if self._scale is None:
            _unused_x, method, n, order = self._state
            return default_scale(method, n, order)
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @property
    def base_step(self):
        """Base step defines the minimum or maximum step when offset==0."""
        if self._base_step is None:
            return get_base_step(self.scale)
        return self._base_step

    @base_step.setter
    def base_step(self, base_step):
        self._base_step = base_step

    @staticmethod
    def _num_step_divisor(method, n, order):
        complex_divisior = 4 if (n > 1 or order >= 4) else 2
        return dict(central=2,
                    central2=2,
                    complex=complex_divisior,
                    multicomplex=2).get(method, 1)

    @property
    def min_num_steps(self):
        """Minimum number of steps required given the differentiation method and order."""
        _unused_x, method, n, order = self._state
        num_steps = int(n + order - 1)
        divisor = self._num_step_divisor(method, n, order)
        return max(num_steps // divisor, 1)

    @property
    def num_steps(self):
        """Defines number of steps generated"""
        min_num_steps = self.min_num_steps
        if self._num_steps is not None:
            num_steps = int(self._num_steps)
            if self.check_num_steps:
                num_steps = max(num_steps, min_num_steps)
            return num_steps
        return min_num_steps + int(self.num_extrap)

    @num_steps.setter
    def num_steps(self, num_steps):
        self._num_steps = num_steps

    @property
    def step_ratio(self):
        """Ratio between sequential steps generated"""
        step_ratio = self._step_ratio
        if step_ratio is None:
            step_ratio = {1: 2.0}.get(self._state.n, 1.6)
        return float(step_ratio)

    @step_ratio.setter
    def step_ratio(self, step_ratio):
        self._step_ratio = step_ratio

    @property
    def step_nom(self):
        """Nominal step"""
        x = self._state.x
        if self._step_nom is None:
            return get_nominal_step(x)
        return np.full(x.shape, fill_value=self._step_nom)

    @step_nom.setter
    def step_nom(self, step_nom):
        self._step_nom = step_nom

    def step_generator_function(self, x, method='forward', n=1, order=2):
        """Step generator function"""
        self._state = _STATE(np.asarray(x), method, n, order)
        base_step, step_ratio = self.base_step * self.step_nom, self.step_ratio
        if self.use_exact_steps:
            base_step = make_exact(base_step)
            step_ratio = make_exact(step_ratio)
        return self._step_generator(base_step=base_step,
                                    step_ratio=step_ratio,
                                    num_steps=self.num_steps,
                                    offset=self.offset)

    def __call__(self, x, method='forward', n=1, order=2):
        step_generator = self.step_generator_function(x, method, n, order)
        return step_generator()


class MaxStepGenerator(MinStepGenerator):

    """
    Generates a sequence of steps

    where
        steps = step_nom * base_step * step_ratio ** (-i + offset)
    for  i = 0, 1, ..., num_steps-1.

    Parameters
    ----------
    base_step : float, array-like, default 2.0
        Defines the maximum step, if None, the value is set to EPS**(1/scale)
    step_ratio : real scalar, optional, default 2 or 1.6
        Ratio between sequential steps generated.
        Note: Ratio > 1
        If None then step_ratio is 2 for n=1 otherwise step_ratio is 1.6
    num_steps : scalar integer, optional, default  min_num_steps + num_extrap
        defines number of steps generated. It should be larger than
        min_num_steps = (n + order - 1) / fact where fact is 1, 2 or 4
        depending on differentiation method used.
    step_nom :  default maximum(log(exp(1)+|x|), 1)
        Nominal step where x is supplied at runtime through the __call__
        method.
    offset : real scalar, optional, default 0
        offset to the base step
    num_extrap : scalar integer, default 0
        number of points used for extrapolation
    check_num_steps : boolean, default True
        If True make sure num_steps is larger than the minimum required steps.
    use_exact_steps : boolean, default True
        If true make sure exact steps are generated
    scale : real scalar, default 500
        scale used in base step.
    """

    _step_generator = BasicMaxStepGenerator

    def __init__(self, base_step=2.0, step_ratio=None, num_steps=15,
                 step_nom=None, offset=0, num_extrap=9,
                 use_exact_steps=False, check_num_steps=True, scale=500):
        super(MaxStepGenerator,
              self).__init__(base_step=base_step, step_ratio=step_ratio,
                             num_steps=num_steps, step_nom=step_nom,
                             offset=offset, num_extrap=num_extrap,
                             use_exact_steps=use_exact_steps,
                             check_num_steps=check_num_steps, scale=scale)


one_step = MinStepGenerator(num_steps=1, scale=None, step_nom=None)

if __name__ == '__main__':
    from numdifftools.testing import test_docstrings
    test_docstrings()
