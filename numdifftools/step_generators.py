import numpy as np
from numdifftools.extrapolation import EPS
_EPS = EPS


def make_exact(h):
    """Make sure h is an exact representable number

    This is important when calculating numerical derivatives and is
    accomplished by adding 1.0 and then subtracting 1.0.
    """
    return (h + 1.0) - 1.0


def valarray(shape, value=np.NaN, typecode=None):
    """Return an array of all value."""
    if typecode is None:
        typecode = bool
    out = np.ones(shape, dtype=typecode) * value

    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


def nom_step(x=None):
    """Return nominal step"""
    if x is None:
        return 1.0
    return np.maximum(np.log1p(np.abs(x)), 1.0)


def default_base_step(x, scale, epsilon=None):
    if epsilon is None:
        h = _EPS ** (1. / scale) * nom_step(x)
    else:
        h = valarray(x.shape, value=epsilon)
    return h


def default_scale(method='forward', n=1, order=2):
    # is_odd = (n % 2) == 1
    high_order = int(n > 1 or order >= 4)
    order2 = max(order // 2 - 1, 0)
    n4 = n // 4
    return (dict(multicomplex=1.35, complex=1.35).get(method, 2.5) +
            int((n - 1)) * dict(multicomplex=0, complex=0.0).get(method, 1.3) +
            order2 * dict(central=3, forward=2, backward=2).get(method, 0) +
            # is_odd * dict(complex=2.65*int(n//2)).get(method, 0) +
            (n % 4 == 1) * high_order * dict(complex=3.65 + n4 * (5 + 1.5**n4)
                                             ).get(method, 0) +
            (n % 4 == 3) * dict(complex=3.65 * 2 + n4 * (5 + 2.1**n4)
                                ).get(method, 0) +
            (n % 4 == 2) * dict(complex=3.65 + n4 * (5 + 1.7**n4)
                                ).get(method, 0) +
            (n % 4 == 0) * dict(complex=(n // 4) * (10 + 1.5 * int(n > 10))
                                ).get(method, 0))


class _StepGenerator(object):
    def __repr__(self):
        class_name = self.__class__.__name__
        kwds = ['{0!s}={1!s}'.format(name, str(getattr(self, name)))
                for name in self.__dict__.keys()]
        return """{0!s}({1!s})""".format(class_name, ','.join(kwds))

    def _default_scale(self, method, n, order):
        scale = self.scale
        if scale is None:
            scale = default_scale(method, n, order)
        return scale

    def __call__(self, x, method='forward', n=1, order=2):
        steps, delta = self._steps(x, method, n, order)
        for step in steps:
            h = delta * step
            if (np.abs(h) > 0).all():
                yield h


class MinStepGenerator(_StepGenerator):

    """
    Generates a sequence of steps

    where steps = base_step * step_ratio ** (np.arange(num_steps) + offset)

    Parameters
    ----------
    base_step : float, array-like, optional
        Defines the base step, if None, then base_step is set to
        EPS**(1/scale)*max(log(1+|x|), 1) where x is supplied at runtime
        through the __call__ method.
    step_ratio : real scalar, optional, default 2
        Ratio between sequential steps generated.
        Note: Ratio > 1
        If None then step_ratio is 2 for n=1 otherwise step_ratio is 1.6
    num_steps : scalar integer, optional, default  n + order - 1 + num_extrap
        defines number of steps generated. It should be larger than
        n + order - 1
    offset : real scalar, optional, default 0
        offset to the base step
    scale : real scalar, optional
        scale used in base step. If not None it will override the default
        computed with the default_scale function.
    """

    def __init__(self, base_step=None, step_ratio=2, num_steps=None,
                 offset=0, scale=None, num_extrap=0, use_exact_steps=True,
                 check_num_steps=True):
        self.base_step = base_step
        self.num_steps = num_steps
        self.step_ratio = step_ratio
        self.offset = offset
        self.scale = scale
        self.check_num_steps = check_num_steps
        self.use_exact_steps = use_exact_steps
        self.num_extrap = num_extrap

    def _default_base_step(self, xi, method, n, order=2):
        scale = self._default_scale(method, n, order)
        base_step = default_base_step(xi, scale, self.base_step)
        if self.use_exact_steps:
            base_step = make_exact(base_step)
        return base_step

    @staticmethod
    def _min_num_steps(method, n, order):
        num_steps = int(n + order - 1)

        if method in ['central', 'central2', 'complex', 'multicomplex']:
            step = 2
            if method == 'complex':
                step = 4 if n > 2 or order >= 4 else 2
            num_steps = num_steps // step
        return max(num_steps, 1)

    def _default_num_steps(self, method, n, order):
        min_num_steps = self._min_num_steps(method, n, order)
        if self.num_steps is not None:
            num_steps = int(self.num_steps)
            if self.check_num_steps:
                num_steps = max(num_steps, min_num_steps)
            return num_steps
        return min_num_steps + int(self.num_extrap)

    def _default_step_ratio(self, n):
        if self.step_ratio is None:
            step_ratio = {1: 2.0}.get(n, 1.6)
        else:
            step_ratio = float(self.step_ratio)
        if self.use_exact_steps:
            step_ratio = make_exact(step_ratio)
        return step_ratio

    def _steps(self, x, method='central', n=1, order=2):
        xi = np.asarray(x)
        base_step = self._default_base_step(xi, method, n, order)
        step_ratio = self._default_step_ratio(n)

        num_steps = self._default_num_steps(method, n, order)
        offset = self.offset
        steps = step_ratio ** (np.arange(num_steps-1, -1, -1) + offset)
        return steps, base_step

    def __call__(self, x, method='central', n=1, order=2):
        return super(MinStepGenerator, self).__call__(x, method, n, order)


class MinMaxStepGenerator(_StepGenerator):
    """
    Generates a sequence of steps

    where
        steps = logspace(log10(step_min), log10(step_max), num_steps)

    Parameters
    ----------
    step_min : float, array-like, optional
       Defines the minimim step. Default value is:
           EPS**(1/scale)*max(log(1+|x|), 1)
       where x and scale are supplied at runtime through the __call__ method.
    step_max : real scalar, optional
        maximum step generated. Default value is:
            exp(log(step_min) * scale / (scale + 1.5))
    num_steps : scalar integer, optional
        defines number of steps generated.
    scale : real scalar, optional
        scale used in base step. If set to a value it will override the scale
        supplied at runtime.
    """

    def __init__(self, step_min=None, step_max=None, num_steps=10, scale=None,
                 num_extrap=0, use_exact_steps=True):
        self.step_min = step_min
        self.num_steps = num_steps
        self.step_max = step_max
        self.scale = scale
        self.num_extrap = num_extrap
        self.use_exact_steps = use_exact_steps

    def _steps(self, x, method, n, order):
        xi = np.asarray(x)
        scale = self._default_scale(method, n, order)

        step_min, step_max = self.step_min, self.step_max
        base_step = default_base_step(xi, scale, step_min)
        if step_min is None:
            step_min = (10 * EPS) ** (1. / scale)
        if step_max is None:
            step_max = np.exp(np.log(step_min) * scale / (scale + 1.5))
        steps = np.logspace(0, np.log10(step_max) - np.log10(step_min),
                            self.num_steps)
        if self.use_exact_steps:
            return make_exact(steps), make_exact(base_step)
        return steps, base_step


class MaxStepGenerator(MinStepGenerator):
    """
    Generates a sequence of steps

    where
        steps = base_step * step_ratio ** (-np.arange(num_steps) + offset)
        base_step = step_max * step_nom

    Parameters
    ----------
    max_step : float, array-like, optional default 2
       Defines the maximum step
    step_ratio : real scalar, optional, default 2
        Ratio between sequential steps generated.
        Note: Ratio > 1
    num_steps : scalar integer, optional, default  n + order - 1 + num_extrap
        defines number of steps generated. It should be larger than
        n + order - 1
    step_nom :  default maximum(log1p(abs(x)), 1)
        Nominal step.
    offset : real scalar, optional, default 0
        offset to the base step: max_step * nom_step
    """

    def __init__(self, step_max=2.0, step_ratio=2.0, num_steps=15,
                 step_nom=None, offset=0, num_extrap=0,
                 use_exact_steps=False, check_num_steps=True):
        self.base_step = None
        self.step_max = step_max
        self.step_ratio = step_ratio
        self.num_steps = num_steps
        self.step_nom = step_nom
        self.offset = offset
        self.num_extrap = num_extrap
        self.check_num_steps = check_num_steps
        self.use_exact_steps = use_exact_steps

    def _default_step_nom(self, x):
        if self.step_nom is None:
            return nom_step(x)
        return valarray(x.shape, value=self.step_nom)

    def _default_base_step(self, xi, method, n, order=1):
        base_step = self.base_step
        if base_step is None:
            base_step = self.step_max * self._default_step_nom(xi)
        if self.use_exact_steps:
            base_step = make_exact(base_step)
        return base_step

    def _steps(self, x, method, n, order):
        xi = np.asarray(x)

        base_step = self._default_base_step(xi, method, n)
        step_ratio = self._default_step_ratio(n)
        num_steps = self._default_num_steps(method, n, order)
        offset = self.offset
        steps = step_ratio ** (-np.arange(num_steps) + offset)
        return steps, base_step

    def __call__(self, x, method='forward', n=1, order=None):
        return super(MaxStepGenerator, self).__call__(x, method, n, order)
