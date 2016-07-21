import pkg_resources
from .info import __doc__
from .core import *
from . import extrapolation, limits, step_generators

from numpy.testing import Tester
try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    __version__ = 'unknown'


test = Tester(raise_warnings="release").test
