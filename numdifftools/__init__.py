import pkg_resources
from .info import __doc__
from .core import *
from numpy.testing import Tester

__version__ = pkg_resources.get_distribution(__name__).version

test = Tester().test
