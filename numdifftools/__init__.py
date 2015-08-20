from .info import __doc__
from .core import *
from ._version import get_versions
from numpy.testing import Tester

__version__ = get_versions()['version']
del get_versions

test = Tester().test
