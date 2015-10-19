import pkg_resources
from .info import __doc__
from .core import *
from numpy.testing import Tester
try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'

test = Tester().test