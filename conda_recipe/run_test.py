import os

import numdifftools
import pytest


path = os.path.join(numdifftools.__path__[0], 'tests')
os.chdir(path)

pytest.main()
