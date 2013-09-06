import doctest
import unittest
from unittest import TextTestRunner
import numdifftools

def suite():
    return doctest.DocTestSuite(numdifftools.core)

def load_tests(loader=None, tests=None, ignore=None):
    if tests is None:
        return suite()
    else:
        tests.addTests(suite())
        return tests

if __name__=='__main__':
    runner = TextTestRunner()
    unittest.main(testRunner=runner)
    #unittest.main(defaultTest='suite')