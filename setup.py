#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for numdifftools.

    This file was generated with PyScaffold 2.4.2, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/

Usage:
Run all tests:
  python setup.py test

  python setup.py doctests

Build documentation

  python setup.py docs

Install
  python setup.py install [, --prefix=$PREFIX]

Build wininstaller

  python setup.py bdist_wininst

  python setup.py bdist_wheel --universal

PyPi upload:
  python setup.py sdist bdist_wheel upload

  python setup.py sdist bdist_wininst upload --show-response

  python setup.py register sdist bdist_wininst upload --show-response
"""

import sys
from setuptools import setup


def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['six', 'pyscaffold>=2.4rc1,<2.5a0'] + sphinx,
          tests_require=['pytest_cov', 'pytest'],
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
