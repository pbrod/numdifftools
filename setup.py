#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for numdifftools.


Usage:
Run all tests:
  python setup.py test

  python setup.py doctests

Build documentation

  python setup.py docs

Install
  python setup.py install [, --prefix=$PREFIX]

Build

  python setup.py bdist_wininst

  python setup.py bdist_wheel --universal

  python setup.py sdist

PyPi upload:
  git pull origin
git shortlog v0.9.20..HEAD -w80 --format="* %s" --reverse > log.txt  # update Changes.rst
# update CHANGELOG.rst with info from log.txt
# update numdifftools.info (this file will be the generated README.rst)
  python build_package.py 0.10.0rc0
  git commit
  git tag v0.10.0rc0 master
  git push --tags
  twine check dist/*   # check
  twine upload dist/*  # wait until the travis report is OK before doing this step.

"""
import os
import re
import sys
from setuptools import setup, Command

HERE = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = 'numdifftools'


def read(*parts):
    with open(os.path.join(*parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(HERE, *file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)  # @UndefinedVariable
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


class Doctest(Command):
    description = 'Run doctests with Sphinx'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.application import Sphinx
        sph = Sphinx('./docs',  # source directory
                     './docs',  # directory containing conf.py
                     './docs/_build',  # output directory
                     './docs/_build/doctrees',  # doctree directory
                     'doctest')  # finally, specify the doctest builder
        sph.build()


def setup_package():
    version = find_version('src', PACKAGE_NAME, "__init__.py")
    print("Version: {}".format(version))

    sphinx_requires = ['sphinx>=1.3.1']
    needs_sphinx = {'build_sphinx'}.intersection(sys.argv)
    sphinx = ['numpydoc',
              'imgmath',
              'sphinx_rtd_theme>=0.1.7'] + sphinx_requires if needs_sphinx else []
    setup(setup_requires=["pytest-runner"] + sphinx,
          version=version,
          cmdclass={'doctests': Doctest},
          extras_require={'build_sphinx': sphinx_requires,},
          )


if __name__ == "__main__":
    setup_package()
