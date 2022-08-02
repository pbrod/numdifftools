#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup file for numdifftools.

Usage:
Run doctests on documentation:
  python setup.py doctest

Build documentation
  python setup.py docs

Build
  python setup.py bdist_wininst
  python setup.py bdist_wheel --universal
  python setup.py sdist

Recommended build
  git pull origin
  git shortlog v0.9.20..HEAD -w80 --format="* %s" --reverse > log.txt
# update CHANGELOG.rst with info from log.txt
# update numdifftools.info (this file will be the generated README.rst)
  python build_package.py 0.10.0rc0
  git commit
  git tag v0.10.0rc0 master
  git push --tags

PyPi upload:
  twine check dist/*   # check
  twine upload dist/*  # wait until the travis report is OK before doing this step.

Notes
-----
Don't use package_data and/or data_files, use include_package_data=True and MANIFEST.in instead!
Don't hard-code the list of packages, use setuptools.find_packages() instead!


See also
--------
https://docs.pytest.org/en/latest/goodpractices.html
https://python-packaging.readthedocs.io/en/latest/
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
https://ep2015.europython.eu/media/conference/slides/less-known-packaging-features-and-tricks.pdf
https://realpython.com/documenting-python-code/#public-and-open-source-projects

"""
import os
import re
import sys
import pkg_resources
from setuptools import setup, Command
pkg_resources.require('setuptools>=39.2') # setuptools >=38.3.0     # version with most `setup.cfg` bugfixes
ROOT = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = 'numdifftools'


def read(file_path, lines=False):
    """Returns contents of file either as a string or list of lines."""
    with open(file_path, 'r') as fp:
        if lines:
            return fp.readlines()
        return fp.read()


def find_version(file_path):
    """Returns version given in the __version__ variable of a module file"""
    version_file = read(file_path)
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
    version = find_version(os.path.join(ROOT, 'src', PACKAGE_NAME, "__init__.py"))
    print("Version: {}".format(version))

    sphinx_requires = ['sphinx>=1.3.1']
    needs_sphinx = {'build_sphinx'}.intersection(sys.argv)
    sphinx = ['numpydoc',
              'imgmath',
              'sphinx_rtd_theme>=0.1.7'] + sphinx_requires if needs_sphinx else []
    setup(
        name=PACKAGE_NAME,
        version=version,
        install_requires=read(os.path.join(ROOT, 'requirements.txt'), lines=True),
        extras_require={'build_sphinx': sphinx_requires},
        setup_requires=["pytest-runner"] + sphinx,
        tests_require=['pytest',
                       'pytest-cov',
                       'pytest-pep8',
                       'hypothesis',
                       'matplotlib',
                       'line_profiler'
                       ],
        cmdclass={'doctest': Doctest},
    )


if __name__ == "__main__":
    # sys.argv.append('docs')
    # sys.argv.append('bdist_wininst')
    setup_package()
