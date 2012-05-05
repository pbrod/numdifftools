"""
Install numdifftools

Usage:

python setup.py install [, --prefix=$PREFIX]

python setup.py bdist_wininst

PyPi upload:

python setup.py sdist bdist_wininst upload --show-response

python setup.py register sdist bdist_wininst upload --show-response


"""
#!/usr/bin/env python
import os, sys

DISTUTILS_DEBUG = True
#sys.argv.append("develop")
#sys.argv.append("install")
# make sure we import from this package, not an installed one:
sys.path.insert(0, os.path.join('numdifftools'))
import info as numdifftools

if  True:#__file__ == 'setupegg.py':
    # http://peak.telecommunity.com/DevCenter/setuptools
    from setuptools import setup, Extension
else:
    from distutils.core import setup
test_dir = os.path.join('numdifftools','test')
doc_dir = os.path.join('numdifftools','doc')
testscripts = [os.path.join('test', f)
               for f in os.listdir(test_dir)
               if not (f.startswith('.') or f.endswith('~') or
                       f.endswith('.old') or f.endswith('.bak'))]
docs = [os.path.join('doc', f) for f in os.listdir(doc_dir) if os.path.isfile(f)]
packagedata = docs + testscripts
#package_data = {'numdifftools': packagedata},
setup(
    name = "Numdifftools",
    version = '0.4.0',
    author="John D'Errico and Per A. Brodtkorb",
    author_email='woodchips at rochester.rr.com, Brodtkorb at frisurf.no',
    description = 'Solves automatic numerical differentiation problems in one or more variables.',
    long_description = numdifftools.__doc__,
    license = "New BSD",
    install_requires = ['numpy>=1.4', 'scipy>=0.8'],
    url='http://code.google.com/p/numdifftools/',
    maintainer='Per A. Brodtkorb',
    maintainer_email = 'Brodtkorb at frisurf.no',
    packages = ['numdifftools'],
    package_data = {'': packagedata},
    classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Topic :: Scientific/Engineering :: Mathematics',
          ],
    )
