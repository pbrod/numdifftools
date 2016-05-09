=========
Changelog
=========

Created with gitcommand: git shortlog v0.9.14..v0.9.15

Version 0.9.15, May 10, 2016
---------------------------------

Cody (2):
      Migrated `%` string formating
      Migrated `%` string formating

Per A Brodtkorb (28):
      Updated README.rst + setup.cfg
      Replaced instance methods with static methods +pep8
      Merge branch 'master' of https://github.com/pbrod/numdifftools
      Fixed a bug: replaced missing triple quote
      updated link to plot
      updated link to example picture
      Added depsy badge
      added .checkignore for quantificode
      Small refactorings
      Fixed failing tests
      Added .codeclimate.yml
      Fixed failing tests
      Changed instance methods to static methods
      Changed instance methods to static methods
      Made untyped exception handlers specific
      Replaced local function with a static method
      Simplified tests
      Removed duplicated code Simplified _Derivative._get_function_name
      exclude tests from testclimate
      Renamed test_functions.py to example_functions.py Added test_example_functions.py
      Fixed import

Per A. Brodtkorb (2):
      Merge pull request #17 from pbrod/autofix/wrapped2_to3_fix
      Merge pull request #18 from pbrod/autofix/wrapped2_to3_fix-0

pbrod (17):
      updated conf.py
      added numpydoc>=0.5, sphinx_rtd_theme>=0.1.7 to setup_requires if sphinx
      updated setup.py
      added requirements.readthedocs.txt
      Updated README.rst with info about how to install it using conda in an anaconda package.
      updated conda install description
      Fixed number of arguments so it does not differs from overridden '_default_base_step' method
      Added codecov to .travis.yml
      Attempt to remove coverage of test-files
      Added directionaldiff function in order to calculate directional derivatives. Fixes issue #16. Also added supporting tests and examples to the documentation.
      Fixed isssue #19 multiple observations mishandled in Jacobian
      Moved rosen function into numdifftools.testing.py
      updated import of rosen function from numdifftools.testing
      Simplified code + pep8 + added TestResidue
      Updated readme.rst and replaced string interpolation with format()
      Cleaned Dea class + pep8
      Updated references for Wynn extrapolation method.



Version 0.9.14, November 10, 2015
---------------------------------

pbrod (53):
      * Updated documentation of setup.py
      * Updated README.rst
      * updated version
      * Added more documentation
      * Updated example
      * Added .landscape.yml     updated .coveragerc, .travis.yml
      * Added coverageall to README.rst.
      * updated docs/index.rst
      * Removed unused code and added tests/test_extrapolation.py
      * updated tests
      * Added more tests
      * Readded c_abs c_atan2
      * Removed dependence on wheel, numpydoc>=0.5 and sphinx_rtd_theme>=0.1.7 (only needed for building documentation)
      * updated conda path in .travis.yml
      * added omnia channel to .travis.yml
      * Added conda_recipe files     Filtered out warnings in limits.py


Version 0.9.13, October 30, 2015
---------------------------------

pbrod (21):
      * Updated README.rst and CHANGES.rst.
      * updated Limits.
      * Made it possible to differentiate complex functions and allow zero'th order derivative.
      * BUG: added missing derivative order, n to Gradient, Hessian, Jacobian.
      * Made test more robust.
      * Updated structure in setup according to pyscaffold version 2.4.2.
      * Updated setup.cfg and deleted duplicate tests folder.
      * removed unused code.
      * Added appveyor.yml.
      * Added required appveyor install scripts
      * Fixed bug in appveyor.yml.
      * added wheel to requirements.txt.
      * updated appveyor.yml.
      * Removed import matplotlib.

Justin Lecher (1):
      * Fix min version for numpy.

kikocorreoso (1):
      * fix some prints on run_benchmark.py to make it work with py3


Version 0.9.12, August 28, 2015
-------------------------------

pbrod (12):
      
      * Updated documentation.
      * Updated version in conf.py.
      * Updated CHANGES.rst.
      * Reimplemented outlier detection and made it more robust.     
      * Added limits.py with tests.
      * Updated main tests folder.        
      * Moved Richardson and dea3 to extrapolation.py.
      * Making a new release in order to upload to pypi.


Version 0.9.11, August 27, 2015
-------------------------------

pbrod (2):
      * Fixed sphinx-build and updated docs.
      * Fixed issue #9 Backward differentiation method fails with additional parameters.


Version 0.9.10, August 26, 2015
-------------------------------

pbrod (7):
      * Fixed sphinx-build and updated docs.
      * Added more tests to nd_algopy.
      * Dropped support for Python 2.6.


Version 0.9.4, August 26, 2015
------------------------------

pbrod (7):
      * Fixed sphinx-build and updated docs.


Version 0.9.3, August 23, 2015
------------------------------

Paul Kienzle (1):
      * more useful benchmark plots.

pbrod (7):
      * Fixed bugs and updated docs.
      * Major rewrite of the easy to use interface to Algopy.
      * Added possibility to calculate n'th order derivative not just for n=1 in nd_algopy.
      * Added tests to the easy to use interface to algopy.



Version 0.9.2, August 20, 2015
------------------------------

pbrod (3):
      * Updated documentation
      * Added parenthesis to a call to the print function
      * Made the test less strict in order to pass the tests on Travis for python 2.6 and 3.2.
      

Version 0.9.1, August 20,2015
-----------------------------

Christoph Deil (1):
      * Fix Sphinx build

pbrod (47):
      * Total remake of numdifftools with slightly different call syntax.
         * Can compute derivatives of order up to 10-14 depending on function and method used. 
         * Updated documentation and tests accordingly.
         * Fixed a bug in dea3.
         * Added StepsGenerator as an replacement for the adaptive option.
         * Added bicomplex class for testing the complex step second derivative.
         * Added fornberg_weights_all for computing optimal finite difference rules in a stable way.
         * Added higher order complex step derivative methods.
      


Version 0.7.7, December 18, 2014
--------------------------------

pbrod (35):
      * Got travis-ci working in order to run the tests automatically.
      * Fixed bugs in Dea class.
      * Fixed better error estimate for the Hessian.
      * Fixed tests for python 2.6.
      * Adding tests as subpackage.
      * Restructerd folders of numdifftools.


Version 0.7.3, December 17, 2014
--------------------------------

pbrod (5):
      * Small cosmetic fixes.
      * pep8 + some refactorings.
      * Simplified code by refactoring.



Version 0.6.0, February 8, 2014
--------------------------------

pbrod (20):
      * Update and rename README.md to README.rst.
      * Simplified call to Derivative: removed step_fix.
      * Deleted unused code.
      * Simplified and Refactored. Now possible to choose step_num=1.
      * Changed default step_nom from max(abs(x0), 0.2) to max(log2(abs(x0)), 0.2).
      * pep8ified code and made sure that all tests pass.


Version 0.5.0, January 10, 2014
-------------------------------

pbrod (9):
      * Updated the examples in Gradient class and in info.py.
      * Added test for vec2mat and docstrings + cosmetic fixes.
      * Refactored code into private methods.
      * Fixed issue #7: Derivative(fun)(numpy.ones((10,5)) * 2) failed.
      * Made print statements compatible with python 3.



Version 0.4.0, May 5, 2012
--------------------------

pbrod (1)
      * Fixed a bug for inf and nan values.




Version 0.3.5, May 19, 2011
---------------------------

pbrod (1)
      * Fixed a bug for inf and nan values.


Version 0.3.4, Feb 24, 2011
---------------------------

pbrod (11)
      * Made automatic choice for the stepsize more robust.
      * Added easy to use interface to the algopy and scientificpython modules.


Version 0.3.1, May 20, 2009
---------------------------

pbrod (4)
      * First version of numdifftools published on google.code


