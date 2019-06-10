=========
Changelog
=========

Version 0.9.39 Jun 10, 2019
===========================

Robert Parini (1):
      * Fix issue #43: numpy future warning



Version 0.9.38 Jun 10, 2019
===========================

Andrew Nelson (1):
      * MAINT: special.factorial instead of misc.factorial

Dougal J. Sutherland (1):
      * include LICENSE.txt in distributions

Per A Brodtkorb (140):
      * Adjusted runtime for hypothesis tests to avoid failure and fixed pep8 failures.
      * Fixed a bug in setup.cfg
      * Replaced valarray function with numpy.full in step_generators.py 
      * Added try except on import of algopy 
      * Updated the badges used in the README.rst 
      * Replaced numpy.testing.Tester with pytest. 
      * Removed dependence on pyscaffold.
      * Simplified setup.py and setup.cfg 
      * Updated .travis.yml configuration.
      * Reorganized the documentation. 
      * Ongoing work to simplify the classes.
      * Replaced unittest with pytest.
      * Added finite_difference.py
      * replaced , with .
      * Reverted to coverage=4.3.4
      * New attempt
      * Fixed conflicting import
      * Missing import of EPS
      * Added missing FD_RULES = {}
      * Removed pinned coverage, removed dependence on pyscaffold
      * Updated .travis.yml and .appveyor.yml
      * Replaced conda channel omnia with conda-forge
      * Removed commented out code. Set pyqt=5 in appveyor.yml
      * Updated codeclimate checks
      * Dropped support for python 3.3 and 3.4. Added support for python 3.6, 3.7
      * Simplified code.
      * Pinned IPython==5.0 in order to make the testserver not crash.
      * Added line_profiler to appveyor.yml
      * Removed line_profiler from requirements.txt
      * Fix issue #37: Unable to install on Python 2.7 
      * Added method='backward' to nd_statsmodels.py 
      * Skip test_profile_numdifftools_profile_hessian and TestDoProfile
      * Added missing import of warnings
      * Added tests for the scripts from profile_numdifftools.py, _find_default_scale.py and run_benchmark.py.
      * Added reason to unittest.skipIf
      * Added line_profiler to requirements.
      * misssing import of warnings fixed.
      * Renamed test so it comes last, because I suspect this test mess up the coverage stats.
      * Reordered the tests.
      * Added more tests.
      * Cleaned up _find_default_scale.py
      * Removed link to depsy
      * Reverted: install of cython and pip install setuptools
      * Disabled sonar-scanner -X for python 3.5 because it crashes.
      * Reverted [options.packages.find] to exclude tests again
      * Added cython and reverted to pip install setuptools
      * Updated sphinx to 1.6.7
      * Try to install setuptools with conda instead.
      * Added hypothesis and pytest to requirements.readthedocs.txt
      * Set version of setuptools==37.0
      * Added algopy, statsmodels and numpy to requirements.readthedocs.txt
      * Restricted sphinx in the hope that the docs will be generated.
      * Removed exclusion of tests/ directory from test coverage.
      * Added dependencies into setup.cfg
      * Readded six as dependency
      * Refactored and removed commented out code.
      * Fixed a bug in the docstring example: Made sure the shape passed on to zeros is an integer.
      * Fixed c_abs so it works with algopy on python 3.6.
      * Fixed flaky test and made it more robust.
      * Fixed bug in .travis.yml
      * Refactored the taylor function into the Taylor class in order to simplify the code.
      * Fixed issue #35 and added tests
      * Attempt to simplify complexity
      * Made doctests more robust
      * Updated project path
      * Changed install of algopy
      * Fixed small bugs
      * Updated docstrings
      * Changed Example and Reference to Examples and References in docstrings to comply with numpydoc-style.
      * Renamed CHANGES.rst to CHANGELOG.rst
      * Renamed source path
      * Hack due to a bug in algopy or changed behaviour.
      * Small fix.
      * Try to reduce complexity
      * Reduced cognitive complexity of min_num_steps
      * Simplified  code in Jacobian
      * Merge branch 'master' of https://github.com/pbrod/numdifftools
      * Fixed issue #34 Licence clarification.
      * Locked coverage=4.3.4 due to a bug in coverage that cause code-climate test-reporter to fail.
      * Added script for finding default scale
      * updated from sonarcube to sonarcloud
      * Made sure shape is an integer.
      * Refactored make_step_generator into a step property
      * Issue warning message to the user when setting the order to something different than 1 or 2 in Hessian.
      * Updated example in Gradient.
      * Reverted --timid option to coverage because it took too long time to run.
      * Reverted --pep8 option
      * pep8 + added --timid in .travis.yml coverage run in order to to increase missed coverage.
      * Refactored taylor to reduce complexity
      * No support for python 3.3. Added python 3.6
      * Fixed a small bug and updated test.
      * Removed unneccasarry perenthesis. Reduced the complexity of do_profile
      * Made python3 compatible
      * Removed assert False
      * Made unittests more forgiving.
      * updated doctest in nd_scipy.py and profiletools.py install line_profiler on travis
      * Made python 3 compatible
      * Updated tests
      * Added test_profiletools.py and capture_stdout_and_stderr in testing.py
      * Optimized numdifftools.core.py for speed: fd_rules are now only computed once.
      * Only keeping html docs in the distribution.
      * Added doctest and updated .pylintrc and requirements.txt
      * Reduced time footprint on tests in the hope that it will pass on Travis CI.
      * Prefer static methods over instance methods
      * Better memory handling: This fixes issue #27
      * Added statsmodels to requirements.txt
      * Added nd_statsmodels.py
      * Simplified input
      * Merge branch 'master' of https://github.com/pbrod/numdifftools
      * Updated link to the documentation.

Robert Parini (4):
      * Avoid RuntimeWarning in _get_logn
      * Allow fd_derivative to take complex valued functions

solarjoe (1):
      * doc: added nd.directionaldiff example



Version 0.9.20, Jan 11, 2017
============================

Per A Brodtkorb (1):
     * Updated the author email address in order for twine to be able to upload to pypi.


Version 0.9.19, Jan 11, 2017
============================

Per A Brodtkorb (1):
      * Updated setup.py in a attempt to get upload to pypi working again.


Version 0.9.18, Jan 11, 2017
============================

Per A Brodtkorb (38):
      * Updated setup
      * Added import statements in help header examples.
      * Added more rigorous tests using hypothesis.
      * Forced to use wxagg backend
      * Moved import of matplotlib.pyplot to main in order to avoid import error on travis.
      * Added fd_derivative function
      * Updated references.
      * Attempt to automate sonarcube analysis
      * Added testcoverage to sonarqube and codeclimate
      * Simplified code
      * Added .pylintrc + pep8
      * Major change in api: class member variable self.f changed to self.fun
      * Fixes issue #25 (Jacobian broken since 0.9.15)


Version 0.9.17, Sep 8, 2016
============================

Andrew Fowlie (1):
      * Fix ReadTheDocs link as mentioned in #21

Per A Brodtkorb (79):
      * Added test for MinMaxStepgenerator
      * Removed obsolete docs from core.py
      * Updated appveyor.yml
      * Fixed sign in inverse matrix
      * Simplified code 
      * Added appveyor badge + synchronised info.py with README.rst.
      * Removed plot in help header
      * Added Programming Language :: Python :: 3.5
      * Simplified code
      * Renamed bicomplex to Bicomplex
      * Simplified example_functions.py
      * Moved MinStepGenerator, MaxStepGeneretor and MinMaxStepGenerator to step_generators.py
		* Unified the step generators
		* Moved step_generator tests to test_step_generators.py
		* Major simplification of step_generators.py
      * Removed duplicated code + pep8
      * Moved fornberg_weights to fornberg.py + added taylor and derivative
      * Fixed print statement
      * Replace xrange with range
      * Added examples + made computation more robust.
      * Made 'backward' and alias for 'reverse' in nd_algopy.py
      * Expanded the tests + added test_docstrings to testing.py
      * Replace string interpolation with format()
      * Removed obsolete parameter
      * Smaller start radius for Fornberg method
      * Simplified "n" and "order" properties
      * Simplified default_scale
      * Removed unecessary parenthesis and code. pep8
      * Fixed a bug in Dea + small refactorings.
      * Added test for EpsAlg
      * Avoid mutable default args and prefer static methods over instance-meth.
      * Refactored to reduce cyclomatic complexity
      * Changed some instance methods to static methods
      * Renamed non-pythonic variable names
      * Turned on xvfb (X Virtual Framebuffer) to imitate a display.
      * Added extra test for Jacobian
      * Replace lambda function with a def
      * Removed unused import
      * Added test for epsalg
      * Fixed test_scalar_to_vector
      * Updated test_docstrings


Version 0.9.15, May 10, 2016
============================

Cody (2):
      * Migrated `%` string formating
      * Migrated `%` string formating

Per A Brodtkorb (28):
      * Updated README.rst + setup.cfg
      * Replaced instance methods with static methods +pep8
      * Merge branch 'master' of https://github.com/pbrod/numdifftools
      * Fixed a bug: replaced missing triple quote
      * Added depsy badge
      * added .checkignore for quantificode
      * Added .codeclimate.yml
      * Fixed failing tests
      * Changed instance methods to static methods
      * Made untyped exception handlers specific
      * Replaced local function with a static method
      * Simplified tests
      * Removed duplicated code Simplified _Derivative._get_function_name
      * exclude tests from testclimate
      * Renamed test_functions.py to example_functions.py Added test_example_functions.py

Per A. Brodtkorb (2):
      * Merge pull request #17 from pbrod/autofix/wrapped2_to3_fix
      * Merge pull request #18 from pbrod/autofix/wrapped2_to3_fix-0

pbrod (17):
      * updated conf.py
      * added numpydoc>=0.5, sphinx_rtd_theme>=0.1.7 to setup_requires if sphinx
      * updated setup.py
      * added requirements.readthedocs.txt
      * Updated README.rst with info about how to install it using conda in an anaconda package.
      * updated conda install description
      * Fixed number of arguments so it does not differs from overridden '_default_base_step' method
      * Added codecov to .travis.yml
      * Attempt to remove coverage of test-files
      * Added directionaldiff function in order to calculate directional derivatives. Fixes issue #16. Also added supporting tests and examples to the documentation.
      * Fixed isssue #19 multiple observations mishandled in Jacobian
      * Moved rosen function into numdifftools.testing.py
      * updated import of rosen function from numdifftools.testing
      * Simplified code + pep8 + added TestResidue
      * Updated readme.rst and replaced string interpolation with format()
      * Cleaned Dea class + pep8
      * Updated references for Wynn extrapolation method.



Version 0.9.14, November 10, 2015
=================================

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
================================

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
===============================

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
===============================

pbrod (2):
      * Fixed sphinx-build and updated docs.
      * Fixed issue #9 Backward differentiation method fails with additional parameters.


Version 0.9.10, August 26, 2015
===============================

pbrod (7):
      * Fixed sphinx-build and updated docs.
      * Added more tests to nd_algopy.
      * Dropped support for Python 2.6.


Version 0.9.4, August 26, 2015
==============================

pbrod (7):
      * Fixed sphinx-build and updated docs.


Version 0.9.3, August 23, 2015
==============================

Paul Kienzle (1):
      * more useful benchmark plots.

pbrod (7):
      * Fixed bugs and updated docs.
      * Major rewrite of the easy to use interface to Algopy.
      * Added possibility to calculate n'th order derivative not just for n=1 in nd_algopy.
      * Added tests to the easy to use interface to algopy.



Version 0.9.2, August 20, 2015
==============================

pbrod (3):
      * Updated documentation
      * Added parenthesis to a call to the print function
      * Made the test less strict in order to pass the tests on Travis for python 2.6 and 3.2.
      

Version 0.9.1, August 20,2015
=============================

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
================================

pbrod (35):
      * Got travis-ci working in order to run the tests automatically.
      * Fixed bugs in Dea class.
      * Fixed better error estimate for the Hessian.
      * Fixed tests for python 2.6.
      * Adding tests as subpackage.
      * Restructerd folders of numdifftools.


Version 0.7.3, December 17, 2014
================================

pbrod (5):
      * Small cosmetic fixes.
      * pep8 + some refactorings.
      * Simplified code by refactoring.



Version 0.6.0, February 8, 2014
===============================

pbrod (20):
      * Update and rename README.md to README.rst.
      * Simplified call to Derivative: removed step_fix.
      * Deleted unused code.
      * Simplified and Refactored. Now possible to choose step_num=1.
      * Changed default step_nom from max(abs(x0), 0.2) to max(log2(abs(x0)), 0.2).
      * pep8ified code and made sure that all tests pass.


Version 0.5.0, January 10, 2014
===============================

pbrod (9):
      * Updated the examples in Gradient class and in info.py.
      * Added test for vec2mat and docstrings + cosmetic fixes.
      * Refactored code into private methods.
      * Fixed issue #7: Derivative(fun)(numpy.ones((10,5)) * 2) failed.
      * Made print statements compatible with python 3.



Version 0.4.0, May 5, 2012
==========================

pbrod (1)
      * Fixed a bug for inf and nan values.




Version 0.3.5, May 19, 2011
===========================

pbrod (1)
      * Fixed a bug for inf and nan values.


Version 0.3.4, Feb 24, 2011
===========================

pbrod (11)
      * Made automatic choice for the stepsize more robust.
      * Added easy to use interface to the algopy and scientificpython modules.


Version 0.3.1, May 20, 2009
===========================

pbrod (4)
      * First version of numdifftools published on google.code


