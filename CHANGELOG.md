# Changelog

## [0.11.1] - 2026-09-02

### 🐛 Bug Fixes

- *(test)* Skip profiletools-dependent tests when the optional dependency is unavailable


## [0.11.0] - 2026-09-01

### 🐛 Bug Fixes

- *(numdifftools.test)* Remove unsupported `plugins` argument

### ⚙️ Maintenance

- Remove unused `typing.Any` import
- *(cliff)* Improve changelog generation and commit grouping
- *(pyproject)* Update metadata, tooling, and lock files

### ♻️ Refactoring

- *(test)* Run pytest in a subprocess
- *(profiletools)* Remove bundled profiling code

### 📚 Documentation

- Update readthedocs.yaml
- *(pyproject)* Improve project metadata and release workflow documentation
- *(pyproject)* Add Git command to extract commit messages since the previous release
- *(changelog)* Modernize release history formatting
- *(changelog)* Add git-cliff configuration


## [0.10.1] - 2026-08-12

### 🚀 Features

- Add `app2md2` for application-to-markdown conversion

### 🐛 Bug Fixes

- *(ci)* Make Sphinx doctests work in GitHub Actions
- *(ci)* Fix workflow syntax and optimize test matrix execution in `test.yml`
- *(typing)* Normalize `ArrayLike` input in `loglimits`
- *(Jacobian)* Ensure Jacobian does not fail for functions of shape (1,)

### 🚜 Refactor

- *(api)* Return `EstimateResult` for `full_output` mode
- *(typing)* Introduce protocols and annotate differentiation core
- *(typing)* Modernize type annotations and protocols across package
- *(test_numdifftools.py)* Replace lambda with named function
- Rename `_atleast_2d` to `_ensure_2d_shape`
- Remove Python 2 compatibility code

### 📚 Documentation

- Update documentation examples using the new API
- Update links and remove redundant Python version references in `info.py` and `README.rst`
- Update maintainability badge path in `info.py`

### 🎨 Styling

- Update benchmark title using f-strings
- *(finite_difference.py)* Reformat code to comply with max line length
- Format `app2md2.py` with Ruff
- Remove obsolete code

### ⚙️ Miscellaneous Tasks

- Drop NumPy < 2.0 support and modernize tooling
- Update `codecov-action` to v6
- Simplify Python 3.15 allow-failure logic and improve CI workflow security
- *(ci)* Improve test and release automation
- Delete obsolete `appveyor` and `conda_recipe` folders
- Update build & dependency configurations (`pyproject.toml`, `build_package.py`, `pdm.lock`, `pdm.lock314`)
- Add missing `_typing.py` and coverage reports

## [0.9.42] - 2025-12-11

### 🐛 Bug Fixes
- Removed deprecated "cross_platform" from pdm.lock
- Updated doctests.
- Updated required modules for doc-generation.
- Skip two tests not working on python=3.14 in test_nd_algopy.
- Replaced trapz with trapezoid in test_extrapolation.py and extrapolation.py.
- The function approx_fprime_cs now handles matrix valued functions correctly in nd_statsmodels.py.
- _Limit._add_error_to_outliers method can now handle complex numbers.

### 📚 Documentation
- Updated configuration for docs generation.
- Updated documentation of dea3.

### ⚙️ Miscellaneous Tasks

- Added .readthedocs.yml, cliff.toml, pyproject.toml.
- Updated pyproject.toml and github workflows.
- Calculate the Hessian of functions with n-D output.
- Deleted obsolete files.
- Added prepare-release script


## [0.9.41] - 2022-11-10

### 🏗️ CI/CD
- Added initial GitHub Actions workflow.
- Added test requirements to CI workflow.
- Migrated from Travis CI and AppVeyor to GitHub Actions.
- Updated CI configuration for Python 3.10.
- Removed Python 2.7 and Python 3.6 from CI.

### 🐛 Bug Fixes
- Replaced deprecated `scipy.ndimage.filters` imports.
- Replaced `np.finfo(float).machar.tiny`.
- Fixed NumPy deprecation warnings.
- Updated doctests to avoid CI failures.

### ⚙️ Maintenance
- Dropped support for Python 3.6.
- Removed obsolete Travis CI scripts.
- Removed Python 2.7 classifiers.
- Updated package configuration and requirements.

### ♻️ Performance
- Improved percentile calculation performance.
- Added workaround for known `np.nanpercentile` issue.

## [0.9.40] - 2021-06-02

### ✨ Features
- Added `LogJacobianRule`, `LogHessdiagRule` and `LogHessianRule`.
- Added Jacobian support to finite difference rules.
- Added `richardson_demo`.
- Added Quadpack and Wynn epsilon references.

### 🐛 Bug Fixes
- Fixed Richardson error estimates for complex rules.
- Fixed Jacobian shape handling.
- Fixed Hessian shape handling.
- Fixed multiple failing doctests.

### ♻️ Refactoring
- Moved finite difference logic from `core.py` to `finite_difference.py`.
- Reduced cyclomatic complexity in extrapolation algorithms.
- Simplified step generator logic.
- Simplified Taylor-series related code.

### 📚 Documentation
- Updated Derivative documentation.
- Updated Richardson documentation.
- Added doctest support to documentation.
- Updated tutorials and examples.

### 🏗️ CI/CD
- Added Python 3.8 and 3.9 support.
- Updated Travis and AppVeyor configurations.
- Improved coverage integration.


## [0.9.39] - 2019-06-10
-   Fix issue #43: numpy future warning


## [0.9.38] - 2019-06-10

### ✨ Features
- Added finite_difference.py.
- Added backward differentiation support to nd_statsmodels.

### 🐛 Bug Fixes
- Replaced deprecated scipy.misc.factorial.
- Fixed Python 2.7 installation issues.
- Fixed flaky tests and documentation examples.

### ♻️ Refactoring
- Replaced unittest with pytest.
- Removed dependence on pyscaffold.
- Simplified setup.py and setup.cfg.

### 🏗️ CI/CD
- Added additional CI tests and profiling coverage.
- Updated Travis CI and AppVeyor configurations.

### 📚 Documentation
- Reorganized documentation.
- Updated package badges and documentation examples.


## [0.9.20] - 2017-01-11

 -   Updated the author email address in order for twine to be able
     to upload to pypi.


## [0.9.19] - 2017-01-11

-   Updated setup.py in a attempt to get upload to pypi working
    again.

## [0.9.18] - 2017-01-11

### ✨ Features
- Added fd_derivative.
- Added more rigorous Hypothesis tests.

### 🐛 Bug Fixes
- Fixed Jacobian regression introduced in 0.9.15.

### ⚙️ Maintenance
- Added .pylintrc and PEP 8 cleanup.
- Updated setup.py and packaging configuration.


## [0.9.17] - 2016-09-08

### ✨ Features

- Added tests for `MinMaxStepGenerator`.
- Added extra Jacobian tests.
- Added examples and improved numerical robustness.
- Expanded test coverage and added `test_docstrings()` to `testing.py`.
- Added support for `'backward'` as an alias for `'reverse'` in `nd_algopy.py`.
- Added AppVeyor badge and synchronized `info.py` with `README.rst`.

### 🐛 Bug Fixes

- Fixed Read the Docs link (issue #21).
- Fixed sign error in inverse matrix calculations.
- Fixed `test_scalar_to_vector`.
- Fixed print statements and replaced `xrange` with `range`.
- Reduced start radius for the Fornberg method.
- Updated doctests and documentation tests.
- Updated and corrected documentation throughout the project.

### ♻️ Refactoring

- Renamed `bicomplex` to `Bicomplex`.
- Moved finite-difference utilities to `fornberg.py`.
- Moved step generator implementations to `step_generators.py`.
- Unified and simplified the step generator hierarchy.
- Simplified `example_functions.py`.
- Simplified handling of `n`, `order`, and `default_scale`.
- Replaced lambda expressions with regular functions where appropriate.
- Reduced cyclomatic complexity across multiple modules.
- Renamed non-Pythonic variable names.
- Converted several instance methods to static methods.
- Avoided mutable default arguments and improved internal APIs.

### 📚 Documentation

- Removed obsolete documentation from `core.py`.
- Removed obsolete parameters and examples.
- Updated package documentation and doctests.
- Replaced old string interpolation examples with `str.format()`.

### 🧪 Testing

- Added tests for `EpsAlg`.
- Added tests for `epsalg`.
- Added step generator tests.
- Expanded coverage of Jacobian-related functionality.

### ⚙️ Maintenance

- Removed unused imports.
- Removed unnecessary parentheses and obsolete code.
- Enabled Xvfb in CI to support graphical tests.
- Continued PEP 8 cleanup and code modernization.

## [0.9.14] - 2015-11-10

### ✨ Features

- Added conda recipe files.
- Added additional unit tests, including `tests/test_extrapolation.py`.
- Reintroduced `c_abs` and `c_atan2`.

### 🐛 Bug Fixes

- Filtered warnings in `limits.py`.
- Updated tests and examples.
- Improved package installation and conda support.

### 📚 Documentation

- Updated `README.rst`.
- Added additional documentation.
- Updated `docs/index.rst`.
- Added coverage information to `README.rst`.
- Improved examples and package documentation.

### 🧪 Testing

- Added more tests.
- Expanded extrapolation test coverage.

### ⚙️ Maintenance

- Updated version information.
- Updated setup and packaging configuration.
- Added `.landscape.yml`.
- Updated `.coveragerc` and `.travis.yml`.
- Added the Omnia conda channel to Travis CI.
- Updated conda installation paths.
- Removed unnecessary runtime dependencies for documentation builds.
- Continued project cleanup and dependency simplification.

## [0.9.13] - 2015-10-30

### ✨ Features

- Added AppVeyor support.
- Added AppVeyor installation scripts.
- Added support for differentiating complex-valued functions.
- Added support for zero-order derivatives.

### 🐛 Bug Fixes

- Fixed a missing derivative-order parameter in `Gradient`, `Hessian`, and `Jacobian`.
- Fixed bugs in AppVeyor configuration.
- Removed unnecessary `matplotlib` import.
- Improved Python 3 compatibility in `run_benchmark.py`.
- Made tests more robust.

### ♻️ Refactoring

- Updated package structure to align with PyScaffold 2.4.2.
- Removed duplicate test directories.
- Removed unused code.

### 📚 Documentation

- Updated `README.rst`.
- Updated `CHANGES.rst`.
- Improved documentation throughout the project.

### 🧪 Testing

- Increased robustness of the test suite.

### ⚙️ Maintenance

- Updated setup configuration.
- Fixed minimum NumPy version requirements.
- Added wheel support.
- Updated AppVeyor configuration and build scripts.


## [0.9.12] - 2015-08-28

-   Updated documentation.
-   Updated version in conf.py.
-   Updated CHANGES.rst.
-   Reimplemented outlier detection and made it more robust.
-   Added limits.py with tests.
-   Updated main tests folder.
-   Moved Richardson and dea3 to extrapolation.py.
-   Making a new release in order to upload to pypi.


## [0.9.11] - 2015-08-27

-   Fixed sphinx-build and updated docs.
-   Fixed issue #9 Backward differentiation method fails with
    additional parameters.


## [0.9.10] - 2015-08-26


-   Fixed sphinx-build and updated docs.
-   Added more tests to nd_algopy.
-   Dropped support for Python 2.6.


## [0.9.4] - 2015-08-26

-   Fixed sphinx-build and updated docs.


## [0.9.3] - 2015-08-23
-   more useful benchmark plots.
-   Fixed bugs and updated docs.
-   Major rewrite of the easy to use interface to Algopy.
-   Added possibility to calculate n\'th order derivative not just
    for n=1 in nd_algopy.
-   Added tests to the easy to use interface to algopy.


## [0.9.2] - 2015-08-20
-   Updated documentation
-   Added parenthesis to a call to the print function
-   Made the test less strict in order to pass the tests on Travis
    for python 2.6 and 3.2.


## [0.9.1] - 2015-08-20

-   Fix Sphinx build
-   Total remake of numdifftools with slightly different call syntax:
    -   Can compute derivatives of order up to 10-14 depending
        on function and method used.
    -   Updated documentation and tests accordingly.
    -   Fixed a bug in dea3.
    -   Added StepsGenerator as an replacement for the adaptive
        option.
    -   Added bicomplex class for testing the complex step
        second derivative.
    -   Added fornberg_weights_all for computing optimal finite
        difference rules in a stable way.
    -   Added higher order complex step derivative methods.


## [0.7.7] - 2014-12-18

-   Got travis-ci working in order to run the tests automatically.
-   Fixed bugs in Dea class.
-   Fixed better error estimate for the Hessian.
-   Fixed tests for python 2.6.
-   Adding tests as subpackage.
-   Restructerd folders of numdifftools.


## [0.7.3] - 2014-12-17

-   Small cosmetic fixes.
-   pep8 + some refactorings.
-   Simplified code by refactoring.


## [0.6.0] - 2014-02-08

-   Update and rename README.md to README.rst.
-   Simplified call to Derivative: removed step_fix.
-   Deleted unused code.
-   Simplified and Refactored. Now possible to choose step_num=1.
-   Changed default step_nom from max(abs(x0), 0.2) to
    max(log2(abs(x0)), 0.2).
-   pep8ified code and made sure that all tests pass.


## [0.5.0] - 2014-01-10

-   Updated the examples in Gradient class and in info.py.
-   Added test for vec2mat and docstrings + cosmetic fixes.
-   Refactored code into private methods.
-   Fixed issue #7: Derivative(fun)(numpy.ones((10,5)) \* 2) failed.
-   Made print statements compatible with python 3.


## [0.4.0] - 2012-05-05

-   Fixed a bug for inf and nan values.


## [0.3.5] - 2011-05-19

-   Fixed a bug for inf and nan values.


## [0.3.4] - 2011-02-24

-   Made automatic choice for the stepsize more robust.
-   Added easy to use interface to the algopy and scientificpython
    modules.


## [0.3.1] - 2009-05-20

-   First version of numdifftools published on google.code
