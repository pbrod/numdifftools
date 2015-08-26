=========
Changelog
=========

Created with gitcommand: git shortlog v0.9.2..v0.9.3


Version 0.9.10
=============

pbrod (7):
      Fixed sphinx-build and updated docs.
      Added more tests to nd_algopy
      Dropped support for Python 2.6


Version 0.9.4
=============

pbrod (7):
      Fixed sphinx-build and updated docs.


Version 0.9.3
=============

Paul Kienzle (1):
      more useful benchmark plots

pbrod (7):
      Fixed bugs and updated docs.
    	Major rewrite of the easy to use interface to Algopy:     
		   Added possibility to calculate n'th order derivative not just for n=1 in nd_algopy.     
         Added tests to the easy to use interface to algopy


Version 0.9.2
=============

pbrod (3):
      Updated documentation
      Added parenthesis to a call to the print function
      Made the test less strict in order to pass the tests on Travis for python 2.6 and 3.2
      

Version 0.9.1
=============

Christoph Deil (1):
      Fix Sphinx build

pbrod (47):
      Total remake of numdifftools with slightly different call syntax.
         Can compute derivatives of order up to 10-14 depending on function and method used. 
         Updated documentation and tests accordingly.
         Fixed a bug in dea3  
         Added StepsGenerator as an replacement for the adaptive option
         Added bicomplex class for testing the complex step second derivative.
         Added fornberg_weights_all for computing optimal finite difference rules in a stable way.
         Added higher order complex step derivative methods
      

Version 0.7.7
=============

pbrod (35):
      Got travis-ci working in order to run the tests automatically.
      Fixed bugs in Dea class
      Fixed better error estimate for the Hessian
      Fixed tests for python 2.6
      Adding tests as subpackage
      Restructerd folders of numdifftools


Version 0.7.0
=============

pbrod (5):
      Small cosmetic fixes
      pep8 + some refactorings
      Simplified code by refactoring


Version 0.6.0
=============

pbrod (20):
      Update and rename README.md to README.rst
      Simplified call to Derivative: removed step_fix     
		Deleted unused code      
      Simplified and Refactored. Now possible to choose step_num=1
      Changed default step_nom from max(abs(x0), 0.2) to max(log2(abs(x0)), 0.2)
      pep8ified code and made sure that all tests pass.


Version 0.5.0
=============

pbrod (9):
      Updated the examples in Gradient class and in info.py
      Added test for vec2mat and docstrings + cosmetic fixes
      Refactored code into private methods
      Fixed issue #7: Derivative(fun)(numpy.ones((10,5)) * 2) failed
      Made print statements compatible with python 3


Version 0.4.0
=============

pbrod (1)
      Fixed a bug for inf and nan values.


Version 0.3.5
=============

pbrod (1)
      Fixed a bug for inf and nan values.


Version 0.3.4
=============

pbrod (11)
      Made automatic choice for the stepsize more robust.
      Added easy to use interface to the algopy and scientificpython modules


Version 0.3.1
=============

pbrod (4)
      First version of numdifftools published on google.code


