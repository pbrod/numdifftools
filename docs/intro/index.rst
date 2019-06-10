.. _introduction:

============
Introduction
============


What is numdifftools?
=====================

Numdifftools is a suite of tools written in `_Python <http://www.python.org/>`_
to solve automatic numerical differentiation problems in one or more variables.
Finite differences are used in an adaptive manner, coupled with a Richardson
extrapolation methodology to provide a maximally accurate result.
The user can configure many options like; changing the order of the method or
the extrapolation, even allowing the user to specify whether complex-step,
central, forward or backward differences are used.

The methods provided are:

- **Derivative**: Compute the derivatives of order 1 through 10 on any scalar function.

- **directionaldiff**: Compute directional derivative of a function of n variables

- **Gradient**: Compute the gradient vector of a scalar function of one or more variables.

- **Jacobian**: Compute the Jacobian matrix of a vector valued function of one or more variables.

- **Hessian**: Compute the Hessian matrix of all 2nd partial derivatives of a scalar function of one or more variables.

- **Hessdiag**: Compute only the diagonal elements of the Hessian matrix

All of these methods also produce error estimates on the result.


Numdifftools also provide an easy to use interface to derivatives calculated
with in `_AlgoPy <https://pythonhosted.org/algopy/>`_. Algopy stands for Algorithmic
Differentiation in Python.
The purpose of AlgoPy is the evaluation of higher-order derivatives in the
`forward` and `reverse` mode of Algorithmic Differentiation (AD) of functions
that are implemented as Python programs.




How the documentation is organized
==================================

Numdifftools has a lot of documentation. A high-level overview of how it's organized
will help you know where to look for certain things:

* :doc:`Tutorials </tutorials/index>` take you by the hand through a series of
  steps to load a CDF container and explore its contents or to
  construct a new dataset and validate it. Start here if you're new to numdifftools.

* :doc:`Topic guides </topics/index>` discuss key topics and concepts at a
  fairly high level and provide useful background information and explanation.

* :doc:`Reference guides </reference/index>` contain technical reference for APIs and
  other aspects of numdifftools' machinery. They describe how it works and how to
  use it but assume that you have a basic understanding of key concepts.

* :doc:`How-to guides </how-to/index>` are recipes. They guide you through the
  steps involved in addressing key problems and use-cases. They are more
  advanced than tutorials and assume some knowledge of how numdifftools works.

