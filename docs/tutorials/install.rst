.. _install:


=============
Install guide
=============

Before you can use numdifftools, you'll need to get it installed. This guide will
guide you through a simple installation
that'll work while you walk through the introduction.


Install Python
==============

Being a Python library, numdifftools requires Python. Preferably you ned version 3.4 or
newer, but you get the latest version of Python at
https://www.python.org/downloads/.

You can verify that Python is installed by typing ``python`` from the command shell;
you should see something like:

.. code-block:: console


    Python 3.6.3 (64-bit)| (default, Oct 15 2017, 03:27:45)
    [MSC v.1900 64 bit (AMD64)] on win32
    Type "help", "copyright", "credits" or "license" for more information.
    >>>


``pip`` is the Python installer. Make sure yours is up-to-date, as earlier versions can be less reliable:


.. code-block:: console

    $ pip install --upgrade pip


Dependencies
============
Numdifftools requires numpy 1.9 or newer, 
scipy 0.8 or newer, and Python 2.7 or 3.3 or newer. 
This tutorial assumes you are using Python 3. 
Optionally you may also want to install Algopy 0.4 or newer and statsmodels 0.6 or newer in order 
to be able to use their easy to use interfaces to their derivative functions. 


Install numdifftools
====================

To install numdifftools simply type in the 'command' shell:

.. code-block:: console

    $ pip install numdifftools

to get the lastest stable version. Using pip also has the advantage 
that all requirements are automatically installed.


Verifying installation
======================
To verify that numdifftools can be seen by Python, type ``python`` from your shell.
Then at the Python prompt, try to import numdifftools:

.. parsed-literal::

    >>> import numdifftools as nd
    >>> print(nd.__version__)
    |release|


To test if the toolbox is working correctly paste the following in an interactive python prompt:

.. parsed-literal::

    nd.test('--doctest-module')


If the result show no errors, you now have installed a fully functional toolbox.
Congratulations!


That's it!
==========

That's it -- you can now :doc:`move onto the getting started tutorial </tutorials/getting_started>`
