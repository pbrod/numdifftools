=================
What to read next
=================

So you've read all the :doc:`introductory material </tutorials/index>` and have
decided you'd like to keep using numdifftools. We've only just scratched the surface
with this intro.

So what's next?

Well, we've always been big fans of learning by doing. At this point you should
know enough to start a project of your own and start fooling around. As you need
to learn new tricks, come back to the documentation.

We've put a lot of effort into making numdifftools's documentation useful, easy to
read and as complete as possible. The rest of this document explains more about
how the documentation works so that you can get the most out of it.


Finding documentation
=====================

Numdifftools got a *lot* of documentation,
so finding what you need can sometimes be tricky. A few good places to start
are the :ref:`search` and the :ref:`genindex`.

Or you can just browse around!

How the documentation is organized
==================================

Numdifftools main documentation is broken up into "chunks" designed to fill
different needs:

* The :doc:`introductory material </tutorials/index>` is designed for people new
  to numdifftools. It doesn't cover anything in depth, but instead gives a hands on
  overview of how to use numdifftools.

* The :doc:`topic guides </topics/index>`, on the other hand, dive deep into
  individual parts of numdifftools from a theoretical perspective.

* We've written a set of :doc:`how-to guides </how-to/index>` that answer
  common "How do I ...?" questions.

* The guides and how-to's don't cover every single class, function, and
  method available in numdifftools -- that would be overwhelming when you're
  trying to learn. Instead, details about individual classes, functions,
  methods, and modules are kept in the :doc:`reference </reference/index>`. This is
  where you'll turn to find the details of a particular function or
  whatever you need.


How documentation is updated
============================

Just as the numdifftools code base is developed and improved on a daily basis, our
documentation is consistently improving. We improve documentation for several
reasons:

* To make content fixes, such as grammar/typo corrections.

* To add information and/or examples to existing sections that need to be
  expanded.

* To document numdifftools features that aren't yet documented. (The list of
  such features is shrinking but exists nonetheless.)

* To add documentation for new features as new features get added, or as
  numdifftools APIs or behaviors change.


In plain text
-------------

For offline reading, or just for convenience, you can read the numdifftools
documentation in plain text.

If you're using an official release of numdifftools, the zipped package (tarball) of
the code includes a ``docs/`` directory, which contains all the documentation
for that release.

If you're using the development version of numdifftools (aka the master branch), the
``docs/`` directory contains all of the documentation. You can update your
Git checkout to get the latest changes.

One low-tech way of taking advantage of the text documentation is by using the
Unix ``grep`` utility to search for a phrase in all of the documentation. For
example, this will show you each mention of the phrase "max_length" in any
numdifftools document:

.. code-block:: console

    $ grep -r max_length /path/to/numdifftools/docs/


As HTML, locally
----------------

You can get a local copy of the HTML documentation following a few easy steps:

* numdifftools's documentation uses a system called Sphinx__ to convert from
  plain text to HTML. You'll need to install Sphinx by either downloading
  and installing the package from the Sphinx website, or with ``pip``:

   .. code-block:: console

        $ pip install Sphinx

* Then, just use the included ``Makefile`` to turn the documentation into
  HTML:

  .. code-block:: console

        $ cd path/to/numdifftools/docs
        $ make html

  You'll need `GNU Make`__ installed for this.

  If you're on Windows you can alternatively use the included batch file:

  .. code-block:: bat

        $ cd path\to\numdifftools\docs
        $ make.bat html

* The HTML documentation will be placed in ``docs/_build/html``.


Using pydoc
-----------
The pydoc module automatically generates documentation from Python modules. 
The documentation can be presented as pages of text on the console, served 
to a Web browser, or saved to HTML files.

For modules, classes, functions and methods, the displayed documentation is 
derived from the docstring (i.e. the __doc__ attribute) of the object, and 
recursively of its documentable members. If there is no docstring, pydoc 
tries to obtain a description from the block of comment lines just above the 
definition of the class, function or method in the source file, or at the top 
of the module (see inspect.getcomments()).

The built-in function help() invokes the online help system in the interactive 
interpreter, which uses pydoc to generate its documentation as text on the 
console. The same text documentation can also be viewed from outside the Python 
interpreter by running pydoc as a script at the operating system's command prompt. 
For example, running

.. code-block:: console

    $ pydoc numdifftools


at a shell prompt will display documentation on the numdifftools module, in a style similar 
to the manual pages shown by the Unix man command. The argument to pydoc can be 
the name of a function, module, or package, or a dotted reference to a class, 
method, or function within a module or module in a package. If the argument to 
pydoc looks like a path (that is, it contains the path separator for your 
operating system, such as a slash in Unix), and refers to an existing Python 
source file, then documentation is produced for that file.

You can also use pydoc to start an HTTP server on the local machine that will 
serve documentation to visiting Web browsers. For example, running

.. code-block:: console

    $ pydoc -b 

will start the server and additionally open a web browser to a module index page. 
Each served page has a navigation bar at the top where you can Get help on an 
individual item, Search all modules with a keyword in their synopsis line, and 
go to the Module index, Topics and Keywords pages.
To quit the server just type

.. code-block:: console

    $ quit 




__ http://sphinx-doc.org/
__ https://www.gnu.org/software/make/

