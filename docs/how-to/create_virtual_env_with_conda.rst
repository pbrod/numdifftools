.. _create_virtual_environments:


========================================================
How to create virtual environments for python with conda
========================================================

In this section we will explain how to work with virtual environments
using conda. A virtual environment is a named, isolated, working copy 
of Python that maintains its own files, directories, and paths so that 
you can work with specific versions of libraries or Python itself 
without affecting other Python projects. Virtual environments make it 
easy to cleanly separate different projects and avoid problems with 
different dependencies and version requirements across components. 
The conda command is the preferred interface for managing installations 
and virtual environments with the Anaconda Python distribution. If you 
have a vanilla Python installation or other Python distribution see 
virtualenv.

In the following we assume that the Anaconda Python distribution 
installed and accessible.



Check conda is installed and in your PATH.
------------------------------------------
Open a terminal client.
Enter ``conda -V`` into the terminal command line and press enter.
If conda is installed you should see somehting like the following.


.. parsed-literal::

    $ conda -V
    conda 4.6.8


Check conda is up to date.
--------------------------
In the terminal client enter

.. parsed-literal::

    conda update conda

Update any packages if necessary by typing ``y`` to proceed.


Create a virtual environment for your project.
----------------------------------------------
In the terminal client enter the following where yourenvname is the 
name you want to call your environment, and replace x.x with the Python 
version you wish to use. (To see a list of available python versions 
first, type ``conda search "^python$"`` and press enter.)

.. parsed-literal::

    conda create -n yourenvname python=x.x anaconda

Press ``y`` to proceed. This will install the Python version and all 
the associated anaconda packaged libraries at 
path_to_your_anaconda_location/anaconda/envs/yourenvname


Activate your virtual environment.
----------------------------------
To activate or switch into your virtual environment, simply type the 
following where yourenvname is the name you gave to your environement 
at creation.

.. parsed-literal::

    conda activate yourenvname

Activating a conda environment modifies the PATH and shell variables 
to point to the specific isolated Python set-up you created. The 
command prompt will change to indicate which conda environemnt you 
are currently in by prepending (yourenvname). To see a list of all 
your environments, use the command ``conda info -e``.


Install additional Python packages to a virtual environment.
------------------------------------------------------------
To install additional packages only to your virtual environment, 
enter the following command where yourenvname is the name of your 
environemnt, and [package] is the name of the package you wish to 
install. Failure to specify ``-n yourenvname`` will install the 
package to the root Python installation.

.. parsed-literal::

    conda install -n yourenvname [package]


Deactivate your virtual environment.
------------------------------------
To end a session in the current environment, enter the following. 
There is no need to specify the envname - which ever is currently 
active will be deactivated, and the PATH and shell variables will 
be returned to normal.

.. parsed-literal::

    conda deactivate


Delete a no longer needed virtual environment.
----------------------------------------------
To delete a conda environment, enter the following, where yourenvname 
is the name of the environment you wish to delete.

.. parsed-literal::

    conda remove -n yourenvname -all


Related info.
-------------
The offical conda documentation can be found here: 
https://conda.io/projects/conda/en/latest/user-guide/overview.html
https://conda.io/projects/conda/en/latest/user-guide/getting-started.html.

