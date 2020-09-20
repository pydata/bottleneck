.. _installing:

Installing Bottleneck
=====================

As bottleneck aims to provide high-performance, optimized numerical functions
to all users, it is distributed as a source package (except via Anaconda) so
that local compilers can perform the relevant optimizations. Accordingly,
installation may take some additional steps compared to packages like numpy.

Anaconda
~~~~~~~~

If you wish to avoid additional steps, we recommend using Anaconda or
Miniconda. A pre-compiled version of bottleneck is installed by default.
Users looking for optimal performance may benefit from uninstalling the
pre-compiled version and following the steps below.

Build dependencies
~~~~~~~~~~~~~~~~~~

Debian & Ubuntu
---------------

The following build packages must be installed prior to installing bottleneck:

.. code-block::

   sudo apt install gcc python3-dev

The Python development headers can be excluded if using Anaconda.

RHEL, Fedora & CentOS
---------------------

.. code-block::

   sudo yum install gcc python3-devel

Windows
-------

The Python Wiki maintains detailed instructions on which Visual Studio
version to install here: https://wiki.python.org/moin/WindowsCompilers


pip & setuptools
~~~~~~~~~~~~~~~~

bottleneck leverages :pep:`517` and thus we generally recommend updating
pip and setuptools before installing to leverage recent improvements.

With Anaconda:

.. code-block::

   conda update setuptools pip

And with pip:

.. code-block::

   pip install --upgrade setuptools pip


Installation
~~~~~~~~~~~~

Finally, simply install with:

.. code-block::

   pip install bottleneck

If you encounter any errors, please open an issue on our GitHub
page: https://github.com/pydata/bottleneck/issues
