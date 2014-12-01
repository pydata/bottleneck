==========
Bottleneck
==========

.. warning:: This version of Bottleneck requires NumPy 1.9.1.

Introduction
============

Bottleneck is a collection of fast NumPy array functions written in Cython:

===================== =======================================================
NumPy/SciPy           ``median, nanmedian, rankdata, ss, nansum, nanmin,
                      nanmax, nanmean, nanstd, nanargmin, nanargmax``
Functions             ``nanrankdata, nanvar, partsort, argpartsort, replace,
                      nn, anynan, allnan``
Moving window         ``move_sum, move_nansum, move_mean, move_nanmean,
                      move_median, move_std, move_nanstd, move_min,
                      move_nanmin, move_max, move_nanmax``
===================== =======================================================

Let's give it a try. Create a NumPy array::

    >>> import numpy as np
    >>> arr = np.array([1, 2, np.nan, 4, 5])

Find the nanmean::

    >>> import bottleneck as bn
    >>> bn.nanmean(arr)
    3.0

Moving window nanmean::

    >>> bn.move_nanmean(arr, window=2)
    array([ nan,  1.5,  2. ,  4. ,  4.5])

Fast
====

Bottleneck comes with a benchmark suite. To run the benchmark::

    >>> bn.bench()
    Bottleneck performance benchmark
        Bottleneck  0.9.0dev
        Numpy (np)  1.9.1
        Scipy (sp)  0.14.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN     no NaN      NaN        NaN        NaN
                     (10,)     (10,10)  (1000,1000)   (10,)     (10,10)  (1000,1000)
    nansum           37.03      19.79       3.86      36.10      20.69       8.97
    nanmean         145.23      77.49       5.18     127.50      67.47      10.33
    nanstd          229.38      91.83       4.24     229.26      97.31       8.51
    nanmax           29.79      13.67       1.07      29.49      13.80       2.84
    move_mean        26.80      19.79      32.17      29.06      20.20      69.31
    move_nanmean     72.88      53.14      66.42      75.15      54.80      66.57

    Reference functions:
    nansum         np.nansum
    nanmean        np.nanmean
    nanstd         np.nanstd
    nanmax         np.nanmax
    move_mean      sp.ndimage.convolve1d based, window=a.shape[-1] // 5
    move_nanmean   sp.ndimage.convolve1d based, window=a.shape[-1] // 5

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtypes result in calls to slower, unaccelerated
functions.

License
=======

Bottleneck is distributed under a Simplified BSD license. Parts of NumPy,
Scipy and numpydoc, all of which have BSD licenses, are included in
Bottleneck. See the LICENSE file, which is distributed with Bottleneck, for
details.

URLs
====

===================   ========================================================
 download             http://pypi.python.org/pypi/Bottleneck
 docs                 http://berkeleyanalytics.com/bottleneck
 code                 http://github.com/kwgoodman/bottleneck
 mailing list         http://groups.google.com/group/bottle-neck
===================   ========================================================

Install
=======

.. warning:: This version of Bottleneck requires NumPy 1.9.1.

Requirements:

======================== ====================================================
Bottleneck               Python 2.6, 2.7, 3.3; NumPy 1.9.1
Compile                  gcc, clang, MinGW
Unit tests               nose
======================== ====================================================

Optional:

======================== ====================================================
SciPy                    portions of benchmark suite
tox, virtualenv          run unit tests across multiple python/numpy versions
======================== ====================================================

Directions for installing a *released* version of Bottleneck (i.e., one
obtained from http://pypi.python.org/pypi/Bottleneck) are given below. Cython
is not required since the Cython files have already been converted to C source
files. (If you obtained bottleneck directly from the repository, then you will
need to generate the C source files using the included Makefile which requires
Cython.)

GNU/Linux, Mac OS X, et al.
---------------------------

To install Bottleneck::

    $ python setup.py build
    $ sudo python setup.py install

Windows
-------

You can compile Bottleneck using the instructions below or you can use the
Windows binaries created by Christoph Gohlke:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck

In order to compile the C code in Bottleneck you need a Windows version of the
gcc compiler. MinGW (Minimalist GNU for Windows) contains gcc.

Install MinGW and add it to your system path. Then install Bottleneck with the
commands::

    python setup.py build --compiler=mingw32
    python setup.py install

Post install
------------

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 24 tests in 20.712s
    OK
    <nose.result.TextTestResult run=24 errors=0 failures=0>
