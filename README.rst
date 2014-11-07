==========
Bottleneck
==========

.. warning:: This version of Bottleneck requires NumPy 1.8.

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

Bottleneck is fast::

    >>> arr = np.random.rand(100, 100)
    >>> timeit np.nansum(arr)
    10000 loops, best of 3: 43.7 us per loop
    >>> timeit bn.nansum(arr)
    100000 loops, best of 3: 7.98 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nansum(arr)
    10000 loops, best of 3: 74 us per loop
    >>> timeit bn.nansum(arr)
    100000 loops, best of 3: 30.6 us per loop

Bottleneck comes with a benchmark suite. To run the benchmark::

    >>> bn.bench(mode='fast', dtype='float64', axis=1)
    Bottleneck performance benchmark
        Bottleneck  0.8.0
        Numpy (np)  1.8.0
        Scipy (sp)  0.13.2
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            8.10       1.04       0.82       8.43       1.70       0.93
    nanmedian       140.72      35.97       7.01     151.70     101.35      13.24
    nansum           16.20       5.49       3.98      15.94       7.83       6.93
    nanmax            7.29       1.98       1.10       7.29       3.49       2.80
    nanmean          34.49       9.23       5.25      34.77      11.82       8.05
    nanstd           53.70       7.82       4.26      54.83      10.12       6.68
    nanargmax        13.76       4.45       4.38      13.75       7.56       7.65
    ss                6.23       2.49       2.40       6.26       2.50       2.38
    rankdata         65.57       8.87       1.46      64.45      21.63       2.17
    partsort          1.21       1.96       2.84       1.33       2.37       4.44
    argpartsort       0.40       1.96       2.22       0.45       1.70       1.42
    replace           3.64       1.87       1.83       3.66       1.87       1.83
    anynan            4.47       2.14       1.50       4.67      10.05     116.63
    move_sum         58.31      54.99      85.41      58.22      57.03      87.54
    move_nansum     168.52     173.22     263.86     167.18     187.21     409.86
    move_mean        98.78      33.89      33.00     102.37      95.10      92.41
    move_nanmean    288.63     105.86     133.04     288.69     110.75     185.44
    move_std        112.53      23.80      35.88     153.28     192.38     291.89
    move_nanstd     307.70      53.50      68.04     341.50      56.74     101.22
    move_max         46.32      24.87      21.93      47.65      68.60      65.75
    move_nanmax     103.47      36.94      33.36     105.58      81.05      90.14

    Reference functions:
    median         np.median
    nanmedian      local copy of sp.stats.nanmedian
    nansum         np.nansum
    nanmax         np.nanmax
    nanmean        local copy of sp.stats.nanmean
    nanstd         local copy of sp.stats.nanstd
    nanargmax      np.nanargmax
    ss             scipy.stats.ss
    rankdata       scipy.stats.rankdata based (axis support added)
    partsort       np.sort, n=max(a.shape[1]/2,1)
    argpartsort    np.argsort, n=max(a.shape[1]/2,1)
    replace        np.putmask based (see bn.slow.replace)
    anynan         np.isnan(arr).any(axis)
    move_sum       sp.ndimage.convolve1d based, window=a.shape[1] // 5
    move_nansum    sp.ndimage.convolve1d based, window=a.shape[1] // 5
    move_mean      sp.ndimage.convolve1d based, window=a.shape[1] // 5
    move_nanmean   sp.ndimage.convolve1d based, window=a.shape[1] // 5
    move_std       sp.ndimage.convolve1d based, window=a.shape[1] // 5
    move_nanstd    sp.ndimage.convolve1d based, window=a.shape[1] // 5
    move_max       sp.ndimage.maximum_filter1d based, window=a.shape[1] // 5
    move_nanmax    sp.ndimage.maximum_filter1d based, window=a.shape[1] // 5

Slow
====

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtype combinations result in calls to slower,
unaccelerated functions.

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

.. warning:: This version of Bottleneck requires NumPy 1.8.

Requirements:

======================== ====================================================
Bottleneck               Python 2.6, 2.7, 3.3; NumPy 1.8
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

Bottleneck takes a few minutes to build on newer machines. On older machines
it can take a lot longer (one user reported 30 minutes!).

GNU/Linux, Mac OS X, et al.
---------------------------

To install Bottleneck::

    $ python setup.py build
    $ sudo python setup.py install

Or, if you wish to specify where Bottleneck is installed, for example inside
``/usr/local``::

    $ python setup.py build
    $ sudo python setup.py install --prefix=/usr/local

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
    Ran 124 tests in 31.197s
    OK
    <nose.result.TextTestResult run=124 errors=0 failures=0>
