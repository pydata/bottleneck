==========
Bottleneck
==========

Introduction
============

Bottleneck is a collection of fast NumPy array functions written in Cython:

===================== =======================================================
NumPy/SciPy           ``median, nanmedian, rankdata, ss, nansum, nanmin,
                      nanmax, nanmean, nanstd, nanargmin, nanargmax`` 
Functions             ``nanrankdata, nanvar, partsort, argpartsort``
Moving window         ``move_sum, move_nansum, move_mean, move_nanmean,
                      move_std, move_nanstd, move_min, move_nanmin, move_max,
                      move_nanmax``
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
    >>> timeit np.nanmax(arr)
    10000 loops, best of 3: 90 us per loop
    >>> timeit bn.nanmax(arr)
    100000 loops, best of 3: 12.6 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nanmax(arr)
    10000 loops, best of 3: 133 us per loop
    >>> timeit bn.nanmax(arr)
    100000 loops, best of 3: 12.6 us per loop

Bottleneck comes with a benchmark suite. To run the benchmark::
    
    >>> bn.bench(mode='fast', dtype='float64', axis=0)
    Bottleneck performance benchmark
        Bottleneck  0.5.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            5.25       2.20       2.27       5.88       4.06       2.87
    nanmedian       128.35      28.31       4.35     152.00      71.91       6.46
    nansum           12.04       6.18       1.71      12.24       7.20       1.71
    nanmax           12.17       6.16       1.68      12.88      10.07       1.70
    nanmean          24.67      13.85       2.99      25.86      29.22       4.97
    nanstd           30.77       9.73       2.69      31.87      17.86       3.70
    nanargmax        11.48       5.78       2.65      11.42       8.82       2.85
    ss                5.27       3.31       1.25       5.27       3.30       1.25
    rankdata         22.14      13.03       8.62      22.16      14.67       9.62
    partsort          1.42       2.20       2.48       1.57       5.05       3.70
    argpartsort       0.78       2.20       1.67       0.67       3.45       1.60
    move_sum         11.91       8.64      14.25      11.78       8.73      13.79
    move_nansum      29.90      19.47      28.85      30.40      25.27      29.40
    move_mean        11.18       4.43      14.37      11.03       8.63      13.97
    move_nanmean     35.93      12.11      29.77      37.45      14.67      30.69
    move_std         18.38       3.42      23.01      24.21      20.98      29.72
    move_nanstd      37.21       6.31      34.89      43.67       7.14      36.14
    move_max          4.35       3.73       9.32       4.79       5.87      11.73
    move_nanmax      22.96       6.40      19.58      27.06      15.19      26.97

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
    partsort       np.sort, n=max(a.shape[0]/2,1)
    argpartsort    np.argsort, n=max(a.shape[0]/2,1)
    move_sum       sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nansum    sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_mean      sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nanmean   sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_std       sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nanstd    sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_max       sp.ndimage.maximum_filter1d based, window=a.shape[0]/5
    move_nanmax    sp.ndimage.maximum_filter1d based, window=a.shape[0]/5

Faster
======

Under the hood Bottleneck uses a separate Cython function for each combination
of ndim, dtype, and axis. A lot of the overhead in bn.nanmax(), for example,
is in checking that the axis is within range, converting non-array data to an
array, and selecting the function to use to calculate the maximum.

You can get rid of the overhead by doing all this before you, say, enter
an inner loop::

    >>> arr = np.random.rand(10,10)
    >>> func, a = bn.func.nanmax_selector(arr, axis=0)
    >>> func
    <built-in function nanmax_2d_float64_axis0> 

Let's see how much faster than runs::
    
    >>> timeit np.nanmax(arr, axis=0)
    10000 loops, best of 3: 24.7 us per loop
    >>> timeit bn.nanmax(arr, axis=0)
    100000 loops, best of 3: 2.1 us per loop
    >>> timeit func(a)
    100000 loops, best of 3: 1.47 us per loop

Note that ``func`` is faster than Numpy's non-NaN version of max::
    
    >>> timeit arr.max(axis=0)
    100000 loops, best of 3: 4.78 us per loop

So adding NaN protection to your inner loops comes at a negative cost!

Benchmarks for the low-level Cython functions::

    >>> bn.bench(mode='faster', dtype='float64', axis=0)
    Bottleneck performance benchmark
        Bottleneck  0.5.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.58       2.24       2.28       7.39       4.25       2.89
    nanmedian       159.56      28.57       4.35     184.05      74.48       6.42
    nansum           20.07       6.70       1.70      20.09       7.93       1.71
    nanmax           18.39       6.44       1.68      19.83      10.83       1.69
    nanmean          37.21      14.46       3.00      38.79      30.73       4.98
    nanstd           42.23       9.96       2.68      44.62      18.35       3.69
    nanargmax        17.15       6.00       2.67      17.45       9.42       2.87
    ss                8.88       3.64       1.25       8.91       3.60       1.24
    rankdata         23.42      12.99       8.51      23.68      14.60       9.53
    partsort          2.03       2.22       2.48       2.37       5.25       3.71
    argpartsort       1.12       2.22       1.67       0.92       3.57       1.59
    move_sum         16.26       8.97      14.19      16.05       9.21      13.45
    move_nansum      39.62      21.51      29.08      40.93      26.15      28.98
    move_mean        14.49       4.49      14.31      14.58       9.08      13.84
    move_nanmean     44.82      12.25      29.59      46.38      14.92      30.18
    move_std         20.66       3.44      22.94      28.44      21.70      29.65
    move_nanstd      41.16       6.35      34.94      49.21       7.21      36.04
    move_max          5.30       3.80       9.34       6.20       6.01      11.81
    move_nanmax      27.02       6.53      19.63      32.31      15.23      27.03

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
    partsort       np.sort, n=max(a.shape[0]/2,1)
    argpartsort    np.argsort, n=max(a.shape[0]/2,1)
    move_sum       sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nansum    sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_mean      sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nanmean   sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_std       sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nanstd    sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_max       sp.ndimage.maximum_filter1d based, window=a.shape[0]/5
    move_nanmax    sp.ndimage.maximum_filter1d based, window=a.shape[0]/5

Slow
====

Currently only 1d, 2d, and 3d input arrays with data type (dtype) int32,
int64, float32, and float64 are accelerated. All other ndim/dtype
combinations result in calls to slower, unaccelerated functions.

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
 mailing list 2       http://mail.scipy.org/mailman/listinfo/scipy-user
===================   ========================================================

Install
=======

Requirements:

======================== ====================================================
Bottleneck               Python, NumPy 1.5.1
Unit tests               nose
Compile                  gcc or MinGW
Optional                 SciPy 0.8.0 or 0.9.0 (portions of benchmark)
======================== ====================================================

Directions for installing a *released* version of Bottleneck (i.e., one
obtained from http://pypi.python.org/pypi/Bottleneck) are given below. Cython
is not required since the Cython files have already been converted to C source
files. (If you obtained bottleneck directly from the repository, then you will
need to generate the C source files using the included Makefile which requires
Cython.)

**GNU/Linux, Mac OS X, et al.**

To install Bottleneck::

    $ python setup.py build
    $ sudo python setup.py install
    
Or, if you wish to specify where Bottleneck is installed, for example inside
``/usr/local``::

    $ python setup.py build
    $ sudo python setup.py install --prefix=/usr/local

**Windows**

You can compile Bottleneck using the instructions below or you can use the
Windows binaries created by Christoph Gohlke:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck

In order to compile the C code in Bottleneck you need a Windows version of the
gcc compiler. MinGW (Minimalist GNU for Windows) contains gcc.

Install MinGW and add it to your system path. Then install Bottleneck with the
commands::

    python setup.py build --compiler=mingw32
    python setup.py install

**Post install**

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 77 tests in 49.602s
    OK
    <nose.result.TextTestResult run=77 errors=0 failures=0> 
