==========
Bottleneck
==========

Introduction
============

Bottleneck is a collection of fast NumPy array functions written in Cython:

===================== =======================================================
NumPy/SciPy           ``median, nanmedian, rankdata, nansum, nanmin, nanmax,
                      nanmean, nanstd, nanargmin, nanargmax`` 
Functions             ``nanrankdata, nanvar``
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
        Bottleneck  0.4.3
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            5.13       2.18       2.26       6.18       4.06       2.89
    nanmedian       123.05      28.56       4.37     153.27      73.41       6.42
    nansum           12.35       6.33       1.72      12.72       7.23       1.70
    nanmax           12.57       6.25       1.68      13.70      10.40       1.68
    nanmean          24.24      13.72       3.00      25.13      29.74       4.87
    nanstd           29.77       9.63       2.63      31.46      17.74       3.65
    nanargmax        11.55       5.76       2.62      12.04       8.80       2.81
    rankdata         22.70      12.89       8.65      22.41      14.57       9.65
    move_sum         10.29       7.99      14.46      11.12       8.41      13.87
    move_nansum      29.09      18.81      29.31      28.62      24.58      29.62
    move_mean        10.01       4.24      14.49      10.16       7.80      14.09
    move_nanmean     31.90      11.75      29.38      32.71      14.49      31.03
    move_std         16.92       3.28      22.77      21.25      20.10      29.58
    move_nanstd      33.67       6.21      34.67      38.34       7.03      36.06
    move_max          3.95       3.72       9.35       4.91       5.50      11.74
    move_nanmax      22.77       6.28      19.57      24.26      14.73      26.61

    Reference functions:
    median         np.median
    nanmedian      local copy of sp.stats.nanmedian
    nansum         np.nansum
    nanmax         np.nanmax
    nanmean        local copy of sp.stats.nanmean
    nanstd         local copy of sp.stats.nanstd
    nanargmax      np.nanargmax
    rankdata       scipy.stats.rankdata based (axis support added)
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
        Bottleneck  0.4.3
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.69       2.20       2.28       8.05       4.20       2.90
    nanmedian       153.28      28.59       4.37     202.89      75.00       6.46
    nansum           19.81       6.64       1.74      19.93       7.78       1.71
    nanmax           19.52       6.42       1.69      21.98      11.03       1.69
    nanmean          37.07      14.51       2.98      39.31      30.74       5.00
    nanstd           42.13       9.84       2.64      44.11      18.21       3.66
    nanargmax        17.91       6.01       2.65      18.81       9.42       2.82
    rankdata         24.50      12.73       8.59      24.32      14.56       9.77
    move_sum         17.19       8.56      14.68      17.17       8.90      14.07
    move_nansum      46.11      21.13      29.80      47.85      26.42      29.95
    move_mean        16.35       4.27      14.30      16.58       8.96      14.23
    move_nanmean     47.75      11.91      29.89      50.23      14.61      30.91
    move_std         22.55       3.36      22.85      31.19      21.64      29.86
    move_nanstd      46.23       6.20      34.91      54.68       7.07      35.87
    move_max          5.64       3.69       9.40       6.72       5.71      11.83
    move_nanmax      29.37       6.38      19.61      36.51      14.68      27.15

    Reference functions:
    median         np.median
    nanmedian      local copy of sp.stats.nanmedian
    nansum         np.nansum
    nanmax         np.nanmax
    nanmean        local copy of sp.stats.nanmean
    nanstd         local copy of sp.stats.nanstd
    nanargmax      np.nanargmax
    rankdata       scipy.stats.rankdata based (axis support added)
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
Optional                 SciPy 0.8.0 (portions of benchmark)
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
    Ran 68 tests in 42.457s
    OK
    <nose.result.TextTestResult run=68 errors=0 failures=0> 
