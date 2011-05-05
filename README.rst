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
    median            5.22       2.18       2.20       5.98       4.02       2.86
    nanmedian       115.05      27.65       4.33     148.55      68.91       6.44
    nansum           12.92       6.28       1.70      12.84       7.06       1.73
    nanmax           12.73       6.15       1.68      12.93      10.21       1.70
    nanmean          24.28      13.73       3.00      25.40      28.95       4.98
    nanstd           30.64       9.82       2.73      31.87      17.48       3.76
    nanargmax        10.82       5.92       2.68      11.36       8.93       2.89
    ss                5.26       3.36       1.25       5.30       3.30       1.25
    rankdata         23.20      13.54       9.10      22.95      15.19      10.32
    partsort          1.43       2.20       2.48       1.52       5.18       3.71
    argpartsort       0.78       2.22       1.65       0.68       3.59       1.59
    move_sum         12.24       8.38      14.15      12.10       8.58      13.73
    move_nansum      29.84      20.39      28.79      30.51      25.29      29.04
    move_mean        11.43       4.41      14.27      11.25       8.59      13.83
    move_nanmean     35.99      12.01      29.60      37.25      14.63      30.52
    move_std         18.15       3.42      23.08      24.11      21.19      29.73
    move_nanstd      37.08       6.33      35.05      43.68       7.18      36.23
    move_max          4.33       3.74       9.46       4.90       5.83      11.75
    move_nanmax      23.49       6.44      19.58      27.26      15.29      26.92

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
    median            5.86       2.23       2.27       7.54       4.23       2.88
    nanmedian       139.97      27.79       4.31     191.21      68.61       6.36
    nansum           19.92       6.72       1.70      19.35       7.89       1.72
    nanmax           19.09       6.49       1.68      20.17      11.07       1.70
    nanmean          35.65      14.48       2.98      37.41      30.76       4.96
    nanstd           41.62      10.02       2.70      43.75      17.98       3.71
    nanargmax        16.91       6.23       2.69      17.61       9.58       2.88
    ss                8.66       3.53       1.25       8.72       3.56       1.25
    rankdata         23.58      13.08       8.69      23.63      14.55       9.68
    partsort          1.99       2.22       2.48       2.24       5.36       3.70
    argpartsort       0.95       2.22       1.66       0.86       3.69       1.59
    move_sum         18.47       8.84      14.20      18.20       9.25      13.71
    move_nansum      45.80      21.58      28.88      48.13      27.28      29.28
    move_mean        16.33       4.51      14.30      16.40       9.16      13.87
    move_nanmean     52.45      12.23      29.48      54.01      14.96      30.43
    move_std         23.02       3.44      23.05      32.55      22.25      29.83
    move_nanstd      47.08       6.35      34.96      57.24       7.18      36.13
    move_max          5.75       3.77       9.29       6.72       6.01      11.74
    move_nanmax      30.40       6.54      19.65      36.18      15.62      27.03

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
