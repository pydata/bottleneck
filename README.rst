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
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            5.03       2.18       2.28       5.77       4.08       2.89
    nanmedian       120.51      29.31       4.40     151.19      75.39       6.56
    nansum           11.79       6.29       1.71      11.77       7.32       1.69
    nanmax           12.31       6.22       1.67      13.25      10.42       1.68
    nanmean          23.02      13.81       2.98      23.71      29.02       5.10
    nanstd           27.88       9.66       2.66      29.18      17.78       3.67
    nanargmax        10.97       5.87       2.65      11.25       8.73       2.81
    rankdata         22.46      12.85       8.76      22.22      14.41       9.84
    move_sum         11.34       8.55      14.29      11.05       9.03      13.84
    move_nansum      29.18      20.42      29.05      29.58      25.72      29.40
    move_mean        10.66       4.41      14.36      10.47       8.90      13.95
    move_nanmean     31.35      12.07      29.77      32.50      15.50      30.61
    move_std         17.40       3.42      22.98      22.37      21.23      29.89
    move_nanstd      34.71       6.36      34.77      40.26       7.14      36.14
    move_max          3.98       3.74       9.30       4.62       5.87      11.68
    move_nanmax      21.64       6.47      19.59      24.89      15.40      26.98

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
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.07       2.20       2.29       7.51       4.13       2.89
    nanmedian       145.48      29.33       4.41     201.33      77.24       6.54
    nansum           20.22       6.66       1.72      20.02       7.95       1.69
    nanmax           19.31       6.48       1.69      21.36      11.07       1.66
    nanmean          37.43      14.38       3.00      39.66      30.58       5.01
    nanstd           42.84       9.90       2.85      44.79      18.23       3.82
    nanargmax        17.83       6.06       2.68      18.13       9.54       2.85
    rankdata         23.62      12.63       8.83      23.59      14.30       9.87
    move_sum         17.53       8.99      13.94      17.01       9.47      13.74
    move_nansum      47.25      21.71      29.44      49.74      27.34      28.95
    move_mean        16.45       4.48      14.27      16.19       9.08      13.85
    move_nanmean     49.29      12.22      29.45      51.56      14.97      30.25
    move_std         22.57       3.46      23.01      31.58      22.34      29.68
    move_nanstd      46.73       6.34      34.83      56.56       7.18      35.95
    move_max          5.69       3.81       9.25       6.70       5.97      11.73
    move_nanmax      29.97       6.48      19.57      35.73      15.92      27.10

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
    Ran 68 tests in 42.457s
    OK
    <nose.result.TextTestResult run=68 errors=0 failures=0> 
