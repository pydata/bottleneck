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
    10000 loops, best of 3: 82.8 us per loop
    >>> timeit bn.nansum(arr)
    100000 loops, best of 3: 16.9 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nansum(arr)
    10000 loops, best of 3: 126 us per loop
    >>> timeit bn.nansum(arr)
    100000 loops, best of 3: 63.1 us per loop

Bottleneck comes with a benchmark suite. To run the benchmark::
    
    >>> bn.bench(mode='fast', dtype='float64', axis=0)
    Bottleneck performance benchmark
        Bottleneck  0.5.0
        Numpy (np)  1.6.0
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            5.12       2.12       2.23       5.54       4.48       2.95
    nanmedian       115.31      28.86       4.23     140.78      70.29       6.49
    nansum           10.14       4.88       1.69      10.21       5.75       1.68
    nanmax            2.26       1.31       1.01       2.51       3.98       1.09
    nanmean          25.02      10.20       2.38      26.41      25.22       4.41
    nanstd           32.22       7.47       2.34      33.76      15.43       3.38
    nanargmax        10.25       5.48       2.59      10.63       8.10       2.78
    ss                5.51       2.55       1.29       5.48       2.62       1.23
    rankdata         22.65      13.36       8.81      23.02      14.44       9.91
    partsort          1.52       2.13       2.43       1.75       5.96       3.79
    argpartsort       0.77       2.15       1.68       0.75       3.71       1.70
    move_sum         10.52       8.33      14.25      10.24       8.90      13.84
    move_nansum      25.53      20.06      29.20      26.02      25.46      29.38
    move_mean        10.84       4.42      14.38      11.10       8.82      13.93
    move_nanmean     32.50      11.64      29.74      34.26      14.34      30.49
    move_std         17.77       3.43      22.94      23.19      21.07      29.77
    move_nanstd      35.07       6.03      34.87      40.64       6.92      36.10
    move_max          4.32       3.73       9.32       4.77       5.86      11.81
    move_nanmax      19.18       6.43      19.70      22.74      15.41      27.09

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
    >>> func, a = bn.func.nansum_selector(arr, axis=0)
    >>> func
    <built-in function nansum_2d_float64_axis0> 

Let's see how much faster than runs::
    
    >>> timeit np.nansum(arr, axis=0)
    10000 loops, best of 3: 20.4 us per loop
    >>> timeit bn.nansum(arr, axis=0)
    100000 loops, best of 3: 2.05 us per loop
    >>> timeit func(a)
    100000 loops, best of 3: 1.14 us per loop

Note that ``func`` is faster than Numpy's non-NaN version of sum::
    
    >>> timeit arr.sum(axis=0)
    100000 loops, best of 3: 3.03 us per loop

So, in this example, adding NaN protection to your inner loop comes at a
negative cost!

Benchmarks for the low-level Cython functions::

    >>> bn.bench(mode='faster', dtype='float64', axis=0)
    Bottleneck performance benchmark
        Bottleneck  0.5.0
        Numpy (np)  1.6.0
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.99       2.14       2.23       8.11       4.57       2.95
    nanmedian       155.70      29.55       4.25     199.93      73.12       6.48
    nansum           16.06       5.14       1.69      16.17       6.07       1.68
    nanmax            3.55       1.35       1.01       4.11       4.29       1.09
    nanmean          37.91      10.77       2.37      38.41      26.50       4.41
    nanstd           45.15       7.71       2.36      47.14      15.85       3.38
    nanargmax        15.12       5.64       2.64      15.74       8.74       2.82
    ss                8.46       2.72       1.21       8.34       2.71       1.22
    rankdata         24.46      13.16       8.94      24.32      14.39       9.82
    partsort          2.33       2.17       2.44       2.82       6.23       3.80
    argpartsort       1.15       2.17       1.67       1.15       3.81       1.67
    move_sum         16.89       8.80      14.17      16.76       9.18      13.80
    move_nansum      42.53      21.27      28.58      44.57      26.91      29.30
    move_mean        15.70       4.55      14.32      15.96       9.01      13.86
    move_nanmean     49.32      11.71      29.30      50.52      14.58      30.32
    move_std         22.64       3.45      22.96      32.23      22.24      29.30
    move_nanstd      46.44       6.10      34.35      56.45       6.91      36.26
    move_max          5.86       3.80       9.32       6.98       6.05      11.70
    move_nanmax      26.30       6.38      19.52      32.37      15.48      27.06

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
Bottleneck               Python 2.5, 2.6, 2.7; NumPy 1.5.1 or 1.6.0
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
    Ran 80 tests in 49.602s
    OK
    <nose.result.TextTestResult run=80 errors=0 failures=0> 
