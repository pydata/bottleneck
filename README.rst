==========
Bottleneck
==========

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
    
    >>> bn.bench(mode='fast', dtype='float64', axis=1)
    Bottleneck performance benchmark
        Bottleneck  0.6.0
        Numpy (np)  1.6.1
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            4.54       1.80       2.35       4.76       1.96       2.32
    nanmedian       119.48      25.86       5.44     125.15      75.04       7.77
    nansum            8.76       6.48       8.84       8.65       6.02      10.17
    nanmax            2.09       1.20       1.03       2.23       3.58       3.94
    nanmean          21.18      12.42      13.37      22.11      26.26      29.84
    nanstd           31.56       9.48      10.71      32.54      15.77      17.91
    nanargmax         7.34       4.83       9.75       7.29       7.62      13.02
    ss                4.67       2.42       4.59       4.53       2.34       4.62
    rankdata         25.44      16.22      12.71      24.58      19.28      15.52
    partsort          1.16       1.92       2.50       1.21       2.29       2.70
    argpartsort       0.44       1.79       2.13       0.48       1.36       1.51
    replace           4.42       3.99       4.38       4.41       3.99       4.44
    anynan            3.14       4.69       5.25       3.19      19.73     369.03
    move_sum          8.96      10.10      39.90       8.91       9.86      39.42
    move_nansum      22.79      23.82      81.63      23.42      30.94      86.81
    move_mean         9.16       4.01      23.03       9.36      10.36      39.40
    move_nanmean     25.67      10.90      48.02      26.42      12.99      49.51
    move_std         13.47       3.02      18.53      20.01      23.94      79.04
    move_nanstd      26.06       5.60      28.83      28.68       6.24      29.39
    move_max          4.35       2.45      16.71       4.40       5.88      36.40
    move_nanmax      18.31       5.88      32.13      18.94      16.22      67.90

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

    >>> bn.bench(mode='faster', dtype='float64', axis=1)
    Bottleneck performance benchmark
        Bottleneck  0.6.0
        Numpy (np)  1.6.1
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.24       1.81       2.34       6.64       2.10       2.31
    nanmedian       158.79      26.09       5.43     169.00      76.07       7.75
    nansum           14.33       6.91       9.03      14.25       6.46      10.30
    nanmax            3.08       1.26       1.03       3.29       3.68       3.88
    nanmean          32.58      13.20      13.41      33.84      27.69      30.11
    nanstd           42.41      10.18      10.75      45.10      16.06      17.98
    nanargmax        11.19       5.05       9.86      11.09       8.12      13.10
    ss                7.26       2.57       4.65       7.27       2.56       4.67
    rankdata         27.45      16.27      12.82      26.57      19.17      15.89
    partsort          1.69       1.93       2.50       1.90       2.36       2.70
    argpartsort       0.63       1.81       2.14       0.73       1.40       1.51
    replace           6.28       4.11       4.43       6.30       4.11       4.36
    anynan            4.62       5.03       5.29       4.95      28.35     393.84
    move_sum         14.28      10.49      40.37      14.31      10.20      39.56
    move_nansum      35.12      24.81      81.78      35.74      32.06      86.84
    move_mean        13.73       4.06      23.08      14.07      10.52      39.62
    move_nanmean     37.06      11.00      48.08      37.41      13.19      49.39
    move_std         17.35       3.03      18.54      29.42      24.48      79.51
    move_nanstd      33.86       5.64      28.87      38.38       6.30      29.46
    move_max          6.08       2.48      16.72       6.17       5.98      36.51
    move_nanmax      26.25       6.00      32.15      27.11      16.48      68.16

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
Bottleneck               Python 2.6, 2.7, or 3.2; NumPy 1.5.1 or 1.6.1
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

Bottleneck takes a few minutes to build on newer machines. On older machines
it can take a lot longer (one user reported 30 minutes!).

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
    Ran 90 tests in 31.197s
    OK
    <nose.result.TextTestResult run=90 errors=0 failures=0> 
