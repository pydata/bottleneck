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
    10000 loops, best of 3: 48.7 us per loop
    >>> timeit bn.nansum(arr)
    100000 loops, best of 3: 7.73 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nansum(arr)
    10000 loops, best of 3: 73.2 us per loop
    >>> timeit bn.nansum(arr)
    100000 loops, best of 3: 28.2 us per loop

Bottleneck comes with a benchmark suite. To run the benchmark::
    
    >>> bn.bench(mode='fast', dtype='float64', axis=1)
    Bottleneck performance benchmark
        Bottleneck  0.6.0
        Numpy (np)  1.6.2
        Scipy (sp)  0.10.1
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            4.47       1.82       2.57       5.08       2.04       2.88
    nanmedian       119.55      26.41       6.20     126.24      74.82      11.09
    nansum            9.01       5.85       5.82       8.94       6.08       6.86
    nanmax            1.96       1.30       1.05       2.06       3.54       3.93
    nanmean          21.48      12.91      10.12      22.19      24.69      25.19
    nanstd           30.96       8.62       8.97      32.84      14.26      16.38
    nanargmax         7.72       4.86       6.49       7.72       7.81       9.20
    ss                4.31       2.38       2.37       4.30       2.38       2.38
    rankdata         25.82      16.17      12.42      24.85      19.26      15.13
    partsort          1.14       1.90       2.78       1.26       2.43       3.83
    argpartsort       0.42       2.07       2.20       0.44       1.89       1.59
    replace           4.02       3.97       4.12       4.10       3.83       4.10
    anynan            3.09       4.64       4.90       3.31      20.98     328.64
    move_sum          8.99       9.86      85.08       8.86      10.04      84.90
    move_nansum      22.96      23.17     173.55      23.80      29.69     188.18
    move_mean         9.10       3.79      32.91       9.17       9.85      84.95
    move_nanmean     27.29      10.24      69.15      28.57      12.40      71.43
    move_std         13.44       2.88      21.21      19.95      23.64     164.76
    move_nanstd      27.56       5.32      32.96      30.65       5.94      33.55
    move_max          4.40       2.08      19.37       4.53       5.04      55.54
    move_nanmax      18.13       5.49      38.82      19.00      15.41     110.81

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
    <function nansum_2d_float64_axis0>

Let's see how much faster than runs::
    
    >>> timeit np.nansum(arr, axis=0)
    10000 loops, best of 3: 11 us per loop
    >>> timeit bn.nansum(arr, axis=0)
    100000 loops, best of 3: 1.2 us per loop
    >>> timeit func(a)
    100000 loops, best of 3: 902 ns per loop

Note that ``func`` is faster than Numpy's non-NaN version of sum::
    
    >>> timeit arr.sum(axis=0)
    100000 loops, best of 3: 1.66 us per loop

So, in this example, adding NaN protection to your inner loop comes at a
negative cost!

Benchmarks for the low-level Cython functions::

    >>> bn.bench(mode='faster', dtype='float64', axis=1)
    Bottleneck performance benchmark
        Bottleneck  0.6.0
        Numpy (np)  1.6.2
        Scipy (sp)  0.10.1
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.20       1.83       2.58       6.65       2.10       2.92
    nanmedian       155.59      26.29       6.22     161.74      77.36      11.18
    nansum           14.00       6.24       5.90      13.54       6.50       6.83
    nanmax            2.96       1.37       1.05       3.01       3.73       3.92
    nanmean          32.04      13.58      10.17      31.91      26.18      24.21
    nanstd           44.75       8.82       9.01      45.96      14.70      16.23
    nanargmax        11.09       5.08       6.50      11.37       8.26       9.50
    ss                7.47       2.61       2.38       7.48       2.58       2.37
    rankdata         27.83      16.31      12.51      26.78      19.31      15.19
    partsort          1.65       1.91       2.78       1.92       2.51       3.82
    argpartsort       0.61       2.09       2.18       0.68       1.90       1.55
    replace           5.97       3.98       4.14       5.94       3.98       4.15
    anynan            4.68       4.91       4.92       4.98      29.22     357.80
    move_sum         14.21      10.17      85.38      14.08      10.41      85.66
    move_nansum      34.51      23.79     174.52      35.39      30.54     190.32
    move_mean        13.17       3.87      32.95      13.51      10.01      85.28
    move_nanmean     40.06      10.46      69.29      41.68      12.50      71.54
    move_std         17.37       2.89      21.24      29.49      24.41     165.67
    move_nanstd      34.36       5.37      33.03      39.60       6.00      33.61
    move_max          6.25       2.10      19.38       6.40       5.08      55.66
    move_nanmax      26.49       5.56      38.77      27.93      15.73     111.53

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
===================   ========================================================

Install
=======

Requirements:

======================== ====================================================
Bottleneck               Python 2.6, 2.7, 3.2; NumPy 1.5.1, 1.6.1, 1.6.2
Unit tests               nose
Compile                  gcc or MinGW
Optional                 SciPy 0.8, 0.9, 0.10 (portions of benchmark)
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
    Ran 91 tests in 31.197s
    OK
    <nose.result.TextTestResult run=91 errors=0 failures=0> 
