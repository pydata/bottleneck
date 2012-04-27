==========
Bottleneck
==========

Introduction
============

Bottleneck is a collection of fast NumPy array functions written in Cython:

===================== =======================================================
NumPy/SciPy           ``median, nanmedian, rankdata, ss, nansum, nanmin,
                      nanmax, nanmean, nanstd, nanargmin, nanargmax`` 
Functions             ``nanrankdata, nanvar, partsort, argpartsort, replace``
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
        Bottleneck  0.6.0
        Numpy (np)  1.6.1
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            4.73       2.02       1.92       4.74       2.49       2.12
    nanmedian       122.28      28.38       3.47     134.84      86.42       5.30
    nansum            8.93       5.84       1.70       8.93       6.21       1.73
    nanmax            1.94       1.21       1.01       2.11       3.50       1.07
    nanmean          22.16      12.72       2.31      22.96      23.99       3.90
    nanstd           28.21       8.42       2.25      30.01      15.28       3.05
    nanargmax         8.30       5.23       2.58       8.34       7.68       2.80
    ss                4.69       2.39       1.21       4.71       2.39       1.21
    rankdata         23.34      13.59       7.65      22.45      15.83       8.73
    partsort          1.24       2.00       2.11       1.42       3.01       2.74
    argpartsort       0.52       2.09       1.45       0.58       2.25       1.19
    replace           3.96       3.94       3.88       3.96       3.94       3.89
    move_sum          9.33       8.48      11.26       9.32       9.00      10.69
    move_nansum      23.22      19.50      22.20      23.83      25.34      22.63
    move_mean         8.57       3.44      11.05       8.84       9.43      10.71
    move_nanmean     26.57       9.51      23.21      27.30      11.66      23.71
    move_std         14.01       2.65      18.15      19.97      20.51      22.82
    move_nanstd      26.99       4.98      28.06      29.05       5.62      29.14
    move_max          4.26       2.14      10.04       4.39       5.22      13.48
    move_nanmax      17.84       4.98      18.57      18.16      14.57      25.09

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
    replace        np.putmask based (see bn.slow.replace)
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
        Bottleneck  0.6.0
        Numpy (np)  1.6.1
        Scipy (sp)  0.9.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.28       2.03       1.95       6.78       2.56       2.13
    nanmedian       163.01      28.58       3.47     173.04      87.96       5.30
    nansum           13.32       6.17       1.71      13.24       6.56       1.72
    nanmax            2.98       1.30       1.01       3.22       3.65       1.07
    nanmean          32.89      13.43       2.31      34.09      25.31       3.91
    nanstd           39.15       8.60       2.25      41.14      14.09       3.05
    nanargmax        12.48       5.49       2.60      12.50       8.19       2.82
    ss                7.47       2.57       1.21       7.45       2.55       1.21
    rankdata         24.97      13.72       7.55      23.96      15.92       8.65
    partsort          1.92       2.80       2.12       2.11       3.09       2.75
    argpartsort       0.75       2.13       1.45       0.84       2.33       1.19
    replace           5.79       4.06       3.90       5.91       3.69       3.91
    move_sum         14.13       8.78      10.86      14.07       9.42      10.70
    move_nansum      35.94      20.68      22.10      36.80      25.56      22.73
    move_mean        12.81       3.50      11.00      13.29       9.74      10.84
    move_nanmean     39.65       9.69      23.12      40.31      11.83      23.66
    move_std         17.05       2.66      18.16      27.04      21.06      22.71
    move_nanstd      34.39       5.03      28.07      37.87       5.68      29.08
    move_max          6.17       2.15      10.17       6.37       5.21      13.55
    move_nanmax      26.29       5.04      18.63      26.45      14.79      25.06

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
    replace        np.putmask based (see bn.slow.replace)
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
Bottleneck               Python 2.6, 2.7, or 3.2; NumPy 1.5.1 or 1.6.0
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
    Ran 84 tests in 28.169s
    OK
    <nose.result.TextTestResult run=84 errors=0 failures=0> 
