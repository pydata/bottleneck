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
        Bottleneck  0.7.0
        Numpy (np)  1.6.2
        Scipy (sp)  0.11.0rc2
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            5.00       1.81       2.58       4.84       2.06       2.97
    nanmedian       127.23      27.24       6.34     124.99      78.77      11.64
    nansum            9.20       5.65       5.51       9.14       6.02       6.44
    nanmax            2.13       1.30       1.04       2.24       3.57       3.95
    nanmean          22.34      12.05       9.54      22.86      24.35      23.37
    nanstd           31.96       8.29       8.74      33.60      14.15      15.93
    nanargmax         8.02       4.76       6.04       8.08       7.87       9.23
    ss                4.33       2.39       2.28       4.32       2.41       2.30
    rankdata         23.57      14.67      11.49      22.97      17.01      13.84
    partsort          1.24       1.90       2.78       1.23       2.43       3.93
    argpartsort       0.49       2.10       2.17       0.51       1.88       1.53
    replace           3.94       3.58       3.83       3.99       3.59       3.84
    anynan            3.08       4.39       4.58       3.15      18.35     298.50
    move_sum          9.12       8.60      84.22       9.13       8.63      83.83
    move_nansum      22.97      20.35     171.77      23.56      26.79     188.21
    move_mean         9.40       3.38      32.21       9.72       9.14      88.76
    move_nanmean     27.01       9.33      67.85      27.55      11.38      70.20
    move_std         13.78       2.57      20.67      19.83      19.66     155.89
    move_nanstd      26.77       4.84      32.11      29.75       5.43      32.44
    move_max          4.33       2.08      19.46       4.43       5.02      55.37
    move_nanmax      18.86       5.17      38.49      19.49      14.22     108.31

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
        Bottleneck  0.7.0
        Numpy (np)  1.6.2
        Scipy (sp)  0.11.0rc2
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.09       1.82       2.58       6.55       2.12       2.99
    nanmedian       150.87      27.36       6.33     161.43      79.39      11.87
    nansum           13.52       5.83       5.42      13.37       6.28       6.43
    nanmax            3.06       1.37       1.06       3.20       3.71       3.91
    nanmean          31.58      12.45       9.52      31.62      25.92      24.40
    nanstd           42.03       8.41       8.64      44.48      14.47      16.05
    nanargmax        11.06       4.88       6.05      10.90       8.26       9.19
    ss                6.73       2.53       2.32       6.71       2.53       2.31
    rankdata         25.02      14.65      11.50      24.24      16.89      13.87
    partsort          1.60       1.90       2.79       1.79       2.48       3.98
    argpartsort       0.67       2.11       2.20       0.71       1.92       1.58
    replace           5.23       3.66       3.83       5.33       3.67       3.83
    anynan            4.56       4.69       4.58       4.86      26.29     322.28
    move_sum         13.47       8.87      84.30      13.81       8.91      84.38
    move_nansum      34.45      20.81     172.65      35.23      27.97     189.21
    move_mean        13.36       3.43      32.28      13.55       9.44      88.83
    move_nanmean     38.66       9.56      67.85      38.90      11.61      70.47
    move_std         16.97       2.58      20.58      28.37      19.83     155.80
    move_nanstd      33.19       4.86      32.10      37.12       5.48      32.50
    move_max          6.03       2.08      19.56       6.01       5.17      56.03
    move_nanmax      25.63       5.16      38.48      25.85      14.60     108.84

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
Bottleneck               Python 2.6, 2.7, 3.2; NumPy 1.5.1, 1.6.2, 1.7.0
Compile                  gcc or MinGW
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
    Ran 124 tests in 31.197s
    OK
    <nose.result.TextTestResult run=124 errors=0 failures=0>
