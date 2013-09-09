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
        Numpy (np)  1.7.1
        Scipy (sp)  0.12.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.30       1.87       2.61       6.53       2.17       3.06
    nanmedian       136.40      33.60       7.48     146.34      97.96      14.43
    nansum           15.02       6.88       5.87      15.10       9.06       8.89
    nanmax            3.32       1.64       1.09       3.47       4.00       3.99
    nanmean          24.40      12.53      10.14      25.03      15.81      13.93
    nanstd           31.50       8.27       8.98      32.43       9.70      10.64
    nanargmax        11.76       6.03       6.64      11.98      11.02      11.98
    ss                6.05       2.80       2.43       6.06       2.77       2.42
    rankdata         55.29       8.21       1.39      54.97      19.70       1.97
    partsort          1.10       1.94       2.86       1.20       2.34       3.98
    argpartsort       0.41       1.96       2.11       0.45       1.69       1.35
    replace           3.95       3.98       4.13       3.97       3.98       4.13
    anynan            4.10       4.83       4.97       4.18      19.83     321.45
    move_sum         49.66      47.17      85.15      48.81      49.07      86.51
    move_nansum     155.94     184.20     465.50     155.08     197.80     598.44
    move_mean        87.25      30.23      32.89      89.58      83.86      91.87
    move_nanmean    225.05     100.44     271.17     234.04     110.13     339.02
    move_std        112.98      22.49      40.67     165.62     182.21     326.84
    move_nanstd     191.92      41.20     114.36     218.17      43.72     134.97
    move_max         40.21      13.32      20.30      41.15      39.72     181.84
    move_nanmax      91.59      25.24      31.85      91.14      53.85     139.09

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
        Numpy (np)  1.7.1
        Scipy (sp)  0.12.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            8.52       1.89       2.61       9.02       2.22       3.07
    nanmedian       173.70      33.68       7.46     186.39      97.89      14.40
    nansum           22.11       7.26       5.89      21.81       9.61       8.88
    nanmax            4.76       1.70       1.10       4.79       4.19       4.00
    nanmean          35.84      13.13      10.15      35.98      16.70      14.05
    nanstd           41.29       8.45       8.96      43.55       9.92      10.56
    nanargmax        16.60       6.34       6.58      16.54      11.75      11.88
    ss                8.78       3.00       2.42       8.91       2.99       2.42
    rankdata         68.29       8.26       1.39      66.59      19.88       1.96
    partsort          1.52       1.96       2.87       1.70       2.41       4.04
    argpartsort       0.52       1.97       2.10       0.57       1.73       1.33
    replace           6.02       4.10       4.12       6.00       4.10       4.10
    anynan            6.21       5.16       4.93       6.63      27.78     355.48
    move_sum         71.35      48.61      85.13      71.46      51.30      86.97
    move_nansum     227.57     191.44     469.08     227.29     204.85     601.81
    move_mean       122.50      30.51      33.02     125.82      86.94      92.38
    move_nanmean    312.10     101.64     271.52     322.43     111.60     339.61
    move_std        145.97      22.44      40.75     231.65     186.41     333.17
    move_nanstd     232.06      41.28     114.41     262.51      43.78     134.59
    move_max         53.27      13.29      20.30      53.63      39.64     183.31
    move_nanmax     123.14      25.22      31.84     123.99      52.86     140.28

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
Bottleneck               Python 2.6, 2.7, 3.3; NumPy 1.5.1, 1.6.2, 1.7.1
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
