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
        Numpy (np)  1.7.0
        Scipy (sp)  0.11.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            6.12       1.88       2.57       6.38       2.22       2.88
    nanmedian       126.25      33.46       7.32     139.59      98.87      14.32
    nansum           14.27       6.84       5.93      14.43       8.81       8.80
    nanmax            3.11       1.49       1.07       3.27       3.70       3.96
    nanmean          25.41      12.60      10.13      26.10      15.90      14.06
    nanstd           32.56       8.54       9.27      33.34      10.14      11.33
    nanargmax        11.46       5.49       6.43      11.36      10.29      11.61
    ss                5.61       2.67       2.42       5.63       2.69       2.43
    rankdata         54.79       8.28       1.39      55.32      18.79       1.94
    partsort          1.05       1.89       2.78       1.16       2.38       3.73
    argpartsort       0.40       2.05       2.15       0.43       1.87       1.43
    replace           4.39       3.92       4.15       4.43       3.91       4.15
    anynan            4.14       4.84       4.93       4.30      20.47     329.55
    move_sum         45.61      44.24      79.55      45.36      44.58      79.60
    move_nansum     144.21     168.47     425.40     144.21     192.27     591.93
    move_mean        80.06      30.47      33.04      80.35      80.13      88.45
    move_nanmean    217.15     101.80     271.80     221.77     111.94     341.67
    move_std        112.71      22.94      41.03     165.00     171.11     307.99
    move_nanstd     183.93      42.36     114.73     201.99      44.67     135.03
    move_max         34.90      12.90      19.34      35.59      35.44     137.10
    move_nanmax      89.25      26.24      33.57      88.48      56.27     149.49

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
        Numpy (np)  1.7.0
        Scipy (sp)  0.11.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=1 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            8.16       1.89       2.57       8.55       2.27       2.87
    nanmedian       166.30      33.51       7.26     179.38     100.87      13.98
    nansum           20.07       7.21       5.91      19.98       9.27       8.88
    nanmax            4.34       1.56       1.09       4.52       3.91       3.99
    nanmean          34.17      13.18      10.26      34.61      16.64      14.04
    nanstd           40.79       8.72       9.24      42.79      10.37      11.35
    nanargmax        15.09       5.66       6.56      14.90      10.84      11.85
    ss                8.48       2.87       2.44       8.49       2.85       2.43
    rankdata         66.37       8.29       1.38      65.97      18.91       1.93
    partsort          1.47       1.91       2.78       1.62       2.44       3.76
    argpartsort       0.56       2.08       2.15       0.59       1.94       1.43
    replace           6.42       4.02       4.14       6.29       4.03       4.14
    anynan            6.18       5.17       4.96       6.60      28.63     353.89
    move_sum         60.69      45.21      79.51      60.37      45.35      79.81
    move_nansum     199.23     172.42     426.42     197.33     198.26     594.46
    move_mean       102.96      30.45      32.99     104.52      81.64      88.64
    move_nanmean    278.86     101.95     271.79     286.26     111.78     342.09
    move_std        129.71      22.78      41.14     207.27     170.57     307.86
    move_nanstd     219.50      42.06     114.69     246.10      44.38     134.57
    move_max         42.80      11.91      19.71      43.43      35.76     161.31
    move_nanmax     108.94      23.55      33.45     108.55      57.20     149.79

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
