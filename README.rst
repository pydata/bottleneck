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
        Bottleneck  0.4.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        High-level functions used (mode='fast')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median            9.34      14.40       7.29       8.27       3.64       2.84
    nanmedian       219.65     127.95       8.21     226.79     176.69       8.10
    nansum           12.16       6.40       1.72      12.10       7.34       1.71
    nanmax           12.78       6.29       1.69      13.56      10.45       1.69
    nanmean          21.97      13.98       3.00      21.93      28.89       4.99
    nanstd           30.06       9.69       2.69      30.61      17.62       3.71
    nanargmax        10.68       6.05       2.68      10.85       9.04       2.88
    rankdata         23.11      12.51       8.33      22.71      14.09       9.36
    move_sum         11.13       8.71      14.53      12.15       8.63      14.11
    move_nansum      29.39      19.52      29.45      28.00      25.40      29.83
    move_mean        11.11       4.25      14.43      11.23       8.36      14.30
    move_nanmean     31.65      11.81      29.86      32.81      14.41      30.93
    move_std         17.33       3.33      22.82      22.30      20.77      29.94
    move_nanstd      34.82       6.18      34.94      40.44       7.06      36.09
    move_max          4.06       3.61       9.26       4.71       5.54      11.65
    move_nanmax      22.16       5.95      19.57      24.74      14.69      27.07

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
        Bottleneck  0.4.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        Low-level functions used (mode='faster')

                     no NaN     no NaN     no NaN      NaN        NaN        NaN    
                    (10,10)   (100,100) (1000,1000)  (10,10)   (100,100) (1000,1000)
    median           14.72      14.75       7.09      11.90       3.64       2.83
    nanmedian       337.43     133.27       8.92     352.73     182.65       8.20
    nansum           20.75       6.72       1.73      20.61       7.96       1.72
    nanmax           20.03       6.58       1.72      22.44      11.11       1.69
    nanmean          38.55      14.44       3.00      39.35      30.52       5.00
    nanstd           41.78       9.85       2.70      44.16      18.17       3.71
    nanargmax        17.97       6.33       2.70      18.50       9.64       2.91
    rankdata         24.43      12.43       8.37      24.37      14.06       9.21
    move_sum         18.29       8.60      14.52      18.13       8.87      13.62
    move_nansum      45.98      20.80      29.33      48.56      26.25      29.29
    move_mean        16.33       4.35      14.33      16.21       8.64      14.15
    move_nanmean     50.79      11.92      29.36      51.63      14.93      30.32
    move_std         23.45       3.36      22.88      33.20      20.18      29.18
    move_nanstd      48.02       6.16      34.61      57.20       7.03      36.13
    move_max          5.82       3.63       9.31       6.70       5.62      11.77
    move_nanmax      29.09       6.02      19.55      36.57      14.83      27.02

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
    Ran 46 tests in 41.457s
    OK
    <nose.result.TextTestResult run=46 errors=0 failures=0> 
