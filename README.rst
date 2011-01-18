==========
Bottleneck
==========

Introduction
============

Bottleneck is a collection of fast NumPy array functions written in Cython:

===================== =======================================================
NumPy/SciPy           ``median, nanmedian, nansum, nanmin, nanmax, nanmean,
                      nanstd, nanargmin, nanargmax`` 
Functions             ``nanvar``
Moving window         ``move_sum, move_nansum, move_mean, move_nanmean,
                      move_std, move_nanstd, move_min, move_nanmin, move_max,
                      move_nanmax``
Group by              ``group_nanmean``
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

Group nanmean::   

    >>> label = ['a', 'a', 'b', 'b', 'a']
    >>> bn.group_nanmean(arr, label)
    (array([ 2.66666667,  4.        ]), ['a', 'b'])

Fast
====

Bottleneck is fast::

    >>> arr = np.random.rand(100, 100)    
    >>> timeit np.nanmax(arr)
    10000 loops, best of 3: 99.6 us per loop
    >>> timeit bn.nanmax(arr)
    100000 loops, best of 3: 15.3 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nanmax(arr)
    10000 loops, best of 3: 146 us per loop
    >>> timeit bn.nanmax(arr)
    100000 loops, best of 3: 15.2 us per loop

Bottleneck comes with a benchmark suite. To run the benchmark::
    
    >>> bn.bench(mode='fast', dtype='float64', axis=0)
    Bottleneck performance benchmark
        Bottleneck  0.3.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        High-level functions used (mode='fast')

                     no NaN   no NaN     no NaN     NaN      NaN        NaN
                    (10,10) (100,100) (1000,1000) (10,10) (100,100) (1000,1000)
    median            6.00    13.49       7.40      5.34     3.29       2.92
    nanmedian       155.51   132.44       8.89    160.37   174.60       8.13
    nansum            7.32     5.83       1.73      7.26     6.63       1.72
    nanmax            8.08     5.86       1.69      8.25     9.11       1.69
    nanmean          15.13    12.82       3.01     16.03    26.48       5.05
    nanstd           18.68     9.40       2.65     19.52    16.63       3.67
    nanargmax         7.48     5.69       2.64      7.53     8.32       2.84
    move_sum          7.33     7.82      11.96      7.42     8.26      13.08
    move_nansum      20.59    19.15      29.42     19.96    24.45      27.26
    move_mean         7.67     4.26      14.29      7.53     8.22      13.40
    move_nanmean     21.46    11.49      29.43     22.42    14.13      28.88
    move_std         13.00     3.31      22.92     15.70    20.14      29.51
    move_nanstd      25.58     6.14      33.55     28.91     6.96      35.87
    move_max          3.26     3.64       9.38      3.70     5.56      11.70
    move_nanmax      16.84     6.23      19.49     17.57    14.37      26.70

    Reference functions:
    median          np.median
    nanmedian       local copy of sp.stats.nanmedian
    nansum          np.nansum
    nanmax          np.nanmax
    nanmean         local copy of sp.stats.nanmean
    nanstd          local copy of sp.stats.nanstd
    nanargmax       np.nanargmax
    move_sum        sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nansum     sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_mean       sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nanmean    sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_std        sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nanstd     sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_max        sp.ndimage.maximum_filter1d based, window=a.shape[0]/5
    move_nanmax     sp.ndimage.maximum_filter1d based, window=a.shape[0]/5

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
    10000 loops, best of 3: 24.9 us per loop
    >>> timeit bn.nanmax(arr, axis=0)
    100000 loops, best of 3: 4.97 us per loop
    >>> timeit func(a)
    100000 loops, best of 3: 2.13 us per loop

Note that ``func`` is faster than Numpy's non-NaN version of max::
    
    >>> timeit arr.max(axis=0)
    100000 loops, best of 3: 4.75 us per loop

So adding NaN protection to your inner loops comes at a negative cost!

Benchmarks for the low-level Cython functions::

    >>> bn.bench(mode='faster', dtype='float64', axis=0)
    Bottleneck performance benchmark
        Bottleneck  0.3.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        Low-level functions used (mode='faster')

                     no NaN   no NaN     no NaN     NaN      NaN        NaN
                    (10,10) (100,100) (1000,1000) (10,10) (100,100) (1000,1000)
    median           14.35    14.16       7.31     11.37     3.59       2.85
    nanmedian       325.28   127.47       8.23    333.08   174.28       8.11
    nansum           20.49     7.04       1.81     20.80     8.31       1.79
    nanmax           19.04     6.73       1.76     20.80    12.46       1.80
    nanmean          37.23    15.11       3.17     39.19    31.45       5.16
    nanstd           43.39    10.34       2.89     46.11    18.26       3.89
    nanargmax        15.82     6.28       2.66     16.97    10.12       2.91
    move_sum         17.58     8.61      14.62     17.90     9.04      14.16
    move_nansum      47.59    21.41      29.52     50.09    27.04      29.81
    move_mean        15.90     4.35      14.50     17.25     8.90      14.29
    move_nanmean     53.71    12.06      29.97     55.72    14.81      31.04
    move_std         22.01     3.34      23.02     31.34    21.59      30.03
    move_nanstd      46.56     6.23      35.16     57.32     7.09      36.30
    move_max          6.05     3.60       9.17      7.28     5.50      11.93
    move_nanmax      29.65     6.21      19.47     38.29    14.38      27.28

    Reference functions:
    median          np.median
    nanmedian       local copy of sp.stats.nanmedian
    nansum          np.nansum
    nanmax          np.nanmax
    nanmean         local copy of sp.stats.nanmean
    nanstd          local copy of sp.stats.nanstd
    nanargmax       np.nanargmax
    move_sum        sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nansum     sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_mean       sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nanmean    sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_std        sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_nanstd     sp.ndimage.convolve1d based, window=a.shape[0]/5
    move_max        sp.ndimage.maximum_filter1d based, window=a.shape[0]/5
    move_nanmax     sp.ndimage.maximum_filter1d based, window=a.shape[0]/5

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
Bottleneck               Python, NumPy 1.4.1+
Unit tests               nose
Compile                  gcc or MinGW
Optional                 SciPy 0.72+ (portions of benchmark)
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
    Ran 44 tests in 39.108s
    OK
    <nose.result.TextTestResult run=44 errors=0 failures=0> 
