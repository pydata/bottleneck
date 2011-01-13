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
                      move_min, move_nanmin, move_max, move_nanmax``
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

Bottleneck comes with a benchmark suite that compares the performance of the
bottleneck functions that have a NumPy/SciPy equivalent. To run the
benchmark::
    
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
    median            3.55     2.17       2.28      4.07     3.99       2.88
    nanmedian        96.98    28.51       4.44    108.27    66.47       6.40
    nansum            5.19     5.64       1.77      5.21     6.46       1.76
    nanmax            6.18     5.91       1.74      6.37    10.06       1.81
    nanmean          12.03    13.01       3.14     12.45    26.52       5.06
    nanstd           16.61     9.53       2.87     17.08    16.68       3.87
    nanargmax         5.55     5.46       2.62      5.75     8.40       2.88
    move_sum          6.22     7.57      14.55      6.44     7.86      14.11
    move_nansum      16.07    18.40      29.32     16.50    23.40      29.77
    move_mean         6.31     4.17      14.33      6.44     7.97      14.15
    move_nanmean     19.39    11.46      29.72     20.59    14.06      30.66
    move_max          2.71     3.59       9.16      2.94     5.21      11.77
    move_nanmax      13.62     5.98      19.15     15.22    13.54      27.19

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

Benchmarks for the low-level Cython version of each function::

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
    median           14.36    14.19       7.25     11.23     3.62       2.82
    nanmedian       329.99   126.77       8.21    346.97   174.34       8.04
    nansum           20.74     7.07       1.78     20.64     8.38       1.77
    nanmax           19.70     6.83       1.74     21.47    12.59       1.83
    nanmean          36.78    15.14       3.12     39.58    31.52       5.10
    nanstd           43.23    10.32       2.88     46.03    18.27       3.88
    nanargmax        16.32     6.29       2.63     17.17    10.22       2.90
    move_sum         19.02     8.70      14.51     19.13     9.13      14.08
    move_nansum      48.83    21.24      29.34     51.22    26.92      29.69
    move_mean        16.63     4.37      14.40     17.61     8.94      14.20
    move_nanmean     53.43    12.09      29.76     55.76    14.82      30.78
    move_max          5.79     3.71       9.26      6.99     5.53      11.83
    move_nanmax      28.19     6.17      19.32     34.70    14.43      27.17

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
    move_max        sp.ndimage.maximum_filter1d based, window=a.shape[0]/5
    move_nanmax     sp.ndimage.maximum_filter1d based, window=a.shape[0]/5

Slow
====

Currently only 1d, 2d, and 3d NumPy arrays with data type (dtype) int32,
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

Directions for installing a *released* version of Bottleneck are given below.
Cython is not required since the Cython files have already been converted to
C source files. (If you obtained bottleneck directly from the repository, then
you will need to generate the C source files using the included Makefile which
requires Cython.)

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
gcc compiler. MinGW (Minimalist GNU for Windows) contains gcc and has been used
to successfully compile Bottleneck on Windows.

Install MinGW and add it to your system path. Then install Bottleneck with the
commands::

    python setup.py build --compiler=mingw32
    python setup.py install

**Post install**

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 40 tests in 75.108s
    OK
    <nose.result.TextTestResult run=40 errors=0 failures=0> 
