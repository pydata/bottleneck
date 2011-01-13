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
Moving window         ``move_mean, move_nanmean, move_min, move_nanmin,
                      move_max, move_nanmax``
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
    median            3.16     2.16       2.29      3.56     3.94       2.88
    nanmedian        88.34    27.40       4.39     96.33    64.26       6.35
    nansum            5.25     5.58       1.79      5.25     6.33       1.78
    nanmax            5.64     5.70       1.75      5.85     9.56       1.81
    nanmean          10.60    12.55       3.14     11.20    25.25       5.14
    nanstd           14.88     9.32       2.69     15.60    16.41       3.71
    nanargmax         5.17     5.38       2.62      5.20     8.09       2.88
    move_mean         6.11     4.16      14.43      6.60     7.87      14.29
    move_nanmean     17.77    11.48      29.94     19.51    14.02      30.71
    move_max          2.76     3.61       9.27      2.94     5.20      11.80
    move_nanmax      13.32     5.91      18.95     14.13    13.57      26.90

    Reference functions:
    median          np.median
    nanmedian       local copy of sp.stats.nanmedian
    nansum          np.nansum
    nanmax          np.nanmax
    nanmean         local copy of sp.stats.nanmean
    nanstd          local copy of sp.stats.nanstd
    nanargmax       np.nanargmax
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
    median           15.12    14.21       7.29     11.65     3.58       2.82
    nanmedian       333.70   125.13       8.14    355.41   171.86       8.04
    nansum           21.44     7.02       1.79     21.47     8.35       1.79
    nanmax           19.78     6.76       1.76     21.62    12.50       1.79
    nanmean          38.10    15.11       3.14     40.50    31.61       5.15
    nanstd           39.65    10.18       2.70     41.16    18.50       3.71
    nanargmax        16.86     6.32       2.62     17.89    10.19       2.88
    move_mean        16.00     3.94      17.15     17.31     8.30      17.13
    move_nanmean     50.89    10.71      35.43     53.21    17.17      36.67
    move_max          6.07     4.04      11.28      6.80     5.63      13.71
    move_nanmax      30.62     6.84      23.50     36.24    14.91      31.57

    Reference functions:
    median          np.median
    nanmedian       local copy of sp.stats.nanmedian
    nansum          np.nansum
    nanmax          np.nanmax
    nanmean         local copy of sp.stats.nanmean
    nanstd          local copy of sp.stats.nanstd
    nanargmax       np.nanargmax
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
    Ran 36 tests in 67.108s
    OK
    <nose.result.TextTestResult run=36 errors=0 failures=0> 
