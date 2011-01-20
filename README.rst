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
    10000 loops, best of 3: 95.2 us per loop
    >>> timeit bn.nanmax(arr)
    100000 loops, best of 3: 13.2 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nanmax(arr)
    10000 loops, best of 3: 141 us per loop
    >>> timeit bn.nanmax(arr)
    100000 loops, best of 3: 13.2 us per loop

Bottleneck comes with a benchmark suite. To run the benchmark::
    
    >>> bn.bench(mode='fast', dtype='float64', axis=0)
    Bottleneck performance benchmark
        Bottleneck  0.4.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; float64 and axis=0 are used
        High-level functions used (mode='fast')

                     no NaN   no NaN     no NaN     NaN      NaN        NaN
                    (10,10) (100,100) (1000,1000) (10,10) (100,100) (1000,1000)
    median           10.14    14.08       7.25      8.63     3.50       2.82
    nanmedian       247.68   140.30       8.18    257.18   188.18       8.08
    nansum           13.56     6.44       1.69     13.45     7.48       1.69
    nanmax           13.44     6.31       1.67     14.37    10.47       1.69
    nanmean          25.13    13.88       2.98     26.21    29.24       4.93
    nanstd           30.41     9.63       2.61     31.16    17.90       3.63
    nanargmax        12.36     5.82       2.56     13.00     9.04       2.74
    move_sum         12.10     8.46      14.53     11.95     8.78      14.10
    move_nansum      30.81    20.37      29.43     30.02    25.92      29.71
    move_mean        11.11     4.28      14.44     11.22     8.70      14.24
    move_nanmean     32.91    11.83      29.79     34.33    14.45      30.73
    move_std         17.29     3.33      22.95     22.53    21.08      29.84
    move_nanstd      34.49     6.19      35.03     39.71     7.01      36.15
    move_max          4.46     3.66       9.35      5.19     5.61      11.80
    move_nanmax      22.39     6.32      19.56     23.91    14.80      27.14

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
    10000 loops, best of 3: 26.2 us per loop
    >>> timeit bn.nanmax(arr, axis=0)
    100000 loops, best of 3: 1.93 us per loop
    >>> timeit func(a)
    100000 loops, best of 3: 1.26 us per loop

Note that ``func`` is faster than Numpy's non-NaN version of max::
    
    >>> timeit arr.max(axis=0)
    100000 loops, best of 3: 5 us per loop

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

                     no NaN   no NaN     no NaN     NaN      NaN        NaN
                    (10,10) (100,100) (1000,1000) (10,10) (100,100) (1000,1000)
    median           15.08    14.72       7.24     11.99     3.54       2.83
    nanmedian       341.83   143.61       8.18    357.21   192.56       8.07
    nansum           21.67     6.76       1.70     21.55     7.92       1.70
    nanmax           21.17     6.57       1.67     23.42    11.20       1.68
    nanmean          38.60    14.42       2.99     40.84    30.70       4.98
    nanstd           43.67     9.85       2.61     46.20    18.34       3.63
    nanargmax        18.19     6.11       2.56     19.28     9.50       2.75
    move_sum         18.01     8.70      14.51     17.99     9.02      14.10
    move_nansum      47.44    21.28      29.40     49.06    26.60      29.68
    move_mean        17.02     4.34      14.43     17.31     8.87      14.23
    move_nanmean     53.00    11.95      29.77     54.41    14.63      30.81
    move_std         23.60     3.35      22.85     33.80    21.60      29.69
    move_nanstd      47.09     6.21      34.87     57.45     7.02      36.03
    move_max          5.93     3.70       9.33      6.77     5.70      11.83
    move_nanmax      30.34     6.39      19.54     36.06    15.04      27.14

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
