==========
Bottleneck
==========

Introduction
============

Bottleneck is a collection of fast NumPy array functions written in Cython.

Bottleneck, which is in the prototype stage, contains the following functions:

===================== =======================================================
NumPy/SciPy           ``median, nanmedian, nanmin, nanmax, nanmean, nanstd,
                      nanargmax`` 
Miscellaneous         ``nanvar``
Moving window         ``move_nanmean``
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
    
    >>> bn.bench(mode='fast')
    Bottleneck performance benchmark
        Bottleneck  0.2.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means all NaNs; axis=0 and float64 are used
    median vs np.median
        4.01  (10,10)         
        3.47  (100,100)       
        2.53  (1000,1000)     
    nanmedian vs local copy of sp.stats.nanmedian
      127.22  (100,100)    NaN
      106.54  (10,10)         
      103.07  (100,100)       
       69.77  (10,10)      NaN
        6.92  (1000,1000)  NaN
        5.83  (1000,1000)     
    nanmax vs np.nanmax
       10.41  (100,100)       
        9.08  (100,100)    NaN
        8.74  (10,10)      NaN
        6.87  (10,10)         
        2.06  (1000,1000)     
        2.02  (1000,1000)  NaN
    nanmin vs np.nanmin
        8.57  (100,100)    NaN
        8.57  (10,10)      NaN
        6.70  (10,10)         
        6.66  (100,100)       
        2.03  (1000,1000)  NaN
        1.76  (1000,1000)     
    nanmean vs local copy of sp.stats.nanmean
       46.12  (100,100)    NaN
       13.96  (10,10)      NaN
       12.96  (100,100)       
       12.41  (10,10)         
        9.35  (1000,1000)  NaN
        3.16  (1000,1000)     
    nanstd vs local copy of sp.stats.nanstd
       51.23  (100,100)    NaN
       19.18  (10,10)      NaN
       16.56  (10,10)         
       11.60  (1000,1000)  NaN
        9.46  (100,100)       
        2.84  (1000,1000)     
    nanargmax vs np.nanargmax
        9.33  (100,100)       
        6.13  (10,10)         
        2.95  (1000,1000)     
    move_nanmean vs sp.ndimage.convolve1d based function
        window = 5
       37.31  (100,100)    NaN
       23.37  (10,10)      NaN
       19.60  (10,10)         
        9.90  (1000,1000)  NaN
        7.50  (100,100)       
        4.53  (1000,1000)     

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

    >>> bn.bench(mode='faster')
    Bottleneck performance benchmark
        Bottleneck  0.2.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means all NaNs; axis=0 and float64 are used
    median_selector vs np.median
       13.91  (10,10)         
        4.84  (100,100)       
        3.04  (1000,1000)     
    nanmedian_selector vs local copy of sp.stats.nanmedian
      327.76  (10,10)         
      259.64  (10,10)      NaN
      128.02  (100,100)       
      123.96  (100,100)    NaN
        6.43  (1000,1000)     
        4.07  (1000,1000)  NaN
    nanmax_selector vs np.nanmax
       27.93  (10,10)      NaN
       23.03  (10,10)         
       12.39  (100,100)       
       11.14  (100,100)    NaN
        2.07  (1000,1000)     
        2.03  (1000,1000)  NaN
    nanmin_selector vs np.nanmin
       27.08  (100,100)    NaN
       26.94  (10,10)      NaN
       21.24  (10,10)         
       21.09  (100,100)       
        2.04  (1000,1000)  NaN
        1.77  (1000,1000)     
    nanmean_selector vs local copy of sp.stats.nanmean
       56.10  (100,100)    NaN
       41.98  (10,10)      NaN
       39.71  (10,10)         
       15.17  (100,100)       
        9.36  (1000,1000)  NaN
        3.18  (1000,1000)     
    nanstd_selector vs local copy of sp.stats.nanstd
       62.91  (100,100)    NaN
       59.28  (10,10)      NaN
       44.34  (10,10)         
       11.62  (1000,1000)  NaN
       10.30  (100,100)       
        2.85  (1000,1000)     
    nanargmax_selector vs np.nanargmax
       19.19  (10,10)         
       11.68  (100,100)       
        2.97  (1000,1000)     
    move_nanmean_selector vs sp.ndimage.convolve1d based function
        window = 5
       66.33  (10,10)      NaN
       51.69  (10,10)         
       45.28  (100,100)    NaN
        9.81  (1000,1000)  NaN
        7.92  (100,100)       
        4.46  (1000,1000)     

Slow
====

Currently only 1d, 2d, and 3d NumPy arrays with dtype int32, int64, float32,
and float64 are accelerated. All other ndim/dtype combinations result in
calls to slower, unaccelerated functions.

License
=======

Bottleneck is distributed under a Simplified BSD license. Parts of NumPy,
Scipy and numpydoc, all of which have BSD licenses, are included in
Bottleneck. See the LICENSE file, which is distributed with Bottleneck, for
details.

URLs
====

===============   =============================================
 download          http://pypi.python.org/pypi/Bottleneck
 docs              http://berkeleyanalytics.com/bottleneck
 code              http://github.com/kwgoodman/bottleneck
 mailing list      http://groups.google.com/group/bottle-neck
===============   =============================================

Install
=======

Requirements:

======================== ===================================
Bottleneck               Python, NumPy 1.4.1+
Unit tests               nose
Compile                  gcc or MinGW
======================== ===================================

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
    Ran 12 tests in 41.756s
    OK
    <nose.result.TextTestResult run=11 errors=0 failures=0> 
