==========
Bottleneck
==========

Introduction
============

Bottleneck is a collection of fast NumPy array functions written in Cython:

===================== =======================================================
NumPy/SciPy           ``median, nanmedian, nanmin, nanmax, nanmean, nanstd,
                      nanargmin, nanargmax`` 
Functions             ``nanvar``
Moving window         ``move_nanmean, move_min, move_max, move_nanmin,
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

Bottleneck comes with a benchmark suite that compares the performance of the
bottleneck functions that have a NumPy/SciPy equivalent. To run the
benchmark::
    
    >>> bn.bench(mode='fast')
    Bottleneck performance benchmark
        Bottleneck  0.3.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; axis=0 and float64 are used
    median vs np.median
        3.48  (10,10)         
        2.41  (1001,1001)     
        2.28  (1000,1000)     
        2.17  (100,100)       
    nanmedian vs local copy of sp.stats.nanmedian
      102.67  (10,10)      NaN
       92.03  (10,10)         
       68.72  (100,100)    NaN
       28.87  (100,100)       
        6.40  (1000,1000)  NaN
        4.41  (1000,1000)     
    nanmax vs np.nanmax
        9.87  (100,100)    NaN
        6.27  (10,10)      NaN
        6.06  (10,10)         
        5.81  (100,100)       
        1.79  (1000,1000)  NaN
        1.76  (1000,1000)     
    nanmean vs local copy of sp.stats.nanmean
       26.11  (100,100)    NaN
       12.88  (100,100)       
       12.05  (10,10)      NaN
       11.58  (10,10)         
        5.12  (1000,1000)  NaN
        3.16  (1000,1000)     
    nanstd vs local copy of sp.stats.nanstd
       16.86  (100,100)    NaN
       16.43  (10,10)      NaN
       15.54  (10,10)         
        9.44  (100,100)       
        3.84  (1000,1000)  NaN
        2.82  (1000,1000)     
    nanargmax vs np.nanargmax
        8.52  (100,100)    NaN
        5.62  (100,100)       
        5.51  (10,10)      NaN
        5.33  (10,10)         
        2.83  (1000,1000)  NaN
        2.57  (1000,1000)     
    move_nanmean vs sp.ndimage.convolve1d based function
        window = 5
       19.92  (10,10)      NaN
       18.86  (10,10)         
       11.11  (100,100)    NaN
        6.24  (100,100)       
        5.31  (1000,1000)  NaN
        4.37  (1000,1000)     
    move_max vs sp.ndimage.maximum_filter1d based function
        window = 5
        3.63  (10,10)         
        1.84  (100,100)       
        1.48  (1000,1000)     
    move_nanmax vs sp.ndimage.maximum_filter1d based function
        window = 5
       14.94  (10,10)      NaN
       14.05  (10,10)         
        8.46  (100,100)    NaN
        4.16  (1000,1000)  NaN
        2.94  (100,100)       
        2.85  (1000,1000)     

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
        Bottleneck  0.3.0
        Numpy (np)  1.5.1
        Scipy (sp)  0.8.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means one-third NaNs; axis=0 and float64 are used
    median_selector vs np.median
       14.49  (100,100)       
       14.14  (10,10)         
        8.03  (1001,1001)     
        7.35  (1000,1000)     
    nanmedian_selector vs local copy of sp.stats.nanmedian
      337.27  (10,10)      NaN
      322.51  (10,10)         
      186.39  (100,100)    NaN
      138.67  (100,100)       
        8.25  (1000,1000)     
        8.11  (1000,1000)  NaN
    nanmax_selector vs np.nanmax
       20.49  (10,10)      NaN
       19.15  (10,10)         
       12.45  (100,100)    NaN
        6.73  (100,100)       
        1.79  (1000,1000)  NaN
        1.76  (1000,1000)     
    nanmean_selector vs local copy of sp.stats.nanmean
       36.52  (10,10)      NaN
       35.08  (10,10)         
       31.36  (100,100)    NaN
       15.04  (100,100)       
        5.16  (1000,1000)  NaN
        3.17  (1000,1000)     
    nanstd_selector vs local copy of sp.stats.nanstd
       44.59  (10,10)      NaN
       42.21  (10,10)         
       18.67  (100,100)    NaN
       10.27  (100,100)       
        3.84  (1000,1000)  NaN
        2.83  (1000,1000)     
    nanargmax_selector vs np.nanargmax
       16.62  (10,10)      NaN
       16.29  (10,10)         
       10.48  (100,100)    NaN
        6.52  (100,100)       
        2.84  (1000,1000)  NaN
        2.57  (1000,1000)     
    move_nanmean_selector vs sp.ndimage.convolve1d based function
        window = 5
       54.54  (10,10)      NaN
       49.65  (10,10)         
       10.72  (100,100)    NaN
        7.07  (100,100)       
        5.38  (1000,1000)  NaN
        4.30  (1000,1000)     
    move_max_selector vs sp.ndimage.maximum_filter1d based function
        window = 5
        9.19  (10,10)         
        1.92  (100,100)       
        1.49  (1000,1000)     
    move_nanmax_selector vs sp.ndimage.maximum_filter1d based function
        window = 5
       39.73  (10,10)      NaN
       33.53  (10,10)         
        8.92  (100,100)    NaN
        4.16  (1000,1000)  NaN
        3.20  (100,100)       
        2.88  (1000,1000)  

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
    Ran 17 tests in 54.756s
    OK
    <nose.result.TextTestResult run=17 errors=0 failures=0> 
