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
        NaN means one-third NaNs; axis=0 and float64 are used
    median vs np.median
        3.59  (10,10)         
        2.43  (1001,1001)     
        2.28  (1000,1000)     
        2.16  (100,100)       
    nanmedian vs local copy of sp.stats.nanmedian
      102.72  (10,10)      NaN
       94.34  (10,10)         
       67.89  (100,100)    NaN
       28.52  (100,100)       
        6.37  (1000,1000)  NaN
        4.41  (1000,1000)     
    nanmax vs np.nanmax
        9.99  (100,100)    NaN
        6.12  (10,10)      NaN
        5.99  (10,10)         
        5.88  (100,100)       
        1.79  (1000,1000)  NaN
        1.76  (1000,1000)     
    nanmean vs local copy of sp.stats.nanmean
       25.95  (100,100)    NaN
       12.85  (100,100)       
       12.26  (10,10)      NaN
       11.89  (10,10)         
        5.15  (1000,1000)  NaN
        3.17  (1000,1000)     
    nanstd vs local copy of sp.stats.nanstd
       16.96  (100,100)    NaN
       15.75  (10,10)      NaN
       15.49  (10,10)         
        9.51  (100,100)       
        3.85  (1000,1000)  NaN
        2.82  (1000,1000)     
    nanargmax vs np.nanargmax
        8.60  (100,100)    NaN
        5.65  (10,10)      NaN
        5.62  (100,100)       
        5.44  (10,10)         
        2.84  (1000,1000)  NaN
        2.58  (1000,1000)     
    move_nanmean vs sp.ndimage.convolve1d based function
        window = 5
       19.52  (10,10)      NaN
       18.55  (10,10)         
       10.56  (100,100)    NaN
        6.67  (100,100)       
        5.19  (1000,1000)  NaN
        4.42  (1000,1000)     

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
        NaN means one-third NaNs; axis=0 and float64 are used
    median_selector vs np.median
       15.29  (10,10)         
       14.19  (100,100)       
        8.04  (1001,1001)     
        7.32  (1000,1000)     
    nanmedian_selector vs local copy of sp.stats.nanmedian
      352.08  (10,10)      NaN
      340.27  (10,10)         
      185.56  (100,100)    NaN
      138.81  (100,100)       
        8.21  (1000,1000)     
        8.09  (1000,1000)  NaN
    nanmax_selector vs np.nanmax
       21.54  (10,10)      NaN
       19.98  (10,10)         
       12.65  (100,100)    NaN
        6.82  (100,100)       
        1.79  (1000,1000)  NaN
        1.76  (1000,1000)     
    nanmean_selector vs local copy of sp.stats.nanmean
       41.08  (10,10)      NaN
       39.05  (10,10)         
       31.74  (100,100)    NaN
       15.24  (100,100)       
        5.13  (1000,1000)  NaN
        3.16  (1000,1000)     
    nanstd_selector vs local copy of sp.stats.nanstd
       44.55  (10,10)      NaN
       43.49  (10,10)         
       18.66  (100,100)    NaN
       10.29  (100,100)       
        3.83  (1000,1000)  NaN
        2.82  (1000,1000)     
    nanargmax_selector vs np.nanargmax
       17.91  (10,10)      NaN
       17.00  (10,10)         
       10.56  (100,100)    NaN
        6.50  (100,100)       
        2.85  (1000,1000)  NaN
        2.59  (1000,1000)     
    move_nanmean_selector vs sp.ndimage.convolve1d based function
        window = 5
       55.96  (10,10)      NaN
       50.82  (10,10)         
       11.77  (100,100)    NaN
        6.93  (100,100)       
        5.56  (1000,1000)  NaN
        4.51  (1000,1000)     

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
    Ran 13 tests in 41.756s
    OK
    <nose.result.TextTestResult run=11 errors=0 failures=0> 
