==========
Bottleneck
==========

Introduction
============

Bottleneck is a collection of fast, NumPy array functions written in Cython.

The three categories of Bottleneck functions:

- Faster, drop-in replacement for NaN functions in NumPy and SciPy
- Moving window functions
- Group functions that bin calculations by like-labeled elements  

Function signatures (using mean as an example):

===============  ================================================
 NaN functions    ``mean(arr, axis=None)``
 Moving window    ``move_mean(arr, window, axis=0)``
 Group by         ``group_mean(arr, label, order=None, axis=0)``
===============  ================================================

Let's give it a try. Create a NumPy array::
    
    >>> import numpy as np
    >>> arr = np.array([1, 2, np.nan, 4, 5])

Find the mean::

    >>> import bottleneck as bn
    >>> bn.mean(arr)
    3.0

Moving window sum::

    >>> bn.move_sum(arr, window=2)
    array([ nan,   3.,   2.,   4.,   9.])

Group mean::   

    >>> label = ['a', 'a', 'b', 'b', 'a']
    >>> bn.group_mean(arr, label)
    (array([ 2.66666667,  4.        ]), ['a', 'b'])

Fast
====

Bottleneck is fast::

    >>> arr = np.random.rand(100, 100)    
    >>> timeit np.nansum(arr)
    10000 loops, best of 3: 68.4 us per loop
    >>> timeit bn.sum(arr)
    100000 loops, best of 3: 17.7 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nansum(arr)
    1000 loops, best of 3: 417 us per loop
    >>> timeit bn.sum(arr)
    10000 loops, best of 3: 64.8 us per loop

Bottleneck comes with a benchmark suite that compares the performance of the
bottleneck functions that have a NumPy/SciPy equivalent. To run the
benchmark::
    
    >>> bn.benchit(verbose=False)
    Bottleneck performance benchmark
        Bottleneck  0.1.0dev
        Numpy       1.5.1
        Scipy       0.8.0
        Speed is numpy (or scipy) time divided by Bottleneck time
        NaN means all NaNs
       Speed   Test                  Shape        dtype    NaN?
       4.8103  nansum(a, axis=-1)    (500,500)    int64  
       5.1392  nansum(a, axis=-1)    (10000,)     float64  
       7.1373  nansum(a, axis=-1)    (500,500)    int32  
       6.0882  nansum(a, axis=-1)    (500,500)    float64  
       7.7081  nansum(a, axis=-1)    (10000,)     int32  
       2.1392  nansum(a, axis=-1)    (10000,)     int64  
       9.8542  nansum(a, axis=-1)    (500,500)    float64  NaN
       7.9069  nansum(a, axis=-1)    (10000,)     float64  NaN
       5.1859  nanmax(a, axis=-1)    (500,500)    int64  
       9.5304  nanmax(a, axis=-1)    (10000,)     float64  
       0.1392  nanmax(a, axis=-1)    (500,500)    int32  
      10.8645  nanmax(a, axis=-1)    (500,500)    float64  
       2.4558  nanmax(a, axis=-1)    (10000,)     int32  
       3.2855  nanmax(a, axis=-1)    (10000,)     int64  
       9.6748  nanmax(a, axis=-1)    (500,500)    float64  NaN
       8.3101  nanmax(a, axis=-1)    (10000,)     float64  NaN
       5.1828  nanmin(a, axis=-1)    (500,500)    int64  
       6.8145  nanmin(a, axis=-1)    (10000,)     float64  
       0.1349  nanmin(a, axis=-1)    (500,500)    int32  
       7.6657  nanmin(a, axis=-1)    (500,500)    float64  
       2.4619  nanmin(a, axis=-1)    (10000,)     int32  
       3.2942  nanmin(a, axis=-1)    (10000,)     int64  
       9.7377  nanmin(a, axis=-1)    (500,500)    float64  NaN
       8.3564  nanmin(a, axis=-1)    (10000,)     float64  NaN
      20.7414  nanmean(a, axis=-1)   (500,500)    int64  
      13.0027  nanmean(a, axis=-1)   (10000,)     float64  
      19.1651  nanmean(a, axis=-1)   (500,500)    int32  
      13.3462  nanmean(a, axis=-1)   (500,500)    float64  
      18.1296  nanmean(a, axis=-1)   (10000,)     int32  
      18.9846  nanmean(a, axis=-1)   (10000,)     int64  
      53.6566  nanmean(a, axis=-1)   (500,500)    float64  NaN
      23.0624  nanmean(a, axis=-1)   (10000,)     float64  NaN
       6.8075  nanstd(a, axis=-1)    (500,500)    int64  
       9.0953  nanstd(a, axis=-1)    (10000,)     float64  
       7.2786  nanstd(a, axis=-1)    (500,500)    int32  
      11.1632  nanstd(a, axis=-1)    (500,500)    float64  
       5.9248  nanstd(a, axis=-1)    (10000,)     int32  
       5.2482  nanstd(a, axis=-1)    (10000,)     int64  
      89.4077  nanstd(a, axis=-1)    (500,500)    float64  NaN
      27.0319  nanstd(a, axis=-1)    (10000,)     float64  NaN

Faster
======

Under the hood Bottleneck uses a separate Cython function for each combination
of ndim, dtype, and axis. A lot of the overhead in bn.max(), for example, is
in checking that the axis is within range, converting non-array data to an
array, and selecting the function to use to calculate the maximum.

You can get rid of the overhead by doing all this before you, say, enter
an inner loop::

    >>> arr = np.random.rand(10,10)
    >>> func, a = bn.func.max_selector(arr, axis=0)
    >>> func
    <built-in function max_2d_float64_axis0> 

Let's see how much faster than runs::
    
    >> timeit np.nanmax(arr, axis=0)
    10000 loops, best of 3: 25.7 us per loop
    >> timeit bn.max(arr, axis=0)
    100000 loops, best of 3: 5.25 us per loop
    >> timeit func(a)
    100000 loops, best of 3: 2.5 us per loop

Note that ``func`` is faster than Numpy's non-NaN version of max::
    
    >> timeit arr.max(axis=0)
    100000 loops, best of 3: 3.28 us per loop

So adding NaN protection to your inner loops comes at a negative cost!           

Functions
=========

Bottleneck is in the prototype stage.

Bottleneck contains the following functions:

=========    ==============   ===============
sum          move_sum         
mean                          group_mean
var                  
std          
min          
max          
=========    ==============   ===============

Currently only 1d, 2d, and 3d NumPy arrays with dtype int32, int64, and
float64 are supported.

License
=======

Bottleneck is distributed under a Simplified BSD license. Parts of NumPy,
Scipy and numpydoc, all of which have BSD licenses, are included in
Bottleneck. See the LICENSE file, which is distributed with Bottleneck, for
details.

Download and install
====================

You can grab Bottleneck from http://github.com/kwgoodman/bottleneck

**GNU/Linux, Mac OS X, et al.**

To install Bottleneck::

    $ python setup.py build
    $ sudo python setup.py install
    
Or, if you wish to specify where Bottleneck is installed, for example inside
``/usr/local``::

    $ python setup.py build
    $ sudo python setup.py install --prefix=/usr/local

**Windows**

In order to compile the C code in dsna you need a Windows version of the gcc
compiler. MinGW (Minimalist GNU for Windows) contains gcc and has been used to successfully compile dsna on Windows.

Install MinGW and add it to your system path. Then install dsna with the
commands::

    python setup.py build --compiler=mingw32
    python setup.py install

**Post install**

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 10 tests in 13.756s
    OK
    <nose.result.TextTestResult run=10 errors=0 failures=0> 
