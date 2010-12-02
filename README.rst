==========
Bottleneck
==========

Introduction
============

Bottleneck is a collection of fast NumPy array functions written in Cython.

The three categories of Bottleneck functions:

- Faster replacement for NumPy and SciPy functions
- Moving window functions
- Group functions that bin calculations by like-labeled elements  

Function signatures (using nanmean as an example):

===============  ===================================================
 Functions        ``nanmean(arr, axis=None)``
 Moving window    ``move_mean(arr, window, axis=0)``
 Group by         ``group_nanmean(arr, label, order=None, axis=0)``
===============  ===================================================

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
    
    >>> bn.benchit(verbose=False)
    Bottleneck performance benchmark
        Bottleneck  0.2.0dev
        Numpy       1.5.1
        Scipy       0.8.0
        Speed is numpy (or scipy) time divided by Bottleneck time
        NaN means all NaNs
       Speed   Test                  Shape        dtype    NaN?
       2.5780  median(a, axis=0)     (500,500)    float64  
       2.3277  median(a, axis=0)     (500,500)    float64  NaN
       4.1347  median(a, axis=0)     (10000,)     float64  
       4.7842  median(a, axis=0)     (10000,)     float64  NaN
       6.7673  nanmax(a, axis=0)     (500,500)    float64  
       4.7826  nanmax(a, axis=0)     (500,500)    float64  NaN
       9.4604  nanmax(a, axis=0)     (10000,)     float64  
       8.3246  nanmax(a, axis=0)     (10000,)     float64  NaN
       4.0559  nanmin(a, axis=0)     (500,500)    float64  
       4.5415  nanmin(a, axis=0)     (500,500)    float64  NaN
       6.7648  nanmin(a, axis=0)     (10000,)     float64  
       8.4638  nanmin(a, axis=0)     (10000,)     float64  NaN
      11.4441  nanmean(a, axis=0)    (500,500)    float64  
      55.3499  nanmean(a, axis=0)    (500,500)    float64  NaN
      12.8546  nanmean(a, axis=0)    (10000,)     float64  
      22.7069  nanmean(a, axis=0)    (10000,)     float64  NaN
       8.5651  nanstd(a, axis=0)     (500,500)    float64  
      62.3893  nanstd(a, axis=0)     (500,500)    float64  NaN
       8.8121  nanstd(a, axis=0)     (10000,)     float64  
      26.3563  nanstd(a, axis=0)     (10000,)     float64  NaN

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
    
    >> timeit np.nanmax(arr, axis=0)
    10000 loops, best of 3: 25.7 us per loop
    >> timeit bn.nanmax(arr, axis=0)
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
median
nanmean      move_nanmean     group_nanmean
nanvar                  
nanstd          
nanmin          
nanmax          
=========    ==============   ===============

Currently only 1d, 2d, and 3d NumPy arrays with dtype int32, int64, and
float64 are supported.

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
Bottleneck               Python, NumPy 1.5.1+, SciPy 0.8.0+
Unit tests               nose
Compile                  gcc or MinGW
======================== ===================================

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
compiler. MinGW (Minimalist GNU for Windows) contains gcc and has been used
to successfully compile dsna on Windows.

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
