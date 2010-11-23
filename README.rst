====
DSNA
====

Introduction
============

DSNA uses the magic of Cython to give you fast, NaN-aware descriptive
statistics of NumPy arrays.

The functions in dsna fall into three categories:

===============  ===============================
 General          sum(arr, axis=None)
 Moving window    move_sum(arr, window, axis=0)
 Group by         group_sum(arr, label, axis=0)
===============  ===============================

For example, create a NumPy array::
    
    >>> import numpy as np
    >>> arr = np.array([1, 2, np.nan, 4, 5])

Then find the sum::

    >>> import dsna as ds
    >>> ds.sum(arr)
    12.0

Moving window sum::

    >>> ds.move_sum(arr, window=2)
    array([ nan,   3.,   2.,   4.,   9.])

Group sum::   

    >>> label = ['a', 'a', 'b', 'b', 'a']
    >>> a, lab = ds.group_sum(arr, label)
    >>> a
    array([ 8.,  4.])
    >>> lab
    ['a', 'b']

Fast
====

DNSA is fast::

    >>> import dsna as ds
    >>> import numpy as np
    >>> arr = np.random.rand(100, 100)
    
    >>> timeit np.nansum(arr)
    10000 loops, best of 3: 68.4 us per loop
    >>> timeit ds.sum(arr)
    100000 loops, best of 3: 17.7 us per loop

Let's not forget to add some NaNs::

    >>> arr[arr > 0.5] = np.nan
    >>> timeit np.nansum(arr)
    1000 loops, best of 3: 417 us per loop
    >>> timeit ds.sum(arr)
    10000 loops, best of 3: 64.8 us per loop

DSNA comes with a benchmark suite that compares the performance of the dsna
functions that have a NumPy/SciPy equivalent. To run the benchmark::
    
    >>> ds.benchit(verbose=False)
    DSNA performance benchmark
        DSNA  0.0.1dev
        Numpy 1.5.1
        Scipy 0.8.0
        Speed is numpy (or scipy) time divided by dsna time
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

Under the hood dsna uses a separate Cython function for each combination of
ndim, dtype, and axis. A lot of the overhead in ds.max, for example, is
in checking that your axis is within range, converting non-array data to an
array, and selecting the function to use to calculate the maximum.

You can get rid of the overhead by doing all this before you, say, enter
an inner loop::

    >>> arr = np.random.rand(10,10)
    >>> func, a = ds.func.max_selector(arr, axis=0)
    >>> func
    <built-in function max_2d_float64_axis0> 

Let's see how much faster than runs::    
    
    >> timeit np.nanmax(arr, axis=0)
    10000 loops, best of 3: 25.7 us per loop
    >> timeit ds.max(arr, axis=0)
    100000 loops, best of 3: 5.25 us per loop
    >> timeit func(a)
    100000 loops, best of 3: 2.5 us per loop

Note that ``func`` is faster than the Numpy's non-nan version of max::
    
    >> timeit arr.max(axis=0)
    100000 loops, best of 3: 3.28 us per loop

So adding NaN protection to your inner loops has a negative cost!           

Functions
=========

DSNA is in the prototype stage.

DSNA contains the following functions (an asterisk means not yet complete): 

=========    ==============   ===============
sum*         move_sum*        group_sum*
mean         move_mean*       group_mean*
var          move_var*        group_var*
std          move_std*        group_std*
min          move_min*        group_min*
max          move_max*        group_max*
median*      move_median*     group_median*
zscore*      move_zscore*     group_zscore*
ranking*     move_ranking*    group_ranking*
quantile*    move_quantile*   group_quantile*
count*       move_count*      group_count*
=========    ==============   ===============

Currently only 1d, 2d, and 3d NumPy arrays with dtype int32, int64, and float64 are supported.

License
=======

DSNA is distributed under a Simplified BSD license. Parts of NumPy and Scipy,
which both have BSD licenses, are included in dsna. See the LICENSE file,
which is distributed with dsna, for details.

Install
=======

You can grab dsna from http://github.com/kwgoodman/dsna

**GNU/Linux, Mac OS X, et al.**

To install dsna::

    $ python setup.py build
    $ sudo python setup.py install
    
Or, if you wish to specify where dsna is installed, for example inside
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

After you have installed dsna, run the suite of unit tests::

    >>> import dsna
    >>> dsna.test()
    <snip>
    Ran 8 tests in 4.692s
    OK
    <nose.result.TextTestResult run=8 errors=0 failures=0> 
