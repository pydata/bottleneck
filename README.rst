====
DSNA
====

DSNA uses the magic of Cython to give you fast, NaN-aware descriptive
statistics of NumPy arrays.

For example::

    >> import dsna as ds
    >> import numpy as np
    >> arr = np.random.rand(100, 100)
    
    >> timeit np.nansum(arr)
    10000 loops, best of 3: 68.4 us per loop
    >> timeit ds.nansum(arr)
    100000 loops, best of 3: 17.7 us per loop

Let's not forget to add some NaNs::

    >> arr[arr > 0.5] = np.nan
    >> timeit np.nansum(arr)
    1000 loops, best of 3: 417 us per loop
    >> timeit ds.nansum(arr)
    10000 loops, best of 3: 64.8 us per loop

Remember, dsna quickly protects your precious data from the corrupting
influence of Mr. Nan.

Fast
====

DSNA comes with a benchmark suite. To run it::
    
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
ndim, dtype, and axis. A lot of the overhead in ds.nanmax, for example, is
in checking that your axis is within range, converting non-array data to an
array, and selecting the function to use to calculate nanmax.

You can get rid of the overhead by doing all this before you, say, enter
an inner loop::

    >>> arr = np.random.rand(10,10)
    >>> axis = 0
    >>> func, a = ds.func.nanmax_selector(arr, axis)
    >>> func.__name__
    'nanmax_2d_float64_axis0'

Let's see how much faster than runs::    
    
    >> timeit np.nanmax(arr, axis=0)
    10000 loops, best of 3: 25.7 us per loop
    >> timeit ds.nanmax(arr, axis=0)
    100000 loops, best of 3: 5.25 us per loop
    >> timeit func(a)
    100000 loops, best of 3: 2.5 us per loop

Note that ``func`` is faster than the Numpy's non-nan version of max::
    
    >> timeit arr.max(axis=0)
    100000 loops, best of 3: 3.28 us per loop

So adding NaN protection to your inner loops has a negative cost!           

Functions
=========

DSNA is in the prototype stage. (Feedback welcomed!)

DSNA currently contains the following functions: nanmax, nanmin, nanmean,
nanstd, nanvar, nansum.

Functions that will appear in later releases of dsna: nanmedian (using a
partial sort).

It may also be useful to add functions that do not currently appear in NumPy
or SciPy: nancumsum, nanprod, etc. And perhaps functions like anynan, which
could short-circuit once a NaN is found.

Currently only 1d, 2d, and 3d arrays with NumPy dtype int32, int64, and float64 are supported.

License
=======

DSNA is distributed under a Simplified BSD license. Parts of NumPy and Scipy,
which both have BSD licenses, are included in dsna. See the LICENSE file,
which is distributed with dsna, for details.

Install
=======

You can grab dsna at http://github.com/kwgoodman/dsna

nansum of ints is only supported by 64-bit operating systems at the moment. 

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
