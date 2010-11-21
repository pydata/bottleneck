=====
Nanny
=====

Nanny uses the magic of Cython to give you a faster, drop-in replacement for
the NaN functions in NumPy and SciPy.

For example::

    >> import nanny as ny
    >> import numpy as np
    >> arr = np.random.rand(100, 100)
    
    >> timeit np.nansum(arr)
    10000 loops, best of 3: 68.4 us per loop
    >> timeit ny.nansum(arr)
    100000 loops, best of 3: 17.7 us per loop

Let's not forget to add some NaNs::

    >> arr[arr > 0.5] = np.nan
    >> timeit np.nansum(arr)
    1000 loops, best of 3: 417 us per loop
    >> timeit ny.nansum(arr)
    10000 loops, best of 3: 64.8 us per loop

Remember, Nanny quickly protects your precious data from the corrupting
influence of Mr. Nan.

Fast
====

Nanny comes with a benchmark suite. To run it::
    
    >>> ny.benchit(verbose=False)
    Nanny performance benchmark
        Nanny 0.0.1dev
        Numpy 1.4.1
        Speed is numpy time divided by nanny time
        NaN means all NaNs
       Speed   Test                Shape        dtype    NaN?
       6.6884  nansum(a, axis=-1)  (500,500)    int64  
       4.7998  nansum(a, axis=-1)  (10000,)     float64  
      13.1576  nansum(a, axis=-1)  (500,500)    int32  
       3.0553  nansum(a, axis=-1)  (500,500)    float64  
      12.2908  nansum(a, axis=-1)  (10000,)     int32  
       6.7347  nansum(a, axis=-1)  (10000,)     int64  
      51.9741  nansum(a, axis=-1)  (500,500)    float64  NaN
      14.2142  nansum(a, axis=-1)  (10000,)     float64  NaN
       6.5308  nanmax(a, axis=-1)  (500,500)    int64  
       9.0980  nanmax(a, axis=-1)  (10000,)     float64  
       0.2057  nanmax(a, axis=-1)  (500,500)    int32  
       6.9793  nanmax(a, axis=-1)  (500,500)    float64  
       5.1443  nanmax(a, axis=-1)  (10000,)     int32  
       6.7844  nanmax(a, axis=-1)  (10000,)     int64  
      49.7016  nanmax(a, axis=-1)  (500,500)    float64  NaN
      14.9575  nanmax(a, axis=-1)  (10000,)     float64  NaN

Faster
======

Under the hood Nanny uses a separate Cython function for each combination of
ndim, dtype, and axis. A lot of the overhead in ny.nanmax, for example, is
in checking the your axis is within range, converting non-array data to an
array, and selecting the function to use to calculate nanmax.

You can get rid of the overhead by doing all this before you, say, enter
an inner loop::

    >>> arr = np.random.rand(10,10)
    >>> axis = 0
    >>> func, a = ny.func.nanmax_selector(arr, axis)
    >>> func.__name__
    'nanmax_2d_float64_axis0'

Let's see how much faster than runs::    
    
    >> timeit np.nanmax(arr, axis=0)
    10000 loops, best of 3: 25.7 us per loop
    >> timeit ny.nanmax(arr, axis=0)
    100000 loops, best of 3: 5.25 us per loop
    >> timeit func(a)
    100000 loops, best of 3: 2.5 us per loop

Note that ``func`` is faster than the Numpy's non-nan version of max::
    
    >> timeit arr.max(axis=0)
    100000 loops, best of 3: 3.28 us per loop

So adding NaN protection to your inner loops has a negative cost!           

Functions
=========

Nanny is in the prototype stage. (Feedback welcomed!)

It currently contains the following functions: nanmax, nanmin, nansum.

Functions that will appear in later releases of Nanny: nanmean, nanstd,
nanmedian (using a partial sort).

It may also be useful to add functions that do not currently appear in NumPy
or SciPy: nancumsum, nanprod, etc. And perhaps functions like anynan, which
could short-circuit once a NaN is found.

License
=======

Nanny is distributed under a Simplified BSD license. Parts of NumPy and Scipy,
which both have BSD licenses, are included in Nanny. See the LICENSE file,
which is distributed with Nanny, for details.

Install
=======

You can grab Nanny at http://github.com/kwgoodman/nanny.

nansum of ints is only supported by 64-bit operating systems at the moment. 

**GNU/Linux, Mac OS X, et al.**

To install Nanny::

    $ python setup.py build
    $ sudo python setup.py install
    
Or, if you wish to specify where Nanny is installed, for example inside
``/usr/local``::

    $ python setup.py build
    $ sudo python setup.py install --prefix=/usr/local

**Windows**

In order to compile the C code in Nanny you need a Windows version of the gcc
compiler. MinGW (Minimalist GNU for Windows) contains gcc and has been used to successfully compile Nanny on Windows.

Install MinGW and add it to your system path. Then install Nanny with the
commands::

    python setup.py build --compiler=mingw32
    python setup.py install

**Post install**

After you have installed Nanny, run the suite of unit tests::

    >>> import nanny
    >>> nanny.test()
    <snip>
    Ran 5 tests in 1.390s
    OK
    <nose.result.TextTestResult run=5 errors=0 failures=0> 
