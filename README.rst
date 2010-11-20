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
    10000 loops, best of 3: 67.5 us per loop
    >> timeit ny.nansum(arr)
    100000 loops, best of 3: 18.2 us per loop

Let's not forget to add some NaNs::

    >> arr[arr > 0.5] = np.nan 
    >> timeit np.nansum(arr)
    1000 loops, best of 3: 411 us per loop
    >> timeit ny.nansum(arr)
    10000 loops, best of 3: 65 us per loop

Nanny uses a separate Cython function for each combination of ndim, dtype, and
axis. You can get rid of a lot of overhead (useful in an inner loop, e.g.) by
using the function that matches your problem::
              
    >> arr = np.random.rand(10, 10)

    >> timeit np.nanmax(arr, axis=1)
    10000 loops, best of 3: 26.5 us per loop
    >> timeit ny.nanmax(arr, axis=1)
    100000 loops, best of 3: 5.82 us per loop
    >> timeit ny.func.nanmax_2d_float64_axis1(arr)
    100000 loops, best of 3: 2.56 us per loop

Note that nanmax_2d_float64_axis1 is faster than the Numpy's non-nan version
of max::

    >> timeit np.max(arr, axis=1)
    100000 loops, best of 3: 3.7 us per loop

I put together Nanny as a way to learn Cython. It currently only supports:

- functions: nansum, nanmax
- Operating systems: 64-bit (accumulator for int32 is hard coded to int64)
- dtype: int32, int64, float64
- ndim: 1, 2, and 3

If there is interest in the project, I could continue adding the remaining NaN
functions from NumPy and SciPy: nanmin, nanmax, nanmean, nanmedian (using a
partial sort), nanstd. But why stop there? How about nancumsum or nanprod? Or
anynan, which could short-circuit once a NaN is found?

Feedback on the code or the direction of the project are welcomed. So is
coding help---without that I doubt the package will ever be completed. Once
nansum is complete, many of the remaining functions will be copy, paste, touch
up operations.

Remember, Nanny quickly protects your precious data from the corrupting
influence of Mr. Nan.


License
=======

Nanny is distributed under a Simplified BSD license. Parts of NumPy, which has
a BSD licenses, are included in Nanny. See the LICENSE file, which is
distributed with Nanny, for details.


Installation
============

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
    Ran 2 tests in 0.640s
    OK
    <nose.result.TextTestResult run=2 errors=0 failures=0> 
