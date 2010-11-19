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

Nanny is in the prototype stage. It currently contains only one NaN function:
nansum. And only np.int64 and np.float64 dtypes are currently supported.

If there is interest in the project, I will continue adding the remaining NaN
functions from NumPy and Scipy: nanmin, nanmax, nanmean, nanmedian, nanstd.
But why stop there? How about nancumsum or nanprod? Or anynan, which could
short-circuit once a NaN is found?

If I'm the only one excited by all this, then I'll probably fold this project
into the labeled-array package, la.

Feedback on the code or the direction of the project are welcomed.


License
=======

Nanny is distributed under a Simplified BSD license. Parts of NumPy, which has
a BSD licenses, are included in Nanny. See the LICENSE file, which is
distributed with Nanny, for details.


Installation
============

You can grab Nanny at http://github.com/kwgoodman/nanny.

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
    Ran 1 tests in 0.008s
    OK
    <nose.result.TextTestResult run=1 errors=0 failures=0> 
