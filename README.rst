==========
Bottleneck
==========

Bottleneck is a collection of fast NumPy array functions written in Cython:

===================== =======================================================
reduce                ``nansum, nanmean, nanstd, nanvar, nanmin, nanmax,
                      median, nanmedian, nanargmin, nanargmax, anynan, allnan,
                      ss``
non-reduce            ``replace``
non-reduce (axis)     ``partsort, argpartsort, rankdata, nanrankdata``
moving window         ``move_sum, move_nansum, move_mean, move_nanmean,
                      move_median, move_std, move_nanstd, move_min,
                      move_nanmin, move_max, move_nanmax``
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

Benchmark
=========

Bottleneck comes with a benchmark suite. To run the benchmark::

    >>> bn.bench()
    Bottleneck performance benchmark
        Bottleneck  1.0.0dev
        Numpy (np)  1.9.1
        Scipy (sp)  0.14.0
        Speed is NumPy or SciPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                     (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum       35.9        3.9       36.2        9.1
        nanmean     154.2        5.2      154.4       10.2
        nanstd      266.3        4.4      252.6        8.6
        nanmax       34.4        1.1       34.5        2.8
        partsort      3.7        2.9        3.8        3.5
        argpartsort   1.0        2.2        1.0        1.4
        replace       9.8        1.3        9.8        1.3
        move_sum     30.2       65.6       29.1       67.6
        move_nansum  66.8      130.3       71.4      122.5
        move_mean    28.4       31.2       28.5       66.4
        move_nanmean 73.6       66.0       77.7       66.2

    Reference functions:
    nansum         np.nansum
    nanmean        np.nanmean
    nanstd         np.nanstd
    nanmax         np.nanmax
    partsort       np.sort, n=max(a.shape[-1]/2,1)
    argpartsort    np.argsort, n=max(a.shape[-1]/2,1)
    replace        np.putmask based (see bn.slow.replace)
    move_sum       sp.ndimage.convolve1d based, window=a.shape[-1] // 5
    move_nansum    sp.ndimage.convolve1d based, window=a.shape[-1] // 5
    move_mean      sp.ndimage.convolve1d based, window=a.shape[-1] // 5
    move_nanmean   sp.ndimage.convolve1d based, window=a.shape[-1] // 5

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtypes result in calls to slower, unaccelerated
functions.

Where
=====

===================   ========================================================
 download             http://pypi.python.org/pypi/Bottleneck
 docs                 http://berkeleyanalytics.com/bottleneck
 code                 http://github.com/kwgoodman/bottleneck
 mailing list         http://groups.google.com/group/bottle-neck
===================   ========================================================

License
=======

Bottleneck is distributed under a Simplified BSD license. See the LICENSE file
for details.

Install
=======

Requirements:

======================== ====================================================
Bottleneck               Python 2.7, 3.4; **NumPy 1.9.1**
Compile                  gcc or clang or MinGW
Unit tests               nose
======================== ====================================================

Optional:

======================== ====================================================
SciPy                    Portions of benchmark suite
tox, virtualenv          Run unit tests across multiple python/numpy versions
Cython                   Development of bottleneck
======================== ====================================================

To install Bottleneck on GNU/Linux, Mac OS X, et al.::

    $ python setup.py build
    $ sudo python setup.py install

To install bottleneck on Windows, first install MinGW and add it to your
system path. Then install Bottleneck with the commands::

    python setup.py build --compiler=mingw32
    python setup.py install

Alternatively, you can use the Windows binaries created by Christoph Gohlke:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck

Unit tests
==========

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 45 tests in 28.712s
    OK
    <nose.result.TextTestResult run=45 errors=0 failures=0>
