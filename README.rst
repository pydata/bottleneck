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
        nansum         36.1        4.0       36.7        9.2
        nanmean       137.8        5.2      141.3       10.3
        nanstd        243.5        4.2      244.2        8.3
        nanmax         32.1        1.1       32.2        2.8
        partsort        3.2        2.9        3.2        3.4
        argpartsort     0.9        2.4        1.0        1.7
        replace        10.4        1.2       10.4        1.3
        move_sum       31.0       66.9       28.7       67.9
        move_nansum    61.7      135.7       62.2      136.4
        move_mean      27.0       32.1       27.3       65.3
        move_nanmean   66.5       65.6       68.9       65.7
        move_std       51.3       22.1       55.6      135.5
        move_nanstd    94.1       34.5       97.3       34.8

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
    move_std       sp.ndimage.convolve1d based, window=a.shape[-1] // 5
    move_nanstd    sp.ndimage.convolve1d based, window=a.shape[-1] // 5

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
    Ran 51 tests in 36.712s
    OK
    <nose.result.TextTestResult run=51 errors=0 failures=0>
