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
moving window         ``move_sum, move_mean, move_std, move_median, move_min,
                      move_max``
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
        nansum         39.8        4.1       40.2        9.3
        nanmean       152.3        5.3      159.6       10.4
        nanstd        273.6        4.2      258.4        8.4
        nanmax         35.4        1.1       35.1        2.9
        partsort        3.9        2.9        4.0        3.5
        argpartsort     0.9        2.4        1.0        1.7
        replace        11.0        1.3       10.9        1.3
        move_sum       37.6       66.5       35.0       67.7
        move_nansum    71.9      135.3       74.7      137.5
        move_mean      31.9       31.8       32.0       68.1
        move_nanmean   76.1       65.6       79.0       65.9
        move_std       57.0       22.1       60.5      132.5
        move_nanstd    99.0       34.6      100.8       34.9
        move_max       14.5       20.2       14.5       60.6
        move_nanmax    49.3       37.8       51.4      105.4
        move_median   358.9       34.4      363.0       22.2

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
    move_max       sp.ndimage.maximum_filter1d based, window=a.shape[-1] // 5
    move_nanmax    sp.ndimage.maximum_filter1d based, window=a.shape[-1] // 5
    move_median    for loop with np.median

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
    Ran 68 tests in 47.712s
    OK
    <nose.result.TextTestResult run=68 errors=0 failures=0>
