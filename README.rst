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
        nansum         41.0        3.9       41.2        9.0
        nanmean       154.0        5.2      154.1       10.1
        nanstd        241.8        4.3      241.1        8.5
        nanmax         33.6        1.1       30.1        2.8
        partsort        3.3        2.9        3.4        3.5
        argpartsort     0.8        2.2        0.8        1.4
        replace        10.9        1.3       10.9        1.3
        move_sum       31.4       65.8       31.0       67.8
        move_nansum    68.6      135.0       72.6      137.9
        move_mean      27.7       31.8       28.9       67.9
        move_nanmean   77.1       66.1       80.3       66.3

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
