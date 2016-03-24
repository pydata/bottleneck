==========
Bottleneck
==========

Bottleneck is a collection of fast NumPy array functions written in Cython.

Let's give it a try. Create a NumPy array::

    >>> import numpy as np
    >>> arr = np.array([1, 2, np.nan, 4, 5])

Find the nanmean::

    >>> import bottleneck as bn
    >>> bn.nanmean(arr)
    3.0

Moving window mean::

    >>> bn.move_mean(arr, window=2, min_count=1)
    array([ 1. ,  1.5,  2. ,  4. ,  4.5])

Benchmark
=========

Bottleneck comes with a benchmark suite::

    >>> bn.bench()
    Bottleneck performance benchmark
        Bottleneck  1.1.0dev
        Numpy (np)  1.10.4
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum         34.1        3.9       33.7        6.5
        nanmean       119.2        5.7      120.9        8.3
        nanstd        209.1        4.9      202.3        7.5
        nanvar        207.2        4.5      210.1        7.0
        nanmin         34.7        1.1       30.5        1.4
        nanmax         29.9        1.0       30.2        1.7
        median         42.7        0.8       51.1        1.0
        nanmedian      49.8        2.5       57.4        6.0
        ss             14.0        3.9        3.1        1.1
        nanargmin      55.2        4.0       55.5        5.2
        nanargmax      55.9        3.9       56.5        6.6
        anynan         13.4        1.5       14.1      175.9
        allnan         13.6      274.4       13.7      199.9
        rankdata       38.4        1.6       38.2        2.3
        nanrankdata    46.0       20.1       42.4       27.8
        partsort        5.0        0.8        5.1        1.1
        argpartsort     2.6        0.8        2.7        0.6
        replace         9.8        1.3       11.5        1.4
        move_sum      296.2      199.2      299.5      360.1
        move_mean     792.5      284.4      710.8      463.6
        move_std     1172.5      172.8     1251.7      458.8
        move_min      225.8       29.0      223.1       57.0
        move_max      231.0       27.7      237.9       74.8
        move_median   386.4       41.0      380.4       62.3

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
Bottleneck               Python 2.7, 3.4; NumPy 1.10.4
Compile                  gcc or clang or MinGW
Unit tests               nose
======================== ====================================================

Optional:

======================== ====================================================
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
    Ran 79 tests in 70.712s
    OK
    <nose.result.TextTestResult run=79 errors=0 failures=0>
