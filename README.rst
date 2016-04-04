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
        Numpy (np)  1.11.0
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum         34.9        3.8       35.7        5.6
        nanmean       125.1        5.8      120.9        8.0
        nanstd        205.8        4.6      210.6        7.2
        nanvar        209.1        4.2      186.1        6.9
        nanmin         31.1        1.1       30.9        1.3
        nanmax         31.4        1.0       30.6        1.7
        median         49.2        0.8       56.8        1.0
        nanmedian      47.9        2.5       55.1        6.0
        ss             14.2        3.9        3.1        1.1
        nanargmin      55.0        3.9       57.2        5.4
        nanargmax      56.1        4.0       56.3        6.6
        anynan         13.6        1.5       14.0      191.9
        allnan         14.0      275.8       14.1      206.4
        rankdata       38.4        1.5       45.0        2.0
        nanrankdata    48.4       20.1       42.9       28.9
        partsort        4.5        0.8        4.8        1.1
        argpartsort     2.7        0.7        2.8        0.6
        replace         9.2        1.3        9.1        1.3
        move_sum      324.3      187.5      324.4      315.5
        move_mean     735.0      245.9      800.3      435.7
        move_std     1235.9      123.7     1320.2      378.9
        move_var     1251.4      179.3     1292.7      374.0
        move_min      231.7       23.8      242.2       49.0
        move_max      246.1       21.7      241.5       72.7
        move_argmax   373.5       75.2      398.8      259.0
        move_median   382.1       36.5      383.3       50.2

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtypes result in calls to slower, unaccelerated
functions. In the rare case of a byte-swapped input array (e.g. a big-endian
array on a little-endian operating system) the function will not be
accelerated regardless of dtype.

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
Bottleneck               Python 2.7, 3.4; NumPy 1.11.0
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
    Ran 85 tests in 70.712s
    OK
    <nose.result.TextTestResult run=85 errors=0 failures=0>
