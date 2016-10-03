.. image:: https://travis-ci.org/kwgoodman/bottleneck.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/bottleneck
.. image:: https://ci.appveyor.com/api/projects/status/github/kwgoodman/bottleneck?svg=true&passingText=passing&failingText=failing&pendingText=pending
    :target: https://ci.appveyor.com/project/kwgoodman/bottleneck
==========
Bottleneck
==========

Bottleneck is a collection of fast NumPy array functions written in C.

Let's give it a try. Create a NumPy array::

    >>> import numpy as np
    >>> a = np.array([1, 2, np.nan, 4, 5])

Find the nanmean::

    >>> import bottleneck as bn
    >>> bn.nanmean(a)
    3.0

Moving window mean::

    >>> bn.move_mean(a, window=2, min_count=1)
    array([ 1. ,  1.5,  2. ,  4. ,  4.5])

Benchmark
=========

Bottleneck comes with a benchmark suite::

    >>> bn.bench()
    Bottleneck performance benchmark
        Bottleneck 1.2.0dev; Numpy 1.11.0
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-fifth NaNs; float64 and axis=-1 are used

                      no NaN      NaN       no NaN      NaN
                       (100,)    (1000,)  (1000,1000)(1000,1000)
        nansum         59.3       16.7        2.3        5.0
        nanmean       208.7       47.0        3.5        6.1
        nanstd        242.9       43.0        2.7        4.9
        nanvar        231.7       40.8        2.7        4.9
        nanmin         32.0        5.3        0.4        0.7
        nanmax         32.1        5.4        0.4        0.8
        median        106.4       53.6        1.0        5.3
        nanmedian     106.4       29.2        4.9       32.4
        ss             27.9        6.5        1.6        1.6
        nanargmin      55.7        9.4        1.2        2.0
        nanargmax      56.4        9.6        1.2        2.3
        anynan         22.3       49.8        0.5      112.8
        allnan         42.1       46.6      114.4       65.9
        rankdata       50.9        8.2        2.7        6.6
        nanrankdata    56.8        8.5        2.9        7.1
        partition       4.0        3.5        1.0        1.7
        argpartition    3.0        2.5        1.1        1.5
        replace        12.5        3.9        1.5        1.4
        push         3432.7     9640.6       19.7       20.0
        move_sum     4799.6     9430.4      194.7      374.2
        move_mean   11372.1    19312.2      250.4      398.1
        move_std     9343.7    12246.2      134.1      242.7
        move_var    12343.5    18427.2      209.8      385.5
        move_min     2343.7     3806.8       25.2       85.1
        move_max     2122.2     4308.0       25.1      109.0
        move_argmin  3703.3     5836.4       40.5      179.1
        move_argmax  3608.2     7683.3       42.9      226.1
        move_median  1772.2     1143.7       42.8      114.2
        move_rank    1282.9      241.4        2.5        7.3

You can also run a detailed benchmark for a single function using, for
example, the command::

    >>> bn.bench_detailed("move_median", fraction_nan=0.3)

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtypes result in calls to slower, unaccelerated
functions. In the rare case of a byte-swapped input array (e.g. a big-endian
array on a little-endian operating system) the function will not be
accelerated regardless of dtype.

Where
=====

===================   ========================================================
 download             https://pypi.python.org/pypi/Bottleneck
 docs                 http://berkeleyanalytics.com/bottleneck
 code                 https://github.com/kwgoodman/bottleneck
 mailing list         https://groups.google.com/group/bottle-neck
===================   ========================================================

License
=======

Bottleneck is distributed under a Simplified BSD license. See the LICENSE file
for details.

Install
=======

Requirements:

======================== ====================================================
Bottleneck               Python 2.7, 3.4, 3.5; NumPy 1.11.0
Compile                  gcc, clang, MinGW or MSVC
Unit tests               nose
======================== ====================================================

To install Bottleneck on GNU/Linux, Mac OS X, et al.::

    $ sudo python setup.py install

To install bottleneck on Windows, first install MinGW and add it to your
system path. Then install Bottleneck with the commands::

    python setup.py install --compiler=mingw32

Alternatively, you can use the Windows binaries created by Christoph Gohlke:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck

Unit tests
==========

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 198 tests in 17.205s
    OK
    <nose.result.TextTestResult run=198 errors=0 failures=0>