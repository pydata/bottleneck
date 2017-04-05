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
        Bottleneck 1.3.0.dev0; Numpy 1.12.1
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-fifth NaNs; float64 used

                  no NaN     no NaN      NaN       no NaN      NaN
                   (100,)  (1000,1000)(1000,1000)(1000,1000)(1000,1000)
                   axis=0     axis=0     axis=0     axis=1     axis=1
    nansum         67.3        0.3        0.7        2.5        2.4
    nanmean       194.8        1.9        2.1        3.4        3.1
    nanstd        241.5        1.6        2.1        2.7        2.6
    nanvar        229.7        1.7        2.1        2.7        2.5
    nanmin         34.1        0.7        1.1        0.8        2.6
    nanmax         45.6        0.7        2.7        1.0        3.7
    median        111.0        1.3        5.6        1.0        4.8
    nanmedian     108.8        5.9        6.7        5.6        6.7
    ss             16.3        1.1        1.2        1.6        1.6
    nanargmin      89.2        2.9        5.1        2.2        5.6
    nanargmax     107.4        3.0        5.4        2.2        5.8
    anynan         19.4        0.3       35.0        0.5       29.9
    allnan         39.9      146.6      128.3      115.8       75.6
    rankdata       55.0        2.6        2.3        2.9        2.8
    nanrankdata    59.8        2.8        2.2        3.2        2.5
    partition       4.4        1.2        1.6        1.0        1.4
    argpartition    3.5        1.1        1.4        1.1        1.6
    replace        17.7        1.4        1.4        1.3        1.4
    push         3440.0        7.8        9.5       20.0       15.5
    move_sum     4743.0       75.7      156.1      195.4      211.1
    move_mean    8760.9      116.2      167.4      252.1      258.8
    move_std     8979.9       96.1      196.3      144.0      256.3
    move_var    11216.8      127.3      243.9      225.9      321.4
    move_min     2245.3       20.6       36.7       23.2       42.1
    move_max     2223.7       20.5       37.2       24.1       42.4
    move_argmin  3664.0       48.2       73.3       40.2       83.9
    move_argmax  3916.9       42.0       75.4       41.5       81.2
    move_median  2023.3      166.8      173.7      153.8      154.3
    move_rank    1208.5        1.9        1.9        2.5        2.8

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
Bottleneck               Python 2.7, 3.5, 3.6; NumPy 1.12.1
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
    Ran 169 tests in 57.205s
    OK
    <nose.result.TextTestResult run=169 errors=0 failures=0>
