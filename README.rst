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
        Bottleneck 1.3.0.dev0; Numpy 1.11.3
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-fifth NaNs; float64 used

                  no NaN     no NaN      NaN       no NaN      NaN
                   (100,)  (1000,1000)(1000,1000)(1000,1000)(1000,1000)
                   axis=0     axis=0     axis=0     axis=1     axis=1
    nansum         62.3        1.6        2.0        2.3        2.5
    nanmean       230.7        2.4        2.4        3.5        2.9
    nanstd        260.2        2.1        2.2        2.7        2.6
    nanvar        250.5        2.1        2.3        2.8        2.6
    nanmin         43.3        0.7        1.9        0.8        2.6
    nanmax         45.9        0.7        2.1        1.0        3.3
    median        108.6        1.3        6.2        1.1        6.2
    nanmedian     108.8        5.7        6.6        5.5        6.6
    ss             27.2        1.2        1.2        1.6        1.6
    nanargmin      80.8        3.1        5.4        2.3        6.0
    nanargmax      95.0        3.2        5.4        2.3        6.0
    anynan         18.1        0.3       34.5        0.5       29.6
    allnan         39.7      146.3      126.9      117.0       96.3
    rankdata       56.1        2.5        2.5        2.8        2.9
    nanrankdata    60.8        2.7        2.7        3.1        3.0
    partition       4.1        1.2        1.6        1.0        1.4
    argpartition    3.0        1.1        1.4        1.1        1.6
    replace        12.3        1.4        1.4        1.4        1.4
    push         3363.6        7.6        9.1       20.2       15.7
    move_sum     5046.7       67.5      147.9      192.9      211.7
    move_mean   12277.3      111.8      180.0      252.9      261.6
    move_std    10677.3       97.0      196.6      145.1      258.0
    move_var    13537.3      123.7      235.5      214.7      324.8
    move_min     2474.2       20.0       36.9       23.5       41.9
    move_max     2416.5       20.2       37.1       23.7       42.3
    move_argmin  3876.9       38.9       72.4       39.6       80.7
    move_argmax  3910.2       40.3       73.9       41.2       81.0
    move_median  2087.3      148.2      161.9      148.4      160.7
    move_rank    1312.5        1.8        2.1        2.3        2.7

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
Bottleneck               Python 2.7, 3.4, 3.5; NumPy 1.11.3
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