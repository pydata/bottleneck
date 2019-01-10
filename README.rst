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
        Bottleneck 1.3.0; Numpy 1.15.4
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-fifth NaNs; float64 used

                  no NaN     no NaN      NaN       no NaN      NaN
                   (100,)  (1000,1000)(1000,1000)(1000,1000)(1000,1000)
                   axis=0     axis=0     axis=0     axis=1     axis=1
    nansum         74.3        1.7        2.0        2.3        2.4
    nanmean       223.8        2.3        2.4        3.3        2.9
    nanstd        295.4        2.0        2.2        2.6        2.6
    nanvar        273.5        1.9        2.1        2.6        2.5
    nanmin         55.7        0.7        2.1        1.0        3.3
    nanmax         49.7        0.7        1.9        0.8        2.6
    median        160.4        1.3        6.1        1.1        6.2
    nanmedian     164.5        7.5        8.6        7.4        8.6
    ss             29.7        1.1        1.1        1.6        1.6
    nanargmin      87.9        3.0        5.3        2.3        5.9
    nanargmax      94.1        3.2        5.4        2.3        5.9
    anynan         21.0        0.3       40.8        0.5       33.7
    allnan         37.5      142.9      121.3      114.5       97.4
    rankdata       68.3        2.4        2.4        2.8        2.8
    nanrankdata    70.6        2.6        2.5        3.1        3.0
    partition       4.0        1.2        1.6        1.0        1.4
    argpartition    3.6        1.1        1.4        1.1        1.6
    replace        14.1        1.6        1.6        1.6        1.6
    push         2947.5        8.4        9.2       20.6       15.7
    move_sum     4561.5       75.7      153.6      189.1      203.5
    move_mean   11003.3      116.2      184.7      224.1      245.9
    move_std    10748.8       95.7      196.4      135.0      249.4
    move_var    13999.7      126.6      240.1      198.8      316.4
    move_min     2259.7       20.1       38.6       25.0       43.8
    move_max     2189.3       20.4       38.9       25.2       43.9
    move_argmin  4389.4       41.7       76.3       42.0       85.1
    move_argmax  4464.5       42.5       76.0       42.4       83.6
    move_median  2969.5      163.4      155.6      154.2      155.0
    move_rank    1395.6        1.9        2.0        2.4        2.5

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
 docs                 https://kwgoodman.github.io/bottleneck-doc
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
Documentation            sphinx, numpydoc
======================== ====================================================

To install Bottleneck on GNU/Linux, Mac OS X, et al.::

    $ sudo python setup.py install

To install bottleneck on Windows, first install MinGW and add it to your
system path. Then install Bottleneck with the command::

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