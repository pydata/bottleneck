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
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN      NaN       no NaN      NaN
                       (10,)      (100,)  (1000,1000)(1000,1000)
        nansum         98.0       56.9        2.3        7.4
        nanmean       380.8      209.8        3.3        8.1
        nanstd        603.4      239.7        2.7        7.1
        nanvar        614.3      246.1        2.6        6.9
        nanmin         87.4       51.5        1.0        1.4
        nanmax         88.2       42.4        1.0        2.1
        median        239.8      182.4        1.1        5.1
        nanmedian     248.9      127.3        5.0       39.8
        ss             50.8       28.4        1.6        1.6
        nanargmin     171.0       97.5        2.3        6.1
        nanargmax     170.8       98.9        2.3        6.5
        anynan         39.4       44.0        0.6       76.7
        allnan         42.8       42.8      110.0       86.4
        rankdata      140.4       50.5        2.7        5.0
        nanrankdata   150.6       57.1        2.9        4.8
        partition       9.3        7.5        1.0        1.1
        argpartition    8.2        5.0        1.1        1.0
        replace        19.2       12.2        1.4        1.4
        push          546.8     3329.4       20.1       27.1
        move_sum      936.2     4056.2      192.1      442.4
        move_mean    2160.0     9471.2      234.2      379.2
        move_std     3411.7     6630.1      129.5      205.5
        move_var     3572.4     9862.9      203.4      351.7
        move_min      689.3     2379.6       25.3       60.4
        move_max      661.3     2549.4       25.2      105.0
        move_argmin  1082.1     3744.3       40.7      240.5
        move_argmax  1130.6     4461.5       43.1      303.4
        move_median  1126.4     1823.9       42.6      142.5
        move_rank    1608.3     2222.5        2.3        9.0

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