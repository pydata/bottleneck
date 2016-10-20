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
        Bottleneck 1.2.0; Numpy 1.11.2
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-fifth NaNs; float64 and axis=-1 are used

                      no NaN      NaN       no NaN      NaN
                       (100,)    (1000,)  (1000,1000)(1000,1000)
        nansum         58.3       16.6        2.3        5.1
        nanmean       258.7       46.1        3.5        5.1
        nanstd        238.4       42.9        2.8        5.0
        nanvar        229.9       41.4        2.7        5.0
        nanmin         44.6       12.9        0.8        0.9
        nanmax         41.8       12.9        0.8        1.8
        median         99.6       51.4        1.1        5.7
        nanmedian     102.1       26.5        5.0       31.2
        ss             27.4        6.4        1.6        1.6
        nanargmin      72.6       24.6        2.3        3.4
        nanargmax      70.1       29.2        2.4        4.6
        anynan         22.1       49.9        0.5      114.6
        allnan         43.3       48.4      115.8       66.7
        rankdata       50.3        8.0        2.6        6.5
        nanrankdata    52.5        8.1        2.9        6.8
        partition       4.1        3.6        1.0        2.0
        argpartition    2.7        2.2        1.1        1.5
        replace        13.7        4.9        1.5        1.5
        push         3231.6     7437.4       20.1       19.6
        move_sum     4173.5     8955.4      194.7      374.8
        move_mean   10265.5    18540.0      222.8      372.2
        move_std     8910.9    12158.5      128.7      234.5
        move_var    11969.4    18323.8      202.7      378.7
        move_min     2164.6     3676.3       23.9       57.2
        move_max     1995.0     4206.0       23.8      108.8
        move_argmin  3380.5     5559.1       40.5      180.5
        move_argmax  3386.5     7278.1       43.0      227.2
        move_median  1762.3     1134.9      157.9      118.5
        move_rank    1203.6      223.2        2.7        7.8

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
Bottleneck               Python 2.7, 3.4, 3.5; NumPy 1.11.2
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
