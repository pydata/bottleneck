.. image:: https://travis-ci.org/kwgoodman/bottleneck.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/bottleneck
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

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum        117.4        2.3      114.6        7.3
        nanmean       433.2        3.5      429.7        8.2
        nanstd        665.9        2.7      680.8        6.9
        nanvar        710.8        2.6      664.4        6.9
        nanmin         96.1        1.0       94.5        1.3
        nanmax         95.0        1.0       98.4        2.1
        median        251.5        1.1      372.5        5.2
        nanmedian     268.9        5.3      335.5       41.8
        ss             58.9        1.6       58.4        1.6
        nanargmin     193.4        2.2      197.8        6.0
        nanargmax     192.9        2.2      195.9        6.4
        anynan         44.3        0.5       49.1       58.2
        allnan         49.3       59.2       49.4       59.7
        rankdata      137.4        2.7      142.2        5.2
        nanrankdata   164.6        3.0      164.2        5.1
        partition       9.0        1.0        9.8        1.1
        argpartition    7.8        1.1        8.7        1.0
        replace        21.3        1.4       21.3        1.4
        push          564.4       22.7      564.1       29.4
        move_sum      932.3      196.2      927.7      445.6
        move_mean    2417.4      226.2     2431.8      373.0
        move_std     3587.7      127.5     3547.3      200.5
        move_var     3676.2      199.0     3690.7      349.5
        move_min      755.3       27.3      769.2       65.8
        move_max      731.9       27.2      753.2      111.9
        move_argmin  1112.9       41.5     1173.6      242.5
        move_argmax   812.3       44.2      780.7      306.5
        move_median  1043.2       42.9     1043.1      142.1
        move_rank    1680.8        2.4     1919.6        9.0

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