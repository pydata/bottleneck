.. image:: https://travis-ci.org/pydata/bottleneck.svg?branch=master
    :target: https://travis-ci.org/pydata/bottleneck
.. image:: https://github.com/pydata/bottleneck/workflows/Github%20Actions/badge.svg
    :target: https://github.com/pydata/bottleneck/actions

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
        Bottleneck 1.3.0.dev0; Numpy 1.16.4
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-fifth NaNs; float64 used

                  no NaN     no NaN      NaN       no NaN      NaN
                   (100,)  (1000,1000)(1000,1000)(1000,1000)(1000,1000)
                   axis=0     axis=0     axis=0     axis=1     axis=1
    nansum         30.9        1.4        1.6        2.0        2.1
    nanmean       101.5        2.0        1.8        3.3        2.5
    nanstd        148.9        1.7        1.8        2.8        2.5
    nanvar        141.0        1.6        1.8        2.8        2.5
    nanmin         27.4        0.5        1.6        1.1        3.7
    nanmax         24.5        0.6        1.6        1.1        3.8
    median        122.2        1.2        4.6        1.1        5.4
    nanmedian     122.6        5.0        5.7        4.8        5.5
    ss             13.8        1.2        1.2        1.6        1.6
    nanargmin      63.9        2.9        7.1        3.3        6.7
    nanargmax      61.8        2.6        4.6        3.0        6.3
    anynan         10.0        0.3       55.0        0.8       44.1
    allnan         15.7      201.8      154.3      147.7      119.2
    rankdata       46.3        1.3        1.2        2.3        2.3
    nanrankdata    52.1        1.4        1.3        2.5        2.4
    partition       3.5        1.1        1.6        1.0        1.5
    argpartition    3.0        1.2        1.5        1.1        1.6
    replace         7.0        1.5        1.6        1.6        1.6
    push         1583.4        5.3        6.7       13.4       11.1
    move_sum     3716.1       33.3       88.7      197.9      184.7
    move_mean    8115.1       68.4      116.8      382.3      257.9
    move_std    10349.8       88.3      169.1      243.5      331.7
    move_var    10868.3       97.2      178.3      281.9      332.9
    move_min     2006.3       13.6       30.5       24.2       45.6
    move_max     2002.1       14.9       29.1       24.8       46.9
    move_argmin  3518.0       35.5       62.7       52.7       89.0
    move_argmax  3418.6       32.5       59.7       46.1       84.4
    move_median  2835.1      155.7      151.6      167.9      166.7
    move_rank    1241.4        1.4        1.0        2.5        1.9

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
 docs                 https://bottleneck.readthedocs.io
 code                 https://github.com/pydata/bottleneck
 mailing list         https://groups.google.com/group/bottle-neck
===================   ========================================================

License
=======

Bottleneck is distributed under a Simplified BSD license. See the LICENSE file
and LICENSES directory for details.

Install
=======

Requirements:

======================== ====================================================
Bottleneck               Python 2.7, 3.5, 3.6; NumPy 1.16.0
Compile                  gcc, clang, MinGW or MSVC
Unit tests               pytest
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
