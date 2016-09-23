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
        nansum         64.1        2.3       63.7        7.4
        nanmean       217.7        3.5      216.3        8.3
        nanstd        355.6        2.7      383.6        7.0
        nanvar        348.8        2.7      370.2        7.0
        nanmin         57.3        1.0       48.1        1.3
        nanmax         51.9        1.0       56.8        2.1
        median        175.6        1.1      231.0        5.1
        nanmedian     158.0        5.0      185.6       39.8
        ss             32.3        1.6       32.4        1.6
        nanargmin      92.9        2.2       94.1        5.8
        nanargmax      96.4        2.2       97.2        6.1
        anynan         22.0        0.5       23.4       73.0
        allnan         26.2      105.8       25.6       80.1
        rankdata      101.8        2.7      100.5        5.1
        nanrankdata   121.8        2.9      124.7        5.1
        partsort        7.4        1.0        8.1        1.0
        argpartsort     6.2        1.1        6.7        0.9
        replace        18.1        1.4       18.0        1.5
        push          304.1       19.6      319.3       26.4
        move_sum      695.6      210.2      645.6      562.7
        move_mean    1615.4      333.9     1612.8      481.4
        move_std     2576.8      168.3     2672.2      396.9
        move_var     2551.7      211.5     2568.8      394.7
        move_min      524.6       25.9      536.6       59.8
        move_max      460.5       25.9      467.3       96.8
        move_argmin   800.6       41.3      839.0      256.9
        move_argmax   833.3       43.4      872.1      305.4
        move_median   928.0       43.4      919.9      143.3
        move_rank    1028.0        3.0     1148.9       11.2

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
