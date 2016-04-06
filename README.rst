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
        nansum         35.2        3.7       35.2        5.8
        nanmean       120.5        5.8      125.0        8.1
        nanstd        212.6        4.6      215.3        7.1
        nanvar        203.7        4.4      206.4        6.8
        nanmin         31.6        1.1       31.1        1.4
        nanmax         31.3        1.0       31.2        1.7
        median         47.8        0.7       55.9        1.0
        nanmedian      47.4        2.4       55.2        5.9
        ss             14.5        3.8        3.1        1.2
        nanargmin      56.2        4.2       56.3        5.4
        nanargmax      56.6        3.9       56.2        6.4
        anynan         13.7        1.5       14.4      186.3
        allnan         14.4      282.9       14.2      202.0
        rankdata       38.5        1.5       38.3        2.3
        nanrankdata    47.3       19.1       44.6       29.0
        partsort        4.7        0.8        5.3        1.0
        argpartsort     2.7        0.7        2.7        0.6
        replace         9.0        1.3        8.8        1.3
        move_sum      320.2      191.4      322.0      321.4
        move_mean     796.0      231.8      804.5      434.6
        move_std     1247.9      125.5     1325.6      374.9
        move_var     1230.0      183.7     1289.2      368.4
        move_min      242.8       23.2      244.8       49.4
        move_max      258.4       20.1      246.4       71.0
        move_argmin   374.0       75.8      388.1      240.0
        move_argmax   385.1       81.1      400.1      262.2
        move_median   385.3       34.5      384.8       55.3

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
    Ran 88 tests in 70.712s
    OK
    <nose.result.TextTestResult run=88 errors=0 failures=0>
