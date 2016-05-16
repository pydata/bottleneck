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
        Bottleneck 1.1.0dev; Numpy 1.11.0
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum         33.2        4.0       33.1        6.5
        nanmean       114.5        5.5      117.8        7.4
        nanstd        191.7        4.3      196.3        7.0
        nanvar        189.8        4.6      192.8        6.4
        nanmin         30.1        1.1       28.7        1.4
        nanmax         30.1        1.0       36.6        1.7
        median         48.6        0.8       61.2        2.9
        nanmedian      47.4        2.5       55.2        6.0
        ss             13.8        3.9        3.1        1.1
        nanargmin      53.0        4.1       53.5        5.6
        nanargmax      53.6        4.0       54.2        6.6
        anynan         14.1        1.5       14.8      183.0
        allnan         14.5      210.9       14.5      195.9
        rankdata       34.3        1.5       35.4        2.3
        nanrankdata    48.2       20.1       43.6       28.8
        partsort        4.6        0.8        4.8        1.1
        argpartsort     2.5        0.7        2.6        0.6
        replace         8.2        2.0        8.2        2.0
        push          155.7       11.1      152.4       14.3
        move_sum      305.9      230.5      308.9      360.9
        move_mean     697.5      247.5      714.2      457.9
        move_std     1168.6      130.3     1236.2      379.6
        move_var     1155.3      188.1     1179.2      390.1
        move_min      238.9       22.6      236.2       48.7
        move_max      232.6       22.3      229.6       71.4
        move_argmin   357.4       76.2      380.5      237.6
        move_argmax   371.3       83.6      409.2      276.1
        move_median   371.4       34.4      377.4       55.3
        move_rank     593.2        4.3      646.7       10.9

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
    Ran 113 tests in 18.978s
    OK
    <nose.result.TextTestResult run=113 errors=0 failures=0>
