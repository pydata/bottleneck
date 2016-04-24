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
        nansum         33.4        3.9       33.9        6.3
        nanmean       116.6        5.3      116.6        6.9
        nanstd        210.6        4.1      195.3        6.8
        nanvar        192.7        4.4      195.0        6.7
        nanmin         29.7        1.1       30.0        1.4
        nanmax         29.4        1.0       30.0        1.7
        median         48.6        0.8       57.0        1.0
        nanmedian      49.5        2.5       58.7        6.0
        ss             13.8        3.8        3.1        1.1
        nanargmin      53.4        3.9       52.8        5.5
        nanargmax      53.6        4.0       54.4        6.6
        anynan         14.1        1.6       14.7      180.9
        allnan         14.5      209.7       14.4      189.3
        rankdata       34.6        1.5       35.2        2.3
        nanrankdata    46.3       20.1       46.3       29.0
        partsort        4.8        0.8        4.9        1.1
        argpartsort     2.6        0.7        2.6        0.6
        replace         8.3        2.0        8.2        2.0
        push          157.4       11.2      153.5       14.2
        move_sum      302.8      228.1      305.2      352.1
        move_mean     741.0      232.7      748.8      447.4
        move_std     1169.6      125.1     1220.6      404.2
        move_var     1187.9      168.0     1222.8      371.8
        move_min      230.0       22.9      232.3       46.9
        move_max      225.3       22.3      229.9       70.3
        move_argmin   355.1       77.6      372.9      228.2
        move_argmax   366.3       80.3      380.5      255.6
        move_median   379.8       33.9      375.7       54.8
        move_rank     612.3        4.2      668.5       10.6

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
    Ran 110 tests in 70.712s
    OK
    <nose.result.TextTestResult run=110 errors=0 failures=0>