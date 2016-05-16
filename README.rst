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
        nansum         34.4        4.0       33.9        6.4
        nanmean       117.9        5.5      117.5        7.3
        nanstd        200.8        4.3      203.2        7.0
        nanvar        193.6        4.4      195.7        6.8
        nanmin         30.0        1.1       30.3        1.4
        nanmax         30.0        1.0       29.7        1.7
        median         48.7        0.8       60.3        2.9
        nanmedian      47.9        2.4       55.6        6.0
        ss             13.6        3.9        3.1        1.1
        nanargmin      54.2        4.0       53.8        5.5
        nanargmax      54.0        4.0       53.5        6.7
        anynan         14.5        1.5       14.7      183.4
        allnan         14.3      211.0       14.3      201.9
        rankdata       31.2        1.5       34.3        2.3
        nanrankdata    46.7       20.2       42.9       28.8
        partsort        4.7        0.8        4.8        1.1
        argpartsort     2.5        0.7        2.6        0.6
        replace         8.2        2.0        8.2        2.0
        push          159.5       11.0      159.1       14.0
        move_sum      307.9      211.3      309.0      337.3
        move_mean     696.8      253.4      703.9      469.5
        move_std     1138.0      129.2     1214.6      383.9
        move_var     1164.8      187.9     1152.1      398.6
        move_min      227.8       23.2      222.8       45.4
        move_max      230.2       23.8      237.4       76.2
        move_argmin   353.7       81.2      379.0      238.1
        move_argmax   363.5       83.9      370.2      255.7
        move_median   427.2       47.2      407.5      128.3
        move_rank     597.7        3.7      652.0       10.9

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