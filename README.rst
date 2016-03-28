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
        Numpy (np)  1.10.4
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum         34.7        3.9       34.8        6.5
        nanmean       121.2        5.6      123.2        8.2
        nanstd        209.0        4.7      212.1        7.2
        nanvar        210.5        4.6      205.5        6.9
        nanmin         30.9        1.1       31.5        1.4
        nanmax         31.3        1.0       30.7        1.8
        median         50.6        0.8       60.2        1.0
        nanmedian      49.8        2.6       57.6        6.1
        ss             14.3        3.8        3.1        1.1
        nanargmin      55.1        3.8       55.6        5.1
        nanargmax      55.4        3.9       56.2        6.5
        anynan         13.6        1.5       14.3      170.2
        allnan         14.1      271.1       13.6      196.6
        rankdata       38.2        1.6       37.8        2.3
        nanrankdata    45.3       20.7       41.3       28.0
        partsort        5.0        0.8        5.3        1.1
        argpartsort     2.6        0.8        2.7        0.6
        replace        10.2        1.3        9.9        1.5
        move_sum      281.9      189.0      296.9      326.2
        move_mean     730.1      224.0      784.0      466.7
        move_std     1122.0      132.5     1158.3      385.9
        move_min      220.6       22.9      225.3       48.7
        move_max      220.2       22.2      221.7       72.6
        move_median   396.1       35.7      387.1       53.8

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
Bottleneck               Python 2.7, 3.4; NumPy 1.10.4
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
    Ran 79 tests in 70.712s
    OK
    <nose.result.TextTestResult run=79 errors=0 failures=0>
