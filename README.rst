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
        nansum         34.9        4.0       34.4        6.4
        nanmean       119.3        5.6      121.0        8.1
        nanstd        207.2        4.7      207.2        7.3
        nanvar        198.7        4.6      198.5        6.9
        nanmin         30.8        1.1       31.0        1.4
        nanmax         30.0        1.1       30.9        1.7
        median         49.3        0.8       60.6        1.0
        nanmedian      49.6        2.7       55.2        6.1
        ss             16.1        4.9        3.2        1.1
        nanargmin      52.3        4.3       55.8        5.9
        nanargmax      55.3        4.5       60.3        6.9
        anynan         13.8        1.5       14.4      243.7
        allnan         13.6      356.8       13.2      276.7
        rankdata       35.5        1.4       37.7        2.5
        nanrankdata    47.0       18.2       42.6       27.9
        partsort        5.0        0.8        5.2        1.1
        argpartsort     2.7        0.8        2.8        0.6
        replace         9.8        1.3        9.9        1.3
        move_sum      284.2      190.3      288.1      306.9
        move_mean     718.1      237.7      731.9      444.4
        move_std     1128.0      124.5     1186.3      378.9
        move_var     1138.9      184.1     1174.8      372.4
        move_min      213.8       22.7      219.9       47.3
        move_max      218.8       21.7      214.3       72.9
        move_median   383.3       35.6      385.7       57.3

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
    Ran 82 tests in 70.712s
    OK
    <nose.result.TextTestResult run=82 errors=0 failures=0>
