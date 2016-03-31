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
        nansum         35.7        3.7       36.0        6.2
        nanmean       111.5        5.5      126.0        8.1
        nanstd        209.7        4.7      214.6        7.2
        nanvar        203.5        4.5      206.3        6.8
        nanmin         30.8        1.1       31.1        1.5
        nanmax         29.9        1.1       29.5        1.7
        median         46.7        0.7       52.4        1.0
        nanmedian      43.7        2.6       54.2        5.5
        ss             13.5        4.2        3.3        1.1
        nanargmin      55.4        4.1       52.8        5.5
        nanargmax      57.5        4.1       56.8        6.8
        anynan         13.7        1.5       15.7      187.9
        allnan         15.6      303.0       13.9      223.8
        rankdata       37.0        1.6       38.5        2.3
        nanrankdata    47.6       20.0       44.3       28.6
        partsort        4.8        0.8        4.9        1.1
        argpartsort     2.7        0.7        2.8        0.6
        replace         9.2        1.3        9.1        1.3
        move_sum      326.7      196.3      328.9      304.7
        move_mean     799.8      244.3      815.5      435.4
        move_std     1230.3      126.3     1307.2      382.2
        move_var     1250.3      183.6     1288.0      378.8
        move_min      236.2       23.7      245.1       47.9
        move_max      250.2       22.5      249.9       72.8
        move_median   394.7       34.8      386.5       54.6

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
