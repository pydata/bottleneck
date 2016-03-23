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
        Numpy (np)  1.9.2
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum         31.9        3.9       33.0        6.9
        nanmean       119.1        5.2      119.5        8.4
        nanstd        168.0        4.8      165.6        7.6
        nanvar        189.1        4.6      193.7        7.2
        nanmin         29.6        1.1       28.9        1.4
        nanmax         29.2        1.0       29.3        1.7
        median         29.8        0.8       32.5        0.9
        nanmedian      38.9        2.1       46.7        4.8
        ss             13.5        3.8        3.0        1.2
        nanargmin      49.0        4.6       53.9        5.3
        nanargmax      54.3        3.9       54.6        6.8
        anynan         12.3        1.6       12.7      158.0
        allnan         11.9      254.6       12.4      198.5
        rankdata       38.0        1.5       38.9        2.2
        nanrankdata    47.1       18.9       43.7       27.7
        partsort        6.0        0.8        6.2        1.1
        argpartsort     2.4        0.8        2.6        0.6
        replace         8.7        1.3        8.8        1.3
        move_sum      290.3      207.8      288.7      325.1
        move_mean     626.2      243.0      634.8      472.9
        move_std     1116.7      127.4     1199.1      393.6
        move_min      218.2       23.8      214.6       47.4
        move_max      208.7       21.5      211.9       70.1
        move_median   372.7       55.1      134.9       74.5

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtypes result in calls to slower, unaccelerated
functions.

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
Bottleneck               Python 2.7, 3.4; **NumPy 1.9.1**
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
