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
        Bottleneck  1.0.0dev
        Numpy (np)  1.9.1
        Speed is NumPy time divided by Bottleneck time
        NaN means approx one-third NaNs; float64 and axis=-1 are used

                     no NaN     no NaN      NaN        NaN
                       (10,)   (1000,1000)   (10,)   (1000,1000)
        nansum         40.0        4.0       40.0        9.3
        nanmean       149.0        5.5      137.0       10.8
        nanstd        241.1        4.2      240.2        8.4
        nanmin         31.1        1.1       31.2        1.7
        nanmax         31.3        1.1       31.1        2.9
        median         44.6        0.8       47.6        0.9
        nanmedian      59.6        2.9       72.6        6.8
        ss             14.4        3.5       14.4        3.7
        nanargmin      62.8        4.2       62.7        7.3
        nanargmax      63.0        4.2       63.0        9.2
        anynan         13.5        1.0       15.2       88.4
        allnan         14.0       99.6       15.1       99.6
        rankdata       53.4        1.4       47.7        2.1
        nanrankdata    61.1       25.9       55.9       38.4
        partsort        3.7        2.7        3.8        3.4
        argpartsort     0.9        2.3        0.9        1.6
        replace         9.6        1.2        9.7        1.2
        move_sum      350.7      124.5      354.1      344.4
        move_mean     839.9       98.4      852.2      434.3
        move_std     1351.4       56.8     1490.9      778.6
        move_min      251.6       21.0      258.9       54.3
        move_max      260.2       21.0      271.0      122.8
        move_median   482.3       44.9      471.1      210.7

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
