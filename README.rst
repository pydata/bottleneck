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
        nansum         36.5        4.0       36.6        9.1
        nanmean       144.5        5.2      146.1        9.2
        nanstd        253.2        4.3      253.1        8.4
        nanvar        241.4        4.2      241.2        8.4
        nanmin         30.6        1.1       30.5        1.7
        nanmax         32.1        1.1       32.2        2.9
        median         43.3        0.8       45.7        0.9
        nanmedian      58.7        2.8       67.5        6.8
        ss             14.3        3.5       14.4        3.4
        nanargmin      60.8        4.1       61.1        7.3
        nanargmax      61.4        4.1       61.3        9.0
        anynan         12.9        1.0       13.5       89.2
        allnan         13.6       98.5       13.5       97.8
        rankdata       45.5        1.4       45.9        1.9
        nanrankdata    60.7       26.3       54.1       37.9
        partsort        3.3        2.8        3.3        3.5
        argpartsort     0.9        2.1        0.9        1.4
        replace         9.9        1.2        9.9        1.2
        move_sum      276.1      121.1      283.2      330.9
        move_mean     714.4       95.7      723.7      415.8
        move_std     1102.5       56.2     1160.7      749.0
        move_min      207.0       20.9      211.0       55.2
        move_max      213.8       21.4      218.4      118.6
        move_median   457.9       43.4      452.7      208.4

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
