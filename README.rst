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
        nansum         31.2        4.0       31.1        6.4
        nanmean       113.1        5.6      113.6        7.5
        nanstd        192.4        4.3      192.1        7.1
        nanvar        185.5        4.5      188.2        6.8
        nanmin         27.3        1.1       27.1        1.4
        nanmax         27.4        1.0       27.4        1.7
        median         43.0        0.8       49.8        1.0
        nanmedian      44.3        2.4       51.6        5.9
        ss             12.5        3.9        2.9        1.1
        nanargmin      53.1        4.0       51.1        5.4
        nanargmax      50.2        4.0       50.4        6.6
        anynan          9.7        1.5       12.7      224.9
        allnan         12.6      261.2       12.0      245.9
        rankdata       36.8        1.6       36.1        2.2
        nanrankdata    46.3       19.2       48.6       33.3
        partsort        4.3        0.8        4.6        1.1
        argpartsort     2.1        0.8        2.2        0.5
        replace         9.0        1.9        8.2        1.9
        push          169.6        8.6      148.0       10.7
        move_sum      278.3      250.5      277.5      355.8
        move_mean     659.7      295.2      683.8      476.9
        move_std      997.3      161.2     1041.5      388.1
        move_var     1039.8      247.0     1149.5      443.9
        move_min      204.1       28.3      206.4       51.1
        move_max      212.6       25.2      212.1       77.7
        move_argmin   365.5       78.3      360.4      224.1
        move_argmax   399.8      102.1      318.6      253.7
        move_median   362.2       35.0      354.1       53.5

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
