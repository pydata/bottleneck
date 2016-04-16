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
        nansum         19.6        4.1       32.6        6.5
        nanmean       111.6        5.6      114.5        7.2
        nanstd        198.4        4.7      194.8        7.2
        nanvar        191.3        4.6      189.8        7.0
        nanmin         29.2        1.1       29.2        1.4
        nanmax         29.1        1.0       29.2        1.7
        median         48.5        0.8       55.7        1.0
        nanmedian      45.9        2.5       52.8        5.9
        ss             13.5        3.8        3.1        1.1
        nanargmin      52.8        4.0       52.5        5.3
        nanargmax      52.5        4.0       53.2        6.7
        anynan         13.8        1.5       14.3      180.1
        allnan         14.0      209.7       14.0      197.4
        rankdata       35.6        1.5       36.8        2.3
        nanrankdata    48.2       20.0       43.0       28.7
        partsort        4.6        0.8        4.8        1.1
        argpartsort     2.6        0.7        2.7        0.6
        replace         8.3        2.0        8.1        2.0
        push          208.3       11.0      205.2       14.6
        move_sum      301.8      204.5      301.5      330.9
        move_mean     740.3      237.7      751.4      481.6
        move_std     1156.1      126.0     1215.1      366.3
        move_var     1082.5      176.5     1107.3      362.9
        move_min      215.1       21.9      218.4       47.3
        move_max      228.5       22.3      232.4       70.2
        move_argmin   362.7       80.8      376.8      236.4
        move_argmax   351.4       77.2      363.3      246.9
        move_median   376.7       33.6      372.0       53.2

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
    Ran 88 tests in 70.712s
    OK
    <nose.result.TextTestResult run=88 errors=0 failures=0>
