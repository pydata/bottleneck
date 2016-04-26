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
        nansum         34.6        3.9       34.3        6.4
        nanmean       116.7        5.5      114.0        7.4
        nanstd        197.0        4.2      187.3        6.9
        nanvar        192.3        4.5      191.9        6.8
        nanmin         29.3        1.1       29.3        1.4
        nanmax         29.3        1.0       29.5        1.7
        median         51.7        0.8       59.3        2.9
        nanmedian      47.1        2.5       54.5        6.0
        ss             13.7        3.8        3.1        1.1
        nanargmin      54.4        4.0       54.1        5.5
        nanargmax      54.2        3.9       54.5        6.6
        anynan         13.8        1.5       14.4      182.5
        allnan         14.2      205.6       14.1      192.7
        rankdata       35.2        1.5       34.9        2.3
        nanrankdata    49.7       20.0       45.8       28.7
        partsort        4.8        0.8        4.9        1.1
        argpartsort     2.5        0.7        2.6        0.6
        replace         8.2        2.0        8.2        2.0
        push          156.3       11.3      150.2       14.5
        move_sum      302.9      223.1      302.6      370.8
        move_mean     682.7      244.1      700.5      466.4
        move_std     1160.5      125.4     1222.2      380.3
        move_var     1138.0      179.9     1170.5      374.4
        move_min      230.1       23.5      214.7       47.3
        move_max      225.1       22.4      229.3       72.1
        move_argmin   363.6       76.1      376.9      228.2
        move_argmax   370.9       82.1      384.4      263.5
        move_median   375.0       34.3      371.8       54.7
        move_rank     628.3        5.0      673.8       12.9

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