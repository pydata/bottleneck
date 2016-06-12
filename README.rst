.. image:: https://travis-ci.org/kwgoodman/bottleneck.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/bottleneck
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
        nansum         55.1        2.3       59.4        7.4
        nanmean       216.9        3.5      218.0        8.3
        nanstd        355.3        2.8      360.7        7.1
        nanvar        368.4        2.8      362.5        7.1
        nanmin         49.7        1.0       49.4        1.4
        nanmax         45.7        1.0       46.1        2.1
        median         80.9        0.8      113.3        5.2
        nanmedian      96.9        3.9      112.3       12.0
        ss             29.1        1.6       29.2        1.6
        nanargmin      91.1        2.9       92.4        7.3
        nanargmax      92.3        2.8       92.7        8.0
        anynan         24.0        0.7       25.7       58.6
        allnan         25.6       61.9       25.2       58.7
        rankdata       45.3        1.5       44.5        2.2
        nanrankdata    51.6       27.4       48.0       42.8
        partsort        5.9        0.9        6.1        1.1
        argpartsort     3.0        0.9        3.2        0.7
        replace         9.9        1.2        9.9        1.1
        push          201.9       21.1      206.0       26.4
        move_sum      354.6      155.1      344.6      409.7
        move_mean     807.0      150.6      863.1      539.6
        move_std     1202.4       78.2     1278.3      369.6
        move_var     1198.7      107.9     1235.3      369.6
        move_min      230.9       23.3      238.9       51.5
        move_max      253.0       23.1      251.4       93.0
        move_argmin   412.9       41.0      431.7      242.8
        move_argmax   389.2       41.6      406.6      282.5
        move_median   507.3       42.3      492.9      161.4
        move_rank     658.8        2.9      718.7       11.1

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
Bottleneck               Python 2.7, 3.4, 3.5; NumPy 1.11.0
Compile                  gcc or clang or MinGW
Unit tests               nose
Cython                   Optional for released version of Bottleneck
======================== ====================================================

If Cython is installed on your computer then you can install either a released
version of Bottleneck (PyPI) or a development version (GitHub).

If Cython is not installed on your computer then you can only install a
released version of Bottleneck (PyPI). Cython is not required because the
Cython files have already been converted to C source files in Bottleneck
releases.

To install Bottleneck on GNU/Linux, Mac OS X, et al.::

    $ sudo python setup.py install

To install bottleneck on Windows, first install MinGW and add it to your
system path. Then install Bottleneck with the commands::

    python setup.py install --compiler=mingw32

Alternatively, you can use the Windows binaries created by Christoph Gohlke:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck

Unit tests
==========

After you have installed Bottleneck, run the suite of unit tests::

    >>> import bottleneck as bn
    >>> bn.test()
    <snip>
    Ran 127 tests in 18.978s
    OK
    <nose.result.TextTestResult run=127 errors=0 failures=0>