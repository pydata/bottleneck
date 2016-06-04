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
        nansum         34.0        3.9       34.1        6.4
        nanmean       117.1        5.6      118.5        7.3
        nanstd        196.5        4.2      202.6        7.0
        nanvar        190.0        4.5      195.6        6.8
        nanmin         29.9        1.1       30.4        1.4
        nanmax         29.8        1.0       30.6        1.8
        median         52.6        0.7       60.0        2.9
        nanmedian      48.0        2.4       55.9        6.0
        ss             13.4        3.8        3.1        1.1
        nanargmin      54.3        4.0       54.3        5.5
        nanargmax      54.7        4.0       55.3        6.6
        anynan         13.9        1.5       14.2      179.1
        allnan         14.4      209.0       14.1      194.1
        rankdata       34.4        1.5       36.1        2.2
        nanrankdata    47.4       20.8       41.6       34.6
        partsort        4.5        0.8        4.9        1.1
        argpartsort     2.5        0.7        2.5        0.5
        replace         8.2        1.9        8.2        2.0
        push          149.9       10.4      152.7       14.5
        move_sum      294.7      219.7      301.4      336.1
        move_mean     709.9      248.0      729.5      489.8
        move_std     1191.6      133.3     1223.3      363.8
        move_var     1165.8      183.3     1202.6      379.1
        move_min      237.6       22.2      235.3       48.6
        move_max      242.4       22.6      239.9       73.1
        move_argmin   364.6       79.9      385.0      235.4
        move_argmax   368.7       80.7      384.2      257.4
        move_median   435.8       48.7      435.2      138.5
        move_rank     587.4        3.8      653.6       10.8

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
    Ran 132 tests in 18.978s
    OK
    <nose.result.TextTestResult run=132 errors=0 failures=0>
